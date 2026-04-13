mod build;
mod encode;
mod params;

use std::sync::Arc;

use zkgpu_babybear::BabyBear;
use zkgpu_core::ZkGpuError;

use crate::async_util;
use crate::buffer::WgpuBuffer;
use crate::device::WgpuDevice;
use crate::dispatch::LinearDispatch;
use crate::profiling::{GpuProfiler, GpuTiming, TimestampSpan};

use super::planner::FourStepPlanConfig;
use super::stockham::NttTimings;

/// GPU four-step NTT plan for BabyBear fields.
///
/// Correct Cooley-Tukey four-step decomposition for N = R × C:
///
///   Phase 1: Transpose R×C → C×R
///   Phase 2: R-point batched row DFTs (C batches — column DFTs of original)
///   Phase 3: Diagonal twiddle multiply omega_N^(k_r × c)
///   Phase 4: Transpose C×R → R×C
///   Phase 5: C-point batched row DFTs (R batches — row DFTs of original)
///   Phase 6: Transpose R×C → C×R (output reordering)
///   Phase 7: Inverse scale (inverse NTT only)
pub(crate) struct FourStepPlan {
    pub(super) leaf_global_pipeline: Arc<wgpu::ComputePipeline>,
    pub(super) leaf_r4_pipeline: Arc<wgpu::ComputePipeline>,
    pub(super) leaf_bgl: Arc<wgpu::BindGroupLayout>,

    pub(super) twiddle_pipeline: Arc<wgpu::ComputePipeline>,
    pub(super) twiddle_bgl: Arc<wgpu::BindGroupLayout>,
    pub(super) twiddle_buffer: wgpu::Buffer,
    pub(super) twiddle_prime_buffer: wgpu::Buffer,
    pub(super) twiddle_param_buffer: wgpu::Buffer,
    pub(super) twiddle_dispatch: LinearDispatch,

    pub(super) transpose_pipeline: Arc<wgpu::ComputePipeline>,
    pub(super) transpose_bgl: Arc<wgpu::BindGroupLayout>,
    /// R×C → C×R transpose params (phases 1 and 6)
    pub(super) transpose_rc_to_cr_params: wgpu::Buffer,
    /// C×R → R×C transpose params (phase 4)
    pub(super) transpose_cr_to_rc_params: wgpu::Buffer,

    pub(super) scale_pipeline: Arc<wgpu::ComputePipeline>,
    pub(super) scale_bgl: Arc<wgpu::BindGroupLayout>,
    pub(super) scale_param_buffer: Option<wgpu::Buffer>,
    pub(super) scale_dispatch: LinearDispatch,

    /// Phase-2 leaf: R-point NTTs (col_leaf config), C batches
    pub(super) phase2_r4_twiddle_buffer: wgpu::Buffer,
    pub(super) phase2_r4_twiddle_prime_buffer: wgpu::Buffer,
    pub(super) phase2_r4_stage_param_buffers: Vec<wgpu::Buffer>,
    pub(super) phase2_r2_twiddle_buffer: wgpu::Buffer,
    pub(super) phase2_r2_twiddle_prime_buffer: wgpu::Buffer,
    pub(super) phase2_r2_stage_param_buffers: Vec<wgpu::Buffer>,

    /// Phase-5 leaf: C-point NTTs (row_leaf config), R batches
    pub(super) phase5_r4_twiddle_buffer: wgpu::Buffer,
    pub(super) phase5_r4_twiddle_prime_buffer: wgpu::Buffer,
    pub(super) phase5_r4_stage_param_buffers: Vec<wgpu::Buffer>,
    pub(super) phase5_r2_twiddle_buffer: wgpu::Buffer,
    pub(super) phase5_r2_twiddle_prime_buffer: wgpu::Buffer,
    pub(super) phase5_r2_stage_param_buffers: Vec<wgpu::Buffer>,

    pub(super) scratch_buffer: wgpu::Buffer,
    pub(super) transpose_scratch_buffer: wgpu::Buffer,

    pub(super) config: FourStepPlanConfig,
}

impl FourStepPlan {
    pub(crate) fn num_dispatches(&self) -> u32 {
        self.config.total_dispatches() + u32::from(self.scale_param_buffer.is_some())
    }

    pub(crate) fn execute_kernels(
        &mut self,
        device: &WgpuDevice,
        buf: &mut WgpuBuffer<BabyBear>,
    ) -> Result<(), ZkGpuError> {
        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Four-step NTT encoder"),
            });

        self.encode_all(&device.device, &mut encoder, buf, &[]);

        device.queue.submit(Some(encoder.finish()));
        device
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| ZkGpuError::BackendError(Box::new(e)))?;
        Ok(())
    }

    pub(crate) fn execute_kernels_profiled(
        &mut self,
        device: &WgpuDevice,
        buf: &mut WgpuBuffer<BabyBear>,
    ) -> Result<Option<NttTimings>, ZkGpuError> {
        let wall_start = std::time::Instant::now();

        let num_dispatches = self.num_dispatches();
        let max_queries = num_dispatches * 2;
        let mut profiler =
            GpuProfiler::new_if_supported(&device.device, &device.queue, &device.caps, max_queries);

        let spans: Vec<TimestampSpan> = if let Some(ref mut p) = profiler {
            (0..num_dispatches).filter_map(|_| p.begin_span()).collect()
        } else {
            Vec::new()
        };

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Four-step NTT profiled encoder"),
            });

        let ts_writes: Vec<Option<wgpu::ComputePassTimestampWrites<'_>>> = (0..num_dispatches
            as usize)
            .map(|i| {
                profiler
                    .as_ref()
                    .and_then(|p| spans.get(i).map(|span| p.pass_timestamp_writes(span)))
            })
            .collect();

        self.encode_all(&device.device, &mut encoder, buf, &ts_writes);

        if let Some(ref profiler) = profiler {
            profiler.resolve(&mut encoder);
        }

        device.queue.submit(Some(encoder.finish()));
        device
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| ZkGpuError::BackendError(Box::new(e)))?;

        let wall_clock = wall_start.elapsed();

        let timings = if let Some(ref profiler) = profiler {
            let timestamps = profiler.read_timestamps(&device.device)?;
            let mut gpu_stage_ns = Vec::with_capacity(spans.len());
            let mut total = 0.0;

            let labels = self.dispatch_labels();
            for (i, span) in spans.iter().enumerate() {
                let dur = profiler.span_duration_ns(&timestamps, span).unwrap_or(0.0);
                total += dur;
                gpu_stage_ns.push(GpuTiming {
                    label: labels
                        .get(i)
                        .cloned()
                        .unwrap_or_else(|| format!("dispatch {i}")),
                    duration_ns: dur,
                });
            }
            Some(NttTimings {
                wall_clock,
                gpu_stage_ns,
                gpu_total_ns: total,
            })
        } else {
            None
        };

        Ok(timings)
    }

    /// Async variant of [`execute_kernels`](Self::execute_kernels). Browser-safe.
    pub(crate) async fn execute_kernels_async(
        &mut self,
        device: &WgpuDevice,
        buf: &mut WgpuBuffer<BabyBear>,
    ) -> Result<(), ZkGpuError> {
        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Four-step NTT encoder"),
            });

        self.encode_all(&device.device, &mut encoder, buf, &[]);

        device.queue.submit(Some(encoder.finish()));
        async_util::wait_for_submission(&device.device, &device.queue).await?;
        Ok(())
    }

    /// Async variant of [`execute_kernels_profiled`](Self::execute_kernels_profiled).
    /// Browser-safe. Uses `web_time::Instant` for wall-clock timing.
    pub(crate) async fn execute_kernels_profiled_async(
        &mut self,
        device: &WgpuDevice,
        buf: &mut WgpuBuffer<BabyBear>,
    ) -> Result<Option<NttTimings>, ZkGpuError> {
        let wall_start = web_time::Instant::now();

        let num_dispatches = self.num_dispatches();
        let max_queries = num_dispatches * 2;
        let mut profiler =
            GpuProfiler::new_if_supported(&device.device, &device.queue, &device.caps, max_queries);

        let spans: Vec<TimestampSpan> = if let Some(ref mut p) = profiler {
            (0..num_dispatches).filter_map(|_| p.begin_span()).collect()
        } else {
            Vec::new()
        };

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Four-step NTT profiled encoder"),
            });

        let ts_writes: Vec<Option<wgpu::ComputePassTimestampWrites<'_>>> = (0..num_dispatches
            as usize)
            .map(|i| {
                profiler
                    .as_ref()
                    .and_then(|p| spans.get(i).map(|span| p.pass_timestamp_writes(span)))
            })
            .collect();

        self.encode_all(&device.device, &mut encoder, buf, &ts_writes);

        if let Some(ref profiler) = profiler {
            profiler.resolve(&mut encoder);
        }

        device.queue.submit(Some(encoder.finish()));

        let wall_clock = wall_start.elapsed();

        let timings = if let Some(ref profiler) = profiler {
            let timestamps = profiler.read_timestamps_async(&device.device).await?;
            let mut gpu_stage_ns = Vec::with_capacity(spans.len());
            let mut total = 0.0;

            let labels = self.dispatch_labels();
            for (i, span) in spans.iter().enumerate() {
                let dur = profiler.span_duration_ns(&timestamps, span).unwrap_or(0.0);
                total += dur;
                gpu_stage_ns.push(GpuTiming {
                    label: labels
                        .get(i)
                        .cloned()
                        .unwrap_or_else(|| format!("dispatch {i}")),
                    duration_ns: dur,
                });
            }
            Some(NttTimings {
                wall_clock,
                gpu_stage_ns,
                gpu_total_ns: total,
            })
        } else {
            async_util::wait_for_submission(&device.device, &device.queue).await?;
            None
        };

        Ok(timings)
    }

    fn dispatch_labels(&self) -> Vec<String> {
        let mut labels = Vec::new();
        labels.push("transpose R\u{d7}C\u{2192}C\u{d7}R".to_string());
        for i in 0..self.config.col_leaf.r4_stage_params.len() {
            let h = i as u32 * 2;
            labels.push(format!("phase2 R-pt r4 stages {}+{}", h, h + 1));
        }
        for i in 0..self.config.col_leaf.global_stage_params.len() {
            let h = self.config.col_leaf.r4_stage_params.len() as u32 * 2 + i as u32;
            labels.push(format!("phase2 R-pt r2 stage {h}"));
        }
        labels.push("twiddle multiply".to_string());
        labels.push("transpose C\u{d7}R\u{2192}R\u{d7}C".to_string());
        for i in 0..self.config.row_leaf.r4_stage_params.len() {
            let h = i as u32 * 2;
            labels.push(format!("phase5 C-pt r4 stages {}+{}", h, h + 1));
        }
        for i in 0..self.config.row_leaf.global_stage_params.len() {
            let h = self.config.row_leaf.r4_stage_params.len() as u32 * 2 + i as u32;
            labels.push(format!("phase5 C-pt r2 stage {h}"));
        }
        labels.push("transpose R\u{d7}C\u{2192}C\u{d7}R (output)".to_string());
        if self.scale_param_buffer.is_some() {
            labels.push("inverse scale".to_string());
        }
        labels
    }
}
