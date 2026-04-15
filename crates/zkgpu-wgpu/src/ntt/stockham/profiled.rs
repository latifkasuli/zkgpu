use zkgpu_babybear::BabyBear;
use zkgpu_core::ZkGpuError;

use crate::async_util;
use crate::buffer::WgpuBuffer;
use crate::device::WgpuDevice;
use crate::profiling::{GpuProfiler, GpuTiming, TimestampSpan};

use super::{NttTimings, StockhamPlan};

impl StockhamPlan {
    /// Execute the full NTT with GPU timestamp profiling.
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
                label: Some("Stockham NTT profiled encoder"),
            });

        let ntt_dispatches = self.config.ntt_dispatches() as usize;
        let ntt_ts_writes: Vec<Option<wgpu::ComputePassTimestampWrites<'_>>> = (0..ntt_dispatches)
            .map(|s| {
                profiler
                    .as_ref()
                    .and_then(|p| spans.get(s).map(|span| p.pass_timestamp_writes(span)))
            })
            .collect();

        self.encode_ntt_stages(&device.device, &mut encoder, buf, &ntt_ts_writes);

        let scale_ts = profiler.as_ref().and_then(|p| {
            spans
                .get(ntt_dispatches)
                .map(|span| p.pass_timestamp_writes(span))
        });
        self.encode_scale(&device.device, &mut encoder, buf, scale_ts);

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
            let r4_count = self.config.r4_stage_params.len();
            let r2_count = self.config.global_stage_params.len();
            for (i, span) in spans.iter().enumerate() {
                let dur = profiler
                    .span_duration_ns(&timestamps, span)
                    .unwrap_or(0.0);
                total += dur;
                let label = if i < r4_count {
                    format!("r4 stages {}+{}", i * 2, i * 2 + 1)
                } else if i < r4_count + r2_count {
                    let r2_idx = i - r4_count;
                    let stage_h = r4_count * 2 + r2_idx;
                    format!("r2 stage {stage_h}")
                } else if i < ntt_dispatches {
                    // Only reached when the tail strategy is LocalFusedR4.
                    // GlobalOnlyR4 fuses the tail into the global R4 chain
                    // above, so its dispatches keep the "r4 stages X+Y" label.
                    "local fused".to_string()
                } else {
                    "inverse scale".to_string()
                };
                gpu_stage_ns.push(GpuTiming {
                    label,
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
                label: Some("Stockham NTT profiled encoder"),
            });

        let ntt_dispatches = self.config.ntt_dispatches() as usize;
        let ntt_ts_writes: Vec<Option<wgpu::ComputePassTimestampWrites<'_>>> = (0..ntt_dispatches)
            .map(|s| {
                profiler
                    .as_ref()
                    .and_then(|p| spans.get(s).map(|span| p.pass_timestamp_writes(span)))
            })
            .collect();

        self.encode_ntt_stages(&device.device, &mut encoder, buf, &ntt_ts_writes);

        let scale_ts = profiler.as_ref().and_then(|p| {
            spans
                .get(ntt_dispatches)
                .map(|span| p.pass_timestamp_writes(span))
        });
        self.encode_scale(&device.device, &mut encoder, buf, scale_ts);

        if let Some(ref profiler) = profiler {
            profiler.resolve(&mut encoder);
        }

        device.queue.submit(Some(encoder.finish()));

        let wall_clock = wall_start.elapsed();

        let timings = if let Some(ref profiler) = profiler {
            let timestamps = profiler.read_timestamps_async(&device.device).await?;
            let mut gpu_stage_ns = Vec::with_capacity(spans.len());
            let mut total = 0.0;
            let r4_count = self.config.r4_stage_params.len();
            let r2_count = self.config.global_stage_params.len();
            for (i, span) in spans.iter().enumerate() {
                let dur = profiler
                    .span_duration_ns(&timestamps, span)
                    .unwrap_or(0.0);
                total += dur;
                let label = if i < r4_count {
                    format!("r4 stages {}+{}", i * 2, i * 2 + 1)
                } else if i < r4_count + r2_count {
                    let r2_idx = i - r4_count;
                    let stage_h = r4_count * 2 + r2_idx;
                    format!("r2 stage {stage_h}")
                } else if i < ntt_dispatches {
                    // Only reached when the tail strategy is LocalFusedR4.
                    // GlobalOnlyR4 fuses the tail into the global R4 chain
                    // above, so its dispatches keep the "r4 stages X+Y" label.
                    "local fused".to_string()
                } else {
                    "inverse scale".to_string()
                };
                gpu_stage_ns.push(GpuTiming {
                    label,
                    duration_ns: dur,
                });
            }
            Some(NttTimings {
                wall_clock,
                gpu_stage_ns,
                gpu_total_ns: total,
            })
        } else {
            // No profiler — still wait for GPU completion.
            async_util::wait_for_submission(&device.device, &device.queue).await?;
            None
        };

        Ok(timings)
    }
}
