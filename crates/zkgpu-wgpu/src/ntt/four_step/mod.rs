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

use super::planner::{FourStepPlanConfig, StockhamPlanConfig};
use super::stockham::NttTimings;

/// Which transpose kernel the four-step plan uses.
///
/// `Tile16` is the legacy 16x16 kernel with one element per thread
/// (`babybear_fourstep_transpose.wgsl`). `Tile32` is the tuned 32x32
/// variant with 4 elements per thread (`babybear_fourstep_transpose_tiled32.wgsl`).
///
/// The active variant is selected via the `ZKGPU_TRANSPOSE_VARIANT`
/// environment variable (`tile16` or `tile32`); unset defaults to `Tile16`.
/// Selection happens once at `FourStepPlan::new` and is frozen for the
/// lifetime of the plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TransposeVariant {
    Tile16,
    // Only reachable via `ZKGPU_TRANSPOSE_VARIANT=tile32`, which the wasm
    // `from_env()` branch ignores (no env on the web). That makes `Tile32`
    // unconstructed on wasm and triggers `dead_code`. Kept in-tree per the
    // NVIDIA scale-up T3.E / FTL A/B measurements — the variant is behind
    // the env gate for future experimentation, not orphaned.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    Tile32,
}

impl TransposeVariant {
    /// Read the variant from `ZKGPU_TRANSPOSE_VARIANT`, defaulting to `Tile16`.
    ///
    /// FTL A/B on 2026-04-14 (Samsung A56 / Xclipse 540 and Pixel 9 Pro /
    /// Mali-G715) measured tile32's transpose win at ≤5.2% of four-step total
    /// GPU time — below the ≥20% keeper threshold. Tile32 stays in-tree behind
    /// the env var for future experimentation, but `Tile16` remains the
    /// universal default.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn from_env() -> Self {
        match std::env::var("ZKGPU_TRANSPOSE_VARIANT").as_deref() {
            Ok("tile32") => Self::Tile32,
            _ => Self::Tile16,
        }
    }

    /// wasm target has no env; always use the legacy variant there.
    #[cfg(target_arch = "wasm32")]
    pub(crate) fn from_env() -> Self {
        Self::Tile16
    }

    /// Tile edge length (shared-memory tile is `tile_size × (tile_size+1)`).
    pub(crate) fn tile_size(self) -> u32 {
        match self {
            Self::Tile16 => 16,
            Self::Tile32 => 32,
        }
    }
}

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
    // NVIDIA scale-up T3.A (2026-04-17): R8 leaf pipeline — 3 logical
    // Stockham stages per dispatch, vs R4's 2 and R2's 1. Planner greedy-
    // factors leaves as R8+R4+R2 via `StockhamPlanConfig::new_global_only`.
    pub(super) leaf_r8_pipeline: Arc<wgpu::ComputePipeline>,
    pub(super) leaf_bgl: Arc<wgpu::BindGroupLayout>,

    pub(super) twiddle_pipeline: Arc<wgpu::ComputePipeline>,
    pub(super) twiddle_bgl: Arc<wgpu::BindGroupLayout>,
    pub(super) twiddle_buffer: wgpu::Buffer,
    // NVIDIA scale-up Tier 2A Option A (2026-04-16): the companion
    // `twiddle_prime_buffer` field was dropped. The diagonal twiddle
    // shader now uses `mod_mul` (10-iter reducer) instead of Shoup's
    // `mod_mul_shoup`, eliminating the 16 MiB prime buffer's L2
    // pressure at log ≥ 22.
    pub(super) twiddle_param_buffer: wgpu::Buffer,
    pub(super) twiddle_dispatch: LinearDispatch,

    pub(super) transpose_pipeline: Arc<wgpu::ComputePipeline>,
    pub(super) transpose_bgl: Arc<wgpu::BindGroupLayout>,
    /// Which transpose kernel is active (frozen at plan construction).
    pub(super) transpose_variant: TransposeVariant,
    /// R×C → C×R transpose params (phases 1 and 6)
    pub(super) transpose_rc_to_cr_params: wgpu::Buffer,
    /// C×R → R×C transpose params (phase 4)
    pub(super) transpose_cr_to_rc_params: wgpu::Buffer,

    pub(super) scale_pipeline: Arc<wgpu::ComputePipeline>,
    pub(super) scale_bgl: Arc<wgpu::BindGroupLayout>,
    pub(super) scale_param_buffer: Option<wgpu::Buffer>,
    pub(super) scale_dispatch: LinearDispatch,

    /// Phase-2 leaf: R-point NTTs (col_leaf config), C batches
    pub(super) phase2_r8_twiddle_buffer: wgpu::Buffer,
    pub(super) phase2_r8_twiddle_prime_buffer: wgpu::Buffer,
    pub(super) phase2_r8_stage_param_buffers: Vec<wgpu::Buffer>,
    pub(super) phase2_r4_twiddle_buffer: wgpu::Buffer,
    pub(super) phase2_r4_twiddle_prime_buffer: wgpu::Buffer,
    pub(super) phase2_r4_stage_param_buffers: Vec<wgpu::Buffer>,
    pub(super) phase2_r2_twiddle_buffer: wgpu::Buffer,
    pub(super) phase2_r2_twiddle_prime_buffer: wgpu::Buffer,
    pub(super) phase2_r2_stage_param_buffers: Vec<wgpu::Buffer>,

    /// Phase-5 leaf: C-point NTTs (row_leaf config), R batches
    pub(super) phase5_r8_twiddle_buffer: wgpu::Buffer,
    pub(super) phase5_r8_twiddle_prime_buffer: wgpu::Buffer,
    pub(super) phase5_r8_stage_param_buffers: Vec<wgpu::Buffer>,
    pub(super) phase5_r4_twiddle_buffer: wgpu::Buffer,
    pub(super) phase5_r4_twiddle_prime_buffer: wgpu::Buffer,
    pub(super) phase5_r4_stage_param_buffers: Vec<wgpu::Buffer>,
    pub(super) phase5_r2_twiddle_buffer: wgpu::Buffer,
    pub(super) phase5_r2_twiddle_prime_buffer: wgpu::Buffer,
    pub(super) phase5_r2_stage_param_buffers: Vec<wgpu::Buffer>,

    // NVIDIA scale-up Tier 1 Fix 2b (2026-04-16): 2D-folded leaf
    // dispatches. Phase 2 and Phase 5 each cover `n/radix` butterflies
    // across their batches — same count across phases. At log_n ≥ 26
    // the 1D grid exceeds 65535 workgroups; 2D fold wraps into gid.y.
    //
    // T3.A (2026-04-17) adds `leaf_r8_dispatch` for R8 kernels that
    // cover n/8 butterflies.
    pub(super) leaf_r8_dispatch: LinearDispatch,
    pub(super) leaf_r4_dispatch: LinearDispatch,
    pub(super) leaf_r2_dispatch: LinearDispatch,

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
        // NVIDIA scale-up T3.A (2026-04-17): leaf labels now emit in
        // dispatch order (R8 → R4 → R2), matching `encode_batched_leaf_r4`.
        // Helper: compute leaf stage labels for a given phase.
        fn leaf_labels(
            phase: &str,
            label_stem: &str,
            leaf: &StockhamPlanConfig,
        ) -> Vec<String> {
            let mut out = Vec::new();
            // R8 stages (3 logical stages each, starting at h = 0, 3, 6, ...)
            for i in 0..leaf.r8_stage_params.len() {
                let h = i as u32 * 3;
                out.push(format!("{phase} {label_stem} r8 stages {}+{}+{}", h, h + 1, h + 2));
            }
            // R4 stages (2 logical stages each, starting after the R8 stages)
            let r4_start_h = leaf.r8_stage_params.len() as u32 * 3;
            for i in 0..leaf.r4_stage_params.len() {
                let h = r4_start_h + i as u32 * 2;
                out.push(format!("{phase} {label_stem} r4 stages {}+{}", h, h + 1));
            }
            // R2 residue (at most one dispatch, at the final h)
            let r2_start_h = r4_start_h + leaf.r4_stage_params.len() as u32 * 2;
            for i in 0..leaf.global_stage_params.len() {
                let h = r2_start_h + i as u32;
                out.push(format!("{phase} {label_stem} r2 stage {h}"));
            }
            out
        }

        let mut labels = Vec::new();
        labels.push("transpose R\u{d7}C\u{2192}C\u{d7}R".to_string());
        labels.extend(leaf_labels("phase2", "R-pt", &self.config.col_leaf));
        labels.push("twiddle multiply".to_string());
        labels.push("transpose C\u{d7}R\u{2192}R\u{d7}C".to_string());
        labels.extend(leaf_labels("phase5", "C-pt", &self.config.row_leaf));
        labels.push("transpose R\u{d7}C\u{2192}C\u{d7}R (output)".to_string());
        if self.scale_param_buffer.is_some() {
            labels.push("inverse scale".to_string());
        }
        labels
    }
}
