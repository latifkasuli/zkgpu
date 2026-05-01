mod build;
mod encode;
mod profiled;

use std::sync::Arc;

use zkgpu_babybear::BabyBear;
use zkgpu_core::ZkGpuError;

use crate::async_util;
use crate::buffer::WgpuBuffer;
use crate::device::WgpuDevice;
use crate::dispatch::LinearDispatch;
use crate::profiling::GpuTiming;

use super::planner::StockhamPlanConfig;

/// GPU NTT execution plan for BabyBear fields using Stockham autosort.
///
/// Two-phase hybrid architecture:
///
/// - **N < BLOCK_SIZE**: all stages run as global DIF Stockham passes
///   (one dispatch per stage, ping-pong between data and scratch).
///
/// - **N >= BLOCK_SIZE**: the first `log_n - LOG_BLOCK` stages run as
///   global DIF Stockham passes, reducing the problem to N/BLOCK_SIZE
///   independent sub-problems. A single workgroup-local dispatch then
///   finishes all remaining LOG_BLOCK stages in shared memory.
///
/// For inverse NTT, a final GPU dispatch multiplies every element by
/// `n^{-1} mod P`, avoiding a host round-trip.
///
/// **Item #3 (Gate 2):** the radix-4 global stage carries its
/// per-stage params via either a uniform buffer (the production
/// default, [`R4ParamSource::Storage`]) or via
/// `wgpu::Features::IMMEDIATES` register writes
/// ([`R4ParamSource::Immediate`]). The choice is fixed at plan
/// build. `WgpuNttPlan::new` always picks Storage per the verdict in
/// `docs/research/r4-immediate-pilot-verdict.md` (no host cleared
/// the propagation gate); `WgpuNttPlan::new_with_r4_param_mode`
/// exposes the Immediate path for benches and downstream callers
/// who measure on their own hardware. R2 global stages and the
/// local-fused kernel keep the uniform path regardless; only R4
/// has a dual implementation.
pub(crate) struct StockhamPlan {
    pub(super) global_pipeline: Arc<wgpu::ComputePipeline>,
    pub(super) r4_pipeline: Arc<wgpu::ComputePipeline>,
    pub(super) local_pipeline: Arc<wgpu::ComputePipeline>,
    pub(super) ntt_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    /// R4 BGL — equal to `ntt_bind_group_layout` in `Storage` mode,
    /// a distinct 4-entry layout (no slot 3) in `Immediate` mode.
    pub(super) r4_bind_group_layout: Arc<wgpu::BindGroupLayout>,

    pub(super) global_twiddle_buffer: wgpu::Buffer,
    pub(super) global_twiddle_prime_buffer: wgpu::Buffer,
    pub(super) global_stage_param_buffers: Vec<wgpu::Buffer>,

    pub(super) r4_twiddle_buffer: wgpu::Buffer,
    pub(super) r4_twiddle_prime_buffer: wgpu::Buffer,
    /// R4 stage params, stored in the shape the active mode needs.
    pub(super) r4_param_source: R4ParamSource,

    pub(super) local_twiddle_buffer: wgpu::Buffer,
    pub(super) local_twiddle_prime_buffer: wgpu::Buffer,
    pub(super) local_param_buffer: wgpu::Buffer,

    pub(super) scratch_buffer: wgpu::Buffer,

    pub(super) scale_pipeline: Arc<wgpu::ComputePipeline>,
    pub(super) scale_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    pub(super) scale_param_buffer: Option<wgpu::Buffer>,
    pub(super) scale_dispatch: LinearDispatch,

    // NVIDIA scale-up Tier 1 Fix 2 (2026-04-16): 2D-folded dispatch
    // grids for the three Stockham dispatch sites. All three compute
    // the same-sized workgroup grid because Stockham stages all cover
    // `n/2` butterflies for R2, `n/4` for R4, and `n/BLOCK_SIZE`
    // local workgroups. Using 2D grids lets log_n ≥ 25 workloads
    // dispatch without exceeding the wgpu per-dimension limit.
    pub(super) r2_dispatch: LinearDispatch,
    pub(super) r4_dispatch: LinearDispatch,
    pub(super) local_dispatch: LinearDispatch,

    pub(super) config: StockhamPlanConfig,
}

/// Optional GPU timing results from a profiled NTT execution.
#[derive(Debug, Clone)]
pub struct NttTimings {
    pub wall_clock: std::time::Duration,
    pub gpu_stage_ns: Vec<GpuTiming>,
    pub gpu_total_ns: f64,
}

/// How R4 stage params reach the kernel. Item #3 of
/// `docs/research/zkgpu-wgpu-speed-opportunities.md`.
///
/// Production callers go through [`crate::WgpuNttPlan::new`] which
/// always picks [`Storage`](Self::Storage) — see the verdict in
/// `docs/research/r4-immediate-pilot-verdict.md` for why no host
/// cleared the propagation gate. This enum is exposed publicly so
/// benches and downstream callers who measure on their own hardware
/// can opt into [`Immediate`](Self::Immediate) explicitly via
/// [`crate::WgpuNttPlan::new_with_r4_param_mode`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum R4ParamMode {
    /// Per-stage uniform buffer at bind-group entry 3. The shipping
    /// default on every backend. Works whether or not
    /// `Features::IMMEDIATES` is advertised.
    Storage,
    /// Per-stage params flow through `wgpu::Features::IMMEDIATES` —
    /// `set_immediates` writes 32 bytes directly to register-resident
    /// space. Drops the bind-group entry and the per-stage
    /// `wgpu::Buffer` allocation. Requires `caps.has_immediates`;
    /// constructing a plan with this variant on a device that
    /// doesn't advertise the feature returns `InvalidNttSize`.
    Immediate,
}

/// Internal binding state for R4 stage params. See [`R4ParamMode`].
pub(super) enum R4ParamSource {
    /// One uniform buffer per stage; bound at entry 3 in encode.
    Storage(Vec<wgpu::Buffer>),
    /// One 32-byte param block per stage; written via
    /// `pass.set_immediates(0, ...)` before each R4 dispatch.
    Immediate(Vec<[u32; 8]>),
}

impl R4ParamSource {
    pub(super) fn len(&self) -> usize {
        match self {
            R4ParamSource::Storage(v) => v.len(),
            R4ParamSource::Immediate(v) => v.len(),
        }
    }
}

impl StockhamPlan {
    /// Total number of GPU dispatches (NTT stages + optional scaling).
    pub(crate) fn num_dispatches(&self) -> u32 {
        self.config.ntt_dispatches() + u32::from(self.scale_param_buffer.is_some())
    }

    /// Tail strategy label for reporting; `None` when the plan has no tail.
    pub(crate) fn tail_strategy_label(&self) -> Option<&'static str> {
        self.config.tail.as_ref().map(|t| t.strategy.as_str())
    }

    /// Tail reason label for reporting; `None` when the plan has no tail.
    pub(crate) fn tail_reason_label(&self) -> Option<&'static str> {
        self.config.tail.as_ref().map(|t| t.reason.as_str())
    }

    /// Per-thread gather stride in bytes for the local-fused tail.
    pub(crate) fn tail_stride_bytes(&self) -> Option<u64> {
        self.config.tail_stride_bytes()
    }

    /// Execute the full NTT (stages + inverse scaling if applicable) in a
    /// single command submission. Entirely GPU-side — no host round-trips.
    pub(crate) fn execute_kernels(
        &mut self,
        device: &WgpuDevice,
        buf: &mut WgpuBuffer<BabyBear>,
    ) -> Result<(), ZkGpuError> {
        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Stockham NTT encoder"),
            });

        self.encode_ntt_stages(
            &device.device,
            &mut encoder,
            buf,
            &[],
            encode::NttEncodeMode::Folded,
        );
        self.encode_scale(&device.device, &mut encoder, buf, None);

        device.queue.submit(Some(encoder.finish()));
        device
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| ZkGpuError::BackendError(Box::new(e)))?;

        Ok(())
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
                label: Some("Stockham NTT encoder"),
            });

        self.encode_ntt_stages(
            &device.device,
            &mut encoder,
            buf,
            &[],
            encode::NttEncodeMode::Folded,
        );
        self.encode_scale(&device.device, &mut encoder, buf, None);

        device.queue.submit(Some(encoder.finish()));
        async_util::wait_for_submission(&device.device, &device.queue).await?;

        Ok(())
    }
}
