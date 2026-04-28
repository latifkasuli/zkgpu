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
pub(crate) struct StockhamPlan {
    pub(super) global_pipeline: Arc<wgpu::ComputePipeline>,
    pub(super) r4_pipeline: Arc<wgpu::ComputePipeline>,
    pub(super) local_pipeline: Arc<wgpu::ComputePipeline>,
    pub(super) ntt_bind_group_layout: Arc<wgpu::BindGroupLayout>,

    pub(super) global_twiddle_buffer: wgpu::Buffer,
    pub(super) global_twiddle_prime_buffer: wgpu::Buffer,
    pub(super) global_stage_param_buffers: Vec<wgpu::Buffer>,

    pub(super) r4_twiddle_buffer: wgpu::Buffer,
    pub(super) r4_twiddle_prime_buffer: wgpu::Buffer,
    pub(super) r4_stage_param_buffers: Vec<wgpu::Buffer>,

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
