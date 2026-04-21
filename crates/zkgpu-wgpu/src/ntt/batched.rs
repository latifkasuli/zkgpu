//! Phase 7.5 — column-batched NTT plan.
//!
//! The existing [`WgpuNttPlan`] is single-poly in-place. This module
//! adds a sibling plan `WgpuBatchedNttPlan` that processes a
//! row-major `h × w` matrix in a single command submission.
//!
//! **Path C.1 (2026-04-21)**: the plan now chains radix-4 stages
//! (pairs of R2 stages fused) for the bulk of the NTT plus a single
//! R2 stage for the remainder when `log_n` is odd. Same kernel shape
//! as the single-column `StockhamPlan`'s R4 + R2 chain, plus a
//! pitched storage layout where each row is padded to a multiple of
//! 8 u32s so coalesced loads stay aligned.
//!
//! Scope: BabyBear, Stockham R4 + R2 remainder, forward + inverse.
//! No workgroup-local fused tail (that's Path C.2, once the portable
//! one-column-per-workgroup variant is designed).
//!
//! Supported `log_n` range: 1..=22 (same storage-buffer cap as
//! Path B).

use std::sync::Arc;

use wgpu::util::DeviceExt;
use zkgpu_babybear::BabyBear;
#[cfg(test)]
use zkgpu_core::GpuDevice;
use zkgpu_core::{GpuBuffer, GpuField, NttDirection, ZkGpuError};

use crate::buffer::WgpuBuffer;
use crate::device::WgpuDevice;
use crate::dispatch::{plan_linear_dispatch, LinearDispatch};

use super::babybear_twiddles::shoup_quotient;
use super::common::{bgl_storage_entry, bgl_uniform_entry};
use super::planner::{MAX_BABYBEAR_LOG_N, WORKGROUP_SIZE};

const BATCHED_R2_SOURCE: &str =
    include_str!("../kernels/portable/babybear_stockham_r2_batched.wgsl");
const BATCHED_R4_SOURCE: &str =
    include_str!("../kernels/portable/babybear_stockham_r4_batched.wgsl");
const SCALE_SOURCE: &str = include_str!("../kernels/portable/babybear_scale.wgsl");

const BATCHED_BGL_LABEL: &str = "Batched Stockham NTT bind group layout";
const BATCHED_SCALE_BGL_LABEL: &str = "Batched scale bind group layout";

const MAX_BATCHED_LOG_N: u32 = 22;

/// Round `width` up to the nearest multiple of 8 (u32 elements), so
/// every row in the batched buffer starts on a 32-byte boundary.
/// Keeps coalesced loads aligned on all typical GPU architectures.
#[inline]
fn round_pitch(width: u32) -> u32 {
    (width + 7) & !7u32
}

/// Column-batched NTT plan for BabyBear. Processes an `h × w`
/// row-major matrix via pitched storage (`h × pitch`, with
/// `pitch = round_up(w, 8)`).
pub struct WgpuBatchedNttPlan {
    // Pipelines + bind-group layout shared between R2 and R4 (same
    // entry-point layout: src/dst/twiddles/params/twiddles_prime).
    r4_pipeline: Arc<wgpu::ComputePipeline>,
    r2_pipeline: Arc<wgpu::ComputePipeline>,
    bind_group_layout: Arc<wgpu::BindGroupLayout>,

    // Twiddle tables. R4 stages need `3 * m4` entries per stage
    // (three twiddles per R4 butterfly); R2 remainder stage needs `m`
    // entries. Stored in separate buffers for clarity even though
    // they could share.
    r4_twiddle_buffer: wgpu::Buffer,
    r4_twiddle_prime_buffer: wgpu::Buffer,
    r2_twiddle_buffer: wgpu::Buffer,
    r2_twiddle_prime_buffer: wgpu::Buffer,

    // Per-stage parameter buffers, in dispatch order: all R4 stages
    // first, then R2 remainder (if `log_n` is odd).
    r4_stage_param_buffers: Vec<wgpu::Buffer>,
    r2_stage_param_buffers: Vec<wgpu::Buffer>,

    scratch_buffer: wgpu::Buffer,

    // Total dispatch for one R4 stage = (n/4) * width threads.
    r4_dispatch: LinearDispatch,
    // Total dispatch for R2 remainder = (n/2) * width threads.
    r2_dispatch: LinearDispatch,

    scale_pipeline: Arc<wgpu::ComputePipeline>,
    scale_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    scale_param_buffer: Option<wgpu::Buffer>,
    scale_dispatch: LinearDispatch,

    log_n: u32,
    n: u32,
    width: u32,
    pitch: u32,
    /// `true` when the final stage writes into scratch and we need to
    /// copy back into the caller's buffer.
    result_in_scratch: bool,

    direction: NttDirection,
}

/// Parameters for the R4 batched kernel. Must match `R4BatchedParams`
/// in `babybear_stockham_r4_batched.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct R4BatchedStageParams {
    n: u32,
    s: u32,
    m4: u32,
    twiddle_offset: u32,
    omega4: u32,
    omega4_prime: u32,
    width: u32,
    pitch: u32,
    groups_per_row: u32,
    _pad0: u32,
}

/// Parameters for the R2 batched kernel. Must match
/// `BatchedStockhamParams` in `babybear_stockham_r2_batched.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct R2BatchedStageParams {
    n: u32,
    s: u32,
    m: u32,
    twiddle_offset: u32,
    width: u32,
    pitch: u32,
    groups_per_row: u32,
    _pad0: u32,
}

impl WgpuBatchedNttPlan {
    /// Build a plan for an `h × w` NTT where `h = 2^log_n`.
    pub fn new(
        device: &WgpuDevice,
        log_n: u32,
        width: u32,
        direction: NttDirection,
    ) -> Result<Self, ZkGpuError> {
        if log_n == 0 || log_n > MAX_BABYBEAR_LOG_N {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "batched plan requires 1 <= log_n <= {MAX_BABYBEAR_LOG_N}, got {log_n}"
            )));
        }
        if log_n > MAX_BATCHED_LOG_N {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "batched plan caps at log_n <= {MAX_BATCHED_LOG_N} (got {log_n}); \
                 use the column-loop path instead"
            )));
        }
        if width == 0 {
            return Err(ZkGpuError::InvalidNttSize(
                "batched plan requires width >= 1".to_string(),
            ));
        }

        let n = 1u32 << log_n;
        let pitch = round_pitch(width);
        let per_buffer_bytes = (n as u64) * (pitch as u64) * 4;

        // Per-buffer preflight.
        let storage_limit = device.caps.max_storage_buffer_size();
        if per_buffer_bytes > storage_limit {
            return Err(ZkGpuError::BufferSize {
                requested: per_buffer_bytes,
                limit: storage_limit,
            });
        }

        // Aggregate live-footprint preflight.
        // Live: user data + scratch + r4 twiddles (3*(n-1) max) +
        // r4 twiddles_prime + r2 twiddles + r2 twiddles_prime.
        // Conservative upper bound on twiddles: 4 * n * 4 bytes.
        let twiddle_bytes = 4u64 * (n as u64) * 4;
        let total_live_bytes = 2 * per_buffer_bytes + twiddle_bytes;
        if total_live_bytes > device.caps.max_buffer_size {
            return Err(ZkGpuError::BufferSize {
                requested: total_live_bytes,
                limit: device.caps.max_buffer_size,
            });
        }

        #[cfg(not(target_arch = "wasm32"))]
        let validation_scope = device.push_validation_scope();
        let reg = &device.pipelines;

        // --- Pipelines: R4 and R2 share the same bind-group layout ---
        let bind_group_layout = reg.get_or_create_bgl(
            &device.device,
            BATCHED_BGL_LABEL,
            &[
                bgl_storage_entry(0, true),
                bgl_storage_entry(1, false),
                bgl_storage_entry(2, true),
                bgl_uniform_entry(3),
                bgl_storage_entry(4, true),
            ],
        );
        let pipeline_layout = device
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Batched Stockham NTT pipeline layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });

        let r4_module = reg.get_or_create_module(
            &device.device,
            BATCHED_R4_SOURCE,
            "Batched Stockham R4 shader",
        );
        let r4_pipeline = reg.get_or_create_pipeline(
            &device.device,
            BATCHED_R4_SOURCE,
            "batched_stockham_r4_butterfly",
            BATCHED_BGL_LABEL,
            &pipeline_layout,
            &r4_module,
            device.pipeline_cache.as_ref(),
        );
        let r2_module = reg.get_or_create_module(
            &device.device,
            BATCHED_R2_SOURCE,
            "Batched Stockham R2 shader",
        );
        let r2_pipeline = reg.get_or_create_pipeline(
            &device.device,
            BATCHED_R2_SOURCE,
            "batched_stockham_r2_butterfly",
            BATCHED_BGL_LABEL,
            &pipeline_layout,
            &r2_module,
            device.pipeline_cache.as_ref(),
        );

        // --- Scale pipeline (reuse single-poly scale kernel) ---
        let scale_bind_group_layout = reg.get_or_create_bgl(
            &device.device,
            BATCHED_SCALE_BGL_LABEL,
            &[bgl_storage_entry(0, false), bgl_uniform_entry(1)],
        );
        let scale_pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Batched scale pipeline layout"),
                    bind_group_layouts: &[Some(&scale_bind_group_layout)],
                    immediate_size: 0,
                });
        let scale_module = reg.get_or_create_module(
            &device.device,
            SCALE_SOURCE,
            "Batched scale shader",
        );
        let scale_pipeline = reg.get_or_create_pipeline(
            &device.device,
            SCALE_SOURCE,
            "scale_elements",
            BATCHED_SCALE_BGL_LABEL,
            &scale_pipeline_layout,
            &scale_module,
            device.pipeline_cache.as_ref(),
        );

        #[cfg(not(target_arch = "wasm32"))]
        device.pop_validation_scope(validation_scope, "batched plan pipeline creation")?;

        // --- Decide R4/R2 stage layout ---
        // R4 fuses stages (h, h+1). With log_n stages total, we can
        // run floor(log_n / 2) R4 dispatches plus (log_n % 2) R2
        // dispatches. R4 stages execute at the "lower" h values first
        // (stages 0+1, 2+3, ...); if log_n is odd the topmost stage
        // (h = log_n - 1) runs as R2.
        let num_r4_pairs: u32 = log_n / 2;
        let has_r2_remainder = (log_n % 2) == 1;

        // --- Twiddle tables ---
        let omega = match direction {
            NttDirection::Forward => BabyBear::root_of_unity(log_n),
            NttDirection::Inverse => BabyBear::root_of_unity(log_n)
                .inv()
                .expect("root of unity is invertible"),
        };

        // omega4 = omega^(n/4) is the principal 4th root of unity in
        // BabyBear (same convention as the single-column R4 kernel).
        let omega4 = omega.pow((n / 4) as u64);
        let omega4_repr = omega4.to_repr();
        let omega4_prime = shoup_quotient(omega4_repr);

        // R4 twiddles: for each R4 pair k ∈ [0, num_r4_pairs), the
        // lower stage has h = 2k, s = 2^h, m = n/2^(h+1), m4 = m/2.
        // For each butterfly p ∈ [0, m4), we store three twiddles:
        //   w1 = omega^(s * p)
        //   w2 = omega^(s * p)^2  = omega^(2 s p)
        //   w3 = omega^(s * p)^3  = omega^(3 s p)
        // The single-column R4 kernel uses the same layout.
        let mut r4_twiddles: Vec<u32> = Vec::new();
        let mut r4_twiddles_prime: Vec<u32> = Vec::new();
        let mut r4_stage_offsets: Vec<u32> = Vec::with_capacity(num_r4_pairs as usize);
        for k in 0..num_r4_pairs {
            let h = 2 * k;
            let s = 1u32 << h;
            let m = n >> (h + 1);
            let m4 = m / 2;
            let step = omega.pow(s as u64);
            r4_stage_offsets.push(r4_twiddles.len() as u32);
            let mut w_cur = BabyBear::ONE;
            for _ in 0..m4 {
                let r1 = w_cur;
                let r2 = r1 * r1;
                let r3 = r2 * r1;
                for r in [r1, r2, r3] {
                    let repr = r.to_repr();
                    r4_twiddles.push(repr);
                    r4_twiddles_prime.push(shoup_quotient(repr));
                }
                w_cur = w_cur * step;
            }
        }

        // R2 remainder twiddles: if log_n is odd, the topmost stage
        // h = log_n - 1 runs as R2, same layout as the R2-only plan.
        let mut r2_twiddles: Vec<u32> = Vec::new();
        let mut r2_twiddles_prime: Vec<u32> = Vec::new();
        let mut r2_stage_offsets: Vec<u32> = Vec::new();
        if has_r2_remainder {
            let h = log_n - 1;
            let s = 1u32 << h;
            let m = n >> (h + 1);
            let step = omega.pow(s as u64);
            r2_stage_offsets.push(0);
            let mut w_cur = BabyBear::ONE;
            for _ in 0..m {
                let repr = w_cur.to_repr();
                r2_twiddles.push(repr);
                r2_twiddles_prime.push(shoup_quotient(repr));
                w_cur = w_cur * step;
            }
        }

        let r4_twiddle_buffer = create_storage_init(
            &device.device,
            "Batched R4 twiddles",
            if r4_twiddles.is_empty() { &[0u32] } else { &r4_twiddles },
        );
        let r4_twiddle_prime_buffer = create_storage_init(
            &device.device,
            "Batched R4 twiddles'",
            if r4_twiddles_prime.is_empty() { &[0u32] } else { &r4_twiddles_prime },
        );
        let r2_twiddle_buffer = create_storage_init(
            &device.device,
            "Batched R2 twiddles",
            if r2_twiddles.is_empty() { &[0u32] } else { &r2_twiddles },
        );
        let r2_twiddle_prime_buffer = create_storage_init(
            &device.device,
            "Batched R2 twiddles'",
            if r2_twiddles_prime.is_empty() { &[0u32] } else { &r2_twiddles_prime },
        );

        // --- Per-stage dispatch planning ---
        // R4 threads per stage = (n/4) * width; R2 remainder threads
        // per stage = (n/2) * width.
        let r4_dispatch = plan_linear_dispatch(
            (n / 4) * width,
            WORKGROUP_SIZE,
            device.caps.max_compute_workgroups_per_dimension,
        )?;
        let r2_dispatch = plan_linear_dispatch(
            (n / 2) * width,
            WORKGROUP_SIZE,
            device.caps.max_compute_workgroups_per_dimension,
        )?;

        // R4 stage params
        let r4_stage_param_buffers: Vec<wgpu::Buffer> = (0..num_r4_pairs)
            .map(|k| {
                let h = 2 * k;
                let s = 1u32 << h;
                let m = n >> (h + 1);
                let m4 = m / 2;
                let params = R4BatchedStageParams {
                    n,
                    s,
                    m4,
                    twiddle_offset: r4_stage_offsets[k as usize],
                    omega4: omega4_repr,
                    omega4_prime,
                    width,
                    pitch,
                    groups_per_row: r4_dispatch.groups_per_row,
                    _pad0: 0,
                };
                device
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Batched R4 stage params"),
                        contents: bytemuck::cast_slice(&[params]),
                        usage: wgpu::BufferUsages::UNIFORM,
                    })
            })
            .collect();

        let r2_stage_param_buffers: Vec<wgpu::Buffer> = if has_r2_remainder {
            let h = log_n - 1;
            let s = 1u32 << h;
            let m = n >> (h + 1);
            let params = R2BatchedStageParams {
                n,
                s,
                m,
                twiddle_offset: 0,
                width,
                pitch,
                groups_per_row: r2_dispatch.groups_per_row,
                _pad0: 0,
            };
            vec![device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Batched R2 remainder stage params"),
                    contents: bytemuck::cast_slice(&[params]),
                    usage: wgpu::BufferUsages::UNIFORM,
                })]
        } else {
            Vec::new()
        };

        // --- Scratch buffer (h × pitch × 4 bytes) ---
        let scratch_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Batched NTT scratch"),
            size: per_buffer_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // --- Scale (only if inverse) ---
        // Scales every element in the h × pitch buffer by 1/n.
        // The padded columns (c ∈ [width, pitch)) hold zero or junk;
        // scaling them is harmless since we never read those columns
        // back in the download pass.
        let scale_total_elems = (n * pitch).div_ceil(4);
        let scale_dispatch = plan_linear_dispatch(
            scale_total_elems,
            WORKGROUP_SIZE,
            device.caps.max_compute_workgroups_per_dimension,
        )?;
        let scale_param_buffer = if direction == NttDirection::Inverse {
            let n_field = BabyBear::new(n);
            let n_inv = n_field.inv().expect("n must be invertible in BabyBear");
            // ScaleParams layout from babybear_scale.wgsl:
            //   { n: u32, scalar: u32, groups_per_row: u32, _pad1: u32 }
            // We pass the full (n * pitch) as the element count.
            let params = [n * pitch, n_inv.to_repr(), scale_dispatch.groups_per_row, 0u32];
            Some(
                device
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Batched inverse scale params"),
                        contents: bytemuck::cast_slice(&params),
                        usage: wgpu::BufferUsages::UNIFORM,
                    }),
            )
        } else {
            None
        };

        device.save_pipeline_cache();

        // Parity of total dispatches determines whether the result
        // lives in buf (even) or scratch (odd) after ping-pong.
        let total_dispatches = num_r4_pairs + r2_stage_param_buffers.len() as u32;
        let result_in_scratch = total_dispatches % 2 == 1;

        Ok(Self {
            r4_pipeline,
            r2_pipeline,
            bind_group_layout,
            r4_twiddle_buffer,
            r4_twiddle_prime_buffer,
            r2_twiddle_buffer,
            r2_twiddle_prime_buffer,
            r4_stage_param_buffers,
            r2_stage_param_buffers,
            scratch_buffer,
            r4_dispatch,
            r2_dispatch,
            scale_pipeline,
            scale_bind_group_layout,
            scale_param_buffer,
            scale_dispatch,
            log_n,
            n,
            width,
            pitch,
            result_in_scratch,
            direction,
        })
    }

    /// Plan's NTT size `h = 2^log_n` (the per-column length).
    pub fn log_n(&self) -> u32 {
        self.log_n
    }

    /// Plan's batch width (number of columns).
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Plan's padded stride in u32 elements (>= width, multiple of 8).
    /// Callers uploading to the batched plan must use this stride.
    pub fn pitch(&self) -> u32 {
        self.pitch
    }

    /// Execute one full batched NTT.
    ///
    /// `buf` must have exactly `h × pitch` elements in pitched
    /// row-major layout (row `r`, column `c` lives at
    /// `buf[r * pitch + c]`). For `c ∈ [width, pitch)` the value is
    /// ignored — padding rows are scratch. See [`Self::pitch`].
    pub fn execute(
        &mut self,
        device: &WgpuDevice,
        buf: &mut WgpuBuffer<BabyBear>,
    ) -> Result<(), ZkGpuError> {
        let expected_len = (self.n as usize) * (self.pitch as usize);
        if buf.len() != expected_len {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "batched buffer length mismatch: plan expects {} elements \
                 (h={}, pitch={}), got {}",
                expected_len,
                self.n,
                self.pitch,
                buf.len()
            )));
        }

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Batched Stockham NTT encoder"),
            });

        let mut dispatch_idx: usize = 0;

        // Phase 1: R4 stages (stage-pairs 0+1, 2+3, ...).
        for param_buffer in self.r4_stage_param_buffers.iter() {
            let (src_buf, dst_buf) = if dispatch_idx % 2 == 0 {
                (&buf.inner, &self.scratch_buffer)
            } else {
                (&self.scratch_buffer, &buf.inner)
            };

            let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Batched R4 stage bind group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: src_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: dst_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.r4_twiddle_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: param_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.r4_twiddle_prime_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Batched R4 stage"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.r4_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(self.r4_dispatch.x, self.r4_dispatch.y, 1);

            dispatch_idx += 1;
        }

        // Phase 2: R2 remainder (only if log_n is odd).
        for param_buffer in self.r2_stage_param_buffers.iter() {
            let (src_buf, dst_buf) = if dispatch_idx % 2 == 0 {
                (&buf.inner, &self.scratch_buffer)
            } else {
                (&self.scratch_buffer, &buf.inner)
            };

            let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Batched R2 remainder bind group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: src_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: dst_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.r2_twiddle_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: param_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.r2_twiddle_prime_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Batched R2 remainder"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.r2_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(self.r2_dispatch.x, self.r2_dispatch.y, 1);

            dispatch_idx += 1;
        }

        // Copy-back if result landed in scratch.
        if self.result_in_scratch {
            let size = (self.n as u64) * (self.pitch as u64) * 4;
            encoder.copy_buffer_to_buffer(&self.scratch_buffer, 0, &buf.inner, 0, size);
        }

        // Inverse scale.
        if let Some(ref scale_params) = self.scale_param_buffer {
            let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Batched scale bind group"),
                layout: &self.scale_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.inner.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: scale_params.as_entire_binding(),
                    },
                ],
            });

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Batched inverse scale"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scale_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(self.scale_dispatch.x, self.scale_dispatch.y, 1);
        }

        device.queue.submit(Some(encoder.finish()));
        device
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| ZkGpuError::BackendError(Box::new(e)))?;

        let _ = self.direction;
        Ok(())
    }
}

fn create_storage_init(
    device: &wgpu::Device,
    label: &str,
    contents: &[u32],
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(contents),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_values(len: usize, seed: u64) -> Vec<BabyBear> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                BabyBear::from_u64(state)
            })
            .collect()
    }

    /// Upload a `h × w` logical matrix into an `h × pitch` pitched
    /// buffer. Padding columns `[w, pitch)` are zero.
    fn upload_pitched(
        device: &WgpuDevice,
        values: &[BabyBear],
        h: usize,
        w: usize,
        pitch: usize,
    ) -> WgpuBuffer<BabyBear> {
        let mut padded: Vec<BabyBear> = vec![BabyBear::ZERO; h * pitch];
        for r in 0..h {
            for c in 0..w {
                padded[r * pitch + c] = values[r * w + c];
            }
        }
        device.upload(&padded).expect("upload padded")
    }

    /// Extract the `w` real columns out of a downloaded `h × pitch`
    /// buffer into a flat `h × w` result.
    fn strip_pitch(
        values: &[BabyBear],
        h: usize,
        w: usize,
        pitch: usize,
    ) -> Vec<BabyBear> {
        let mut out = vec![BabyBear::ZERO; h * w];
        for r in 0..h {
            for c in 0..w {
                out[r * w + c] = values[r * pitch + c];
            }
        }
        out
    }

    #[test]
    fn batched_r4_matches_single_column_stockham() {
        use super::super::WgpuNttPlan;

        let device = match WgpuDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping batched R4 test — no GPU device: {e}");
                return;
            }
        };

        // Test both even log_n (R4-only) and odd log_n (R4 + R2 rem).
        for &log_n in &[6u32, 7, 8, 10] {
            for &w in &[4usize, 8, 16] {
                let h = 1usize << log_n;
                let pitch = round_pitch(w as u32) as usize;

                let input = random_values(h * w, 0xCAFE_BABE_DEAD_BEEF ^ (log_n as u64) ^ ((w as u64) << 32));

                let mut batched_buf = upload_pitched(&device, &input, h, w, pitch);
                let mut plan = WgpuBatchedNttPlan::new(&device, log_n, w as u32, NttDirection::Forward)
                    .expect("batched plan build");
                plan.execute(&device, &mut batched_buf)
                    .expect("batched execute");
                let batched_raw = batched_buf.read_to_vec().expect("batched readback");
                let batched_out = strip_pitch(&batched_raw, h, w, pitch);

                // Single-column reference.
                let mut per_column_out = vec![BabyBear::ZERO; h * w];
                for c in 0..w {
                    let col: Vec<BabyBear> = (0..h).map(|r| input[r * w + c]).collect();
                    let mut col_buf = device.upload(&col).expect("upload col");
                    let mut plan =
                        WgpuNttPlan::new(&device, log_n, NttDirection::Forward)
                            .expect("single plan build");
                    plan.execute_kernels(&device, &mut col_buf)
                        .expect("single execute");
                    let col_out = col_buf.read_to_vec().expect("single readback");
                    for r in 0..h {
                        per_column_out[r * w + c] = col_out[r];
                    }
                }

                for (i, (a, b)) in batched_out.iter().zip(per_column_out.iter()).enumerate() {
                    assert_eq!(
                        a.to_repr(),
                        b.to_repr(),
                        "log_n={log_n} w={w} element {i} differs: batched={:?} per_column={:?}",
                        a, b
                    );
                }
            }
        }
    }

    #[test]
    fn batched_r4_inverse_roundtrip() {
        let device = match WgpuDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping inverse roundtrip — no GPU device: {e}");
                return;
            }
        };

        for &log_n in &[6u32, 8] {
            for &w in &[4usize, 8] {
                let h = 1usize << log_n;
                let pitch = round_pitch(w as u32) as usize;

                let input = random_values(h * w, 0x1234_5678_9ABC_DEF0 ^ (log_n as u64));

                // Forward NTT
                let mut buf = upload_pitched(&device, &input, h, w, pitch);
                let mut fwd = WgpuBatchedNttPlan::new(&device, log_n, w as u32, NttDirection::Forward)
                    .expect("fwd plan");
                fwd.execute(&device, &mut buf).expect("fwd exec");

                // Inverse NTT (operates on the pitched buffer directly)
                let mut inv = WgpuBatchedNttPlan::new(&device, log_n, w as u32, NttDirection::Inverse)
                    .expect("inv plan");
                inv.execute(&device, &mut buf).expect("inv exec");

                let raw_out = buf.read_to_vec().expect("readback");
                let out = strip_pitch(&raw_out, h, w, pitch);

                for (i, (a, b)) in out.iter().zip(input.iter()).enumerate() {
                    assert_eq!(
                        a.to_repr(),
                        b.to_repr(),
                        "log_n={log_n} w={w} element {i}: roundtrip differs: got={:?} want={:?}",
                        a, b
                    );
                }
            }
        }
    }
}
