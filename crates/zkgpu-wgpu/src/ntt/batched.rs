//! Phase 7.5 — column-batched NTT plan.
//!
//! The existing [`WgpuNttPlan`] is single-poly in-place. This module
//! adds a minimal sibling plan `WgpuBatchedNttPlan` that processes a
//! row-major `h × w` matrix in a single command submission: one
//! dispatch per stage, all `w` columns handled in parallel via threads
//! indexed `(butterfly_index, column_index)`.
//!
//! Scope: BabyBear, Stockham R2 (all stages), forward + inverse. No
//! R4 fast-path and no workgroup-local tail — those optimisations
//! belong in a follow-up if data justifies the extra WGSL. The R2-only
//! path is sufficient to answer the question "does batching beat the
//! column-loop path in [`zkgpu_plonky3::GpuDft`]?" because the win
//! comes from amortising upload/download/launch overhead, not from the
//! per-stage butterfly radix.
//!
//! Supported `log_n` range: 1..=22 on most GPUs (limited by
//! `max_storage_buffer_size` — `h × w × 4` bytes must fit). At
//! `log_n ≥ 23` the batched plan returns `BufferSize` and callers
//! should fall back to the column-loop path in
//! `zkgpu-plonky3::run_column_loop`.

use std::sync::Arc;

use wgpu::util::DeviceExt;
use zkgpu_babybear::BabyBear;
use zkgpu_core::{GpuBuffer, GpuField, NttDirection, ZkGpuError};

use crate::buffer::WgpuBuffer;
use crate::device::WgpuDevice;
use crate::dispatch::{plan_linear_dispatch, LinearDispatch};

use super::babybear_twiddles::shoup_quotient;
use super::common::{bgl_storage_entry, bgl_uniform_entry};
use super::planner::{MAX_BABYBEAR_LOG_N, WORKGROUP_SIZE};

const BATCHED_R2_SOURCE: &str =
    include_str!("../kernels/portable/babybear_stockham_r2_batched.wgsl");
const SCALE_SOURCE: &str = include_str!("../kernels/portable/babybear_scale.wgsl");

const BATCHED_BGL_LABEL: &str = "Batched Stockham NTT bind group layout";
const BATCHED_SCALE_BGL_LABEL: &str = "Batched scale bind group layout";

/// Upper limit on `log_n` for the batched plan. Below this the plan
/// allocates `h × w × 4` bytes for scratch, which on most GPUs caps
/// around `max_storage_buffer_size = 2 GiB` — plenty of headroom at
/// `log_n = 22, w = 16` (64 MiB). Above `log_n = 22` the batched path
/// is unlikely to amortize the larger allocation cost even if the GPU
/// has the memory, so callers fall back to the column-loop path.
const MAX_BATCHED_LOG_N: u32 = 22;

/// Column-batched NTT plan for BabyBear.
///
/// Processes an `h × w` row-major matrix where each column is an
/// independent forward (or inverse) NTT of length `h = 2^log_n`.
pub struct WgpuBatchedNttPlan {
    pipeline: Arc<wgpu::ComputePipeline>,
    bind_group_layout: Arc<wgpu::BindGroupLayout>,
    twiddle_buffer: wgpu::Buffer,
    twiddle_prime_buffer: wgpu::Buffer,
    stage_param_buffers: Vec<wgpu::Buffer>,
    scratch_buffer: wgpu::Buffer,
    dispatch: LinearDispatch,

    scale_pipeline: Arc<wgpu::ComputePipeline>,
    scale_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    scale_param_buffer: Option<wgpu::Buffer>,
    scale_dispatch: LinearDispatch,

    log_n: u32,
    n: u32,
    width: u32,
    /// `true` when the final stage writes into scratch and we need to
    /// copy back into the user's buffer.
    result_in_scratch: bool,

    direction: NttDirection,
}

/// Parameters uploaded as one uniform buffer per NTT stage.
///
/// Must match `BatchedStockhamParams` in
/// `babybear_stockham_r2_batched.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BatchedStageParams {
    n: u32,
    s: u32,
    m: u32,
    twiddle_offset: u32,
    width: u32,
    groups_per_row: u32,
    _pad0: u32,
    _pad1: u32,
}

impl WgpuBatchedNttPlan {
    /// Build a plan for an `h × w` NTT where `h = 2^log_n`.
    ///
    /// Returns `ZkGpuError::BufferSize` if the batched scratch would
    /// exceed the device's max storage-buffer size. Callers should
    /// fall back to the column-loop path in that case.
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
        let per_buffer_bytes = (n as u64) * (width as u64) * 4;
        let twiddle_bytes = ((n - 1) as u64) * 4;

        // Per-buffer preflight: each individual buffer must fit the
        // device's single-storage-buffer cap. Matches the pattern in
        // `WgpuNttPlan::new_with_policy` (ntt/mod.rs). The largest
        // individual buffer here is user-data or scratch at h*w*4.
        let storage_limit = device.caps.max_storage_buffer_size();
        if per_buffer_bytes > storage_limit {
            return Err(ZkGpuError::BufferSize {
                requested: per_buffer_bytes,
                limit: storage_limit,
            });
        }

        // Aggregate preflight: total live footprint at execute time is
        //   user data + scratch + twiddles + twiddles_prime
        //   = 2*(h*w*4) + 2*((n-1)*4)
        // Check against max_buffer_size as a practical-device-capacity
        // proxy, same as `WgpuNttPlan::new_with_policy` does for the
        // single-column case.
        let total_live_bytes = 2 * per_buffer_bytes + 2 * twiddle_bytes;
        if total_live_bytes > device.caps.max_buffer_size {
            return Err(ZkGpuError::BufferSize {
                requested: total_live_bytes,
                limit: device.caps.max_buffer_size,
            });
        }

        #[cfg(not(target_arch = "wasm32"))]
        let validation_scope = device.push_validation_scope();
        let reg = &device.pipelines;

        // --- Pipelines ---
        let bind_group_layout = reg.get_or_create_bgl(
            &device.device,
            BATCHED_BGL_LABEL,
            &[
                bgl_storage_entry(0, true),  // src
                bgl_storage_entry(1, false), // dst
                bgl_storage_entry(2, true),  // twiddles
                bgl_uniform_entry(3),
                bgl_storage_entry(4, true), // twiddles_prime
            ],
        );
        let pipeline_layout = device
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Batched Stockham NTT pipeline layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });
        let module = reg.get_or_create_module(
            &device.device,
            BATCHED_R2_SOURCE,
            "Batched Stockham R2 shader",
        );
        let pipeline = reg.get_or_create_pipeline(
            &device.device,
            BATCHED_R2_SOURCE,
            "batched_stockham_r2_butterfly",
            BATCHED_BGL_LABEL,
            &pipeline_layout,
            &module,
            device.pipeline_cache.as_ref(),
        );

        // --- Scale pipeline (reuse single-poly scale.wgsl with
        //     n = h * w; every element gets the same 1/h scalar). ---
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

        // --- Twiddle table ---
        // Stage `k` (0..log_n) has m_k = n / 2^(k+1) twiddles.
        // Total twiddles = n - 1. We store them flat; `twiddle_offset`
        // picks the stage's slice.
        let omega = match direction {
            NttDirection::Forward => BabyBear::root_of_unity(log_n),
            NttDirection::Inverse => BabyBear::root_of_unity(log_n)
                .inv()
                .expect("root of unity is invertible"),
        };
        let mut twiddles: Vec<u32> = Vec::with_capacity((n - 1) as usize);
        let mut twiddles_prime: Vec<u32> = Vec::with_capacity((n - 1) as usize);
        let mut stage_offsets: Vec<u32> = Vec::with_capacity(log_n as usize);
        for h in 0..log_n {
            let s = 1u32 << h;
            let m = n >> (h + 1);
            let step = omega.pow(s as u64);
            stage_offsets.push(twiddles.len() as u32);
            let mut w_cur = BabyBear::ONE;
            for _ in 0..m {
                let repr = w_cur.to_repr();
                twiddles.push(repr);
                twiddles_prime.push(shoup_quotient(repr));
                w_cur = w_cur * step;
            }
        }

        let twiddle_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Batched NTT twiddles"),
                contents: bytemuck::cast_slice(&twiddles),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let twiddle_prime_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Batched NTT twiddles'"),
                contents: bytemuck::cast_slice(&twiddles_prime),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // --- Per-stage dispatch & param buffers ---
        // Total threads per stage = (n / 2) * w.
        let total_threads = (n / 2) * width;
        let dispatch = plan_linear_dispatch(
            total_threads,
            WORKGROUP_SIZE,
            device.caps.max_compute_workgroups_per_dimension,
        )?;

        let stage_param_buffers: Vec<wgpu::Buffer> = (0..log_n)
            .map(|h| {
                let s = 1u32 << h;
                let m = n >> (h + 1);
                let params = BatchedStageParams {
                    n,
                    s,
                    m,
                    twiddle_offset: stage_offsets[h as usize],
                    width,
                    groups_per_row: dispatch.groups_per_row,
                    _pad0: 0,
                    _pad1: 0,
                };
                device
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Batched stage params"),
                        contents: bytemuck::cast_slice(&[params]),
                        usage: wgpu::BufferUsages::UNIFORM,
                    })
            })
            .collect();

        // --- Scratch buffer (h × w × 4 bytes) ---
        let scratch_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Batched NTT scratch"),
            size: per_buffer_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // --- Scale (only if inverse) ---
        let scale_dispatch = plan_linear_dispatch(
            (n * width).div_ceil(4),
            WORKGROUP_SIZE,
            device.caps.max_compute_workgroups_per_dimension,
        )?;
        let scale_param_buffer = if direction == NttDirection::Inverse {
            let n_field = BabyBear::new(n);
            let n_inv = n_field.inv().expect("n must be invertible in BabyBear");
            // ScaleParams layout from babybear_scale.wgsl:
            //   { n: u32, scalar: u32, groups_per_row: u32, _pad1: u32 }
            let params = [n * width, n_inv.to_repr(), scale_dispatch.groups_per_row, 0u32];
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

        // After `log_n` ping-pong stages, the result lives in the
        // user's buffer if `log_n` is even, and in scratch if odd.
        // (Stage 0 reads buf → writes scratch; stage 1 reads scratch
        // → writes buf; etc.)
        let result_in_scratch = log_n % 2 == 1;

        Ok(Self {
            pipeline,
            bind_group_layout,
            twiddle_buffer,
            twiddle_prime_buffer,
            stage_param_buffers,
            scratch_buffer,
            dispatch,
            scale_pipeline,
            scale_bind_group_layout,
            scale_param_buffer,
            scale_dispatch,
            log_n,
            n,
            width,
            result_in_scratch,
            direction,
        })
    }

    /// Plan's NTT size `h = 2^log_n` (the per-column length).
    pub fn log_n(&self) -> u32 {
        self.log_n
    }

    /// Plan's batch width.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Execute one full batched NTT on the given buffer.
    ///
    /// `buf` must have exactly `h × w` elements (row-major, so
    /// `buf[r * w + c]` is matrix entry `(r, c)`).
    pub fn execute(
        &mut self,
        device: &WgpuDevice,
        buf: &mut WgpuBuffer<BabyBear>,
    ) -> Result<(), ZkGpuError> {
        let expected_len = (self.n as usize) * (self.width as usize);
        if buf.len() != expected_len {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "batched buffer length mismatch: plan expects {} elements (h={}, w={}), got {}",
                expected_len,
                self.n,
                self.width,
                buf.len()
            )));
        }

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Batched Stockham NTT encoder"),
            });

        for (stage_idx, param_buffer) in self.stage_param_buffers.iter().enumerate() {
            let (src_buf, dst_buf) = if stage_idx % 2 == 0 {
                (&buf.inner, &self.scratch_buffer)
            } else {
                (&self.scratch_buffer, &buf.inner)
            };

            let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Batched stage bind group"),
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
                        resource: self.twiddle_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: param_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.twiddle_prime_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Batched NTT stage"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(self.dispatch.x, self.dispatch.y, 1);
        }

        // If the final stage wrote into scratch, copy back to the
        // user's buffer before returning.
        if self.result_in_scratch {
            let size = (self.n as u64) * (self.width as u64) * 4;
            encoder.copy_buffer_to_buffer(&self.scratch_buffer, 0, &buf.inner, 0, size);
        }

        // Inverse scale (if applicable).
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

        // Suppress `direction unused` when not profiled — we stored
        // it for future timestamp reporting and potential debug output.
        let _ = self.direction;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    //! Local sanity check: batched plan output matches single-column
    //! plan output, column by column. Lives here so we exercise the
    //! fallback-free GPU path directly; the zkgpu-plonky3 adapter has
    //! the broader Plonky3-level differential tests.
    use super::*;
    use zkgpu_core::GpuDevice;

    fn random_values(len: usize, seed: u64) -> Vec<BabyBear> {
        // Xorshift — good enough for differential input generation.
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

    #[test]
    fn batched_matches_single_column_stockham() {
        use super::super::WgpuNttPlan;

        let device = match WgpuDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping batched test — no GPU device: {e}");
                return;
            }
        };

        let log_n = 8u32;
        let h = 1usize << log_n;
        let w = 4usize;

        let input = random_values(h * w, 0xCAFE_BABE_DEAD_BEEF);

        // Batched expected output.
        let mut batched_buf = device.upload(&input).expect("upload batched");
        let mut batched_plan =
            WgpuBatchedNttPlan::new(&device, log_n, w as u32, NttDirection::Forward)
                .expect("batched plan build");
        batched_plan
            .execute(&device, &mut batched_buf)
            .expect("batched execute");
        let batched_out = batched_buf.read_to_vec().expect("batched readback");

        // Single-column reference: run the production WgpuNttPlan on
        // each column individually.
        let mut per_column_out = vec![BabyBear::ZERO; h * w];
        for c in 0..w {
            let col: Vec<BabyBear> = (0..h).map(|r| input[r * w + c]).collect();
            let mut col_buf = device.upload(&col).expect("upload col");
            let mut plan = WgpuNttPlan::new(&device, log_n, NttDirection::Forward)
                .expect("single plan build");
            plan.execute_kernels(&device, &mut col_buf)
                .expect("single execute");
            let col_out = col_buf.read_to_vec().expect("single readback");
            for r in 0..h {
                per_column_out[r * w + c] = col_out[r];
            }
        }

        assert_eq!(
            batched_out.len(),
            per_column_out.len(),
            "batched vs per-column length mismatch"
        );
        for (i, (a, b)) in batched_out.iter().zip(per_column_out.iter()).enumerate() {
            assert_eq!(
                a.to_repr(),
                b.to_repr(),
                "element {i} differs: batched={:?} per_column={:?}",
                a,
                b
            );
        }
    }
}
