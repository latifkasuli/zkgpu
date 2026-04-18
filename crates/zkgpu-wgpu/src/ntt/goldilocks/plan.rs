//! [`WgpuGoldilocksNttPlan`] — portable u32x2 Goldilocks NTT on the GPU.
//!
//! Phase B.3 deliverable. Implements a Stockham DIF auto-sort NTT over
//! Goldilocks using the u32x2 WGSL arithmetic primitives validated in
//! Phase B.1. The plan dispatches the radix-4 kernel when `log_n` is
//! even and falls back to radix-2 when odd. Global-only (no local-fused
//! tail), one butterfly per thread, ping-pong stages between the
//! caller's buffer and an internal scratch buffer. Inverse direction
//! applies the `goldilocks_scale` kernel at the end to divide by n.
//!
//! Dispatches are 2D-folded via [`crate::dispatch::plan_linear_dispatch`]
//! so large `log_n` honors WebGPU's baseline
//! `max_compute_workgroups_per_dimension = 65535`.
//!
//! Not wired into the main [`crate::WgpuNttPlan`] dispatch — BabyBear
//! and Goldilocks plans live side by side for now. Phase E of the
//! roadmap adds a `field` parameter to the harness and routes
//! Goldilocks suites here.

use std::sync::Arc;

use zkgpu_core::{GpuBuffer, GpuDevice, GpuField, NttDirection, ZkGpuError};
use zkgpu_goldilocks::Goldilocks;

use crate::buffer::WgpuBuffer;
use crate::device::WgpuDevice;
use crate::dispatch::{plan_linear_dispatch, LinearDispatch};
use crate::ntt::goldilocks::resolve::{
    resolve_variant, GoldilocksKernelOverride, ResolvedGoldilocksKernel,
};

/// Maximum `log_n` this portable plan accepts.
///
/// Capped at 31 because the stage-indexing path uses `u32` counters
/// (`n = 1u32 << log_n`, `half = n >> (s+1)`, etc.). Goldilocks'
/// 2-adicity is 32, so a log_n = 32 NTT is mathematically defined —
/// but in practice a 2^32 × 8-byte buffer is 32 GB, far beyond any
/// GPU's `max_buffer_size`. A future plan can widen the indexing path
/// to `u64` if / when that ceiling becomes interesting.
pub const MAX_GOLDILOCKS_LOG_N: u32 = 31;

/// WGSL source: helpers + Stockham R2 body, concatenated at compile
/// time. See `goldilocks_arith_helpers.wgsl` for the prelude design.
const STOCKHAM_R2_WGSL: &str = concat!(
    include_str!("../../kernels/portable/goldilocks_arith_helpers.wgsl"),
    include_str!("../../kernels/portable/goldilocks_stockham_r2.wgsl"),
);

/// WGSL source: helpers + Stockham R4 body.
const STOCKHAM_R4_WGSL: &str = concat!(
    include_str!("../../kernels/portable/goldilocks_arith_helpers.wgsl"),
    include_str!("../../kernels/portable/goldilocks_stockham_r4.wgsl"),
);

/// WGSL source: helpers + scale body.
const SCALE_WGSL: &str = concat!(
    include_str!("../../kernels/portable/goldilocks_arith_helpers.wgsl"),
    include_str!("../../kernels/portable/goldilocks_scale.wgsl"),
);

const STOCKHAM_R2_BGL_LABEL: &str = "Goldilocks Stockham R2 BGL";
const STOCKHAM_R4_BGL_LABEL: &str = "Goldilocks Stockham R4 BGL";
const SCALE_BGL_LABEL: &str = "Goldilocks scale BGL";
const STOCKHAM_R2_ENTRY: &str = "gl_stockham_r2";
const STOCKHAM_R4_ENTRY: &str = "gl_stockham_r4";
const SCALE_ENTRY: &str = "gl_scale";
const WORKGROUP_SIZE: u32 = 64;

// Stage-uniform layout must match `StockhamR2Params` in WGSL. 32 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct StageUniformR2 {
    n: u32,
    half: u32,
    half_total: u32,
    twiddle_offset: u32,
    row_stride: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// Stage-uniform layout must match `StockhamR4Params` in WGSL. 32 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct StageUniformR4 {
    n: u32,
    quarter: u32,
    total_bfly: u32,
    twiddle_offset: u32,
    i_n_lo: u32,
    i_n_hi: u32,
    row_stride: u32,
    _pad: u32,
}

// Scale-uniform layout must match `ScaleParams` in WGSL. 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ScaleUniform {
    n: u32,
    scalar_lo: u32,
    scalar_hi: u32,
    row_stride: u32,
}

/// Portable Goldilocks NTT plan.
///
/// **Concrete over `Goldilocks`.** The WGSL helpers hardcode the
/// Goldilocks modulus and reduction identity — any other
/// `GpuField<Repr = u64>` would compile against a generic shape of
/// this type and then silently execute Goldilocks arithmetic on the
/// GPU, producing wrong results. Keeping the type concrete rules that
/// out. When the shader layer becomes field-parametric, generalise at
/// that point.
///
/// Construction precomputes twiddles, allocates scratch, and builds
/// all `log_n` stage-uniform buffers. `execute` reuses those across
/// dispatches; only the bind groups (which reference the caller's
/// input buffer) are created per-call.
pub struct WgpuGoldilocksNttPlan {
    direction: NttDirection,
    log_n: u32,
    n: u32,

    /// `true` when this plan uses the Radix-4 Stockham kernel (log_n
    /// even); `false` when it falls back to Radix-2 (log_n odd).
    uses_r4: bool,

    /// Precomputed twiddles for this direction, uploaded once. Layout
    /// differs per radix:
    ///   - R2: `N - 1` entries, one twiddle per butterfly per stage.
    ///   - R4: `3 · (N/4) · log4_n` entries, three twiddles per
    ///     butterfly (tw1, tw2, tw3) laid out consecutively.
    twiddles_buf: WgpuBuffer<Goldilocks>,

    /// Ping-pong scratch buffer, length N.
    scratch_buf: WgpuBuffer<Goldilocks>,

    /// One uniform per Stockham stage. For R4, size is
    /// `size_of::<StageUniformR4>()`; for R2,
    /// `size_of::<StageUniformR2>()`.
    stage_uniforms: Vec<wgpu::Buffer>,

    /// Uniform for the post-inverse scale kernel. `None` for forward
    /// plans.
    scale_uniform: Option<wgpu::Buffer>,

    /// 2D-folded dispatch shape for the Stockham stages. Same shape is
    /// reused across every stage — all R2 (or R4) stages have the same
    /// butterfly count, so a single plan suffices.
    stockham_dispatch: LinearDispatch,

    /// 2D-folded dispatch shape for the inverse-scale kernel. `None`
    /// for forward plans. Distinct from `stockham_dispatch` because the
    /// scale kernel runs one thread per element (N total) rather than
    /// one per butterfly (N/2 or N/4).
    scale_dispatch: Option<LinearDispatch>,

    // R2 path (used when log_n is odd).
    stockham_r2_bgl: Arc<wgpu::BindGroupLayout>,
    stockham_r2_pipeline: Arc<wgpu::ComputePipeline>,

    // R4 path (used when log_n is even).
    stockham_r4_bgl: Arc<wgpu::BindGroupLayout>,
    stockham_r4_pipeline: Arc<wgpu::ComputePipeline>,

    scale_bgl: Arc<wgpu::BindGroupLayout>,
    scale_pipeline: Arc<wgpu::ComputePipeline>,

    resolved: ResolvedGoldilocksKernel,
}

impl WgpuGoldilocksNttPlan {
    /// Construct a plan for `2^log_n`-point NTT in `direction`.
    /// Currently hardcoded to the `Auto` resolver path, which in
    /// Phase A/B always resolves to `PortableU32x2`.
    ///
    /// Returns `ZkGpuError::InvalidNttSize` if `log_n == 0` or
    /// `log_n > MAX_GOLDILOCKS_LOG_N`, and `ZkGpuError::BufferSize` if
    /// the 2 × N × 8-byte working set (user buffer + internal scratch)
    /// exceeds the device's `max_buffer_size`.
    pub fn new(
        device: &WgpuDevice,
        log_n: u32,
        direction: NttDirection,
    ) -> Result<Self, ZkGpuError> {
        // ---- Size contract ------------------------------------------
        if log_n == 0 {
            return Err(ZkGpuError::InvalidNttSize(
                "log_n must be >= 1".to_string(),
            ));
        }
        if log_n > MAX_GOLDILOCKS_LOG_N {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "log_n={log_n} exceeds portable Goldilocks plan cap \
                 (MAX_GOLDILOCKS_LOG_N={MAX_GOLDILOCKS_LOG_N})"
            )));
        }
        // Now safe: log_n in [1, 31], so `1u32 << log_n` fits in u32.
        let n = 1u32 << log_n;

        // Buffer-budget preflight. Two checks, mirroring the BabyBear
        // `WgpuNttPlan` convention:
        //
        //   1. Per-buffer: each N-sized buffer is bound via
        //      `as_entire_binding()`, so its size must fit under
        //      `max_storage_buffer_size() = min(max_buffer_size,
        //      max_storage_buffer_binding_size)`. The binding cap is
        //      sometimes tighter than the raw buffer cap on
        //      mobile/integrated adapters, so we check both here by
        //      querying the effective minimum.
        //   2. Aggregate: the plan holds three N-sized buffers
        //      simultaneously (user buf + scratch + twiddles), so the
        //      total working set is 3·N·8. We cap that against
        //      `max_buffer_size` as a proxy for device-total memory;
        //      wgpu doesn't expose total GPU memory directly.
        let elem_bytes = 8u64; // Goldilocks = 8 bytes per element.
        let per_buffer = (n as u64) * elem_bytes;
        let storage_limit = device.caps.max_storage_buffer_size();
        if per_buffer > storage_limit {
            return Err(ZkGpuError::BufferSize {
                requested: per_buffer,
                limit: storage_limit,
            });
        }
        let aggregate = 3u64 * per_buffer;
        let buf_limit = device.caps.max_buffer_size;
        if aggregate > buf_limit {
            return Err(ZkGpuError::BufferSize {
                requested: aggregate,
                limit: buf_limit,
            });
        }

        let resolved = resolve_variant(GoldilocksKernelOverride::Auto, device.caps())
            .map_err(|e| ZkGpuError::BackendError(Box::new(e)))?;

        // R4 is dispatched only when log_n is even. Odd log_n keeps the
        // B.2 R2 path because a mixed R4+R2 dispatch adds bind-group and
        // parity complexity that isn't warranted until a concrete
        // odd-log_n workload justifies it.
        let uses_r4 = log_n % 2 == 0;

        // Plan the 2D-folded dispatch once — it's constant per plan
        // (all Stockham stages have the same butterfly count). The
        // shader reconstructs the flat butterfly index as
        // `gid.x + gid.y * row_stride`, avoiding the WebGPU baseline
        // limit `max_compute_workgroups_per_dimension >= 65535` that
        // a raw 1D dispatch exceeds at `log_n >= 23` (R2) or
        // `log_n >= 24` (R4).
        let max_wg_per_dim = device.caps.max_compute_workgroups_per_dimension;
        let stockham_bfly = if uses_r4 { n / 4 } else { n / 2 };
        let stockham_dispatch =
            plan_linear_dispatch(stockham_bfly, WORKGROUP_SIZE, max_wg_per_dim)?;
        let stockham_row_stride = stockham_dispatch.groups_per_row * WORKGROUP_SIZE;

        // ---- Twiddles + stage uniforms -------------------------------
        let (twiddles, stage_uniforms) = if uses_r4 {
            let tw = precompute_stockham_r4_twiddles(log_n, direction);
            let i_n_u64 = compute_i_n::<Goldilocks>(log_n, direction).to_repr();
            let i_n_lo = i_n_u64 as u32;
            let i_n_hi = (i_n_u64 >> 32) as u32;
            let total_bfly = n / 4;
            let mut uniforms = Vec::with_capacity((log_n / 2) as usize);
            let mut twiddle_offset: u32 = 0;
            for s in 0..(log_n / 2) {
                let quarter = n >> (2 * s + 2);
                let uniform = create_uniform(
                    device.raw_device(),
                    &StageUniformR4 {
                        n,
                        quarter,
                        total_bfly,
                        twiddle_offset,
                        i_n_lo,
                        i_n_hi,
                        row_stride: stockham_row_stride,
                        _pad: 0,
                    },
                    "goldilocks stockham r4 stage uniform",
                );
                uniforms.push(uniform);
                twiddle_offset += 3 * quarter;
            }
            (tw, uniforms)
        } else {
            let tw = precompute_stockham_r2_twiddles(log_n, direction);
            let half_total = n / 2;
            let mut uniforms = Vec::with_capacity(log_n as usize);
            let mut twiddle_offset: u32 = 0;
            for s in 0..log_n {
                let half = n >> (s + 1);
                let uniform = create_uniform(
                    device.raw_device(),
                    &StageUniformR2 {
                        n,
                        half,
                        half_total,
                        twiddle_offset,
                        row_stride: stockham_row_stride,
                        _pad0: 0,
                        _pad1: 0,
                        _pad2: 0,
                    },
                    "goldilocks stockham r2 stage uniform",
                );
                uniforms.push(uniform);
                twiddle_offset += half;
            }
            (tw, uniforms)
        };
        let twiddles_buf = device.upload::<Goldilocks>(&twiddles)?;

        // ---- Scratch -------------------------------------------------
        let scratch_buf = device.alloc_zeros::<Goldilocks>(n as usize)?;

        // ---- Scale uniform (inverse only) ---------------------------
        // The scale kernel dispatches one thread per *element* (N total),
        // not per butterfly, so it needs its own 2D-folded dispatch
        // plan. Without this, log_n >= 22 exceeds
        // `max_compute_workgroups_per_dimension >= 65535` on a raw 1D
        // grid (N/64 > 65535 at log_n = 22).
        let (scale_uniform, scale_dispatch) = match direction {
            NttDirection::Forward => (None, None),
            NttDirection::Inverse => {
                let scale_dispatch =
                    plan_linear_dispatch(n, WORKGROUP_SIZE, max_wg_per_dim)?;
                let scale_row_stride =
                    scale_dispatch.groups_per_row * WORKGROUP_SIZE;
                // n_inv = (n as field element)^{-1}, packed as (lo, hi).
                let n_field = <Goldilocks as GpuField>::from_u64(n as u64);
                let n_inv = n_field.inverse().ok_or_else(|| {
                    ZkGpuError::InvalidNttSize(format!(
                        "field is not invertible at n=2^{log_n}"
                    ))
                })?;
                let n_inv_u64: u64 = n_inv.to_repr();
                let scalar_lo = n_inv_u64 as u32;
                let scalar_hi = (n_inv_u64 >> 32) as u32;
                let uniform = create_uniform(
                    device.raw_device(),
                    &ScaleUniform {
                        n,
                        scalar_lo,
                        scalar_hi,
                        row_stride: scale_row_stride,
                    },
                    "goldilocks scale uniform",
                );
                (Some(uniform), Some(scale_dispatch))
            }
        };

        // ---- Pipelines + bind group layouts -------------------------
        //
        // We build both R2 and R4 pipelines up front even though this
        // plan instance only dispatches one of them (based on `uses_r4`).
        // The registry dedupes across plans, so a future log_n-odd plan
        // in the same process pays no extra compile cost.
        let registry = device.pipeline_registry();

        // R2 pipeline
        let r2_module = registry.get_or_create_module(
            device.raw_device(),
            STOCKHAM_R2_WGSL,
            "goldilocks_stockham_r2",
        );
        let stockham_r2_bgl = registry.get_or_create_bgl(
            device.raw_device(),
            STOCKHAM_R2_BGL_LABEL,
            &[
                bgl_storage_entry(0, true),
                bgl_storage_entry(1, false),
                bgl_storage_entry(2, true),
                bgl_uniform_entry(3),
            ],
        );
        let r2_layout =
            device
                .raw_device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Goldilocks Stockham R2 pipeline layout"),
                    bind_group_layouts: &[Some(&stockham_r2_bgl)],
                    immediate_size: 0,
                });
        let stockham_r2_pipeline = registry.get_or_create_pipeline(
            device.raw_device(),
            STOCKHAM_R2_WGSL,
            STOCKHAM_R2_ENTRY,
            STOCKHAM_R2_BGL_LABEL,
            &r2_layout,
            &r2_module,
            None,
        );

        // R4 pipeline
        let r4_module = registry.get_or_create_module(
            device.raw_device(),
            STOCKHAM_R4_WGSL,
            "goldilocks_stockham_r4",
        );
        let stockham_r4_bgl = registry.get_or_create_bgl(
            device.raw_device(),
            STOCKHAM_R4_BGL_LABEL,
            &[
                bgl_storage_entry(0, true),
                bgl_storage_entry(1, false),
                bgl_storage_entry(2, true),
                bgl_uniform_entry(3),
            ],
        );
        let r4_layout =
            device
                .raw_device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Goldilocks Stockham R4 pipeline layout"),
                    bind_group_layouts: &[Some(&stockham_r4_bgl)],
                    immediate_size: 0,
                });
        let stockham_r4_pipeline = registry.get_or_create_pipeline(
            device.raw_device(),
            STOCKHAM_R4_WGSL,
            STOCKHAM_R4_ENTRY,
            STOCKHAM_R4_BGL_LABEL,
            &r4_layout,
            &r4_module,
            None,
        );

        let scale_module = registry.get_or_create_module(
            device.raw_device(),
            SCALE_WGSL,
            "goldilocks_scale",
        );
        let scale_bgl = registry.get_or_create_bgl(
            device.raw_device(),
            SCALE_BGL_LABEL,
            &[
                bgl_storage_entry(0, false),
                bgl_uniform_entry(1),
            ],
        );
        let scale_layout =
            device
                .raw_device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Goldilocks scale pipeline layout"),
                    bind_group_layouts: &[Some(&scale_bgl)],
                    immediate_size: 0,
                });
        let scale_pipeline = registry.get_or_create_pipeline(
            device.raw_device(),
            SCALE_WGSL,
            SCALE_ENTRY,
            SCALE_BGL_LABEL,
            &scale_layout,
            &scale_module,
            None,
        );

        Ok(Self {
            direction,
            log_n,
            n,
            uses_r4,
            twiddles_buf,
            scratch_buf,
            stage_uniforms,
            scale_uniform,
            stockham_dispatch,
            scale_dispatch,
            stockham_r2_bgl,
            stockham_r2_pipeline,
            stockham_r4_bgl,
            stockham_r4_pipeline,
            scale_bgl,
            scale_pipeline,
            resolved,
        })
    }

    pub fn log_n(&self) -> u32 {
        self.log_n
    }
    pub fn direction(&self) -> NttDirection {
        self.direction
    }
    pub fn kernel_variant(&self) -> &'static str {
        self.resolved.variant.label()
    }
    pub fn kernel_reason(&self) -> &'static str {
        self.resolved.reason.label()
    }
    pub fn storage_abi(&self) -> &'static str {
        self.resolved.storage_abi.label()
    }

    /// Execute the NTT in place on `buf`.
    ///
    /// `buf` must have length `2^log_n`. Scratch is internal to the
    /// plan. Result lands back in `buf` (extra scratch→buf copy is
    /// inserted automatically for odd log_n).
    pub fn execute(
        &mut self,
        device: &WgpuDevice,
        buf: &mut WgpuBuffer<Goldilocks>,
    ) -> Result<(), ZkGpuError> {
        if buf.len() != self.n as usize {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "buffer length {} does not match plan n={}",
                buf.len(),
                self.n
            )));
        }

        // Dispatch geometry depends on radix:
        //   R2: N/2 butterflies per stage, log_n stages.
        //   R4: N/4 butterflies per stage, log_n/2 stages.
        //
        // Dispatch shape (`stockham_dispatch`) was computed once at plan
        // construction via `plan_linear_dispatch` and threaded into the
        // per-stage uniform's `row_stride` field so the shader can
        // reconstruct the flat butterfly index.
        let (num_stages, bgl, pipeline) = if self.uses_r4 {
            (
                self.log_n / 2,
                &self.stockham_r4_bgl,
                &self.stockham_r4_pipeline,
            )
        } else {
            (
                self.log_n,
                &self.stockham_r2_bgl,
                &self.stockham_r2_pipeline,
            )
        };
        let dispatch = self.stockham_dispatch;

        let mut encoder = device.raw_device().create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("goldilocks ntt encoder"),
            },
        );

        // --- Pre-build per-stage bind groups -------------------------
        // Each stage alternates which of (buf, scratch) plays
        // input/output. Bind groups reference those concrete buffers,
        // so we have to build them per-execute.
        let mut bind_groups: Vec<wgpu::BindGroup> = Vec::with_capacity(num_stages as usize);
        for s in 0..num_stages {
            let (in_buf, out_buf) = if s % 2 == 0 {
                (&buf.inner, &self.scratch_buf.inner)
            } else {
                (&self.scratch_buf.inner, &buf.inner)
            };
            let bg = device.raw_device().create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("goldilocks stockham stage bg"),
                layout: bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: in_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: out_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.twiddles_buf.inner.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.stage_uniforms[s as usize].as_entire_binding(),
                    },
                ],
            });
            bind_groups.push(bg);
        }

        // --- Stockham stages in one compute pass ---------------------
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("goldilocks ntt stages"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            for bg in &bind_groups {
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(dispatch.x, dispatch.y, 1);
            }
        }

        // --- If `num_stages` is odd, final result lives in scratch;
        // copy back into buf so the caller always sees output there.
        //
        // For R2: num_stages = log_n, so odd iff log_n is odd.
        // For R4: num_stages = log_n/2, so odd iff log_n ≡ 2 (mod 4).
        if num_stages % 2 == 1 {
            encoder.copy_buffer_to_buffer(
                &self.scratch_buf.inner,
                0,
                &buf.inner,
                0,
                buf.byte_size(),
            );
        }

        // --- Inverse normalisation -----------------------------------
        if self.direction == NttDirection::Inverse {
            let scale_uniform = self
                .scale_uniform
                .as_ref()
                .expect("inverse plan must have scale uniform");
            let scale_bg = device.raw_device().create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("goldilocks scale bg"),
                layout: &self.scale_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.inner.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: scale_uniform.as_entire_binding(),
                    },
                ],
            });
            let scale_dispatch = self
                .scale_dispatch
                .expect("inverse plan must have scale dispatch");
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("goldilocks inverse scale"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scale_pipeline);
            pass.set_bind_group(0, &scale_bg, &[]);
            pass.dispatch_workgroups(scale_dispatch.x, scale_dispatch.y, 1);
        }

        device.raw_queue().submit(Some(encoder.finish()));

        // Force GPU completion before the caller can do a map-based
        // readback on UMA targets — same sync point as arith_test.rs.
        device
            .raw_device()
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| ZkGpuError::BackendError(Box::new(e)))?;

        Ok(())
    }
}

// --- Host-side twiddle precomputation -----------------------------------

/// Precompute the flat twiddle array for a Stockham radix-2 NTT.
///
/// Layout (per stage `s`, `0 ≤ s < log_n`):
///   - `n_groups = 2^s`
///   - `half = N / 2^(s+1)`
///   - Entries at indices `offset..offset + half` are `ω^(j · n_groups)`
///     for `j = 0..half`, where ω is the primitive `N`-th root of
///     unity (or its inverse for the inverse NTT).
///   - `offset` for stage `s` = `sum over s' < s of half(s')`.
///
/// Total length = `N - 1`. Elements are uploaded directly via
/// `WgpuDevice::upload::<Goldilocks>`.
///
/// Assumes `log_n <= MAX_GOLDILOCKS_LOG_N` (caller's responsibility;
/// `WgpuGoldilocksNttPlan::new` enforces this).
fn precompute_stockham_r2_twiddles(
    log_n: u32,
    direction: NttDirection,
) -> Vec<Goldilocks> {
    let n = 1u32 << log_n;
    let omega = Goldilocks::root_of_unity(log_n);
    let omega = match direction {
        NttDirection::Forward => omega,
        NttDirection::Inverse => omega
            .inv()
            .expect("root of unity must be invertible"),
    };

    let mut tw = Vec::with_capacity((n - 1) as usize);
    for s in 0..log_n {
        let n_groups = 1u64 << s;
        let half = n >> (s + 1);
        let step = omega.pow(n_groups);
        let mut w = Goldilocks::new(1);
        for _ in 0..half {
            tw.push(w);
            w = w * step;
        }
    }
    tw
}

/// Precompute Stockham radix-4 twiddles.
///
/// Layout: per R4 stage `s` (`0 ≤ s < log_n/2`), three twiddles per
/// butterfly laid out consecutively:
///   `tw1(j) = ω^(j · n_groups)`
///   `tw2(j) = ω^(j · n_groups · 2)`
///   `tw3(j) = ω^(j · n_groups · 3)`
/// for `j = 0..quarter` where `n_groups = 4^s` and
/// `quarter = N / 4^(s+1)`.
///
/// Total length = `3 · sum over s of quarter(s) = 3 · (N/4 + N/16 + ...)`.
/// For `log_n = 2k`, the sum is `(N - 1) / 3`, so the total is exactly
/// `N - 1`. The buffer layout is strictly per-stage (tw offsets don't
/// overlap across stages).
///
/// Caller must only invoke this with even `log_n`.
fn precompute_stockham_r4_twiddles(
    log_n: u32,
    direction: NttDirection,
) -> Vec<Goldilocks> {
    debug_assert!(
        log_n % 2 == 0,
        "R4 twiddles only valid for even log_n; plan must fall back to R2"
    );
    let n = 1u32 << log_n;
    let omega = Goldilocks::root_of_unity(log_n);
    let omega = match direction {
        NttDirection::Forward => omega,
        NttDirection::Inverse => omega
            .inv()
            .expect("root of unity must be invertible"),
    };

    // Upper bound: 3 * sum of quarters.
    let mut tw = Vec::with_capacity(n as usize);
    for s in 0..(log_n / 2) {
        let n_groups = 1u64 << (2 * s);
        let quarter = n >> (2 * s + 2);
        let step = omega.pow(n_groups);
        // Emit (tw1, tw2, tw3) per j, with w = ω^(j · n_groups).
        let mut w = Goldilocks::new(1);
        for _ in 0..quarter {
            let w2 = w * w;
            let w3 = w2 * w;
            tw.push(w);
            tw.push(w2);
            tw.push(w3);
            w = w * step;
        }
    }
    tw
}

/// Compute `i_N = ω^(N/4)`, the primitive 4th root of unity in the
/// Goldilocks field. Used by the R4 kernel as its √-1 analogue.
///
/// For the inverse NTT, the direction-adjusted ω is `ω^(-1)`, so
/// `i_N_inverse = (ω^(-1))^(N/4) = (ω^(N/4))^(-1) = -i_N_forward`.
/// Both are valid primitive 4th roots; the R4 kernel uses whichever
/// matches the direction-adjusted twiddles.
fn compute_i_n<F: GpuField>(log_n: u32, direction: NttDirection) -> F {
    let omega = F::root_of_unity(log_n);
    let omega = match direction {
        NttDirection::Forward => omega,
        NttDirection::Inverse => omega
            .inverse()
            .expect("root of unity must be invertible"),
    };
    // ω^(N/4) = ω^(2^(log_n - 2))
    omega.pow(1u64 << (log_n - 2))
}

// --- Binding-entry helpers (local to this module to keep the plan
// self-contained) --------------------------------------------------------

fn bgl_storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn create_uniform<U: bytemuck::Pod>(
    device: &wgpu::Device,
    data: &U,
    label: &'static str,
) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::bytes_of(data),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkgpu_goldilocks::Goldilocks;
    use zkgpu_ntt::ntt_cpu_reference;

    fn try_device() -> Option<WgpuDevice> {
        WgpuDevice::new().ok()
    }

    /// Deterministic input vector: `[0, 1, ..., N-1]`. Used for the
    /// forward / inverse / roundtrip canaries so a failure pinpoints a
    /// specific (stage, butterfly) divergence.
    fn sequential_input(n: u32) -> Vec<Goldilocks> {
        (0..n as u64).map(Goldilocks::new).collect()
    }

    /// GPU forward NTT must bit-match `ntt_cpu_reference::<Goldilocks>`
    /// for the same input. Any mismatch indicates either a broken
    /// arithmetic primitive (should have fired in Phase B.1) or a
    /// Stockham indexing / twiddle bug.
    #[test]
    fn goldilocks_gpu_forward_matches_cpu_reference() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        for log_n in [3u32, 4, 5, 6, 10] {
            let n = 1u32 << log_n;
            let mut gpu_data = sequential_input(n);
            let mut cpu_data = gpu_data.clone();

            let mut plan =
                WgpuGoldilocksNttPlan::new(&device, log_n, NttDirection::Forward)
                    .expect("plan should build");

            let mut gpu_buf = device.upload::<Goldilocks>(&gpu_data).unwrap();
            plan.execute(&device, &mut gpu_buf).unwrap();
            gpu_data = gpu_buf.read_to_vec_blocking().unwrap();

            ntt_cpu_reference::<Goldilocks>(&mut cpu_data, NttDirection::Forward);

            assert_eq!(
                gpu_data, cpu_data,
                "forward NTT mismatch at log_n={log_n}"
            );
        }
    }

    /// Inverse NTT of a forward NTT output must recover the original.
    /// Exercises the scale kernel + inverse twiddles together.
    #[test]
    fn goldilocks_gpu_forward_inverse_roundtrip() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        for log_n in [3u32, 4, 5, 6, 10] {
            let n = 1u32 << log_n;
            let original = sequential_input(n);
            let mut buf = device.upload::<Goldilocks>(&original).unwrap();

            let mut fwd =
                WgpuGoldilocksNttPlan::new(&device, log_n, NttDirection::Forward)
                    .unwrap();
            fwd.execute(&device, &mut buf).unwrap();

            let mut inv =
                WgpuGoldilocksNttPlan::new(&device, log_n, NttDirection::Inverse)
                    .unwrap();
            inv.execute(&device, &mut buf).unwrap();

            let recovered = buf.read_to_vec_blocking().unwrap();
            assert_eq!(
                recovered, original,
                "fwd+inv roundtrip failed at log_n={log_n}"
            );
        }
    }

    /// Invalid log_n (0 or > 31) must produce a structured
    /// `InvalidNttSize` error, not a panic, overflow, or assert.
    #[test]
    fn plan_rejects_out_of_range_log_n() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        // log_n = 0: explicit error
        let err = WgpuGoldilocksNttPlan::new(&device, 0, NttDirection::Forward);
        assert!(matches!(err, Err(ZkGpuError::InvalidNttSize(_))));

        // log_n = 32: above MAX_GOLDILOCKS_LOG_N
        let err = WgpuGoldilocksNttPlan::new(&device, 32, NttDirection::Forward);
        assert!(matches!(err, Err(ZkGpuError::InvalidNttSize(_))));

        // log_n = MAX_GOLDILOCKS_LOG_N itself may hit the buffer-size
        // preflight on most devices (2 × 2^31 × 8 bytes = 32 GB > any
        // GPU's max_buffer_size). That's acceptable — it must still be
        // a structured error, not a panic.
        let err = WgpuGoldilocksNttPlan::new(&device, MAX_GOLDILOCKS_LOG_N, NttDirection::Forward);
        assert!(matches!(
            err,
            Err(ZkGpuError::InvalidNttSize(_)) | Err(ZkGpuError::BufferSize { .. })
        ));
    }

    /// B.3 acceptance gate: the R4 path at log_n = 18 must bit-match
    /// the CPU reference. 262 144 elements; ~50–100 ms CPU reference,
    /// a few ms GPU. Runs on both Metal and Vulkan.
    #[test]
    fn goldilocks_gpu_forward_matches_cpu_log18() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let log_n = 18u32;
        let n = 1u32 << log_n;
        // Deterministic SplitMix64 inputs so any mismatch is
        // bit-reproducible from the seed.
        let mut rng_state: u64 = 0xCAFE_BABE_DEAD_BEEF;
        let mut next = || {
            rng_state = rng_state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = rng_state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        let input: Vec<Goldilocks> = (0..n).map(|_| Goldilocks::new(next())).collect();

        let mut cpu = input.clone();
        ntt_cpu_reference::<Goldilocks>(&mut cpu, NttDirection::Forward);

        let mut plan =
            WgpuGoldilocksNttPlan::new(&device, log_n, NttDirection::Forward).unwrap();
        let mut gpu_buf = device.upload::<Goldilocks>(&input).unwrap();
        plan.execute(&device, &mut gpu_buf).unwrap();
        let gpu = gpu_buf.read_to_vec_blocking().unwrap();

        assert_eq!(gpu.len(), cpu.len());
        if gpu != cpu {
            // Report the first mismatch for debuggability.
            let first_mismatch = gpu
                .iter()
                .zip(cpu.iter())
                .position(|(a, b)| a != b)
                .unwrap();
            panic!(
                "log18 forward mismatch at index {first_mismatch}: \
                 gpu={} cpu={}",
                gpu[first_mismatch].0, cpu[first_mismatch].0
            );
        }
    }

    /// B.3 acceptance gate: GPU determinism at log_n = 20. Two forward
    /// runs of the same input must produce bit-identical output. Catches
    /// race conditions in the ping-pong dispatch and any nondeterministic
    /// scheduling in the R4 kernel.
    ///
    /// 1 M elements. No CPU reference (too slow for tight inner loops) —
    /// just checks GPU vs. GPU.
    #[test]
    fn goldilocks_gpu_determinism_log20() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let log_n = 20u32;
        let n = 1u32 << log_n;
        // Skip if the plan would fail the buffer-size preflight on the
        // current adapter — log 20 needs ~16 MB × 2 = 32 MB working set
        // which is fine on any real GPU, but a software adapter could
        // reject it.
        let input: Vec<Goldilocks> = (0..n as u64).map(Goldilocks::new).collect();

        let mut plan =
            match WgpuGoldilocksNttPlan::new(&device, log_n, NttDirection::Forward) {
                Ok(p) => p,
                Err(ZkGpuError::BufferSize { .. }) => {
                    eprintln!("skipping determinism_log20: adapter buffer cap too small");
                    return;
                }
                Err(e) => panic!("plan construction failed: {e}"),
            };

        let mut buf1 = device.upload::<Goldilocks>(&input).unwrap();
        plan.execute(&device, &mut buf1).unwrap();
        let out1 = buf1.read_to_vec_blocking().unwrap();

        let mut buf2 = device.upload::<Goldilocks>(&input).unwrap();
        plan.execute(&device, &mut buf2).unwrap();
        let out2 = buf2.read_to_vec_blocking().unwrap();

        assert_eq!(out1, out2, "log20 forward NTT not deterministic across runs");
    }

    /// Plan reports the resolved variant, reason, and storage ABI —
    /// Phase E's harness consumer depends on these accessors.
    #[test]
    fn plan_reports_portable_variant() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let plan =
            WgpuGoldilocksNttPlan::new(&device, 4, NttDirection::Forward).unwrap();
        assert_eq!(plan.kernel_variant(), "PortableU32x2");
        assert_eq!(plan.storage_abi(), "Limb32x2Le");
        // Reason depends on whether we're on browser (BrowserWgslNoInt64)
        // or native (AutoPortableDefault). Either is legitimate.
        let reason = plan.kernel_reason();
        assert!(
            reason == "AutoPortableDefault" || reason == "BrowserWgslNoInt64",
            "unexpected reason: {reason}"
        );
    }
}
