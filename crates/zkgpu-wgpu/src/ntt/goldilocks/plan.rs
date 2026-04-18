//! [`WgpuGoldilocksNttPlan`] — portable u32x2 Goldilocks NTT on the GPU.
//!
//! Phase B.2 deliverable. Implements a radix-2 Stockham DIF auto-sort
//! NTT over Goldilocks using the u32x2 WGSL arithmetic primitives
//! validated in Phase B.1. Global-only (no local-fused tail), one
//! butterfly per thread, log_n ping-pong stages between the caller's
//! buffer and an internal scratch buffer. Inverse direction applies
//! the `goldilocks_scale` kernel at the end to divide by n.
//!
//! Not wired into the main [`crate::WgpuNttPlan`] dispatch — BabyBear
//! and Goldilocks plans live side by side for now. Phase E of the
//! roadmap adds a `field` parameter to the harness and routes
//! Goldilocks suites here.

use std::sync::Arc;

use zkgpu_core::{GpuBuffer, GpuDevice, GpuField, NttDirection, ZkGpuError};

use crate::buffer::WgpuBuffer;
use crate::device::WgpuDevice;
use crate::ntt::goldilocks::resolve::{
    resolve_variant, GoldilocksKernelOverride, ResolvedGoldilocksKernel,
};

/// WGSL source: helpers + Stockham R2 body, concatenated at compile
/// time. See `goldilocks_arith_helpers.wgsl` for the prelude design.
const STOCKHAM_R2_WGSL: &str = concat!(
    include_str!("../../kernels/portable/goldilocks_arith_helpers.wgsl"),
    include_str!("../../kernels/portable/goldilocks_stockham_r2.wgsl"),
);

/// WGSL source: helpers + scale body.
const SCALE_WGSL: &str = concat!(
    include_str!("../../kernels/portable/goldilocks_arith_helpers.wgsl"),
    include_str!("../../kernels/portable/goldilocks_scale.wgsl"),
);

const STOCKHAM_BGL_LABEL: &str = "Goldilocks Stockham R2 BGL";
const SCALE_BGL_LABEL: &str = "Goldilocks scale BGL";
const STOCKHAM_ENTRY: &str = "gl_stockham_r2";
const SCALE_ENTRY: &str = "gl_scale";
const WORKGROUP_SIZE: u32 = 64;

// Stage-uniform layout must match `StockhamR2Params` in WGSL.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct StageUniform {
    n: u32,
    half: u32,
    half_total: u32,
    twiddle_offset: u32,
}

// Scale-uniform layout must match `ScaleParams` in WGSL.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ScaleUniform {
    n: u32,
    scalar_lo: u32,
    scalar_hi: u32,
    _pad: u32,
}

/// Portable Goldilocks NTT plan.
///
/// Construction precomputes twiddles, allocates scratch, and builds
/// all `log_n` stage-uniform buffers. `execute` reuses those across
/// dispatches; only the bind groups (which reference the caller's
/// input buffer) are created per-call.
pub struct WgpuGoldilocksNttPlan<F: GpuField<Repr = u64>> {
    direction: NttDirection,
    log_n: u32,
    n: u32,

    /// Precomputed twiddles for this direction, uploaded once.
    twiddles_buf: WgpuBuffer<F>,

    /// Ping-pong scratch buffer, length N.
    scratch_buf: WgpuBuffer<F>,

    /// One uniform per Stockham stage. Written during construction;
    /// read-only at execute-time.
    stage_uniforms: Vec<wgpu::Buffer>,

    /// Uniform for the post-inverse scale kernel. `None` for forward
    /// plans.
    scale_uniform: Option<wgpu::Buffer>,

    stockham_bgl: Arc<wgpu::BindGroupLayout>,
    stockham_pipeline: Arc<wgpu::ComputePipeline>,
    scale_bgl: Arc<wgpu::BindGroupLayout>,
    scale_pipeline: Arc<wgpu::ComputePipeline>,

    resolved: ResolvedGoldilocksKernel,

    _marker: std::marker::PhantomData<F>,
}

impl<F: GpuField<Repr = u64>> WgpuGoldilocksNttPlan<F> {
    /// Construct a plan for `2^log_n`-point NTT in `direction`.
    /// Currently hardcoded to the `Auto` resolver path, which in
    /// Phase A/B always resolves to `PortableU32x2`.
    pub fn new(
        device: &WgpuDevice,
        log_n: u32,
        direction: NttDirection,
    ) -> Result<Self, ZkGpuError> {
        let resolved = resolve_variant(GoldilocksKernelOverride::Auto, device.caps())
            .map_err(|e| ZkGpuError::BackendError(Box::new(e)))?;

        let n = 1u32 << log_n;

        // ---- Twiddles ------------------------------------------------
        let twiddles = precompute_stockham_r2_twiddles::<F>(log_n, direction);
        let twiddles_buf = device.upload::<F>(&twiddles)?;

        // ---- Scratch -------------------------------------------------
        let scratch_buf = device.alloc_zeros::<F>(n as usize)?;

        // ---- Stage uniforms -----------------------------------------
        let mut stage_uniforms = Vec::with_capacity(log_n as usize);
        let mut twiddle_offset: u32 = 0;
        let half_total = n / 2;
        for s in 0..log_n {
            let half = n >> (s + 1);
            let uniform = create_uniform(
                device.raw_device(),
                &StageUniform {
                    n,
                    half,
                    half_total,
                    twiddle_offset,
                },
                "goldilocks stockham r2 stage uniform",
            );
            stage_uniforms.push(uniform);
            twiddle_offset += half;
        }

        // ---- Scale uniform (inverse only) ---------------------------
        let scale_uniform = match direction {
            NttDirection::Forward => None,
            NttDirection::Inverse => {
                // n_inv = (n as field element)^{-1}, packed as (lo, hi).
                let n_field = F::from_u64(n as u64);
                let n_inv = n_field.inverse().ok_or_else(|| {
                    ZkGpuError::InvalidNttSize(format!(
                        "field is not invertible at n=2^{log_n}"
                    ))
                })?;
                let n_inv_u64: u64 = n_inv.to_repr();
                let scalar_lo = n_inv_u64 as u32;
                let scalar_hi = (n_inv_u64 >> 32) as u32;
                Some(create_uniform(
                    device.raw_device(),
                    &ScaleUniform {
                        n,
                        scalar_lo,
                        scalar_hi,
                        _pad: 0,
                    },
                    "goldilocks scale uniform",
                ))
            }
        };

        // ---- Pipelines + bind group layouts -------------------------
        let registry = device.pipeline_registry();
        let stockham_module = registry.get_or_create_module(
            device.raw_device(),
            STOCKHAM_R2_WGSL,
            "goldilocks_stockham_r2",
        );
        let stockham_bgl = registry.get_or_create_bgl(
            device.raw_device(),
            STOCKHAM_BGL_LABEL,
            &[
                bgl_storage_entry(0, true),
                bgl_storage_entry(1, false),
                bgl_storage_entry(2, true),
                bgl_uniform_entry(3),
            ],
        );
        let stockham_layout =
            device
                .raw_device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Goldilocks Stockham R2 pipeline layout"),
                    bind_group_layouts: &[Some(&stockham_bgl)],
                    immediate_size: 0,
                });
        let stockham_pipeline = registry.get_or_create_pipeline(
            device.raw_device(),
            STOCKHAM_R2_WGSL,
            STOCKHAM_ENTRY,
            STOCKHAM_BGL_LABEL,
            &stockham_layout,
            &stockham_module,
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
            twiddles_buf,
            scratch_buf,
            stage_uniforms,
            scale_uniform,
            stockham_bgl,
            stockham_pipeline,
            scale_bgl,
            scale_pipeline,
            resolved,
            _marker: std::marker::PhantomData,
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
        buf: &mut WgpuBuffer<F>,
    ) -> Result<(), ZkGpuError> {
        if buf.len() != self.n as usize {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "buffer length {} does not match plan n={}",
                buf.len(),
                self.n
            )));
        }

        let half_total = self.n / 2;
        let num_groups = half_total.div_ceil(WORKGROUP_SIZE);

        let mut encoder = device.raw_device().create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("goldilocks ntt encoder"),
            },
        );

        // --- Pre-build per-stage bind groups -------------------------
        // Each stage alternates which of (buf, scratch) plays
        // input/output. Bind groups reference those concrete buffers,
        // so we have to build them per-execute.
        let mut bind_groups: Vec<wgpu::BindGroup> = Vec::with_capacity(self.log_n as usize);
        for s in 0..self.log_n {
            let (in_buf, out_buf) = if s % 2 == 0 {
                (&buf.inner, &self.scratch_buf.inner)
            } else {
                (&self.scratch_buf.inner, &buf.inner)
            };
            let bg = device.raw_device().create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("goldilocks stockham r2 stage bg"),
                layout: &self.stockham_bgl,
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
            pass.set_pipeline(&self.stockham_pipeline);
            for bg in &bind_groups {
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(num_groups, 1, 1);
            }
        }

        // --- If log_n is odd, final result lives in scratch; copy ----
        // back into buf so the caller always sees output there.
        if self.log_n % 2 == 1 {
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
            let scale_groups = self.n.div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("goldilocks inverse scale"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scale_pipeline);
            pass.set_bind_group(0, &scale_bg, &[]);
            pass.dispatch_workgroups(scale_groups, 1, 1);
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
/// Total length = `N - 1`. Elements are written as `F` so the caller
/// can upload them directly via `WgpuDevice::upload`.
fn precompute_stockham_r2_twiddles<F: GpuField<Repr = u64>>(
    log_n: u32,
    direction: NttDirection,
) -> Vec<F> {
    let n = 1u32 << log_n;
    let omega = F::root_of_unity(log_n);
    let omega = match direction {
        NttDirection::Forward => omega,
        NttDirection::Inverse => omega
            .inverse()
            .expect("root of unity must be invertible"),
    };

    let mut tw = Vec::with_capacity((n - 1) as usize);
    for s in 0..log_n {
        let n_groups = 1u64 << s;
        let half = n >> (s + 1);
        let step = omega.pow(n_groups);
        let mut w = F::from_u64(1); // F::ONE but through the canonical constructor
        for _ in 0..half {
            tw.push(w);
            w = w.mul(step);
        }
    }
    tw
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
                WgpuGoldilocksNttPlan::<Goldilocks>::new(&device, log_n, NttDirection::Forward)
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
                WgpuGoldilocksNttPlan::<Goldilocks>::new(&device, log_n, NttDirection::Forward)
                    .unwrap();
            fwd.execute(&device, &mut buf).unwrap();

            let mut inv =
                WgpuGoldilocksNttPlan::<Goldilocks>::new(&device, log_n, NttDirection::Inverse)
                    .unwrap();
            inv.execute(&device, &mut buf).unwrap();

            let recovered = buf.read_to_vec_blocking().unwrap();
            assert_eq!(
                recovered, original,
                "fwd+inv roundtrip failed at log_n={log_n}"
            );
        }
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
            WgpuGoldilocksNttPlan::<Goldilocks>::new(&device, 4, NttDirection::Forward).unwrap();
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
