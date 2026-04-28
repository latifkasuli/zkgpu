//! [`WgpuBabyBearPoseidon2Plan`] — Phase F.1 GPU Poseidon2 plan.

use std::sync::Arc;

use zkgpu_babybear::BabyBear;
use zkgpu_core::{GpuBuffer, GpuDevice, GpuField, ZkGpuError};
use zkgpu_poseidon2::{Poseidon2Params, WIDTH};

use crate::async_util;
use crate::buffer::WgpuBuffer;
use crate::device::WgpuDevice;
use crate::dispatch::{plan_linear_dispatch, LinearDispatch};

const POSEIDON2_WGSL: &str =
    include_str!("../kernels/portable/babybear_poseidon2.wgsl");
const POSEIDON2_UNIFORM_WGSL: &str =
    include_str!("../kernels/portable/babybear_poseidon2_uniform.wgsl");

const POSEIDON2_BGL_LABEL: &str = "BabyBear Poseidon2 BGL";
const POSEIDON2_UNIFORM_BGL_LABEL: &str = "BabyBear Poseidon2 Uniform BGL";
const POSEIDON2_ENTRY: &str = "poseidon2_permute";
const WORKGROUP_SIZE: u32 = 64;

/// How Poseidon2 round constants reach the kernel.
///
/// Item #6 of `docs/research/zkgpu-wgpu-speed-opportunities.md` (Gate 2,
/// safer portable path). The two variants execute the same Poseidon2
/// algorithm and must produce bit-identical output — the only
/// difference is the binding type used for the round constants and
/// internal diagonal.
///
/// On Apple Silicon (Metal `constant` address space) and discrete GPUs
/// (Vulkan/SPIR-V `Uniform` storage class → cmem on NVIDIA) the
/// `Uniform` variant routes constants through dedicated constant-cache
/// hardware, which serves the broadcast access pattern (every thread
/// in a warp reads the same constant index at the same time) more
/// efficiently than general storage-buffer reads. Whether that
/// translates to a measurable wall-time win depends on the device
/// and the size of the rest of the working set; see the pilot bench
/// in `benches/poseidon2_constants_source.rs`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Poseidon2ConstantsSource {
    /// Three `var<storage, read>` arrays — `external_constants`,
    /// `internal_constants`, `internal_diagonal`. The shipping default;
    /// every existing caller uses this implicitly.
    #[default]
    Storage,
    /// One `var<uniform>` struct holding all three vectors packed as
    /// `array<vec4<u32>, _>` (the canonical WGSL way to avoid 4×
    /// padding waste in std140-like uniform layout). Pilot scope as
    /// of this commit: standalone `WgpuBabyBearPoseidon2Plan` only.
    Uniform,
}

/// Uniform layout mirrors the `Poseidon2Params` WGSL struct (4 × u32).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ParamsUniform {
    num_permutations: u32,
    rounds_f_half: u32,
    rounds_p: u32,
    row_stride: u32,
}

/// Packed-uniform layout for `Poseidon2ConstantsSource::Uniform`.
/// Mirrors the `Poseidon2W16ConstantsUniform` struct in
/// `babybear_poseidon2_uniform.wgsl` exactly:
///
/// * `external` — 32 × vec4<u32> = 128 u32 = 8 rounds × 16 width
/// * `internal` —  8 × vec4<u32> ≥ 32 partial rounds (BabyBear
///                 standard uses 13; trailing slots are zero-padded
///                 and never read)
/// * `diagonal` —  4 × vec4<u32> = 16 diagonal slots
///
/// Total: 44 × 16 B = 704 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ConstantsUniformW16 {
    external: [[u32; 4]; 32],
    internal: [[u32; 4]; 8],
    diagonal: [[u32; 4]; 4],
}

impl ConstantsUniformW16 {
    fn pack(
        external_flat: &[BabyBear],
        internal: &[BabyBear],
        diagonal: &[BabyBear],
    ) -> Result<Self, ZkGpuError> {
        if external_flat.len() > 32 * 4 {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "Poseidon2 external_constants exceeds uniform layout: \
                 {} > 128 (8 rounds × 16 width)",
                external_flat.len(),
            )));
        }
        if internal.len() > 8 * 4 {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "Poseidon2 internal_constants exceeds uniform layout: \
                 {} > 32 partial rounds",
                internal.len(),
            )));
        }
        if diagonal.len() > 4 * 4 {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "Poseidon2 internal_diagonal exceeds uniform layout: \
                 {} > 16 (W16 width)",
                diagonal.len(),
            )));
        }
        let mut u = Self {
            external: [[0u32; 4]; 32],
            internal: [[0u32; 4]; 8],
            diagonal: [[0u32; 4]; 4],
        };
        for (i, v) in external_flat.iter().enumerate() {
            u.external[i / 4][i % 4] = v.to_repr();
        }
        for (i, v) in internal.iter().enumerate() {
            u.internal[i / 4][i % 4] = v.to_repr();
        }
        for (i, v) in diagonal.iter().enumerate() {
            u.diagonal[i / 4][i % 4] = v.to_repr();
        }
        Ok(u)
    }
}

/// Internal binding state. Storage variant holds the three storage
/// buffers separately (today's path); Uniform variant holds a single
/// packed uniform buffer. The variant is fixed at plan construction
/// and drives both BGL choice and bind-group population in `encode`.
enum ConstantsBindings {
    Storage {
        external: WgpuBuffer<BabyBear>,
        internal: WgpuBuffer<BabyBear>,
        diagonal: WgpuBuffer<BabyBear>,
    },
    Uniform {
        packed: wgpu::Buffer,
    },
}

/// GPU Poseidon2 permutation plan for BabyBear at width-16.
///
/// Construction uploads the round-constant vectors + internal diagonal
/// exactly once and builds the compute pipeline. `execute` runs the
/// permutation in place on a caller-provided state buffer of length
/// `num_permutations * WIDTH`.
///
/// The plan is monomorphic over [`BabyBear`] because the WGSL kernel
/// hardcodes the BabyBear modulus (`P = 2^31 - 2^27 + 1`) and
/// reduction. The Goldilocks sibling in Phase F.2 will be its own
/// plan type, same pattern as the NTT split between [`crate::WgpuNttPlan`]
/// and [`crate::WgpuGoldilocksNttPlan`].
pub struct WgpuBabyBearPoseidon2Plan {
    /// Per-plan params for reference (rounds counts, alpha) — exposed
    /// via [`Self::params_info`] so callers can assert they're running
    /// the configuration they expect. The actual constant data lives
    /// in the GPU buffers below.
    rounds_f_half: u32,
    rounds_p: u32,

    /// Round constants binding state — either three storage buffers
    /// (today's default) or one packed uniform buffer (item #6 pilot).
    constants: ConstantsBindings,

    /// Uniform holding `(num_permutations, rounds_f_half, rounds_p,
    /// row_stride)`. Rewritten per `execute` call since
    /// `num_permutations` and `row_stride` depend on the batch size.
    params_uniform: wgpu::Buffer,

    bgl: Arc<wgpu::BindGroupLayout>,
    pipeline: Arc<wgpu::ComputePipeline>,
}

impl WgpuBabyBearPoseidon2Plan {
    /// S-box exponent hardcoded by [`BABYBEAR_POSEIDON2_WGSL`'s][W] `sbox7`
    /// function — four multiplies producing `x^7`. Any other `alpha` is
    /// rejected at construction because the kernel genuinely cannot
    /// execute it: swapping the S-box requires a shader edit, not a
    /// uniform update.
    ///
    /// [W]: ../../kernels/portable/babybear_poseidon2.wgsl
    pub const SUPPORTED_ALPHA: u64 = 7;

    /// Build a plan from a fully-parametrised [`Poseidon2Params`].
    ///
    /// Consumes the params: the round constants and internal diagonal
    /// are uploaded to GPU storage buffers once and live inside the
    /// plan. To change constants, build a new plan.
    ///
    /// Returns [`ZkGpuError::InvalidNttSize`] when the params carry a
    /// configuration the WGSL kernel cannot honour — today that's
    /// `alpha != 7` (the shader hardcodes the `x^7` S-box) or a
    /// constant-vector length that disagrees with `rounds_f_half` /
    /// `rounds_p`. Width is enforced at the type level via
    /// `Poseidon2Params<BabyBear, WIDTH>`.
    pub fn new(
        device: &WgpuDevice,
        params: Poseidon2Params<BabyBear, WIDTH>,
    ) -> Result<Self, ZkGpuError> {
        Self::new_with_constants_source(
            device,
            params,
            Poseidon2ConstantsSource::default(),
        )
    }

    /// Build a plan with an explicit choice of round-constant binding
    /// strategy. See [`Poseidon2ConstantsSource`] for the rationale and
    /// the pilot scope. Default behavior matches [`Self::new`].
    ///
    /// The constructed plan is functionally interchangeable with one
    /// built via [`Self::new`] — same input/output contract, same
    /// bit-identical output for a given input. Only the GPU-side
    /// memory access path for the round constants differs.
    pub fn new_with_constants_source(
        device: &WgpuDevice,
        params: Poseidon2Params<BabyBear, WIDTH>,
        constants_source: Poseidon2ConstantsSource,
    ) -> Result<Self, ZkGpuError> {
        // Phase F.1 post-review: reject unsupported alpha up front.
        // The shader's `sbox7` fn is four multiplies hardcoded to
        // `x^7`. A hypothetical `Poseidon2Params::<BabyBear, _>` with
        // `alpha = 3` is cryptographically meaningful (coprime to
        // p - 1 = 2^27 · ... for BabyBear), but this plan would
        // silently execute the wrong S-box. Surface the mismatch as a
        // structured error so callers either fix their params or wait
        // for a future plan variant that exposes alpha as a
        // shader-defs override.
        if params.alpha != Self::SUPPORTED_ALPHA {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "Poseidon2 alpha={} is not supported by \
                 WgpuBabyBearPoseidon2Plan (shader hardcodes x^{}); \
                 build a plan variant with a matching S-box first",
                params.alpha,
                Self::SUPPORTED_ALPHA,
            )));
        }

        // Sanity: the WGSL kernel assumes the canonical external
        // length (2 * rounds_f_half rows, each WIDTH wide) and a
        // rounds_p-length internal constant vector. Params::new()
        // already asserts these, but we re-check here so a plan built
        // from hand-assembled params doesn't silently desync.
        let expected_external_rows = 2 * params.rounds_f_half;
        if params.external_constants.len() != expected_external_rows {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "Poseidon2 external_constants: expected {} rows, got {}",
                expected_external_rows,
                params.external_constants.len(),
            )));
        }
        if params.internal_constants.len() != params.rounds_p {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "Poseidon2 internal_constants: expected {} entries, got {}",
                params.rounds_p,
                params.internal_constants.len(),
            )));
        }

        // --- Flatten constants (shape-validated above) ------------------
        let mut external_flat =
            Vec::with_capacity(expected_external_rows * WIDTH);
        for row in &params.external_constants {
            external_flat.extend_from_slice(row);
        }

        // --- Build constants binding (variant-specific) -----------------
        let constants = match constants_source {
            Poseidon2ConstantsSource::Storage => ConstantsBindings::Storage {
                external: device.upload::<BabyBear>(&external_flat)?,
                internal: device.upload::<BabyBear>(&params.internal_constants)?,
                diagonal: device.upload::<BabyBear>(&params.internal_diagonal)?,
            },
            Poseidon2ConstantsSource::Uniform => {
                let packed = ConstantsUniformW16::pack(
                    &external_flat,
                    &params.internal_constants,
                    &params.internal_diagonal,
                )?;
                ConstantsBindings::Uniform {
                    packed: create_uniform(
                        device.raw_device(),
                        &packed,
                        "babybear poseidon2 constants uniform",
                    ),
                }
            }
        };

        // --- Params uniform (zeroed at construction; filled per-execute) -
        let params_uniform = create_uniform(
            device.raw_device(),
            &ParamsUniform {
                num_permutations: 0,
                rounds_f_half: params.rounds_f_half as u32,
                rounds_p: params.rounds_p as u32,
                row_stride: 0,
            },
            "babybear poseidon2 params uniform",
        );

        // --- Pipeline + BGL (variant-specific) --------------------------
        //
        // Pipeline cache identity: the two variants use different WGSL
        // sources (different `&'static str` pointer → different
        // `SourceId::Static`) and different BGL labels, so they end up
        // as separate cache entries naturally — no manual cache-key
        // discrimination needed beyond what the foundation commit
        // already wired up.
        let registry = device.pipeline_registry();
        let (wgsl_src, bgl_label, module_label, bgl_entries) =
            match constants_source {
                Poseidon2ConstantsSource::Storage => (
                    POSEIDON2_WGSL,
                    POSEIDON2_BGL_LABEL,
                    "babybear_poseidon2",
                    vec![
                        // 0: state (read_write)
                        bgl_storage_entry(0, false),
                        // 1: external_constants (read)
                        bgl_storage_entry(1, true),
                        // 2: internal_constants (read)
                        bgl_storage_entry(2, true),
                        // 3: internal_diagonal (read)
                        bgl_storage_entry(3, true),
                        // 4: params (uniform)
                        bgl_uniform_entry(4),
                    ],
                ),
                Poseidon2ConstantsSource::Uniform => (
                    POSEIDON2_UNIFORM_WGSL,
                    POSEIDON2_UNIFORM_BGL_LABEL,
                    "babybear_poseidon2_uniform",
                    vec![
                        // 0: state (read_write)
                        bgl_storage_entry(0, false),
                        // 1: packed constants (uniform)
                        bgl_uniform_entry(1),
                        // 2: params (uniform)
                        bgl_uniform_entry(2),
                    ],
                ),
            };

        // Wrap pipeline construction in a validation scope so any
        // shader-compile / BGL / pipeline-layout error surfaces as a
        // structured `ZkGpuError::GpuValidation` rather than a silent
        // no-op pipeline. This caught a real failure class during
        // item #6 development: the Uniform-variant WGSL initially
        // named struct fields `external` and `internal`, both
        // reserved keywords in WGSL — Naga rejected the module, wgpu
        // returned a tombstone pipeline, and the only symptom was
        // GPU output being all zeros. Keep the scope.
        //
        // Native-only — `pop_validation_scope` is `#[cfg(not(wasm32))]`
        // because wasm needs the async variant. Mirrors the gating
        // pattern in `ntt::stockham::build`, `ntt::four_step::build`,
        // and `ntt::batched`. The browser/WebGPU portability claim
        // breaks if this scope leaks to a wasm build.
        #[cfg(not(target_arch = "wasm32"))]
        let scope = device.push_validation_scope();

        let module = registry.get_or_create_module(
            device.raw_device(),
            wgsl_src,
            module_label,
        );
        let bgl = registry.get_or_create_bgl(
            device.raw_device(),
            bgl_label,
            &bgl_entries,
        );
        let layout =
            device
                .raw_device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("BabyBear Poseidon2 pipeline layout"),
                    bind_group_layouts: &[Some(&bgl)],
                    immediate_size: 0,
                });
        let pipeline = registry.get_or_create_pipeline(
            device.raw_device(),
            wgsl_src,
            POSEIDON2_ENTRY,
            bgl_label,
            &layout,
            &module,
            None,
        );

        #[cfg(not(target_arch = "wasm32"))]
        device.pop_validation_scope(scope, "Poseidon2 plan build")?;

        Ok(Self {
            rounds_f_half: params.rounds_f_half as u32,
            rounds_p: params.rounds_p as u32,
            constants,
            params_uniform,
            bgl,
            pipeline,
        })
    }

    /// `(rounds_f_half, rounds_p)`. Lets callers assert the plan
    /// matches the CPU params they're validating against.
    pub fn params_info(&self) -> (u32, u32) {
        (self.rounds_f_half, self.rounds_p)
    }

    /// State width in field elements. Always [`WIDTH`] for this plan;
    /// surfaced as an accessor so harness code doesn't need to import
    /// the poseidon2 crate just to read the constant.
    pub fn width(&self) -> usize {
        WIDTH
    }

    /// Permute each `WIDTH`-element block of `buf` in place (blocking).
    ///
    /// `buf.len()` must equal `num_permutations * WIDTH` — the kernel
    /// treats the buffer as a flat concatenation of independent
    /// state vectors. For browser / async callers use
    /// [`execute_async`](Self::execute_async).
    pub fn execute(
        &mut self,
        device: &WgpuDevice,
        buf: &mut WgpuBuffer<BabyBear>,
    ) -> Result<(), ZkGpuError> {
        let Some(encoder) = self.encode(device, buf)? else {
            return Ok(()); // empty batch
        };
        device.raw_queue().submit(Some(encoder.finish()));
        device
            .raw_device()
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| ZkGpuError::BackendError(Box::new(e)))?;
        Ok(())
    }

    /// Async variant of [`execute`](Self::execute). Browser-safe.
    ///
    /// Phase F.3.d — mirrors `WgpuGoldilocksNttPlan::execute_async`
    /// (Phase E.2.a). Encode path is identical to the sync version;
    /// only the post-submit synchronisation differs. On native the
    /// async helper drives `device.poll(wait_indefinitely)` internally;
    /// on wasm it's a no-op (map_async downstream handles sync), so
    /// callers don't block the browser event loop.
    pub async fn execute_async(
        &mut self,
        device: &WgpuDevice,
        buf: &mut WgpuBuffer<BabyBear>,
    ) -> Result<(), ZkGpuError> {
        let Some(encoder) = self.encode(device, buf)? else {
            return Ok(()); // empty batch
        };
        device.raw_queue().submit(Some(encoder.finish()));
        async_util::wait_for_submission(device.raw_device(), device.raw_queue())
            .await
    }

    /// Build the compute-pass command buffer without submitting.
    ///
    /// Shared encode path for [`execute`](Self::execute) and
    /// [`execute_async`](Self::execute_async). Returns:
    /// - `Ok(Some(encoder))` when there's work to submit
    /// - `Ok(None)` when the batch is empty (zero permutations —
    ///   caller should return success without submitting)
    /// - `Err(_)` on mis-sized input or dispatch-plan failure
    ///
    /// The caller is responsible for `queue.submit` and the completion
    /// wait (sync `device.poll` vs async `async_util::wait_for_submission`).
    /// A future F.3.* sub-phase will extend this helper to accept an
    /// optional `ComputePassTimestampWrites` for profiled-execute, same
    /// shape as the Goldilocks NTT `encode` helper.
    fn encode(
        &mut self,
        device: &WgpuDevice,
        buf: &WgpuBuffer<BabyBear>,
    ) -> Result<Option<wgpu::CommandEncoder>, ZkGpuError> {
        if buf.len() % WIDTH != 0 {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "Poseidon2 state buffer length {} is not a multiple of \
                 WIDTH ({WIDTH})",
                buf.len(),
            )));
        }
        let num_permutations = (buf.len() / WIDTH) as u32;
        if num_permutations == 0 {
            return Ok(None);
        }

        // --- 2D-folded dispatch planning ------------------------------
        let max_wg = device.caps.max_compute_workgroups_per_dimension;
        let dispatch: LinearDispatch =
            plan_linear_dispatch(num_permutations, WORKGROUP_SIZE, max_wg)?;
        let row_stride = dispatch.groups_per_row * WORKGROUP_SIZE;

        // --- Refresh params uniform for this batch --------------------
        let params = ParamsUniform {
            num_permutations,
            rounds_f_half: self.rounds_f_half,
            rounds_p: self.rounds_p,
            row_stride,
        };
        device.raw_queue().write_buffer(
            &self.params_uniform,
            0,
            bytemuck::bytes_of(&params),
        );

        // --- Bind group (variant-specific layout) ---------------------
        let entries: Vec<wgpu::BindGroupEntry> = match &self.constants {
            ConstantsBindings::Storage {
                external,
                internal,
                diagonal,
            } => vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: external.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: internal.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: diagonal.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.params_uniform.as_entire_binding(),
                },
            ],
            ConstantsBindings::Uniform { packed } => vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: packed.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.params_uniform.as_entire_binding(),
                },
            ],
        };
        let bind_group = device.raw_device().create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: Some("babybear poseidon2 bg"),
                layout: &self.bgl,
                entries: &entries,
            },
        );

        // --- Encode + dispatch ----------------------------------------
        let mut encoder = device.raw_device().create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("babybear poseidon2 encoder"),
            },
        );
        {
            let mut pass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("babybear poseidon2 pass"),
                    timestamp_writes: None,
                });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch.x, dispatch.y, 1);
        }

        Ok(Some(encoder))
    }
}

// --- Bind-group helpers (local; mirror the NTT plan file) ---------------

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
    use zkgpu_core::GpuField;
    use zkgpu_poseidon2::Poseidon2;

    fn try_device() -> Option<WgpuDevice> {
        WgpuDevice::new().ok()
    }

    /// Single-permutation bit-parity: the GPU must produce exactly the
    /// same output as the CPU reference for the pinned regression
    /// anchor at `state[0] = 1`. This catches any drift in WGSL
    /// arithmetic, round ordering, or external-matrix implementation.
    #[test]
    fn poseidon2_gpu_matches_cpu_regression_state_0001() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let params = Poseidon2Params::<BabyBear, WIDTH>::babybear_default();
        let cpu = Poseidon2::new(params.clone());

        // CPU reference output
        let mut cpu_state = [BabyBear::new(0); WIDTH];
        cpu_state[0] = BabyBear::new(1);
        cpu.permute(&mut cpu_state);

        // GPU output for same input
        let mut gpu_state = vec![BabyBear::new(0); WIDTH];
        gpu_state[0] = BabyBear::new(1);
        let mut buf = device.upload::<BabyBear>(&gpu_state).unwrap();
        let mut plan = WgpuBabyBearPoseidon2Plan::new(&device, params).unwrap();
        plan.execute(&device, &mut buf).unwrap();
        let gpu_out = buf.read_to_vec_blocking().unwrap();

        assert_eq!(
            gpu_out.as_slice(),
            cpu_state.as_slice(),
            "GPU Poseidon2 must match CPU reference on regression anchor",
        );

        // Extra assertion: GPU result must also match the pinned
        // `babybear_regression_state_0001` expected array so changes
        // to the GPU kernel can't drift silently from the CPU pin.
        let expected: [u32; WIDTH] = [
            1646996371, 1788689999, 602438123, 1506086531,
            748277907, 1860416619, 1005521241, 522487477,
            1853726457, 740563310, 1495084457, 816004387,
            268492728, 1545584133, 820438449, 558558427,
        ];
        let got: [u32; WIDTH] = std::array::from_fn(|i| gpu_out[i].to_repr());
        assert_eq!(
            got, expected,
            "GPU regression_state_0001 drift — CPU and GPU disagree"
        );
    }

    /// Batch differential: feeding N distinct inputs through the GPU
    /// plan must yield the same outputs as running the CPU reference
    /// on each input independently. Proves the per-thread state
    /// isolation (no cross-permutation leakage, correct addressing).
    #[test]
    fn poseidon2_gpu_batch_matches_cpu_per_input() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let params = Poseidon2Params::<BabyBear, WIDTH>::babybear_default();
        let cpu = Poseidon2::new(params.clone());

        // 17 permutations — prime so we catch off-by-one in 2D-fold.
        let num = 17usize;
        let mut gpu_flat: Vec<BabyBear> = Vec::with_capacity(num * WIDTH);
        let mut cpu_blocks: Vec<[BabyBear; WIDTH]> = Vec::with_capacity(num);
        for p in 0..num {
            let mut s = [BabyBear::new(0); WIDTH];
            for (i, slot) in s.iter_mut().enumerate() {
                *slot = BabyBear::new(((p as u32) * 31 + i as u32 + 1) & 0x7FFF_FFFF);
            }
            gpu_flat.extend_from_slice(&s);
            cpu_blocks.push(s);
        }

        // CPU reference (per-block in-place permute)
        for block in cpu_blocks.iter_mut() {
            cpu.permute(block);
        }

        // GPU batch
        let mut buf = device.upload::<BabyBear>(&gpu_flat).unwrap();
        let mut plan = WgpuBabyBearPoseidon2Plan::new(&device, params).unwrap();
        plan.execute(&device, &mut buf).unwrap();
        let gpu_out = buf.read_to_vec_blocking().unwrap();

        for p in 0..num {
            let gpu_slice = &gpu_out[p * WIDTH..(p + 1) * WIDTH];
            assert_eq!(
                gpu_slice,
                cpu_blocks[p].as_slice(),
                "GPU Poseidon2 mismatch in batch slot {p}",
            );
        }
    }

    /// Empty-batch smoke: `execute` on a zero-length buffer must return
    /// `Ok(())` without dispatching anything. Harness code builds the
    /// batch dynamically and an empty batch is a legitimate no-op.
    #[test]
    fn poseidon2_gpu_empty_batch_is_noop() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let params = Poseidon2Params::<BabyBear, WIDTH>::babybear_default();
        let empty: Vec<BabyBear> = Vec::new();
        let mut buf = device.upload::<BabyBear>(&empty).unwrap();
        let mut plan = WgpuBabyBearPoseidon2Plan::new(&device, params).unwrap();
        plan.execute(&device, &mut buf).unwrap();
    }

    /// Mis-sized state buffer (length not a multiple of WIDTH) must
    /// produce a structured `InvalidNttSize` error, not a panic or a
    /// silent truncation.
    #[test]
    fn poseidon2_gpu_rejects_mis_sized_buffer() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let params = Poseidon2Params::<BabyBear, WIDTH>::babybear_default();
        // 25 elements — not divisible by 16.
        let data = vec![BabyBear::new(0); 25];
        let mut buf = device.upload::<BabyBear>(&data).unwrap();
        let mut plan = WgpuBabyBearPoseidon2Plan::new(&device, params).unwrap();
        let err = plan.execute(&device, &mut buf);
        assert!(
            matches!(err, Err(ZkGpuError::InvalidNttSize(_))),
            "expected InvalidNttSize, got {err:?}"
        );
    }

    /// Phase F.1 post-review: the plan must refuse to build when
    /// `params.alpha` doesn't match the WGSL kernel's hardcoded S-box.
    /// Without this gate, a caller could pass `alpha = 3` (valid
    /// Poseidon2 parameter, wrong for this shader) and get a plan
    /// that runs silently with the wrong permutation.
    ///
    /// No GPU needed — the check fires before device work.
    #[test]
    fn poseidon2_rejects_unsupported_alpha() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        // Build default BabyBear params (alpha=7), then override to
        // an unsupported alpha. Constants lengths stay consistent, so
        // only the alpha check should fire.
        let mut params = Poseidon2Params::<BabyBear, WIDTH>::babybear_default();
        params.alpha = 3;
        let err = WgpuBabyBearPoseidon2Plan::new(&device, params);
        // Can't `{:?}` the Ok branch since the plan doesn't derive Debug
        // (holds `Arc<wgpu::ComputePipeline>`). Check via match + string
        // inspection, which is sufficient.
        match err {
            Err(ZkGpuError::InvalidNttSize(msg)) => {
                assert!(
                    msg.contains("alpha=3"),
                    "error should name the rejected alpha: {msg}"
                );
                assert!(
                    msg.contains("x^7") || msg.contains("7"),
                    "error should name the supported alpha: {msg}"
                );
            }
            Err(e) => panic!(
                "expected InvalidNttSize for alpha=3, got different error: {e}"
            ),
            Ok(_) => panic!(
                "plan built successfully for alpha=3 — kernel would run \
                 silently wrong permutation"
            ),
        }
    }

    /// Positive gate: the default BabyBear params build cleanly — so
    /// the alpha check isn't false-positiving on its own sanctioned
    /// configuration. Explicit because the rejection test above would
    /// still pass if alpha=7 itself were being blocked.
    #[test]
    fn poseidon2_accepts_alpha_7() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let params = Poseidon2Params::<BabyBear, WIDTH>::babybear_default();
        assert_eq!(params.alpha, WgpuBabyBearPoseidon2Plan::SUPPORTED_ALPHA);
        let _plan = WgpuBabyBearPoseidon2Plan::new(&device, params)
            .expect("default BabyBear params (alpha=7) must build cleanly");
    }

    /// Phase F.3.d canary: `execute_async` must produce bit-identical
    /// output to the sync `execute`. Only the post-submit sync path
    /// differs, but the encode refactor (sync + async now share a
    /// single `encode` helper) is exactly the kind of change this
    /// test catches if a bind-group or dispatch parameter silently
    /// diverges between the two code paths.
    #[test]
    fn poseidon2_execute_async_matches_sync() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let params = Poseidon2Params::<BabyBear, WIDTH>::babybear_default();

        // 17 permutations (prime — exercises 2D-fold) with distinct
        // per-slot inputs.
        let num = 17usize;
        let mut input: Vec<BabyBear> = Vec::with_capacity(num * WIDTH);
        for p in 0..num {
            for i in 0..WIDTH {
                input.push(BabyBear::new(((p as u32) * 31 + i as u32 + 1) & 0x7FFF_FFFF));
            }
        }

        // Sync reference
        let mut sync_plan =
            WgpuBabyBearPoseidon2Plan::new(&device, params.clone()).unwrap();
        let mut sync_buf = device.upload::<BabyBear>(&input).unwrap();
        sync_plan.execute(&device, &mut sync_buf).unwrap();
        let sync_out = sync_buf.read_to_vec_blocking().unwrap();

        // Async under pollster — drives device.poll on native.
        let mut async_plan =
            WgpuBabyBearPoseidon2Plan::new(&device, params).unwrap();
        let mut async_buf = device.upload::<BabyBear>(&input).unwrap();
        pollster::block_on(async_plan.execute_async(&device, &mut async_buf))
            .unwrap();
        let async_out = async_buf.read_to_vec_blocking().unwrap();

        assert_eq!(
            sync_out, async_out,
            "execute_async must match sync execute bit-for-bit",
        );
    }

    /// Empty-batch path must short-circuit before submit on both
    /// sync and async paths. Regression guard for the `encode`
    /// returning `Ok(None)` contract.
    #[test]
    fn poseidon2_execute_async_empty_batch_is_noop() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let params = Poseidon2Params::<BabyBear, WIDTH>::babybear_default();
        let mut plan = WgpuBabyBearPoseidon2Plan::new(&device, params).unwrap();
        let empty: Vec<BabyBear> = Vec::new();
        let mut buf = device.upload::<BabyBear>(&empty).unwrap();
        pollster::block_on(plan.execute_async(&device, &mut buf)).unwrap();
    }

    /// Determinism: running the same input twice (including across
    /// separate `execute` calls on the same plan) must produce
    /// bit-identical output.
    #[test]
    fn poseidon2_gpu_is_deterministic() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let params = Poseidon2Params::<BabyBear, WIDTH>::babybear_default();
        let mut plan = WgpuBabyBearPoseidon2Plan::new(&device, params).unwrap();

        let mut input = [BabyBear::new(0); WIDTH];
        for (i, slot) in input.iter_mut().enumerate() {
            *slot = BabyBear::new((i as u32) + 1);
        }

        let mut buf1 = device.upload::<BabyBear>(&input).unwrap();
        plan.execute(&device, &mut buf1).unwrap();
        let out1 = buf1.read_to_vec_blocking().unwrap();

        let mut buf2 = device.upload::<BabyBear>(&input).unwrap();
        plan.execute(&device, &mut buf2).unwrap();
        let out2 = buf2.read_to_vec_blocking().unwrap();

        assert_eq!(out1, out2, "Poseidon2 must be deterministic across runs");
    }

    // -----------------------------------------------------------------------
    // Item #6 pilot: Poseidon2ConstantsSource::Uniform parity tests.
    //
    // These mirror the Storage-path tests above but exercise the new
    // uniform-bound kernel. The acceptance criterion is bit-identical
    // output to the Storage path on the pinned regression vector and on
    // a multi-input batch — anything else means the WGSL or pack/upload
    // logic drifted.
    // -----------------------------------------------------------------------

    /// Uniform path: regression-vector bit-parity. Locks the kernel
    /// against any drift in the WGSL `vec4`-lane unpack vs the Rust
    /// `pack` helper.
    #[test]
    fn poseidon2_uniform_gpu_matches_cpu_regression_state_0001() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let params = Poseidon2Params::<BabyBear, WIDTH>::babybear_default();
        let cpu = Poseidon2::new(params.clone());

        let mut cpu_state = [BabyBear::new(0); WIDTH];
        cpu_state[0] = BabyBear::new(1);
        cpu.permute(&mut cpu_state);

        let mut gpu_state = vec![BabyBear::new(0); WIDTH];
        gpu_state[0] = BabyBear::new(1);
        let mut buf = device.upload::<BabyBear>(&gpu_state).unwrap();
        let mut plan = WgpuBabyBearPoseidon2Plan::new_with_constants_source(
            &device,
            params,
            Poseidon2ConstantsSource::Uniform,
        )
        .unwrap();
        plan.execute(&device, &mut buf).unwrap();
        let gpu_out = buf.read_to_vec_blocking().unwrap();

        assert_eq!(
            gpu_out.as_slice(),
            cpu_state.as_slice(),
            "uniform-path Poseidon2 must match CPU reference",
        );

        // Same pinned-output array as the Storage-path test — locks
        // both variants against the same byte-exact reference, so any
        // drift between them is caught immediately.
        let expected: [u32; WIDTH] = [
            1646996371, 1788689999, 602438123, 1506086531,
            748277907, 1860416619, 1005521241, 522487477,
            1853726457, 740563310, 1495084457, 816004387,
            268492728, 1545584133, 820438449, 558558427,
        ];
        let got: [u32; WIDTH] = std::array::from_fn(|i| gpu_out[i].to_repr());
        assert_eq!(
            got, expected,
            "uniform-path regression_state_0001 drift",
        );
    }

    /// Uniform path: cross-variant bit-parity over a 17-permutation
    /// batch. Storage and Uniform paths must agree element-for-element
    /// — they execute the same algorithm, only the constant binding
    /// type differs.
    #[test]
    fn poseidon2_uniform_matches_storage_batch() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let params = Poseidon2Params::<BabyBear, WIDTH>::babybear_default();

        // Build a 17-permutation batch (prime; catches off-by-one
        // in 2D-fold dispatch + lane unpack).
        let num = 17usize;
        let mut input: Vec<BabyBear> = Vec::with_capacity(num * WIDTH);
        for p in 0..num {
            for i in 0..WIDTH {
                input.push(BabyBear::new(
                    ((p as u32) * 31 + i as u32 + 1) & 0x7FFF_FFFF,
                ));
            }
        }

        let mut storage_plan =
            WgpuBabyBearPoseidon2Plan::new(&device, params.clone()).unwrap();
        let mut uniform_plan =
            WgpuBabyBearPoseidon2Plan::new_with_constants_source(
                &device,
                params,
                Poseidon2ConstantsSource::Uniform,
            )
            .unwrap();

        let mut storage_buf = device.upload::<BabyBear>(&input).unwrap();
        storage_plan.execute(&device, &mut storage_buf).unwrap();
        let storage_out = storage_buf.read_to_vec_blocking().unwrap();

        let mut uniform_buf = device.upload::<BabyBear>(&input).unwrap();
        uniform_plan.execute(&device, &mut uniform_buf).unwrap();
        let uniform_out = uniform_buf.read_to_vec_blocking().unwrap();

        assert_eq!(
            uniform_out, storage_out,
            "Uniform variant must match Storage variant byte-for-byte"
        );
    }

    /// Uniform path: empty batch must remain a no-op (mirrors the
    /// Storage-path test — the encode helper is shared but the bind
    /// group construction differs).
    #[test]
    fn poseidon2_uniform_empty_batch_is_noop() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let params = Poseidon2Params::<BabyBear, WIDTH>::babybear_default();
        let mut plan = WgpuBabyBearPoseidon2Plan::new_with_constants_source(
            &device,
            params,
            Poseidon2ConstantsSource::Uniform,
        )
        .unwrap();

        let empty: Vec<BabyBear> = Vec::new();
        let mut buf = device.upload::<BabyBear>(&empty).unwrap();
        plan.execute(&device, &mut buf).expect("empty batch must be Ok");
    }
}
