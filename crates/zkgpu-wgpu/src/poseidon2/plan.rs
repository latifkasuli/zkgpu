//! [`WgpuBabyBearPoseidon2Plan`] — Phase F.1 GPU Poseidon2 plan.

use std::sync::Arc;

use zkgpu_babybear::BabyBear;
use zkgpu_core::{GpuBuffer, GpuDevice, ZkGpuError};
use zkgpu_poseidon2::{Poseidon2Params, WIDTH};

use crate::buffer::WgpuBuffer;
use crate::device::WgpuDevice;
use crate::dispatch::{plan_linear_dispatch, LinearDispatch};

const POSEIDON2_WGSL: &str =
    include_str!("../kernels/portable/babybear_poseidon2.wgsl");

const POSEIDON2_BGL_LABEL: &str = "BabyBear Poseidon2 BGL";
const POSEIDON2_ENTRY: &str = "poseidon2_permute";
const WORKGROUP_SIZE: u32 = 64;

/// Uniform layout mirrors the `Poseidon2Params` WGSL struct (4 × u32).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ParamsUniform {
    num_permutations: u32,
    rounds_f_half: u32,
    rounds_p: u32,
    row_stride: u32,
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

    /// Flattened external-round constants:
    /// `external_constants[round * WIDTH + slot]`. Length
    /// `2 * rounds_f_half * WIDTH`.
    external_constants_buf: WgpuBuffer<BabyBear>,

    /// Internal-round constants, one per internal round. Length
    /// `rounds_p`.
    internal_constants_buf: WgpuBuffer<BabyBear>,

    /// Internal-layer diagonal `D = diag(d_0..d_{WIDTH-1})`. Length
    /// [`WIDTH`].
    internal_diagonal_buf: WgpuBuffer<BabyBear>,

    /// Uniform holding `(num_permutations, rounds_f_half, rounds_p,
    /// row_stride)`. Rewritten per `execute` call since
    /// `num_permutations` and `row_stride` depend on the batch size.
    params_uniform: wgpu::Buffer,

    bgl: Arc<wgpu::BindGroupLayout>,
    pipeline: Arc<wgpu::ComputePipeline>,
}

impl WgpuBabyBearPoseidon2Plan {
    /// Build a plan from a fully-parametrised [`Poseidon2Params`].
    ///
    /// Consumes the params: the round constants and internal diagonal
    /// are uploaded to GPU storage buffers once and live inside the
    /// plan. To change constants, build a new plan.
    pub fn new(
        device: &WgpuDevice,
        params: Poseidon2Params<BabyBear, WIDTH>,
    ) -> Result<Self, ZkGpuError> {
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

        // --- Flatten + upload constants ---------------------------------
        let mut external_flat =
            Vec::with_capacity(expected_external_rows * WIDTH);
        for row in &params.external_constants {
            external_flat.extend_from_slice(row);
        }
        let external_constants_buf =
            device.upload::<BabyBear>(&external_flat)?;
        let internal_constants_buf =
            device.upload::<BabyBear>(&params.internal_constants)?;
        let internal_diagonal_buf =
            device.upload::<BabyBear>(&params.internal_diagonal)?;

        // --- Uniform (zeroed at construction; filled per-execute) -------
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

        // --- Pipeline + BGL ---------------------------------------------
        let registry = device.pipeline_registry();
        let module = registry.get_or_create_module(
            device.raw_device(),
            POSEIDON2_WGSL,
            "babybear_poseidon2",
        );
        let bgl = registry.get_or_create_bgl(
            device.raw_device(),
            POSEIDON2_BGL_LABEL,
            &[
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
            POSEIDON2_WGSL,
            POSEIDON2_ENTRY,
            POSEIDON2_BGL_LABEL,
            &layout,
            &module,
            None,
        );

        Ok(Self {
            rounds_f_half: params.rounds_f_half as u32,
            rounds_p: params.rounds_p as u32,
            external_constants_buf,
            internal_constants_buf,
            internal_diagonal_buf,
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

    /// Permute each `WIDTH`-element block of `buf` in place.
    ///
    /// `buf.len()` must equal `num_permutations * WIDTH` — the kernel
    /// treats the buffer as a flat concatenation of independent
    /// state vectors.
    pub fn execute(
        &mut self,
        device: &WgpuDevice,
        buf: &mut WgpuBuffer<BabyBear>,
    ) -> Result<(), ZkGpuError> {
        if buf.len() % WIDTH != 0 {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "Poseidon2 state buffer length {} is not a multiple of \
                 WIDTH ({WIDTH})",
                buf.len(),
            )));
        }
        let num_permutations = (buf.len() / WIDTH) as u32;
        if num_permutations == 0 {
            // Empty batch — nothing to dispatch. Caller still gets a
            // successful result so the harness can treat batch=0 as
            // a no-op instead of a configuration error.
            return Ok(());
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

        // --- Bind group -----------------------------------------------
        let bind_group = device.raw_device().create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: Some("babybear poseidon2 bg"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.inner.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self
                            .external_constants_buf
                            .inner
                            .as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self
                            .internal_constants_buf
                            .inner
                            .as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self
                            .internal_diagonal_buf
                            .inner
                            .as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.params_uniform.as_entire_binding(),
                    },
                ],
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

        device.raw_queue().submit(Some(encoder.finish()));
        device
            .raw_device()
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| ZkGpuError::BackendError(Box::new(e)))?;
        Ok(())
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
}
