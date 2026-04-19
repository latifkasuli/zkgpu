//! [`WgpuGoldilocksPoseidon2Plan`] — Phase F.2 Goldilocks GPU plan.
//!
//! Portable u32x2 Goldilocks twin of [`super::WgpuBabyBearPoseidon2Plan`].
//! Shares the single-pass structure, the 2D-folded dispatch, and the
//! once-at-construction constant upload; only the element type, the
//! arithmetic helpers (`gl_add` / `gl_sub` / `gl_mul` prepended from
//! `goldilocks_arith_helpers.wgsl`), and the per-element byte width
//! (8 bytes) differ.
//!
//! Concrete over [`Goldilocks`] for the same reason the
//! [`crate::WgpuGoldilocksNttPlan`] is concrete: the WGSL helpers
//! hardcode the Goldilocks modulus + reduction, so a generic
//! `F: GpuField<Repr = u64>` plan would compile but silently run
//! Goldilocks arithmetic on any other 64-bit field.

use std::sync::Arc;

use zkgpu_core::{GpuBuffer, GpuDevice, ZkGpuError};
use zkgpu_goldilocks::Goldilocks;
use zkgpu_poseidon2::{Poseidon2Params, WIDTH};

use crate::buffer::WgpuBuffer;
use crate::device::WgpuDevice;
use crate::dispatch::{plan_linear_dispatch, LinearDispatch};

/// Compile-time WGSL concatenation: helpers prelude + Poseidon2 body.
/// Same pattern as the Goldilocks NTT plans.
const GOLDILOCKS_POSEIDON2_WGSL: &str = concat!(
    include_str!("../kernels/portable/goldilocks_arith_helpers.wgsl"),
    include_str!("../kernels/portable/goldilocks_poseidon2.wgsl"),
);

const POSEIDON2_BGL_LABEL: &str = "Goldilocks Poseidon2 BGL";
const POSEIDON2_ENTRY: &str = "gl_poseidon2_permute";
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

/// GPU Poseidon2 permutation plan for Goldilocks at width-16.
///
/// See [`super::WgpuBabyBearPoseidon2Plan`] for the algorithm and
/// batch-model docs. This plan differs only in:
/// - Element type: `Goldilocks` (8 bytes / `vec2<u32>` limb pair).
/// - Shader: `goldilocks_poseidon2.wgsl` prepended with the Goldilocks
///   arith helpers.
/// - S-box exponent: `x^7` (same as BabyBear — coprime to p-1 for
///   Goldilocks because none of 7's prime factors divide p-1).
///
/// Rejects `params.alpha != 7` at construction for the same reason as
/// the BabyBear plan — the shader's `gl_sbox7` fn is four multiplies
/// hardcoded to x^7.
pub struct WgpuGoldilocksPoseidon2Plan {
    rounds_f_half: u32,
    rounds_p: u32,

    external_constants_buf: WgpuBuffer<Goldilocks>,
    internal_constants_buf: WgpuBuffer<Goldilocks>,
    internal_diagonal_buf: WgpuBuffer<Goldilocks>,
    params_uniform: wgpu::Buffer,

    bgl: Arc<wgpu::BindGroupLayout>,
    pipeline: Arc<wgpu::ComputePipeline>,
}

impl WgpuGoldilocksPoseidon2Plan {
    /// S-box exponent hardcoded by the WGSL kernel. See the BabyBear
    /// plan's constant of the same name for the rationale — any other
    /// alpha is rejected at construction because the shader's
    /// `gl_sbox7` fn is structurally `x^7`.
    pub const SUPPORTED_ALPHA: u64 = 7;

    /// Build a plan from a fully-parametrised [`Poseidon2Params`].
    ///
    /// Returns [`ZkGpuError::InvalidNttSize`] for unsupported
    /// configurations (`alpha != 7`, mis-sized constant vectors).
    /// Width is enforced at the type level via
    /// `Poseidon2Params<Goldilocks, WIDTH>`.
    pub fn new(
        device: &WgpuDevice,
        params: Poseidon2Params<Goldilocks, WIDTH>,
    ) -> Result<Self, ZkGpuError> {
        if params.alpha != Self::SUPPORTED_ALPHA {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "Poseidon2 alpha={} is not supported by \
                 WgpuGoldilocksPoseidon2Plan (shader hardcodes x^{})",
                params.alpha,
                Self::SUPPORTED_ALPHA,
            )));
        }

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

        // --- Flatten + upload constants -------------------------------
        let mut external_flat =
            Vec::with_capacity(expected_external_rows * WIDTH);
        for row in &params.external_constants {
            external_flat.extend_from_slice(row);
        }
        let external_constants_buf =
            device.upload::<Goldilocks>(&external_flat)?;
        let internal_constants_buf =
            device.upload::<Goldilocks>(&params.internal_constants)?;
        let internal_diagonal_buf =
            device.upload::<Goldilocks>(&params.internal_diagonal)?;

        let params_uniform = create_uniform(
            device.raw_device(),
            &ParamsUniform {
                num_permutations: 0,
                rounds_f_half: params.rounds_f_half as u32,
                rounds_p: params.rounds_p as u32,
                row_stride: 0,
            },
            "goldilocks poseidon2 params uniform",
        );

        // --- Pipeline + BGL -------------------------------------------
        let registry = device.pipeline_registry();
        let module = registry.get_or_create_module(
            device.raw_device(),
            GOLDILOCKS_POSEIDON2_WGSL,
            "goldilocks_poseidon2",
        );
        let bgl = registry.get_or_create_bgl(
            device.raw_device(),
            POSEIDON2_BGL_LABEL,
            &[
                bgl_storage_entry(0, false),
                bgl_storage_entry(1, true),
                bgl_storage_entry(2, true),
                bgl_storage_entry(3, true),
                bgl_uniform_entry(4),
            ],
        );
        let layout =
            device
                .raw_device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Goldilocks Poseidon2 pipeline layout"),
                    bind_group_layouts: &[Some(&bgl)],
                    immediate_size: 0,
                });
        let pipeline = registry.get_or_create_pipeline(
            device.raw_device(),
            GOLDILOCKS_POSEIDON2_WGSL,
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

    /// `(rounds_f_half, rounds_p)`. See the BabyBear twin for rationale.
    pub fn params_info(&self) -> (u32, u32) {
        (self.rounds_f_half, self.rounds_p)
    }

    pub fn width(&self) -> usize {
        WIDTH
    }

    /// Permute each `WIDTH`-element block of `buf` in place.
    pub fn execute(
        &mut self,
        device: &WgpuDevice,
        buf: &mut WgpuBuffer<Goldilocks>,
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
            return Ok(());
        }

        let max_wg = device.caps.max_compute_workgroups_per_dimension;
        let dispatch: LinearDispatch =
            plan_linear_dispatch(num_permutations, WORKGROUP_SIZE, max_wg)?;
        let row_stride = dispatch.groups_per_row * WORKGROUP_SIZE;

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

        let bind_group = device.raw_device().create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: Some("goldilocks poseidon2 bg"),
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

        let mut encoder = device.raw_device().create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("goldilocks poseidon2 encoder"),
            },
        );
        {
            let mut pass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("goldilocks poseidon2 pass"),
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

// --- Bind-group helpers -------------------------------------------------

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

    /// Single-permutation bit-parity against the CPU reference, plus a
    /// pinned assertion against the `goldilocks_regression_state_0001`
    /// anchor in `zkgpu-poseidon2`. Catches any drift in the u32x2
    /// Goldilocks arithmetic, round ordering, or external-matrix
    /// implementation independently of the CPU pin.
    #[test]
    fn gl_poseidon2_gpu_matches_cpu_regression_state_0001() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let params = Poseidon2Params::<Goldilocks, WIDTH>::goldilocks_default();
        let cpu = Poseidon2::new(params.clone());

        let mut cpu_state = [Goldilocks::new(0); WIDTH];
        cpu_state[0] = Goldilocks::new(1);
        cpu.permute(&mut cpu_state);

        let mut gpu_state = vec![Goldilocks::new(0); WIDTH];
        gpu_state[0] = Goldilocks::new(1);
        let mut buf = device.upload::<Goldilocks>(&gpu_state).unwrap();
        let mut plan =
            WgpuGoldilocksPoseidon2Plan::new(&device, params).unwrap();
        plan.execute(&device, &mut buf).unwrap();
        let gpu_out = buf.read_to_vec_blocking().unwrap();

        assert_eq!(
            gpu_out.as_slice(),
            cpu_state.as_slice(),
            "GPU Goldilocks Poseidon2 must match CPU reference on \
             regression anchor",
        );

        // Also pin to the literal u64 array — independent check in
        // case the CPU pin itself ever drifts.
        let expected: [u64; WIDTH] = [
            2188074367496775180,
            5885830467094400763,
            3310312858303912864,
            3736067622965212886,
            14578661067055100765,
            7771460173176447673,
            16719455422659298413,
            12878948591403318662,
            9702470661204462942,
            13973340836048818744,
            6474905020163181456,
            7259406223061710650,
            10585164742300593476,
            1863897901684124130,
            12515919369749004495,
            15547016035687218023,
        ];
        let got: [u64; WIDTH] = std::array::from_fn(|i| gpu_out[i].to_repr());
        assert_eq!(
            got, expected,
            "Goldilocks GPU regression_state_0001 drift — CPU and GPU \
             disagree on the pinned anchor"
        );
    }

    /// Batch differential (same shape as the BabyBear twin). 17
    /// distinct inputs, prime to catch any 2D-fold dispatch off-by-one.
    #[test]
    fn gl_poseidon2_gpu_batch_matches_cpu_per_input() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let params = Poseidon2Params::<Goldilocks, WIDTH>::goldilocks_default();
        let cpu = Poseidon2::new(params.clone());

        let num = 17usize;
        let mut gpu_flat: Vec<Goldilocks> = Vec::with_capacity(num * WIDTH);
        let mut cpu_blocks: Vec<[Goldilocks; WIDTH]> = Vec::with_capacity(num);
        for p in 0..num {
            let mut s = [Goldilocks::new(0); WIDTH];
            for (i, slot) in s.iter_mut().enumerate() {
                // Mix p and i into a 64-bit value; large values
                // exercise the upper half of the u32x2 limb pair.
                let v = ((p as u64).wrapping_mul(0x9E3779B97F4A7C15))
                    .wrapping_add((i as u64).wrapping_mul(0xBF58476D1CE4E5B9));
                *slot = Goldilocks::new(v);
            }
            gpu_flat.extend_from_slice(&s);
            cpu_blocks.push(s);
        }

        for block in cpu_blocks.iter_mut() {
            cpu.permute(block);
        }

        let mut buf = device.upload::<Goldilocks>(&gpu_flat).unwrap();
        let mut plan =
            WgpuGoldilocksPoseidon2Plan::new(&device, params).unwrap();
        plan.execute(&device, &mut buf).unwrap();
        let gpu_out = buf.read_to_vec_blocking().unwrap();

        for p in 0..num {
            let gpu_slice = &gpu_out[p * WIDTH..(p + 1) * WIDTH];
            assert_eq!(
                gpu_slice,
                cpu_blocks[p].as_slice(),
                "GPU Goldilocks Poseidon2 mismatch in batch slot {p}",
            );
        }
    }

    #[test]
    fn gl_poseidon2_gpu_empty_batch_is_noop() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let params = Poseidon2Params::<Goldilocks, WIDTH>::goldilocks_default();
        let empty: Vec<Goldilocks> = Vec::new();
        let mut buf = device.upload::<Goldilocks>(&empty).unwrap();
        let mut plan =
            WgpuGoldilocksPoseidon2Plan::new(&device, params).unwrap();
        plan.execute(&device, &mut buf).unwrap();
    }

    #[test]
    fn gl_poseidon2_gpu_rejects_mis_sized_buffer() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let params = Poseidon2Params::<Goldilocks, WIDTH>::goldilocks_default();
        let data = vec![Goldilocks::new(0); 25];
        let mut buf = device.upload::<Goldilocks>(&data).unwrap();
        let mut plan =
            WgpuGoldilocksPoseidon2Plan::new(&device, params).unwrap();
        let err = plan.execute(&device, &mut buf);
        assert!(
            matches!(err, Err(ZkGpuError::InvalidNttSize(_))),
            "expected InvalidNttSize, got {err:?}"
        );
    }

    #[test]
    fn gl_poseidon2_gpu_is_deterministic() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let params = Poseidon2Params::<Goldilocks, WIDTH>::goldilocks_default();
        let mut plan =
            WgpuGoldilocksPoseidon2Plan::new(&device, params).unwrap();

        let mut input = [Goldilocks::new(0); WIDTH];
        for (i, slot) in input.iter_mut().enumerate() {
            *slot = Goldilocks::new((i as u64) + 1);
        }

        let mut buf1 = device.upload::<Goldilocks>(&input).unwrap();
        plan.execute(&device, &mut buf1).unwrap();
        let out1 = buf1.read_to_vec_blocking().unwrap();

        let mut buf2 = device.upload::<Goldilocks>(&input).unwrap();
        plan.execute(&device, &mut buf2).unwrap();
        let out2 = buf2.read_to_vec_blocking().unwrap();

        assert_eq!(
            out1, out2,
            "Goldilocks Poseidon2 must be deterministic across runs"
        );
    }

    /// Post-F.1-review parity: reject unsupported alpha on the
    /// Goldilocks path too.
    #[test]
    fn gl_poseidon2_rejects_unsupported_alpha() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let mut params =
            Poseidon2Params::<Goldilocks, WIDTH>::goldilocks_default();
        params.alpha = 3;
        let err = WgpuGoldilocksPoseidon2Plan::new(&device, params);
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
}
