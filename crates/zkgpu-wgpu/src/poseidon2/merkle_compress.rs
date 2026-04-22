//! Phase 7 Step 3.b — GPU Poseidon2 Merkle-tree compression plan
//! (width-16, Plonky3 variant).
//!
//! Mirrors `p3_symmetric::TruncatedPermutation<Poseidon2BabyBear<16>, 2, 8, 16>`:
//! takes an input layer of `2 * num_outputs` child digests
//! (row-major, 8 u32s each) and writes `num_outputs` parent digests to
//! a separate output buffer. Each parent is the first 8 state slots
//! after running Plonky3 Poseidon2 W16 over `left ∥ right`.
//!
//! Consumes [`Poseidon2Params<BabyBear, 16>`] built via
//! `zkgpu_plonky3::poseidon2_bridge::babybear_plonky3_params::<16>(..)`
//! — the *compression* permutation is distinct from the W24
//! *leaf-sponge* permutation; both have their own random constants in
//! Plonky3's canonical MMCS config.
//!
//! Same 3-storage + 1-uniform BGL shape as the leaf-sponge plan — the
//! three constant tables are packed into one `poseidon_constants`
//! buffer so the BGL fits inside the WebGPU baseline cap of 4 storage
//! buffers per shader stage. See the Step 3.a plan module doc for the
//! rationale and the P2 review that surfaced this trap.

use std::sync::Arc;

use zkgpu_babybear::BabyBear;
use zkgpu_core::{GpuBuffer, GpuDevice, ZkGpuError};
use zkgpu_poseidon2::{M4Variant, Poseidon2Params};

use crate::buffer::WgpuBuffer;
use crate::device::WgpuDevice;
use crate::dispatch::{plan_linear_dispatch, LinearDispatch};

const COMPRESS_WGSL: &str =
    include_str!("../kernels/portable/babybear_poseidon2_merkle_compress.wgsl");
const BGL_LABEL: &str = "BabyBear Poseidon2 Merkle Compress BGL";
const ENTRY: &str = "merkle_compress";
const WORKGROUP_SIZE: u32 = 64;

const WIDTH: usize = 16;
const DIGEST_LEN: usize = 8;

/// Matches `MerkleCompressParams` in the WGSL kernel.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ParamsUniform {
    num_outputs: u32,
    rounds_f_half: u32,
    rounds_p: u32,
    row_stride: u32,
    internal_rc_offset: u32,
    internal_diag_offset: u32,
    _pad0: u32,
    _pad1: u32,
}

/// GPU plan that runs one level of Merkle tree compression:
/// `num_outputs` pairs of 8-element digests → `num_outputs` 8-element
/// digests via Plonky3 Poseidon2 W16 truncated permutation.
pub struct WgpuPoseidon2MerkleCompressPlan {
    rounds_f_half: u32,
    rounds_p: u32,
    internal_rc_offset: u32,
    internal_diag_offset: u32,

    poseidon_constants_buf: WgpuBuffer<BabyBear>,
    params_uniform: wgpu::Buffer,

    bgl: Arc<wgpu::BindGroupLayout>,
    pipeline: Arc<wgpu::ComputePipeline>,
}

impl WgpuPoseidon2MerkleCompressPlan {
    /// Build the compression plan from Plonky3-compatible width-16
    /// Poseidon2 params. Rejects the HorizonLabs M_4 variant and any
    /// α ≠ 7 up front.
    pub fn new(
        device: &WgpuDevice,
        params: Poseidon2Params<BabyBear, 16>,
    ) -> Result<Self, ZkGpuError> {
        if params.m4_variant != M4Variant::Plonky3 {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "Merkle compress plan requires M4Variant::Plonky3, got {:?}",
                params.m4_variant,
            )));
        }
        if params.alpha != 7 {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "Merkle compress plan requires alpha=7, got {}",
                params.alpha,
            )));
        }
        let expected_external_rows = 2 * params.rounds_f_half;
        if params.external_constants.len() != expected_external_rows {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "Merkle compress plan: expected {} external rows, got {}",
                expected_external_rows,
                params.external_constants.len(),
            )));
        }
        if params.internal_constants.len() != params.rounds_p {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "Merkle compress plan: expected {} internal constants, got {}",
                params.rounds_p,
                params.internal_constants.len(),
            )));
        }

        // Pack constants: [ external | internal_rc | internal_diag ].
        let external_len = expected_external_rows * WIDTH;
        let internal_rc_offset = external_len;
        let internal_diag_offset = internal_rc_offset + params.rounds_p;
        let total_len = internal_diag_offset + WIDTH;

        let mut packed = Vec::with_capacity(total_len);
        for row in &params.external_constants {
            packed.extend_from_slice(row);
        }
        packed.extend_from_slice(&params.internal_constants);
        packed.extend_from_slice(&params.internal_diagonal);
        debug_assert_eq!(packed.len(), total_len);
        let poseidon_constants_buf = device.upload::<BabyBear>(&packed)?;

        let params_uniform = create_uniform(
            device.raw_device(),
            &ParamsUniform {
                num_outputs: 0,
                rounds_f_half: params.rounds_f_half as u32,
                rounds_p: params.rounds_p as u32,
                row_stride: 0,
                internal_rc_offset: internal_rc_offset as u32,
                internal_diag_offset: internal_diag_offset as u32,
                _pad0: 0,
                _pad1: 0,
            },
            "merkle compress params uniform",
        );

        let registry = device.pipeline_registry();
        let module = registry.get_or_create_module(
            device.raw_device(),
            COMPRESS_WGSL,
            "babybear_poseidon2_merkle_compress",
        );
        let bgl = registry.get_or_create_bgl(
            device.raw_device(),
            BGL_LABEL,
            &[
                bgl_storage_entry(0, true),  // input_digests
                bgl_storage_entry(1, false), // output_digests
                bgl_storage_entry(2, true),  // poseidon_constants (packed)
                bgl_uniform_entry(3),        // params
            ],
        );
        let layout = device.raw_device().create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("BabyBear Poseidon2 Merkle Compress pipeline layout"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            },
        );
        let pipeline = registry.get_or_create_pipeline(
            device.raw_device(),
            COMPRESS_WGSL,
            ENTRY,
            BGL_LABEL,
            &layout,
            &module,
            None,
        );

        Ok(Self {
            rounds_f_half: params.rounds_f_half as u32,
            rounds_p: params.rounds_p as u32,
            internal_rc_offset: internal_rc_offset as u32,
            internal_diag_offset: internal_diag_offset as u32,
            poseidon_constants_buf,
            params_uniform,
            bgl,
            pipeline,
        })
    }

    /// Run one compression level.
    ///
    /// * `input.len()` must be **at least** `2 * num_outputs * DIGEST_LEN`
    ///   — the kernel reads positions `[0, 2*num_outputs*DIGEST_LEN)`
    ///   and ignores any trailing slots.
    /// * `output.len()` must be **at least** `num_outputs * DIGEST_LEN`
    ///   — the kernel writes positions `[0, num_outputs*DIGEST_LEN)`
    ///   and leaves any trailing slots untouched.
    ///
    /// Both buffers are required to be distinct — the kernel reads
    /// from `input` and writes to `output`, no aliasing. The ≥-sized
    /// contract lets the commit orchestrator ping-pong between two
    /// `h/2`-sized scratch buffers across every compression level
    /// (each successive level writes half as many outputs but reuses
    /// the same allocations).
    pub fn compress_level(
        &mut self,
        device: &WgpuDevice,
        input: &WgpuBuffer<BabyBear>,
        output: &mut WgpuBuffer<BabyBear>,
        num_outputs: u32,
    ) -> Result<(), ZkGpuError> {
        if num_outputs == 0 {
            return Ok(());
        }
        let required_input_len = 2 * (num_outputs as usize) * DIGEST_LEN;
        if input.len() < required_input_len {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "Merkle compress input length {} < 2*num_outputs*{DIGEST_LEN} = {required_input_len}",
                input.len(),
            )));
        }
        let required_output_len = (num_outputs as usize) * DIGEST_LEN;
        if output.len() < required_output_len {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "Merkle compress output length {} < num_outputs*{DIGEST_LEN} = {required_output_len}",
                output.len(),
            )));
        }

        let max_wg = device.caps.max_compute_workgroups_per_dimension;
        let dispatch: LinearDispatch =
            plan_linear_dispatch(num_outputs, WORKGROUP_SIZE, max_wg)?;
        let row_stride = dispatch.groups_per_row * WORKGROUP_SIZE;

        let uniform = ParamsUniform {
            num_outputs,
            rounds_f_half: self.rounds_f_half,
            rounds_p: self.rounds_p,
            row_stride,
            internal_rc_offset: self.internal_rc_offset,
            internal_diag_offset: self.internal_diag_offset,
            _pad0: 0,
            _pad1: 0,
        };
        device.raw_queue().write_buffer(
            &self.params_uniform,
            0,
            bytemuck::bytes_of(&uniform),
        );

        let bind_group = device.raw_device().create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: Some("merkle compress bg"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input.inner.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: output.inner.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.poseidon_constants_buf.inner.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.params_uniform.as_entire_binding(),
                    },
                ],
            },
        );

        let mut encoder = device.raw_device().create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("merkle compress encoder"),
            },
        );
        {
            let mut pass = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor {
                    label: Some("merkle compress pass"),
                    timestamp_writes: None,
                },
            );
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

// --- Local BGL helpers (same shape as the other poseidon2 plan files) ---

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
