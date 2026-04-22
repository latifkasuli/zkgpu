//! Phase 7 Step 3.a — GPU Poseidon2 leaf sponge plan.
//!
//! Runs `PaddingFreeSponge<Perm24, 24, 16, 8>` over each row of a
//! row-major `h × w` BabyBear matrix, producing `h * 8` digest
//! elements in the output buffer.
//!
//! Consumes [`Poseidon2Params<BabyBear, 24>`] configured for the
//! Plonky3 variant — use `zkgpu_plonky3::poseidon2_bridge::
//! babybear_plonky3_params::<24>(..)` to build it. The plan rejects
//! `M4Variant::HorizonLabs` params up front.
//!
//! Designed with Step 2 in mind:
//! [`WgpuPoseidon2MerkleLeafPlan::hash_rows`] accepts the matrix
//! buffer as a GPU-resident `WgpuBuffer<BabyBear>`, so once Step 2
//! lands a GPU-resident `coset_lde_batch`, the LDE output can feed
//! this plan without a host round-trip. The host-fed entry point
//! [`WgpuPoseidon2MerkleLeafPlan::hash_host_matrix`] is a convenience
//! wrapper around the GPU-fed path.

use std::sync::Arc;

use zkgpu_babybear::BabyBear;
use zkgpu_core::{GpuBuffer, GpuDevice, ZkGpuError};
use zkgpu_poseidon2::{M4Variant, Poseidon2Params};

use crate::buffer::WgpuBuffer;
use crate::device::WgpuDevice;
use crate::dispatch::{plan_linear_dispatch, LinearDispatch};

const MERKLE_LEAF_WGSL: &str =
    include_str!("../kernels/portable/babybear_poseidon2_merkle_leaf.wgsl");
const BGL_LABEL: &str = "BabyBear Poseidon2 Merkle Leaf BGL";
const ENTRY: &str = "merkle_leaf_hash";
const WORKGROUP_SIZE: u32 = 64;

const WIDTH: usize = 24;
/// Digest length (OUT in `PaddingFreeSponge<_, 24, 16, 8>`).
pub const DIGEST_LEN: usize = 8;

/// Must match `MerkleLeafParams` in the WGSL kernel.
///
/// The WebGPU baseline caps `max_storage_buffers_per_shader_stage = 4`,
/// so the three Poseidon2 constant tables (external RC, internal RC,
/// internal diagonal) are packed into one `poseidon_constants` buffer
/// with kernel-side offsets. That leaves one storage slot each for
/// `input_matrix` and `digests`, and one slot free for future
/// extensions (total: 3 storage + 1 uniform).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ParamsUniform {
    num_leaves: u32,
    row_width: u32,
    rounds_f_half: u32,
    rounds_p: u32,
    row_stride: u32,
    internal_rc_offset: u32,
    internal_diag_offset: u32,
    _pad: u32,
}

/// GPU plan that hashes each row of a matrix into an 8-element
/// Poseidon2 digest using `PaddingFreeSponge<Perm24, 24, 16, 8>`.
pub struct WgpuPoseidon2MerkleLeafPlan {
    rounds_f_half: u32,
    rounds_p: u32,
    internal_rc_offset: u32,
    internal_diag_offset: u32,

    /// Packed `[external | internal_rc | internal_diag]`.
    poseidon_constants_buf: WgpuBuffer<BabyBear>,
    params_uniform: wgpu::Buffer,

    bgl: Arc<wgpu::BindGroupLayout>,
    pipeline: Arc<wgpu::ComputePipeline>,
}

impl WgpuPoseidon2MerkleLeafPlan {
    /// Build the plan from Plonky3-compatible width-24 Poseidon2 params.
    pub fn new(
        device: &WgpuDevice,
        params: Poseidon2Params<BabyBear, 24>,
    ) -> Result<Self, ZkGpuError> {
        if params.m4_variant != M4Variant::Plonky3 {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "Merkle leaf plan requires M4Variant::Plonky3, got {:?}",
                params.m4_variant,
            )));
        }
        if params.alpha != 7 {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "Merkle leaf plan requires alpha=7, got {}",
                params.alpha,
            )));
        }
        let expected_external_rows = 2 * params.rounds_f_half;
        if params.external_constants.len() != expected_external_rows {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "Merkle leaf plan: expected {} external rows, got {}",
                expected_external_rows,
                params.external_constants.len(),
            )));
        }
        if params.internal_constants.len() != params.rounds_p {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "Merkle leaf plan: expected {} internal constants, got {}",
                params.rounds_p,
                params.internal_constants.len(),
            )));
        }

        // Pack all Poseidon2 constants into one flat buffer:
        //   [ external (2*rf_half*WIDTH) | internal_rc (rp) | internal_diag (WIDTH) ]
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
                num_leaves: 0,
                row_width: 0,
                rounds_f_half: params.rounds_f_half as u32,
                rounds_p: params.rounds_p as u32,
                row_stride: 0,
                internal_rc_offset: internal_rc_offset as u32,
                internal_diag_offset: internal_diag_offset as u32,
                _pad: 0,
            },
            "merkle leaf params uniform",
        );

        let registry = device.pipeline_registry();
        let module = registry.get_or_create_module(
            device.raw_device(),
            MERKLE_LEAF_WGSL,
            "babybear_poseidon2_merkle_leaf",
        );
        let bgl = registry.get_or_create_bgl(
            device.raw_device(),
            BGL_LABEL,
            &[
                bgl_storage_entry(0, true),  // input_matrix
                bgl_storage_entry(1, false), // digests
                bgl_storage_entry(2, true),  // poseidon_constants (packed)
                bgl_uniform_entry(3),        // params
            ],
        );
        let layout = device.raw_device().create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("BabyBear Poseidon2 Merkle Leaf pipeline layout"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            },
        );
        let pipeline = registry.get_or_create_pipeline(
            device.raw_device(),
            MERKLE_LEAF_WGSL,
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

    /// Digest length produced per leaf.
    pub fn digest_len(&self) -> usize {
        DIGEST_LEN
    }

    /// Hash each row of a GPU-resident matrix into an 8-element digest.
    ///
    /// `input` is the `h × w` row-major matrix as a flat GPU buffer of
    /// length `num_leaves * row_width`. Output `digests` must be a
    /// GPU buffer of length `num_leaves * DIGEST_LEN`.
    ///
    /// This is the GPU-resident entry point — Step 2's future GPU
    /// `coset_lde_batch` output will feed directly into this without a
    /// host round-trip.
    pub fn hash_rows(
        &mut self,
        device: &WgpuDevice,
        input: &WgpuBuffer<BabyBear>,
        digests: &mut WgpuBuffer<BabyBear>,
        num_leaves: u32,
        row_width: u32,
    ) -> Result<(), ZkGpuError> {
        if num_leaves == 0 {
            return Ok(());
        }
        let expected_input_len = (num_leaves as usize) * (row_width as usize);
        if input.len() != expected_input_len {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "Merkle leaf input buffer length {} != num_leaves*row_width ({}*{}={})",
                input.len(),
                num_leaves,
                row_width,
                expected_input_len,
            )));
        }
        let expected_digest_len = (num_leaves as usize) * DIGEST_LEN;
        if digests.len() != expected_digest_len {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "Merkle leaf digest buffer length {} != num_leaves*{DIGEST_LEN} = {expected_digest_len}",
                digests.len(),
            )));
        }

        let max_wg = device.caps.max_compute_workgroups_per_dimension;
        let dispatch: LinearDispatch =
            plan_linear_dispatch(num_leaves, WORKGROUP_SIZE, max_wg)?;
        let row_stride = dispatch.groups_per_row * WORKGROUP_SIZE;

        let uniform = ParamsUniform {
            num_leaves,
            row_width,
            rounds_f_half: self.rounds_f_half,
            rounds_p: self.rounds_p,
            row_stride,
            internal_rc_offset: self.internal_rc_offset,
            internal_diag_offset: self.internal_diag_offset,
            _pad: 0,
        };
        device.raw_queue().write_buffer(
            &self.params_uniform,
            0,
            bytemuck::bytes_of(&uniform),
        );

        let bind_group = device.raw_device().create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: Some("merkle leaf bg"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input.inner.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: digests.inner.as_entire_binding(),
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
                label: Some("merkle leaf encoder"),
            },
        );
        {
            let mut pass = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor {
                    label: Some("merkle leaf pass"),
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

    /// Convenience: hash rows of a host-side matrix. Uploads the
    /// matrix, allocates a digest buffer, runs `hash_rows`, and
    /// downloads the digests as `Vec<BabyBear>` of length
    /// `num_leaves * 8`.
    ///
    /// For production use, prefer [`Self::hash_rows`] with
    /// GPU-resident buffers to avoid the upload/download round-trip.
    pub fn hash_host_matrix(
        &mut self,
        device: &WgpuDevice,
        matrix: &[BabyBear],
        num_leaves: u32,
        row_width: u32,
    ) -> Result<Vec<BabyBear>, ZkGpuError> {
        // Empty-batch shortcut: wgpu rejects zero-size buffer inits,
        // so skip the GPU round-trip entirely. Semantically matches
        // the num_leaves==0 early-return in `hash_rows`.
        if num_leaves == 0 {
            return Ok(Vec::new());
        }
        // row_width==0 is a legal Plonky3 input (every row's `hash_iter`
        // yields an all-zero digest). Bypass the GPU dispatch and
        // return a freshly zeroed host vec — cheaper than uploading a
        // dummy input just to satisfy the size check.
        if row_width == 0 {
            return Ok(vec![BabyBear::new(0); (num_leaves as usize) * DIGEST_LEN]);
        }
        let input = device.upload::<BabyBear>(matrix)?;
        let mut digests = device.alloc_zeros::<BabyBear>((num_leaves as usize) * DIGEST_LEN)?;
        self.hash_rows(device, &input, &mut digests, num_leaves, row_width)?;
        digests.read_to_vec()
    }
}

// --- Local BGL helpers (same as the other poseidon2 plan files) ------------

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
