//! Phase 3.d Stage 1a — GPU Poseidon2 leaf sponge plan, **W16 / RATE=8**
//! variant (OpenVM-compatible shape).
//!
//! Sibling to [`super::merkle_leaf::WgpuPoseidon2MerkleLeafPlan`]
//! (the Plonky3 canonical W24/RATE=16 shape). Runs
//! `PaddingFreeSponge<Perm16, 16, 8, 8>::hash_iter` over each row of
//! an `h × w` BabyBear matrix, producing `h * 8` digest elements.
//!
//! This is the leaf-sponge shape that matches OpenVM's
//! `stark-backend` BabyBear Poseidon2 config:
//!
//! ```ignore
//! // openvm-org/stark-backend/crates/stark-sdk/src/config/baby_bear_poseidon2.rs
//! const WIDTH: usize = 16;
//! const RATE: usize = 8;
//! const DIGEST_WIDTH: usize = 8;
//! type Hash<P> = PaddingFreeSponge<P, WIDTH, RATE, DIGEST_WIDTH>;
//! ```
//!
//! Consumes [`Poseidon2Params<BabyBear, 16>`] (Plonky3-variant, α=7)
//! — use `zkgpu_plonky3::poseidon2_bridge::babybear_plonky3_params::<16>(..)`
//! to build it.
//!
//! # Why a sibling module instead of a const-generic generic
//!
//! WGSL can't be parameterised over width / rate at runtime without
//! string-templating the kernel source. Each shape needs its own
//! `@compute` entry (different state-array size, different absorption
//! loop, different `mul_external` block count). Rust-side, the two
//! plan types share almost all their logic but carry the WGSL file
//! and shape constants as compile-time identity, which keeps the
//! type system honest about "don't cross shape boundaries."
//!
//! The mixed-height DAG engine (Stage 1b) takes either shape via a
//! small internal trait so the commit orchestration is shared.

use std::sync::Arc;

use zkgpu_babybear::BabyBear;
use zkgpu_core::{GpuBuffer, GpuDevice, ZkGpuError};
use zkgpu_poseidon2::{M4Variant, Poseidon2Params};

use crate::buffer::WgpuBuffer;
use crate::device::WgpuDevice;
use crate::dispatch::{plan_linear_dispatch, LinearDispatch};

const MERKLE_LEAF_W16_WGSL: &str =
    include_str!("../kernels/portable/babybear_poseidon2_merkle_leaf_w16.wgsl");
const BGL_LABEL: &str = "BabyBear Poseidon2 Merkle Leaf W16 BGL";
const ENTRY: &str = "merkle_leaf_hash";
const WORKGROUP_SIZE: u32 = 64;

const WIDTH: usize = 16;
#[allow(dead_code)] // referenced in compile-time checks at the bottom of the file
const RATE: usize = 8;
/// Digest length (OUT in `PaddingFreeSponge<_, 16, 8, 8>`). Same as
/// the W24 plan and the shared backend's `DIGEST_LEN`.
pub const DIGEST_LEN: usize = 8;

/// Matches the WGSL `MerkleLeafParams` block (3 storage + 1 uniform).
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
/// Poseidon2 digest using `PaddingFreeSponge<Perm16, 16, 8, 8>`.
///
/// Paired with [`super::merkle_leaf::WgpuPoseidon2MerkleLeafPlan`]
/// (the W24/RATE=16 Plonky3 canonical variant) by a small internal
/// trait in the shared backend so the mixed-height DAG orchestrator
/// can consume either shape without duplicating commit logic.
pub struct WgpuPoseidon2MerkleLeafW16R8Plan {
    rounds_f_half: u32,
    rounds_p: u32,
    internal_rc_offset: u32,
    internal_diag_offset: u32,

    poseidon_constants_buf: WgpuBuffer<BabyBear>,
    params_uniform: wgpu::Buffer,

    bgl: Arc<wgpu::BindGroupLayout>,
    pipeline: Arc<wgpu::ComputePipeline>,
}

impl WgpuPoseidon2MerkleLeafW16R8Plan {
    /// Build the plan from Plonky3-compatible width-16 Poseidon2 params.
    pub fn new(
        device: &WgpuDevice,
        params: Poseidon2Params<BabyBear, 16>,
    ) -> Result<Self, ZkGpuError> {
        if params.m4_variant != M4Variant::Plonky3 {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "W16 leaf plan requires M4Variant::Plonky3, got {:?}",
                params.m4_variant,
            )));
        }
        if params.alpha != 7 {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "W16 leaf plan requires alpha=7, got {}",
                params.alpha,
            )));
        }
        let expected_external_rows = 2 * params.rounds_f_half;
        if params.external_constants.len() != expected_external_rows {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "W16 leaf plan: expected {} external rows, got {}",
                expected_external_rows,
                params.external_constants.len(),
            )));
        }
        if params.internal_constants.len() != params.rounds_p {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "W16 leaf plan: expected {} internal constants, got {}",
                params.rounds_p,
                params.internal_constants.len(),
            )));
        }

        // Pack constants: [ external | internal_rc | internal_diag ].
        // Same layout as the W24 kernel, just shorter total length
        // because external rows are WIDTH=16 elements wide (vs 24).
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
            "merkle leaf w16 params uniform",
        );

        let registry = device.pipeline_registry();
        let module = registry.get_or_create_module(
            device.raw_device(),
            MERKLE_LEAF_W16_WGSL,
            "babybear_poseidon2_merkle_leaf_w16",
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
                label: Some("BabyBear Poseidon2 Merkle Leaf W16 pipeline layout"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            },
        );
        let pipeline = registry.get_or_create_pipeline(
            device.raw_device(),
            MERKLE_LEAF_W16_WGSL,
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

    /// Digest length produced per leaf (8, always).
    pub const fn digest_len(&self) -> usize {
        DIGEST_LEN
    }

    /// Hash each row of a GPU-resident matrix into an 8-element
    /// digest. Semantics match
    /// `PaddingFreeSponge<Perm16, 16, 8, 8>::hash_iter`.
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
                "W16 leaf input buffer length {} != num_leaves*row_width ({}*{}={})",
                input.len(),
                num_leaves,
                row_width,
                expected_input_len,
            )));
        }
        let expected_digest_len = (num_leaves as usize) * DIGEST_LEN;
        if digests.len() != expected_digest_len {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "W16 leaf digest buffer length {} != num_leaves*{DIGEST_LEN} = {expected_digest_len}",
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
        device
            .raw_queue()
            .write_buffer(&self.params_uniform, 0, bytemuck::bytes_of(&uniform));

        let bind_group = device.raw_device().create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: Some("merkle leaf w16 bg"),
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
                label: Some("merkle leaf w16 encoder"),
            },
        );
        {
            let mut pass = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor {
                    label: Some("merkle leaf w16 pass"),
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

    /// Convenience: upload host matrix, run hash_rows, read digests.
    ///
    /// Mirrors the W24 plan's `hash_host_matrix` contract bit-for-bit
    /// including the shape-overflow and zero-shape guards.
    pub fn hash_host_matrix(
        &mut self,
        device: &WgpuDevice,
        matrix: &[BabyBear],
        num_leaves: u32,
        row_width: u32,
    ) -> Result<Vec<BabyBear>, ZkGpuError> {
        let expected_len = (num_leaves as usize)
            .checked_mul(row_width as usize)
            .ok_or_else(|| {
                ZkGpuError::InvalidNttSize(format!(
                    "W16 leaf host matrix shape overflow: {num_leaves} * {row_width}"
                ))
            })?;
        if matrix.len() != expected_len {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "W16 leaf host matrix length {} != num_leaves*row_width ({}*{}={})",
                matrix.len(),
                num_leaves,
                row_width,
                expected_len,
            )));
        }
        if num_leaves == 0 {
            return Ok(Vec::new());
        }
        if row_width == 0 {
            return Ok(vec![BabyBear::new(0); (num_leaves as usize) * DIGEST_LEN]);
        }
        let input = device.upload::<BabyBear>(matrix)?;
        let mut digests = device.alloc_zeros::<BabyBear>((num_leaves as usize) * DIGEST_LEN)?;
        self.hash_rows(device, &input, &mut digests, num_leaves, row_width)?;
        digests.read_to_vec()
    }
}

// --- Local BGL helpers (identical to the W24 sibling) ---

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

// Compile-time check that RATE is consistent with WIDTH usage.
const _: () = {
    assert!(RATE < WIDTH, "sponge RATE must be < WIDTH");
    assert!(DIGEST_LEN <= RATE, "DIGEST_LEN must fit in the rate slots");
};
