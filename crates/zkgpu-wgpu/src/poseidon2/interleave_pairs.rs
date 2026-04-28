//! GPU pair-interleave plan for the mixed-height Merkle commit DAG.
//!
//! Replaces the host-side `interleave_pairs_host` memcpy step in
//! [`super::merkle_commit_dag`] with a small compute kernel. Used at
//! injection levels of the mixed-height commit engine to combine the
//! previous level's pairwise compression output (`left`) with the
//! current level's injected leaf-hash digests (`right`) into one
//! 2N-digest interleaved buffer that the next compress pass consumes.
//!
//! Why this matters: pre-item-#1 the engine downloaded `temp_gpu` to
//! host, allocated an interleaved Vec on host, and re-uploaded it to
//! GPU before the second compression. With the GPU-resident
//! interleave, the data never leaves the device — eliminating the
//! per-injection-level host round-trip that dominates mixed-height
//! commit latency on discrete GPUs.

use std::sync::Arc;

use zkgpu_babybear::BabyBear;
use zkgpu_core::{GpuBuffer, ZkGpuError};

use crate::buffer::WgpuBuffer;
use crate::device::WgpuDevice;
use crate::dispatch::{plan_linear_dispatch, LinearDispatch};

const INTERLEAVE_WGSL: &str =
    include_str!("../kernels/portable/babybear_poseidon2_interleave_pairs.wgsl");
const BGL_LABEL: &str = "BabyBear Poseidon2 Interleave Pairs BGL";
const ENTRY: &str = "interleave_pairs";
const WORKGROUP_SIZE: u32 = 64;
const DIGEST_LEN: usize = 8;

/// Matches `InterleaveParams` in the WGSL kernel.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ParamsUniform {
    /// Number of digests per input buffer (`left.len() / DIGEST_LEN`).
    n: u32,
    /// 2D dispatch fold: number of workgroups across the x dimension
    /// before wrapping into y. Sourced from
    /// [`crate::dispatch::LinearDispatch::groups_per_row`].
    groups_per_row: u32,
    _pad0: u32,
    _pad1: u32,
}

/// GPU plan that interleaves two N-digest buffers into a 2N-digest
/// buffer with pair-i = `(left[i], right[i])`. Single-dispatch.
///
/// Cheap to construct: no Poseidon constants, no specialized
/// permutation. Holds one BGL + one pipeline + one params uniform.
pub struct WgpuPoseidon2InterleavePairsPlan {
    params_uniform: wgpu::Buffer,
    bgl: Arc<wgpu::BindGroupLayout>,
    pipeline: Arc<wgpu::ComputePipeline>,
}

impl WgpuPoseidon2InterleavePairsPlan {
    /// Build the plan. No Poseidon-specific configuration; the kernel
    /// is a pure data-shuffling step.
    pub fn new(device: &WgpuDevice) -> Result<Self, ZkGpuError> {
        let raw = device.raw_device();
        let registry = device.pipeline_registry();

        let module = registry.get_or_create_module(
            raw,
            INTERLEAVE_WGSL,
            "babybear_poseidon2_interleave_pairs",
        );

        let bgl = registry.get_or_create_bgl(
            raw,
            BGL_LABEL,
            &[
                // binding 0: left (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: right (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 2: out (read-write storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 3: params (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        );

        let layout = raw.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Interleave pairs pipeline layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = registry.get_or_create_pipeline(
            raw,
            INTERLEAVE_WGSL,
            ENTRY,
            BGL_LABEL,
            &layout,
            &module,
            None,
        );

        let params_uniform = raw.create_buffer(&wgpu::BufferDescriptor {
            label: Some("interleave pairs params uniform"),
            size: std::mem::size_of::<ParamsUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            params_uniform,
            bgl,
            pipeline,
        })
    }

    /// Run one interleave dispatch.
    ///
    /// * `left` and `right` must each contain at least `n * DIGEST_LEN`
    ///   elements. Trailing slots are ignored.
    /// * `output` must contain at least `2 * n * DIGEST_LEN` elements.
    ///   Trailing slots are left untouched.
    /// * The three buffers must be distinct (no aliasing).
    pub fn interleave(
        &mut self,
        device: &WgpuDevice,
        left: &WgpuBuffer<BabyBear>,
        right: &WgpuBuffer<BabyBear>,
        output: &mut WgpuBuffer<BabyBear>,
        n: u32,
    ) -> Result<(), ZkGpuError> {
        if n == 0 {
            return Ok(());
        }
        let required_in = (n as usize) * DIGEST_LEN;
        if left.len() < required_in {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "interleave left length {} < n*{DIGEST_LEN} = {required_in}",
                left.len(),
            )));
        }
        if right.len() < required_in {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "interleave right length {} < n*{DIGEST_LEN} = {required_in}",
                right.len(),
            )));
        }
        let required_out = 2 * required_in;
        if output.len() < required_out {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "interleave output length {} < 2*n*{DIGEST_LEN} = {required_out}",
                output.len(),
            )));
        }

        // 2D-folded dispatch via the shared `plan_linear_dispatch`
        // helper. The previous flat 1D dispatch worked at small `n`
        // but became a hard ceiling at large mixed-height shapes:
        // with the typical 65,535 per-dimension limit and
        // WORKGROUP_SIZE = 64, an injection level with `n = 2^22`
        // digests already needs 65,536 workgroups, so a tallest
        // matrix at `h_max = 2^23` with an injected group at
        // `h_max/2` would have failed before dispatch. With 2D
        // folding the only remaining bound is
        // `max_compute_workgroups_per_dimension^2` (≈ 2^32 total
        // workgroups on most backends), well above any reachable
        // BabyBear shape.
        let max_wg = device.caps.max_compute_workgroups_per_dimension;
        let dispatch: LinearDispatch =
            plan_linear_dispatch(n, WORKGROUP_SIZE, max_wg)?;

        let uniform = ParamsUniform {
            n,
            groups_per_row: dispatch.groups_per_row,
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
                label: Some("interleave pairs bg"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: left.inner.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: right.inner.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.inner.as_entire_binding(),
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
                label: Some("interleave pairs encoder"),
            },
        );
        {
            let mut pass = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor {
                    label: Some("interleave pairs pass"),
                    timestamp_writes: None,
                },
            );
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch.x, dispatch.y, 1);
        }
        device.raw_queue().submit(Some(encoder.finish()));

        Ok(())
    }
}
