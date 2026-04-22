//! Phase 7 Step 1.5b — Plonky3-variant Poseidon2 plans at widths 16
//! and 24.
//!
//! Sibling to [`super::plan::WgpuBabyBearPoseidon2Plan`] (the width-16
//! HorizonLabs-variant plan from Phase F.1). Two plan types here:
//!
//! - [`WgpuBabyBearPoseidon2PlonkyW16Plan`] — width-16, Plonky3
//!   `M_4 = circ(2, 3, 1, 1)` matrix. Consumed by Plonky3's
//!   `TruncatedPermutation<Poseidon2BabyBear<16>, ..>` in the
//!   Merkle-tree compression path.
//! - [`WgpuBabyBearPoseidon2PlonkyW24Plan`] — width-24, Plonky3
//!   matrix. Consumed by `PaddingFreeSponge<Poseidon2BabyBear<24>, ..>`
//!   in the Merkle-tree leaf hashing path.
//!
//! Both plans accept a [`Poseidon2Params`] constructed via the
//! `zkgpu-plonky3` bridge (`poseidon2_bridge::babybear_plonky3_params`)
//! so the GPU kernel matches `p3_baby_bear::Poseidon2BabyBear` output
//! bit-for-bit.

use std::sync::Arc;

use zkgpu_babybear::BabyBear;
use zkgpu_core::{GpuBuffer, GpuDevice, ZkGpuError};
use zkgpu_poseidon2::{M4Variant, Poseidon2Params};

use crate::async_util;
use crate::buffer::WgpuBuffer;
use crate::device::WgpuDevice;
use crate::dispatch::{plan_linear_dispatch, LinearDispatch};

const PLONKY3_W16_WGSL: &str =
    include_str!("../kernels/portable/babybear_poseidon2_plonky3_w16.wgsl");
const PLONKY3_W24_WGSL: &str =
    include_str!("../kernels/portable/babybear_poseidon2_plonky3_w24.wgsl");

const WORKGROUP_SIZE: u32 = 64;

/// Uniform layout — same shape as the HorizonLabs-variant kernel's
/// Poseidon2Params, which the Plonky3 kernels mirror exactly.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ParamsUniform {
    num_permutations: u32,
    rounds_f_half: u32,
    rounds_p: u32,
    row_stride: u32,
}

// ---------------------------------------------------------------------------
// Width-16 plan
// ---------------------------------------------------------------------------

/// GPU Poseidon2 plan for BabyBear at width 16, Plonky3 M_4 variant.
pub struct WgpuBabyBearPoseidon2PlonkyW16Plan {
    inner: PlanInner<16>,
}

impl WgpuBabyBearPoseidon2PlonkyW16Plan {
    /// Build a plan from [`Poseidon2Params`] configured for Plonky3
    /// interop (via `zkgpu-plonky3::poseidon2_bridge::babybear_plonky3_params`).
    ///
    /// Rejects params whose `m4_variant` is not [`M4Variant::Plonky3`]
    /// — this plan is specialised for the Plonky3 matrix choice; the
    /// HorizonLabs variant has its own sibling plan type.
    pub fn new(
        device: &WgpuDevice,
        params: Poseidon2Params<BabyBear, 16>,
    ) -> Result<Self, ZkGpuError> {
        require_plonky3_variant(&params)?;
        let inner = PlanInner::<16>::build(
            device,
            params,
            PLONKY3_W16_WGSL,
            "babybear_poseidon2_plonky3_w16",
            "poseidon2_permute_w16",
            "BabyBear Poseidon2 Plonky3 W16 BGL",
        )?;
        Ok(Self { inner })
    }

    pub fn width(&self) -> usize {
        16
    }

    pub fn params_info(&self) -> (u32, u32) {
        (self.inner.rounds_f_half, self.inner.rounds_p)
    }

    pub fn execute(
        &mut self,
        device: &WgpuDevice,
        buf: &mut WgpuBuffer<BabyBear>,
    ) -> Result<(), ZkGpuError> {
        self.inner.execute_sync(device, buf)
    }

    pub async fn execute_async(
        &mut self,
        device: &WgpuDevice,
        buf: &mut WgpuBuffer<BabyBear>,
    ) -> Result<(), ZkGpuError> {
        self.inner.execute_async(device, buf).await
    }
}

// ---------------------------------------------------------------------------
// Width-24 plan
// ---------------------------------------------------------------------------

/// GPU Poseidon2 plan for BabyBear at width 24, Plonky3 M_4 variant.
pub struct WgpuBabyBearPoseidon2PlonkyW24Plan {
    inner: PlanInner<24>,
}

impl WgpuBabyBearPoseidon2PlonkyW24Plan {
    /// Build a plan from Plonky3-compatible [`Poseidon2Params`].
    pub fn new(
        device: &WgpuDevice,
        params: Poseidon2Params<BabyBear, 24>,
    ) -> Result<Self, ZkGpuError> {
        require_plonky3_variant(&params)?;
        let inner = PlanInner::<24>::build(
            device,
            params,
            PLONKY3_W24_WGSL,
            "babybear_poseidon2_plonky3_w24",
            "poseidon2_permute_w24",
            "BabyBear Poseidon2 Plonky3 W24 BGL",
        )?;
        Ok(Self { inner })
    }

    pub fn width(&self) -> usize {
        24
    }

    pub fn params_info(&self) -> (u32, u32) {
        (self.inner.rounds_f_half, self.inner.rounds_p)
    }

    pub fn execute(
        &mut self,
        device: &WgpuDevice,
        buf: &mut WgpuBuffer<BabyBear>,
    ) -> Result<(), ZkGpuError> {
        self.inner.execute_sync(device, buf)
    }

    pub async fn execute_async(
        &mut self,
        device: &WgpuDevice,
        buf: &mut WgpuBuffer<BabyBear>,
    ) -> Result<(), ZkGpuError> {
        self.inner.execute_async(device, buf).await
    }
}

// ---------------------------------------------------------------------------
// Shared internals
// ---------------------------------------------------------------------------

struct PlanInner<const W: usize> {
    rounds_f_half: u32,
    rounds_p: u32,

    external_constants_buf: WgpuBuffer<BabyBear>,
    internal_constants_buf: WgpuBuffer<BabyBear>,
    internal_diagonal_buf: WgpuBuffer<BabyBear>,
    params_uniform: wgpu::Buffer,

    bgl: Arc<wgpu::BindGroupLayout>,
    pipeline: Arc<wgpu::ComputePipeline>,
}

const SUPPORTED_ALPHA: u64 = 7;

fn require_plonky3_variant<const W: usize>(
    params: &Poseidon2Params<BabyBear, W>,
) -> Result<(), ZkGpuError> {
    if params.m4_variant != M4Variant::Plonky3 {
        return Err(ZkGpuError::InvalidNttSize(format!(
            "Plonky3 Poseidon2 plan requires M4Variant::Plonky3, got {:?}",
            params.m4_variant,
        )));
    }
    if params.alpha != SUPPORTED_ALPHA {
        return Err(ZkGpuError::InvalidNttSize(format!(
            "Plonky3 Poseidon2 plan requires alpha={SUPPORTED_ALPHA}, got {}",
            params.alpha,
        )));
    }
    Ok(())
}

impl<const W: usize> PlanInner<W> {
    fn build(
        device: &WgpuDevice,
        params: Poseidon2Params<BabyBear, W>,
        wgsl_source: &'static str,
        wgsl_label: &'static str,
        entry_point: &'static str,
        bgl_label: &'static str,
    ) -> Result<Self, ZkGpuError> {
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

        // Flatten + upload constants.
        let mut external_flat = Vec::with_capacity(expected_external_rows * W);
        for row in &params.external_constants {
            external_flat.extend_from_slice(row);
        }
        let external_constants_buf = device.upload::<BabyBear>(&external_flat)?;
        let internal_constants_buf = device.upload::<BabyBear>(&params.internal_constants)?;
        let internal_diagonal_buf = device.upload::<BabyBear>(&params.internal_diagonal)?;

        let params_uniform = create_uniform(
            device.raw_device(),
            &ParamsUniform {
                num_permutations: 0,
                rounds_f_half: params.rounds_f_half as u32,
                rounds_p: params.rounds_p as u32,
                row_stride: 0,
            },
            "babybear poseidon2 plonky3 params uniform",
        );

        let registry = device.pipeline_registry();
        let module = registry.get_or_create_module(
            device.raw_device(),
            wgsl_source,
            wgsl_label,
        );
        let bgl = registry.get_or_create_bgl(
            device.raw_device(),
            bgl_label,
            &[
                bgl_storage_entry(0, false), // state
                bgl_storage_entry(1, true),  // external_constants
                bgl_storage_entry(2, true),  // internal_constants
                bgl_storage_entry(3, true),  // internal_diagonal
                bgl_uniform_entry(4),        // params
            ],
        );
        let layout = device.raw_device().create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("BabyBear Poseidon2 Plonky3 pipeline layout"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            },
        );
        let pipeline = registry.get_or_create_pipeline(
            device.raw_device(),
            wgsl_source,
            entry_point,
            bgl_label,
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

    fn encode(
        &mut self,
        device: &WgpuDevice,
        buf: &WgpuBuffer<BabyBear>,
    ) -> Result<Option<wgpu::CommandEncoder>, ZkGpuError> {
        if buf.len() % W != 0 {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "Poseidon2 state buffer length {} is not a multiple of W={W}",
                buf.len(),
            )));
        }
        let num_permutations = (buf.len() / W) as u32;
        if num_permutations == 0 {
            return Ok(None);
        }

        let max_wg = device.caps.max_compute_workgroups_per_dimension;
        let dispatch: LinearDispatch =
            plan_linear_dispatch(num_permutations, WORKGROUP_SIZE, max_wg)?;
        let row_stride = dispatch.groups_per_row * WORKGROUP_SIZE;

        let uniform = ParamsUniform {
            num_permutations,
            rounds_f_half: self.rounds_f_half,
            rounds_p: self.rounds_p,
            row_stride,
        };
        device.raw_queue().write_buffer(
            &self.params_uniform,
            0,
            bytemuck::bytes_of(&uniform),
        );

        let bind_group = device.raw_device().create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: Some("babybear poseidon2 plonky3 bg"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.inner.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.external_constants_buf.inner.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.internal_constants_buf.inner.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.internal_diagonal_buf.inner.as_entire_binding(),
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
                label: Some("babybear poseidon2 plonky3 encoder"),
            },
        );
        {
            let mut pass = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor {
                    label: Some("babybear poseidon2 plonky3 pass"),
                    timestamp_writes: None,
                },
            );
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch.x, dispatch.y, 1);
        }
        Ok(Some(encoder))
    }

    fn execute_sync(
        &mut self,
        device: &WgpuDevice,
        buf: &mut WgpuBuffer<BabyBear>,
    ) -> Result<(), ZkGpuError> {
        let Some(encoder) = self.encode(device, buf)? else {
            return Ok(());
        };
        device.raw_queue().submit(Some(encoder.finish()));
        device
            .raw_device()
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| ZkGpuError::BackendError(Box::new(e)))?;
        Ok(())
    }

    async fn execute_async(
        &mut self,
        device: &WgpuDevice,
        buf: &mut WgpuBuffer<BabyBear>,
    ) -> Result<(), ZkGpuError> {
        let Some(encoder) = self.encode(device, buf)? else {
            return Ok(());
        };
        device.raw_queue().submit(Some(encoder.finish()));
        async_util::wait_for_submission(device.raw_device(), device.raw_queue()).await
    }
}

// --- BGL helpers ------------------------------------------------------------
//
// Duplicated from plan.rs — these are tiny wrappers and having them
// here removes the need to export them from plan.rs just for this
// sibling module.

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
