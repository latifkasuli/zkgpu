use wgpu::util::DeviceExt;
use zkgpu_babybear::BabyBear;
use zkgpu_core::{GpuField, NttDirection, ZkGpuError};

use crate::device::WgpuDevice;
use crate::dispatch::plan_linear_dispatch;

use super::StockhamPlan;
use super::super::common::{bgl_storage_entry, bgl_uniform_entry};
use super::super::local_kernel::{resolve_local_kernel, ResolvedLocalKernel};
use super::super::planner::{LocalKernelHint, StockhamPlanConfig, WORKGROUP_SIZE};
use super::super::twiddles::{
    precompute_local_r4_twiddles, precompute_stockham_r4_twiddles,
    precompute_subgroup_local_twiddles, shoup_quotient,
};

const STOCKHAM_R2_SOURCE: &str =
    include_str!("../../kernels/portable/babybear_stockham_r2.wgsl");
const STOCKHAM_R4_SOURCE: &str =
    include_str!("../../kernels/portable/babybear_stockham_r4.wgsl");
const STOCKHAM_LOCAL_R4_SOURCE: &str =
    include_str!("../../kernels/portable/babybear_stockham_local_r4.wgsl");
const SCALE_SOURCE: &str =
    include_str!("../../kernels/portable/babybear_scale.wgsl");

/// Pre-compiled SPIR-V for the subgroup-accelerated DIT local kernel.
/// Only included when the `subgroup-vulkan-spirv` feature is active.
#[cfg(feature = "subgroup-vulkan-spirv")]
const STOCKHAM_LOCAL_SUBGROUP_SPIRV: &[u8] =
    include_bytes!("../../kernels/native/babybear_stockham_local_subgroup.spv");

const NTT_BGL_LABEL: &str = "Stockham NTT bind group layout";
const SCALE_BGL_LABEL: &str = "Scale bind group layout";

impl StockhamPlan {
    /// Create a Stockham NTT plan from a pre-validated config.
    pub(crate) fn new(
        device: &WgpuDevice,
        config: StockhamPlanConfig,
        direction: NttDirection,
        local_hint: LocalKernelHint,
    ) -> Result<Self, ZkGpuError> {
        let log_n = config.log_n;

        #[cfg(not(target_arch = "wasm32"))]
        let validation_scope = device.push_validation_scope();

        let reg = &device.pipelines;

        // --- NTT bind group layout (shared between global and local pipelines) ---
        let ntt_bind_group_layout = reg.get_or_create_bgl(
            &device.device,
            NTT_BGL_LABEL,
            &[
                bgl_storage_entry(0, true),  // src (read-only)
                bgl_storage_entry(1, false), // dst (read-write)
                bgl_storage_entry(2, true),  // twiddles (read-only)
                bgl_uniform_entry(3),
                bgl_storage_entry(4, true),  // twiddles_prime (read-only)
            ],
        );

        let ntt_pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Stockham NTT pipeline layout"),
                    bind_group_layouts: &[Some(&ntt_bind_group_layout)],
                    immediate_size: 0,
                });

        // --- Global R2 pipeline ---
        let global_module = reg.get_or_create_module(
            &device.device,
            STOCKHAM_R2_SOURCE,
            "Stockham global R2 shader",
        );
        let global_pipeline = reg.get_or_create_pipeline(
            &device.device,
            STOCKHAM_R2_SOURCE,
            "stockham_butterfly",
            NTT_BGL_LABEL,
            &ntt_pipeline_layout,
            &global_module,
            device.pipeline_cache.as_ref(),
        );

        // --- Radix-4 global pipeline ---
        let r4_module = reg.get_or_create_module(
            &device.device,
            STOCKHAM_R4_SOURCE,
            "Stockham R4 shader",
        );
        let r4_pipeline = reg.get_or_create_pipeline(
            &device.device,
            STOCKHAM_R4_SOURCE,
            "stockham_r4_butterfly",
            NTT_BGL_LABEL,
            &ntt_pipeline_layout,
            &r4_module,
            device.pipeline_cache.as_ref(),
        );

        // --- Local pipeline (subgroup-accelerated DIT or portable R4 DIF) ---
        //
        // The resolver checks cargo features, backend, subgroup caps, and
        // the experimental env flag. ForceSubgroup errors if unsatisfiable;
        // Auto silently falls back to PortableR4.
        let (resolved_local, _reason) = resolve_local_kernel(local_hint, &device.caps)?;
        let use_subgroup_local = resolved_local == ResolvedLocalKernel::SubgroupSpirV;

        let local_pipeline = if use_subgroup_local {
            #[cfg(feature = "subgroup-vulkan-spirv")]
            {
                let local_module = reg.get_or_create_module_spirv(
                    &device.device,
                    STOCKHAM_LOCAL_SUBGROUP_SPIRV,
                    "Stockham local subgroup shader (SPIR-V)",
                );
                reg.get_or_create_pipeline_keyed(
                    &device.device,
                    STOCKHAM_LOCAL_SUBGROUP_SPIRV.as_ptr() as usize,
                    "main",
                    NTT_BGL_LABEL,
                    &ntt_pipeline_layout,
                    &local_module,
                    device.pipeline_cache.as_ref(),
                )
            }
            #[cfg(not(feature = "subgroup-vulkan-spirv"))]
            unreachable!("SubgroupSpirV requires the subgroup-vulkan-spirv cargo feature")
        } else {
            let local_module = reg.get_or_create_module(
                &device.device,
                STOCKHAM_LOCAL_R4_SOURCE,
                "Stockham local R4 shader",
            );
            reg.get_or_create_pipeline(
                &device.device,
                STOCKHAM_LOCAL_R4_SOURCE,
                "stockham_local_r4",
                NTT_BGL_LABEL,
                &ntt_pipeline_layout,
                &local_module,
                device.pipeline_cache.as_ref(),
            )
        };

        // --- Scale pipeline (in-place element-wise multiply) ---
        let scale_bind_group_layout = reg.get_or_create_bgl(
            &device.device,
            SCALE_BGL_LABEL,
            &[
                bgl_storage_entry(0, false), // data (read-write)
                bgl_uniform_entry(1),
            ],
        );

        let scale_pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Scale pipeline layout"),
                    bind_group_layouts: &[Some(&scale_bind_group_layout)],
                    immediate_size: 0,
                });

        let scale_module = reg.get_or_create_module(
            &device.device,
            SCALE_SOURCE,
            "Scale shader",
        );
        let scale_pipeline = reg.get_or_create_pipeline(
            &device.device,
            SCALE_SOURCE,
            "scale_elements",
            SCALE_BGL_LABEL,
            &scale_pipeline_layout,
            &scale_module,
            device.pipeline_cache.as_ref(),
        );
        #[cfg(not(target_arch = "wasm32"))]
        device.pop_validation_scope(validation_scope, "stockham pipeline creation")?;

        let scale_dispatch = plan_linear_dispatch(
            config.n.div_ceil(4),
            WORKGROUP_SIZE,
            device.caps.max_compute_workgroups_per_dimension,
        )?;

        let scale_param_buffer = if direction == NttDirection::Inverse {
            let n_field = BabyBear::new(config.n);
            let n_inv = n_field.inv().expect("n must be invertible in BabyBear");
            let params = [
                config.n,
                n_inv.to_repr(),
                scale_dispatch.groups_per_row,
                0u32,
            ];
            Some(
                device
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Inverse scale params"),
                        contents: bytemuck::cast_slice(&params),
                        usage: wgpu::BufferUsages::UNIFORM,
                    }),
            )
        } else {
            None
        };

        // --- Global twiddles and per-stage param buffers ---
        // R2 remainder stages (if any)
        let r2_start_h = config.r4_stage_params.len() as u32 * 2;
        let (r2_twiddles, r2_twiddles_prime) = if !config.global_stage_params.is_empty() {
            let n_val = config.n;
            let omega = BabyBear::root_of_unity(log_n);
            let omega = match direction {
                NttDirection::Forward => omega,
                NttDirection::Inverse => omega.inv().expect("root invertible"),
            };
            let mut tw = Vec::new();
            let mut tw_prime = Vec::new();
            for (i, _sp) in config.global_stage_params.iter().enumerate() {
                let h = r2_start_h + i as u32;
                let s = 1u32 << h;
                let m = n_val >> (h + 1);
                let step = omega.pow(s as u64);
                let mut w = BabyBear::ONE;
                for _ in 0..m {
                    let repr = w.to_repr();
                    tw.push(repr);
                    tw_prime.push(shoup_quotient(repr));
                    w = w * step;
                }
            }
            (tw, tw_prime)
        } else {
            (vec![0u32], vec![0u32])
        };
        let global_twiddle_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Stockham R2 twiddles"),
                    contents: bytemuck::cast_slice(&r2_twiddles),
                    usage: wgpu::BufferUsages::STORAGE,
                });
        let global_twiddle_prime_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Stockham R2 twiddles prime"),
                    contents: bytemuck::cast_slice(&r2_twiddles_prime),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let mut global_stage_param_buffers = Vec::with_capacity(config.global_stage_params.len());
        for sp in &config.global_stage_params {
            let params = [sp.n, sp.s, sp.m, sp.twiddle_offset];
            let buf = device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Stockham R2 stage params"),
                    contents: bytemuck::cast_slice(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            global_stage_param_buffers.push(buf);
        }

        // --- R4 twiddles and per-stage param buffers ---
        let (r4_twiddles_raw, r4_twiddles_prime, omega4, omega4_prime) =
            if !config.r4_stage_params.is_empty() {
                precompute_stockham_r4_twiddles(log_n, direction, &config.r4_twiddle_spec)
            } else {
                (vec![0u32], vec![0u32], 0u32, 0u32)
            };
        let r4_twiddle_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Stockham R4 twiddles"),
                    contents: bytemuck::cast_slice(&r4_twiddles_raw),
                    usage: wgpu::BufferUsages::STORAGE,
                });
        let r4_twiddle_prime_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Stockham R4 twiddles prime"),
                    contents: bytemuck::cast_slice(&r4_twiddles_prime),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let mut r4_stage_param_buffers = Vec::with_capacity(config.r4_stage_params.len());
        for sp in &config.r4_stage_params {
            let params = [sp.n, sp.s, sp.m4, sp.twiddle_offset, omega4, omega4_prime, 0u32, 0u32];
            let buf = device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Stockham R4 stage params"),
                    contents: bytemuck::cast_slice(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            r4_stage_param_buffers.push(buf);
        }

        // --- Local twiddles and params (DIT for subgroup, DIF for R4) ---
        let (local_twiddles, local_twiddles_prime, local_omega4, local_omega4_prime) =
            if use_subgroup_local {
                precompute_subgroup_local_twiddles(direction)
            } else {
                precompute_local_r4_twiddles(direction)
            };
        let local_label = if use_subgroup_local { "subgroup" } else { "R4" };
        let local_twiddle_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Stockham local {local_label} twiddles")),
                    contents: bytemuck::cast_slice(&local_twiddles),
                    usage: wgpu::BufferUsages::STORAGE,
                });
        let local_twiddle_prime_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Stockham local {local_label} twiddles prime")),
                    contents: bytemuck::cast_slice(&local_twiddles_prime),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let subgroup_log = if use_subgroup_local {
            device.caps.min_subgroup_size.trailing_zeros()
        } else {
            0
        };
        let local_params = [config.local_stride, local_omega4, local_omega4_prime, subgroup_log];
        let local_param_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Stockham local {local_label} params")),
                    contents: bytemuck::cast_slice(&local_params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        // --- Scratch buffer ---
        let scratch_size = (config.n as u64) * std::mem::size_of::<u32>() as u64;
        let scratch_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Stockham scratch buffer"),
            size: scratch_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            global_pipeline,
            r4_pipeline,
            local_pipeline,
            ntt_bind_group_layout,
            global_twiddle_buffer,
            global_twiddle_prime_buffer,
            global_stage_param_buffers,
            r4_twiddle_buffer,
            r4_twiddle_prime_buffer,
            r4_stage_param_buffers,
            local_twiddle_buffer,
            local_twiddle_prime_buffer,
            local_param_buffer,
            scratch_buffer,
            scale_pipeline,
            scale_bind_group_layout,
            scale_param_buffer,
            scale_dispatch,
            config,
        })
    }
}
