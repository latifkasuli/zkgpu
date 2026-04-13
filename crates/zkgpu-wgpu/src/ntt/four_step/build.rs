use wgpu::util::DeviceExt;
use zkgpu_babybear::BabyBear;
use zkgpu_core::{GpuField, NttDirection, ZkGpuError};

use crate::device::WgpuDevice;
use crate::dispatch::plan_linear_dispatch;

use super::FourStepPlan;
use super::params::{build_batched_r2_stage_params, build_batched_r4_stage_params};

use super::super::common::{bgl_storage_entry, bgl_uniform_entry};
use super::super::planner::{FourStepPlanConfig, WORKGROUP_SIZE};
use super::super::twiddles::{
    precompute_fourstep_twiddles, precompute_single_r2_twiddles,
    precompute_stockham_r4_twiddles,
};

const FOURSTEP_LEAF_R2_SOURCE: &str =
    include_str!("../../kernels/portable/babybear_fourstep_leaf_r2.wgsl");
const FOURSTEP_LEAF_R4_SOURCE: &str =
    include_str!("../../kernels/portable/babybear_fourstep_leaf_r4.wgsl");
const FOURSTEP_TWIDDLE_SOURCE: &str =
    include_str!("../../kernels/portable/babybear_fourstep_twiddle.wgsl");
const FOURSTEP_TRANSPOSE_SOURCE: &str =
    include_str!("../../kernels/portable/babybear_fourstep_transpose.wgsl");
const SCALE_SOURCE: &str =
    include_str!("../../kernels/portable/babybear_scale.wgsl");

const LEAF_BGL_LABEL: &str = "Four-step leaf BGL";
const TWIDDLE_BGL_LABEL: &str = "Four-step twiddle BGL";
const TRANSPOSE_BGL_LABEL: &str = "Four-step transpose BGL";
const SCALE_BGL_LABEL: &str = "Four-step scale BGL";

impl FourStepPlan {
    pub(crate) fn new(
        device: &WgpuDevice,
        config: FourStepPlanConfig,
        direction: NttDirection,
    ) -> Result<Self, ZkGpuError> {
        #[cfg(not(target_arch = "wasm32"))]
        let validation_scope = device.push_validation_scope();

        let reg = &device.pipelines;

        // --- Leaf bind group layout (shared between R2 and R4 leaf pipelines) ---
        let leaf_bgl = reg.get_or_create_bgl(
            &device.device,
            LEAF_BGL_LABEL,
            &[
                bgl_storage_entry(0, true),
                bgl_storage_entry(1, false),
                bgl_storage_entry(2, true),
                bgl_uniform_entry(3),
                bgl_storage_entry(4, true), // twiddles_prime (read-only)
            ],
        );

        let leaf_pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Four-step leaf pipeline layout"),
                    bind_group_layouts: &[Some(&leaf_bgl)],
                    immediate_size: 0,
                });

        let leaf_global_module = reg.get_or_create_module(
            &device.device,
            FOURSTEP_LEAF_R2_SOURCE,
            "Four-step batched leaf shader",
        );
        let leaf_global_pipeline = reg.get_or_create_pipeline(
            &device.device,
            FOURSTEP_LEAF_R2_SOURCE,
            "batched_stockham_butterfly",
            LEAF_BGL_LABEL,
            &leaf_pipeline_layout,
            &leaf_global_module,
            device.pipeline_cache.as_ref(),
        );

        let leaf_r4_module = reg.get_or_create_module(
            &device.device,
            FOURSTEP_LEAF_R4_SOURCE,
            "Four-step batched R4 leaf shader",
        );
        let leaf_r4_pipeline = reg.get_or_create_pipeline(
            &device.device,
            FOURSTEP_LEAF_R4_SOURCE,
            "batched_stockham_r4_butterfly",
            LEAF_BGL_LABEL,
            &leaf_pipeline_layout,
            &leaf_r4_module,
            device.pipeline_cache.as_ref(),
        );

        // --- Twiddle multiply pipeline ---
        let twiddle_bgl = reg.get_or_create_bgl(
            &device.device,
            TWIDDLE_BGL_LABEL,
            &[
                bgl_storage_entry(0, false),
                bgl_storage_entry(1, true),
                bgl_uniform_entry(2),
                bgl_storage_entry(3, true), // twiddle_prime (read-only)
            ],
        );

        let twiddle_pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Four-step twiddle pipeline layout"),
                    bind_group_layouts: &[Some(&twiddle_bgl)],
                    immediate_size: 0,
                });

        let twiddle_module = reg.get_or_create_module(
            &device.device,
            FOURSTEP_TWIDDLE_SOURCE,
            "Four-step twiddle shader",
        );
        let twiddle_pipeline = reg.get_or_create_pipeline(
            &device.device,
            FOURSTEP_TWIDDLE_SOURCE,
            "fourstep_twiddle",
            TWIDDLE_BGL_LABEL,
            &twiddle_pipeline_layout,
            &twiddle_module,
            device.pipeline_cache.as_ref(),
        );

        // --- Transpose pipeline ---
        let transpose_bgl = reg.get_or_create_bgl(
            &device.device,
            TRANSPOSE_BGL_LABEL,
            &[
                bgl_storage_entry(0, true),
                bgl_storage_entry(1, false),
                bgl_uniform_entry(2),
            ],
        );

        let transpose_pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Four-step transpose pipeline layout"),
                    bind_group_layouts: &[Some(&transpose_bgl)],
                    immediate_size: 0,
                });

        let transpose_module = reg.get_or_create_module(
            &device.device,
            FOURSTEP_TRANSPOSE_SOURCE,
            "Four-step transpose shader",
        );
        let transpose_pipeline = reg.get_or_create_pipeline(
            &device.device,
            FOURSTEP_TRANSPOSE_SOURCE,
            "transpose_tiles",
            TRANSPOSE_BGL_LABEL,
            &transpose_pipeline_layout,
            &transpose_module,
            device.pipeline_cache.as_ref(),
        );

        // --- Scale pipeline ---
        let scale_bgl = reg.get_or_create_bgl(
            &device.device,
            SCALE_BGL_LABEL,
            &[bgl_storage_entry(0, false), bgl_uniform_entry(1)],
        );

        let scale_pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Four-step scale pipeline layout"),
                    bind_group_layouts: &[Some(&scale_bgl)],
                    immediate_size: 0,
                });

        let scale_module = reg.get_or_create_module(
            &device.device,
            SCALE_SOURCE,
            "Four-step scale shader",
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
        device.pop_validation_scope(validation_scope, "four-step pipeline creation")?;

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
                        label: Some("Four-step inverse scale params"),
                        contents: bytemuck::cast_slice(&params),
                        usage: wgpu::BufferUsages::UNIFORM,
                    }),
            )
        } else {
            None
        };

        // --- Precompute diagonal twiddle table in C×R layout ---
        // After Phase 1 transpose, data is C×R. Twiddle at position (c, k_r)
        // is omega_N^(k_r * c), stored at flat index c*R + k_r.
        let (fourstep_twiddles, fourstep_twiddles_prime) =
            precompute_fourstep_twiddles(&config, direction);
        let twiddle_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Four-step diagonal twiddles"),
                contents: bytemuck::cast_slice(&fourstep_twiddles),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let twiddle_prime_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Four-step diagonal twiddles prime"),
                contents: bytemuck::cast_slice(&fourstep_twiddles_prime),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let twiddle_dispatch = plan_linear_dispatch(
            config.n.div_ceil(4),
            WORKGROUP_SIZE,
            device.caps.max_compute_workgroups_per_dimension,
        )?;
        let twiddle_params = [config.n, twiddle_dispatch.groups_per_row, 0u32, 0u32];
        let twiddle_param_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Four-step twiddle params"),
                    contents: bytemuck::cast_slice(&twiddle_params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        // --- Phase 2: R-point NTTs (col_leaf), C batches ---
        // R4 twiddles
        let (phase2_r4_tw, phase2_r4_tw_prime, phase2_omega4, phase2_omega4_prime) =
            precompute_stockham_r4_twiddles(
                config.row_log_n,
                direction,
                &config.col_leaf.r4_twiddle_spec,
            );
        let phase2_r4_twiddle_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Four-step phase2 R4 twiddles"),
                    contents: bytemuck::cast_slice(if phase2_r4_tw.is_empty() {
                        &[0u32]
                    } else {
                        &phase2_r4_tw
                    }),
                    usage: wgpu::BufferUsages::STORAGE,
                });
        let phase2_r4_twiddle_prime_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Four-step phase2 R4 twiddles prime"),
                    contents: bytemuck::cast_slice(if phase2_r4_tw_prime.is_empty() {
                        &[0u32]
                    } else {
                        &phase2_r4_tw_prime
                    }),
                    usage: wgpu::BufferUsages::STORAGE,
                });
        let phase2_r4_stage_param_buffers = build_batched_r4_stage_params(
            &device.device,
            &config.col_leaf,
            config.cols,
            phase2_omega4,
            phase2_omega4_prime,
        );

        // R2 remainder twiddles
        let (phase2_r2_tw, phase2_r2_tw_prime) =
            if !config.col_leaf.global_stage_params.is_empty() {
                let h = config.col_leaf.r4_stage_params.len() as u32 * 2;
                precompute_single_r2_twiddles(config.row_log_n, direction, h)
            } else {
                (Vec::new(), Vec::new())
            };
        let phase2_r2_twiddle_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Four-step phase2 R2 twiddles"),
                    contents: bytemuck::cast_slice(if phase2_r2_tw.is_empty() {
                        &[0u32]
                    } else {
                        &phase2_r2_tw
                    }),
                    usage: wgpu::BufferUsages::STORAGE,
                });
        let phase2_r2_twiddle_prime_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Four-step phase2 R2 twiddles prime"),
                    contents: bytemuck::cast_slice(if phase2_r2_tw_prime.is_empty() {
                        &[0u32]
                    } else {
                        &phase2_r2_tw_prime
                    }),
                    usage: wgpu::BufferUsages::STORAGE,
                });
        let phase2_r2_stage_param_buffers =
            build_batched_r2_stage_params(&device.device, &config.col_leaf, config.cols);

        // --- Phase 5: C-point NTTs (row_leaf), R batches ---
        // R4 twiddles
        let (phase5_r4_tw, phase5_r4_tw_prime, phase5_omega4, phase5_omega4_prime) =
            precompute_stockham_r4_twiddles(
                config.col_log_n,
                direction,
                &config.row_leaf.r4_twiddle_spec,
            );
        let phase5_r4_twiddle_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Four-step phase5 R4 twiddles"),
                    contents: bytemuck::cast_slice(if phase5_r4_tw.is_empty() {
                        &[0u32]
                    } else {
                        &phase5_r4_tw
                    }),
                    usage: wgpu::BufferUsages::STORAGE,
                });
        let phase5_r4_twiddle_prime_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Four-step phase5 R4 twiddles prime"),
                    contents: bytemuck::cast_slice(if phase5_r4_tw_prime.is_empty() {
                        &[0u32]
                    } else {
                        &phase5_r4_tw_prime
                    }),
                    usage: wgpu::BufferUsages::STORAGE,
                });
        let phase5_r4_stage_param_buffers = build_batched_r4_stage_params(
            &device.device,
            &config.row_leaf,
            config.rows,
            phase5_omega4,
            phase5_omega4_prime,
        );

        // R2 remainder twiddles
        let (phase5_r2_tw, phase5_r2_tw_prime) =
            if !config.row_leaf.global_stage_params.is_empty() {
                let h = config.row_leaf.r4_stage_params.len() as u32 * 2;
                precompute_single_r2_twiddles(config.col_log_n, direction, h)
            } else {
                (Vec::new(), Vec::new())
            };
        let phase5_r2_twiddle_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Four-step phase5 R2 twiddles"),
                    contents: bytemuck::cast_slice(if phase5_r2_tw.is_empty() {
                        &[0u32]
                    } else {
                        &phase5_r2_tw
                    }),
                    usage: wgpu::BufferUsages::STORAGE,
                });
        let phase5_r2_twiddle_prime_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Four-step phase5 R2 twiddles prime"),
                    contents: bytemuck::cast_slice(if phase5_r2_tw_prime.is_empty() {
                        &[0u32]
                    } else {
                        &phase5_r2_tw_prime
                    }),
                    usage: wgpu::BufferUsages::STORAGE,
                });
        let phase5_r2_stage_param_buffers =
            build_batched_r2_stage_params(&device.device, &config.row_leaf, config.rows);

        // --- Scratch buffers ---
        let buf_size = (config.n as u64) * std::mem::size_of::<u32>() as u64;
        let scratch_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Four-step NTT scratch"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let transpose_scratch_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Four-step transpose scratch"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Transpose param buffers ---
        // R×C → C×R: params = (rows, cols)
        let rc_to_cr = [config.rows, config.cols, 0u32, 0u32];
        let transpose_rc_to_cr_params =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Four-step transpose R\u{d7}C\u{2192}C\u{d7}R params"),
                    contents: bytemuck::cast_slice(&rc_to_cr),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
        // C×R → R×C: params = (cols, rows)
        let cr_to_rc = [config.cols, config.rows, 0u32, 0u32];
        let transpose_cr_to_rc_params =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Four-step transpose C\u{d7}R\u{2192}R\u{d7}C params"),
                    contents: bytemuck::cast_slice(&cr_to_rc),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        Ok(Self {
            leaf_global_pipeline,
            leaf_r4_pipeline,
            leaf_bgl,
            twiddle_pipeline,
            twiddle_bgl,
            twiddle_buffer,
            twiddle_prime_buffer,
            twiddle_param_buffer,
            twiddle_dispatch,
            transpose_pipeline,
            transpose_bgl,
            transpose_rc_to_cr_params,
            transpose_cr_to_rc_params,
            scale_pipeline,
            scale_bgl,
            scale_param_buffer,
            scale_dispatch,
            phase2_r4_twiddle_buffer,
            phase2_r4_twiddle_prime_buffer,
            phase2_r4_stage_param_buffers,
            phase2_r2_twiddle_buffer,
            phase2_r2_twiddle_prime_buffer,
            phase2_r2_stage_param_buffers,
            phase5_r4_twiddle_buffer,
            phase5_r4_twiddle_prime_buffer,
            phase5_r4_stage_param_buffers,
            phase5_r2_twiddle_buffer,
            phase5_r2_twiddle_prime_buffer,
            phase5_r2_stage_param_buffers,
            scratch_buffer,
            transpose_scratch_buffer,
            config,
        })
    }
}
