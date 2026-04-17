use wgpu::util::DeviceExt;
use zkgpu_core::{NttDirection, ZkGpuError};

use crate::device::WgpuDevice;
use crate::dispatch::plan_linear_dispatch;

use super::{FourStepPlan, TransposeVariant};
use super::params::{
    build_batched_r2_stage_params, build_batched_r4_stage_params,
    build_batched_r8_stage_params,
};

use super::super::common::{bgl_storage_entry, bgl_uniform_entry};
use super::super::planner::{FourStepPlanConfig, WORKGROUP_SIZE};
use super::super::twiddles::{
    precompute_fourstep_twiddles, precompute_single_r2_twiddles,
    precompute_stockham_r4_twiddles, precompute_stockham_r8_twiddles,
};

const FOURSTEP_LEAF_R2_SOURCE: &str =
    include_str!("../../kernels/portable/babybear_fourstep_leaf_r2.wgsl");
const FOURSTEP_LEAF_R4_SOURCE: &str =
    include_str!("../../kernels/portable/babybear_fourstep_leaf_r4.wgsl");
const FOURSTEP_LEAF_R8_SOURCE: &str =
    include_str!("../../kernels/portable/babybear_fourstep_leaf_r8.wgsl");
const FOURSTEP_TWIDDLE_SOURCE: &str =
    include_str!("../../kernels/portable/babybear_fourstep_twiddle.wgsl");
const FOURSTEP_TRANSPOSE_SOURCE_TILE16: &str =
    include_str!("../../kernels/portable/babybear_fourstep_transpose.wgsl");
const FOURSTEP_TRANSPOSE_SOURCE_TILE32: &str =
    include_str!("../../kernels/portable/babybear_fourstep_transpose_tiled32.wgsl");
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

        // NVIDIA scale-up Tier 3 Option A (T3.A, 2026-04-17): R8 leaf
        // pipeline. Shares the leaf BGL because the R8 kernel uses the
        // same 5 bindings (src, dst, twiddles, params, twiddles_prime) as
        // the R4 kernel. R8 stages consume 3 logical Stockham stages per
        // dispatch, halving memory round-trips vs R4's 2 stages/dispatch.
        let leaf_r8_module = reg.get_or_create_module(
            &device.device,
            FOURSTEP_LEAF_R8_SOURCE,
            "Four-step batched R8 leaf shader",
        );
        let leaf_r8_pipeline = reg.get_or_create_pipeline(
            &device.device,
            FOURSTEP_LEAF_R8_SOURCE,
            "batched_stockham_r8_butterfly",
            LEAF_BGL_LABEL,
            &leaf_pipeline_layout,
            &leaf_r8_module,
            device.pipeline_cache.as_ref(),
        );

        // --- Twiddle multiply pipeline ---
        //
        // NVIDIA scale-up Tier 2A Option A (2026-04-16): dropped the
        // binding-3 `twiddle_prime` storage buffer. The diagonal twiddle
        // pass now uses `mod_mul` (10-iteration reducer) instead of
        // `mod_mul_shoup`. Rationale: at log 22 the 16 MiB prime buffer
        // was half of the twiddle working set that caused partial-fit L2
        // cache-thrashing on RTX 4090. See
        // `research/benchmarks/nvidia-scale-up-2026-04-16/tier-2a-
        // log22-cliff-investigation.md` §Option A.
        let twiddle_bgl = reg.get_or_create_bgl(
            &device.device,
            TWIDDLE_BGL_LABEL,
            &[
                bgl_storage_entry(0, false),
                bgl_storage_entry(1, true),
                bgl_uniform_entry(2),
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

        let transpose_variant = TransposeVariant::from_env();
        let (transpose_source, transpose_label) = match transpose_variant {
            TransposeVariant::Tile16 => (
                FOURSTEP_TRANSPOSE_SOURCE_TILE16,
                "Four-step transpose shader (tile16)",
            ),
            TransposeVariant::Tile32 => (
                FOURSTEP_TRANSPOSE_SOURCE_TILE32,
                "Four-step transpose shader (tile32)",
            ),
        };
        let transpose_module =
            reg.get_or_create_module(&device.device, transpose_source, transpose_label);
        let transpose_pipeline = reg.get_or_create_pipeline(
            &device.device,
            transpose_source,
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
        // NVIDIA scale-up Tier 2B Option A (2026-04-16): the Phase-7
        // scale-by-1/N pass is now unconditionally skipped because the
        // `1/N` normalization is folded into the Phase-3 diagonal
        // twiddle table by `precompute_fourstep_twiddles`. Saves
        // ~193 µs per inverse call at log 22. `scale_param_buffer`
        // is therefore always `None`; `encode.rs`'s Phase-7 block
        // guards on `scale_param_buffer.is_some()` so Phase 7 simply
        // doesn't fire. `scale_dispatch`, `scale_pipeline`, and
        // `scale_bgl` are still plumbed through the struct so a
        // future revert (e.g. for numerical-stability reasons) is
        // a one-line change.
        let scale_param_buffer: Option<wgpu::Buffer> = None;

        // NVIDIA scale-up Tier 1 Fix 2b (2026-04-16): 2D-folded leaf
        // dispatch grids. Phase 2 R4 + Phase 5 R4 each cover `n/4`
        // butterflies; Phase 2 R2 + Phase 5 R2 each cover `n/2`.
        // Constant across all stages within a phase, and across
        // phases (Phase 2 and Phase 5 have the same dispatch shape).
        let max_dim = device.caps.max_compute_workgroups_per_dimension;
        // T3.A (2026-04-17): R8 leaf dispatches cover n/8 butterflies.
        let leaf_r8_dispatch = plan_linear_dispatch(config.n / 8, WORKGROUP_SIZE, max_dim)?;
        let leaf_r4_dispatch = plan_linear_dispatch(config.n / 4, WORKGROUP_SIZE, max_dim)?;
        let leaf_r2_dispatch = plan_linear_dispatch(config.n / 2, WORKGROUP_SIZE, max_dim)?;

        // --- Precompute diagonal twiddle table in C×R layout ---
        // After Phase 1 transpose, data is C×R. Twiddle at position (c, k_r)
        // is omega_N^(k_r * c), stored at flat index c*R + k_r.
        // NVIDIA scale-up Tier 2A Option A (2026-04-16): we still call
        // `precompute_fourstep_twiddles` for the `twiddle` buffer but
        // drop the `fourstep_twiddles_prime` allocation. The shader no
        // longer uses Shoup reduction for the diagonal twiddle pass;
        // shrinking this buffer from 2 × 16 MiB → 16 MiB at log 22
        // moves the working set back under RTX 4090 L2 capacity.
        let (fourstep_twiddles, _fourstep_twiddles_prime) =
            precompute_fourstep_twiddles(&config, direction);
        let twiddle_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Four-step diagonal twiddles"),
                contents: bytemuck::cast_slice(&fourstep_twiddles),
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
        //
        // T3.A (2026-04-17): R8 twiddles + stage params come first in the
        // leaf chain (smallest `s`), then R4, then R2 residue. The chain's
        // parity of dispatches determines src/dst ping-pong in encode.rs.
        let (
            phase2_r8_tw,
            phase2_r8_tw_prime,
            phase2_omega8,
            phase2_omega8_prime,
            phase2_omega4_from_r8,
            phase2_omega4_prime_from_r8,
            phase2_omega8_cubed,
            phase2_omega8_cubed_prime,
        ) = precompute_stockham_r8_twiddles(
            config.row_log_n,
            direction,
            &config.col_leaf.r8_twiddle_spec,
        );
        let phase2_r8_twiddle_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Four-step phase2 R8 twiddles"),
                    contents: bytemuck::cast_slice(if phase2_r8_tw.is_empty() {
                        &[0u32]
                    } else {
                        &phase2_r8_tw
                    }),
                    usage: wgpu::BufferUsages::STORAGE,
                });
        let phase2_r8_twiddle_prime_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Four-step phase2 R8 twiddles prime"),
                    contents: bytemuck::cast_slice(if phase2_r8_tw_prime.is_empty() {
                        &[0u32]
                    } else {
                        &phase2_r8_tw_prime
                    }),
                    usage: wgpu::BufferUsages::STORAGE,
                });
        let phase2_r8_stage_param_buffers = build_batched_r8_stage_params(
            &device.device,
            &config.col_leaf,
            config.cols,
            phase2_omega8,
            phase2_omega8_prime,
            phase2_omega4_from_r8,
            phase2_omega4_prime_from_r8,
            phase2_omega8_cubed,
            phase2_omega8_cubed_prime,
            leaf_r8_dispatch.groups_per_row,
        );

        // R4 twiddles (after R8 in the leaf stage chain)
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
            leaf_r4_dispatch.groups_per_row,
        );

        // R2 remainder twiddles — T3.A (2026-04-17): the R2 residue stage now
        // starts after the R8 stages plus the R4 stages. Recover `h` from the
        // R2 stage's `s` (= 2^h) stored in global_stage_params.
        let (phase2_r2_tw, phase2_r2_tw_prime) =
            if let Some(sp) = config.col_leaf.global_stage_params.first() {
                let h = sp.s.trailing_zeros();
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
        let phase2_r2_stage_param_buffers = build_batched_r2_stage_params(
            &device.device,
            &config.col_leaf,
            config.cols,
            leaf_r2_dispatch.groups_per_row,
        );

        // --- Phase 5: C-point NTTs (row_leaf), R batches ---
        //
        // T3.A (2026-04-17): R8 + R4 + R2 leaf chain, same pattern as Phase 2.
        let (
            phase5_r8_tw,
            phase5_r8_tw_prime,
            phase5_omega8,
            phase5_omega8_prime,
            phase5_omega4_from_r8,
            phase5_omega4_prime_from_r8,
            phase5_omega8_cubed,
            phase5_omega8_cubed_prime,
        ) = precompute_stockham_r8_twiddles(
            config.col_log_n,
            direction,
            &config.row_leaf.r8_twiddle_spec,
        );
        let phase5_r8_twiddle_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Four-step phase5 R8 twiddles"),
                    contents: bytemuck::cast_slice(if phase5_r8_tw.is_empty() {
                        &[0u32]
                    } else {
                        &phase5_r8_tw
                    }),
                    usage: wgpu::BufferUsages::STORAGE,
                });
        let phase5_r8_twiddle_prime_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Four-step phase5 R8 twiddles prime"),
                    contents: bytemuck::cast_slice(if phase5_r8_tw_prime.is_empty() {
                        &[0u32]
                    } else {
                        &phase5_r8_tw_prime
                    }),
                    usage: wgpu::BufferUsages::STORAGE,
                });
        let phase5_r8_stage_param_buffers = build_batched_r8_stage_params(
            &device.device,
            &config.row_leaf,
            config.rows,
            phase5_omega8,
            phase5_omega8_prime,
            phase5_omega4_from_r8,
            phase5_omega4_prime_from_r8,
            phase5_omega8_cubed,
            phase5_omega8_cubed_prime,
            leaf_r8_dispatch.groups_per_row,
        );

        // R4 twiddles (after R8 in the leaf stage chain)
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
            leaf_r4_dispatch.groups_per_row,
        );

        // R2 remainder twiddles — T3.A (2026-04-17): recover `h` from the R2
        // stage's `s` (= 2^h) stored in global_stage_params (now starts after
        // both R8 and R4 stages).
        let (phase5_r2_tw, phase5_r2_tw_prime) =
            if let Some(sp) = config.row_leaf.global_stage_params.first() {
                let h = sp.s.trailing_zeros();
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
        let phase5_r2_stage_param_buffers = build_batched_r2_stage_params(
            &device.device,
            &config.row_leaf,
            config.rows,
            leaf_r2_dispatch.groups_per_row,
        );

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
            leaf_r8_pipeline,
            leaf_bgl,
            twiddle_pipeline,
            twiddle_bgl,
            twiddle_buffer,
            twiddle_param_buffer,
            twiddle_dispatch,
            transpose_pipeline,
            transpose_bgl,
            transpose_variant,
            transpose_rc_to_cr_params,
            transpose_cr_to_rc_params,
            scale_pipeline,
            scale_bgl,
            scale_param_buffer,
            scale_dispatch,
            phase2_r8_twiddle_buffer,
            phase2_r8_twiddle_prime_buffer,
            phase2_r8_stage_param_buffers,
            phase2_r4_twiddle_buffer,
            phase2_r4_twiddle_prime_buffer,
            phase2_r4_stage_param_buffers,
            phase2_r2_twiddle_buffer,
            phase2_r2_twiddle_prime_buffer,
            phase2_r2_stage_param_buffers,
            phase5_r8_twiddle_buffer,
            phase5_r8_twiddle_prime_buffer,
            phase5_r8_stage_param_buffers,
            phase5_r4_twiddle_buffer,
            phase5_r4_twiddle_prime_buffer,
            phase5_r4_stage_param_buffers,
            phase5_r2_twiddle_buffer,
            phase5_r2_twiddle_prime_buffer,
            phase5_r2_stage_param_buffers,
            leaf_r8_dispatch,
            leaf_r4_dispatch,
            leaf_r2_dispatch,
            scratch_buffer,
            transpose_scratch_buffer,
            config,
        })
    }
}
