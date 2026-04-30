use std::sync::Arc;

use wgpu::util::DeviceExt;
use zkgpu_babybear::BabyBear;
use zkgpu_core::{GpuField, NttDirection, ZkGpuError};

use crate::device::WgpuDevice;
use crate::dispatch::plan_linear_dispatch;

use super::{R4ParamMode, R4ParamSource, StockhamPlan};
use super::super::common::{bgl_storage_entry, bgl_uniform_entry};
use super::super::planner::{StockhamPlanConfig, WORKGROUP_SIZE};
use super::super::babybear_twiddles::{
    precompute_local_r4_twiddles, precompute_stockham_r4_twiddles, shoup_quotient,
};

const STOCKHAM_R2_SOURCE: &str =
    include_str!("../../kernels/portable/babybear_stockham_r2.wgsl");
const STOCKHAM_R4_SOURCE: &str =
    include_str!("../../kernels/portable/babybear_stockham_r4.wgsl");
const STOCKHAM_R4_IMMEDIATE_SOURCE: &str =
    include_str!("../../kernels/portable/babybear_stockham_r4_immediate.wgsl");
const STOCKHAM_LOCAL_R4_SOURCE: &str =
    include_str!("../../kernels/portable/babybear_stockham_local_r4.wgsl");
const SCALE_SOURCE: &str =
    include_str!("../../kernels/portable/babybear_scale.wgsl");

const NTT_BGL_LABEL: &str = "Stockham NTT bind group layout";
const R4_IMMEDIATE_BGL_LABEL: &str = "Stockham R4 Immediate bind group layout";
const SCALE_BGL_LABEL: &str = "Scale bind group layout";

/// Item #3 (immediates) drift-prevention: the R4 immediate-mode param
/// block size in bytes. This single value is the source of truth for
/// BOTH `PipelineLayoutDescriptor::immediate_size` AND the cache-key
/// `PipelineSpec::immediate_size` — see the design note at the top of
/// `pipeline_registry.rs`. The two fields cannot drift because they're
/// both derived from this constant. The size matches the WGSL
/// `R4Params` struct in `babybear_stockham_r4_immediate.wgsl` (8 ×
/// u32 = 32 bytes).
const R4_IMMEDIATE_SIZE_BYTES: u32 = 32;

impl StockhamPlan {
    /// Create a Stockham NTT plan from a pre-validated config.
    ///
    /// `r4_param_mode` selects how the radix-4 stage params reach the
    /// kernel. `Storage` is the original path (per-stage uniform
    /// buffer); `Immediate` is the item #3 pilot path
    /// (`set_immediates`). The caller is responsible for picking a
    /// mode the device actually supports (see
    /// `device.caps.has_immediates`); attempting to build an
    /// `Immediate` plan on a device without `Features::IMMEDIATES`
    /// returns `ZkGpuError::InvalidNttSize`. The trampoline at
    /// `WgpuNttPlan::new` does the auto-detection so most callers
    /// don't think about this.
    pub(crate) fn new(
        device: &WgpuDevice,
        config: StockhamPlanConfig,
        direction: NttDirection,
        r4_param_mode: R4ParamMode,
    ) -> Result<Self, ZkGpuError> {
        if r4_param_mode == R4ParamMode::Immediate && !device.caps.has_immediates {
            return Err(ZkGpuError::InvalidNttSize(
                "R4ParamMode::Immediate requires wgpu::Features::IMMEDIATES, \
                 which the active device does not advertise"
                    .to_string(),
            ));
        }
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

        // --- Radix-4 global pipeline (mode-dependent) ---
        //
        // Storage mode: shares the 5-entry NTT BGL + pipeline layout
        // with R2 and local. Today's path.
        //
        // Immediate mode: distinct 4-entry BGL (no slot 3) +
        // pipeline layout with `immediate_size = 32`. The
        // `R4_IMMEDIATE_SIZE_BYTES` constant feeds BOTH the layout
        // descriptor and the `PipelineSpec` cache key — drift-
        // prevention rule from the design note in
        // `pipeline_registry.rs`. The `_with_spec` pipeline path
        // ensures the cache distinguishes the two specializations
        // (different `immediate_size` → different cache slot →
        // separately compiled pipeline).
        let (r4_bind_group_layout, r4_pipeline) = match r4_param_mode {
            R4ParamMode::Storage => {
                let r4_module = reg.get_or_create_module(
                    &device.device,
                    STOCKHAM_R4_SOURCE,
                    "Stockham R4 shader",
                );
                let pipe = reg.get_or_create_pipeline(
                    &device.device,
                    STOCKHAM_R4_SOURCE,
                    "stockham_r4_butterfly",
                    NTT_BGL_LABEL,
                    &ntt_pipeline_layout,
                    &r4_module,
                    device.pipeline_cache.as_ref(),
                );
                (Arc::clone(&ntt_bind_group_layout), pipe)
            }
            R4ParamMode::Immediate => {
                let r4_imm_bgl = reg.get_or_create_bgl(
                    &device.device,
                    R4_IMMEDIATE_BGL_LABEL,
                    &[
                        bgl_storage_entry(0, true),  // src
                        bgl_storage_entry(1, false), // dst
                        bgl_storage_entry(2, true),  // twiddles
                        // binding 3 deliberately absent — params via
                        // var<immediate>, not a uniform buffer.
                        bgl_storage_entry(4, true), // twiddles_prime
                    ],
                );
                let r4_imm_layout = device.device.create_pipeline_layout(
                    &wgpu::PipelineLayoutDescriptor {
                        label: Some("Stockham R4 Immediate pipeline layout"),
                        bind_group_layouts: &[Some(&r4_imm_bgl)],
                        immediate_size: R4_IMMEDIATE_SIZE_BYTES,
                    },
                );
                let r4_imm_module = reg.get_or_create_module(
                    &device.device,
                    STOCKHAM_R4_IMMEDIATE_SOURCE,
                    "Stockham R4 Immediate shader",
                );
                let pipe = reg.get_or_create_pipeline_with_spec(
                    &device.device,
                    STOCKHAM_R4_IMMEDIATE_SOURCE,
                    "stockham_r4_butterfly",
                    R4_IMMEDIATE_BGL_LABEL,
                    &r4_imm_layout,
                    &r4_imm_module,
                    device.pipeline_cache.as_ref(),
                    &crate::pipeline_registry::PipelineSpec {
                        immediate_size: R4_IMMEDIATE_SIZE_BYTES,
                        ..crate::pipeline_registry::PipelineSpec::default()
                    },
                );
                (r4_imm_bgl, pipe)
            }
        };

        // --- Local pipeline (portable radix-4 DIF) ---
        //
        // Speed-opportunities item #2: opt out of the workgroup-memory
        // zero-init that wgpu adds by default. The kernel's gather
        // phase writes `padded_idx(0..1024)` across all 256 threads
        // before any read; subsequent stages each fully write their
        // destination buffer (`shmem_a` / `shmem_b`) before the next
        // stage reads it. The 32 padding physical slots
        // (`tile[r*33 + 32]`-style) are never read. See
        // `crates/zkgpu-wgpu/src/kernels/portable/babybear_stockham_local_r4.wgsl`
        // for the exact write/read pattern audit.
        let local_module = reg.get_or_create_module(
            &device.device,
            STOCKHAM_LOCAL_R4_SOURCE,
            "Stockham local R4 shader",
        );
        let local_pipeline = reg.get_or_create_pipeline_with_spec(
            &device.device,
            STOCKHAM_LOCAL_R4_SOURCE,
            "stockham_local_r4",
            NTT_BGL_LABEL,
            &ntt_pipeline_layout,
            &local_module,
            device.pipeline_cache.as_ref(),
            &crate::pipeline_registry::PipelineSpec {
                zero_initialize_workgroup_memory: false,
                ..crate::pipeline_registry::PipelineSpec::default()
            },
        );

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

        // NVIDIA scale-up Tier 1 Fix 2 (2026-04-16): 2D-folded dispatch
        // grids. Each stage covers a fixed number of butterflies
        // regardless of stage index (Stockham invariant), so a single
        // dispatch shape per dispatch type is sufficient.
        let max_dim = device.caps.max_compute_workgroups_per_dimension;
        let r2_dispatch = plan_linear_dispatch(config.n / 2, WORKGROUP_SIZE, max_dim)?;
        let r4_dispatch = plan_linear_dispatch(config.n / 4, WORKGROUP_SIZE, max_dim)?;
        // Local kernel: one workgroup per BLOCK_SIZE-element sub-problem.
        // Each workgroup is its own unit (no workgroup_size multiplier
        // applied), so pass `WORKGROUP_SIZE=1` so `plan_linear_dispatch`
        // treats `local_workgroups` as the direct workgroup count.
        let local_dispatch = if config.local_workgroups > 0 {
            plan_linear_dispatch(config.local_workgroups, 1, max_dim)?
        } else {
            crate::dispatch::LinearDispatch { x: 0, y: 0, groups_per_row: 1 }
        };

        let mut global_stage_param_buffers = Vec::with_capacity(config.global_stage_params.len());
        for sp in &config.global_stage_params {
            let params = [
                sp.n,
                sp.s,
                sp.m,
                sp.twiddle_offset,
                r2_dispatch.groups_per_row,
                0u32,
                0u32,
                0u32,
            ];
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

        // R4 stage params: shape depends on mode.
        // - Storage: pre-built `wgpu::Buffer` per stage, bound at slot 3.
        // - Immediate: pre-computed [u32; 8] per stage, written via
        //   `pass.set_immediates(0, &bytes)` before each dispatch.
        let r4_param_source = match r4_param_mode {
            R4ParamMode::Storage => {
                let mut buffers = Vec::with_capacity(config.r4_stage_params.len());
                for sp in &config.r4_stage_params {
                    let params = [
                        sp.n,
                        sp.s,
                        sp.m4,
                        sp.twiddle_offset,
                        omega4,
                        omega4_prime,
                        r4_dispatch.groups_per_row,
                        0u32,
                    ];
                    let buf = device.device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("Stockham R4 stage params"),
                            contents: bytemuck::cast_slice(&params),
                            usage: wgpu::BufferUsages::UNIFORM,
                        },
                    );
                    buffers.push(buf);
                }
                R4ParamSource::Storage(buffers)
            }
            R4ParamMode::Immediate => {
                let mut bytes = Vec::with_capacity(config.r4_stage_params.len());
                for sp in &config.r4_stage_params {
                    bytes.push([
                        sp.n,
                        sp.s,
                        sp.m4,
                        sp.twiddle_offset,
                        omega4,
                        omega4_prime,
                        r4_dispatch.groups_per_row,
                        0u32,
                    ]);
                }
                R4ParamSource::Immediate(bytes)
            }
        };

        // --- Local twiddles and params (radix-4 DIF) ---
        let (local_twiddles, local_twiddles_prime, local_omega4, local_omega4_prime) =
            precompute_local_r4_twiddles(direction);
        let local_twiddle_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Stockham local R4 twiddles"),
                    contents: bytemuck::cast_slice(&local_twiddles),
                    usage: wgpu::BufferUsages::STORAGE,
                });
        let local_twiddle_prime_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Stockham local R4 twiddles prime"),
                    contents: bytemuck::cast_slice(&local_twiddles_prime),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        // Slot 3 (`_pad0` in the WGSL struct) now carries the 2D-dispatch
        // `groups_per_row` for workgroup-index reconstruction.
        let local_params = [
            config.local_stride,
            local_omega4,
            local_omega4_prime,
            local_dispatch.groups_per_row,
        ];
        let local_param_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Stockham local R4 params"),
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
            r4_bind_group_layout,
            global_twiddle_buffer,
            global_twiddle_prime_buffer,
            global_stage_param_buffers,
            r4_twiddle_buffer,
            r4_twiddle_prime_buffer,
            r4_param_source,
            local_twiddle_buffer,
            local_twiddle_prime_buffer,
            local_param_buffer,
            scratch_buffer,
            scale_pipeline,
            scale_bind_group_layout,
            scale_param_buffer,
            scale_dispatch,
            r2_dispatch,
            r4_dispatch,
            local_dispatch,
            config,
        })
    }
}
