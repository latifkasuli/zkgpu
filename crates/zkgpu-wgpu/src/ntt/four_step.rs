use std::sync::Arc;

use wgpu::util::DeviceExt;
use zkgpu_babybear::BabyBear;
use zkgpu_core::{GpuField, NttDirection, ZkGpuError};

use crate::async_util;
use crate::buffer::WgpuBuffer;
use crate::device::WgpuDevice;
use crate::dispatch::{plan_linear_dispatch, LinearDispatch};
use crate::profiling::{GpuProfiler, GpuTiming, TimestampSpan};

use super::planner::{FourStepPlanConfig, StockhamPlanConfig, WORKGROUP_SIZE};
use super::stockham::NttTimings;
use super::twiddles::{
    precompute_fourstep_twiddles, precompute_single_r2_twiddles,
    precompute_stockham_r4_twiddles,
};

const FOURSTEP_LEAF_R2_SOURCE: &str =
    include_str!("../kernels/portable/babybear_fourstep_leaf_r2.wgsl");
const FOURSTEP_LEAF_R4_SOURCE: &str =
    include_str!("../kernels/portable/babybear_fourstep_leaf_r4.wgsl");
const FOURSTEP_TWIDDLE_SOURCE: &str =
    include_str!("../kernels/portable/babybear_fourstep_twiddle.wgsl");
const FOURSTEP_TRANSPOSE_SOURCE: &str =
    include_str!("../kernels/portable/babybear_fourstep_transpose.wgsl");
const SCALE_SOURCE: &str =
    include_str!("../kernels/portable/babybear_scale.wgsl");

const LEAF_BGL_LABEL: &str = "Four-step leaf BGL";
const TWIDDLE_BGL_LABEL: &str = "Four-step twiddle BGL";
const TRANSPOSE_BGL_LABEL: &str = "Four-step transpose BGL";
const SCALE_BGL_LABEL: &str = "Four-step scale BGL";

/// GPU four-step NTT plan for BabyBear fields.
///
/// Correct Cooley-Tukey four-step decomposition for N = R × C:
///
///   Phase 1: Transpose R×C → C×R
///   Phase 2: R-point batched row DFTs (C batches — column DFTs of original)
///   Phase 3: Diagonal twiddle multiply omega_N^(k_r × c)
///   Phase 4: Transpose C×R → R×C
///   Phase 5: C-point batched row DFTs (R batches — row DFTs of original)
///   Phase 6: Transpose R×C → C×R (output reordering)
///   Phase 7: Inverse scale (inverse NTT only)
pub(crate) struct FourStepPlan {
    leaf_global_pipeline: Arc<wgpu::ComputePipeline>,
    leaf_r4_pipeline: Arc<wgpu::ComputePipeline>,
    leaf_bgl: Arc<wgpu::BindGroupLayout>,

    twiddle_pipeline: Arc<wgpu::ComputePipeline>,
    twiddle_bgl: Arc<wgpu::BindGroupLayout>,
    twiddle_buffer: wgpu::Buffer,
    twiddle_prime_buffer: wgpu::Buffer,
    twiddle_param_buffer: wgpu::Buffer,
    twiddle_dispatch: LinearDispatch,

    transpose_pipeline: Arc<wgpu::ComputePipeline>,
    transpose_bgl: Arc<wgpu::BindGroupLayout>,
    /// R×C → C×R transpose params (phases 1 and 6)
    transpose_rc_to_cr_params: wgpu::Buffer,
    /// C×R → R×C transpose params (phase 4)
    transpose_cr_to_rc_params: wgpu::Buffer,

    scale_pipeline: Arc<wgpu::ComputePipeline>,
    scale_bgl: Arc<wgpu::BindGroupLayout>,
    scale_param_buffer: Option<wgpu::Buffer>,
    scale_dispatch: LinearDispatch,

    /// Phase-2 leaf: R-point NTTs (col_leaf config), C batches
    phase2_r4_twiddle_buffer: wgpu::Buffer,
    phase2_r4_twiddle_prime_buffer: wgpu::Buffer,
    phase2_r4_stage_param_buffers: Vec<wgpu::Buffer>,
    phase2_r2_twiddle_buffer: wgpu::Buffer,
    phase2_r2_twiddle_prime_buffer: wgpu::Buffer,
    phase2_r2_stage_param_buffers: Vec<wgpu::Buffer>,

    /// Phase-5 leaf: C-point NTTs (row_leaf config), R batches
    phase5_r4_twiddle_buffer: wgpu::Buffer,
    phase5_r4_twiddle_prime_buffer: wgpu::Buffer,
    phase5_r4_stage_param_buffers: Vec<wgpu::Buffer>,
    phase5_r2_twiddle_buffer: wgpu::Buffer,
    phase5_r2_twiddle_prime_buffer: wgpu::Buffer,
    phase5_r2_stage_param_buffers: Vec<wgpu::Buffer>,

    scratch_buffer: wgpu::Buffer,
    transpose_scratch_buffer: wgpu::Buffer,

    config: FourStepPlanConfig,
}

impl FourStepPlan {
    pub(crate) fn new(
        device: &WgpuDevice,
        config: FourStepPlanConfig,
        direction: NttDirection,
    ) -> Result<Self, ZkGpuError> {
        #[cfg(not(target_arch = "wasm32"))]
        device.push_validation_scope();

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
                    bind_group_layouts: &[&leaf_bgl],
                    push_constant_ranges: &[],
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
                    bind_group_layouts: &[&twiddle_bgl],
                    push_constant_ranges: &[],
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
                    bind_group_layouts: &[&transpose_bgl],
                    push_constant_ranges: &[],
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
                    bind_group_layouts: &[&scale_bgl],
                    push_constant_ranges: &[],
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
        device.pop_validation_scope("four-step pipeline creation")?;

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
                    label: Some("Four-step transpose R×C→C×R params"),
                    contents: bytemuck::cast_slice(&rc_to_cr),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
        // C×R → R×C: params = (cols, rows)
        let cr_to_rc = [config.cols, config.rows, 0u32, 0u32];
        let transpose_cr_to_rc_params =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Four-step transpose C×R→R×C params"),
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

    pub(crate) fn num_dispatches(&self) -> u32 {
        self.config.total_dispatches() + u32::from(self.scale_param_buffer.is_some())
    }

    pub(crate) fn execute_kernels(
        &mut self,
        device: &WgpuDevice,
        buf: &mut WgpuBuffer<BabyBear>,
    ) -> Result<(), ZkGpuError> {
        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Four-step NTT encoder"),
            });

        self.encode_all(&device.device, &mut encoder, buf, &[]);

        device.queue.submit(Some(encoder.finish()));
        device.device.poll(wgpu::Maintain::Wait);
        Ok(())
    }

    pub(crate) fn execute_kernels_profiled(
        &mut self,
        device: &WgpuDevice,
        buf: &mut WgpuBuffer<BabyBear>,
    ) -> Result<Option<NttTimings>, ZkGpuError> {
        let wall_start = std::time::Instant::now();

        let num_dispatches = self.num_dispatches();
        let max_queries = num_dispatches * 2;
        let mut profiler =
            GpuProfiler::new_if_supported(&device.device, &device.queue, &device.caps, max_queries);

        let spans: Vec<TimestampSpan> = if let Some(ref mut p) = profiler {
            (0..num_dispatches).filter_map(|_| p.begin_span()).collect()
        } else {
            Vec::new()
        };

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Four-step NTT profiled encoder"),
            });

        let ts_writes: Vec<Option<wgpu::ComputePassTimestampWrites<'_>>> = (0..num_dispatches
            as usize)
            .map(|i| {
                profiler
                    .as_ref()
                    .and_then(|p| spans.get(i).map(|span| p.pass_timestamp_writes(span)))
            })
            .collect();

        self.encode_all(&device.device, &mut encoder, buf, &ts_writes);

        if let Some(ref profiler) = profiler {
            profiler.resolve(&mut encoder);
        }

        device.queue.submit(Some(encoder.finish()));
        device.device.poll(wgpu::Maintain::Wait);

        let wall_clock = wall_start.elapsed();

        let timings = if let Some(ref profiler) = profiler {
            let timestamps = profiler.read_timestamps(&device.device)?;
            let mut gpu_stage_ns = Vec::with_capacity(spans.len());
            let mut total = 0.0;

            let labels = self.dispatch_labels();
            for (i, span) in spans.iter().enumerate() {
                let dur = profiler.span_duration_ns(&timestamps, span).unwrap_or(0.0);
                total += dur;
                gpu_stage_ns.push(GpuTiming {
                    label: labels
                        .get(i)
                        .cloned()
                        .unwrap_or_else(|| format!("dispatch {i}")),
                    duration_ns: dur,
                });
            }
            Some(NttTimings {
                wall_clock,
                gpu_stage_ns,
                gpu_total_ns: total,
            })
        } else {
            None
        };

        Ok(timings)
    }

    /// Async variant of [`execute_kernels`](Self::execute_kernels). Browser-safe.
    pub(crate) async fn execute_kernels_async(
        &mut self,
        device: &WgpuDevice,
        buf: &mut WgpuBuffer<BabyBear>,
    ) -> Result<(), ZkGpuError> {
        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Four-step NTT encoder"),
            });

        self.encode_all(&device.device, &mut encoder, buf, &[]);

        device.queue.submit(Some(encoder.finish()));
        async_util::wait_for_submission(&device.device, &device.queue).await;
        Ok(())
    }

    /// Async variant of [`execute_kernels_profiled`](Self::execute_kernels_profiled).
    /// Browser-safe. Uses `web_time::Instant` for wall-clock timing.
    pub(crate) async fn execute_kernels_profiled_async(
        &mut self,
        device: &WgpuDevice,
        buf: &mut WgpuBuffer<BabyBear>,
    ) -> Result<Option<NttTimings>, ZkGpuError> {
        let wall_start = web_time::Instant::now();

        let num_dispatches = self.num_dispatches();
        let max_queries = num_dispatches * 2;
        let mut profiler =
            GpuProfiler::new_if_supported(&device.device, &device.queue, &device.caps, max_queries);

        let spans: Vec<TimestampSpan> = if let Some(ref mut p) = profiler {
            (0..num_dispatches).filter_map(|_| p.begin_span()).collect()
        } else {
            Vec::new()
        };

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Four-step NTT profiled encoder"),
            });

        let ts_writes: Vec<Option<wgpu::ComputePassTimestampWrites<'_>>> = (0..num_dispatches
            as usize)
            .map(|i| {
                profiler
                    .as_ref()
                    .and_then(|p| spans.get(i).map(|span| p.pass_timestamp_writes(span)))
            })
            .collect();

        self.encode_all(&device.device, &mut encoder, buf, &ts_writes);

        if let Some(ref profiler) = profiler {
            profiler.resolve(&mut encoder);
        }

        device.queue.submit(Some(encoder.finish()));

        let wall_clock = wall_start.elapsed();

        let timings = if let Some(ref profiler) = profiler {
            let timestamps = profiler.read_timestamps_async(&device.device).await?;
            let mut gpu_stage_ns = Vec::with_capacity(spans.len());
            let mut total = 0.0;

            let labels = self.dispatch_labels();
            for (i, span) in spans.iter().enumerate() {
                let dur = profiler.span_duration_ns(&timestamps, span).unwrap_or(0.0);
                total += dur;
                gpu_stage_ns.push(GpuTiming {
                    label: labels
                        .get(i)
                        .cloned()
                        .unwrap_or_else(|| format!("dispatch {i}")),
                    duration_ns: dur,
                });
            }
            Some(NttTimings {
                wall_clock,
                gpu_stage_ns,
                gpu_total_ns: total,
            })
        } else {
            async_util::wait_for_submission(&device.device, &device.queue).await;
            None
        };

        Ok(timings)
    }

    fn dispatch_labels(&self) -> Vec<String> {
        let mut labels = Vec::new();
        labels.push("transpose R×C→C×R".to_string());
        for i in 0..self.config.col_leaf.r4_stage_params.len() {
            let h = i as u32 * 2;
            labels.push(format!("phase2 R-pt r4 stages {}+{}", h, h + 1));
        }
        for i in 0..self.config.col_leaf.global_stage_params.len() {
            let h = self.config.col_leaf.r4_stage_params.len() as u32 * 2 + i as u32;
            labels.push(format!("phase2 R-pt r2 stage {h}"));
        }
        labels.push("twiddle multiply".to_string());
        labels.push("transpose C×R→R×C".to_string());
        for i in 0..self.config.row_leaf.r4_stage_params.len() {
            let h = i as u32 * 2;
            labels.push(format!("phase5 C-pt r4 stages {}+{}", h, h + 1));
        }
        for i in 0..self.config.row_leaf.global_stage_params.len() {
            let h = self.config.row_leaf.r4_stage_params.len() as u32 * 2 + i as u32;
            labels.push(format!("phase5 C-pt r2 stage {h}"));
        }
        labels.push("transpose R×C→C×R (output)".to_string());
        if self.scale_param_buffer.is_some() {
            labels.push("inverse scale".to_string());
        }
        labels
    }

    /// Encode all six phases of the four-step NTT.
    fn encode_all(
        &self,
        wgpu_device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        buf: &WgpuBuffer<BabyBear>,
        ts_writes: &[Option<wgpu::ComputePassTimestampWrites<'_>>],
    ) {
        let buf_size = (self.config.n as u64) * std::mem::size_of::<u32>() as u64;
        let mut dispatch_idx = 0usize;

        // Phase 1: Transpose R×C → C×R (buf → transpose_scratch → buf)
        dispatch_idx = self.encode_transpose(
            wgpu_device,
            encoder,
            &buf.inner,
            &self.transpose_scratch_buffer,
            &self.transpose_rc_to_cr_params,
            self.config.transpose_workgroups_x,
            self.config.transpose_workgroups_y,
            ts_writes,
            dispatch_idx,
        );
        encoder.copy_buffer_to_buffer(&self.transpose_scratch_buffer, 0, &buf.inner, 0, buf_size);

        // Phase 2: R-point batched row DFTs (C batches)
        // Data is now C×R in buf. C independent R-point NTTs on contiguous rows.
        dispatch_idx = self.encode_batched_leaf_r4(
            wgpu_device,
            encoder,
            buf,
            &self.config.col_leaf,
            self.config.cols,
            &self.phase2_r4_twiddle_buffer,
            &self.phase2_r4_twiddle_prime_buffer,
            &self.phase2_r4_stage_param_buffers,
            &self.phase2_r2_twiddle_buffer,
            &self.phase2_r2_twiddle_prime_buffer,
            &self.phase2_r2_stage_param_buffers,
            ts_writes,
            dispatch_idx,
        );

        // Phase 3: Twiddle multiply on C×R data (in-place on buf)
        {
            let bind_group = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.twiddle_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.inner.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.twiddle_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.twiddle_param_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.twiddle_prime_buffer.as_entire_binding(),
                    },
                ],
            });

            let ts = ts_writes.get(dispatch_idx).and_then(|t| t.clone());
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: ts,
                });
                pass.set_pipeline(&self.twiddle_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(self.twiddle_dispatch.x, self.twiddle_dispatch.y, 1);
            }
            dispatch_idx += 1;
        }

        // Phase 4: Transpose C×R → R×C (buf → transpose_scratch → buf)
        dispatch_idx = self.encode_transpose(
            wgpu_device,
            encoder,
            &buf.inner,
            &self.transpose_scratch_buffer,
            &self.transpose_cr_to_rc_params,
            self.config.transpose_workgroups_y, // C×R: x-tiles = R/tile, y-tiles = C/tile
            self.config.transpose_workgroups_x,
            ts_writes,
            dispatch_idx,
        );
        encoder.copy_buffer_to_buffer(&self.transpose_scratch_buffer, 0, &buf.inner, 0, buf_size);

        // Phase 5: C-point batched row DFTs (R batches)
        // Data is now R×C in buf. R independent C-point NTTs on contiguous rows.
        dispatch_idx = self.encode_batched_leaf_r4(
            wgpu_device,
            encoder,
            buf,
            &self.config.row_leaf,
            self.config.rows,
            &self.phase5_r4_twiddle_buffer,
            &self.phase5_r4_twiddle_prime_buffer,
            &self.phase5_r4_stage_param_buffers,
            &self.phase5_r2_twiddle_buffer,
            &self.phase5_r2_twiddle_prime_buffer,
            &self.phase5_r2_stage_param_buffers,
            ts_writes,
            dispatch_idx,
        );

        // Phase 6: Transpose R×C → C×R (final output reordering)
        dispatch_idx = self.encode_transpose(
            wgpu_device,
            encoder,
            &buf.inner,
            &self.transpose_scratch_buffer,
            &self.transpose_rc_to_cr_params,
            self.config.transpose_workgroups_x,
            self.config.transpose_workgroups_y,
            ts_writes,
            dispatch_idx,
        );
        encoder.copy_buffer_to_buffer(&self.transpose_scratch_buffer, 0, &buf.inner, 0, buf_size);

        // Phase 7: Inverse scale
        if let Some(ref param_buf) = self.scale_param_buffer {
            let bind_group = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.scale_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.inner.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: param_buf.as_entire_binding(),
                    },
                ],
            });

            let ts = ts_writes.get(dispatch_idx).and_then(|t| t.clone());
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: ts,
                });
                pass.set_pipeline(&self.scale_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(self.scale_dispatch.x, self.scale_dispatch.y, 1);
            }
        }
    }

    /// Encode a single transpose dispatch (src → dst).
    #[allow(clippy::too_many_arguments)]
    fn encode_transpose(
        &self,
        wgpu_device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::Buffer,
        dst: &wgpu::Buffer,
        param_buffer: &wgpu::Buffer,
        workgroups_x: u32,
        workgroups_y: u32,
        ts_writes: &[Option<wgpu::ComputePassTimestampWrites<'_>>],
        dispatch_idx: usize,
    ) -> usize {
        let bind_group = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.transpose_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: src.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dst.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: param_buffer.as_entire_binding(),
                },
            ],
        });

        let ts = ts_writes.get(dispatch_idx).and_then(|t| t.clone());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: ts,
            });
            pass.set_pipeline(&self.transpose_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }
        dispatch_idx + 1
    }

    /// Encode batched R4 + R2 global Stockham stages for leaf NTTs.
    ///
    /// Dispatches R4 stages first using `leaf_r4_pipeline`, then any R2
    /// remainder stages using `leaf_global_pipeline`. The src/dst ping-pong
    /// continues across both R4 and R2 dispatches.
    #[allow(clippy::too_many_arguments)]
    fn encode_batched_leaf_r4(
        &self,
        wgpu_device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        buf: &WgpuBuffer<BabyBear>,
        leaf_config: &StockhamPlanConfig,
        batch_count: u32,
        r4_twiddle_buffer: &wgpu::Buffer,
        r4_twiddle_prime_buffer: &wgpu::Buffer,
        r4_stage_param_buffers: &[wgpu::Buffer],
        r2_twiddle_buffer: &wgpu::Buffer,
        r2_twiddle_prime_buffer: &wgpu::Buffer,
        r2_stage_param_buffers: &[wgpu::Buffer],
        ts_writes: &[Option<wgpu::ComputePassTimestampWrites<'_>>],
        start_dispatch: usize,
    ) -> usize {
        let mut dispatch_idx = start_dispatch;
        let mut parity = 0usize;

        // R4 dispatches: each R4 butterfly processes 4 elements → leaf_n/4 butterflies per batch
        for param_buffer in r4_stage_param_buffers {
            let (src_buf, dst_buf) = if parity % 2 == 0 {
                (&buf.inner, &self.scratch_buffer)
            } else {
                (&self.scratch_buffer, &buf.inner)
            };

            let bind_group = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.leaf_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: src_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: dst_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: r4_twiddle_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: param_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: r4_twiddle_prime_buffer.as_entire_binding(),
                    },
                ],
            });

            let total_r4_butterflies = batch_count * (leaf_config.n / 4);
            let workgroups = total_r4_butterflies.div_ceil(WORKGROUP_SIZE);

            let ts = ts_writes.get(dispatch_idx).and_then(|t| t.clone());
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: ts,
                });
                pass.set_pipeline(&self.leaf_r4_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            dispatch_idx += 1;
            parity += 1;
        }

        // R2 remainder dispatches
        for param_buffer in r2_stage_param_buffers {
            let (src_buf, dst_buf) = if parity % 2 == 0 {
                (&buf.inner, &self.scratch_buffer)
            } else {
                (&self.scratch_buffer, &buf.inner)
            };

            let bind_group = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.leaf_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: src_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: dst_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: r2_twiddle_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: param_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: r2_twiddle_prime_buffer.as_entire_binding(),
                    },
                ],
            });

            let total_butterflies = batch_count * (leaf_config.n / 2);
            let workgroups = total_butterflies.div_ceil(WORKGROUP_SIZE);

            let ts = ts_writes.get(dispatch_idx).and_then(|t| t.clone());
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: ts,
                });
                pass.set_pipeline(&self.leaf_global_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            dispatch_idx += 1;
            parity += 1;
        }

        // Copy result back to buf if it ended in scratch
        if leaf_config.result_in_scratch {
            let size = (self.config.n as u64) * std::mem::size_of::<u32>() as u64;
            encoder.copy_buffer_to_buffer(&self.scratch_buffer, 0, &buf.inner, 0, size);
        }

        dispatch_idx
    }
}

fn build_batched_r4_stage_params(
    device: &wgpu::Device,
    leaf: &StockhamPlanConfig,
    batch_count: u32,
    omega4: u32,
    omega4_prime: u32,
) -> Vec<wgpu::Buffer> {
    leaf.r4_stage_params
        .iter()
        .map(|sp| {
            let params = [
                sp.n,
                sp.s,
                sp.m4,
                sp.twiddle_offset,
                batch_count,
                omega4,
                omega4_prime,
                0u32,
            ];
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Batched R4 leaf stage params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            })
        })
        .collect()
}

fn build_batched_r2_stage_params(
    device: &wgpu::Device,
    leaf: &StockhamPlanConfig,
    batch_count: u32,
) -> Vec<wgpu::Buffer> {
    leaf.global_stage_params
        .iter()
        .map(|sp| {
            let params = [
                sp.n,
                sp.s,
                sp.m,
                sp.twiddle_offset,
                batch_count,
                0u32,
                0u32,
                0u32,
            ];
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Batched R2 leaf stage params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            })
        })
        .collect()
}

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
