use std::sync::Arc;

use wgpu::util::DeviceExt;
use zkgpu_babybear::BabyBear;
use zkgpu_core::{GpuField, NttDirection, ZkGpuError};
use crate::async_util;
use crate::buffer::WgpuBuffer;
use crate::device::WgpuDevice;
use crate::dispatch::{plan_linear_dispatch, LinearDispatch};
use crate::profiling::{GpuProfiler, GpuTiming, TimestampSpan};

use super::planner::{LocalKernelHint, StockhamPlanConfig, WORKGROUP_SIZE};
use super::twiddles::{
    precompute_local_r4_twiddles, precompute_stockham_r4_twiddles,
    precompute_subgroup_local_twiddles, shoup_quotient,
};

const STOCKHAM_R2_SOURCE: &str =
    include_str!("../kernels/portable/babybear_stockham_r2.wgsl");
const STOCKHAM_R4_SOURCE: &str =
    include_str!("../kernels/portable/babybear_stockham_r4.wgsl");
const STOCKHAM_LOCAL_R4_SOURCE: &str =
    include_str!("../kernels/portable/babybear_stockham_local_r4.wgsl");
const STOCKHAM_LOCAL_SUBGROUP_SOURCE: &str =
    include_str!("../kernels/native/babybear_stockham_local_subgroup.wgsl");
const SCALE_SOURCE: &str =
    include_str!("../kernels/portable/babybear_scale.wgsl");

const NTT_BGL_LABEL: &str = "Stockham NTT bind group layout";
const SCALE_BGL_LABEL: &str = "Scale bind group layout";

/// GPU NTT execution plan for BabyBear fields using Stockham autosort.
///
/// Two-phase hybrid architecture:
///
/// - **N < BLOCK_SIZE**: all stages run as global DIF Stockham passes
///   (one dispatch per stage, ping-pong between data and scratch).
///
/// - **N >= BLOCK_SIZE**: the first `log_n - LOG_BLOCK` stages run as
///   global DIF Stockham passes, reducing the problem to N/BLOCK_SIZE
///   independent sub-problems. A single workgroup-local dispatch then
///   finishes all remaining LOG_BLOCK stages in shared memory.
///
/// For inverse NTT, a final GPU dispatch multiplies every element by
/// `n^{-1} mod P`, avoiding a host round-trip.
pub(crate) struct StockhamPlan {
    global_pipeline: Arc<wgpu::ComputePipeline>,
    r4_pipeline: Arc<wgpu::ComputePipeline>,
    local_pipeline: Arc<wgpu::ComputePipeline>,
    ntt_bind_group_layout: Arc<wgpu::BindGroupLayout>,

    global_twiddle_buffer: wgpu::Buffer,
    global_twiddle_prime_buffer: wgpu::Buffer,
    global_stage_param_buffers: Vec<wgpu::Buffer>,

    r4_twiddle_buffer: wgpu::Buffer,
    r4_twiddle_prime_buffer: wgpu::Buffer,
    r4_stage_param_buffers: Vec<wgpu::Buffer>,

    local_twiddle_buffer: wgpu::Buffer,
    local_twiddle_prime_buffer: wgpu::Buffer,
    local_param_buffer: wgpu::Buffer,

    scratch_buffer: wgpu::Buffer,

    scale_pipeline: Arc<wgpu::ComputePipeline>,
    scale_bind_group_layout: Arc<wgpu::BindGroupLayout>,
    scale_param_buffer: Option<wgpu::Buffer>,
    scale_dispatch: LinearDispatch,

    config: StockhamPlanConfig,
}

/// Optional GPU timing results from a profiled NTT execution.
#[derive(Debug, Clone)]
pub struct NttTimings {
    pub wall_clock: std::time::Duration,
    pub gpu_stage_ns: Vec<GpuTiming>,
    pub gpu_total_ns: f64,
}

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
        device.push_validation_scope();

        let reg = &device.pipelines;

        // --- NTT bind group layout (shared between global and local pipelines) ---
        let ntt_bind_group_layout = reg.get_or_create_bgl(
            &device.device,
            NTT_BGL_LABEL,
            &[
                bgl_storage_entry(0, true),  // src (read-only)
                bgl_storage_entry(1, false), // dst (read-write)
                bgl_storage_entry(2, true),  // twiddles (read-only)
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
                bgl_storage_entry(4, true),  // twiddles_prime (read-only)
            ],
        );

        let ntt_pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Stockham NTT pipeline layout"),
                    bind_group_layouts: &[&ntt_bind_group_layout],
                    push_constant_ranges: &[],
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
        let use_subgroup_local = match local_hint {
            LocalKernelHint::ForceSubgroup => {
                if !device.caps.has_subgroup || device.caps.min_subgroup_size < 32 {
                    return Err(ZkGpuError::GpuValidation(
                        "ForceSubgroup requires SUBGROUP feature with min_subgroup_size >= 32"
                            .into(),
                    ));
                }
                true
            }
            LocalKernelHint::ForcePortable => false,
            LocalKernelHint::Auto => {
                device.caps.has_subgroup && device.caps.min_subgroup_size >= 32
            }
        };

        let local_pipeline = if use_subgroup_local {
            let local_module = reg.get_or_create_module(
                &device.device,
                STOCKHAM_LOCAL_SUBGROUP_SOURCE,
                "Stockham local subgroup shader",
            );
            reg.get_or_create_pipeline(
                &device.device,
                STOCKHAM_LOCAL_SUBGROUP_SOURCE,
                "stockham_local_subgroup",
                NTT_BGL_LABEL,
                &ntt_pipeline_layout,
                &local_module,
                device.pipeline_cache.as_ref(),
            )
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
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

        let scale_pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Scale pipeline layout"),
                    bind_group_layouts: &[&scale_bind_group_layout],
                    push_constant_ranges: &[],
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
        device.pop_validation_scope("stockham pipeline creation")?;

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

    /// Total number of GPU dispatches (NTT stages + optional scaling).
    pub(crate) fn num_dispatches(&self) -> u32 {
        self.config.ntt_dispatches() + u32::from(self.scale_param_buffer.is_some())
    }
    /// Execute the full NTT (stages + inverse scaling if applicable) in a
    /// single command submission. Entirely GPU-side — no host round-trips.
    pub(crate) fn execute_kernels(
        &mut self,
        device: &WgpuDevice,
        buf: &mut WgpuBuffer<BabyBear>,
    ) -> Result<(), ZkGpuError> {
        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Stockham NTT encoder"),
            });

        self.encode_ntt_stages(&device.device, &mut encoder, buf, &[]);
        self.encode_scale(&device.device, &mut encoder, buf, None);

        device.queue.submit(Some(encoder.finish()));
        device.device.poll(wgpu::Maintain::Wait);

        Ok(())
    }

    /// Execute the full NTT with GPU timestamp profiling.
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
                label: Some("Stockham NTT profiled encoder"),
            });

        let ntt_dispatches = self.config.ntt_dispatches() as usize;
        let ntt_ts_writes: Vec<Option<wgpu::ComputePassTimestampWrites<'_>>> = (0..ntt_dispatches)
            .map(|s| {
                profiler
                    .as_ref()
                    .and_then(|p| spans.get(s).map(|span| p.pass_timestamp_writes(span)))
            })
            .collect();

        self.encode_ntt_stages(&device.device, &mut encoder, buf, &ntt_ts_writes);

        let scale_ts = profiler.as_ref().and_then(|p| {
            spans
                .get(ntt_dispatches)
                .map(|span| p.pass_timestamp_writes(span))
        });
        self.encode_scale(&device.device, &mut encoder, buf, scale_ts);

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
            let r4_count = self.config.r4_stage_params.len();
            let r2_count = self.config.global_stage_params.len();
            for (i, span) in spans.iter().enumerate() {
                let dur = profiler
                    .span_duration_ns(&timestamps, span)
                    .unwrap_or(0.0);
                total += dur;
                let label = if i < r4_count {
                    format!("r4 stages {}+{}", i * 2, i * 2 + 1)
                } else if i < r4_count + r2_count {
                    let r2_idx = i - r4_count;
                    let stage_h = r4_count * 2 + r2_idx;
                    format!("r2 stage {stage_h}")
                } else if i < ntt_dispatches {
                    "local fused".to_string()
                } else {
                    "inverse scale".to_string()
                };
                gpu_stage_ns.push(GpuTiming {
                    label,
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
                label: Some("Stockham NTT encoder"),
            });

        self.encode_ntt_stages(&device.device, &mut encoder, buf, &[]);
        self.encode_scale(&device.device, &mut encoder, buf, None);

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
                label: Some("Stockham NTT profiled encoder"),
            });

        let ntt_dispatches = self.config.ntt_dispatches() as usize;
        let ntt_ts_writes: Vec<Option<wgpu::ComputePassTimestampWrites<'_>>> = (0..ntt_dispatches)
            .map(|s| {
                profiler
                    .as_ref()
                    .and_then(|p| spans.get(s).map(|span| p.pass_timestamp_writes(span)))
            })
            .collect();

        self.encode_ntt_stages(&device.device, &mut encoder, buf, &ntt_ts_writes);

        let scale_ts = profiler.as_ref().and_then(|p| {
            spans
                .get(ntt_dispatches)
                .map(|span| p.pass_timestamp_writes(span))
        });
        self.encode_scale(&device.device, &mut encoder, buf, scale_ts);

        if let Some(ref profiler) = profiler {
            profiler.resolve(&mut encoder);
        }

        device.queue.submit(Some(encoder.finish()));

        let wall_clock = wall_start.elapsed();

        let timings = if let Some(ref profiler) = profiler {
            let timestamps = profiler.read_timestamps_async(&device.device).await?;
            let mut gpu_stage_ns = Vec::with_capacity(spans.len());
            let mut total = 0.0;
            let r4_count = self.config.r4_stage_params.len();
            let r2_count = self.config.global_stage_params.len();
            for (i, span) in spans.iter().enumerate() {
                let dur = profiler
                    .span_duration_ns(&timestamps, span)
                    .unwrap_or(0.0);
                total += dur;
                let label = if i < r4_count {
                    format!("r4 stages {}+{}", i * 2, i * 2 + 1)
                } else if i < r4_count + r2_count {
                    let r2_idx = i - r4_count;
                    let stage_h = r4_count * 2 + r2_idx;
                    format!("r2 stage {stage_h}")
                } else if i < ntt_dispatches {
                    "local fused".to_string()
                } else {
                    "inverse scale".to_string()
                };
                gpu_stage_ns.push(GpuTiming {
                    label,
                    duration_ns: dur,
                });
            }
            Some(NttTimings {
                wall_clock,
                gpu_stage_ns,
                gpu_total_ns: total,
            })
        } else {
            // No profiler — still wait for GPU completion.
            async_util::wait_for_submission(&device.device, &device.queue).await;
            None
        };

        Ok(timings)
    }

    /// Encode NTT stages into the given command encoder.
    ///
    /// Dispatch order: R4 global stages -> R2 global stages -> local fused.
    fn encode_ntt_stages(
        &self,
        wgpu_device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        buf: &WgpuBuffer<BabyBear>,
        ts_writes: &[Option<wgpu::ComputePassTimestampWrites<'_>>],
    ) {
        let mut dispatch_idx: usize = 0;
        let r4_workgroups = (self.config.n / 4).div_ceil(WORKGROUP_SIZE);

        // Phase 1a: Radix-4 global dispatches
        for param_buffer in self.r4_stage_param_buffers.iter() {
            let (src_buf, dst_buf) = if dispatch_idx % 2 == 0 {
                (&buf.inner, &self.scratch_buffer)
            } else {
                (&self.scratch_buffer, &buf.inner)
            };

            let bind_group = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.ntt_bind_group_layout,
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
                        resource: self.r4_twiddle_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: param_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.r4_twiddle_prime_buffer.as_entire_binding(),
                    },
                ],
            });

            let ts = ts_writes.get(dispatch_idx).and_then(|t| t.clone());

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: ts,
                });
                pass.set_pipeline(&self.r4_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(r4_workgroups, 1, 1);
            }

            dispatch_idx += 1;
        }

        // Phase 1b: Radix-2 remainder global dispatches
        for param_buffer in &self.global_stage_param_buffers {
            let (src_buf, dst_buf) = if dispatch_idx % 2 == 0 {
                (&buf.inner, &self.scratch_buffer)
            } else {
                (&self.scratch_buffer, &buf.inner)
            };

            let bind_group = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.ntt_bind_group_layout,
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
                        resource: self.global_twiddle_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: param_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.global_twiddle_prime_buffer.as_entire_binding(),
                    },
                ],
            });

            let ts = ts_writes.get(dispatch_idx).and_then(|t| t.clone());

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: ts,
                });
                pass.set_pipeline(&self.global_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(self.config.global_workgroups, 1, 1);
            }

            dispatch_idx += 1;
        }

        // Phase 2: workgroup-local fused dispatch
        if self.config.use_local_kernel {
            let (src_buf, dst_buf) = if dispatch_idx % 2 == 0 {
                (&buf.inner, &self.scratch_buffer)
            } else {
                (&self.scratch_buffer, &buf.inner)
            };

            let bind_group = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.ntt_bind_group_layout,
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
                        resource: self.local_twiddle_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.local_param_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.local_twiddle_prime_buffer.as_entire_binding(),
                    },
                ],
            });

            let ts = ts_writes.get(dispatch_idx).and_then(|t| t.clone());

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: ts,
                });
                pass.set_pipeline(&self.local_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(self.config.local_workgroups, 1, 1);
            }
        }

        // Copy-back if the final result landed in the scratch buffer
        if self.config.result_in_scratch {
            let size = (self.config.n as u64) * std::mem::size_of::<u32>() as u64;
            encoder.copy_buffer_to_buffer(&self.scratch_buffer, 0, &buf.inner, 0, size);
        }
    }

    /// Encode the inverse scaling dispatch if this is an inverse plan.
    fn encode_scale(
        &self,
        wgpu_device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        buf: &WgpuBuffer<BabyBear>,
        ts: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        let Some(ref param_buf) = self.scale_param_buffer else {
            return;
        };

        let bind_group = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.scale_bind_group_layout,
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
