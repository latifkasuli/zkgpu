//! Canary compute dispatch — verifies the GPU driver can execute
//! the actual NTT kernel shader and produce correct results.
//!
//! The canary runs two validation passes at device init:
//!
//! 1. **Default kernel** — the kernel selected by `PlannerPolicy::from_caps`
//!    (Auto). Dispatched **twice** in separate submissions to catch broken
//!    Vulkan drivers (e.g. PowerVR Rogue) that crash on the second
//!    `vkQueueSubmit`.
//!
//! 2. **Subgroup SPIR-V kernel** (when hardware supports it but the Auto
//!    path didn't select it). This covers the `ForceSubgroup` opt-in path
//!    so callers who explicitly request subgroup acceleration already have
//!    canary coverage. Dispatched once (double-dispatch is for crash
//!    detection, not relevant for the subgroup path on healthy Vulkan
//!    drivers).
//!
//! After each pass, the GPU output is compared element-by-element against
//! a CPU reference NTT. This catches silent arithmetic corruption from
//! miscompiled shaders or driver bugs.
//!
//! Runs once during device init. On healthy hardware this adds <4 ms.
//! Before dispatching, the canary calls `is_gpu_usable()` to check for
//! known-broken drivers (e.g. PowerVR Rogue/Volcanic) and returns a clean
//! `GpuComputeUnsupported` error instead of letting the driver SIGSEGV.

use wgpu::util::DeviceExt;
use zkgpu_babybear::BabyBear;
use zkgpu_core::{GpuField, NttDirection, ZkGpuError};
use zkgpu_ntt::ntt_cpu_reference;

use crate::async_util::map_buffer_read;
use crate::caps::CapabilityProfile;
use crate::ntt::PlannerPolicy;
use crate::ntt::local_kernel::{resolve_local_kernel, ResolvedLocalKernel};
use crate::ntt::twiddles::{precompute_local_r4_twiddles, precompute_subgroup_local_twiddles};

/// Portable local radix-4 DIF NTT shader.
const NTT_LOCAL_R4_WGSL: &str =
    include_str!("kernels/portable/babybear_stockham_local_r4.wgsl");

/// Pre-compiled SPIR-V for the subgroup-accelerated DIT local kernel.
#[cfg(feature = "subgroup-vulkan-spirv")]
const NTT_LOCAL_SUBGROUP_SPIRV: &[u8] =
    include_bytes!("kernels/native/babybear_stockham_local_subgroup.spv");

/// Number of dispatches for the default kernel canary. Two submissions
/// catch drivers that corrupt state after the first `vkQueueSubmit`.
const DEFAULT_CANARY_DISPATCHES: usize = 2;

/// Canary test size: a single 1024-element local NTT block.
const CANARY_N: u32 = 1024;

/// Run canary compute dispatches at device init.
///
/// Validates **both** the default kernel path (selected by `Auto` policy)
/// and, when hardware supports it, the subgroup SPIR-V fast path. This
/// ensures that callers using `ForceSubgroup` later get the same
/// init-time coverage as the default path.
///
/// Returns `Ok(())` on healthy drivers, or
/// `Err(ZkGpuError::GpuComputeUnsupported)` if either kernel produces
/// incorrect results. On truly broken drivers (segfault), this crashes
/// at init time rather than mid-workload.
pub(crate) async fn canary_dispatch(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline_cache: Option<&wgpu::PipelineCache>,
    caps: &CapabilityProfile,
) -> Result<(), ZkGpuError> {
    // --- Pre-flight: refuse to dispatch on known-broken drivers ---
    crate::caps::is_gpu_usable(caps)?;

    // Shared bind group layout — both local kernels use the same 5 bindings.
    let bind_group_layout = create_canary_bgl(device);
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("zkgpu canary layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    // ---- Pass 1: Default (Auto) kernel validation ----
    //
    // Uses the same resolver as stockham/build.rs so platform overrides
    // (e.g. browser ForcePortable) are respected.
    let policy = PlannerPolicy::from_caps(caps);
    let (resolved, _reason) = resolve_local_kernel(policy.local_kernel_hint(), caps)
        .map_err(|e| ZkGpuError::GpuComputeUnsupported(format!(
            "canary local kernel resolution failed: {e}"
        )))?;
    let default_is_subgroup = resolved == ResolvedLocalKernel::SubgroupSpirV;

    let shader_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let (default_module, default_entry) = if default_is_subgroup {
        #[cfg(feature = "subgroup-vulkan-spirv")]
        {
            let spirv_words = crate::pipeline_registry::spirv_bytes_to_words(NTT_LOCAL_SUBGROUP_SPIRV);
            let m = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("zkgpu canary ntt local (SPIR-V)"),
                source: wgpu::ShaderSource::SpirV(spirv_words.into()),
            });
            (m, "main")
        }
        #[cfg(not(feature = "subgroup-vulkan-spirv"))]
        unreachable!("SubgroupSpirV requires the subgroup-vulkan-spirv cargo feature")
    } else {
        let m = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("zkgpu canary ntt local"),
            source: wgpu::ShaderSource::Wgsl(NTT_LOCAL_R4_WGSL.into()),
        });
        (m, "stockham_local_r4")
    };
    if let Some(err) = shader_scope.pop().await {
        return Err(ZkGpuError::GpuComputeUnsupported(format!(
            "canary NTT shader compilation error: {err}"
        )));
    }

    let default_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("zkgpu canary pipeline"),
        layout: Some(&pipeline_layout),
        module: &default_module,
        entry_point: Some(default_entry),
        compilation_options: Default::default(),
        cache: pipeline_cache,
    });

    let (tw, tw_p, o4, o4p) = if default_is_subgroup {
        precompute_subgroup_local_twiddles(NttDirection::Forward)
    } else {
        precompute_local_r4_twiddles(NttDirection::Forward)
    };
    let subgroup_log = if default_is_subgroup {
        caps.min_subgroup_size.trailing_zeros()
    } else {
        0
    };
    let kernel_name = if default_is_subgroup { "subgroup SPIR-V" } else { "portable R4" };

    run_canary_dispatches(
        device, queue, &bind_group_layout, &default_pipeline,
        &tw, &tw_p, o4, o4p, subgroup_log,
        DEFAULT_CANARY_DISPATCHES, kernel_name,
    ).await?;

    // ---- Pass 2: Subgroup SPIR-V validation (when hw supports but Auto didn't select) ----
    //
    // If the Auto policy already selected SubgroupSpirV (Pass 1 just
    // validated it), skip. Otherwise, when the hardware *could* run
    // the subgroup shader, validate it here so ForceSubgroup callers
    // have init-time coverage.
    #[cfg(feature = "subgroup-vulkan-spirv")]
    if !default_is_subgroup
        && caps.backend == wgpu::Backend::Vulkan
        && caps.has_subgroup
        && caps.min_subgroup_size >= 32
    {
        let spirv_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let spirv_words = crate::pipeline_registry::spirv_bytes_to_words(NTT_LOCAL_SUBGROUP_SPIRV);
        let spirv_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("zkgpu canary subgroup spirv"),
            source: wgpu::ShaderSource::SpirV(spirv_words.into()),
        });
        if let Some(err) = spirv_scope.pop().await {
            return Err(ZkGpuError::GpuComputeUnsupported(format!(
                "canary subgroup SPIR-V shader compilation error: {err}"
            )));
        }

        let spirv_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("zkgpu canary subgroup pipeline"),
            layout: Some(&pipeline_layout),
            module: &spirv_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: pipeline_cache,
        });

        let (tw_sg, tw_p_sg, o4_sg, o4p_sg) =
            precompute_subgroup_local_twiddles(NttDirection::Forward);
        let sg_log = caps.min_subgroup_size.trailing_zeros();

        run_canary_dispatches(
            device, queue, &bind_group_layout, &spirv_pipeline,
            &tw_sg, &tw_p_sg, o4_sg, o4p_sg, sg_log,
            1, "subgroup SPIR-V (pre-validation)",
        ).await?;
    }

    Ok(())
}

/// Create the 5-binding bind group layout shared by all local NTT kernels.
fn create_canary_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("zkgpu canary bgl"),
        entries: &[
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
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
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
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

/// Dispatch the given canary pipeline `dispatch_count` times, read back
/// the final output, and verify against the CPU reference NTT.
///
/// This is the shared core of both the default-kernel and subgroup-SPIR-V
/// canary passes. Each pass provides its own pipeline, twiddles, and params.
async fn run_canary_dispatches(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bind_group_layout: &wgpu::BindGroupLayout,
    pipeline: &wgpu::ComputePipeline,
    twiddle_data: &[u32],
    twiddle_prime_data: &[u32],
    omega4: u32,
    omega4_prime: u32,
    subgroup_log: u32,
    dispatch_count: usize,
    kernel_name: &str,
) -> Result<(), ZkGpuError> {
    let buf_size = (CANARY_N as u64) * 4;

    let twiddle_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("zkgpu canary tw"),
        contents: bytemuck::cast_slice(twiddle_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let params_data: [u32; 4] = [1, omega4, omega4_prime, subgroup_log];
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("zkgpu canary params"),
        contents: bytemuck::cast_slice(&params_data),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let twiddle_prime_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("zkgpu canary tw_prime"),
        contents: bytemuck::cast_slice(twiddle_prime_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("zkgpu canary staging"),
        size: buf_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let src_data: Vec<u32> = (0..CANARY_N).collect();

    // --- Dispatch loop ---
    //
    // Some Vulkan drivers (PowerVR Rogue GE8322) survive one dispatch
    // but crash with SIGSEGV on the second vkQueueSubmit with the same
    // pipeline. The default kernel pass dispatches twice to catch this.
    // The subgroup pass dispatches once (crash detection is less relevant
    // for the SPIR-V path on healthy Vulkan drivers).
    for round in 0..dispatch_count {
        let src_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("zkgpu canary src"),
            contents: bytemuck::cast_slice(&src_data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let dst_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("zkgpu canary dst"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("zkgpu canary bg"),
            layout: bind_group_layout,
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
                    resource: twiddle_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: twiddle_prime_buf.as_entire_binding(),
                },
            ],
        });

        let dispatch_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("zkgpu canary encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("zkgpu canary pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        if round == dispatch_count - 1 {
            encoder.copy_buffer_to_buffer(&dst_buf, 0, &staging_buf, 0, buf_size);
        }
        queue.submit(std::iter::once(encoder.finish()));

        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| ZkGpuError::BackendError(Box::new(e)))?;

        if let Some(err) = dispatch_scope.pop().await {
            return Err(ZkGpuError::GpuComputeUnsupported(format!(
                "canary NTT ({kernel_name}) dispatch {round} validation error: {err}"
            )));
        }
    }

    // --- Read back and verify against CPU reference ---
    let slice = staging_buf.slice(..);
    map_buffer_read(device, slice).await?;

    let data = slice.get_mapped_range();
    let gpu_result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging_buf.unmap();

    let mut expected: Vec<BabyBear> = (0..CANARY_N).map(BabyBear::new).collect();
    ntt_cpu_reference(&mut expected, NttDirection::Forward);
    let expected_u32: Vec<u32> = expected.iter().map(|f| f.to_repr()).collect();

    let mut mismatch_count = 0usize;
    let mut first_mismatch: Option<(usize, u32, u32)> = None;
    for (i, (&got, &want)) in gpu_result.iter().zip(expected_u32.iter()).enumerate() {
        if got != want {
            mismatch_count += 1;
            if first_mismatch.is_none() {
                first_mismatch = Some((i, got, want));
            }
        }
    }

    if mismatch_count > 0 {
        let (idx, got, want) = first_mismatch.unwrap();
        return Err(ZkGpuError::GpuComputeUnsupported(format!(
            "canary NTT ({kernel_name}): {mismatch_count}/1024 elements wrong; \
             first mismatch at [{idx}]: GPU={got}, CPU={want} — \
             GPU driver produces incorrect modular arithmetic"
        )));
    }

    Ok(())
}
