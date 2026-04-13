//! Canary compute dispatch — verifies the GPU driver can execute
//! the actual NTT kernel shader and produce correct results.
//!
//! The canary compiles the same local NTT shader the runtime would
//! select for this device (portable R4 or subgroup-accelerated DIT)
//! and dispatches a 1024-element forward NTT with real twiddle factors
//! **twice**. The double dispatch catches broken Vulkan drivers (e.g.
//! PowerVR Rogue on budget Unisoc hardware) that survive a single
//! `vkQueueSubmit` but crash with SIGSEGV on the second submission.
//!
//! After the dispatches, the GPU output is compared element-by-element
//! against a CPU reference NTT. This catches silent arithmetic
//! corruption from miscompiled shaders or driver bugs that produce
//! plausible but wrong modular-arithmetic results.
//!
//! Runs once during device init. On healthy hardware this adds <2 ms.
//! Before dispatching, the canary calls `is_gpu_usable()` to check for
//! known-broken drivers (e.g. PowerVR Rogue/Volcanic) and returns a clean
//! `GpuComputeUnsupported` error instead of letting the driver SIGSEGV.

use wgpu::util::DeviceExt;
use zkgpu_babybear::BabyBear;
use zkgpu_core::{GpuField, NttDirection, ZkGpuError};
use zkgpu_ntt::ntt_cpu_reference;

use crate::async_util::map_buffer_read;
use crate::caps::CapabilityProfile;
use crate::ntt::{LocalKernelHint, PlannerPolicy};
use crate::ntt::twiddles::{precompute_local_r4_twiddles, precompute_subgroup_local_twiddles};

/// Portable local radix-4 DIF NTT shader.
const NTT_LOCAL_R4_WGSL: &str =
    include_str!("kernels/portable/babybear_stockham_local_r4.wgsl");

/// Subgroup-accelerated local DIT NTT shader.
const NTT_LOCAL_SUBGROUP_WGSL: &str =
    include_str!("kernels/native/babybear_stockham_local_subgroup.wgsl");

/// Number of canary dispatches. Two submissions are needed to catch
/// drivers that corrupt internal state after the first `vkQueueSubmit`.
const CANARY_DISPATCHES: usize = 2;

/// Run canary compute dispatches using the real NTT shader with
/// real root-of-unity twiddle factors and a 1024-element transform.
///
/// The shader is selected using the same `PlannerPolicy` as the
/// runtime — including platform overrides like `ForcePortable` for
/// browser WebGPU — so the canary always exercises the exact local
/// kernel the runtime would pick.
///
/// Dispatches the shader **twice** in separate queue submissions to
/// catch drivers that crash on repeated `vkQueueSubmit` calls with
/// the same pipeline (e.g. PowerVR Rogue GE8322).
///
/// After the final dispatch, the GPU output is compared element-by-
/// element against a CPU reference NTT. This catches silent compute
/// corruption, not just crashes.
///
/// Returns `Ok(())` on healthy drivers, or
/// `Err(ZkGpuError::GpuComputeUnsupported)` if the result is wrong.
/// On truly broken drivers (segfault), this crashes at init time rather
/// than mid-workload — a better failure mode.
pub(crate) async fn canary_dispatch(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline_cache: Option<&wgpu::PipelineCache>,
    caps: &CapabilityProfile,
) -> Result<(), ZkGpuError> {
    // --- Pre-flight: refuse to dispatch on known-broken drivers ---
    //
    // On Android, SIGSEGV kills the process with no recovery. Check
    // driver quirks *before* touching the GPU to return a clean error.
    crate::caps::is_gpu_usable(caps)?;

    // --- Select the same local kernel the runtime would pick ---
    //
    // Derive from PlannerPolicy so platform overrides (e.g. browser's
    // ForcePortable) are respected — not just raw subgroup capability.
    let policy = PlannerPolicy::from_caps(caps);
    let use_subgroup = match policy.local_kernel_hint() {
        LocalKernelHint::ForceSubgroup => true,
        LocalKernelHint::ForcePortable => false,
        LocalKernelHint::Auto => caps.has_subgroup && caps.min_subgroup_size >= 32,
    };

    let (shader_source, entry_point) = if use_subgroup {
        (NTT_LOCAL_SUBGROUP_WGSL, "stockham_local_subgroup")
    } else {
        (NTT_LOCAL_R4_WGSL, "stockham_local_r4")
    };

    // --- Compile the shader ---
    let shader_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("zkgpu canary ntt local"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });
    if let Some(err) = shader_scope.pop().await {
        return Err(ZkGpuError::GpuComputeUnsupported(format!(
            "canary NTT shader compilation error: {err}"
        )));
    }

    // Bind group layout matches both local kernels (same 5 bindings):
    // binding(0): src (read-only storage)
    // binding(1): dst (read-write storage)
    // binding(2): twiddles (read-only storage)
    // binding(3): params (uniform) — { stride, omega4, omega4_prime, subgroup_log/_pad0 }
    // binding(4): twiddles_prime (read-only storage)
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("zkgpu canary layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    // Use the same pipeline cache as the NTT path — on some Vulkan
    // drivers the cache blob influences shader compilation behaviour.
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("zkgpu canary pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: pipeline_cache,
    });

    // --- Prepare test data for a 1024-element local NTT ---
    //
    // Uses REAL root-of-unity twiddle factors (forward direction) —
    // the same values the runtime produces for a real NTT dispatch.
    const N: u32 = 1024;

    let (twiddle_data, twiddle_prime_data, omega4, omega4_prime) = if use_subgroup {
        precompute_subgroup_local_twiddles(NttDirection::Forward)
    } else {
        precompute_local_r4_twiddles(NttDirection::Forward)
    };

    let src_data: Vec<u32> = (0..N).collect();

    // Params layout: { stride, omega4, omega4_prime, subgroup_log/_pad0 }
    let params_w3 = if use_subgroup {
        caps.min_subgroup_size.trailing_zeros()
    } else {
        0
    };
    let params_data: [u32; 4] = [1, omega4, omega4_prime, params_w3];

    let buf_size = (N as u64) * 4;

    let twiddle_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("zkgpu canary tw"),
        contents: bytemuck::cast_slice(&twiddle_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("zkgpu canary params"),
        contents: bytemuck::cast_slice(&params_data),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let twiddle_prime_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("zkgpu canary tw_prime"),
        contents: bytemuck::cast_slice(&twiddle_prime_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("zkgpu canary staging"),
        size: buf_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // --- Dispatch CANARY_DISPATCHES times in separate submissions ---
    //
    // Some Vulkan drivers (PowerVR Rogue GE8322) survive one dispatch
    // but crash with SIGSEGV on the second vkQueueSubmit with the same
    // pipeline. By dispatching twice, we catch these stateful driver bugs.
    for round in 0..CANARY_DISPATCHES {
        // Fresh src/dst buffers each round, matching the NTT's pattern
        // of uploading fresh data each execution.
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
            layout: &bind_group_layout,
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
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        // On the final round, copy to staging for readback verification.
        if round == CANARY_DISPATCHES - 1 {
            encoder.copy_buffer_to_buffer(&dst_buf, 0, &staging_buf, 0, buf_size);
        }
        queue.submit(std::iter::once(encoder.finish()));

        // Synchronise with the GPU — matches the NTT's device.poll(Wait).
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| ZkGpuError::BackendError(Box::new(e)))?;

        if let Some(err) = dispatch_scope.pop().await {
            return Err(ZkGpuError::GpuComputeUnsupported(format!(
                "canary NTT dispatch {round} validation error: {err}"
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

    // Compute the exact expected output using the CPU reference NTT.
    let mut expected: Vec<BabyBear> = (0..N).map(BabyBear::new).collect();
    ntt_cpu_reference(&mut expected, NttDirection::Forward);
    let expected_u32: Vec<u32> = expected.iter().map(|f| f.to_repr()).collect();

    // Element-by-element comparison — catches silent arithmetic corruption.
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
        let kernel = if use_subgroup { "subgroup" } else { "portable R4" };
        return Err(ZkGpuError::GpuComputeUnsupported(format!(
            "canary NTT ({kernel}): {mismatch_count}/1024 elements wrong; \
             first mismatch at [{idx}]: GPU={got}, CPU={want} — \
             GPU driver produces incorrect modular arithmetic"
        )));
    }

    Ok(())
}
