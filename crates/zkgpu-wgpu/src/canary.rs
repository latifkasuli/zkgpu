//! Canary compute dispatch — verifies the GPU driver can execute
//! the actual NTT kernel shader and produce correct results.
//!
//! The canary runs the portable radix-4 local Stockham kernel **twice**
//! in separate submissions to catch broken Vulkan drivers (e.g. PowerVR
//! Rogue) that crash on the second `vkQueueSubmit`.
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
use crate::ntt::twiddles::precompute_local_r4_twiddles;

/// Portable local radix-4 DIF NTT shader.
const NTT_LOCAL_R4_WGSL: &str =
    include_str!("kernels/portable/babybear_stockham_local_r4.wgsl");

/// Number of dispatches for the canary. Two submissions catch drivers
/// that corrupt state after the first `vkQueueSubmit`.
const CANARY_DISPATCHES: usize = 2;

/// Canary test size: a single 1024-element local NTT block.
const CANARY_N: u32 = 1024;

/// Run canary compute dispatches at device init.
///
/// Returns `Ok(())` on healthy drivers, or
/// `Err(ZkGpuError::GpuComputeUnsupported)` if the kernel produces
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

    let bind_group_layout = create_canary_bgl(device);
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("zkgpu canary layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    let shader_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("zkgpu canary ntt local"),
        source: wgpu::ShaderSource::Wgsl(NTT_LOCAL_R4_WGSL.into()),
    });
    if let Some(err) = shader_scope.pop().await {
        return Err(ZkGpuError::GpuComputeUnsupported(format!(
            "canary NTT shader compilation error: {err}"
        )));
    }

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("zkgpu canary pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("stockham_local_r4"),
        compilation_options: Default::default(),
        cache: pipeline_cache,
    });

    let (tw, tw_p, o4, o4p) = precompute_local_r4_twiddles(NttDirection::Forward);

    run_canary_dispatches(
        device, queue, &bind_group_layout, &pipeline,
        &tw, &tw_p, o4, o4p,
        CANARY_DISPATCHES, "portable R4",
    ).await?;

    Ok(())
}

/// Create the 5-binding bind group layout for the local NTT kernel.
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
async fn run_canary_dispatches(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bind_group_layout: &wgpu::BindGroupLayout,
    pipeline: &wgpu::ComputePipeline,
    twiddle_data: &[u32],
    twiddle_prime_data: &[u32],
    omega4: u32,
    omega4_prime: u32,
    dispatch_count: usize,
    kernel_name: &str,
) -> Result<(), ZkGpuError> {
    let buf_size = (CANARY_N as u64) * 4;

    let twiddle_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("zkgpu canary tw"),
        contents: bytemuck::cast_slice(twiddle_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let params_data: [u32; 4] = [1, omega4, omega4_prime, 0];
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
    // pipeline. Dispatching twice catches this.
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
