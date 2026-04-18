//! GPU differential test for the portable u32x2 Goldilocks arithmetic
//! primitives. Dispatches the `gl_test_arith` kernel in
//! `kernels/portable/goldilocks_arith.wgsl` on a set of Goldilocks
//! value pairs, reads the results back, and compares them against the
//! CPU-side [`zkgpu_goldilocks::Goldilocks`] implementation.
//!
//! Phase B.1 deliverable. Proves that the u32x2 limb arithmetic + the
//! 128-bit reduction in the WGSL library compute the same values as
//! the Rust reference on real hardware. Phase B.2's Stockham kernels
//! inline the same primitive helpers — if anything here breaks, the
//! NTT will too, and this test will fire first with a much smaller
//! failure surface.

use zkgpu_core::{GpuDevice, GpuField, ZkGpuError};

use crate::device::WgpuDevice;
use crate::pipeline_registry::PipelineRegistry;

/// WGSL source for the Goldilocks arithmetic primitives + test kernel.
///
/// Kept as `pub(crate)` so Phase B.2's Stockham kernels can re-include
/// the same source via string concatenation. (WGSL has no preprocessor,
/// so we own one canonical copy here.)
pub(crate) const GOLDILOCKS_ARITH_WGSL: &str =
    include_str!("../../kernels/portable/goldilocks_arith.wgsl");

const KERNEL_ENTRY_POINT: &str = "gl_test_arith";
const BIND_GROUP_LAYOUT_LABEL: &str = "Goldilocks arith test BGL";
const PIPELINE_LABEL: &str = "Goldilocks arith test pipeline";
const WORKGROUP_SIZE: u32 = 64;

/// Result of dispatching the test kernel on a pair of input vectors.
///
/// The three output vectors are elementwise `a op b` for each
/// operation. Guaranteed same length as the inputs.
#[derive(Debug, Clone)]
#[allow(dead_code)] // Used by tests + future differential harness.
pub(crate) struct GoldilocksArithResults<F: GpuField> {
    pub add: Vec<F>,
    pub sub: Vec<F>,
    pub mul: Vec<F>,
}

/// Run the portable-Goldilocks arithmetic primitives on the GPU and
/// return elementwise `a + b`, `a - b`, `a * b`.
///
/// `F` must be a 64-bit `GpuField` whose in-memory byte representation
/// matches the WGSL `vec2<u32>` limb layout — namely `Goldilocks`
/// (`Repr = u64`, little-endian, 8 bytes per element). Callers pass
/// `F = Goldilocks` at use sites; the generic signature exists so
/// the integration test in Phase B.2 can reuse this path with the
/// same field type without importing `zkgpu-goldilocks` into the
/// runtime crate's non-test surface.
#[allow(dead_code)] // Exercised by the #[cfg(test)] module below.
pub(crate) fn run_goldilocks_arith_gpu<F: GpuField<Repr = u64>>(
    device: &WgpuDevice,
    a: &[F],
    b: &[F],
) -> Result<GoldilocksArithResults<F>, ZkGpuError> {
    assert_eq!(
        a.len(),
        b.len(),
        "arith test requires equal-length input vectors"
    );
    let n = a.len();
    if n == 0 {
        return Ok(GoldilocksArithResults {
            add: vec![],
            sub: vec![],
            mul: vec![],
        });
    }

    dispatch_and_read(device, a, b, KERNEL_ENTRY_POINT)
}

/// Shared dispatch helper — runs either the real `gl_test_arith`
/// entry point or the `gl_test_sentinel` diagnostic entry point.
/// Returns (add, sub, mul) regions of the packed output buffer.
fn dispatch_and_read<F: GpuField<Repr = u64>>(
    device: &WgpuDevice,
    a: &[F],
    b: &[F],
    entry_point: &'static str,
) -> Result<GoldilocksArithResults<F>, ZkGpuError> {
    let n = a.len();

    // WebGPU baseline caps `max_storage_buffers_per_shader_stage` at 4,
    // so we pack all three results into a single 3N-length output
    // buffer laid out as [add | sub | mul]. One uniform carries `n` so
    // the shader can index each region without a second arrayLength()
    // call.
    let a_buf = device.upload::<F>(a)?;
    let b_buf = device.upload::<F>(b)?;
    let out_buf = device.alloc_zeros::<F>(3 * n)?;

    // Build a 16-byte uniform — four u32s, first is `n`, rest padding
    // for WGSL's 16-byte minimum uniform-struct alignment.
    let params_bytes: [u32; 4] = [n as u32, 0, 0, 0];
    let params_buf = device.raw_device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("goldilocks arith params"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    device
        .raw_queue()
        .write_buffer(&params_buf, 0, bytemuck::cast_slice(&params_bytes));

    let registry = device.pipeline_registry();
    let module = registry.get_or_create_module(
        device.raw_device(),
        GOLDILOCKS_ARITH_WGSL,
        "goldilocks_arith.wgsl",
    );
    let bgl = registry.get_or_create_bgl(
        device.raw_device(),
        BIND_GROUP_LAYOUT_LABEL,
        &[
            bind_storage_entry(0, /* read_only = */ true),
            bind_storage_entry(1, /* read_only = */ true),
            bind_storage_entry(2, /* read_only = */ false),
            bind_uniform_entry(3),
        ],
    );
    let pipeline_layout =
        device
            .raw_device()
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(PIPELINE_LABEL),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });
    let pipeline = registry.get_or_create_pipeline(
        device.raw_device(),
        GOLDILOCKS_ARITH_WGSL,
        entry_point,
        BIND_GROUP_LAYOUT_LABEL,
        &pipeline_layout,
        &module,
        /* cache = */ None,
    );

    let bind_group = device
        .raw_device()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("goldilocks arith test bind group"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buf.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buf.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buf.inner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

    let num_groups = (n as u32).div_ceil(WORKGROUP_SIZE);
    let mut encoder = device.raw_device().create_command_encoder(
        &wgpu::CommandEncoderDescriptor {
            label: Some("goldilocks arith test encoder"),
        },
    );
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("goldilocks arith test pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(num_groups, 1, 1);
    }
    device.raw_queue().submit(Some(encoder.finish()));

    // Force GPU work to complete before readback. On UMA targets with
    // `MAPPABLE_PRIMARY_BUFFERS`, the direct-map readback path can
    // otherwise observe pre-dispatch zeros because the map doesn't
    // wait for the submit fence.
    device
        .raw_device()
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| ZkGpuError::BackendError(Box::new(e)))?;

    // Read back the packed buffer as one Vec<F> of length 3*n, then
    // split into three regions.
    let flat = out_buf.read_to_vec_blocking()?;
    assert_eq!(flat.len(), 3 * n);
    let mut iter = flat.into_iter();
    let add: Vec<F> = iter.by_ref().take(n).collect();
    let sub: Vec<F> = iter.by_ref().take(n).collect();
    let mul: Vec<F> = iter.by_ref().take(n).collect();

    Ok(GoldilocksArithResults { add, sub, mul })
}

/// Helper: build a storage-buffer `BindGroupLayoutEntry` for the
/// goldilocks arith test, at a given binding index.
fn bind_storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
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

/// Helper: uniform-buffer `BindGroupLayoutEntry`.
fn bind_uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

// Accessors on `WgpuDevice` / `WgpuBuffer` that the test needs.
// These are `pub(crate)` getters with narrow scope; they exist so the
// Goldilocks arith path can reach the underlying `wgpu` device and
// registry without leaking them to the wider crate API.

impl crate::device::WgpuDevice {
    pub(crate) fn raw_device(&self) -> &wgpu::Device {
        &self.device
    }
    pub(crate) fn raw_queue(&self) -> &wgpu::Queue {
        &self.queue
    }
    pub(crate) fn pipeline_registry(&self) -> &PipelineRegistry {
        &self.pipelines
    }
}

impl<F: GpuField> crate::buffer::WgpuBuffer<F> {
    /// Blocking readback. Tests drive this path to avoid threading an
    /// async runtime into the assertion logic.
    pub(crate) fn read_to_vec_blocking(&self) -> Result<Vec<F>, ZkGpuError> {
        <Self as zkgpu_core::GpuBuffer<F>>::read_to_vec(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkgpu_goldilocks::{Goldilocks, P};

    /// Build a pseudo-random Goldilocks test vector. SplitMix64 for
    /// reproducibility — the same seed always produces the same
    /// inputs, so any test failure is bit-debuggable.
    fn build_inputs(seed: u64, n: usize) -> Vec<Goldilocks> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                Goldilocks::new(z ^ (z >> 31))
            })
            .collect()
    }

    fn try_device() -> Option<WgpuDevice> {
        // `WgpuDevice::new` is already blocking on native targets (wraps
        // pollster internally). Skip the test rather than fail it when
        // no GPU adapter is reachable — keeps the suite tractable on
        // machines without GPU access.
        WgpuDevice::new().ok()
    }

    /// Diagnostic — confirms dispatch plumbing is live before we test
    /// arithmetic. The `gl_test_sentinel` kernel writes a fixed pattern
    /// regardless of inputs. If this fails, the problem is in the
    /// Rust-side dispatch, not the WGSL arithmetic.
    #[test]
    fn goldilocks_gpu_dispatch_plumbing_sentinel() {
        let Some(device) = try_device() else {
            eprintln!("skipping sentinel test: no GPU adapter available");
            return;
        };
        let n = 8usize;
        let a: Vec<Goldilocks> = (0..n as u64).map(Goldilocks::new).collect();
        let b: Vec<Goldilocks> = (0..n as u64).map(Goldilocks::new).collect();
        let results =
            super::dispatch_and_read::<Goldilocks>(&device, &a, &b, "gl_test_sentinel").unwrap();

        // Expected sentinel: add = (0xDEADBEEF, 0xCAFEBABE) as vec2<u32>,
        // which is u64 = lo | (hi << 32) = 0xCAFEBABE_DEADBEEF.
        let expected_add = 0xCAFEBABE_DEADBEEFu64;
        for (i, v) in results.add.iter().enumerate() {
            assert_eq!(v.0, expected_add, "sentinel add[{i}] = {:#x}", v.0);
        }
        let expected_sub = 0x22222222_11111111u64;
        for (i, v) in results.sub.iter().enumerate() {
            assert_eq!(v.0, expected_sub, "sentinel sub[{i}] = {:#x}", v.0);
        }
        // mul[i] = (i, i) as vec2<u32> = i | (i << 32)
        for (i, v) in results.mul.iter().enumerate() {
            let want = (i as u64) | ((i as u64) << 32);
            assert_eq!(v.0, want, "sentinel mul[{i}] = {:#x}", v.0);
        }
    }

    #[test]
    fn goldilocks_gpu_arith_primitives_match_cpu() {
        let Some(device) = try_device() else {
            eprintln!("skipping goldilocks_gpu_arith: no GPU adapter available");
            return;
        };

        let a = build_inputs(0xA1B2_C3D4_E5F6_0718, 256);
        let b = build_inputs(0x0123_4567_89AB_CDEF, 256);
        let results = run_goldilocks_arith_gpu::<Goldilocks>(&device, &a, &b)
            .expect("dispatch should succeed");

        assert_eq!(results.add.len(), a.len());
        assert_eq!(results.sub.len(), a.len());
        assert_eq!(results.mul.len(), a.len());

        for i in 0..a.len() {
            assert_eq!(
                results.add[i],
                a[i] + b[i],
                "add mismatch at i={i}: a={}, b={}",
                a[i].0,
                b[i].0
            );
            assert_eq!(
                results.sub[i],
                a[i] - b[i],
                "sub mismatch at i={i}: a={}, b={}",
                a[i].0,
                b[i].0
            );
            assert_eq!(
                results.mul[i],
                a[i] * b[i],
                "mul mismatch at i={i}: a={}, b={}",
                a[i].0,
                b[i].0
            );
        }
    }

    #[test]
    fn goldilocks_gpu_arith_boundary_values() {
        // The random test above covers the broad middle of the input
        // space; this one hand-picks the adversarial corners that hit
        // every wrap-correction branch in the WGSL:
        //
        // - (p - 1) + 1                → add overflow, carry + final reduce
        // - 0 - 1                      → sub underflow, borrow correction
        // - (p - 1) * (p - 1) = 1      → reduction boundary
        // - p / 2 + something          → mid-range
        // - max u32x2 limb values      → stress the mul limb decomposition
        let Some(device) = try_device() else {
            eprintln!("skipping goldilocks_gpu_arith_boundary: no GPU adapter");
            return;
        };

        let a_raw = [
            P - 1,
            0,
            P - 1,
            P / 2,
            0x1234_5678_9ABC_DEF0,
            P - 2,
            1,
            0xFFFF_FFFF_0000_0000,  // high-limb-only
            0x0000_0000_FFFF_FFFF,  // low-limb-only
            P - 1,
        ];
        let b_raw = [
            1,
            1,
            P - 1,
            P / 2 + 1,
            0xFEDC_BA98_7654_3210,
            2,
            P - 1,
            0xFFFF_FFFF,
            0xFFFF_FFFF_0000_0000,
            P - 1,
        ];
        let a: Vec<Goldilocks> = a_raw.iter().map(|&v| Goldilocks::new(v)).collect();
        let b: Vec<Goldilocks> = b_raw.iter().map(|&v| Goldilocks::new(v)).collect();

        let results = run_goldilocks_arith_gpu::<Goldilocks>(&device, &a, &b)
            .expect("dispatch should succeed");

        for i in 0..a.len() {
            assert_eq!(results.add[i], a[i] + b[i], "add boundary mismatch at {i}");
            assert_eq!(results.sub[i], a[i] - b[i], "sub boundary mismatch at {i}");
            assert_eq!(results.mul[i], a[i] * b[i], "mul boundary mismatch at {i}");
        }
    }
}
