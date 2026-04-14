use zkgpu_babybear::BabyBear;
use zkgpu_core::{GpuBuffer, GpuDevice, NttDirection, NttPlan, ZkGpuError};
use zkgpu_ntt::ntt_cpu_reference;
use zkgpu_wgpu::{PlannerPolicy, WgpuDevice, WgpuNttPlan};

fn init_device() -> WgpuDevice {
    WgpuDevice::new().expect("GPU device required for integration tests")
}

fn assert_gpu_matches_cpu(log_n: u32, direction: NttDirection, input: &[BabyBear]) {
    let device = init_device();
    let mut plan = WgpuNttPlan::new(&device, log_n, direction).expect("failed to create plan");

    let mut buf = device.upload(input).expect("upload failed");
    plan.execute(&device, &mut buf).expect("GPU NTT failed");
    let gpu_result = buf.read_to_vec().expect("readback failed");

    let mut cpu_result = input.to_vec();
    ntt_cpu_reference(&mut cpu_result, direction);

    for (i, (g, c)) in gpu_result.iter().zip(cpu_result.iter()).enumerate() {
        assert_eq!(
            g, c,
            "mismatch at idx={i}, log_n={log_n}, direction={direction:?}: GPU={g}, CPU={c}"
        );
    }
}

// --- Forward: global-only path (N < 512) ---

#[test]
fn forward_sequential_log4() {
    let n = 1 << 4;
    let data: Vec<BabyBear> = (0..n as u32).map(BabyBear::new).collect();
    assert_gpu_matches_cpu(4, NttDirection::Forward, &data);
}

#[test]
fn forward_sequential_log8() {
    let n = 1 << 8;
    let data: Vec<BabyBear> = (0..n as u32).map(BabyBear::new).collect();
    assert_gpu_matches_cpu(8, NttDirection::Forward, &data);
}

// --- Forward: local kernel path, even global stages (no copy-back) ---

#[test]
fn forward_sequential_log10() {
    let n = 1 << 10;
    let data: Vec<BabyBear> = (0..n as u32).map(BabyBear::new).collect();
    assert_gpu_matches_cpu(10, NttDirection::Forward, &data);
}

#[test]
fn forward_sequential_log14() {
    let n = 1 << 14;
    let data: Vec<BabyBear> = (0..n as u32).map(BabyBear::new).collect();
    assert_gpu_matches_cpu(14, NttDirection::Forward, &data);
}

// --- Forward: local kernel path, odd total swaps (exercises copy-back) ---

#[test]
fn forward_sequential_log9() {
    let n = 1 << 9;
    let data: Vec<BabyBear> = (0..n as u32).map(BabyBear::new).collect();
    assert_gpu_matches_cpu(9, NttDirection::Forward, &data);
}

#[test]
fn forward_sequential_log11() {
    let n = 1 << 11;
    let data: Vec<BabyBear> = (0..n as u32).map(BabyBear::new).collect();
    assert_gpu_matches_cpu(11, NttDirection::Forward, &data);
}

#[test]
fn forward_sequential_log13() {
    let n = 1 << 13;
    let data: Vec<BabyBear> = (0..n as u32).map(BabyBear::new).collect();
    assert_gpu_matches_cpu(13, NttDirection::Forward, &data);
}

// --- Forward: edge cases ---

#[test]
fn forward_large_values() {
    let n = 1 << 10;
    let data: Vec<BabyBear> = (0..n as u32)
        .map(|i| BabyBear::new(zkgpu_babybear::P - 1 - i))
        .collect();
    assert_gpu_matches_cpu(10, NttDirection::Forward, &data);
}

#[test]
fn forward_all_ones() {
    let n = 1 << 8;
    let data = vec![BabyBear::new(1); n];
    assert_gpu_matches_cpu(8, NttDirection::Forward, &data);
}

#[test]
fn forward_all_zeros() {
    let n = 1 << 8;
    let data = vec![BabyBear::new(0); n];
    assert_gpu_matches_cpu(8, NttDirection::Forward, &data);
}

#[test]
fn forward_random_vectors_log10() {
    let n = 1 << 10;
    let data: Vec<BabyBear> = (0..n)
        .map(|i| {
            let pseudo_random = (i as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407)
                >> 33;
            BabyBear::new((pseudo_random as u32) % zkgpu_babybear::P)
        })
        .collect();
    assert_gpu_matches_cpu(10, NttDirection::Forward, &data);
}

// --- Inverse ---

#[test]
fn inverse_sequential_log4() {
    let n = 1 << 4;
    let data: Vec<BabyBear> = (0..n as u32).map(BabyBear::new).collect();
    assert_gpu_matches_cpu(4, NttDirection::Inverse, &data);
}

#[test]
fn inverse_sequential_log10() {
    let n = 1 << 10;
    let data: Vec<BabyBear> = (0..n as u32).map(BabyBear::new).collect();
    assert_gpu_matches_cpu(10, NttDirection::Inverse, &data);
}

#[test]
fn inverse_sequential_log9() {
    let n = 1 << 9;
    let data: Vec<BabyBear> = (0..n as u32).map(BabyBear::new).collect();
    assert_gpu_matches_cpu(9, NttDirection::Inverse, &data);
}

// --- Roundtrips ---

#[test]
fn forward_inverse_roundtrip_log8() {
    let n = 1 << 8;
    let original: Vec<BabyBear> = (0..n as u32).map(BabyBear::new).collect();

    let device = init_device();

    let mut fwd_plan =
        WgpuNttPlan::new(&device, 8, NttDirection::Forward).expect("forward plan failed");
    let mut inv_plan =
        WgpuNttPlan::new(&device, 8, NttDirection::Inverse).expect("inverse plan failed");

    let mut buf = device.upload(&original).expect("upload failed");
    fwd_plan.execute(&device, &mut buf).expect("forward failed");
    inv_plan.execute(&device, &mut buf).expect("inverse failed");

    let result = buf.read_to_vec().expect("readback failed");
    for (i, (got, want)) in result.iter().zip(original.iter()).enumerate() {
        assert_eq!(
            got, want,
            "roundtrip mismatch at idx={i}: got={got}, want={want}"
        );
    }
}

#[test]
fn forward_inverse_roundtrip_log9() {
    let n = 1 << 9;
    let original: Vec<BabyBear> = (0..n as u32).map(BabyBear::new).collect();

    let device = init_device();

    let mut fwd_plan =
        WgpuNttPlan::new(&device, 9, NttDirection::Forward).expect("forward plan failed");
    let mut inv_plan =
        WgpuNttPlan::new(&device, 9, NttDirection::Inverse).expect("inverse plan failed");

    let mut buf = device.upload(&original).expect("upload failed");
    fwd_plan.execute(&device, &mut buf).expect("forward failed");
    inv_plan.execute(&device, &mut buf).expect("inverse failed");

    let result = buf.read_to_vec().expect("readback failed");
    for (i, (got, want)) in result.iter().zip(original.iter()).enumerate() {
        assert_eq!(
            got, want,
            "roundtrip mismatch at idx={i}: got={got}, want={want}"
        );
    }
}

#[test]
fn forward_inverse_roundtrip_log11() {
    let n = 1 << 11;
    let original: Vec<BabyBear> = (0..n as u32).map(BabyBear::new).collect();

    let device = init_device();

    let mut fwd_plan =
        WgpuNttPlan::new(&device, 11, NttDirection::Forward).expect("forward plan failed");
    let mut inv_plan =
        WgpuNttPlan::new(&device, 11, NttDirection::Inverse).expect("inverse plan failed");

    let mut buf = device.upload(&original).expect("upload failed");
    fwd_plan.execute(&device, &mut buf).expect("forward failed");
    inv_plan.execute(&device, &mut buf).expect("inverse failed");

    let result = buf.read_to_vec().expect("readback failed");
    for (i, (got, want)) in result.iter().zip(original.iter()).enumerate() {
        assert_eq!(
            got, want,
            "roundtrip mismatch at idx={i}: got={got}, want={want}"
        );
    }
}

#[test]
fn forward_inverse_roundtrip_log12() {
    let n = 1 << 12;
    let original: Vec<BabyBear> = (0..n as u32).map(BabyBear::new).collect();

    let device = init_device();

    let mut fwd_plan =
        WgpuNttPlan::new(&device, 12, NttDirection::Forward).expect("forward plan failed");
    let mut inv_plan =
        WgpuNttPlan::new(&device, 12, NttDirection::Inverse).expect("inverse plan failed");

    let mut buf = device.upload(&original).expect("upload failed");
    fwd_plan.execute(&device, &mut buf).expect("forward failed");
    inv_plan.execute(&device, &mut buf).expect("inverse failed");

    let result = buf.read_to_vec().expect("readback failed");
    for (i, (got, want)) in result.iter().zip(original.iter()).enumerate() {
        assert_eq!(
            got, want,
            "roundtrip mismatch at idx={i}: got={got}, want={want}"
        );
    }
}

// --- Four-step family (forced via explicit policy to ensure coverage on UMA devices) ---

fn force_four_step_policy() -> PlannerPolicy {
    PlannerPolicy::force_four_step()
}

fn assert_four_step_matches_cpu(log_n: u32, direction: NttDirection, input: &[BabyBear]) {
    let device = init_device();
    let policy = force_four_step_policy();
    let mut plan = WgpuNttPlan::new_with_policy(&device, log_n, direction, &policy)
        .expect("failed to create four-step plan");

    let mut buf = device.upload(input).expect("upload failed");
    plan.execute(&device, &mut buf).expect("GPU NTT failed");
    let gpu_result = buf.read_to_vec().expect("readback failed");

    let mut cpu_result = input.to_vec();
    ntt_cpu_reference(&mut cpu_result, direction);

    for (i, (gpu, cpu)) in gpu_result.iter().zip(cpu_result.iter()).enumerate() {
        assert_eq!(
            gpu, cpu,
            "mismatch at idx={i}: gpu={gpu}, cpu={cpu} (log_n={log_n}, dir={direction:?})"
        );
    }
}

#[test]
fn four_step_forward_sequential_log20() {
    let n = 1u32 << 20;
    let data: Vec<BabyBear> = (0..n).map(BabyBear::new).collect();
    assert_four_step_matches_cpu(20, NttDirection::Forward, &data);
}

#[test]
fn four_step_inverse_sequential_log20() {
    let n = 1u32 << 20;
    let data: Vec<BabyBear> = (0..n).map(BabyBear::new).collect();
    assert_four_step_matches_cpu(20, NttDirection::Inverse, &data);
}

#[test]
fn four_step_roundtrip_log20() {
    let n = 1u32 << 20;
    let original: Vec<BabyBear> = (0..n).map(BabyBear::new).collect();

    let device = init_device();
    let policy = force_four_step_policy();

    let mut fwd_plan = WgpuNttPlan::new_with_policy(&device, 20, NttDirection::Forward, &policy)
        .expect("forward plan failed");
    let mut inv_plan = WgpuNttPlan::new_with_policy(&device, 20, NttDirection::Inverse, &policy)
        .expect("inverse plan failed");

    let mut buf = device.upload(&original).expect("upload failed");
    fwd_plan.execute(&device, &mut buf).expect("forward failed");
    inv_plan.execute(&device, &mut buf).expect("inverse failed");

    let result = buf.read_to_vec().expect("readback failed");
    for (i, (got, want)) in result.iter().zip(original.iter()).enumerate() {
        assert_eq!(
            got, want,
            "roundtrip mismatch at idx={i}: got={got}, want={want}"
        );
    }
}

#[test]
fn four_step_forward_pseudorandom_log20() {
    let n = 1u32 << 20;
    let data: Vec<BabyBear> = (0..n as u64)
        .map(|i| {
            let pseudo_random = i
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407)
                >> 33;
            BabyBear::new((pseudo_random as u32) % zkgpu_babybear::P)
        })
        .collect();
    assert_four_step_matches_cpu(20, NttDirection::Forward, &data);
}

#[test]
fn four_step_profiled_inverse_log20() {
    let device = init_device();
    let n = 1u32 << 20;
    let data: Vec<BabyBear> = (0..n).map(BabyBear::new).collect();

    let policy = force_four_step_policy();
    let mut plan = WgpuNttPlan::new_with_policy(&device, 20, NttDirection::Inverse, &policy)
        .expect("inverse plan failed");
    let mut buf = device.upload(&data).expect("upload failed");

    let timings = plan
        .execute_kernels_profiled(&device, &mut buf)
        .expect("profiled execution failed");

    if let Some(t) = timings {
        let expected_dispatches = plan.num_dispatches() as usize;
        assert_eq!(
            t.gpu_stage_ns.len(),
            expected_dispatches,
            "profiled span count should match num_dispatches()"
        );
        let last = t
            .gpu_stage_ns
            .last()
            .expect("should have at least one span");
        assert_eq!(
            last.label, "inverse scale",
            "last span should be the inverse scale dispatch"
        );
    }
}

// --- Validation: out-of-range log_n ---

#[test]
fn rejects_log_n_zero() {
    let device = init_device();
    let err = WgpuNttPlan::new(&device, 0, NttDirection::Forward);
    assert!(matches!(err, Err(ZkGpuError::InvalidNttSize(_))));
}

#[test]
fn rejects_log_n_exceeds_babybear_adicity() {
    let device = init_device();
    let err = WgpuNttPlan::new(&device, 28, NttDirection::Forward);
    assert!(matches!(err, Err(ZkGpuError::InvalidNttSize(_))));
}

// --- Profiling: inverse plan includes scaling span ---

#[test]
fn profiled_inverse_includes_scale_span() {
    let device = init_device();
    let n = 1 << 10;
    let data: Vec<BabyBear> = (0..n as u32).map(BabyBear::new).collect();

    let mut plan =
        WgpuNttPlan::new(&device, 10, NttDirection::Inverse).expect("inverse plan failed");
    let mut buf = device.upload(&data).expect("upload failed");

    let timings = plan
        .execute_kernels_profiled(&device, &mut buf)
        .expect("profiled execution failed");

    if let Some(t) = timings {
        let expected_dispatches = plan.num_dispatches() as usize;
        assert_eq!(
            t.gpu_stage_ns.len(),
            expected_dispatches,
            "profiled span count should match num_dispatches()"
        );
        let last = t
            .gpu_stage_ns
            .last()
            .expect("should have at least one span");
        assert_eq!(
            last.label, "inverse scale",
            "last span should be the inverse scale dispatch"
        );
    }
}

#[test]
fn profiled_forward_has_no_scale_span() {
    let device = init_device();
    let n = 1 << 10;
    let data: Vec<BabyBear> = (0..n as u32).map(BabyBear::new).collect();

    let mut plan =
        WgpuNttPlan::new(&device, 10, NttDirection::Forward).expect("forward plan failed");
    let mut buf = device.upload(&data).expect("upload failed");

    let timings = plan
        .execute_kernels_profiled(&device, &mut buf)
        .expect("profiled execution failed");

    if let Some(t) = timings {
        for span in &t.gpu_stage_ns {
            assert_ne!(
                span.label, "inverse scale",
                "forward plan should have no scale span"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Local kernel variant: portable R4 DIF (forced regardless of device caps)
// ---------------------------------------------------------------------------

fn force_portable_local_policy() -> PlannerPolicy {
    PlannerPolicy::stockham_only()
}

fn assert_portable_local_matches_cpu(log_n: u32, direction: NttDirection, input: &[BabyBear]) {
    let device = init_device();
    let policy = force_portable_local_policy();
    let mut plan = WgpuNttPlan::new_with_policy(&device, log_n, direction, &policy)
        .expect("failed to create portable-local plan");

    let mut buf = device.upload(input).expect("upload failed");
    plan.execute(&device, &mut buf).expect("GPU NTT failed");
    let gpu_result = buf.read_to_vec().expect("readback failed");

    let mut cpu_result = input.to_vec();
    ntt_cpu_reference(&mut cpu_result, direction);

    for (i, (gpu, cpu)) in gpu_result.iter().zip(cpu_result.iter()).enumerate() {
        assert_eq!(
            gpu, cpu,
            "portable-local mismatch at idx={i}: gpu={gpu}, cpu={cpu} (log_n={log_n}, dir={direction:?})"
        );
    }
}

#[test]
fn portable_local_forward_log10() {
    let n = 1u32 << 10;
    let data: Vec<BabyBear> = (0..n).map(BabyBear::new).collect();
    assert_portable_local_matches_cpu(10, NttDirection::Forward, &data);
}

#[test]
fn portable_local_inverse_log10() {
    let n = 1u32 << 10;
    let data: Vec<BabyBear> = (0..n).map(BabyBear::new).collect();
    assert_portable_local_matches_cpu(10, NttDirection::Inverse, &data);
}

#[test]
fn portable_local_forward_log11() {
    let n = 1u32 << 11;
    let data: Vec<BabyBear> = (0..n).map(BabyBear::new).collect();
    assert_portable_local_matches_cpu(11, NttDirection::Forward, &data);
}

#[test]
fn portable_local_forward_log14() {
    let n = 1u32 << 14;
    let data: Vec<BabyBear> = (0..n).map(BabyBear::new).collect();
    assert_portable_local_matches_cpu(14, NttDirection::Forward, &data);
}

#[test]
fn portable_local_roundtrip_log13() {
    let n = 1u32 << 13;
    let original: Vec<BabyBear> = (0..n).map(BabyBear::new).collect();

    let device = init_device();
    let policy = force_portable_local_policy();

    let mut fwd_plan = WgpuNttPlan::new_with_policy(&device, 13, NttDirection::Forward, &policy)
        .expect("forward plan failed");
    let mut inv_plan = WgpuNttPlan::new_with_policy(&device, 13, NttDirection::Inverse, &policy)
        .expect("inverse plan failed");

    let mut buf = device.upload(&original).expect("upload failed");
    fwd_plan.execute(&device, &mut buf).expect("forward failed");
    inv_plan.execute(&device, &mut buf).expect("inverse failed");

    let result = buf.read_to_vec().expect("readback failed");
    for (i, (got, want)) in result.iter().zip(original.iter()).enumerate() {
        assert_eq!(
            got, want,
            "portable-local roundtrip mismatch at idx={i}: got={got}, want={want}"
        );
    }
}

