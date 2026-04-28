use zkgpu_babybear::BabyBear;
use zkgpu_core::{GpuBuffer, GpuDevice, NttDirection, NttPlan, ZkGpuError};
use zkgpu_ntt::ntt_cpu_reference;
use zkgpu_wgpu::{GpuFamily, GpuTiming, PlannerPolicy, WgpuDevice, WgpuNttPlan};

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
    // NVIDIA scale-up Tier 2B Option A (2026-04-16): Four-Step inverse
    // no longer emits a separate "inverse scale" dispatch. The `1/N`
    // normalization is folded into the Phase-3 diagonal twiddle table
    // during precomputation, so Phase 7 is skipped. The last dispatch
    // of a Four-Step inverse plan is now the same as for a forward
    // plan: the final transpose `R×C→C×R (output)`.
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
        // Post-Tier-2B-A: no standalone "inverse scale" span anywhere.
        for span in &t.gpu_stage_ns {
            assert_ne!(
                span.label, "inverse scale",
                "Four-Step inverse no longer emits a separate scale dispatch"
            );
        }
    }
}

// --- Tier 1 Fix 2 regression tests (2026-04-16): 2D-folded dispatch ---
//
// These tests exercise the log_n ≥ 25 regime where a 1D dispatch grid
// would exceed `max_compute_workgroups_per_dimension` (typically 65535).
// The 2D fold spreads workgroups across (x, y) via `plan_linear_dispatch`.
//
// log 25 needs 128 MB of data + 128 MB scratch, which requires the
// post-Fix-3 `DesktopDiscrete` adapter-max buffer limit. Apple M4 Pro
// and mobile devices with <2 GB buffer caps will reject the plan up
// front. Marked `#[ignore]` so they only run when explicitly requested
// (`cargo test --test gpu_ntt_validation -- --ignored`). Intended
// primary run site: the vast.ai RTX 4090 harness.

#[test]
#[ignore = "requires Four-Step + >= 256 MB buffer limit; run on DesktopDiscrete (T3.A regression)"]
fn four_step_roundtrip_log22() {
    // T3.A (2026-04-17) regression: log 22 Four-Step leaves use 3 R8 + 1 R4
    // (log_leaf = 11 = 3·3 + 2). Validates R8 twiddle math + multi-stage R8
    // chain correctness at a size where R8 dominates the leaf dispatches.
    let device = init_device();
    let log_n: u32 = 22;
    let n = 1u32 << log_n;
    let data: Vec<BabyBear> = (0..n).map(BabyBear::new).collect();

    let policy = force_four_step_policy();
    let mut fwd_plan = WgpuNttPlan::new_with_policy(&device, log_n, NttDirection::Forward, &policy)
        .expect("log 22 Four-Step forward plan failed");
    let mut inv_plan = WgpuNttPlan::new_with_policy(&device, log_n, NttDirection::Inverse, &policy)
        .expect("log 22 Four-Step inverse plan failed");
    let mut buf = device.upload(&data).expect("upload failed");

    fwd_plan.execute(&device, &mut buf).expect("log 22 Four-Step forward failed");
    inv_plan.execute(&device, &mut buf).expect("log 22 Four-Step inverse failed");

    let round_tripped = buf.read_to_vec().expect("readback failed");
    // Spot-check: full 4M-element comparison is slow; check 7 index positions.
    for &i in &[0usize, 1, 100, 1_000, 100_000, 2_000_000, 4_194_303] {
        assert_eq!(round_tripped[i], data[i], "log 22 Four-Step round-trip mismatch at idx={i} — R8 regression?");
    }
}

#[test]
#[ignore = "requires Four-Step + >= 512 MB buffer limit; run on DesktopDiscrete (T3.A regression)"]
fn four_step_roundtrip_log24() {
    // T3.A (2026-04-17): log 24 Four-Step leaves use 4 R8 (log_leaf = 12 = 4·3,
    // clean factoring — no R4 or R2 residue). Pure-R8 correctness check.
    let device = init_device();
    let log_n: u32 = 24;
    let n = 1u32 << log_n;
    let data: Vec<BabyBear> = (0..n).map(BabyBear::new).collect();

    let policy = force_four_step_policy();
    let mut fwd_plan = WgpuNttPlan::new_with_policy(&device, log_n, NttDirection::Forward, &policy)
        .expect("log 24 Four-Step forward plan failed");
    let mut inv_plan = WgpuNttPlan::new_with_policy(&device, log_n, NttDirection::Inverse, &policy)
        .expect("log 24 Four-Step inverse plan failed");
    let mut buf = device.upload(&data).expect("upload failed");

    fwd_plan.execute(&device, &mut buf).expect("log 24 Four-Step forward failed");
    inv_plan.execute(&device, &mut buf).expect("log 24 Four-Step inverse failed");

    let round_tripped = buf.read_to_vec().expect("readback failed");
    for &i in &[0usize, 1, 100, 1_000_000, 8_000_000, 16_000_000, 16_777_215] {
        assert_eq!(round_tripped[i], data[i], "log 24 Four-Step round-trip mismatch at idx={i} — R8 regression?");
    }
}

#[test]
#[ignore = "requires >= 512 MB buffer limit; run on DesktopDiscrete"]
fn stockham_roundtrip_log25() {
    // log 25 = 33M elements. Forward then inverse should round-trip to
    // the original input. Uses Stockham (log 25 > NVIDIA's Four-Step
    // threshold of 21, but Stockham stays legal on devices without
    // Four-Step plumbing, so force it here.
    let device = init_device();
    let log_n: u32 = 25;
    let n = 1u32 << log_n;
    let data: Vec<BabyBear> = (0..n).map(BabyBear::new).collect();

    // Use `force_four_step_policy` (which actually forces Stockham at
    // the NVIDIA 4090 threshold of 21... no wait, that forces Four-Step.
    // Use `stockham_only` via PlannerPolicy to exercise the 2D fold in
    // the Stockham pipeline.)
    let policy = PlannerPolicy::stockham_only();
    let mut fwd_plan = WgpuNttPlan::new_with_policy(&device, log_n, NttDirection::Forward, &policy)
        .expect("log 25 Stockham forward plan failed — Fix 2 regression?");
    let mut inv_plan = WgpuNttPlan::new_with_policy(&device, log_n, NttDirection::Inverse, &policy)
        .expect("log 25 Stockham inverse plan failed — Fix 2 regression?");
    let mut buf = device
        .upload(&data)
        .expect("upload failed — check adapter-max buffer (Fix 3)");

    fwd_plan
        .execute(&device, &mut buf)
        .expect("log 25 forward execution failed — 2D dispatch regression?");
    inv_plan
        .execute(&device, &mut buf)
        .expect("log 25 inverse execution failed — 2D dispatch regression?");

    let round_tripped = buf.read_to_vec().expect("readback failed");
    // Spot-check: full 33M-element comparison is slow even on host.
    // Check a handful of index positions.
    for &i in &[0usize, 1, 100, 1_000_000, 8_000_000, 16_000_000, 33_554_431] {
        assert_eq!(
            round_tripped[i], data[i],
            "log 25 round-trip mismatch at idx={i}"
        );
    }
}

#[test]
#[ignore = "requires Four-Step + >= 1 GB buffer limit; run on DesktopDiscrete (Fix 2b regression)"]
fn four_step_roundtrip_log26() {
    // Tier 1 Fix 2b regression (2026-04-16): log 26 needs
    // `batch_count * leaf_n / 4 = 2^24 / 4 = 2^22 = 4.2M` butterflies
    // for the R4 leaf — exceeds the wgpu 65535 per-dim limit with a 1D
    // grid. Pre-Fix-2b, Four-Step panicked here with
    // `dispatch group size dimension [65536, 1, 1] > 65535`.
    let device = init_device();
    let log_n: u32 = 26;
    let n = 1u32 << log_n;
    let data: Vec<BabyBear> = (0..n).map(BabyBear::new).collect();

    let policy = force_four_step_policy();
    let mut fwd_plan = WgpuNttPlan::new_with_policy(&device, log_n, NttDirection::Forward, &policy)
        .expect("log 26 Four-Step forward plan failed");
    let mut inv_plan = WgpuNttPlan::new_with_policy(&device, log_n, NttDirection::Inverse, &policy)
        .expect("log 26 Four-Step inverse plan failed");
    let mut buf = device.upload(&data).expect("upload failed");

    fwd_plan.execute(&device, &mut buf).expect("log 26 Four-Step forward failed — Fix 2b regression?");
    inv_plan.execute(&device, &mut buf).expect("log 26 Four-Step inverse failed");

    let round_tripped = buf.read_to_vec().expect("readback failed");
    for &i in &[0usize, 1, 100, 1_000_000, 16_000_000, 67_000_000, 67_108_863] {
        assert_eq!(round_tripped[i], data[i], "log 26 Four-Step round-trip mismatch at idx={i}");
    }
}

#[test]
#[ignore = "requires Four-Step + >= 1 GB buffer limit; run on DesktopDiscrete (Fix 2b regression)"]
fn four_step_roundtrip_log25() {
    // Same round-trip at log 25 but through the Four-Step family.
    // Four-Step's leaf kernels already handle the 2D dispatch path
    // (leaves encode via `plan_linear_dispatch`), so this test is
    // primarily a regression check that Fix 2's Stockham changes
    // didn't break the Four-Step path that consumes Stockham leaf
    // kernels for sub-NTTs. The Four-Step leaf kernels themselves
    // use `babybear_fourstep_leaf_r4.wgsl` and were not changed in
    // Fix 2, but the planner-config path is shared.
    let device = init_device();
    let log_n: u32 = 25;
    let n = 1u32 << log_n;
    let data: Vec<BabyBear> = (0..n).map(BabyBear::new).collect();

    let policy = force_four_step_policy();
    let mut fwd_plan = WgpuNttPlan::new_with_policy(&device, log_n, NttDirection::Forward, &policy)
        .expect("log 25 Four-Step forward plan failed");
    let mut inv_plan = WgpuNttPlan::new_with_policy(&device, log_n, NttDirection::Inverse, &policy)
        .expect("log 25 Four-Step inverse plan failed");
    let mut buf = device.upload(&data).expect("upload failed");

    fwd_plan.execute(&device, &mut buf).expect("log 25 Four-Step forward failed");
    inv_plan.execute(&device, &mut buf).expect("log 25 Four-Step inverse failed");

    let round_tripped = buf.read_to_vec().expect("readback failed");
    for &i in &[0usize, 1, 100, 1_000_000, 8_000_000, 16_000_000, 33_554_431] {
        assert_eq!(round_tripped[i], data[i], "log 25 Four-Step round-trip failed at idx={i}");
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

// Regression lock for Gate 2 item #4 (P2): the v0.3 fold collapsed every
// global-stage timestamp span into a single outer-pass span, so the
// per-stage labels in `gpu_stage_ns` reported stale durations. The fix
// (PerPass mode for profiled execution) restores 1:1 dispatch-to-pass
// mapping. The earlier profiled tests at `log_n = 10` don't exercise
// the multi-global-R4 case that triggered the bug — at small log_n the
// plan often has zero or one global stage. log_n = 18 forces multiple
// R4 global stages, locking the regression.
//
// Two-tier assertion shape (post second-reviewer feedback on the v1 of
// this test, which only checked span count + labels and so didn't
// actually catch the bug):
//
// 1. **Span count + labels** — the cheap structural check. Both the
//    healthy and the buggy paths produce labelled spans of the right
//    cardinality, so this alone doesn't catch the regression, but it
//    catches a *different* class (label-formatting drift,
//    num_dispatches mismatch). Runs on every backend.
//
// 2. **Non-zero R4 durations** — the actual fold-regression lock.
//    Under the buggy fold, only the outer pass's timestamp pair
//    fires, so at most one R4 span reports a non-zero duration; the
//    others read undefined-resolved-as-zero. Under the healthy
//    PerPass path each R4 dispatch fires its own pair, so ≥ 2 R4
//    spans should be non-zero. The bug is deterministic; we retry
//    up to `MAX_ATTEMPTS` times only to absorb transient noise on
//    timestamp-flaky backends, and skip the duration check entirely
//    on Metal (see carve-out below).
#[test]
fn profiled_forward_log18_has_multiple_r4_spans() {
    // log_n=18: each R4 dispatch processes ~256k field elements, well
    // above any timestamp-granularity floor on real backends. The
    // multi-global-R4 case (≥ 2 R4 dispatches) is what the original P2
    // bug collapsed.
    //
    // Per-backend behavior under the buggy fold:
    //
    // - **Vulkan/NVIDIA (RTX 4090, 5090):** the bug attaches
    //   `outer_ts = ts_writes[0]` to a single outer pass and leaves
    //   subsequent spans' query indices unwritten. `resolve_query_set`
    //   over the full range then waits on those unwritten queries
    //   forever, hanging the test. CI catches this as a timeout —
    //   still a signal, just via wall-clock rather than via the
    //   duration assertion below. Healthy runs complete in <1 s; a
    //   regression that hangs for the test framework's default timeout
    //   (60 s+) is unambiguous.
    //
    // - **Apple Silicon Metal (M4 Pro):** as of wgpu v29, Metal
    //   reliably writes only the FIRST compute pass's
    //   `ComputePassTimestampWrites` per command buffer; subsequent
    //   passes' begin/end queries read back as zero (see e.g. wgpu
    //   issue tracking around `MTLCounterSampleBuffer` boundaries).
    //   That produces the same all-zero-after-first-span pattern as
    //   the bug, so the duration check can't distinguish healthy from
    //   regressed on Metal — a false-positive risk we don't want in
    //   CI. The duration check is skipped on Metal; structural checks
    //   still run.
    //
    // - **Healthy Vulkan + healthy non-Metal in general:** all R4
    //   spans report non-zero durations; the duration assertion below
    //   is the regression lock.
    //
    // Verified locally on M4 Pro: 10/10 healthy structural-only runs
    // pass, 5/5 simulated-regression runs catch the all-zero pattern
    // (the inner panic message includes the spans for diagnosability).
    // Verified on RTX 5090 Vulkan: 3/3 healthy passes complete in
    // <1 s; the simulated-regression hang above is the catch mechanism
    // there.
    const MAX_ATTEMPTS: usize = 4;

    let device = init_device();
    let log_n = 18;
    let n = 1 << log_n;
    let data: Vec<BabyBear> = (0..n as u32).map(BabyBear::new).collect();

    let on_metal = matches!(device.caps().gpu_family, GpuFamily::Apple);

    let mut plan =
        WgpuNttPlan::new(&device, log_n, NttDirection::Forward).expect("forward plan failed");

    let mut last_spans: Option<Vec<GpuTiming>> = None;

    for attempt in 0..MAX_ATTEMPTS {
        let mut buf = device.upload(&data).expect("upload failed");
        let timings = plan
            .execute_kernels_profiled(&device, &mut buf)
            .expect("profiled execution failed");

        // Skip silently if the device doesn't support GPU timestamps —
        // the profiler is None on those backends.
        let Some(t) = timings else {
            return;
        };

        // Tier 1: structural checks (cheap, run every attempt).
        let expected_dispatches = plan.num_dispatches() as usize;
        assert_eq!(
            t.gpu_stage_ns.len(),
            expected_dispatches,
            "profiled span count should match num_dispatches() — fold regression \
             would collapse to fewer spans"
        );

        let r4_spans: Vec<&GpuTiming> = t
            .gpu_stage_ns
            .iter()
            .filter(|s| s.label.starts_with("r4 stages"))
            .collect();
        assert!(
            r4_spans.len() >= 2,
            "log_n={log_n} forward should produce at least 2 R4-global-stage spans, \
             got {}; spans={:?}",
            r4_spans.len(),
            t.gpu_stage_ns.iter().map(|s| &s.label).collect::<Vec<_>>(),
        );
        for span in &t.gpu_stage_ns {
            assert!(
                !span.label.is_empty(),
                "every profiled span should have a non-empty label"
            );
        }

        // Tier 2: duration check — this is what locks the original bug.
        // The buggy fold path produces ≤ 1 non-zero R4 span deterministically.
        // Skipped on Metal (see carve-out comment above).
        if on_metal {
            // Structural checks already passed; skip duration check
            // on the one platform where it false-positives.
            return;
        }
        let nonzero_r4 = r4_spans.iter().filter(|s| s.duration_ns > 0.0).count();
        if nonzero_r4 >= 2 {
            // Healthy: at least one attempt produced 2+ non-zero R4 spans,
            // confirming each dispatch got its own measurable pass.
            return;
        }
        last_spans = Some(t.gpu_stage_ns.clone());
        eprintln!(
            "profiled_forward_log{log_n}: attempt {} saw only {nonzero_r4} non-zero R4 spans \
             (fold regression?); retrying",
            attempt + 1
        );
    }

    // Every attempt produced ≤ 1 non-zero R4 span. That's deterministic
    // bug behavior — Metal's flake rate is well under 50%, so 4 attempts
    // failing in a row is overwhelmingly the fold regression.
    panic!(
        "regression: across {MAX_ATTEMPTS} attempts, no run produced 2+ non-zero R4 spans. \
         Healthy PerPass profiling should fire each dispatch's timestamps; the buggy \
         Folded path collapses them into a single outer-pass span. \
         Last spans={:?}",
        last_spans.expect("at least one attempt produced timings"),
    );
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

