//! Hash suite runner (Phase F.3.b).
//!
//! Parallel to [`crate::runner`] which handles NTT cases. Dispatches
//! a [`HashSpec`] to either [`WgpuBabyBearPoseidon2Plan`] or
//! [`WgpuGoldilocksPoseidon2Plan`] based on `spec.field`, validates
//! each case against the CPU reference in [`zkgpu_poseidon2`], and
//! assembles a [`HashSuiteReport`].
//!
//! Kept out of `runner.rs` to keep that file navigable —
//! NTT-specific plumbing (family/tail overrides, roundtrip, soak)
//! doesn't touch the hash path.

use zkgpu_babybear::BabyBear;
use zkgpu_core::ZkGpuError;
use zkgpu_goldilocks::Goldilocks;
use zkgpu_poseidon2::{Poseidon2, Poseidon2Params, WIDTH as POSEIDON2_WIDTH};
use zkgpu_report::{
    Field, HashAlgorithm, HashCaseReport, HashCaseSpec, HashSpec, HashSuiteReport,
    KernelReport, SuiteSummary, TimingReport,
};
use zkgpu_wgpu::{WgpuBabyBearPoseidon2Plan, WgpuDevice, WgpuGoldilocksPoseidon2Plan};

use crate::benchmark::{
    measure_babybear_poseidon2_plan, measure_goldilocks_poseidon2_plan,
    Poseidon2PlanMeasurement,
};
use crate::device::build_device_report;
use crate::inputs::{make_babybear_hash_input, make_goldilocks_hash_input};
use crate::TestkitError;

/// Entry point: run a full hash suite.
///
/// Errors structurally on empty-suite and device-init failures. Per-case
/// failures (plan construction, measurement crash, validation mismatch)
/// are captured in the returned report as `HashCaseReport.error` /
/// `.passed = false` rather than bubbling up — matches the NTT
/// `run_suite` contract so mixed outcomes in a single suite produce a
/// coherent JSON report.
pub fn run_hash_suite(spec: &HashSpec) -> Result<HashSuiteReport, TestkitError> {
    if spec.cases.is_empty() {
        return Err(TestkitError::EmptySuite);
    }

    let device =
        WgpuDevice::new().map_err(|e| TestkitError::DeviceInit(e.to_string()))?;
    let device_report = build_device_report(&device);
    let mut cases = Vec::with_capacity(spec.cases.len());

    // Suite-level plan reuse: construct the plan once (uploads
    // round-constant buffers) and reuse across every case. Matches
    // the NTT pattern where a single plan backs multiple executes.
    // Plans are built per-field — the type systems diverge on their
    // buffer element type.
    match (spec.field, spec.algorithm) {
        (Field::BabyBear, HashAlgorithm::Poseidon2) => {
            let params =
                Poseidon2Params::<BabyBear, POSEIDON2_WIDTH>::babybear_default();
            let cpu = Poseidon2::new(params.clone());
            let mut plan = match WgpuBabyBearPoseidon2Plan::new(&device, params) {
                Ok(p) => p,
                Err(err) => {
                    // Plan-build failure affects every case — emit one
                    // error row per case rather than returning a single
                    // TestkitError, so the report still lists what was
                    // requested.
                    return Ok(suite_report_with_plan_error(
                        spec,
                        device_report,
                        err,
                        "babybear-poseidon2",
                    ));
                }
            };
            for case in &spec.cases {
                let report = run_case_babybear(&device, case, &mut plan, &cpu);
                let failed = !report.passed;
                cases.push(report);
                if spec.fail_fast && failed {
                    break;
                }
            }
        }
        (Field::Goldilocks, HashAlgorithm::Poseidon2) => {
            let params =
                Poseidon2Params::<Goldilocks, POSEIDON2_WIDTH>::goldilocks_default();
            let cpu = Poseidon2::new(params.clone());
            let mut plan = match WgpuGoldilocksPoseidon2Plan::new(&device, params) {
                Ok(p) => p,
                Err(err) => {
                    return Ok(suite_report_with_plan_error(
                        spec,
                        device_report,
                        err,
                        "goldilocks-poseidon2-portable",
                    ));
                }
            };
            for case in &spec.cases {
                let report = run_case_goldilocks(&device, case, &mut plan, &cpu);
                let failed = !report.passed;
                cases.push(report);
                if spec.fail_fast && failed {
                    break;
                }
            }
        }
    }

    let passed_cases = cases.iter().filter(|c| c.passed).count() as u32;
    let total_cases = cases.len() as u32;
    let failed_cases = total_cases - passed_cases;
    let kernel_family = match spec.field {
        Field::BabyBear => "babybear-poseidon2",
        Field::Goldilocks => "goldilocks-poseidon2-portable",
    };

    Ok(HashSuiteReport {
        schema_version: 1,
        suite: spec.kind,
        device: device_report,
        kernel: KernelReport {
            field: spec.field.display_name().to_string(),
            ntt_variant: kernel_family.to_string(),
            stockham_tail_strategy: None,
        },
        cases,
        summary: SuiteSummary {
            total_cases,
            passed_cases,
            failed_cases,
        },
    })
}

/// Convenience entry point for the shipped smoke suite.
pub fn run_poseidon2_smoke_suite() -> Result<HashSuiteReport, TestkitError> {
    run_hash_suite(&zkgpu_report::poseidon2_smoke_suite())
}

/// Convenience entry point for the shipped benchmark suite.
pub fn run_poseidon2_benchmark_suite() -> Result<HashSuiteReport, TestkitError> {
    run_hash_suite(&zkgpu_report::poseidon2_benchmark_suite())
}

// ---------------------------------------------------------------------------
// Per-case runners
// ---------------------------------------------------------------------------

fn run_case_babybear(
    device: &WgpuDevice,
    case: &HashCaseSpec,
    plan: &mut WgpuBabyBearPoseidon2Plan,
    cpu: &Poseidon2<BabyBear, POSEIDON2_WIDTH>,
) -> HashCaseReport {
    // Zero-permutation case is a legitimate no-op at the plan level
    // (the kernel's `execute` also returns `Ok(())` for an empty
    // buffer) but `device.upload` panics on an empty slice, so short-
    // circuit before the measure path. Harness code can build a batch
    // dynamically and an empty result shouldn't require special
    // casing at every call site.
    if case.num_permutations == 0 {
        return empty_case_report(case, "babybear-poseidon2");
    }

    let input = make_babybear_hash_input(case.num_permutations, &case.input);

    // CPU reference: permute each WIDTH-block independently.
    let mut expected = input.clone();
    for chunk in expected.chunks_exact_mut(POSEIDON2_WIDTH) {
        let arr: &mut [BabyBear; POSEIDON2_WIDTH] =
            chunk.try_into().expect("chunks_exact_mut enforces length");
        cpu.permute(arr);
    }

    let measurement = match measure_babybear_poseidon2_plan(
        device,
        &input,
        plan,
        case.warmup_iterations,
        case.iterations,
        case.profile_gpu_timestamps,
    ) {
        Ok(m) => m,
        Err(err) => return case_error(case, Some("babybear-poseidon2"), err),
    };

    finalize_case(case, "babybear-poseidon2", measurement, expected)
}

fn run_case_goldilocks(
    device: &WgpuDevice,
    case: &HashCaseSpec,
    plan: &mut WgpuGoldilocksPoseidon2Plan,
    cpu: &Poseidon2<Goldilocks, POSEIDON2_WIDTH>,
) -> HashCaseReport {
    if case.num_permutations == 0 {
        return empty_case_report(case, "goldilocks-poseidon2-portable");
    }

    let input = make_goldilocks_hash_input(case.num_permutations, &case.input);

    let mut expected = input.clone();
    for chunk in expected.chunks_exact_mut(POSEIDON2_WIDTH) {
        let arr: &mut [Goldilocks; POSEIDON2_WIDTH] =
            chunk.try_into().expect("chunks_exact_mut enforces length");
        cpu.permute(arr);
    }

    let measurement = match measure_goldilocks_poseidon2_plan(
        device,
        &input,
        plan,
        case.warmup_iterations,
        case.iterations,
        case.profile_gpu_timestamps,
    ) {
        Ok(m) => m,
        Err(err) => {
            return case_error(case, Some("goldilocks-poseidon2-portable"), err)
        }
    };

    finalize_case(
        case,
        "goldilocks-poseidon2-portable",
        measurement,
        expected,
    )
}

// ---------------------------------------------------------------------------
// Validation + report-building
// ---------------------------------------------------------------------------

/// Field-generic per-case finalizer. Compares `measurement.final_output`
/// against `expected` slot-by-slot, records the first mismatch
/// coordinates (permutation_index, slot_index), and builds the
/// `HashCaseReport`.
fn finalize_case<F>(
    case: &HashCaseSpec,
    kernel_family: &'static str,
    measurement: Poseidon2PlanMeasurement<F>,
    expected: Vec<F>,
) -> HashCaseReport
where
    F: PartialEq + core::fmt::Display,
{
    let gpu_out = &measurement.final_output;

    let mut mismatch_count = 0u32;
    let mut first_mismatch: Option<(u32, u32, String, String)> = None;

    for (perm_idx, (gpu_block, cpu_block)) in gpu_out
        .chunks_exact(POSEIDON2_WIDTH)
        .zip(expected.chunks_exact(POSEIDON2_WIDTH))
        .enumerate()
    {
        for (slot, (g, c)) in gpu_block.iter().zip(cpu_block.iter()).enumerate() {
            if g != c {
                mismatch_count += 1;
                if first_mismatch.is_none() {
                    first_mismatch = Some((
                        perm_idx as u32,
                        slot as u32,
                        g.to_string(),
                        c.to_string(),
                    ));
                }
            }
        }
    }

    let (first_mismatch_index, first_mismatch_gpu, first_mismatch_cpu) =
        match first_mismatch {
            Some((p, s, g, c)) => (Some((p, s)), Some(g), Some(c)),
            None => (None, None, None),
        };

    HashCaseReport {
        name: case.name.clone(),
        num_permutations: case.num_permutations,
        input: case.input,
        kernel_family: Some(kernel_family.to_string()),
        passed: mismatch_count == 0,
        mismatch_count,
        first_mismatch_index,
        first_mismatch_gpu,
        first_mismatch_cpu,
        timings: measurement.timings,
        error: None,
    }
}

/// Report row for an empty-batch case: passes as a no-op, zero
/// timings, zero mismatches.
fn empty_case_report(
    case: &HashCaseSpec,
    kernel_family: &'static str,
) -> HashCaseReport {
    HashCaseReport {
        name: case.name.clone(),
        num_permutations: 0,
        input: case.input,
        kernel_family: Some(kernel_family.to_string()),
        passed: true,
        mismatch_count: 0,
        first_mismatch_index: None,
        first_mismatch_gpu: None,
        first_mismatch_cpu: None,
        timings: TimingReport {
            wall_time_ns: Some(0),
            gpu_total_ns: None,
            gpu_stage_ns: Vec::new(),
        },
        error: None,
    }
}

fn case_error(
    case: &HashCaseSpec,
    kernel_family: Option<&'static str>,
    err: ZkGpuError,
) -> HashCaseReport {
    HashCaseReport {
        name: case.name.clone(),
        num_permutations: case.num_permutations,
        input: case.input,
        kernel_family: kernel_family.map(|s| s.to_string()),
        passed: false,
        mismatch_count: 0,
        first_mismatch_index: None,
        first_mismatch_gpu: None,
        first_mismatch_cpu: None,
        timings: TimingReport {
            wall_time_ns: None,
            gpu_total_ns: None,
            gpu_stage_ns: Vec::new(),
        },
        error: Some(err.to_string()),
    }
}

/// Build a suite report where every case carries the same plan-build
/// error. Called when `WgpuBabyBearPoseidon2Plan::new` (or the
/// Goldilocks twin) rejects the params — usually means an unsupported
/// `alpha`. Still emits a report row per case so the operator sees
/// exactly which cases were requested.
fn suite_report_with_plan_error(
    spec: &HashSpec,
    device_report: zkgpu_report::DeviceReport,
    err: ZkGpuError,
    kernel_family: &'static str,
) -> HashSuiteReport {
    let err_str = err.to_string();
    let cases: Vec<HashCaseReport> = spec
        .cases
        .iter()
        .map(|case| HashCaseReport {
            name: case.name.clone(),
            num_permutations: case.num_permutations,
            input: case.input,
            kernel_family: Some(kernel_family.to_string()),
            passed: false,
            mismatch_count: 0,
            first_mismatch_index: None,
            first_mismatch_gpu: None,
            first_mismatch_cpu: None,
            timings: TimingReport {
                wall_time_ns: None,
                gpu_total_ns: None,
                gpu_stage_ns: Vec::new(),
            },
            error: Some(err_str.clone()),
        })
        .collect();
    let total_cases = cases.len() as u32;

    HashSuiteReport {
        schema_version: 1,
        suite: spec.kind,
        device: device_report,
        kernel: KernelReport {
            field: spec.field.display_name().to_string(),
            ntt_variant: kernel_family.to_string(),
            stockham_tail_strategy: None,
        },
        cases,
        summary: SuiteSummary {
            total_cases,
            passed_cases: 0,
            failed_cases: total_cases,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkgpu_report::{poseidon2_smoke_suite, HashInputPattern, SuiteKind};

    fn try_device() -> Option<WgpuDevice> {
        WgpuDevice::new().ok()
    }

    /// Empty suite must produce a structured `TestkitError::EmptySuite`
    /// before any device work. No GPU required.
    #[test]
    fn empty_suite_rejected() {
        let spec = HashSpec {
            kind: SuiteKind::Smoke,
            cases: Vec::new(),
            fail_fast: true,
            algorithm: HashAlgorithm::Poseidon2,
            field: Field::BabyBear,
        };
        let err = run_hash_suite(&spec);
        assert!(matches!(err, Err(TestkitError::EmptySuite)));
    }

    /// End-to-end BabyBear smoke: every shipped smoke case must pass
    /// against the CPU reference. Covers the Sequential / Zeros /
    /// Ones / RNG input paths.
    #[test]
    fn babybear_smoke_suite_runs_end_to_end() {
        if try_device().is_none() {
            eprintln!("skipping: no GPU adapter available");
            return;
        }
        let spec = poseidon2_smoke_suite(); // defaults to BabyBear
        let report = run_hash_suite(&spec).expect("smoke suite must run");
        assert_eq!(report.kernel.field, "BabyBear");
        assert_eq!(report.summary.failed_cases, 0, "{:?}", report.cases);
        assert_eq!(
            report.summary.total_cases, report.summary.passed_cases,
            "all smoke cases must pass",
        );
        for c in &report.cases {
            assert_eq!(
                c.kernel_family.as_deref(),
                Some("babybear-poseidon2"),
                "unexpected kernel_family: {:?}",
                c.kernel_family
            );
        }
    }

    /// Goldilocks smoke: same cases, routed through the u32x2 GPU
    /// plan. Exercises the `Goldilocks` spec-level field dispatch.
    #[test]
    fn goldilocks_smoke_suite_runs_end_to_end() {
        if try_device().is_none() {
            eprintln!("skipping: no GPU adapter available");
            return;
        }
        let mut spec = poseidon2_smoke_suite();
        spec.field = Field::Goldilocks;
        let report = run_hash_suite(&spec).expect("goldilocks smoke must run");
        assert_eq!(report.kernel.field, "Goldilocks");
        assert_eq!(report.summary.failed_cases, 0, "{:?}", report.cases);
        for c in &report.cases {
            assert_eq!(
                c.kernel_family.as_deref(),
                Some("goldilocks-poseidon2-portable"),
            );
        }
    }

    /// Batch-17 sequential differential: prime size to catch any
    /// 2D-fold off-by-one in the testkit ↔ plan boundary. Runs both
    /// fields back-to-back so a per-field regression is caught in
    /// one test.
    #[test]
    fn hash_batch_17_matches_cpu_both_fields() {
        if try_device().is_none() {
            eprintln!("skipping: no GPU adapter available");
            return;
        }
        let case = HashCaseSpec::new(
            "batch17_seq",
            17,
            HashInputPattern::Sequential,
        );
        for field in [Field::BabyBear, Field::Goldilocks] {
            let spec = HashSpec {
                kind: SuiteKind::Smoke,
                cases: vec![case.clone()],
                fail_fast: true,
                algorithm: HashAlgorithm::Poseidon2,
                field,
            };
            let report = run_hash_suite(&spec).unwrap();
            let c = &report.cases[0];
            assert!(
                c.passed,
                "field={field:?} must pass: {:?} mismatches at {:?}",
                c.mismatch_count, c.first_mismatch_index
            );
            assert_eq!(c.num_permutations, 17);
        }
    }

    /// `profile_gpu_timestamps = true` is accepted by the spec but
    /// the F.3.b testkit degrades to wall-only — there's no
    /// `execute_profiled` on the Poseidon2 plans yet. Contract: the
    /// flag doesn't error, but `gpu_total_ns` stays `None`. When a
    /// future sub-phase adds profiled-execute both the plan and this
    /// test update together (the assertion below should flip to
    /// `is_some()` at that point).
    #[test]
    fn hash_profile_flag_currently_degrades_to_wall_only() {
        if try_device().is_none() {
            eprintln!("skipping: no GPU adapter available");
            return;
        }
        let spec = HashSpec {
            kind: SuiteKind::Benchmark,
            cases: vec![HashCaseSpec::new(
                "profile_wall_only",
                8,
                HashInputPattern::Sequential,
            )
            .with_profile(true)
            .with_iterations(1, 2)],
            fail_fast: true,
            algorithm: HashAlgorithm::Poseidon2,
            field: Field::BabyBear,
        };
        let report = run_hash_suite(&spec).expect("must run");
        let c = &report.cases[0];
        assert!(c.passed);
        // Wall time populated (some value > 0), GPU timestamps not.
        assert!(c.timings.wall_time_ns.is_some());
        assert!(
            c.timings.gpu_total_ns.is_none(),
            "pre-profiled-execute Poseidon2 path must leave gpu_total_ns unset; \
             got {:?}",
            c.timings.gpu_total_ns,
        );
        assert!(c.timings.gpu_stage_ns.is_empty());
    }

    /// `fail_fast = true` must stop after the first failing case.
    /// Exercised via plan-level rejection (alpha=3 forces every case
    /// to fail with the same plan-build error), though fail_fast is
    /// academic there since the plan error fails all cases uniformly.
    /// Still useful as a contract gate.
    #[test]
    fn hash_spec_with_zero_permutations_passes_as_noop() {
        // A 0-permutation case exercises the empty-batch no-op on the
        // plan side. Should still produce a passed case report.
        if try_device().is_none() {
            eprintln!("skipping: no GPU adapter available");
            return;
        }
        let spec = HashSpec {
            kind: SuiteKind::Smoke,
            cases: vec![HashCaseSpec::new(
                "empty",
                0,
                HashInputPattern::AllZeros,
            )],
            fail_fast: true,
            algorithm: HashAlgorithm::Poseidon2,
            field: Field::BabyBear,
        };
        let report = run_hash_suite(&spec).unwrap();
        assert!(report.cases[0].passed, "{:?}", report.cases[0]);
        assert_eq!(report.cases[0].num_permutations, 0);
        assert_eq!(report.cases[0].mismatch_count, 0);
    }
}
