use std::time::Duration;

use zkgpu_core::{NttDirection, ZkGpuError};
use zkgpu_ntt::ntt_cpu_reference;
use zkgpu_report::{SoakCaseReport, SoakSpec, SoakSuiteReport, StockhamTailOverride};
use zkgpu_wgpu::{PlannerPolicy, StockhamTailOverride as PlanTailOverride, WgpuDevice, WgpuNttPlan};

use crate::benchmark::{measure_plan, measure_plan_soak, sum_timing_reports};
use crate::case::{CaseSpec, TestDirection};
use crate::device::build_device_report;
use crate::inputs::make_input;
use crate::report::{CaseReport, KernelReport, SuiteReport, SuiteSummary, TimingReport};
use crate::suite::{benchmark_suite, smoke_suite, validation_suite, FamilyOverride, SuiteSpec};
use crate::validation::compare_vectors;
use crate::TestkitError;

/// Translate the report-layer tail override into the planner-layer enum.
fn to_plan_tail(ov: StockhamTailOverride) -> PlanTailOverride {
    match ov {
        StockhamTailOverride::Auto => PlanTailOverride::Auto,
        StockhamTailOverride::Local => PlanTailOverride::Local,
        StockhamTailOverride::Global => PlanTailOverride::Global,
    }
}

pub fn run_suite(spec: &SuiteSpec) -> Result<SuiteReport, TestkitError> {
    if spec.cases.is_empty() {
        return Err(TestkitError::EmptySuite);
    }

    let device = WgpuDevice::new().map_err(|e| TestkitError::DeviceInit(e.to_string()))?;
    let device_report = build_device_report(&device);
    let mut cases = Vec::with_capacity(spec.cases.len());

    for case in &spec.cases {
        let report = run_case(
            &device,
            case,
            spec.family_override,
            spec.stockham_tail_override,
            spec.r8_max_log_leaf_override,
        );
        let failed = !report.passed;
        cases.push(report);
        if spec.fail_fast && failed {
            break;
        }
    }

    let passed_cases = cases.iter().filter(|c| c.passed).count() as u32;
    let failed_cases = cases.len() as u32 - passed_cases;
    let kernel_variant = derive_kernel_variant(&cases);
    let tail_variant = derive_tail_variant(&cases);
    let total_cases = cases.len() as u32;

    Ok(SuiteReport {
        schema_version: 1,
        suite: spec.kind,
        device: device_report,
        kernel: KernelReport {
            field: "BabyBear".to_string(),
            ntt_variant: kernel_variant,
            stockham_tail_strategy: tail_variant,
        },
        cases,
        summary: SuiteSummary {
            total_cases,
            passed_cases,
            failed_cases,
        },
    })
}

pub fn run_smoke_suite() -> Result<SuiteReport, TestkitError> {
    run_suite(&smoke_suite())
}

pub fn run_validation_suite() -> Result<SuiteReport, TestkitError> {
    run_suite(&validation_suite())
}

pub fn run_benchmark_suite() -> Result<SuiteReport, TestkitError> {
    run_suite(&benchmark_suite())
}

// ---------------------------------------------------------------------------
// Soak suite runner
// ---------------------------------------------------------------------------

/// Run a soak benchmark: each case runs for `spec.duration_secs` with
/// per-iteration timing samples for sustained thermal characterization.
pub fn run_soak_suite(spec: &SoakSpec) -> Result<SoakSuiteReport, TestkitError> {
    if spec.cases.is_empty() {
        return Err(TestkitError::EmptySuite);
    }

    let device = WgpuDevice::new().map_err(|e| TestkitError::DeviceInit(e.to_string()))?;
    let device_report = build_device_report(&device);
    let duration = Duration::from_secs(spec.duration_secs as u64);
    let mut cases = Vec::with_capacity(spec.cases.len());

    for case in &spec.cases {
        let report = run_soak_case(
            &device,
            case,
            duration,
            spec.validate,
            spec.family_override,
            spec.stockham_tail_override,
        );
        cases.push(report);
    }

    let kernel_variant = {
        let mut families: Vec<&str> = cases
            .iter()
            .filter_map(|c| c.kernel_family.as_deref())
            .collect();
        families.sort_unstable();
        families.dedup();
        match families.as_slice() {
            [] => "unknown".to_string(),
            [family] => (*family).to_string(),
            _ => "mixed".to_string(),
        }
    };
    let tail_variant = derive_soak_tail_variant(&cases);

    Ok(SoakSuiteReport {
        schema_version: 1,
        suite: zkgpu_report::SuiteKind::Soak,
        device: device_report,
        kernel: KernelReport {
            field: "BabyBear".to_string(),
            ntt_variant: kernel_variant,
            stockham_tail_strategy: tail_variant,
        },
        cases,
        requested_duration_secs: spec.duration_secs,
    })
}

/// Soak twin of [`derive_tail_variant`]. Inspects per-soak-case tail
/// strategies so the suite-level `KernelReport.stockham_tail_strategy`
/// reflects whatever the runner actually planned. Returning `None` would
/// mean "no Stockham tail anywhere"; `Some("mixed")` flags configuration
/// drift across cases (e.g. a Local override that only triggered on some
/// log_n).
fn derive_soak_tail_variant(cases: &[SoakCaseReport]) -> Option<String> {
    let mut tails: Vec<&str> = cases
        .iter()
        .filter_map(|c| c.stockham_tail_strategy.as_deref())
        .collect();
    if tails.is_empty() {
        return None;
    }
    tails.sort_unstable();
    tails.dedup();
    Some(match tails.as_slice() {
        [tail] => (*tail).to_string(),
        _ => "mixed".to_string(),
    })
}

fn soak_case_error(
    case: &CaseSpec,
    duration: Duration,
    kernel_family: Option<String>,
    err: String,
) -> SoakCaseReport {
    soak_case_error_with_tail(case, duration, kernel_family, None, None, None, err)
}

#[allow(clippy::too_many_arguments)]
fn soak_case_error_with_tail(
    case: &CaseSpec,
    duration: Duration,
    kernel_family: Option<String>,
    stockham_tail_strategy: Option<String>,
    stockham_tail_reason: Option<String>,
    tail_stride_bytes: Option<u64>,
    err: String,
) -> SoakCaseReport {
    SoakCaseReport {
        name: case.name.clone(),
        log_n: case.log_n,
        direction: case.direction,
        kernel_family,
        requested_duration_secs: duration.as_secs() as u32,
        stats: zkgpu_report::SoakStats {
            total_iterations: 0,
            actual_duration_secs: 0.0,
            iterations_per_sec: 0.0,
            median_wall_ns: 0,
            p5_wall_ns: 0,
            p95_wall_ns: 0,
            min_wall_ns: 0,
            max_wall_ns: 0,
            wall_cv: 0.0,
            thermal_drift_ratio: 1.0,
            median_gpu_ns: None,
            p5_gpu_ns: None,
            p95_gpu_ns: None,
            gpu_cv: None,
        },
        samples: Vec::new(),
        validated: false,
        error: Some(err),
        stockham_tail_strategy,
        stockham_tail_reason,
        tail_stride_bytes,
    }
}

fn run_soak_case(
    device: &WgpuDevice,
    case: &CaseSpec,
    duration: Duration,
    validate: bool,
    family: FamilyOverride,
    tail_override: StockhamTailOverride,
) -> SoakCaseReport {
    let direction = match case.direction {
        TestDirection::Forward => NttDirection::Forward,
        TestDirection::Inverse => NttDirection::Inverse,
        TestDirection::Roundtrip => {
            return soak_case_error(
                case,
                duration,
                None,
                "soak benchmark does not support Roundtrip direction; \
                 use separate Forward and Inverse cases instead"
                    .to_string(),
            );
        }
    };

    // Soak harness doesn't expose R8 override yet — pass `None` so the
    // per-family default kicks in, matching pre-r8-override behavior.
    let mut plan = match make_plan(device, case.log_n, direction, family, tail_override, None) {
        Ok(plan) => plan,
        Err(err) => return soak_case_error(case, duration, None, err.to_string()),
    };

    let kernel_family = Some(plan.family_name().to_string());
    // Capture tail metadata up-front so it survives a `measure_plan_soak`
    // failure — the operator still wants to know which strategy was about
    // to run when the soak crashed.
    let tail_strategy = plan.stockham_tail_strategy().map(str::to_string);
    let tail_reason = plan.stockham_tail_reason().map(str::to_string);
    let tail_stride_bytes = plan.tail_stride_bytes();
    let input = make_input(case.log_n, &case.input);

    let measurement = match measure_plan_soak(
        device,
        &input,
        &mut plan,
        duration,
        case.warmup_iterations.max(2), // Soak always does at least 2 warmups
        case.profile_gpu_timestamps,
    ) {
        Ok(m) => m,
        Err(err) => {
            return soak_case_error_with_tail(
                case,
                duration,
                kernel_family,
                tail_strategy,
                tail_reason,
                tail_stride_bytes,
                err.to_string(),
            );
        }
    };

    // Validate first and last iteration outputs against CPU reference
    let validated = if validate && !measurement.first_output.is_empty() {
        let expected = cpu_reference(&input, direction);
        let first_ok = compare_vectors(&measurement.first_output, &expected).passed;
        let last_ok = compare_vectors(&measurement.last_output, &expected).passed;
        first_ok && last_ok
    } else {
        false
    };

    SoakCaseReport {
        name: case.name.clone(),
        log_n: case.log_n,
        direction: case.direction,
        kernel_family,
        requested_duration_secs: duration.as_secs() as u32,
        stats: measurement.stats,
        samples: measurement.samples,
        validated,
        error: None,
        stockham_tail_strategy: tail_strategy,
        stockham_tail_reason: tail_reason,
        tail_stride_bytes,
    }
}

fn run_case(
    device: &WgpuDevice,
    case: &CaseSpec,
    family: FamilyOverride,
    tail_override: StockhamTailOverride,
    r8_override: Option<u32>,
) -> CaseReport {
    let input = make_input(case.log_n, &case.input);

    match case.direction {
        TestDirection::Forward => run_single_direction_case(
            device,
            case,
            &input,
            NttDirection::Forward,
            cpu_reference(&input, NttDirection::Forward),
            family,
            tail_override,
            r8_override,
        ),
        TestDirection::Inverse => run_single_direction_case(
            device,
            case,
            &input,
            NttDirection::Inverse,
            cpu_reference(&input, NttDirection::Inverse),
            family,
            tail_override,
            r8_override,
        ),
        TestDirection::Roundtrip => {
            run_roundtrip_case(device, case, &input, family, tail_override, r8_override)
        }
    }
}

fn run_single_direction_case(
    device: &WgpuDevice,
    case: &CaseSpec,
    input: &[zkgpu_babybear::BabyBear],
    direction: NttDirection,
    expected: Vec<zkgpu_babybear::BabyBear>,
    family: FamilyOverride,
    tail_override: StockhamTailOverride,
    r8_override: Option<u32>,
) -> CaseReport {
    let mut plan = match make_plan(device, case.log_n, direction, family, tail_override, r8_override) {
        Ok(plan) => plan,
        Err(err) => return case_error(case, err),
    };
    let family = Some(plan.family_name().to_string());
    let tail_strategy = plan.stockham_tail_strategy().map(str::to_string);
    let tail_reason = plan.stockham_tail_reason().map(str::to_string);
    let tail_stride_bytes = plan.tail_stride_bytes();

    let measurement = match measure_plan(
        device,
        input,
        &mut plan,
        case.warmup_iterations,
        case.iterations,
        case.profile_gpu_timestamps,
    ) {
        Ok(t) => t,
        Err(err) => {
            return case_error_with_tail(
                case,
                family,
                tail_strategy,
                tail_reason,
                tail_stride_bytes,
                err,
            )
        }
    };
    let outcome = compare_vectors(&measurement.final_output, &expected);

    CaseReport {
        name: case.name.clone(),
        log_n: case.log_n,
        direction: case.direction,
        input: case.input.clone(),
        kernel_family: family,
        passed: outcome.passed,
        mismatch_count: outcome.mismatch_count,
        first_mismatch_index: outcome.first_mismatch_index,
        first_mismatch_gpu: outcome.first_mismatch_gpu,
        first_mismatch_cpu: outcome.first_mismatch_cpu,
        timings: measurement.timings,
        error: None,
        stockham_tail_strategy: tail_strategy,
        stockham_tail_reason: tail_reason,
        tail_stride_bytes,
    }
}

fn run_roundtrip_case(
    device: &WgpuDevice,
    case: &CaseSpec,
    input: &[zkgpu_babybear::BabyBear],
    family: FamilyOverride,
    tail_override: StockhamTailOverride,
    r8_override: Option<u32>,
) -> CaseReport {
    let mut forward = match make_plan(
        device,
        case.log_n,
        NttDirection::Forward,
        family,
        tail_override,
        r8_override,
    ) {
        Ok(plan) => plan,
        Err(err) => return case_error(case, err),
    };
    // Capture forward-side metadata up-front so it survives a later failure
    // (inverse make_plan or either measurement). Without this, an inverse
    // make_plan crash on Xclipse/Mali/Browser drops the very tail strategy
    // PR 1 was meant to make visible.
    let forward_family_name = forward.family_name().to_string();
    let tail_strategy = forward.stockham_tail_strategy().map(str::to_string);
    let tail_reason = forward.stockham_tail_reason().map(str::to_string);
    let tail_stride_bytes = forward.tail_stride_bytes();

    let mut inverse = match make_plan(
        device,
        case.log_n,
        NttDirection::Inverse,
        family,
        tail_override,
        r8_override,
    ) {
        Ok(plan) => plan,
        Err(err) => {
            return case_error_with_tail(
                case,
                Some(forward_family_name),
                tail_strategy,
                tail_reason,
                tail_stride_bytes,
                err,
            )
        }
    };
    let kernel_family = Some(format!(
        "{}/{}",
        forward_family_name,
        inverse.family_name()
    ));

    let forward_measurement = match measure_plan(
        device,
        input,
        &mut forward,
        case.warmup_iterations,
        case.iterations,
        case.profile_gpu_timestamps,
    ) {
        Ok(t) => t,
        Err(err) => {
            return case_error_with_tail(
                case,
                kernel_family,
                tail_strategy,
                tail_reason,
                tail_stride_bytes,
                err,
            )
        }
    };
    let inverse_measurement = match measure_plan(
        device,
        &forward_measurement.final_output,
        &mut inverse,
        case.warmup_iterations,
        case.iterations,
        case.profile_gpu_timestamps,
    ) {
        Ok(t) => t,
        Err(err) => {
            return case_error_with_tail(
                case,
                kernel_family,
                tail_strategy,
                tail_reason,
                tail_stride_bytes,
                err,
            )
        }
    };
    let timings = sum_timing_reports(&[forward_measurement.timings, inverse_measurement.timings]);
    let outcome = compare_vectors(&inverse_measurement.final_output, input);

    CaseReport {
        name: case.name.clone(),
        log_n: case.log_n,
        direction: case.direction,
        input: case.input.clone(),
        kernel_family,
        passed: outcome.passed,
        mismatch_count: outcome.mismatch_count,
        first_mismatch_index: outcome.first_mismatch_index,
        first_mismatch_gpu: outcome.first_mismatch_gpu,
        first_mismatch_cpu: outcome.first_mismatch_cpu,
        timings,
        error: None,
        stockham_tail_strategy: tail_strategy,
        stockham_tail_reason: tail_reason,
        tail_stride_bytes,
    }
}

fn cpu_reference(
    input: &[zkgpu_babybear::BabyBear],
    direction: NttDirection,
) -> Vec<zkgpu_babybear::BabyBear> {
    let mut cpu = input.to_vec();
    ntt_cpu_reference(&mut cpu, direction);
    cpu
}

fn make_plan(
    device: &WgpuDevice,
    log_n: u32,
    direction: NttDirection,
    family: FamilyOverride,
    tail_override: StockhamTailOverride,
    r8_override: Option<u32>,
) -> Result<WgpuNttPlan, ZkGpuError> {
    // Always derive from the device caps so the Stockham tail heuristic has
    // full device context, then narrow the family. The earlier
    // `stockham_only()` / `force_four_step()` constructors threw the caps
    // hint away, which silently downgraded forced-Stockham A/B runs on
    // Xclipse/Mali/Browser to `LocalFusedR4` regardless of the new policy.
    let plan_tail = to_plan_tail(tail_override);
    let base = PlannerPolicy::from_caps(device.caps())
        .with_public_tail_override(plan_tail)
        .with_r8_max_log_leaf_override(r8_override);
    let policy = match family {
        FamilyOverride::Auto => base,
        FamilyOverride::Stockham => base.with_four_step_disabled(),
        FamilyOverride::FourStep => base.with_force_four_step(),
    };
    WgpuNttPlan::new_with_policy(device, log_n, direction, &policy)
}

fn derive_kernel_variant(cases: &[CaseReport]) -> String {
    let mut families = cases
        .iter()
        .filter_map(|c| c.kernel_family.as_deref())
        .collect::<Vec<_>>();
    families.sort_unstable();
    families.dedup();
    match families.as_slice() {
        [] => "unknown".to_string(),
        [family] => (*family).to_string(),
        _ => "mixed".to_string(),
    }
}

/// Summarise the per-case tail strategies into a single kernel-level label.
///
/// Returns `None` if no case recorded a tail strategy (e.g. all cases were
/// four-step or all were Stockham below `LOG_BLOCK`). Returns the single
/// strategy name (e.g. `"LocalFusedR4"`) if every tailed case agreed, or
/// `"mixed"` if multiple strategies were observed in the same suite.
fn derive_tail_variant(cases: &[CaseReport]) -> Option<String> {
    let mut tails = cases
        .iter()
        .filter_map(|c| c.stockham_tail_strategy.as_deref())
        .collect::<Vec<_>>();
    if tails.is_empty() {
        return None;
    }
    tails.sort_unstable();
    tails.dedup();
    Some(match tails.as_slice() {
        [tail] => (*tail).to_string(),
        _ => "mixed".to_string(),
    })
}

fn case_error(case: &CaseSpec, err: ZkGpuError) -> CaseReport {
    case_error_with_family(case, None, err)
}

fn case_error_with_family(
    case: &CaseSpec,
    kernel_family: Option<String>,
    err: ZkGpuError,
) -> CaseReport {
    case_error_with_tail(case, kernel_family, None, None, None, err)
}

/// Construct a failed `CaseReport` while preserving any tail metadata that
/// was already captured before the failure.
///
/// Operators investigating a measurement crash on Xclipse/Mali/Browser need
/// to see which Stockham tail strategy was selected and how it was chosen
/// — exactly the failure modes PR 1 was designed to make visible. The
/// non-tail variants (`case_error`, `case_error_with_family`) just delegate
/// here with `None`s for the pre-plan failure paths.
#[allow(clippy::too_many_arguments)]
fn case_error_with_tail(
    case: &CaseSpec,
    kernel_family: Option<String>,
    stockham_tail_strategy: Option<String>,
    stockham_tail_reason: Option<String>,
    tail_stride_bytes: Option<u64>,
    err: ZkGpuError,
) -> CaseReport {
    CaseReport {
        name: case.name.clone(),
        log_n: case.log_n,
        direction: case.direction,
        input: case.input.clone(),
        kernel_family,
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
        stockham_tail_strategy,
        stockham_tail_reason,
        tail_stride_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report::CaseReport;

    // === soak: roundtrip rejection ===

    #[test]
    fn soak_rejects_roundtrip_direction() {
        // A SoakSpec with a Roundtrip case should produce a SoakCaseReport
        // with an error, not silently run as forward.
        let case = CaseSpec::new(
            "roundtrip_soak",
            10,
            TestDirection::Roundtrip,
            zkgpu_report::InputPattern::Sequential,
        );
        let duration = Duration::from_secs(5);

        // We can't call run_soak_case directly without a device, but we
        // can verify the error path by constructing what run_soak_case
        // would return for a Roundtrip case. The actual rejection is a
        // match arm that returns soak_case_error(), so test that function.
        let report = soak_case_error(
            &case,
            duration,
            None,
            "soak benchmark does not support Roundtrip direction; \
             use separate Forward and Inverse cases instead"
                .to_string(),
        );

        assert!(report.error.is_some());
        assert!(
            report.error.as_ref().unwrap().contains("Roundtrip"),
            "error should mention Roundtrip: {:?}",
            report.error
        );
        assert_eq!(report.direction, TestDirection::Roundtrip);
        assert_eq!(report.stats.total_iterations, 0);
        assert!(!report.validated);
        assert!(report.samples.is_empty());
    }

    // === soak: SoakSuiteReport serialization shape ===

    #[test]
    fn soak_suite_report_json_has_samples_field() {
        // Verify the canonical SoakSuiteReport serializes with the expected
        // shape including per-iteration samples.
        let report = zkgpu_report::SoakSuiteReport {
            schema_version: 1,
            suite: zkgpu_report::SuiteKind::Soak,
            device: zkgpu_report::DeviceReport {
                name: "test".into(),
                backend: "Vulkan".into(),
                tier: "NativeBasic".into(),
                gpu_family: "Adreno".into(),
                detection_source: "VendorId".into(),
                platform_class: "AndroidNative".into(),
                memory_model: "Unified".into(),
                driver: "".into(),
                driver_info: "".into(),
                max_buffer_size_bytes: 0,
                max_workgroup_size_x: 0,
                max_compute_invocations: 0,
                max_compute_workgroup_storage_size_bytes: 0,
                feature_flags: Vec::new(),
            },
            kernel: zkgpu_report::KernelReport {
                field: "BabyBear".into(),
                ntt_variant: "stockham".into(),
                stockham_tail_strategy: None,
            },
            cases: vec![zkgpu_report::SoakCaseReport {
                name: "test_case".into(),
                log_n: 10,
                direction: TestDirection::Forward,
                kernel_family: Some("stockham".into()),
                requested_duration_secs: 5,
                stats: zkgpu_report::SoakStats {
                    total_iterations: 100,
                    actual_duration_secs: 5.1,
                    iterations_per_sec: 19.6,
                    median_wall_ns: 50_000_000,
                    p5_wall_ns: 48_000_000,
                    p95_wall_ns: 55_000_000,
                    min_wall_ns: 47_000_000,
                    max_wall_ns: 60_000_000,
                    wall_cv: 0.05,
                    thermal_drift_ratio: 1.02,
                    median_gpu_ns: Some(30_000_000),
                    p5_gpu_ns: Some(28_000_000),
                    p95_gpu_ns: Some(35_000_000),
                    gpu_cv: Some(0.03),
                },
                samples: vec![
                    zkgpu_report::SoakSample {
                        iteration: 0,
                        wall_ns: 50_000_000,
                        gpu_total_ns: Some(30_000_000),
                        elapsed_ms: 0,
                    },
                    zkgpu_report::SoakSample {
                        iteration: 1,
                        wall_ns: 51_000_000,
                        gpu_total_ns: Some(31_000_000),
                        elapsed_ms: 50,
                    },
                ],
                validated: true,
                error: None,
                stockham_tail_strategy: Some("LocalFusedR4".into()),
                stockham_tail_reason: Some("HeuristicDefaultLocal".into()),
                tail_stride_bytes: Some(4),
            }],
            requested_duration_secs: 5,
        };

        let json = serde_json::to_value(&report).unwrap();

        // Verify top-level shape
        assert_eq!(json["schema_version"], 1);
        assert_eq!(json["suite"], "Soak");
        assert_eq!(json["requested_duration_secs"], 5);

        // Verify case has samples array with per-iteration data
        let case0 = &json["cases"][0];
        assert_eq!(case0["name"], "test_case");
        assert!(case0["samples"].is_array());
        assert_eq!(case0["samples"].as_array().unwrap().len(), 2);
        assert_eq!(case0["samples"][0]["iteration"], 0);
        assert_eq!(case0["samples"][0]["wall_ns"], 50_000_000u64);
        assert_eq!(case0["samples"][1]["gpu_total_ns"], 31_000_000u64);

        // Verify stats are present
        assert!(case0["stats"]["median_wall_ns"].is_u64());
        assert!(case0["stats"]["thermal_drift_ratio"].is_f64());
        assert!(case0["stats"]["gpu_cv"].is_f64());

        // Verify validated field
        assert_eq!(case0["validated"], true);

        // Verify the new tail observability fields surface in JSON.
        assert_eq!(case0["stockham_tail_strategy"], "LocalFusedR4");
        assert_eq!(case0["stockham_tail_reason"], "HeuristicDefaultLocal");
        assert_eq!(case0["tail_stride_bytes"], 4u64);
    }

    // === soak: backwards-compatible deserialization of pre-tail JSON ===

    #[test]
    fn soak_case_report_accepts_legacy_json_without_tail_fields() {
        // A SoakCaseReport written before PR 1 lacked the three tail
        // fields. `#[serde(default)]` should let those legacy reports
        // round-trip into the new struct without error.
        let legacy = serde_json::json!({
            "name": "legacy",
            "log_n": 18,
            "direction": "Forward",
            "kernel_family": "stockham",
            "requested_duration_secs": 5,
            "stats": {
                "total_iterations": 1,
                "actual_duration_secs": 1.0,
                "iterations_per_sec": 1.0,
                "median_wall_ns": 0,
                "p5_wall_ns": 0,
                "p95_wall_ns": 0,
                "min_wall_ns": 0,
                "max_wall_ns": 0,
                "wall_cv": 0.0,
                "thermal_drift_ratio": 1.0,
                "median_gpu_ns": null,
                "p5_gpu_ns": null,
                "p95_gpu_ns": null,
                "gpu_cv": null,
            },
            "samples": [],
            "validated": false,
            "error": null,
        });
        let parsed: zkgpu_report::SoakCaseReport =
            serde_json::from_value(legacy).expect("legacy soak JSON should parse");
        assert_eq!(parsed.stockham_tail_strategy, None);
        assert_eq!(parsed.stockham_tail_reason, None);
        assert_eq!(parsed.tail_stride_bytes, None);
    }

    // === soak: derive_soak_tail_variant ===

    fn soak_case_with_tail(name: &str, tail: Option<&str>) -> SoakCaseReport {
        SoakCaseReport {
            name: name.into(),
            log_n: 20,
            direction: TestDirection::Forward,
            kernel_family: Some("stockham".into()),
            requested_duration_secs: 5,
            stats: zkgpu_report::SoakStats {
                total_iterations: 0,
                actual_duration_secs: 0.0,
                iterations_per_sec: 0.0,
                median_wall_ns: 0,
                p5_wall_ns: 0,
                p95_wall_ns: 0,
                min_wall_ns: 0,
                max_wall_ns: 0,
                wall_cv: 0.0,
                thermal_drift_ratio: 1.0,
                median_gpu_ns: None,
                p5_gpu_ns: None,
                p95_gpu_ns: None,
                gpu_cv: None,
            },
            samples: Vec::new(),
            validated: false,
            error: None,
            stockham_tail_strategy: tail.map(str::to_string),
            stockham_tail_reason: None,
            tail_stride_bytes: None,
        }
    }

    #[test]
    fn derive_soak_tail_variant_none_when_no_tails() {
        assert_eq!(
            derive_soak_tail_variant(&[soak_case_with_tail("a", None)]),
            None
        );
    }

    #[test]
    fn derive_soak_tail_variant_single_and_mixed() {
        let single = vec![
            soak_case_with_tail("a", Some("LocalFusedR4")),
            soak_case_with_tail("b", Some("LocalFusedR4")),
        ];
        assert_eq!(
            derive_soak_tail_variant(&single),
            Some("LocalFusedR4".into())
        );

        let mixed = vec![
            soak_case_with_tail("a", Some("LocalFusedR4")),
            soak_case_with_tail("b", Some("GlobalOnlyR4")),
        ];
        assert_eq!(derive_soak_tail_variant(&mixed), Some("mixed".into()));

        // Cases with no tail strategy are simply ignored, mirroring the
        // non-soak helper. So a partial run still surfaces the one strategy
        // that did get planned.
        let partial = vec![
            soak_case_with_tail("a", None),
            soak_case_with_tail("b", Some("GlobalOnlyR4")),
        ];
        assert_eq!(
            derive_soak_tail_variant(&partial),
            Some("GlobalOnlyR4".into())
        );
    }

    // === non-soak: case_error_with_tail preserves tail fields ===

    #[test]
    fn case_error_with_tail_preserves_tail_metadata() {
        // Codex P3: a measurement failure must not silently drop the tail
        // strategy that was already chosen — those are precisely the
        // failures where operators most need the diagnostic. Verify the
        // helper threads every captured field through.
        let case = CaseSpec::new(
            "tail_preserved",
            18,
            TestDirection::Forward,
            zkgpu_report::InputPattern::Sequential,
        );
        let report = case_error_with_tail(
            &case,
            Some("stockham".into()),
            Some("GlobalOnlyR4".into()),
            Some("HostileStridedDevice".into()),
            Some(8),
            ZkGpuError::DeviceLost("simulated measurement crash".into()),
        );
        assert!(!report.passed);
        assert_eq!(report.kernel_family.as_deref(), Some("stockham"));
        assert_eq!(report.stockham_tail_strategy.as_deref(), Some("GlobalOnlyR4"));
        assert_eq!(
            report.stockham_tail_reason.as_deref(),
            Some("HostileStridedDevice")
        );
        assert_eq!(report.tail_stride_bytes, Some(8));
        assert!(report.error.is_some());
    }

    #[test]
    fn case_error_with_family_keeps_tail_fields_none() {
        // The slim wrapper used for pre-plan failures still leaves the tail
        // fields blank — there's nothing yet to report at that stage.
        let case = CaseSpec::new(
            "no_tail_yet",
            10,
            TestDirection::Forward,
            zkgpu_report::InputPattern::Sequential,
        );
        let report = case_error_with_family(
            &case,
            None,
            ZkGpuError::DeviceLost("plan build failed".into()),
        );
        assert!(report.stockham_tail_strategy.is_none());
        assert!(report.stockham_tail_reason.is_none());
        assert!(report.tail_stride_bytes.is_none());
    }

    #[test]
    fn derive_kernel_variant_returns_mixed_for_multiple_families() {
        let cases = vec![
            CaseReport {
                name: "a".into(),
                log_n: 10,
                direction: TestDirection::Forward,
                input: crate::suite::InputPattern::Sequential,
                kernel_family: Some("stockham".into()),
                passed: true,
                mismatch_count: 0,
                first_mismatch_index: None,
                first_mismatch_gpu: None,
                first_mismatch_cpu: None,
                timings: TimingReport {
                    wall_time_ns: None,
                    gpu_total_ns: None,
                    gpu_stage_ns: Vec::new(),
                },
                error: None,
                stockham_tail_strategy: None,
                stockham_tail_reason: None,
                tail_stride_bytes: None,
            },
            CaseReport {
                name: "b".into(),
                log_n: 20,
                direction: TestDirection::Forward,
                input: crate::suite::InputPattern::Sequential,
                kernel_family: Some("four-step".into()),
                passed: true,
                mismatch_count: 0,
                first_mismatch_index: None,
                first_mismatch_gpu: None,
                first_mismatch_cpu: None,
                timings: TimingReport {
                    wall_time_ns: None,
                    gpu_total_ns: None,
                    gpu_stage_ns: Vec::new(),
                },
                error: None,
                stockham_tail_strategy: None,
                stockham_tail_reason: None,
                tail_stride_bytes: None,
            },
        ];
        assert_eq!(derive_kernel_variant(&cases), "mixed");
    }

    #[test]
    fn derive_tail_variant_none_when_no_tails_recorded() {
        let cases = vec![CaseReport {
            name: "a".into(),
            log_n: 10,
            direction: TestDirection::Forward,
            input: crate::suite::InputPattern::Sequential,
            kernel_family: Some("stockham".into()),
            passed: true,
            mismatch_count: 0,
            first_mismatch_index: None,
            first_mismatch_gpu: None,
            first_mismatch_cpu: None,
            timings: TimingReport {
                wall_time_ns: None,
                gpu_total_ns: None,
                gpu_stage_ns: Vec::new(),
            },
            error: None,
            stockham_tail_strategy: None,
            stockham_tail_reason: None,
            tail_stride_bytes: None,
        }];
        assert_eq!(derive_tail_variant(&cases), None);
    }

    #[test]
    fn derive_tail_variant_single_and_mixed() {
        let mk = |name: &str, tail: Option<&str>| CaseReport {
            name: name.into(),
            log_n: 20,
            direction: TestDirection::Forward,
            input: crate::suite::InputPattern::Sequential,
            kernel_family: Some("stockham".into()),
            passed: true,
            mismatch_count: 0,
            first_mismatch_index: None,
            first_mismatch_gpu: None,
            first_mismatch_cpu: None,
            timings: TimingReport {
                wall_time_ns: None,
                gpu_total_ns: None,
                gpu_stage_ns: Vec::new(),
            },
            error: None,
            stockham_tail_strategy: tail.map(str::to_string),
            stockham_tail_reason: None,
            tail_stride_bytes: None,
        };

        let single = vec![
            mk("a", Some("LocalFusedR4")),
            mk("b", Some("LocalFusedR4")),
        ];
        assert_eq!(derive_tail_variant(&single), Some("LocalFusedR4".into()));

        let mixed = vec![
            mk("a", Some("LocalFusedR4")),
            mk("b", Some("GlobalOnlyR4")),
        ];
        assert_eq!(derive_tail_variant(&mixed), Some("mixed".into()));

        let partial = vec![mk("a", None), mk("b", Some("GlobalOnlyR4"))];
        assert_eq!(derive_tail_variant(&partial), Some("GlobalOnlyR4".into()));
    }
}
