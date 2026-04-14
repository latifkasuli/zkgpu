use std::time::Duration;

use zkgpu_core::{NttDirection, ZkGpuError};
use zkgpu_ntt::ntt_cpu_reference;
use zkgpu_report::{SoakCaseReport, SoakSpec, SoakSuiteReport};
use zkgpu_wgpu::{PlannerPolicy, WgpuDevice, WgpuNttPlan};

use crate::benchmark::{measure_plan, measure_plan_soak, sum_timing_reports};
use crate::case::{CaseSpec, TestDirection};
use crate::device::build_device_report;
use crate::inputs::make_input;
use crate::report::{CaseReport, KernelReport, SuiteReport, SuiteSummary, TimingReport};
use crate::suite::{benchmark_suite, smoke_suite, validation_suite, FamilyOverride, SuiteSpec};
use crate::validation::compare_vectors;
use crate::TestkitError;

pub fn run_suite(spec: &SuiteSpec) -> Result<SuiteReport, TestkitError> {
    if spec.cases.is_empty() {
        return Err(TestkitError::EmptySuite);
    }

    let device = WgpuDevice::new().map_err(|e| TestkitError::DeviceInit(e.to_string()))?;
    let device_report = build_device_report(&device);
    let mut cases = Vec::with_capacity(spec.cases.len());

    for case in &spec.cases {
        let report = run_case(&device, case, spec.family_override);
        let failed = !report.passed;
        cases.push(report);
        if spec.fail_fast && failed {
            break;
        }
    }

    let passed_cases = cases.iter().filter(|c| c.passed).count() as u32;
    let failed_cases = cases.len() as u32 - passed_cases;
    let kernel_variant = derive_kernel_variant(&cases);
    let total_cases = cases.len() as u32;

    Ok(SuiteReport {
        schema_version: 1,
        suite: spec.kind,
        device: device_report,
        kernel: KernelReport {
            field: "BabyBear".to_string(),
            ntt_variant: kernel_variant,
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
        let report = run_soak_case(&device, case, duration, spec.validate, spec.family_override);
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

    Ok(SoakSuiteReport {
        schema_version: 1,
        suite: zkgpu_report::SuiteKind::Soak,
        device: device_report,
        kernel: KernelReport {
            field: "BabyBear".to_string(),
            ntt_variant: kernel_variant,
        },
        cases,
        requested_duration_secs: spec.duration_secs,
    })
}

fn soak_case_error(
    case: &CaseSpec,
    duration: Duration,
    kernel_family: Option<String>,
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
    }
}

fn run_soak_case(
    device: &WgpuDevice,
    case: &CaseSpec,
    duration: Duration,
    validate: bool,
    family: FamilyOverride,
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

    let mut plan = match make_plan(device, case.log_n, direction, family) {
        Ok(plan) => plan,
        Err(err) => return soak_case_error(case, duration, None, err.to_string()),
    };

    let kernel_family = Some(plan.family_name().to_string());
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
            return soak_case_error(case, duration, kernel_family, err.to_string());
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
    }
}

fn run_case(device: &WgpuDevice, case: &CaseSpec, family: FamilyOverride) -> CaseReport {
    let input = make_input(case.log_n, &case.input);

    match case.direction {
        TestDirection::Forward => run_single_direction_case(
            device,
            case,
            &input,
            NttDirection::Forward,
            cpu_reference(&input, NttDirection::Forward),
            family,
        ),
        TestDirection::Inverse => run_single_direction_case(
            device,
            case,
            &input,
            NttDirection::Inverse,
            cpu_reference(&input, NttDirection::Inverse),
            family,
        ),
        TestDirection::Roundtrip => run_roundtrip_case(device, case, &input, family),
    }
}

fn run_single_direction_case(
    device: &WgpuDevice,
    case: &CaseSpec,
    input: &[zkgpu_babybear::BabyBear],
    direction: NttDirection,
    expected: Vec<zkgpu_babybear::BabyBear>,
    family: FamilyOverride,
) -> CaseReport {
    let mut plan = match make_plan(device, case.log_n, direction, family) {
        Ok(plan) => plan,
        Err(err) => return case_error(case, err),
    };
    let family = Some(plan.family_name().to_string());

    let measurement = match measure_plan(
        device,
        input,
        &mut plan,
        case.warmup_iterations,
        case.iterations,
        case.profile_gpu_timestamps,
    ) {
        Ok(t) => t,
        Err(err) => return case_error_with_family(case, family, err),
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
    }
}

fn run_roundtrip_case(
    device: &WgpuDevice,
    case: &CaseSpec,
    input: &[zkgpu_babybear::BabyBear],
    family: FamilyOverride,
) -> CaseReport {
    let mut forward = match make_plan(device, case.log_n, NttDirection::Forward, family) {
        Ok(plan) => plan,
        Err(err) => return case_error(case, err),
    };
    let mut inverse = match make_plan(device, case.log_n, NttDirection::Inverse, family) {
        Ok(plan) => plan,
        Err(err) => return case_error(case, err),
    };
    let kernel_family = Some(format!(
        "{}/{}",
        forward.family_name(),
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
        Err(err) => return case_error_with_family(case, kernel_family, err),
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
        Err(err) => return case_error_with_family(case, kernel_family, err),
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
) -> Result<WgpuNttPlan, ZkGpuError> {
    match family {
        FamilyOverride::Auto => WgpuNttPlan::new(device, log_n, direction),
        FamilyOverride::Stockham => {
            let policy = PlannerPolicy::stockham_only();
            WgpuNttPlan::new_with_policy(device, log_n, direction, &policy)
        }
        FamilyOverride::FourStep => {
            let policy = PlannerPolicy::force_four_step();
            WgpuNttPlan::new_with_policy(device, log_n, direction, &policy)
        }
    }
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

fn case_error(case: &CaseSpec, err: ZkGpuError) -> CaseReport {
    case_error_with_family(case, None, err)
}

fn case_error_with_family(
    case: &CaseSpec,
    kernel_family: Option<String>,
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
            },
        ];
        assert_eq!(derive_kernel_variant(&cases), "mixed");
    }
}
