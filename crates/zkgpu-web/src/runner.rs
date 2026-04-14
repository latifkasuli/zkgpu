//! Async test runner for browser WebGPU.
//!
//! Uses the async API surface (`execute_kernels_async`,
//! `read_to_vec_async`, etc.) so it never blocks the browser event loop.
//!
//! Device ownership: the runner clones an `Rc<WgpuDevice>` from the
//! thread-local via `device::clone_device()`. The `Rc` keeps the device
//! alive across await points — no raw pointers, no dangling risk even
//! if a second `gpu_init()` replaces the global device.
//!
//! Benchmark semantics: `warmup_iterations` + `iterations` match the
//! native runner's `measure_plan`. Warmup runs are discarded; measured
//! runs are averaged for wall and GPU timing. Roundtrip cases honour
//! the caller's warmup/iteration/profiling flags for both directions,
//! then `sum_timing_reports` to match native.

use std::rc::Rc;

use zkgpu_babybear::BabyBear;
use zkgpu_core::{GpuDevice, NttDirection};
use zkgpu_ntt::ntt_cpu_reference;
use zkgpu_report::{
    CaseReport, CaseSpec, DeviceReport, FamilyOverride, KernelReport, StageTimingReport,
    SuiteReport, SuiteSpec, SuiteSummary, TestDirection, TimingReport,
};
use zkgpu_wgpu::{NttTimings, PlannerPolicy, WgpuDevice, WgpuNttPlan};

use crate::device;
use crate::inputs::make_input;
use crate::validation::compare_vectors;

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Run a full suite asynchronously.
pub(crate) async fn run_suite_async(spec: &SuiteSpec) -> Result<SuiteReport, String> {
    if spec.cases.is_empty() {
        return Err("suite must contain at least one case".to_string());
    }

    let dev = device::clone_device()?;
    let device_report = build_device_report_from(&dev);
    let mut cases = Vec::with_capacity(spec.cases.len());

    for case in &spec.cases {
        let report = run_case_inner(case, spec.family_override, &dev).await;
        let failed = !report.passed;
        cases.push(report);
        if spec.fail_fast && failed {
            break;
        }
    }

    let passed_cases = cases.iter().filter(|c| c.passed).count() as u32;
    let total_cases = cases.len() as u32;
    let failed_cases = total_cases - passed_cases;
    let kernel_variant = derive_kernel_variant(&cases);

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

/// Run a single case exposed to JS.
pub(crate) async fn run_single_case_async(case: &CaseSpec) -> Result<CaseReport, String> {
    let dev = device::clone_device()?;
    Ok(run_case_inner(case, FamilyOverride::Auto, &dev).await)
}

// ---------------------------------------------------------------------------
// Case dispatch
// ---------------------------------------------------------------------------

async fn run_case_inner(
    case: &CaseSpec,
    family: FamilyOverride,
    dev: &Rc<WgpuDevice>,
) -> CaseReport {
    let input = make_input(case.log_n, &case.input);

    match case.direction {
        TestDirection::Forward => run_direction(case, &input, NttDirection::Forward, family, dev).await,
        TestDirection::Inverse => run_direction(case, &input, NttDirection::Inverse, family, dev).await,
        TestDirection::Roundtrip => run_roundtrip(case, &input, family, dev).await,
    }
}

// ---------------------------------------------------------------------------
// Single-direction case
// ---------------------------------------------------------------------------

async fn run_direction(
    case: &CaseSpec,
    input: &[BabyBear],
    direction: NttDirection,
    family: FamilyOverride,
    dev: &Rc<WgpuDevice>,
) -> CaseReport {
    let expected = cpu_reference(input, direction);

    let mut plan = match make_plan(dev, case.log_n, direction, family) {
        Ok(p) => p,
        Err(e) => return case_error(case, e.to_string()),
    };
    let family_name = plan.family_name().to_string();

    let measurement = match measure_plan_async(
        dev,
        input,
        &mut plan,
        case.warmup_iterations,
        case.iterations,
        case.profile_gpu_timestamps,
    )
    .await
    {
        Ok(m) => m,
        Err(e) => return case_error_family(case, Some(family_name), e),
    };

    let outcome = compare_vectors(&measurement.final_output, &expected);

    CaseReport {
        name: case.name.clone(),
        log_n: case.log_n,
        direction: case.direction,
        input: case.input.clone(),
        kernel_family: Some(family_name),
        passed: outcome.passed,
        mismatch_count: outcome.mismatch_count,
        first_mismatch_index: outcome.first_mismatch_index,
        first_mismatch_gpu: outcome.first_mismatch_gpu,
        first_mismatch_cpu: outcome.first_mismatch_cpu,
        timings: measurement.timings,
        error: None,
    }
}

// ---------------------------------------------------------------------------
// Roundtrip case — mirrors native run_roundtrip_case exactly
// ---------------------------------------------------------------------------

async fn run_roundtrip(
    case: &CaseSpec,
    input: &[BabyBear],
    family: FamilyOverride,
    dev: &Rc<WgpuDevice>,
) -> CaseReport {
    // Create both plans up-front.
    let mut forward_plan = match make_plan(dev, case.log_n, NttDirection::Forward, family) {
        Ok(p) => p,
        Err(e) => return case_error(case, e.to_string()),
    };
    let mut inverse_plan = match make_plan(dev, case.log_n, NttDirection::Inverse, family) {
        Ok(p) => p,
        Err(e) => return case_error(case, e.to_string()),
    };
    let kernel_family = Some(format!(
        "{}/{}",
        forward_plan.family_name(),
        inverse_plan.family_name(),
    ));

    // Measure forward pass with the caller's warmup/iteration/profiling flags.
    let forward_measurement = match measure_plan_async(
        dev,
        input,
        &mut forward_plan,
        case.warmup_iterations,
        case.iterations,
        case.profile_gpu_timestamps,
    )
    .await
    {
        Ok(m) => m,
        Err(e) => return case_error_family(case, kernel_family, e),
    };

    // Measure inverse pass using forward's final output as input,
    // with the same warmup/iteration/profiling settings.
    let inverse_measurement = match measure_plan_async(
        dev,
        &forward_measurement.final_output,
        &mut inverse_plan,
        case.warmup_iterations,
        case.iterations,
        case.profile_gpu_timestamps,
    )
    .await
    {
        Ok(m) => m,
        Err(e) => return case_error_family(case, kernel_family, e),
    };

    // Sum timing reports (same as native sum_timing_reports).
    let timings = sum_timing_reports(&[forward_measurement.timings, inverse_measurement.timings]);

    // Correctness check: inverse(forward(input)) == input.
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

// ---------------------------------------------------------------------------
// Async measure_plan — mirrors native benchmark::measure_plan
// ---------------------------------------------------------------------------

/// Output of a plan measurement (async equivalent of native PlanMeasurement).
struct PlanMeasurement {
    timings: TimingReport,
    final_output: Vec<BabyBear>,
}

/// Async equivalent of `benchmark::measure_plan`.
///
/// Runs `warmup_iterations` discarded passes, then `iterations` measured
/// passes. Wall and GPU timings are averaged across measured runs. The
/// final output is read back from the last measured iteration.
async fn measure_plan_async(
    dev: &Rc<WgpuDevice>,
    input: &[BabyBear],
    plan: &mut WgpuNttPlan,
    warmup_iterations: u32,
    iterations: u32,
    profile_gpu_timestamps: bool,
) -> Result<PlanMeasurement, String> {
    // --- Warmup iterations (discarded) ---
    for _ in 0..warmup_iterations {
        let mut warmup_buf = dev
            .upload(input)
            .map_err(|e| e.to_string())?;
        if profile_gpu_timestamps {
            plan.execute_kernels_profiled_async(dev, &mut warmup_buf)
                .await
                .map_err(|e| e.to_string())?;
        } else {
            plan.execute_kernels_async(dev, &mut warmup_buf)
                .await
                .map_err(|e| e.to_string())?;
        }
    }

    // --- Measured iterations ---
    let measured = iterations.max(1);
    let mut wall_total_ns: u64 = 0;
    let mut gpu_total_ns_accum: f64 = 0.0;
    let mut stage_totals: Option<Vec<(String, f64)>> = None;
    let mut profiled_samples: u32 = 0;
    let mut final_output: Option<Vec<BabyBear>> = None;

    for iter_idx in 0..measured {
        let mut buf = dev
            .upload(input)
            .map_err(|e| e.to_string())?;

        let wall_start = web_time::Instant::now();
        let exec_result = if profile_gpu_timestamps {
            plan.execute_kernels_profiled_async(dev, &mut buf).await
        } else {
            match plan.execute_kernels_async(dev, &mut buf).await {
                Ok(()) => Ok(None),
                Err(e) => Err(e),
            }
        };
        wall_total_ns += wall_start.elapsed().as_nanos() as u64;

        let ntt_timings = exec_result.map_err(|e| e.to_string())?;

        // Accumulate profiled GPU timings.
        if let Some(ref timings) = ntt_timings {
            gpu_total_ns_accum += timings.gpu_total_ns;
            accumulate_profiled(&mut stage_totals, timings);
            profiled_samples += 1;
        }

        // Read back on last measured iteration.
        if iter_idx + 1 == measured {
            final_output = Some(
                buf.read_to_vec_async()
                    .await
                    .map_err(|e| e.to_string())?,
            );
        }
    }

    let wall_avg_ns = wall_total_ns / measured as u64;
    let timings = build_averaged_timing(wall_avg_ns, profiled_samples, gpu_total_ns_accum, stage_totals);

    Ok(PlanMeasurement {
        timings,
        final_output: final_output.expect("measured loop always captures final output"),
    })
}

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

/// Accumulate per-stage profiled GPU timings across iterations.
/// Mirrors `benchmark::accumulate_profiled` in the native runner.
fn accumulate_profiled(stage_totals: &mut Option<Vec<(String, f64)>>, timings: &NttTimings) {
    let accum = stage_totals.get_or_insert_with(|| {
        timings
            .gpu_stage_ns
            .iter()
            .map(|s| (s.label.clone(), 0.0))
            .collect()
    });

    for (acc, sample) in accum.iter_mut().zip(timings.gpu_stage_ns.iter()) {
        acc.1 += sample.duration_ns;
    }
}

/// Build an averaged timing report from accumulated measurements.
fn build_averaged_timing(
    wall_avg_ns: u64,
    profiled_samples: u32,
    gpu_total_ns_accum: f64,
    stage_totals: Option<Vec<(String, f64)>>,
) -> TimingReport {
    if profiled_samples > 0 {
        let divisor = profiled_samples as f64;
        TimingReport {
            wall_time_ns: Some(wall_avg_ns),
            gpu_total_ns: Some((gpu_total_ns_accum / divisor) as u64),
            gpu_stage_ns: stage_totals
                .unwrap_or_default()
                .into_iter()
                .map(|(label, total)| StageTimingReport {
                    label,
                    duration_ns: (total / divisor) as u64,
                })
                .collect(),
        }
    } else {
        TimingReport {
            wall_time_ns: Some(wall_avg_ns),
            gpu_total_ns: None,
            gpu_stage_ns: Vec::new(),
        }
    }
}

/// Sum timing reports from multiple passes (forward + inverse for roundtrip).
/// Mirrors `benchmark::sum_timing_reports` in the native runner.
fn sum_timing_reports(parts: &[TimingReport]) -> TimingReport {
    let wall_time_ns = parts
        .iter()
        .map(|p| p.wall_time_ns.unwrap_or(0))
        .sum::<u64>();
    let any_profiled = parts.iter().any(|p| p.gpu_total_ns.is_some());
    let gpu_total_ns = any_profiled.then_some(
        parts
            .iter()
            .map(|p| p.gpu_total_ns.unwrap_or(0))
            .sum::<u64>(),
    );
    let gpu_stage_ns = parts
        .iter()
        .flat_map(|p| p.gpu_stage_ns.iter().cloned())
        .collect();

    TimingReport {
        wall_time_ns: Some(wall_time_ns),
        gpu_total_ns,
        gpu_stage_ns,
    }
}

// ---------------------------------------------------------------------------
// Other helpers
// ---------------------------------------------------------------------------

fn cpu_reference(input: &[BabyBear], direction: NttDirection) -> Vec<BabyBear> {
    let mut cpu = input.to_vec();
    ntt_cpu_reference(&mut cpu, direction);
    cpu
}

fn make_plan(
    device: &WgpuDevice,
    log_n: u32,
    direction: NttDirection,
    family: FamilyOverride,
) -> Result<WgpuNttPlan, zkgpu_core::ZkGpuError> {
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

fn build_device_report_from(device: &WgpuDevice) -> DeviceReport {
    let caps = device.caps();
    DeviceReport {
        name: caps.device_name.clone(),
        backend: format!("{:?}", caps.backend),
        tier: format!("{:?}", caps.tier),
        gpu_family: format!("{:?}", caps.gpu_family),
        detection_source: format!("{:?}", caps.detection_source),
        platform_class: format!("{:?}", caps.platform_class),
        memory_model: format!("{:?}", caps.memory_model),
        driver: caps.driver.clone(),
        driver_info: caps.driver_info.clone(),
        max_buffer_size_bytes: caps.max_buffer_size,
        max_workgroup_size_x: caps.max_compute_workgroup_size_x,
        max_compute_invocations: caps.max_compute_invocations_per_workgroup,
        max_compute_workgroup_storage_size_bytes: caps.max_compute_workgroup_storage_size,
        feature_flags: caps.feature_flags_flat(),
    }
}

fn derive_kernel_variant(cases: &[CaseReport]) -> String {
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
}

fn case_error(case: &CaseSpec, error: String) -> CaseReport {
    case_error_family(case, None, error)
}

fn case_error_family(
    case: &CaseSpec,
    kernel_family: Option<String>,
    error: String,
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
        error: Some(error),
    }
}
