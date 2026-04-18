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
use zkgpu_goldilocks::Goldilocks;
use zkgpu_ntt::ntt_cpu_reference;
use zkgpu_report::{
    CaseReport, CaseSpec, DeviceReport, FamilyOverride, Field, KernelReport, StageTimingReport,
    StockhamTailOverride, SuiteReport, SuiteSpec, SuiteSummary, TestDirection, TimingReport,
};
use zkgpu_wgpu::{
    NttTimings, PlannerPolicy, StockhamTailOverride as PlanTailOverride, WgpuDevice,
    WgpuGoldilocksNttPlan, WgpuNttPlan,
};

/// Translate the report-layer tail override into the planner-layer enum.
fn to_plan_tail(ov: StockhamTailOverride) -> PlanTailOverride {
    match ov {
        StockhamTailOverride::Auto => PlanTailOverride::Auto,
        StockhamTailOverride::Local => PlanTailOverride::Local,
        StockhamTailOverride::Global => PlanTailOverride::Global,
    }
}

use crate::device;
use crate::inputs::{make_goldilocks_input, make_input};
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

    // Phase E.2.b: field dispatch at the suite boundary. Mirrors the
    // native testkit pattern — BabyBear cases go through the existing
    // `run_case_inner`, Goldilocks cases route to a parallel async
    // path backed by `WgpuGoldilocksNttPlan::execute_async`.
    // Profiled-timestamp cases are rejected per-case on the Goldilocks
    // path until profiled-execute lands in Phase E.2.c.
    for case in &spec.cases {
        let report = match spec.field {
            Field::BabyBear => {
                run_case_inner(
                    case,
                    spec.family_override,
                    spec.stockham_tail_override,
                    spec.r8_max_log_leaf_override,
                    &dev,
                )
                .await
            }
            Field::Goldilocks => run_case_goldilocks_inner(case, &dev).await,
        };
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
    let tail_variant = derive_tail_variant(&cases);

    Ok(SuiteReport {
        schema_version: 1,
        suite: spec.kind,
        device: device_report,
        kernel: KernelReport {
            // Source from the spec so a future Goldilocks browser path
            // produces correctly-labeled reports. Today the early
            // rejection above guarantees this is always `"BabyBear"`.
            field: spec.field.display_name().to_string(),
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

/// Run a single case exposed to JS.
///
/// Phase E.2.b post-review: takes an explicit [`Field`] so browser
/// callers can request a one-off Goldilocks case, matching the suite
/// API's `spec.field` surface. Prior `CaseSpec`-only variants of this
/// function routed unconditionally to BabyBear, which broke parity
/// with [`run_suite_async`]. The wasm entry point in `lib.rs` accepts
/// both the new envelope (`{case, field}`) and the legacy bare
/// `CaseSpec` (assumed BabyBear) for backward compatibility.
pub(crate) async fn run_single_case_async(
    case: &CaseSpec,
    field: Field,
) -> Result<CaseReport, String> {
    let dev = device::clone_device()?;
    Ok(match field {
        Field::BabyBear => {
            run_case_inner(case, FamilyOverride::Auto, StockhamTailOverride::Auto, None, &dev).await
        }
        Field::Goldilocks => run_case_goldilocks_inner(case, &dev).await,
    })
}

// ---------------------------------------------------------------------------
// Case dispatch
// ---------------------------------------------------------------------------

async fn run_case_inner(
    case: &CaseSpec,
    family: FamilyOverride,
    tail_override: StockhamTailOverride,
    r8_override: Option<u32>,
    dev: &Rc<WgpuDevice>,
) -> CaseReport {
    let input = make_input(case.log_n, &case.input);

    match case.direction {
        TestDirection::Forward => {
            run_direction(
                case, &input, NttDirection::Forward, family, tail_override, r8_override, dev,
            )
            .await
        }
        TestDirection::Inverse => {
            run_direction(
                case, &input, NttDirection::Inverse, family, tail_override, r8_override, dev,
            )
            .await
        }
        TestDirection::Roundtrip => {
            run_roundtrip(case, &input, family, tail_override, r8_override, dev).await
        }
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
    tail_override: StockhamTailOverride,
    r8_override: Option<u32>,
    dev: &Rc<WgpuDevice>,
) -> CaseReport {
    let expected = cpu_reference(input, direction);

    let mut plan = match make_plan(dev, case.log_n, direction, family, tail_override, r8_override) {
        Ok(p) => p,
        Err(e) => return case_error(case, e.to_string()),
    };
    let family_name = plan.family_name().to_string();
    let tail_strategy = plan.stockham_tail_strategy().map(str::to_string);
    let tail_reason = plan.stockham_tail_reason().map(str::to_string);
    let tail_stride_bytes = plan.tail_stride_bytes();

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
        Err(e) => {
            return case_error_with_tail(
                case,
                Some(family_name),
                tail_strategy,
                tail_reason,
                tail_stride_bytes,
                e,
            )
        }
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
        stockham_tail_strategy: tail_strategy,
        stockham_tail_reason: tail_reason,
        tail_stride_bytes,
    }
}

// ---------------------------------------------------------------------------
// Roundtrip case — mirrors native run_roundtrip_case exactly
// ---------------------------------------------------------------------------

async fn run_roundtrip(
    case: &CaseSpec,
    input: &[BabyBear],
    family: FamilyOverride,
    tail_override: StockhamTailOverride,
    r8_override: Option<u32>,
    dev: &Rc<WgpuDevice>,
) -> CaseReport {
    // Create both plans up-front.
    let mut forward_plan = match make_plan(
        dev,
        case.log_n,
        NttDirection::Forward,
        family,
        tail_override,
        r8_override,
    ) {
        Ok(p) => p,
        Err(e) => return case_error(case, e.to_string()),
    };
    // Capture forward-side metadata up-front so it survives a later failure
    // (inverse make_plan or either measurement). Mirrors the testkit twin
    // — without this an inverse make_plan crash drops the very tail strategy
    // PR 1 was meant to make visible.
    let forward_family_name = forward_plan.family_name().to_string();
    let tail_strategy = forward_plan.stockham_tail_strategy().map(str::to_string);
    let tail_reason = forward_plan.stockham_tail_reason().map(str::to_string);
    let tail_stride_bytes = forward_plan.tail_stride_bytes();

    let mut inverse_plan = match make_plan(
        dev,
        case.log_n,
        NttDirection::Inverse,
        family,
        tail_override,
        r8_override,
    ) {
        Ok(p) => p,
        Err(e) => {
            return case_error_with_tail(
                case,
                Some(forward_family_name),
                tail_strategy,
                tail_reason,
                tail_stride_bytes,
                e.to_string(),
            )
        }
    };
    let kernel_family = Some(format!(
        "{}/{}",
        forward_family_name,
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
        Err(e) => {
            return case_error_with_tail(
                case,
                kernel_family,
                tail_strategy,
                tail_reason,
                tail_stride_bytes,
                e,
            )
        }
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
        Err(e) => {
            return case_error_with_tail(
                case,
                kernel_family,
                tail_strategy,
                tail_reason,
                tail_stride_bytes,
                e,
            )
        }
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
        stockham_tail_strategy: tail_strategy,
        stockham_tail_reason: tail_reason,
        tail_stride_bytes,
    }
}

// ---------------------------------------------------------------------------
// Goldilocks async runner (Phase E.2.b)
// ---------------------------------------------------------------------------
//
// Parallel to `run_case_inner` / `run_direction` / `run_roundtrip`. The
// Goldilocks plan has no family-override or tail-override surface and
// no profiled-execute path yet, so the async routes are thinner:
//   - make_goldilocks_input + goldilocks_cpu_reference for I/O
//   - execute_async (non-profiled) for the GPU pass
//   - case.profile_gpu_timestamps=true → structured "not supported"
//     error pending Phase E.2.c's profiled-execute wiring
//
// Once `WgpuGoldilocksNttPlan::execute_profiled_async` exists, fold
// back into a single generic `measure_plan_async<F>` via a trait.

async fn run_case_goldilocks_inner(case: &CaseSpec, dev: &Rc<WgpuDevice>) -> CaseReport {
    // `case.profile_gpu_timestamps` is intentionally ignored on the
    // Goldilocks path — mirrors the native testkit's
    // `measure_goldilocks_plan`, which wall-times and reports
    // `gpu_total_ns: None` regardless of the flag. Keeping parity lets
    // a shared `SuiteSpec` produce semantically identical reports on
    // either runner. When E.2.c adds `execute_profiled_async` we flip
    // to honouring the flag on both sides in the same commit.
    let input = make_goldilocks_input(case.log_n, &case.input);

    match case.direction {
        TestDirection::Forward => {
            run_direction_goldilocks(case, &input, NttDirection::Forward, dev).await
        }
        TestDirection::Inverse => {
            run_direction_goldilocks(case, &input, NttDirection::Inverse, dev).await
        }
        TestDirection::Roundtrip => run_roundtrip_goldilocks(case, &input, dev).await,
    }
}

async fn run_direction_goldilocks(
    case: &CaseSpec,
    input: &[Goldilocks],
    direction: NttDirection,
    dev: &Rc<WgpuDevice>,
) -> CaseReport {
    let expected = goldilocks_cpu_reference(input, direction);

    let mut plan = match WgpuGoldilocksNttPlan::new(dev, case.log_n, direction) {
        Ok(p) => p,
        Err(e) => return case_error(case, e.to_string()),
    };
    let kernel_family = Some(goldilocks_family_label(case.log_n));

    let measurement = match measure_goldilocks_plan_async(
        dev,
        input,
        &mut plan,
        case.warmup_iterations,
        case.iterations,
    )
    .await
    {
        Ok(m) => m,
        Err(e) => return case_error_with_tail(case, kernel_family, None, None, None, e),
    };

    let outcome = compare_vectors(&measurement.final_output, &expected);

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
        timings: measurement.timings,
        error: None,
        stockham_tail_strategy: None,
        stockham_tail_reason: None,
        tail_stride_bytes: None,
    }
}

async fn run_roundtrip_goldilocks(
    case: &CaseSpec,
    input: &[Goldilocks],
    dev: &Rc<WgpuDevice>,
) -> CaseReport {
    let mut forward_plan = match WgpuGoldilocksNttPlan::new(dev, case.log_n, NttDirection::Forward)
    {
        Ok(p) => p,
        Err(e) => return case_error(case, e.to_string()),
    };
    let mut inverse_plan = match WgpuGoldilocksNttPlan::new(dev, case.log_n, NttDirection::Inverse)
    {
        Ok(p) => p,
        Err(e) => {
            return case_error_with_tail(
                case,
                Some(goldilocks_family_label(case.log_n)),
                None,
                None,
                None,
                e.to_string(),
            )
        }
    };
    let kernel_family = Some(goldilocks_family_label(case.log_n));

    let forward_measurement = match measure_goldilocks_plan_async(
        dev,
        input,
        &mut forward_plan,
        case.warmup_iterations,
        case.iterations,
    )
    .await
    {
        Ok(m) => m,
        Err(e) => return case_error_with_tail(case, kernel_family, None, None, None, e),
    };
    let inverse_measurement = match measure_goldilocks_plan_async(
        dev,
        &forward_measurement.final_output,
        &mut inverse_plan,
        case.warmup_iterations,
        case.iterations,
    )
    .await
    {
        Ok(m) => m,
        Err(e) => return case_error_with_tail(case, kernel_family, None, None, None, e),
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
        stockham_tail_strategy: None,
        stockham_tail_reason: None,
        tail_stride_bytes: None,
    }
}

/// Wall-time-only async measurement for the Goldilocks plan.
/// Counterpart of [`measure_plan_async`] for BabyBear; same shape minus
/// the profiled branch (Phase E.2.c).
struct GoldilocksPlanMeasurement {
    timings: TimingReport,
    final_output: Vec<Goldilocks>,
}

async fn measure_goldilocks_plan_async(
    dev: &Rc<WgpuDevice>,
    input: &[Goldilocks],
    plan: &mut WgpuGoldilocksNttPlan,
    warmup_iterations: u32,
    iterations: u32,
) -> Result<GoldilocksPlanMeasurement, String> {
    for _ in 0..warmup_iterations {
        let mut buf = dev.upload(input).map_err(|e| e.to_string())?;
        plan.execute_async(dev, &mut buf)
            .await
            .map_err(|e| e.to_string())?;
    }

    let measured = iterations.max(1);
    let mut wall_total_ns: u64 = 0;
    let mut final_output: Option<Vec<Goldilocks>> = None;

    for iter_idx in 0..measured {
        let mut buf = dev.upload(input).map_err(|e| e.to_string())?;
        let wall_start = web_time::Instant::now();
        plan.execute_async(dev, &mut buf)
            .await
            .map_err(|e| e.to_string())?;
        wall_total_ns += wall_start.elapsed().as_nanos() as u64;

        if iter_idx + 1 == measured {
            final_output = Some(
                buf.read_to_vec_async()
                    .await
                    .map_err(|e| e.to_string())?,
            );
        }
    }

    let wall_avg_ns = wall_total_ns / measured as u64;
    Ok(GoldilocksPlanMeasurement {
        timings: TimingReport {
            wall_time_ns: Some(wall_avg_ns),
            gpu_total_ns: None,
            gpu_stage_ns: Vec::new(),
        },
        final_output: final_output.expect("measured loop always captures final output"),
    })
}

fn goldilocks_cpu_reference(input: &[Goldilocks], direction: NttDirection) -> Vec<Goldilocks> {
    let mut cpu = input.to_vec();
    ntt_cpu_reference(&mut cpu, direction);
    cpu
}

fn goldilocks_family_label(log_n: u32) -> String {
    if log_n % 2 == 0 {
        "goldilocks-portable-r4".to_string()
    } else {
        "goldilocks-portable-r2".to_string()
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
    tail_override: StockhamTailOverride,
    r8_override: Option<u32>,
) -> Result<WgpuNttPlan, zkgpu_core::ZkGpuError> {
    // Always derive from the device caps so the Stockham tail heuristic has
    // full device context, then narrow the family. See the testkit twin for
    // the rationale — `stockham_only()` would drop the caps hint and force
    // `LocalFusedR4` on every device, defeating the new policy.
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

/// Summarise the per-case Stockham tail strategies into a single label.
/// Returns `None` if no case recorded a tail strategy; returns the strategy
/// name if every tailed case agreed, or `"mixed"` otherwise.
fn derive_tail_variant(cases: &[CaseReport]) -> Option<String> {
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

fn case_error(case: &CaseSpec, error: String) -> CaseReport {
    case_error_family(case, None, error)
}

fn case_error_family(
    case: &CaseSpec,
    kernel_family: Option<String>,
    error: String,
) -> CaseReport {
    case_error_with_tail(case, kernel_family, None, None, None, error)
}

/// Failed `CaseReport` that preserves any tail metadata captured before the
/// failure. Mirrors the testkit twin so browser-side measurement crashes on
/// Xclipse/Mali still surface the planned Stockham tail strategy.
#[allow(clippy::too_many_arguments)]
fn case_error_with_tail(
    case: &CaseSpec,
    kernel_family: Option<String>,
    stockham_tail_strategy: Option<String>,
    stockham_tail_reason: Option<String>,
    tail_stride_bytes: Option<u64>,
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
        stockham_tail_strategy,
        stockham_tail_reason,
        tail_stride_bytes,
    }
}

#[cfg(test)]
mod tests {
    //! Native-target tests for the failure-helper plumbing.
    //!
    //! These do not need a GPU, do not need wasm, and run with the
    //! ordinary `cargo test -p zkgpu-web` invocation. The browser-side
    //! companion test in `tests/browser_smoke.rs` covers the success-path
    //! end-to-end through the wasm boundary; together they close Codex's
    //! residual structural-parity gap.
    use super::*;
    use zkgpu_report::InputPattern;

    #[test]
    fn case_error_with_tail_preserves_tail_metadata() {
        // PR 1's whole point is that operators can see *which* tail
        // strategy fired even when measurement crashed. Mirror of the
        // testkit twin — verifies the helper threads every captured field
        // through to the report.
        let case = CaseSpec {
            name: "tail_preserved".into(),
            log_n: 18,
            direction: TestDirection::Forward,
            input: InputPattern::Sequential,
            iterations: 1,
            warmup_iterations: 0,
            profile_gpu_timestamps: false,
        };
        let report = case_error_with_tail(
            &case,
            Some("stockham".into()),
            Some("GlobalOnlyR4".into()),
            Some("HostileStridedDevice".into()),
            Some(8),
            "simulated measurement crash".into(),
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

    // Phase E.2.b post-review (P2): `profile_gpu_timestamps=true` on a
    // Goldilocks case must degrade to wall-only timing on the browser
    // runner, matching the native testkit's `measure_goldilocks_plan`
    // which ignores the flag and returns `gpu_total_ns: None`. An
    // earlier draft of this path returned a structured per-case error;
    // that diverged from native and made shared `SuiteSpec` specs
    // behave differently on the two runners. The GPU-backed parity
    // check lives in `tests/browser_smoke.rs` where a wasm-bindgen-test
    // can actually run the browser path.

    /// Goldilocks family labels mirror the native testkit convention so
    /// mixed-field JSON reports are interpretable across runners.
    #[test]
    fn goldilocks_family_label_picks_radix_from_log_n_parity() {
        assert_eq!(goldilocks_family_label(10), "goldilocks-portable-r4");
        assert_eq!(goldilocks_family_label(11), "goldilocks-portable-r2");
        assert_eq!(goldilocks_family_label(18), "goldilocks-portable-r4");
        assert_eq!(goldilocks_family_label(19), "goldilocks-portable-r2");
    }

    #[test]
    fn case_error_family_keeps_tail_fields_none() {
        // Pre-plan failures (e.g. an invalid log_n caught by make_plan)
        // have no tail metadata to surface — the slim wrapper still
        // leaves the three fields blank.
        let case = CaseSpec {
            name: "no_tail_yet".into(),
            log_n: 10,
            direction: TestDirection::Forward,
            input: InputPattern::Sequential,
            iterations: 1,
            warmup_iterations: 0,
            profile_gpu_timestamps: false,
        };
        let report = case_error_family(&case, None, "plan build failed".into());
        assert!(report.stockham_tail_strategy.is_none());
        assert!(report.stockham_tail_reason.is_none());
        assert!(report.tail_stride_bytes.is_none());
    }
}
