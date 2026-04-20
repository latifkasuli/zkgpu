use std::time::{Duration, Instant};

use zkgpu_babybear::BabyBear;
use zkgpu_core::{GpuBuffer, GpuDevice, NttPlan};
use zkgpu_goldilocks::Goldilocks;
use zkgpu_wgpu::{GpuTiming, NttTimings, WgpuDevice, WgpuGoldilocksNttPlan, WgpuNttPlan};

use crate::report::{StageTimingReport, TimingReport};
use zkgpu_report::{SoakSample, SoakStats};

pub struct PlanMeasurement {
    pub timings: TimingReport,
    pub final_output: Vec<BabyBear>,
}

pub fn measure_plan(
    device: &WgpuDevice,
    data: &[BabyBear],
    plan: &mut WgpuNttPlan,
    warmup_iterations: u32,
    iterations: u32,
    profile_gpu_timestamps: bool,
) -> Result<PlanMeasurement, zkgpu_core::ZkGpuError> {
    let warmups = warmup_iterations;
    let measured = iterations.max(1);

    for _ in 0..warmups {
        let mut buf = device.upload(data)?;
        if profile_gpu_timestamps {
            let _ = plan.execute_kernels_profiled(device, &mut buf)?;
        } else {
            plan.execute(device, &mut buf)?;
        }
    }

    let mut wall_total = Duration::ZERO;
    let mut gpu_total_ns = 0.0f64;
    let mut stage_totals: Option<Vec<(String, f64)>> = None;
    let mut profiled_samples = 0u32;
    let mut final_output = None;

    for iter_idx in 0..measured {
        let mut buf = device.upload(data)?;
        let start = Instant::now();
        if profile_gpu_timestamps {
            if let Some(timings) = plan.execute_kernels_profiled(device, &mut buf)? {
                accumulate_profiled(&mut stage_totals, &timings);
                gpu_total_ns += timings.gpu_total_ns;
                profiled_samples += 1;
            }
        } else {
            plan.execute(device, &mut buf)?;
        }
        wall_total += start.elapsed();

        if iter_idx + 1 == measured {
            final_output = Some(buf.read_to_vec()?);
        }
    }

    let wall_avg_ns = (wall_total.as_secs_f64() * 1_000_000_000.0 / measured as f64) as u64;
    let (gpu_total_ns, gpu_stage_ns) = if profiled_samples > 0 {
        let gpu_total = (gpu_total_ns / profiled_samples as f64) as u64;
        let stages = stage_totals
            .unwrap_or_default()
            .into_iter()
            .map(|(label, total)| StageTimingReport {
                label,
                duration_ns: (total / profiled_samples as f64) as u64,
            })
            .collect();
        (Some(gpu_total), stages)
    } else {
        (None, Vec::new())
    };

    Ok(PlanMeasurement {
        timings: TimingReport {
            wall_time_ns: Some(wall_avg_ns),
            gpu_total_ns,
            gpu_stage_ns,
        },
        final_output: final_output.expect("measured loop always captures the final GPU output"),
    })
}

pub fn sum_timing_reports(parts: &[TimingReport]) -> TimingReport {
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

fn accumulate_profiled(stage_totals: &mut Option<Vec<(String, f64)>>, timings: &NttTimings) {
    let accum = stage_totals.get_or_insert_with(|| {
        timings
            .gpu_stage_ns
            .iter()
            .map(|s: &GpuTiming| (s.label.clone(), 0.0))
            .collect()
    });

    debug_assert_eq!(
        accum.len(),
        timings.gpu_stage_ns.len(),
        "profiled stage count changed across iterations",
    );
    for (acc, sample) in accum.iter_mut().zip(timings.gpu_stage_ns.iter()) {
        debug_assert_eq!(
            acc.0, sample.label,
            "profiled stage order changed across iterations",
        );
        acc.1 += sample.duration_ns;
    }
}

// ---------------------------------------------------------------------------
// Soak benchmark: sustained-run measurement with per-iteration samples
// ---------------------------------------------------------------------------

pub struct SoakMeasurement {
    pub samples: Vec<SoakSample>,
    pub stats: SoakStats,
    /// Output from the first iteration (for validation).
    pub first_output: Vec<BabyBear>,
    /// Output from the last iteration (for validation).
    pub last_output: Vec<BabyBear>,
}

/// Run a sustained soak benchmark for `duration` seconds.
///
/// Executes warmup iterations first, then runs the plan repeatedly until
/// `duration` elapses, capturing per-iteration wall and GPU timing samples.
/// Validates first and last iteration outputs are available for correctness
/// checking by the caller.
pub fn measure_plan_soak(
    device: &WgpuDevice,
    data: &[BabyBear],
    plan: &mut WgpuNttPlan,
    duration: Duration,
    warmup_iterations: u32,
    profile_gpu_timestamps: bool,
) -> Result<SoakMeasurement, zkgpu_core::ZkGpuError> {
    // --- Warmup phase (discarded, not timed) ---
    for _ in 0..warmup_iterations {
        let mut buf = device.upload(data)?;
        if profile_gpu_timestamps {
            let _ = plan.execute_kernels_profiled(device, &mut buf)?;
        } else {
            plan.execute(device, &mut buf)?;
        }
    }

    // --- Soak phase ---
    let soak_start = Instant::now();
    let mut samples = Vec::with_capacity(1024);
    let mut first_output: Option<Vec<BabyBear>> = None;
    let mut last_output = Vec::new();
    let mut iteration = 0u32;

    while soak_start.elapsed() < duration {
        let mut buf = device.upload(data)?;
        let iter_start = Instant::now();

        let gpu_ns = if profile_gpu_timestamps {
            plan.execute_kernels_profiled(device, &mut buf)?
                .map(|t| t.gpu_total_ns as u64)
        } else {
            plan.execute(device, &mut buf)?;
            None
        };

        let wall_ns = iter_start.elapsed().as_nanos() as u64;
        let elapsed_ms = soak_start.elapsed().as_millis() as u64;

        samples.push(SoakSample {
            iteration,
            wall_ns,
            gpu_total_ns: gpu_ns,
            elapsed_ms,
        });

        // Capture first and last outputs for validation
        if first_output.is_none() {
            first_output = Some(buf.read_to_vec()?);
        }
        // Always keep the latest output; only read on the last iteration
        // (we check elapsed *after* push, so this is cheap — we only
        // actually read the last one by overwriting the vec).
        if soak_start.elapsed() >= duration {
            last_output = buf.read_to_vec()?;
        }

        iteration += 1;
    }

    // If only one iteration ran, last == first
    if last_output.is_empty() {
        if let Some(ref first) = first_output {
            last_output = first.clone();
        }
    }

    let actual_duration = soak_start.elapsed();
    let stats = compute_soak_stats(&samples, actual_duration);

    Ok(SoakMeasurement {
        samples,
        stats,
        first_output: first_output.unwrap_or_default(),
        last_output,
    })
}

/// Compute aggregate statistics from soak samples.
pub fn compute_soak_stats(samples: &[SoakSample], actual_duration: Duration) -> SoakStats {
    let n = samples.len();
    if n == 0 {
        return SoakStats {
            total_iterations: 0,
            actual_duration_secs: actual_duration.as_secs_f64(),
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
        };
    }

    let actual_secs = actual_duration.as_secs_f64();
    let iterations_per_sec = n as f64 / actual_secs;

    // Wall time stats
    let mut wall_sorted: Vec<u64> = samples.iter().map(|s| s.wall_ns).collect();
    wall_sorted.sort_unstable();

    let median_wall_ns = percentile(&wall_sorted, 50);
    let p5_wall_ns = percentile(&wall_sorted, 5);
    let p95_wall_ns = percentile(&wall_sorted, 95);
    let min_wall_ns = wall_sorted[0];
    let max_wall_ns = wall_sorted[n - 1];

    let wall_mean = wall_sorted.iter().copied().sum::<u64>() as f64 / n as f64;
    let wall_variance = wall_sorted
        .iter()
        .map(|&v| {
            let diff = v as f64 - wall_mean;
            diff * diff
        })
        .sum::<f64>()
        / n as f64;
    let wall_cv = if wall_mean > 0.0 {
        wall_variance.sqrt() / wall_mean
    } else {
        0.0
    };

    // Thermal drift: mean of last 10% / mean of first 10%
    let bucket_size = (n / 10).max(1);
    let first_bucket_mean = samples[..bucket_size]
        .iter()
        .map(|s| s.wall_ns as f64)
        .sum::<f64>()
        / bucket_size as f64;
    let last_bucket_mean = samples[n.saturating_sub(bucket_size)..]
        .iter()
        .map(|s| s.wall_ns as f64)
        .sum::<f64>()
        / bucket_size as f64;
    let thermal_drift_ratio = if first_bucket_mean > 0.0 {
        last_bucket_mean / first_bucket_mean
    } else {
        1.0
    };

    // GPU stats (if profiled)
    let has_gpu = samples.iter().any(|s| s.gpu_total_ns.is_some());
    let (median_gpu_ns, p5_gpu_ns, p95_gpu_ns, gpu_cv) = if has_gpu {
        let mut gpu_sorted: Vec<u64> = samples
            .iter()
            .filter_map(|s| s.gpu_total_ns)
            .collect();
        gpu_sorted.sort_unstable();
        if gpu_sorted.is_empty() {
            (None, None, None, None)
        } else {
            let gn = gpu_sorted.len();
            let gpu_mean = gpu_sorted.iter().copied().sum::<u64>() as f64 / gn as f64;
            let gpu_var = gpu_sorted
                .iter()
                .map(|&v| {
                    let diff = v as f64 - gpu_mean;
                    diff * diff
                })
                .sum::<f64>()
                / gn as f64;
            let cv = if gpu_mean > 0.0 {
                gpu_var.sqrt() / gpu_mean
            } else {
                0.0
            };
            (
                Some(percentile(&gpu_sorted, 50)),
                Some(percentile(&gpu_sorted, 5)),
                Some(percentile(&gpu_sorted, 95)),
                Some(cv),
            )
        }
    } else {
        (None, None, None, None)
    };

    SoakStats {
        total_iterations: n as u32,
        actual_duration_secs: actual_secs,
        iterations_per_sec,
        median_wall_ns,
        p5_wall_ns,
        p95_wall_ns,
        min_wall_ns,
        max_wall_ns,
        wall_cv,
        thermal_drift_ratio,
        median_gpu_ns,
        p5_gpu_ns,
        p95_gpu_ns,
        gpu_cv,
    }
}

/// Compute the k-th percentile from a sorted slice (nearest-rank method).
// ---------------------------------------------------------------------------
// Goldilocks measurement path (Phase E.1.c, profiled in E.2.d)
// ---------------------------------------------------------------------------
//
// Parallel to [`measure_plan`] rather than generified via a trait — the
// plan and buffer element types differ enough that a trait would force
// lifetime/HRTB gymnastics for marginal DRY. Phase E.2.d threads
// `profile_gpu_timestamps` through so the signature matches the
// BabyBear path and the shared `CaseSpec.profile_gpu_timestamps`
// contract means the same thing across runners.

pub struct GoldilocksPlanMeasurement {
    pub timings: TimingReport,
    pub final_output: Vec<Goldilocks>,
}

/// Averaged per-iteration measurement for the Goldilocks NTT plan.
///
/// Counterpart of [`measure_plan`] for BabyBear. When
/// `profile_gpu_timestamps = true`, the measured loop routes through
/// [`WgpuGoldilocksNttPlan::execute_profiled`] and averages GPU
/// timestamps across iterations; `gpu_total_ns` and `gpu_stage_ns` are
/// populated on adapters that advertise `TIMESTAMP_QUERY`, `None` /
/// empty otherwise (matching the BabyBear contract). When the flag is
/// `false`, uses the fast non-profiled `execute` and returns
/// wall-time-only, just as BabyBear does.
pub fn measure_goldilocks_plan(
    device: &WgpuDevice,
    data: &[Goldilocks],
    plan: &mut WgpuGoldilocksNttPlan,
    warmup_iterations: u32,
    iterations: u32,
    profile_gpu_timestamps: bool,
) -> Result<GoldilocksPlanMeasurement, zkgpu_core::ZkGpuError> {
    let measured = iterations.max(1);

    // Warmup mirrors the profiling choice so pipeline state matches
    // what the measured loop will see. BabyBear's `measure_plan` does
    // the same — keeping the behaviour aligned avoids cross-field
    // cold-start asymmetries in the first measured iteration.
    for _ in 0..warmup_iterations {
        let mut buf = device.upload(data)?;
        if profile_gpu_timestamps {
            let _ = plan.execute_profiled(device, &mut buf)?;
        } else {
            plan.execute(device, &mut buf)?;
        }
    }

    let mut wall_total = Duration::ZERO;
    let mut gpu_total_ns_accum: f64 = 0.0;
    let mut stage_totals: Option<Vec<(String, f64)>> = None;
    let mut profiled_samples: u32 = 0;
    let mut final_output = None;

    for iter_idx in 0..measured {
        let mut buf = device.upload(data)?;
        let start = Instant::now();
        let ntt_timings = if profile_gpu_timestamps {
            plan.execute_profiled(device, &mut buf)?
        } else {
            plan.execute(device, &mut buf)?;
            None
        };
        wall_total += start.elapsed();

        if let Some(ref t) = ntt_timings {
            gpu_total_ns_accum += t.gpu_total_ns;
            let accum = stage_totals.get_or_insert_with(|| {
                t.gpu_stage_ns
                    .iter()
                    .map(|s| (s.label.clone(), 0.0))
                    .collect()
            });
            for (acc, sample) in accum.iter_mut().zip(t.gpu_stage_ns.iter()) {
                acc.1 += sample.duration_ns;
            }
            profiled_samples += 1;
        }

        if iter_idx + 1 == measured {
            final_output = Some(buf.read_to_vec()?);
        }
    }

    let wall_avg_ns = (wall_total.as_secs_f64() * 1_000_000_000.0 / measured as f64) as u64;
    let (gpu_total_ns, gpu_stage_ns) = if profiled_samples > 0 {
        let divisor = profiled_samples as f64;
        let total = (gpu_total_ns_accum / divisor) as u64;
        let stages = stage_totals
            .unwrap_or_default()
            .into_iter()
            .map(|(label, total)| StageTimingReport {
                label,
                duration_ns: (total / divisor) as u64,
            })
            .collect();
        (Some(total), stages)
    } else {
        (None, Vec::new())
    };

    Ok(GoldilocksPlanMeasurement {
        timings: TimingReport {
            wall_time_ns: Some(wall_avg_ns),
            gpu_total_ns,
            gpu_stage_ns,
        },
        final_output: final_output
            .expect("measured loop always captures the final GPU output"),
    })
}

/// Goldilocks twin of [`measure_plan_soak`] — Phase E.2.c, gate
/// added in E.2.d.
///
/// When `profile_gpu_timestamps = true`, uses
/// [`WgpuGoldilocksNttPlan::execute_profiled`] so each soak sample
/// carries `gpu_total_ns`. When `false`, uses the non-profiled
/// `execute` and samples carry `gpu_total_ns: None` — matching
/// BabyBear soak semantics for the same `CaseSpec` flag so operators
/// get wall-only stats when they explicitly opt out of profiling
/// (e.g. to reduce per-iteration overhead on long soaks).
pub fn measure_goldilocks_plan_soak(
    device: &WgpuDevice,
    data: &[Goldilocks],
    plan: &mut WgpuGoldilocksNttPlan,
    duration: Duration,
    warmup_iterations: u32,
    profile_gpu_timestamps: bool,
) -> Result<GoldilocksSoakMeasurement, zkgpu_core::ZkGpuError> {
    // --- Warmup (discarded, not timed) ---
    // Mirror the profiling flag so cold-state transitions in the first
    // measured iteration don't carry a timestamp-query setup cost the
    // rest of the soak doesn't see. Matches BabyBear's measure_plan_soak.
    for _ in 0..warmup_iterations {
        let mut buf = device.upload(data)?;
        if profile_gpu_timestamps {
            let _ = plan.execute_profiled(device, &mut buf)?;
        } else {
            plan.execute(device, &mut buf)?;
        }
    }

    // --- Soak phase ---
    let soak_start = Instant::now();
    let mut samples = Vec::with_capacity(1024);
    let mut first_output: Option<Vec<Goldilocks>> = None;
    let mut last_output = Vec::new();
    let mut iteration = 0u32;

    while soak_start.elapsed() < duration {
        let mut buf = device.upload(data)?;
        let iter_start = Instant::now();

        let gpu_ns = if profile_gpu_timestamps {
            plan.execute_profiled(device, &mut buf)?
                .map(|t| t.gpu_total_ns as u64)
        } else {
            plan.execute(device, &mut buf)?;
            None
        };

        let wall_ns = iter_start.elapsed().as_nanos() as u64;
        let elapsed_ms = soak_start.elapsed().as_millis() as u64;

        samples.push(SoakSample {
            iteration,
            wall_ns,
            gpu_total_ns: gpu_ns,
            elapsed_ms,
        });

        if first_output.is_none() {
            first_output = Some(buf.read_to_vec()?);
        }
        if soak_start.elapsed() >= duration {
            last_output = buf.read_to_vec()?;
        }

        iteration += 1;
    }

    if last_output.is_empty() {
        if let Some(ref first) = first_output {
            last_output = first.clone();
        }
    }

    let actual_duration = soak_start.elapsed();
    let stats = compute_soak_stats(&samples, actual_duration);

    Ok(GoldilocksSoakMeasurement {
        samples,
        stats,
        first_output: first_output.unwrap_or_default(),
        last_output,
    })
}

/// Goldilocks twin of [`SoakMeasurement`]. Separate type because the
/// element vec is `Vec<Goldilocks>` rather than `Vec<BabyBear>`;
/// everything else is identical.
pub struct GoldilocksSoakMeasurement {
    pub samples: Vec<SoakSample>,
    pub stats: SoakStats,
    pub first_output: Vec<Goldilocks>,
    pub last_output: Vec<Goldilocks>,
}

// ---------------------------------------------------------------------------
// Poseidon2 measurement paths (Phase F.3.b)
// ---------------------------------------------------------------------------
//
// Parallel to `measure_plan` / `measure_goldilocks_plan`. Shipped
// wall-only because the Poseidon2 plans only expose sync non-profiled
// `execute` today.
//
// Contract (Phase F.3.d post-review): when `profile_gpu_timestamps =
// true`, these helpers return a structured error rather than
// silently degrading. That keeps spec semantics honest — a caller
// that asks for GPU timestamps gets either the timestamps or a
// visible rejection, never a success report with a silently-dropped
// field. The shipped `poseidon2_benchmark_suite()` constructor and
// CLI hash mode both set the flag to `false`, so the rejection only
// fires on hand-constructed specs. When `execute_profiled` lands on
// the plans, this branch is replaced with the profiled path and the
// error goes away — the contract flips from "rejected" to
// "supported" without a caller-side change.

use zkgpu_wgpu::{WgpuBabyBearPoseidon2Plan, WgpuGoldilocksPoseidon2Plan};

pub struct Poseidon2PlanMeasurement<F> {
    pub timings: TimingReport,
    pub final_output: Vec<F>,
}

/// Error payload returned when a hand-written spec asks for GPU
/// timestamps on the Poseidon2 path — the plans have no
/// `execute_profiled` variant yet. Shared helper so both the
/// BabyBear and Goldilocks measurement paths return the same
/// diagnostic string.
fn poseidon2_profiling_unsupported_error() -> zkgpu_core::ZkGpuError {
    zkgpu_core::ZkGpuError::InvalidNttSize(
        "profile_gpu_timestamps=true is not yet supported on the \
         Poseidon2 path — the plans have no execute_profiled variant. \
         Rerun with profile_gpu_timestamps=false, or wait for a \
         future F.3.* sub-phase that adds profiled-execute to both \
         plans and flips this from rejection to a measured profiled \
         path."
            .to_string(),
    )
}

pub fn measure_babybear_poseidon2_plan(
    device: &WgpuDevice,
    data: &[BabyBear],
    plan: &mut WgpuBabyBearPoseidon2Plan,
    warmup_iterations: u32,
    iterations: u32,
    profile_gpu_timestamps: bool,
) -> Result<Poseidon2PlanMeasurement<BabyBear>, zkgpu_core::ZkGpuError> {
    if profile_gpu_timestamps {
        return Err(poseidon2_profiling_unsupported_error());
    }
    let measured = iterations.max(1);

    // Warmup (discarded, non-profiled).
    for _ in 0..warmup_iterations {
        let mut buf = device.upload(data)?;
        plan.execute(device, &mut buf)?;
    }

    let mut wall_total = Duration::ZERO;
    let mut final_output = None;

    for iter_idx in 0..measured {
        let mut buf = device.upload(data)?;
        let start = Instant::now();
        plan.execute(device, &mut buf)?;
        wall_total += start.elapsed();

        if iter_idx + 1 == measured {
            final_output = Some(buf.read_to_vec()?);
        }
    }

    let wall_avg_ns =
        (wall_total.as_secs_f64() * 1_000_000_000.0 / measured as f64) as u64;

    Ok(Poseidon2PlanMeasurement {
        timings: TimingReport {
            wall_time_ns: Some(wall_avg_ns),
            gpu_total_ns: None,
            gpu_stage_ns: Vec::new(),
        },
        final_output: final_output
            .expect("measured loop always captures the final GPU output"),
    })
}

pub fn measure_goldilocks_poseidon2_plan(
    device: &WgpuDevice,
    data: &[Goldilocks],
    plan: &mut WgpuGoldilocksPoseidon2Plan,
    warmup_iterations: u32,
    iterations: u32,
    profile_gpu_timestamps: bool,
) -> Result<Poseidon2PlanMeasurement<Goldilocks>, zkgpu_core::ZkGpuError> {
    if profile_gpu_timestamps {
        return Err(poseidon2_profiling_unsupported_error());
    }

    let measured = iterations.max(1);

    for _ in 0..warmup_iterations {
        let mut buf = device.upload(data)?;
        plan.execute(device, &mut buf)?;
    }

    let mut wall_total = Duration::ZERO;
    let mut final_output = None;

    for iter_idx in 0..measured {
        let mut buf = device.upload(data)?;
        let start = Instant::now();
        plan.execute(device, &mut buf)?;
        wall_total += start.elapsed();

        if iter_idx + 1 == measured {
            final_output = Some(buf.read_to_vec()?);
        }
    }

    let wall_avg_ns =
        (wall_total.as_secs_f64() * 1_000_000_000.0 / measured as f64) as u64;

    Ok(Poseidon2PlanMeasurement {
        timings: TimingReport {
            wall_time_ns: Some(wall_avg_ns),
            gpu_total_ns: None,
            gpu_stage_ns: Vec::new(),
        },
        final_output: final_output
            .expect("measured loop always captures the final GPU output"),
    })
}

fn percentile(sorted: &[u64], pct: u32) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((pct as f64 / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn sum_timing_reports_preserves_zero_gpu_measurements() {
        let a = TimingReport {
            wall_time_ns: Some(10),
            gpu_total_ns: Some(0),
            gpu_stage_ns: Vec::new(),
        };
        let b = TimingReport {
            wall_time_ns: Some(20),
            gpu_total_ns: Some(0),
            gpu_stage_ns: Vec::new(),
        };

        let sum = sum_timing_reports(&[a, b]);
        assert_eq!(sum.wall_time_ns, Some(30));
        assert_eq!(sum.gpu_total_ns, Some(0));
    }

    #[test]
    fn sum_timing_reports_leaves_gpu_total_missing_when_unprofiled() {
        let a = TimingReport {
            wall_time_ns: Some(10),
            gpu_total_ns: None,
            gpu_stage_ns: Vec::new(),
        };
        let b = TimingReport {
            wall_time_ns: Some(20),
            gpu_total_ns: None,
            gpu_stage_ns: Vec::new(),
        };

        let sum = sum_timing_reports(&[a, b]);
        assert_eq!(sum.wall_time_ns, Some(30));
        assert_eq!(sum.gpu_total_ns, None);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "profiled stage count changed across iterations")]
    fn accumulate_profiled_rejects_stage_count_drift() {
        let mut stage_totals = None;
        accumulate_profiled(
            &mut stage_totals,
            &NttTimings {
                wall_clock: Duration::ZERO,
                gpu_stage_ns: vec![GpuTiming {
                    label: "stage0".into(),
                    duration_ns: 1.0,
                }],
                gpu_total_ns: 1.0,
            },
        );
        accumulate_profiled(
            &mut stage_totals,
            &NttTimings {
                wall_clock: Duration::ZERO,
                gpu_stage_ns: vec![
                    GpuTiming {
                        label: "stage0".into(),
                        duration_ns: 1.0,
                    },
                    GpuTiming {
                        label: "stage1".into(),
                        duration_ns: 1.0,
                    },
                ],
                gpu_total_ns: 2.0,
            },
        );
    }

    // === Soak stats computation ===

    fn make_samples(wall_values: &[u64]) -> Vec<SoakSample> {
        wall_values
            .iter()
            .enumerate()
            .map(|(i, &wall_ns)| SoakSample {
                iteration: i as u32,
                wall_ns,
                gpu_total_ns: None,
                elapsed_ms: i as u64 * 10,
            })
            .collect()
    }

    fn make_profiled_samples(wall_values: &[u64], gpu_values: &[u64]) -> Vec<SoakSample> {
        wall_values
            .iter()
            .zip(gpu_values.iter())
            .enumerate()
            .map(|(i, (&wall_ns, &gpu_ns))| SoakSample {
                iteration: i as u32,
                wall_ns,
                gpu_total_ns: Some(gpu_ns),
                elapsed_ms: i as u64 * 10,
            })
            .collect()
    }

    #[test]
    fn soak_stats_empty_samples() {
        let stats = compute_soak_stats(&[], Duration::from_secs(1));
        assert_eq!(stats.total_iterations, 0);
        assert_eq!(stats.iterations_per_sec, 0.0);
        assert!((stats.thermal_drift_ratio - 1.0).abs() < 1e-9);
    }

    #[test]
    fn soak_stats_single_sample() {
        let samples = make_samples(&[1000]);
        let stats = compute_soak_stats(&samples, Duration::from_millis(10));
        assert_eq!(stats.total_iterations, 1);
        assert_eq!(stats.median_wall_ns, 1000);
        assert_eq!(stats.min_wall_ns, 1000);
        assert_eq!(stats.max_wall_ns, 1000);
        assert!((stats.wall_cv - 0.0).abs() < 1e-9);
    }

    #[test]
    fn soak_stats_stable_run_has_low_cv() {
        // 100 iterations, all at ~10ms with tiny jitter
        let samples = make_samples(
            &(0..100)
                .map(|i| 10_000_000 + (i % 3) * 1000) // 10ms ± 2μs
                .collect::<Vec<_>>(),
        );
        let stats = compute_soak_stats(&samples, Duration::from_secs(1));
        assert_eq!(stats.total_iterations, 100);
        assert!(stats.wall_cv < 0.001, "expected very low CV for stable run, got {}", stats.wall_cv);
        assert!(
            (stats.thermal_drift_ratio - 1.0).abs() < 0.01,
            "expected no drift, got {}",
            stats.thermal_drift_ratio
        );
    }

    #[test]
    fn soak_stats_throttled_run_has_drift() {
        // First 50 iterations at 10ms, last 50 at 15ms (50% thermal drift)
        let mut values = vec![10_000_000u64; 50];
        values.extend(vec![15_000_000u64; 50]);
        let samples = make_samples(&values);
        let stats = compute_soak_stats(&samples, Duration::from_secs(1));
        assert!(
            stats.thermal_drift_ratio > 1.3,
            "expected drift > 1.3, got {}",
            stats.thermal_drift_ratio
        );
        assert!(
            stats.wall_cv > 0.1,
            "expected high CV for throttled run, got {}",
            stats.wall_cv
        );
    }

    #[test]
    fn soak_stats_percentiles_ordered() {
        let samples = make_samples(
            &(0..200).map(|i| (i + 1) * 100_000).collect::<Vec<_>>(),
        );
        let stats = compute_soak_stats(&samples, Duration::from_secs(2));
        assert!(stats.p5_wall_ns <= stats.median_wall_ns);
        assert!(stats.median_wall_ns <= stats.p95_wall_ns);
        assert!(stats.min_wall_ns <= stats.p5_wall_ns);
        assert!(stats.p95_wall_ns <= stats.max_wall_ns);
    }

    #[test]
    fn soak_stats_with_gpu_profiling() {
        let wall = vec![10_000_000u64; 50];
        let gpu = vec![8_000_000u64; 50];
        let samples = make_profiled_samples(&wall, &gpu);
        let stats = compute_soak_stats(&samples, Duration::from_millis(500));
        assert_eq!(stats.median_gpu_ns, Some(8_000_000));
        assert!(stats.gpu_cv.is_some());
        assert!(stats.gpu_cv.unwrap() < 0.001);
    }

    #[test]
    fn percentile_boundary_values() {
        let sorted = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
        assert_eq!(percentile(&sorted, 0), 10);
        assert_eq!(percentile(&sorted, 100), 100);
        // 50th percentile of 10 items: idx = round(0.5 * 9) = round(4.5) = 5 → value 60
        assert_eq!(percentile(&sorted, 50), 60);
        // Odd-length array: clear median
        let sorted_odd = vec![10, 20, 30, 40, 50];
        assert_eq!(percentile(&sorted_odd, 50), 30);
    }
}
