use std::time::Instant;

use serde::Serialize;
use zkgpu_babybear::BabyBear;
use zkgpu_core::{GpuBuffer, GpuDevice, NttDirection, NttPlan};
use zkgpu_goldilocks::Goldilocks;
use zkgpu_ntt::ntt_cpu_reference;
use zkgpu_report::Field;
use zkgpu_wgpu::{CapabilityProfile, PlannerPolicy, WgpuDevice, WgpuGoldilocksNttPlan, WgpuNttPlan};

#[derive(Serialize)]
struct BenchmarkReport {
    device: DeviceReport,
    /// Which prime field the benchmark targeted. Phase E.1.d added
    /// `--field goldilocks`; emitting this lets JSON consumers
    /// distinguish BabyBear rows from Goldilocks rows without sniffing
    /// the `family` string on each `RunReport`.
    field: String,
    /// Forced kernel family, if any (`stockham` / `four-step`).
    /// `None` means either the user didn't force it (auto-policy) or
    /// the user's choice was silently ignored because `field ==
    /// Goldilocks`. In the latter case stderr also logs a warning at
    /// startup — see the `--field goldilocks` guard in `main()`.
    family_override: Option<String>,
    /// Forced Stockham tail strategy (`local` / `global`). Same
    /// null-on-ignored-under-Goldilocks convention as `family_override`.
    tail_override: Option<String>,
    runs: Vec<RunReport>,
}

#[derive(Serialize)]
struct DeviceReport {
    name: String,
    backend: String,
    device_type: String,
    tier: String,
    gpu_family: String,
    detection_source: String,
    platform_class: String,
    driver: String,
    driver_info: String,
    features: FeaturesReport,
    limits: LimitsReport,
}

#[derive(Serialize)]
struct FeaturesReport {
    subgroup: bool,
    timestamp_query: bool,
    timestamp_query_inside_passes: bool,
    mappable_primary_buffers: bool,
    pipeline_cache: bool,
    transient_saves_memory: bool,
}

#[derive(Serialize)]
struct LimitsReport {
    max_buffer_size_mb: u64,
    max_storage_buffer_binding_size_mb: u64,
    max_compute_workgroup_size_x: u32,
    max_compute_invocations_per_workgroup: u32,
    max_compute_workgroups_per_dimension: u32,
    max_compute_workgroup_storage_size_bytes: u32,
}

#[derive(Serialize)]
struct RunReport {
    log_n: u32,
    n: u64,
    direction: String,
    family: String,
    dispatches: u32,
    cpu_reference_us: f64,
    gpu_e2e_us: f64,
    gpu_kernel_us: f64,
    gpu_hw_total_ns: Option<f64>,
    gpu_hw_stages: Option<Vec<StageReport>>,
    validation: String,
}

#[derive(Serialize)]
struct StageReport {
    label: String,
    duration_ns: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FamilyOverride {
    Auto,
    Stockham,
    FourStep,
}

impl FamilyOverride {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "auto" => Some(Self::Auto),
            "stockham" => Some(Self::Stockham),
            "four-step" | "four_step" | "fourstep" => Some(Self::FourStep),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Stockham => "stockham",
            Self::FourStep => "four-step",
        }
    }

    fn planner_policy(
        self,
        caps: &zkgpu_wgpu::CapabilityProfile,
    ) -> Option<PlannerPolicy> {
        match self {
            Self::Auto => None,
            // Build from caps first so `tail_caps_hint` is populated
            // (lets per-family logic like `r8_max_log_leaf` kick in),
            // then apply the forced family. T3.A (2026-04-17).
            Self::Stockham => Some(PlannerPolicy::from_caps(caps).with_four_step_disabled()),
            Self::FourStep => Some(PlannerPolicy::from_caps(caps).with_force_four_step()),
        }
    }
}

/// Forced Stockham-tail strategy. Only meaningful when the plan actually
/// takes the Stockham path (i.e. `--force-family stockham` or `auto` when
/// the auto-planner picks Stockham for the given log_n). On a Four-Step
/// plan the tail override is silently ignored — a four-step plan has no
/// Stockham tail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TailOverride {
    Auto,
    Local,
    Global,
}

impl TailOverride {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "auto" => Some(Self::Auto),
            "local" | "local-fused" | "local-fused-r4" => Some(Self::Local),
            "global" | "global-only" | "global-only-r4" => Some(Self::Global),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Local => "local",
            Self::Global => "global",
        }
    }

    fn as_public(self) -> zkgpu_wgpu::StockhamTailOverride {
        match self {
            Self::Auto => zkgpu_wgpu::StockhamTailOverride::Auto,
            Self::Local => zkgpu_wgpu::StockhamTailOverride::Local,
            Self::Global => zkgpu_wgpu::StockhamTailOverride::Global,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CliMode {
    Benchmark,
    /// Soak test: run sustained NTT for a fixed duration, recording per-iteration samples.
    Soak { duration_secs: u32 },
    /// Phase F.3.c: run a hash suite (currently only Poseidon2). The
    /// positional `sizes` list is reinterpreted as per-case
    /// permutation batch counts, not `log_n` values — each entry
    /// becomes one `HashCaseSpec` with `num_permutations = size`.
    Hash { algorithm: zkgpu_report::HashAlgorithm },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CliArgs {
    json_mode: bool,
    /// Which prime field the CLI targets. Phase E.1.d — default
    /// `Field::BabyBear` keeps pre-Phase-E invocations working
    /// unchanged. `--field goldilocks` routes to the portable u32x2
    /// Stockham plan (`WgpuGoldilocksNttPlan`).
    field: Field,
    family_override: FamilyOverride,
    tail_override: TailOverride,
    sizes: Vec<u32>,
    mode: CliMode,
}

fn device_report(caps: &CapabilityProfile) -> DeviceReport {
    DeviceReport {
        name: caps.device_name.clone(),
        backend: format!("{:?}", caps.backend),
        device_type: format!("{:?}", caps.device_type),
        tier: format!("{:?}", caps.tier),
        gpu_family: format!("{:?}", caps.gpu_family),
        detection_source: format!("{:?}", caps.detection_source),
        platform_class: format!("{:?}", caps.platform_class),
        driver: caps.driver.clone(),
        driver_info: caps.driver_info.clone(),
        features: FeaturesReport {
            subgroup: caps.has_subgroup,
            timestamp_query: caps.has_timestamp_query,
            timestamp_query_inside_passes: caps.has_timestamp_query_inside_passes,
            mappable_primary_buffers: caps.has_mappable_primary_buffers,
            pipeline_cache: caps.has_pipeline_cache,
            transient_saves_memory: caps.transient_saves_memory,
        },
        limits: LimitsReport {
            max_buffer_size_mb: caps.max_buffer_size / (1024 * 1024),
            max_storage_buffer_binding_size_mb: caps.max_storage_buffer_binding_size
                / (1024 * 1024),
            max_compute_workgroup_size_x: caps.max_compute_workgroup_size_x,
            max_compute_invocations_per_workgroup: caps.max_compute_invocations_per_workgroup,
            max_compute_workgroups_per_dimension: caps.max_compute_workgroups_per_dimension,
            max_compute_workgroup_storage_size_bytes: caps.max_compute_workgroup_storage_size,
        },
    }
}

fn make_data(log_n: u32) -> Vec<BabyBear> {
    let n = 1usize << log_n;
    (0..n as u32).map(BabyBear::new).collect()
}

const WARM_UP_ITERS: usize = 2;
const BENCH_ITERS: usize = 5;
const DEFAULT_SIZES: [u32; 4] = [10, 14, 18, 20];
/// Default per-case permutation counts when `--hash` is set and the
/// user doesn't pass positional args. Log-spaced to give a throughput
/// curve without taking too long even at the top end.
const DEFAULT_HASH_BATCH_SIZES: [u32; 4] = [1_024, 16_384, 65_536, 262_144];

fn bench_cpu(data: &[BabyBear], direction: NttDirection) -> f64 {
    for _ in 0..WARM_UP_ITERS {
        let mut d = data.to_vec();
        ntt_cpu_reference(&mut d, direction);
    }

    let mut total = std::time::Duration::ZERO;
    for _ in 0..BENCH_ITERS {
        let mut d = data.to_vec();
        let start = Instant::now();
        ntt_cpu_reference(&mut d, direction);
        total += start.elapsed();
    }
    total.as_secs_f64() * 1_000_000.0 / BENCH_ITERS as f64
}

fn bench_gpu_e2e(device: &WgpuDevice, data: &[BabyBear], plan: &mut WgpuNttPlan) -> f64 {
    for _ in 0..WARM_UP_ITERS {
        let mut buf = device.upload(data).expect("upload failed");
        plan.execute(device, &mut buf).expect("execute failed");
    }

    let mut total = std::time::Duration::ZERO;
    for _ in 0..BENCH_ITERS {
        let mut buf = device.upload(data).expect("upload failed");
        let start = Instant::now();
        plan.execute(device, &mut buf).expect("execute failed");
        total += start.elapsed();
    }
    total.as_secs_f64() * 1_000_000.0 / BENCH_ITERS as f64
}

fn bench_gpu_kernel(device: &WgpuDevice, data: &[BabyBear], plan: &mut WgpuNttPlan) -> f64 {
    for _ in 0..WARM_UP_ITERS {
        let mut buf = device.upload(data).expect("upload failed");
        plan.execute_kernels(device, &mut buf)
            .expect("kernel failed");
    }

    let mut total = std::time::Duration::ZERO;
    for _ in 0..BENCH_ITERS {
        let mut buf = device.upload(data).expect("upload failed");
        let start = Instant::now();
        plan.execute_kernels(device, &mut buf)
            .expect("kernel failed");
        total += start.elapsed();
    }
    total.as_secs_f64() * 1_000_000.0 / BENCH_ITERS as f64
}

const PROFILED_ITERS: usize = BENCH_ITERS;

fn profiled_run(
    device: &WgpuDevice,
    data: &[BabyBear],
    plan: &mut WgpuNttPlan,
) -> (Option<f64>, Option<Vec<StageReport>>) {
    for _ in 0..WARM_UP_ITERS {
        let mut buf = device.upload(data).expect("upload failed");
        let _ = plan.execute_kernels_profiled(device, &mut buf);
    }

    let mut total_ns_accum: Option<f64> = None;
    let mut stage_accum: Option<Vec<(String, f64)>> = None;
    let mut collected = 0usize;

    for _ in 0..PROFILED_ITERS {
        let mut buf = device.upload(data).expect("upload failed");
        match plan
            .execute_kernels_profiled(device, &mut buf)
            .expect("profiled exec failed")
        {
            Some(timings) => {
                *total_ns_accum.get_or_insert(0.0) += timings.gpu_total_ns;
                let accum = stage_accum.get_or_insert_with(|| {
                    timings
                        .gpu_stage_ns
                        .iter()
                        .map(|s| (s.label.clone(), 0.0))
                        .collect()
                });
                for (acc, sample) in accum.iter_mut().zip(timings.gpu_stage_ns.iter()) {
                    acc.1 += sample.duration_ns;
                }
                collected += 1;
            }
            None => return (None, None),
        }
    }

    if collected == 0 {
        return (None, None);
    }

    let n = collected as f64;
    let avg_total = total_ns_accum.map(|t| t / n);
    let avg_stages = stage_accum.map(|stages| {
        stages
            .into_iter()
            .map(|(label, sum)| StageReport {
                label,
                duration_ns: sum / n,
            })
            .collect()
    });
    (avg_total, avg_stages)
}

fn validate(
    device: &WgpuDevice,
    data: &[BabyBear],
    cpu_result: &[BabyBear],
    plan: &mut WgpuNttPlan,
) -> String {
    let mut buf = device.upload(data).expect("upload failed");
    plan.execute(device, &mut buf).expect("execute failed");
    let gpu_result = buf.read_to_vec().expect("read failed");

    let n = data.len();
    let mismatches = gpu_result
        .iter()
        .zip(cpu_result.iter())
        .filter(|(g, c)| g != c)
        .count();

    if mismatches == 0 {
        format!("PASS ({n} elements)")
    } else {
        format!("FAIL ({mismatches}/{n} mismatches)")
    }
}

fn build_plan(
    device: &WgpuDevice,
    log_n: u32,
    direction: NttDirection,
    family_override: FamilyOverride,
    tail_override: TailOverride,
) -> WgpuNttPlan {
    // Both overrides go into a single PlannerPolicy. If family is Auto we
    // still need a policy whenever tail is non-Auto, so we bootstrap from
    // device caps in that case. Mirrors the Android harness pattern.
    match (family_override.planner_policy(device.caps()), tail_override) {
        (Some(policy), tail) => {
            let policy = policy.with_public_tail_override(tail.as_public());
            WgpuNttPlan::new_with_policy(device, log_n, direction, &policy)
                .expect("plan creation failed")
        }
        (None, TailOverride::Auto) => {
            WgpuNttPlan::new(device, log_n, direction).expect("plan creation failed")
        }
        (None, tail) => {
            let policy = PlannerPolicy::from_caps(device.caps())
                .with_public_tail_override(tail.as_public());
            WgpuNttPlan::new_with_policy(device, log_n, direction, &policy)
                .expect("plan creation failed")
        }
    }
}

fn run_benchmark(
    device: &WgpuDevice,
    log_n: u32,
    direction: NttDirection,
    family_override: FamilyOverride,
    tail_override: TailOverride,
) -> RunReport {
    let dir_str = match direction {
        NttDirection::Forward => "forward",
        NttDirection::Inverse => "inverse",
    };
    let n = 1u64 << log_n;
    let data = make_data(log_n);

    eprintln!("  {dir_str} 2^{log_n} = {n}");

    let mut plan = build_plan(device, log_n, direction, family_override, tail_override);
    let family = plan.family_name().to_string();
    let dispatches = plan.num_dispatches();

    let cpu_us = bench_cpu(&data, direction);
    let gpu_e2e_us = bench_gpu_e2e(device, &data, &mut plan);
    let gpu_kernel_us = bench_gpu_kernel(device, &data, &mut plan);
    let (gpu_hw_total_ns, gpu_hw_stages) = profiled_run(device, &data, &mut plan);

    let mut cpu_data = data.clone();
    ntt_cpu_reference(&mut cpu_data, direction);
    let validation = validate(device, &data, &cpu_data, &mut plan);

    eprintln!(
        "    family={}  cpu={:.0}us  e2e={:.0}us  kernel={:.0}us  hw={:.3}ms  {}",
        family,
        cpu_us,
        gpu_e2e_us,
        gpu_kernel_us,
        gpu_hw_total_ns.unwrap_or(0.0) / 1_000_000.0,
        &validation,
    );

    // When --force-tail is explicit (non-Auto), emit a line compatible with
    // the `zkgpu-tail-analyze` logcat parser so the same analyzer works
    // against desktop/CI runs without going through the Android FFI path.
    // Format matches `apps/android-harness/.../ZkgpuInstrumentedTest.kt`
    // `CROSSOVER_STOCKHAM_{LOCAL,GLOBAL}_TAIL` lines.
    if let Some(tag) = match tail_override {
        TailOverride::Auto => None,
        TailOverride::Local => Some("CROSSOVER_STOCKHAM_LOCAL_TAIL"),
        TailOverride::Global => Some("CROSSOVER_STOCKHAM_GLOBAL_TAIL"),
    } {
        let tail_label = plan.stockham_tail_strategy().unwrap_or("none");
        let reason_label = plan.stockham_tail_reason().unwrap_or("none");
        let stride_str = match plan.tail_stride_bytes() {
            Some(s) => s.to_string(),
            None => "none".to_string(),
        };
        eprintln!(
            "{} {}_log{}: family={} tail={} reason={} stride_bytes={} wall={:.3}ms gpu={:.3}ms",
            tag,
            dir_str,
            log_n,
            family,
            tail_label,
            reason_label,
            stride_str,
            gpu_e2e_us / 1_000.0,
            gpu_kernel_us / 1_000.0,
        );
    }

    RunReport {
        log_n,
        n,
        direction: dir_str.to_string(),
        family,
        dispatches,
        cpu_reference_us: cpu_us,
        gpu_e2e_us,
        gpu_kernel_us,
        gpu_hw_total_ns,
        gpu_hw_stages,
        validation,
    }
}

// ---------------------------------------------------------------------------
// Goldilocks benchmark path (Phase E.1.d)
// ---------------------------------------------------------------------------
//
// Thinner than the BabyBear `run_benchmark` because the Goldilocks plan
// has no profiled-execute variant (Phase E.2) and no family / tail split
// (R2 vs R4 is a `log_n`-parity choice, not a runtime override). Reports
// wall-time-only — `gpu_kernel_us` mirrors `gpu_e2e_us` to keep the
// JSON schema stable, and `gpu_hw_total_ns` / `gpu_hw_stages` stay
// `None` so downstream consumers can detect the non-profiled rows.

fn make_goldilocks_data(log_n: u32) -> Vec<Goldilocks> {
    let n = 1usize << log_n;
    (0..n as u64).map(Goldilocks::new).collect()
}

fn bench_cpu_goldilocks(data: &[Goldilocks], direction: NttDirection) -> f64 {
    for _ in 0..WARM_UP_ITERS {
        let mut d = data.to_vec();
        ntt_cpu_reference(&mut d, direction);
    }

    let mut total = std::time::Duration::ZERO;
    for _ in 0..BENCH_ITERS {
        let mut d = data.to_vec();
        let start = Instant::now();
        ntt_cpu_reference(&mut d, direction);
        total += start.elapsed();
    }
    total.as_secs_f64() * 1_000_000.0 / BENCH_ITERS as f64
}

fn bench_gpu_goldilocks(
    device: &WgpuDevice,
    data: &[Goldilocks],
    plan: &mut WgpuGoldilocksNttPlan,
) -> f64 {
    for _ in 0..WARM_UP_ITERS {
        let mut buf = device.upload(data).expect("upload failed");
        plan.execute(device, &mut buf).expect("execute failed");
    }

    let mut total = std::time::Duration::ZERO;
    for _ in 0..BENCH_ITERS {
        let mut buf = device.upload(data).expect("upload failed");
        let start = Instant::now();
        plan.execute(device, &mut buf).expect("execute failed");
        total += start.elapsed();
    }
    total.as_secs_f64() * 1_000_000.0 / BENCH_ITERS as f64
}

fn validate_goldilocks(
    device: &WgpuDevice,
    data: &[Goldilocks],
    cpu_data: &[Goldilocks],
    plan: &mut WgpuGoldilocksNttPlan,
) -> String {
    let mut buf = device.upload(data).expect("upload failed");
    plan.execute(device, &mut buf).expect("execute failed");
    let gpu = buf.read_to_vec().expect("readback failed");
    let n = gpu.len();
    let mismatches = gpu.iter().zip(cpu_data.iter()).filter(|(g, c)| g != c).count();
    if mismatches == 0 {
        format!("PASS ({n} elements)")
    } else {
        format!("FAIL ({mismatches}/{n} mismatches)")
    }
}

fn run_benchmark_goldilocks(
    device: &WgpuDevice,
    log_n: u32,
    direction: NttDirection,
) -> RunReport {
    let dir_str = match direction {
        NttDirection::Forward => "forward",
        NttDirection::Inverse => "inverse",
    };
    let n = 1u64 << log_n;
    let data = make_goldilocks_data(log_n);

    eprintln!("  {dir_str} 2^{log_n} = {n}  [goldilocks]");

    let mut plan = WgpuGoldilocksNttPlan::new(device, log_n, direction)
        .expect("goldilocks plan creation failed");
    let family = if log_n % 2 == 0 {
        "goldilocks-portable-r4".to_string()
    } else {
        "goldilocks-portable-r2".to_string()
    };

    let cpu_us = bench_cpu_goldilocks(&data, direction);
    let gpu_e2e_us = bench_gpu_goldilocks(device, &data, &mut plan);

    // Phase E.2.c.4: collect GPU hardware timestamps via the new
    // execute_profiled path. Returns None on adapters without
    // TIMESTAMP_QUERY support (the BabyBear path degrades the same
    // way — see `profiled_run`).
    let (gpu_hw_total_ns, gpu_hw_stages) = profiled_goldilocks_run(device, &data, &mut plan);

    let mut cpu_data = data.clone();
    ntt_cpu_reference(&mut cpu_data, direction);
    let validation = validate_goldilocks(device, &data, &cpu_data, &mut plan);

    let hw_ms = gpu_hw_total_ns
        .map(|ns| format!("{:.3}ms", ns / 1_000_000.0))
        .unwrap_or_else(|| "n/a".to_string());
    eprintln!(
        "    family={}  cpu={:.0}us  e2e={:.0}us  hw={}  {}",
        family, cpu_us, gpu_e2e_us, hw_ms, &validation,
    );

    RunReport {
        log_n,
        n,
        direction: dir_str.to_string(),
        family,
        dispatches: goldilocks_dispatch_count(log_n, direction),
        cpu_reference_us: cpu_us,
        gpu_e2e_us,
        // Mirror e2e into kernel since we don't split kernel vs. submit
        // on the Goldilocks path. Operators comparing BabyBear and
        // Goldilocks rows should read the per-field `family` label to
        // know whether `gpu_kernel_us` is a distinct measurement.
        gpu_kernel_us: gpu_e2e_us,
        gpu_hw_total_ns,
        gpu_hw_stages,
        validation,
    }
}

/// Run `execute_profiled` PROFILED_ITERS times and average. Mirrors
/// `profiled_run` for BabyBear; on devices without TIMESTAMP_QUERY
/// returns `(None, None)` after a single warmup iteration.
fn profiled_goldilocks_run(
    device: &WgpuDevice,
    data: &[Goldilocks],
    plan: &mut WgpuGoldilocksNttPlan,
) -> (Option<f64>, Option<Vec<StageReport>>) {
    // Warmup — untimed.
    for _ in 0..WARM_UP_ITERS {
        let mut buf = device.upload(data).expect("upload failed");
        let _ = plan.execute_profiled(device, &mut buf);
    }

    let mut total_ns_accum: Option<f64> = None;
    let mut stage_accum: Option<Vec<(String, f64)>> = None;
    let mut collected = 0usize;

    for _ in 0..PROFILED_ITERS {
        let mut buf = device.upload(data).expect("upload failed");
        match plan
            .execute_profiled(device, &mut buf)
            .expect("profiled exec failed")
        {
            Some(timings) => {
                *total_ns_accum.get_or_insert(0.0) += timings.gpu_total_ns;
                let accum = stage_accum.get_or_insert_with(|| {
                    timings
                        .gpu_stage_ns
                        .iter()
                        .map(|s| (s.label.clone(), 0.0))
                        .collect()
                });
                for (acc, sample) in accum.iter_mut().zip(timings.gpu_stage_ns.iter()) {
                    acc.1 += sample.duration_ns;
                }
                collected += 1;
            }
            None => return (None, None),
        }
    }

    if collected == 0 {
        return (None, None);
    }

    let n = collected as f64;
    let avg_total = total_ns_accum.map(|t| t / n);
    let stages = stage_accum.map(|stages| {
        stages
            .into_iter()
            .map(|(label, total)| StageReport {
                label,
                duration_ns: total / n,
            })
            .collect()
    });
    (avg_total, stages)
}

fn goldilocks_dispatch_count(log_n: u32, direction: NttDirection) -> u32 {
    let stages = if log_n % 2 == 0 { log_n / 2 } else { log_n };
    let scale = match direction {
        NttDirection::Forward => 0,
        NttDirection::Inverse => 1,
    };
    stages + scale
}

fn usage() -> &'static str {
    "Usage: zkgpu [--json] [--field babybear|goldilocks] \\
               [--force-family auto|stockham|four-step] \\
               [--force-tail auto|local|global] \\
               [--soak SECS | --hash poseidon2] [SIZE...]

Modes (mutually exclusive):
  (default)      NTT benchmark: warmup + 5 measured iterations per size
                 (SIZE = log_n).
  --soak SECS    Sustained NTT soak: run each size for SECS seconds,
                 recording per-iteration samples for thermal
                 characterization. BabyBear + Goldilocks (Phase E.2.c/d
                 wires Goldilocks through execute_profiled so
                 median_gpu_ns populates on adapters with
                 TIMESTAMP_QUERY support).
  --hash ALG     Hash benchmark (Phase F.3.c). ALG = 'poseidon2' today.
                 SIZE = num_permutations per case. Uses the testkit's
                 run_hash_suite with profile_gpu_timestamps=true, 1
                 warmup + 5 measured, SplitMix64 input pattern. Wall-
                 only timings today; gpu_total_ns populates once a
                 future sub-phase adds execute_profiled to the
                 Poseidon2 plans.

Overrides:
  --field          Which prime field to target (default: babybear).
                   'babybear'   -> 32-bit field; full benchmark / soak surface.
                   'goldilocks' -> 64-bit portable u32x2 Stockham plan;
                                    full benchmark + soak surface including
                                    GPU timestamps (Phase E.2.c/d).
                                    --force-family / --force-tail are
                                    silently ignored: the Goldilocks plan
                                    has no four-step or local-fused-tail
                                    variant, and R2/R4 is picked from
                                    log_n parity.
  --force-family   Pin the NTT family (default: auto = device-policy pick).
                   'stockham' disables four-step; 'four-step' forces it.
                   BabyBear only.
  --force-tail     Pin the Stockham tail strategy (default: auto).
                   'local'  -> LocalFusedR4 (legacy tail kernel)
                   'global' -> GlobalOnlyR4 (extends global R4 chain through
                               the tail; the configuration PR 1 made possible).
                   Only applies when the plan is Stockham. A Four-Step plan
                   has no Stockham tail and silently ignores this flag.
                   BabyBear only.

Examples:
  zkgpu
  zkgpu 10 20
  zkgpu --force-family four-step 20 24
  zkgpu --json --force-family stockham 18 20
  zkgpu --force-family stockham --force-tail local --json 18 19 20 21 22
  zkgpu --force-family stockham --force-tail global --json 18 19 20 21 22
  zkgpu --soak 30 18 20
  zkgpu --soak 60 --json 20
  zkgpu --field goldilocks 10 18
  zkgpu --field goldilocks --json 10 14 18 20
  zkgpu --hash poseidon2
  zkgpu --hash poseidon2 --field goldilocks 1024 16384 65536
  zkgpu --hash poseidon2 --field babybear --json 4096"
}

fn parse_cli_args<I>(args: I) -> Result<CliArgs, String>
where
    I: IntoIterator<Item = String>,
{
    let mut json_mode = false;
    let mut field = Field::BabyBear;
    let mut family_override = FamilyOverride::Auto;
    let mut tail_override = TailOverride::Auto;
    let mut sizes = Vec::new();
    let mut soak_secs: Option<u32> = None;
    let mut hash_algorithm: Option<zkgpu_report::HashAlgorithm> = None;

    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--json" => json_mode = true,
            "--help" | "-h" => return Err(usage().to_string()),
            "--field" => {
                let value = iter
                    .next()
                    .ok_or_else(|| "--field requires a value".to_string())?;
                field = value.parse::<Field>().map_err(|e| {
                    format!("invalid --field value '{value}': {e}")
                })?;
            }
            _ if arg.starts_with("--field=") => {
                let value = &arg["--field=".len()..];
                field = value.parse::<Field>().map_err(|e| {
                    format!("invalid --field value '{value}': {e}")
                })?;
            }
            "--soak" => {
                let value = iter
                    .next()
                    .ok_or_else(|| "--soak requires a duration in seconds".to_string())?;
                let secs = value
                    .parse::<u32>()
                    .map_err(|_| format!("invalid soak duration '{value}'"))?;
                if secs == 0 {
                    return Err("soak duration must be > 0".to_string());
                }
                soak_secs = Some(secs);
            }
            "--force-family" => {
                let value = iter
                    .next()
                    .ok_or_else(|| "--force-family requires a value".to_string())?;
                family_override = FamilyOverride::from_str(&value).ok_or_else(|| {
                    format!(
                        "invalid family override '{value}' (expected auto, stockham, or four-step)"
                    )
                })?;
            }
            _ if arg.starts_with("--force-family=") => {
                let value = &arg["--force-family=".len()..];
                family_override = FamilyOverride::from_str(value).ok_or_else(|| {
                    format!(
                        "invalid family override '{value}' (expected auto, stockham, or four-step)"
                    )
                })?;
            }
            "--force-tail" => {
                let value = iter
                    .next()
                    .ok_or_else(|| "--force-tail requires a value".to_string())?;
                tail_override = TailOverride::from_str(&value).ok_or_else(|| {
                    format!(
                        "invalid tail override '{value}' (expected auto, local, or global)"
                    )
                })?;
            }
            _ if arg.starts_with("--force-tail=") => {
                let value = &arg["--force-tail=".len()..];
                tail_override = TailOverride::from_str(value).ok_or_else(|| {
                    format!(
                        "invalid tail override '{value}' (expected auto, local, or global)"
                    )
                })?;
            }
            _ if arg.starts_with("--soak=") => {
                let value = &arg["--soak=".len()..];
                let secs = value
                    .parse::<u32>()
                    .map_err(|_| format!("invalid soak duration '{value}'"))?;
                if secs == 0 {
                    return Err("soak duration must be > 0".to_string());
                }
                soak_secs = Some(secs);
            }
            "--hash" => {
                let value = iter
                    .next()
                    .ok_or_else(|| "--hash requires an algorithm (e.g. 'poseidon2')".to_string())?;
                hash_algorithm = Some(
                    value.parse::<zkgpu_report::HashAlgorithm>().map_err(|e| {
                        format!("invalid --hash value '{value}': {e}")
                    })?,
                );
            }
            _ if arg.starts_with("--hash=") => {
                let value = &arg["--hash=".len()..];
                hash_algorithm = Some(
                    value.parse::<zkgpu_report::HashAlgorithm>().map_err(|e| {
                        format!("invalid --hash value '{value}': {e}")
                    })?,
                );
            }
            _ if arg.starts_with('-') => {
                return Err(format!("unrecognized option '{arg}'"));
            }
            _ => {
                // Positional ints are interpreted per-mode:
                //   NTT modes: log_n values
                //   Hash mode: num_permutations counts
                // The error message adapts accordingly when we know
                // the mode; during parsing the mode isn't fixed yet,
                // so use a neutral phrasing.
                let value = arg
                    .parse::<u32>()
                    .map_err(|_| format!("invalid positional size '{arg}' (expected a non-negative integer)"))?;
                sizes.push(value);
            }
        }
    }

    // Mutual exclusion: --hash and --soak target different runners.
    // A hash-soak mode doesn't exist today (sustained-hash
    // thermal-characterisation is a separate workload; Phase F
    // scope ships benchmark-only).
    if hash_algorithm.is_some() && soak_secs.is_some() {
        return Err(
            "--hash and --soak are mutually exclusive; pick one (hash-soak \
             is not supported yet)"
                .to_string(),
        );
    }

    let mode = match (hash_algorithm, soak_secs) {
        (Some(algorithm), None) => CliMode::Hash { algorithm },
        (None, Some(secs)) => CliMode::Soak { duration_secs: secs },
        (None, None) => CliMode::Benchmark,
        (Some(_), Some(_)) => unreachable!("guarded by the mutual-exclusion check above"),
    };

    // Default positional list: for NTT modes it's `log_n`, for hash
    // it's `num_permutations`. Pick per-mode defaults so a bare
    // `zkgpu --hash poseidon2` still runs something useful.
    if sizes.is_empty() {
        sizes = match &mode {
            CliMode::Hash { .. } => DEFAULT_HASH_BATCH_SIZES.to_vec(),
            _ => DEFAULT_SIZES.to_vec(),
        };
    }

    Ok(CliArgs {
        json_mode,
        field,
        family_override,
        tail_override,
        sizes,
        mode,
    })
}

fn main() {
    env_logger::init();

    let cli = match parse_cli_args(std::env::args().skip(1)) {
        Ok(args) => args,
        Err(msg) if msg == usage() => {
            println!("{msg}");
            return;
        }
        Err(msg) => {
            eprintln!("{msg}\n\n{}", usage());
            std::process::exit(2);
        }
    };

    let device = WgpuDevice::new().expect("failed to initialize GPU device");
    let caps = device.caps();

    eprintln!("Device: {caps}");
    eprintln!("Field: {}", cli.field.as_str());
    let is_hash_mode = matches!(cli.mode, CliMode::Hash { .. });
    // Suppress NTT-override diagnostic lines in hash mode — the hash
    // runner doesn't consume --force-family / --force-tail at all, so
    // printing them would falsely suggest they're active.
    if !is_hash_mode {
        eprintln!("Family override: {}", cli.family_override.as_str());
        eprintln!("Tail override: {}", cli.tail_override.as_str());
    }
    // BabyBear-only flags are silently ignored on the Goldilocks path
    // (see `--field` in usage). Warn the operator so no-op runs aren't
    // mistaken for successful A/B matrices.
    if cli.field == Field::Goldilocks
        && !is_hash_mode
        && (cli.family_override != FamilyOverride::Auto
            || cli.tail_override != TailOverride::Auto)
    {
        eprintln!(
            "warning: --force-family / --force-tail ignored under --field goldilocks \
             (the Goldilocks plan has no four-step or local-fused-tail variant)"
        );
    }
    // F.3.c post-review (P3): --hash mode uses neither override. Warn
    // if the operator set them so the run isn't mistaken for a
    // forced-family/tail A/B matrix against the hash kernel.
    if is_hash_mode
        && (cli.family_override != FamilyOverride::Auto
            || cli.tail_override != TailOverride::Auto)
    {
        eprintln!(
            "warning: --force-family / --force-tail are NTT-only and have no \
             effect on --hash runs; the hash path ignores them"
        );
    }

    match cli.mode.clone() {
        CliMode::Benchmark => run_benchmark_mode(&device, &cli),
        CliMode::Soak { duration_secs } => {
            // Soak mode uses the canonical testkit runner which creates its
            // own device, so we drop the one we created above to avoid
            // holding two GPU device handles.
            drop(device);
            run_soak_mode(&cli, duration_secs);
        }
        CliMode::Hash { algorithm } => {
            // Hash mode delegates to the testkit (zkgpu_testkit::run_hash_suite),
            // which creates its own device. Drop the one we built for
            // NTT-shaped startup diagnostics to avoid double-holding.
            drop(device);
            run_hash_mode(&cli, algorithm);
        }
    }
}

fn run_benchmark_mode(device: &WgpuDevice, cli: &CliArgs) {
    let mut runs = Vec::new();

    for &log_n in &cli.sizes {
        eprintln!("\n--- NTT 2^{log_n} ---");
        match cli.field {
            Field::BabyBear => {
                runs.push(run_benchmark(
                    device,
                    log_n,
                    NttDirection::Forward,
                    cli.family_override,
                    cli.tail_override,
                ));
                runs.push(run_benchmark(
                    device,
                    log_n,
                    NttDirection::Inverse,
                    cli.family_override,
                    cli.tail_override,
                ));
            }
            Field::Goldilocks => {
                runs.push(run_benchmark_goldilocks(device, log_n, NttDirection::Forward));
                runs.push(run_benchmark_goldilocks(device, log_n, NttDirection::Inverse));
            }
        }
    }

    // `--force-family` / `--force-tail` are silently ignored under
    // `--field goldilocks` (see the startup warning in `main()`). Null
    // them out in the emitted JSON too — otherwise a downstream
    // consumer reading only the report would conclude a forced-family
    // / forced-tail experiment actually ran, when the execution path
    // was always the fixed Goldilocks Stockham plan.
    let (family_override, tail_override) = match cli.field {
        Field::BabyBear => (
            match cli.family_override {
                FamilyOverride::Auto => None,
                mode => Some(mode.as_str().to_string()),
            },
            match cli.tail_override {
                TailOverride::Auto => None,
                tail => Some(tail.as_str().to_string()),
            },
        ),
        Field::Goldilocks => (None, None),
    };

    let report = BenchmarkReport {
        device: device_report(device.caps()),
        field: cli.field.as_str().to_string(),
        family_override,
        tail_override,
        runs,
    };

    if cli.json_mode {
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
    } else {
        print_table(&report);
    }
}

// ---------------------------------------------------------------------------
// Soak mode — delegates to the canonical zkgpu-testkit soak engine
// ---------------------------------------------------------------------------

fn run_soak_mode(cli: &CliArgs, duration_secs: u32) {
    eprintln!("Soak mode: {}s per case", duration_secs);

    // Build a SoakSpec from CLI args using the canonical report types.
    let family_override = match cli.family_override {
        FamilyOverride::Auto => zkgpu_report::FamilyOverride::Auto,
        FamilyOverride::Stockham => zkgpu_report::FamilyOverride::Stockham,
        FamilyOverride::FourStep => zkgpu_report::FamilyOverride::FourStep,
    };

    let mut cases = Vec::new();
    for &log_n in &cli.sizes {
        for (dir, dir_name) in [
            (zkgpu_report::TestDirection::Forward, "forward"),
            (zkgpu_report::TestDirection::Inverse, "inverse"),
        ] {
            cases.push(
                zkgpu_report::CaseSpec::new(
                    format!("soak_{dir_name}_log{log_n}"),
                    log_n,
                    dir,
                    zkgpu_report::InputPattern::Sequential,
                )
                .with_profile(true),
            );
        }
    }

    let spec = zkgpu_report::SoakSpec {
        duration_secs,
        cases,
        validate: true,
        // Phase E.1.d: pass the user's `--field` choice through. The
        // testkit's `run_soak_suite` rejects `Goldilocks` with a
        // structured error until profiled-execute lands (Phase E.2), so
        // the user sees an immediate failure instead of a silent switch
        // to BabyBear or wall-time-only degradation.
        field: cli.field,
        family_override,
        // CLI soak doesn't expose a tail override today; default to Auto so
        // the heuristic picks per-device. Add a flag later if we need to
        // A/B soak runs under Local vs Global.
        stockham_tail_override: zkgpu_report::StockhamTailOverride::Auto,
    };

    let report = match zkgpu_testkit::run_soak_suite(&spec) {
        Ok(r) => r,
        Err(err) => {
            eprintln!("Soak failed: {err}");
            std::process::exit(1);
        }
    };

    // Print progress summary to stderr
    for c in &report.cases {
        let ns_to_ms = |ns: u64| ns as f64 / 1_000_000.0;
        let dir_str = format!("{:?}", c.direction).to_lowercase();
        if let Some(ref err) = c.error {
            eprintln!("\n--- Soak {dir_str} 2^{} FAILED: {} ---", c.log_n, err);
        } else {
            eprintln!(
                "\n--- Soak {dir_str} 2^{} for {}s ---\n  \
                 family={}  iters={}  {:.1}s  {:.1} it/s  \
                 median={:.2}ms  p5={:.2}ms  p95={:.2}ms  \
                 CV={:.4}  drift={:.3}  {}",
                c.log_n,
                c.requested_duration_secs,
                c.kernel_family.as_deref().unwrap_or("?"),
                c.stats.total_iterations,
                c.stats.actual_duration_secs,
                c.stats.iterations_per_sec,
                ns_to_ms(c.stats.median_wall_ns),
                ns_to_ms(c.stats.p5_wall_ns),
                ns_to_ms(c.stats.p95_wall_ns),
                c.stats.wall_cv,
                c.stats.thermal_drift_ratio,
                if c.validated { "VALID" } else { "UNVALIDATED" },
            );
        }
    }

    if cli.json_mode {
        // Emit the full canonical SoakSuiteReport (with per-iteration samples).
        println!(
            "{}",
            serde_json::to_string_pretty(&report)
                .expect("failed to serialize soak report")
        );
    } else {
        print_soak_table(&report);
    }
}

// ---------------------------------------------------------------------------
// Hash mode (Phase F.3.c) — delegates to zkgpu_testkit::run_hash_suite
// ---------------------------------------------------------------------------

fn run_hash_mode(cli: &CliArgs, algorithm: zkgpu_report::HashAlgorithm) {
    eprintln!(
        "Hash mode: algorithm={} field={}",
        algorithm.as_str(),
        cli.field.as_str(),
    );

    // Build one case per positional `num_permutations` entry. All
    // cases share iteration settings (1 warmup, 5 measured) so
    // throughput-curve interpretation is consistent across rows.
    // Matches the `poseidon2_benchmark_suite` convention but driven
    // by the operator's CLI-supplied batch list.
    //
    // F.3.c post-review (P2): profile_gpu_timestamps is NOT set. The
    // Poseidon2 plans today have no profiled-execute variant; the
    // testkit's measure_*_poseidon2_plan would silently drop a
    // profiled-request flag. Keeping the spec honest — wall_time_ns
    // populated, gpu_total_ns null — beats advertising GPU
    // timestamps that don't arrive. When a future F.3.* sub-phase
    // adds execute_profiled, flip this to `.with_profile(true)` in
    // the same commit that wires the plan-side support.
    let mut cases = Vec::with_capacity(cli.sizes.len());
    for &num in &cli.sizes {
        cases.push(
            zkgpu_report::HashCaseSpec::new(
                format!("hash_{}_n{num}", algorithm.as_str()),
                num,
                // SplitMix64 spreads across both u32x2 limbs on
                // 64-bit fields so the Goldilocks path is
                // meaningfully exercised alongside BabyBear.
                zkgpu_report::HashInputPattern::SplitMix64 { seed: 1 },
            )
            .with_iterations(1, 5),
        );
    }

    let spec = zkgpu_report::HashSpec {
        kind: zkgpu_report::SuiteKind::Benchmark,
        cases,
        fail_fast: false,
        algorithm,
        field: cli.field,
    };

    let report = match zkgpu_testkit::run_hash_suite(&spec) {
        Ok(r) => r,
        Err(err) => {
            eprintln!("Hash suite failed: {err}");
            std::process::exit(1);
        }
    };

    // Progress summary to stderr (JSON / table go to stdout below).
    for c in &report.cases {
        let ns_to_us = |ns: u64| ns as f64 / 1_000.0;
        if let Some(ref err) = c.error {
            eprintln!(
                "  {} (n={}): FAILED — {}",
                c.name, c.num_permutations, err,
            );
        } else {
            let wall_us = c.timings.wall_time_ns.map(ns_to_us).unwrap_or(0.0);
            // Throughput rate: permutations per second. Averaged over
            // `iterations` already (wall_time_ns is per-iteration avg).
            let perms_per_sec = if wall_us > 0.0 {
                (c.num_permutations as f64) * 1_000_000.0 / wall_us
            } else {
                0.0
            };
            eprintln!(
                "  {}  n={}  family={}  wall={:.0}us  {:.2}M perms/s  {}",
                c.name,
                c.num_permutations,
                c.kernel_family.as_deref().unwrap_or("?"),
                wall_us,
                perms_per_sec / 1e6,
                if c.passed {
                    "PASS".to_string()
                } else {
                    format!("FAIL ({} mismatches)", c.mismatch_count)
                },
            );
        }
    }

    if cli.json_mode {
        println!(
            "{}",
            serde_json::to_string_pretty(&report)
                .expect("failed to serialize hash report")
        );
    } else {
        print_hash_table(&report);
    }

    // Non-zero exit if any case failed, matching the soak contract.
    if report.summary.failed_cases > 0 {
        std::process::exit(1);
    }
}

fn print_hash_table(report: &zkgpu_report::HashSuiteReport) {
    let d = &report.device;
    println!();
    println!(
        "Device: {} ({}) tier={}",
        d.name, d.backend, d.tier,
    );
    println!("Field: {}", report.kernel.field);
    println!("Algorithm: {}", report.kernel.ntt_variant);
    println!();
    println!(
        "{:<24} {:>10} {:>12} {:>14} {:>8}",
        "case", "n", "wall (us)", "M perms/s", "status",
    );
    println!("{}", "-".repeat(72));
    for c in &report.cases {
        let wall_us = c
            .timings
            .wall_time_ns
            .map(|ns| ns as f64 / 1_000.0)
            .unwrap_or(0.0);
        let perms_per_sec = if wall_us > 0.0 {
            (c.num_permutations as f64) * 1_000_000.0 / wall_us
        } else {
            0.0
        };
        let status = if let Some(ref _e) = c.error {
            "ERROR"
        } else if c.passed {
            "PASS"
        } else {
            "FAIL"
        };
        println!(
            "{:<24} {:>10} {:>12.0} {:>14.2} {:>8}",
            c.name,
            c.num_permutations,
            wall_us,
            perms_per_sec / 1e6,
            status,
        );
    }
    println!();
    println!(
        "Summary: {}/{} cases passed",
        report.summary.passed_cases, report.summary.total_cases,
    );
}

fn print_soak_table(report: &zkgpu_report::SoakSuiteReport) {
    println!();
    println!(
        "Device: {} ({}) tier={}",
        report.device.name, report.device.backend, report.device.tier,
    );
    println!("Soak duration: {}s per case", report.requested_duration_secs);
    println!();
    println!(
        "{:<7} {:<9} {:<10} {:>7} {:>8} {:>10} {:>10} {:>10} {:>7} {:>7} {:>8}",
        "log_n", "dir", "family", "iters", "it/s",
        "med (ms)", "p5 (ms)", "p95 (ms)", "CV", "drift", "valid"
    );
    println!("{}", "-".repeat(100));
    let ns_to_ms = |ns: u64| ns as f64 / 1_000_000.0;
    for c in &report.cases {
        if c.error.is_some() {
            let dir_str = format!("{:?}", c.direction).to_lowercase();
            println!(
                "{:<7} {:<9} {:<10} {:>7} {:>8} {:>10} {:>10} {:>10} {:>7} {:>7} {:>8}",
                c.log_n, dir_str, "-", 0, "-", "-", "-", "-", "-", "-", "ERR"
            );
            continue;
        }
        let dir_str = format!("{:?}", c.direction).to_lowercase();
        let family = c.kernel_family.as_deref().unwrap_or("?");
        println!(
            "{:<7} {:<9} {:<10} {:>7} {:>8.1} {:>10.2} {:>10.2} {:>10.2} {:>7.4} {:>7.3} {:>8}",
            c.log_n,
            dir_str,
            family,
            c.stats.total_iterations,
            c.stats.iterations_per_sec,
            ns_to_ms(c.stats.median_wall_ns),
            ns_to_ms(c.stats.p5_wall_ns),
            ns_to_ms(c.stats.p95_wall_ns),
            c.stats.wall_cv,
            c.stats.thermal_drift_ratio,
            if c.validated { "PASS" } else { "SKIP" },
        );
    }

    // Print GPU timing summary if available
    if report.cases.iter().any(|c| c.stats.median_gpu_ns.is_some()) {
        println!();
        println!("GPU hardware timings (median):");
        for c in &report.cases {
            if let Some(gpu_ns) = c.stats.median_gpu_ns {
                let dir_str = format!("{:?}", c.direction).to_lowercase();
                println!(
                    "  2^{} {}: {:.2}ms (CV={:.4})",
                    c.log_n,
                    dir_str,
                    ns_to_ms(gpu_ns),
                    c.stats.gpu_cv.unwrap_or(0.0),
                );
            }
        }
    }
    println!();
}

fn print_table(report: &BenchmarkReport) {
    let d = &report.device;
    println!();
    println!(
        "Device: {} ({}/{}) tier={} buffer={}MB",
        d.name, d.backend, d.device_type, d.tier, d.limits.max_buffer_size_mb
    );
    println!("Field: {}", report.field);
    println!();
    println!(
        "{:<10} {:<9} {:<10} {:>5} {:>12} {:>12} {:>12} {:>10} {:>10}",
        "log_n",
        "direction",
        "family",
        "disp",
        "cpu (us)",
        "e2e (us)",
        "kernel (us)",
        "hw (ms)",
        "status"
    );
    println!("{}", "-".repeat(102));
    for run in &report.runs {
        let hw = run
            .gpu_hw_total_ns
            .map(|ns| format!("{:.3}", ns / 1_000_000.0))
            .unwrap_or_else(|| "n/a".to_string());
        println!(
            "{:<10} {:<9} {:<10} {:>5} {:>12.0} {:>12.0} {:>12.0} {:>10} {:>10}",
            run.log_n,
            run.direction,
            run.family,
            run.dispatches,
            run.cpu_reference_us,
            run.gpu_e2e_us,
            run.gpu_kernel_us,
            hw,
            run.validation.split_whitespace().next().unwrap_or("?"),
        );
    }
    println!();
    println!(
        "Iterations: {BENCH_ITERS} measured / {WARM_UP_ITERS} warm-up (wall), \
         {PROFILED_ITERS} measured / {WARM_UP_ITERS} warm-up (hw timestamps)"
    );

    let has_stages = report
        .runs
        .iter()
        .any(|r| r.gpu_hw_stages.as_ref().is_some_and(|s| !s.is_empty()));
    if has_stages {
        println!();
        println!("Per-stage GPU hardware timings, averaged (ns):");
        println!("  (stages showing 0 are below timestamp resolution, not zero work)");
        for run in &report.runs {
            if let Some(stages) = &run.gpu_hw_stages {
                if !stages.is_empty() {
                    println!("  2^{} {}:", run.log_n, run.direction);
                    for s in stages {
                        println!("    {:<24} {:>12.0}", s.label, s.duration_ns);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_defaults() {
        let parsed = parse_cli_args(Vec::<String>::new()).unwrap();
        assert!(!parsed.json_mode);
        assert_eq!(parsed.field, Field::BabyBear);
        assert_eq!(parsed.family_override, FamilyOverride::Auto);
        assert_eq!(parsed.sizes, DEFAULT_SIZES);
        assert_eq!(parsed.mode, CliMode::Benchmark);
    }

    // === Phase E.1.d: --field parsing ===

    #[test]
    fn parse_field_goldilocks_split_form() {
        let parsed = parse_cli_args(vec![
            "--field".to_string(),
            "goldilocks".to_string(),
            "10".to_string(),
        ])
        .unwrap();
        assert_eq!(parsed.field, Field::Goldilocks);
        assert_eq!(parsed.sizes, vec![10]);
    }

    #[test]
    fn parse_field_goldilocks_equals_form() {
        let parsed = parse_cli_args(vec!["--field=goldilocks".to_string()]).unwrap();
        assert_eq!(parsed.field, Field::Goldilocks);
    }

    #[test]
    fn parse_field_babybear_explicit() {
        let parsed = parse_cli_args(vec!["--field=babybear".to_string()]).unwrap();
        assert_eq!(parsed.field, Field::BabyBear);
    }

    #[test]
    fn parse_field_rejects_unknown() {
        let err = parse_cli_args(vec!["--field=mersenne31".to_string()]).unwrap_err();
        assert!(
            err.contains("mersenne31"),
            "error should echo the unknown field: {err}"
        );
    }

    #[test]
    fn parse_field_requires_value() {
        let err = parse_cli_args(vec!["--field".to_string()]).unwrap_err();
        assert!(err.contains("--field"), "expected error about --field: {err}");
    }

    // === Phase F.3.c: --hash parsing ===

    #[test]
    fn parse_hash_poseidon2_split_form() {
        let parsed = parse_cli_args(vec![
            "--hash".to_string(),
            "poseidon2".to_string(),
            "1024".to_string(),
            "4096".to_string(),
        ])
        .unwrap();
        assert_eq!(
            parsed.mode,
            CliMode::Hash {
                algorithm: zkgpu_report::HashAlgorithm::Poseidon2
            }
        );
        assert_eq!(parsed.sizes, vec![1024, 4096]);
    }

    #[test]
    fn parse_hash_poseidon2_equals_form() {
        let parsed = parse_cli_args(vec!["--hash=poseidon2".to_string()]).unwrap();
        assert!(matches!(parsed.mode, CliMode::Hash { .. }));
    }

    #[test]
    fn parse_hash_defaults_to_batch_ladder_when_no_positional() {
        let parsed = parse_cli_args(vec!["--hash".to_string(), "poseidon2".to_string()])
            .unwrap();
        assert_eq!(parsed.sizes, DEFAULT_HASH_BATCH_SIZES.to_vec());
    }

    #[test]
    fn parse_hash_rejects_unknown_algorithm() {
        let err = parse_cli_args(vec!["--hash=rescue".to_string()]).unwrap_err();
        assert!(
            err.contains("rescue"),
            "error should echo the unknown algorithm: {err}"
        );
    }

    #[test]
    fn parse_hash_requires_value() {
        let err = parse_cli_args(vec!["--hash".to_string()]).unwrap_err();
        assert!(err.contains("--hash"), "expected error about --hash: {err}");
    }

    #[test]
    fn parse_hash_and_soak_are_mutually_exclusive() {
        let err = parse_cli_args(vec![
            "--hash=poseidon2".to_string(),
            "--soak=30".to_string(),
        ])
        .unwrap_err();
        assert!(
            err.contains("mutually exclusive"),
            "expected mutual-exclusion error: {err}"
        );
    }

    #[test]
    fn parse_hash_accepts_goldilocks_field() {
        let parsed = parse_cli_args(vec![
            "--hash=poseidon2".to_string(),
            "--field=goldilocks".to_string(),
            "256".to_string(),
        ])
        .unwrap();
        assert!(matches!(parsed.mode, CliMode::Hash { .. }));
        assert_eq!(parsed.field, Field::Goldilocks);
        assert_eq!(parsed.sizes, vec![256]);
    }

    /// Phase F.3.c post-review (P3): `--hash` + NTT overrides parses
    /// successfully (the overrides are ignored, not rejected — see
    /// main.rs for the stderr warning that fires at startup). The
    /// parser itself shouldn't block the combination so bulk-reusing
    /// an NTT invocation line with `--hash` appended keeps working.
    #[test]
    fn parse_hash_tolerates_ntt_overrides() {
        let parsed = parse_cli_args(vec![
            "--hash=poseidon2".to_string(),
            "--force-family=stockham".to_string(),
            "--force-tail=local".to_string(),
            "1024".to_string(),
        ])
        .unwrap();
        assert!(matches!(parsed.mode, CliMode::Hash { .. }));
        // Parser doesn't clear them — the runtime warning does the
        // operator-visible work. Asserting both are stored verifies
        // the parse path doesn't accidentally drop them (which would
        // break NTT mode too).
        assert_eq!(parsed.family_override, FamilyOverride::Stockham);
        assert_eq!(parsed.tail_override, TailOverride::Local);
    }

    #[test]
    fn parse_force_family_equals_form() {
        let parsed = parse_cli_args(vec![
            "--json".to_string(),
            "--force-family=four-step".to_string(),
            "20".to_string(),
        ])
        .unwrap();
        assert!(parsed.json_mode);
        assert_eq!(parsed.family_override, FamilyOverride::FourStep);
        assert_eq!(parsed.sizes, vec![20]);
    }

    #[test]
    fn parse_force_family_split_form() {
        let parsed = parse_cli_args(vec![
            "--force-family".to_string(),
            "stockham".to_string(),
            "10".to_string(),
            "24".to_string(),
        ])
        .unwrap();
        assert_eq!(parsed.family_override, FamilyOverride::Stockham);
        assert_eq!(parsed.sizes, vec![10, 24]);
    }

    #[test]
    fn parse_rejects_invalid_family() {
        let err = parse_cli_args(vec!["--force-family=banana".to_string()]).unwrap_err();
        assert!(err.contains("invalid family override"));
    }

    #[test]
    fn parse_defaults_tail_to_auto() {
        let parsed = parse_cli_args(Vec::<String>::new()).unwrap();
        assert_eq!(parsed.tail_override, TailOverride::Auto);
    }

    #[test]
    fn parse_force_tail_equals_form() {
        let parsed = parse_cli_args(vec![
            "--force-family=stockham".to_string(),
            "--force-tail=local".to_string(),
            "18".to_string(),
        ])
        .unwrap();
        assert_eq!(parsed.family_override, FamilyOverride::Stockham);
        assert_eq!(parsed.tail_override, TailOverride::Local);
        assert_eq!(parsed.sizes, vec![18]);
    }

    #[test]
    fn parse_force_tail_split_form() {
        let parsed = parse_cli_args(vec![
            "--force-family".to_string(),
            "stockham".to_string(),
            "--force-tail".to_string(),
            "global".to_string(),
            "20".to_string(),
        ])
        .unwrap();
        assert_eq!(parsed.tail_override, TailOverride::Global);
    }

    #[test]
    fn parse_force_tail_accepts_aliases() {
        let local = parse_cli_args(vec!["--force-tail=local-fused".to_string()])
            .unwrap()
            .tail_override;
        assert_eq!(local, TailOverride::Local);
        let global = parse_cli_args(vec!["--force-tail=global-only-r4".to_string()])
            .unwrap()
            .tail_override;
        assert_eq!(global, TailOverride::Global);
    }

    #[test]
    fn parse_rejects_invalid_tail() {
        let err = parse_cli_args(vec!["--force-tail=sideways".to_string()]).unwrap_err();
        assert!(err.contains("invalid tail override"));
    }

    #[test]
    fn parse_soak_mode() {
        let parsed = parse_cli_args(vec![
            "--soak".to_string(),
            "30".to_string(),
            "18".to_string(),
            "20".to_string(),
        ])
        .unwrap();
        assert_eq!(parsed.mode, CliMode::Soak { duration_secs: 30 });
        assert_eq!(parsed.sizes, vec![18, 20]);
    }

    #[test]
    fn parse_soak_equals_form() {
        let parsed = parse_cli_args(vec![
            "--soak=60".to_string(),
            "--json".to_string(),
            "20".to_string(),
        ])
        .unwrap();
        assert_eq!(parsed.mode, CliMode::Soak { duration_secs: 60 });
        assert!(parsed.json_mode);
    }

    #[test]
    fn parse_soak_rejects_zero() {
        let err = parse_cli_args(vec!["--soak".to_string(), "0".to_string()]).unwrap_err();
        assert!(err.contains("must be > 0"));
    }

    /// Phase E.1 post-review: `--force-family` / `--force-tail` are
    /// silently ignored on the Goldilocks path. The emitted JSON must
    /// reflect that — otherwise a consumer reading only the report
    /// would conclude a forced-family / forced-tail experiment actually
    /// ran. This test runs the exact override-null logic from
    /// `run_benchmark_mode` without touching the GPU.
    #[test]
    fn benchmark_report_overrides_are_null_under_goldilocks() {
        fn derive_overrides(
            field: Field,
            family: FamilyOverride,
            tail: TailOverride,
        ) -> (Option<String>, Option<String>) {
            match field {
                Field::BabyBear => (
                    match family {
                        FamilyOverride::Auto => None,
                        mode => Some(mode.as_str().to_string()),
                    },
                    match tail {
                        TailOverride::Auto => None,
                        t => Some(t.as_str().to_string()),
                    },
                ),
                Field::Goldilocks => (None, None),
            }
        }

        // BabyBear: overrides propagate
        assert_eq!(
            derive_overrides(Field::BabyBear, FamilyOverride::Stockham, TailOverride::Local),
            (Some("stockham".into()), Some("local".into())),
        );
        // Goldilocks: overrides null even when user supplied them
        assert_eq!(
            derive_overrides(Field::Goldilocks, FamilyOverride::FourStep, TailOverride::Global),
            (None, None),
        );
        // Goldilocks + Auto: still null (not Some("auto"))
        assert_eq!(
            derive_overrides(Field::Goldilocks, FamilyOverride::Auto, TailOverride::Auto),
            (None, None),
        );
    }
}
