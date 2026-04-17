use std::time::Instant;

use serde::Serialize;
use zkgpu_babybear::BabyBear;
use zkgpu_core::{GpuBuffer, GpuDevice, NttDirection, NttPlan};
use zkgpu_ntt::ntt_cpu_reference;
use zkgpu_wgpu::{CapabilityProfile, PlannerPolicy, WgpuDevice, WgpuNttPlan};

#[derive(Serialize)]
struct BenchmarkReport {
    device: DeviceReport,
    family_override: Option<String>,
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CliArgs {
    json_mode: bool,
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

fn usage() -> &'static str {
    "Usage: zkgpu [--json] [--force-family auto|stockham|four-step] \\
               [--force-tail auto|local|global] [--soak SECS] [log_n...]

Modes:
  (default)      Short benchmark: warmup + 5 measured iterations per size
  --soak SECS    Sustained soak: run each size for SECS seconds, recording
                 per-iteration samples for thermal characterization

Overrides:
  --force-family   Pin the NTT family (default: auto = device-policy pick).
                   'stockham' disables four-step; 'four-step' forces it.
  --force-tail     Pin the Stockham tail strategy (default: auto).
                   'local'  -> LocalFusedR4 (legacy tail kernel)
                   'global' -> GlobalOnlyR4 (extends global R4 chain through
                               the tail; the configuration PR 1 made possible).
                   Only applies when the plan is Stockham. A Four-Step plan
                   has no Stockham tail and silently ignores this flag.

Examples:
  zkgpu
  zkgpu 10 20
  zkgpu --force-family four-step 20 24
  zkgpu --json --force-family stockham 18 20
  zkgpu --force-family stockham --force-tail local --json 18 19 20 21 22
  zkgpu --force-family stockham --force-tail global --json 18 19 20 21 22
  zkgpu --soak 30 18 20
  zkgpu --soak 60 --json 20"
}

fn parse_cli_args<I>(args: I) -> Result<CliArgs, String>
where
    I: IntoIterator<Item = String>,
{
    let mut json_mode = false;
    let mut family_override = FamilyOverride::Auto;
    let mut tail_override = TailOverride::Auto;
    let mut sizes = Vec::new();
    let mut soak_secs: Option<u32> = None;

    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--json" => json_mode = true,
            "--help" | "-h" => return Err(usage().to_string()),
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
            _ if arg.starts_with('-') => {
                return Err(format!("unrecognized option '{arg}'"));
            }
            _ => {
                let log_n = arg
                    .parse::<u32>()
                    .map_err(|_| format!("invalid log_n '{arg}'"))?;
                sizes.push(log_n);
            }
        }
    }

    if sizes.is_empty() {
        sizes = DEFAULT_SIZES.to_vec();
    }

    let mode = match soak_secs {
        Some(secs) => CliMode::Soak { duration_secs: secs },
        None => CliMode::Benchmark,
    };

    Ok(CliArgs {
        json_mode,
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
    eprintln!("Family override: {}", cli.family_override.as_str());
    eprintln!("Tail override: {}", cli.tail_override.as_str());

    match cli.mode {
        CliMode::Benchmark => run_benchmark_mode(&device, &cli),
        CliMode::Soak { duration_secs } => {
            // Soak mode uses the canonical testkit runner which creates its
            // own device, so we drop the one we created above to avoid
            // holding two GPU device handles.
            drop(device);
            run_soak_mode(&cli, duration_secs);
        }
    }
}

fn run_benchmark_mode(device: &WgpuDevice, cli: &CliArgs) {
    let mut runs = Vec::new();

    for &log_n in &cli.sizes {
        eprintln!("\n--- NTT 2^{log_n} ---");
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

    let report = BenchmarkReport {
        device: device_report(device.caps()),
        family_override: match cli.family_override {
            FamilyOverride::Auto => None,
            mode => Some(mode.as_str().to_string()),
        },
        tail_override: match cli.tail_override {
            TailOverride::Auto => None,
            tail => Some(tail.as_str().to_string()),
        },
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
        assert_eq!(parsed.family_override, FamilyOverride::Auto);
        assert_eq!(parsed.sizes, DEFAULT_SIZES);
        assert_eq!(parsed.mode, CliMode::Benchmark);
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
}
