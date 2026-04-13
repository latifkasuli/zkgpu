//! Shared report, spec, and harness types for zkgpu.
//!
//! This crate contains pure data types (no GPU dependencies) that are
//! shared between native runners (`zkgpu-testkit`, `zkgpu-ffi`, `zkgpu-cli`)
//! and browser runners (`zkgpu-web`, web harness). All types derive
//! `serde::Serialize` + `serde::Deserialize` for JSON interchange.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Enums shared between specs and reports
// ---------------------------------------------------------------------------

/// Direction of the NTT test case.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TestDirection {
    Forward,
    Inverse,
    Roundtrip,
}

/// Input data pattern for a test case.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum InputPattern {
    Sequential,
    AllZeros,
    AllOnes,
    LargeValuesDescending,
    PseudoRandomDeterministic { seed: u64 },
}

/// Which built-in suite to run.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SuiteKind {
    Smoke,
    Validation,
    Benchmark,
    /// Sustained-run soak benchmark: runs real NTT workloads for a fixed
    /// duration (30s, 60s, 120s) and records per-iteration timing samples
    /// instead of averaging. Used to characterize thermal behavior.
    Soak,
}

/// Override which NTT kernel family to use.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum FamilyOverride {
    Auto,
    Stockham,
    FourStep,
}

// ---------------------------------------------------------------------------
// Spec types (inputs to a test run)
// ---------------------------------------------------------------------------

/// Specification for a single test case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseSpec {
    pub name: String,
    pub log_n: u32,
    pub direction: TestDirection,
    pub input: InputPattern,
    #[serde(default)]
    pub profile_gpu_timestamps: bool,
    #[serde(default = "default_iterations")]
    pub iterations: u32,
    #[serde(default)]
    pub warmup_iterations: u32,
}

fn default_iterations() -> u32 {
    1
}

impl CaseSpec {
    pub fn new(
        name: impl Into<String>,
        log_n: u32,
        direction: TestDirection,
        input: InputPattern,
    ) -> Self {
        Self {
            name: name.into(),
            log_n,
            direction,
            input,
            profile_gpu_timestamps: false,
            iterations: 1,
            warmup_iterations: 0,
        }
    }

    pub fn with_profile(mut self, enabled: bool) -> Self {
        self.profile_gpu_timestamps = enabled;
        self
    }

    pub fn with_iterations(mut self, warmup_iterations: u32, iterations: u32) -> Self {
        self.warmup_iterations = warmup_iterations;
        self.iterations = iterations;
        self
    }
}

/// Specification for a full test suite.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuiteSpec {
    pub kind: SuiteKind,
    pub cases: Vec<CaseSpec>,
    pub fail_fast: bool,
    #[serde(default = "default_family_override")]
    pub family_override: FamilyOverride,
}

fn default_family_override() -> FamilyOverride {
    FamilyOverride::Auto
}

// ---------------------------------------------------------------------------
// Report types (outputs from a test run)
// ---------------------------------------------------------------------------

/// Device information included in reports.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceReport {
    pub name: String,
    pub backend: String,
    pub tier: String,
    pub gpu_family: String,
    /// How the GPU family was determined: "VendorId", "NameFallback",
    /// "MetalDefault", or "Unknown".
    #[serde(default)]
    pub detection_source: String,
    pub platform_class: String,
    pub memory_model: String,
    /// Raw driver name from the adapter (e.g. "qualcomm/adreno").
    #[serde(default)]
    pub driver: String,
    /// Driver version / info string from the adapter.
    #[serde(default)]
    pub driver_info: String,
    pub max_buffer_size_bytes: u64,
    pub max_workgroup_size_x: u32,
    pub max_compute_invocations: u32,
    pub feature_flags: Vec<String>,
}

/// Per-stage GPU timing measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageTimingReport {
    pub label: String,
    pub duration_ns: u64,
}

/// Timing information for a test case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingReport {
    pub wall_time_ns: Option<u64>,
    pub gpu_total_ns: Option<u64>,
    pub gpu_stage_ns: Vec<StageTimingReport>,
}

// ---------------------------------------------------------------------------
// Soak benchmark types — per-iteration telemetry for sustained runs
// ---------------------------------------------------------------------------

/// A single timing sample from one iteration of a soak run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoakSample {
    /// Iteration index (0-based).
    pub iteration: u32,
    /// Wall-clock time for this iteration in nanoseconds.
    pub wall_ns: u64,
    /// GPU total time for this iteration in nanoseconds (None if unprofiled).
    pub gpu_total_ns: Option<u64>,
    /// Elapsed time since the soak run started, in milliseconds.
    /// Used to plot timing drift over the run duration.
    pub elapsed_ms: u64,
}

/// Aggregate statistics computed from soak samples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoakStats {
    /// Total number of iterations completed.
    pub total_iterations: u32,
    /// Actual run duration in seconds (may slightly exceed the requested duration).
    pub actual_duration_secs: f64,
    /// Throughput: iterations per second.
    pub iterations_per_sec: f64,
    /// Median wall time per iteration in nanoseconds.
    pub median_wall_ns: u64,
    /// P5 wall time (5th percentile) in nanoseconds.
    pub p5_wall_ns: u64,
    /// P95 wall time (95th percentile) in nanoseconds.
    pub p95_wall_ns: u64,
    /// Min wall time in nanoseconds.
    pub min_wall_ns: u64,
    /// Max wall time in nanoseconds.
    pub max_wall_ns: u64,
    /// Coefficient of variation (std_dev / mean) as a ratio.
    /// Low values (~0.01-0.05) indicate stable thermals;
    /// high values (~0.10+) suggest throttling or contention.
    pub wall_cv: f64,
    /// Wall time of the last 10% of iterations divided by the first 10%.
    /// Values > 1.0 indicate thermal drift (later iterations slower).
    pub thermal_drift_ratio: f64,
    /// Same stats for GPU timestamps (None if unprofiled).
    pub median_gpu_ns: Option<u64>,
    pub p5_gpu_ns: Option<u64>,
    pub p95_gpu_ns: Option<u64>,
    pub gpu_cv: Option<f64>,
}

/// Report for a single soak case (one log_n + direction combo).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoakCaseReport {
    pub name: String,
    pub log_n: u32,
    pub direction: TestDirection,
    pub kernel_family: Option<String>,
    /// Requested soak duration in seconds.
    pub requested_duration_secs: u32,
    /// Aggregate statistics.
    pub stats: SoakStats,
    /// Per-iteration timing samples. Preserved in full for offline analysis.
    pub samples: Vec<SoakSample>,
    /// Whether the case was validated (first and last iteration outputs
    /// checked against CPU reference).
    pub validated: bool,
    /// Error message, if the soak run failed.
    pub error: Option<String>,
}

/// Complete soak suite report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoakSuiteReport {
    pub schema_version: u32,
    pub suite: SuiteKind,
    pub device: DeviceReport,
    pub kernel: KernelReport,
    pub cases: Vec<SoakCaseReport>,
    /// Requested duration for each case, in seconds.
    pub requested_duration_secs: u32,
}

/// Specification for a soak benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoakSpec {
    /// How long each case should run, in seconds.
    pub duration_secs: u32,
    /// Cases to soak. Each case runs for `duration_secs`.
    pub cases: Vec<CaseSpec>,
    /// Whether to validate first and last iteration against CPU reference.
    #[serde(default = "default_true")]
    pub validate: bool,
    #[serde(default = "default_family_override")]
    pub family_override: FamilyOverride,
}

fn default_true() -> bool {
    true
}

/// Report for a single test case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseReport {
    pub name: String,
    pub log_n: u32,
    pub direction: TestDirection,
    pub input: InputPattern,
    pub kernel_family: Option<String>,
    pub passed: bool,
    pub mismatch_count: u32,
    pub first_mismatch_index: Option<u32>,
    pub first_mismatch_gpu: Option<String>,
    pub first_mismatch_cpu: Option<String>,
    pub timings: TimingReport,
    pub error: Option<String>,
}

/// Summary counts for a test suite.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuiteSummary {
    pub total_cases: u32,
    pub passed_cases: u32,
    pub failed_cases: u32,
}

/// Kernel metadata included in reports.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelReport {
    pub field: String,
    pub ntt_variant: String,
}

/// Complete test suite report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuiteReport {
    pub schema_version: u32,
    pub suite: SuiteKind,
    pub device: DeviceReport,
    pub kernel: KernelReport,
    pub cases: Vec<CaseReport>,
    pub summary: SuiteSummary,
}

// ---------------------------------------------------------------------------
// Harness request/response (shared between FFI and web harness)
// ---------------------------------------------------------------------------

/// A request to the test harness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessRequest {
    pub suite: Option<SuiteKind>,
    pub spec: Option<SuiteSpec>,
    #[serde(default)]
    pub family_override: Option<FamilyOverride>,
}

/// Response from the test harness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessResponse {
    pub ok: bool,
    pub report: Option<SuiteReport>,
    pub error: Option<String>,
}

/// Version information for the harness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionResponse {
    pub crate_name: String,
    pub version: String,
    pub ffi_api_version: u32,
}

// ---------------------------------------------------------------------------
// Timing metadata (browser vs native differentiation)
// ---------------------------------------------------------------------------

/// Metadata about how timing values were collected.
///
/// Lets consumers distinguish native GPU timestamps from browser
/// wall-clock times and handle privacy-quantized browser timestamps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingMetadata {
    /// Clock source: "native-gpu", "browser-wall", "native-wall".
    pub clock_source: String,
    /// Whether browser timestamps may be privacy-quantized (e.g. 100us).
    #[serde(default)]
    pub timestamp_quantized: bool,
    /// Whether the test ran in a dedicated worker.
    #[serde(default)]
    pub worker: bool,
    /// Whether the test ran in a secure context.
    #[serde(default)]
    pub secure_context: bool,
    /// Browser user agent string (empty for native).
    #[serde(default)]
    pub user_agent: String,
    /// GPU adapter description from the browser/runtime.
    #[serde(default)]
    pub adapter_info: String,
}

impl Default for TimingMetadata {
    fn default() -> Self {
        Self {
            clock_source: "native-wall".to_string(),
            timestamp_quantized: false,
            worker: false,
            secure_context: false,
            user_agent: String::new(),
            adapter_info: String::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Built-in suite presets
// ---------------------------------------------------------------------------

/// Standard smoke suite: two quick cases for basic sanity.
pub fn smoke_suite() -> SuiteSpec {
    SuiteSpec {
        kind: SuiteKind::Smoke,
        cases: vec![
            CaseSpec::new(
                "forward_log10_sequential",
                10,
                TestDirection::Forward,
                InputPattern::Sequential,
            ),
            CaseSpec::new(
                "inverse_log10_sequential",
                10,
                TestDirection::Inverse,
                InputPattern::Sequential,
            ),
        ],
        fail_fast: true,
        family_override: FamilyOverride::Auto,
    }
}

/// Validation suite: broad coverage across sizes and input patterns.
pub fn validation_suite() -> SuiteSpec {
    SuiteSpec {
        kind: SuiteKind::Validation,
        cases: vec![
            CaseSpec::new(
                "forward_log4_sequential",
                4,
                TestDirection::Forward,
                InputPattern::Sequential,
            ),
            CaseSpec::new(
                "forward_log8_sequential",
                8,
                TestDirection::Forward,
                InputPattern::Sequential,
            ),
            CaseSpec::new(
                "forward_log10_sequential",
                10,
                TestDirection::Forward,
                InputPattern::Sequential,
            ),
            CaseSpec::new(
                "forward_log14_sequential",
                14,
                TestDirection::Forward,
                InputPattern::Sequential,
            ),
            CaseSpec::new(
                "forward_log10_all_zeros",
                10,
                TestDirection::Forward,
                InputPattern::AllZeros,
            ),
            CaseSpec::new(
                "forward_log8_all_ones",
                8,
                TestDirection::Forward,
                InputPattern::AllOnes,
            ),
            CaseSpec::new(
                "forward_log10_large_values",
                10,
                TestDirection::Forward,
                InputPattern::LargeValuesDescending,
            ),
            CaseSpec::new(
                "forward_log10_pseudorandom",
                10,
                TestDirection::Forward,
                InputPattern::PseudoRandomDeterministic { seed: 1 },
            ),
            CaseSpec::new(
                "inverse_log4_sequential",
                4,
                TestDirection::Inverse,
                InputPattern::Sequential,
            ),
            CaseSpec::new(
                "inverse_log10_sequential",
                10,
                TestDirection::Inverse,
                InputPattern::Sequential,
            ),
            CaseSpec::new(
                "roundtrip_log8_sequential",
                8,
                TestDirection::Roundtrip,
                InputPattern::Sequential,
            ),
            CaseSpec::new(
                "roundtrip_log12_sequential",
                12,
                TestDirection::Roundtrip,
                InputPattern::Sequential,
            ),
        ],
        fail_fast: false,
        family_override: FamilyOverride::Auto,
    }
}

/// Benchmark suite: a few sizes with profiling enabled.
pub fn benchmark_suite() -> SuiteSpec {
    let benchmark_case = |name: &str, log_n: u32| {
        CaseSpec::new(
            name,
            log_n,
            TestDirection::Forward,
            InputPattern::Sequential,
        )
        .with_profile(true)
        .with_iterations(1, 5)
    };

    SuiteSpec {
        kind: SuiteKind::Benchmark,
        cases: vec![
            benchmark_case("benchmark_forward_log10", 10),
            benchmark_case("benchmark_forward_log14", 14),
            benchmark_case("benchmark_forward_log18", 18),
            benchmark_case("benchmark_forward_log20", 20),
        ],
        fail_fast: false,
        family_override: FamilyOverride::Auto,
    }
}

// ---------------------------------------------------------------------------
// Built-in soak suite presets
// ---------------------------------------------------------------------------

/// Short soak (30 seconds per case): quick thermal characterization.
pub fn soak_suite_30s() -> SoakSpec {
    soak_spec(30)
}

/// Medium soak (60 seconds per case): standard sustained-run test.
pub fn soak_suite_60s() -> SoakSpec {
    soak_spec(60)
}

/// Long soak (120 seconds per case): full thermal characterization.
pub fn soak_suite_120s() -> SoakSpec {
    soak_spec(120)
}

fn soak_spec(duration_secs: u32) -> SoakSpec {
    let soak_case = |name: &str, log_n: u32, direction: TestDirection| {
        CaseSpec::new(name, log_n, direction, InputPattern::Sequential)
            .with_profile(true)
    };

    SoakSpec {
        duration_secs,
        cases: vec![
            // Representative sizes that exercise both kernel families
            soak_case("soak_forward_log14", 14, TestDirection::Forward),
            soak_case("soak_forward_log18", 18, TestDirection::Forward),
            soak_case("soak_forward_log20", 20, TestDirection::Forward),
            soak_case("soak_inverse_log20", 20, TestDirection::Inverse),
        ],
        validate: true,
        family_override: FamilyOverride::Auto,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_suite_has_two_cases() {
        let suite = smoke_suite();
        assert_eq!(suite.kind, SuiteKind::Smoke);
        assert_eq!(suite.cases.len(), 2);
        assert!(suite.fail_fast);
    }

    #[test]
    fn benchmark_suite_enables_profiling() {
        let suite = benchmark_suite();
        assert!(suite
            .cases
            .iter()
            .all(|c| c.profile_gpu_timestamps && c.iterations == 5 && c.warmup_iterations == 1));
    }

    #[test]
    fn harness_request_roundtrips_json() {
        let req = HarnessRequest {
            suite: Some(SuiteKind::Smoke),
            spec: None,
            family_override: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: HarnessRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.suite, Some(SuiteKind::Smoke));
    }

    #[test]
    fn timing_metadata_defaults() {
        let meta = TimingMetadata::default();
        assert_eq!(meta.clock_source, "native-wall");
        assert!(!meta.timestamp_quantized);
        assert!(!meta.worker);
    }

    #[test]
    fn soak_suite_30s_has_four_cases() {
        let spec = soak_suite_30s();
        assert_eq!(spec.duration_secs, 30);
        assert_eq!(spec.cases.len(), 4);
        assert!(spec.validate);
        assert!(spec.cases.iter().all(|c| c.profile_gpu_timestamps));
    }

    #[test]
    fn soak_suite_120s_has_correct_duration() {
        let spec = soak_suite_120s();
        assert_eq!(spec.duration_secs, 120);
        assert_eq!(spec.cases.len(), 4);
    }

    #[test]
    fn soak_spec_roundtrips_json() {
        let spec = soak_suite_60s();
        let json = serde_json::to_string(&spec).unwrap();
        let parsed: SoakSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.duration_secs, 60);
        assert_eq!(parsed.cases.len(), 4);
        assert!(parsed.validate);
    }

    #[test]
    fn soak_sample_roundtrips_json() {
        let sample = SoakSample {
            iteration: 42,
            wall_ns: 5_000_000,
            gpu_total_ns: Some(3_200_000),
            elapsed_ms: 12_500,
        };
        let json = serde_json::to_string(&sample).unwrap();
        let parsed: SoakSample = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.iteration, 42);
        assert_eq!(parsed.gpu_total_ns, Some(3_200_000));
    }
}
