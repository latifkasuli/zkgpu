mod benchmark;
mod case;
mod device;
mod hash_runner;
mod inputs;
mod report;
mod runner;
mod suite;
mod validation;

use thiserror::Error;

pub use case::{CaseSpec, TestDirection};
pub use report::{
    CaseReport, DeviceReport, KernelReport, StageTimingReport, SuiteReport, SuiteSummary,
    TimingReport,
};
pub use runner::{run_benchmark_suite, run_smoke_suite, run_soak_suite, run_suite, run_validation_suite};
pub use suite::{
    benchmark_suite, smoke_suite, validation_suite, FamilyOverride, Field, InputPattern,
    SuiteKind, SuiteSpec,
};
pub use benchmark::compute_soak_stats;
pub use zkgpu_report::{
    SoakCaseReport, SoakSample, SoakSpec, SoakStats, SoakSuiteReport,
    soak_suite_30s, soak_suite_60s, soak_suite_120s,
};

// Phase F.3.a: re-export the hash spec/report surface so consumers
// that already depend on zkgpu-testkit can build Poseidon2 suites
// without adding a direct zkgpu-report dep. The runner-side wiring
// (run_hash_suite, measure_poseidon2_plan) lands in Phase F.3.b —
// this re-export ships the type surface so the CLI / web entry
// points can start accepting HashSpec JSON in parallel.
pub use zkgpu_report::{
    poseidon2_benchmark_suite, poseidon2_smoke_suite, HashAlgorithm, HashCaseReport,
    HashCaseSpec, HashInputPattern, HashSpec, HashSuiteReport,
};
pub use hash_runner::{
    run_hash_suite, run_poseidon2_benchmark_suite, run_poseidon2_smoke_suite,
};

#[derive(Debug, Error)]
pub enum TestkitError {
    #[error("GPU backend error: {0}")]
    Backend(#[from] zkgpu_core::ZkGpuError),
    #[error("suite must contain at least one case")]
    EmptySuite,
    #[error("failed to create GPU device: {0}")]
    DeviceInit(String),
}
