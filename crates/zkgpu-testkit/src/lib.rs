mod benchmark;
mod case;
mod device;
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
    benchmark_suite, smoke_suite, validation_suite, FamilyOverride, InputPattern, SuiteKind,
    SuiteSpec,
};
pub use benchmark::compute_soak_stats;
pub use zkgpu_report::{
    SoakCaseReport, SoakSample, SoakSpec, SoakStats, SoakSuiteReport,
    soak_suite_30s, soak_suite_60s, soak_suite_120s,
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
