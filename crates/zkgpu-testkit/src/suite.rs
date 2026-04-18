// Re-export from zkgpu-report for backward compatibility.
pub use zkgpu_report::{
    benchmark_suite, smoke_suite, validation_suite, FamilyOverride, Field, InputPattern,
    SuiteKind, SuiteSpec,
};

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
}
