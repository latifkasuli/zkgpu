/// Tuning knobs for the drift pacer.
#[derive(Debug, Clone)]
pub struct DriftPacerConfig {
    /// Number of initial batches to collect for baseline (per OpKey).
    pub baseline_count: u32,
    /// Wall-time ratio threshold above which a batch is "hot".
    /// E.g. 1.20 means 20% above baseline.
    pub wall_drift_threshold: f64,
    /// GPU-time ratio threshold (used when GPU timestamps are available).
    /// E.g. 1.15 means 15% above baseline.
    pub gpu_drift_threshold: f64,
    /// Number of consecutive hot batches before pacing kicks in.
    pub hot_trigger_count: u32,
    /// Number of consecutive stable batches before pacing backs off.
    pub recovery_count: u32,
    /// Ratio below which a batch is considered stable for recovery.
    /// E.g. 1.08 means within 8% of baseline.
    pub recovery_threshold: f64,
    /// Idle step size when ramping up, in milliseconds.
    pub idle_step_ms: u64,
    /// Maximum idle duration, in milliseconds.
    pub max_idle_ms: u64,
}

impl Default for DriftPacerConfig {
    fn default() -> Self {
        Self {
            baseline_count: 8,
            wall_drift_threshold: 1.20,
            gpu_drift_threshold: 1.15,
            hot_trigger_count: 3,
            recovery_count: 10,
            recovery_threshold: 1.08,
            idle_step_ms: 3,
            max_idle_ms: 25,
        }
    }
}
