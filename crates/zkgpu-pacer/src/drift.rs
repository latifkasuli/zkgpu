//! Adaptive drift-based pacer.
//!
//! Baselines on the first N homogeneous batches per [`OpKey`], then
//! triggers idle gaps after sustained timing drift. Recovers gradually
//! when drift subsides.

use std::collections::HashMap;
use std::time::Duration;

use crate::observation::{
    ExecutionObservation, OpKey, PaceDecision, PaceLevel, PaceReason, ThermalHint,
    ThermalHintSource, ThermalSeverity,
};

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

/// Per-OpKey tracking state.
#[derive(Debug)]
struct OpState {
    /// Baseline samples collected so far (wall_ns).
    baseline_wall: Vec<u64>,
    /// Baseline samples collected so far (gpu_ns).
    baseline_gpu: Vec<u64>,
    /// Computed baseline wall time (median of best samples). None during warmup.
    baseline_wall_ns: Option<u64>,
    /// Computed baseline GPU time. None if unprofiled or during warmup.
    baseline_gpu_ns: Option<u64>,
    /// Consecutive batches above drift threshold.
    consecutive_hot: u32,
    /// Consecutive batches below recovery threshold (while pacing).
    consecutive_stable: u32,
    /// Current idle duration being applied.
    current_idle: Duration,
}

impl OpState {
    fn new() -> Self {
        Self {
            baseline_wall: Vec::with_capacity(16),
            baseline_gpu: Vec::with_capacity(16),
            baseline_wall_ns: None,
            baseline_gpu_ns: None,
            consecutive_hot: 0,
            consecutive_stable: 0,
            current_idle: Duration::ZERO,
        }
    }

    /// Compute baseline from collected samples.
    ///
    /// Uses median of the best half of samples (lower half of sorted
    /// values) to resist outliers from cold starts or GC pauses.
    fn compute_baselines(&mut self) {
        self.baseline_wall_ns = Some(best_half_median(&self.baseline_wall));
        if !self.baseline_gpu.is_empty() {
            self.baseline_gpu_ns = Some(best_half_median(&self.baseline_gpu));
        }
    }
}

/// Median of the lower half of values — resistant to high outliers.
fn best_half_median(samples: &[u64]) -> u64 {
    if samples.is_empty() {
        return 0;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let half = (sorted.len() / 2).max(1);
    let best = &sorted[..half];
    best[best.len() / 2]
}

/// Cross-platform adaptive pacer driven by timing drift.
///
/// # Usage
///
/// ```ignore
/// let mut pacer = DriftPacer::new(DriftPacerConfig::default());
///
/// loop {
///     let start = Instant::now();
///     execute_ntt(&plan);
///     let wall_ns = start.elapsed().as_nanos() as u64;
///
///     let decision = pacer.observe(&ExecutionObservation {
///         op_key: key.clone(),
///         observed_wall_ns: wall_ns,
///         gpu_total_ns: None,
///         ..
///     });
///
///     if !decision.idle_for.is_zero() {
///         std::thread::sleep(decision.idle_for);
///     }
/// }
/// ```
pub struct DriftPacer {
    config: DriftPacerConfig,
    states: HashMap<OpKey, OpState>,
    thermal_source: Option<Box<dyn ThermalHintSource>>,
    last_thermal: Option<ThermalHint>,
    /// Nanoseconds accumulated since the last thermal poll.
    /// Initialized to `thermal_poll_interval_ns` so the first
    /// `observe()` call triggers an immediate poll.
    ns_since_thermal_poll: u64,
    /// Cached `min_poll_interval()` from the thermal source, in
    /// nanoseconds. Zero when no source is attached.
    thermal_poll_interval_ns: u64,
    /// Idle duration (ns) recommended by the previous `observe()` call.
    /// Added to the cadence accumulator on the next call so that the
    /// thermal poll tracks real elapsed time, not just batch execution
    /// time. Without this, once the pacer starts inserting sleeps the
    /// cadence undercounts and stale thermal overrides persist.
    last_recommended_idle_ns: u64,
}

impl DriftPacer {
    /// Create a new pacer with the given configuration.
    pub fn new(config: DriftPacerConfig) -> Self {
        Self {
            config,
            states: HashMap::new(),
            thermal_source: None,
            last_thermal: None,
            ns_since_thermal_poll: 0,
            thermal_poll_interval_ns: 0,
            last_recommended_idle_ns: 0,
        }
    }

    /// Create a pacer with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(DriftPacerConfig::default())
    }

    /// Attach a platform thermal hint source.
    ///
    /// The pacer respects `source.min_poll_interval()` by accumulating
    /// estimated real elapsed time (batch wall time + recommended idle)
    /// and only calling `poll()` once enough time has passed. The first
    /// `observe()` always polls.
    pub fn with_thermal_source(mut self, source: Box<dyn ThermalHintSource>) -> Self {
        self.thermal_poll_interval_ns = source.min_poll_interval().as_nanos() as u64;
        // Seed at the interval value so the first observe() immediately
        // crosses the threshold and triggers the initial poll.
        self.ns_since_thermal_poll = self.thermal_poll_interval_ns;
        self.thermal_source = Some(source);
        self
    }

    /// Process one execution observation and return a pacing decision.
    ///
    /// Call this after each NTT batch completes.
    pub fn observe(&mut self, obs: &ExecutionObservation) -> PaceDecision {
        let decision = self.decide(obs);
        // Record the idle we recommended so the *next* call's cadence
        // accumulator accounts for the sleep the caller was told to apply.
        self.last_recommended_idle_ns = decision.idle_for.as_nanos() as u64;
        decision
    }

    /// Internal: poll thermal source (rate-limited), evaluate drift,
    /// and produce a pacing decision.
    fn decide(&mut self, obs: &ExecutionObservation) -> PaceDecision {
        // Poll thermal hint if available and enough time has elapsed.
        //
        // The cadence accumulates *estimated real elapsed time* between
        // observe() calls: batch wall time + the idle the caller was
        // told to apply after the previous observation. This avoids two
        // failure modes:
        //   1. Over-polling coarse APIs like Android ADPF (10 s cadence).
        //   2. Under-counting once pacing inserts sleeps, which would
        //      keep stale Critical/Serious overrides stuck indefinitely.
        if let Some(ref mut source) = self.thermal_source {
            self.ns_since_thermal_poll = self
                .ns_since_thermal_poll
                .saturating_add(obs.observed_wall_ns)
                .saturating_add(self.last_recommended_idle_ns);
            if self.ns_since_thermal_poll >= self.thermal_poll_interval_ns {
                self.ns_since_thermal_poll = 0;
                if let Some(hint) = source.poll() {
                    self.last_thermal = Some(hint);
                }
                // Re-read in case the source adjusted its interval
                // (e.g., ADPF headroom degraded to status-only mode).
                self.thermal_poll_interval_ns = source.min_poll_interval().as_nanos() as u64;
            }
        }

        let state = self
            .states
            .entry(obs.op_key.clone())
            .or_insert_with(OpState::new);

        // --- Phase 1: Baseline collection ---
        if state.baseline_wall_ns.is_none() {
            state.baseline_wall.push(obs.observed_wall_ns);
            if let Some(gpu) = obs.gpu_total_ns {
                state.baseline_gpu.push(gpu);
            }

            if state.baseline_wall.len() >= self.config.baseline_count as usize {
                state.compute_baselines();
            } else {
                return PaceDecision::baseline();
            }
        }

        let baseline_wall = state.baseline_wall_ns.unwrap();
        let wall_ratio = if baseline_wall > 0 {
            obs.observed_wall_ns as f64 / baseline_wall as f64
        } else {
            1.0
        };

        // GPU ratio (secondary signal)
        let gpu_ratio = match (obs.gpu_total_ns, state.baseline_gpu_ns) {
            (Some(gpu), Some(base)) if base > 0 => Some(gpu as f64 / base as f64),
            _ => None,
        };

        // --- Check thermal hints for override ---
        if let Some(ref hint) = self.last_thermal {
            if hint.severity == ThermalSeverity::Critical {
                let idle = Duration::from_millis(self.config.max_idle_ms);
                state.current_idle = idle;
                return PaceDecision {
                    idle_for: idle,
                    level: PaceLevel::Heavy,
                    reason: PaceReason::ThermalHint {
                        severity: ThermalSeverity::Critical,
                    },
                };
            }
            if hint.severity == ThermalSeverity::Serious {
                let idle = Duration::from_millis(self.config.max_idle_ms * 2 / 3);
                state.current_idle = idle;
                return PaceDecision {
                    idle_for: idle,
                    level: PaceLevel::Heavy,
                    reason: PaceReason::ThermalHint {
                        severity: ThermalSeverity::Serious,
                    },
                };
            }
        }

        // --- Phase 2: Drift detection ---
        let is_hot_wall = wall_ratio >= self.config.wall_drift_threshold;
        let is_hot_gpu = gpu_ratio
            .map(|r| r >= self.config.gpu_drift_threshold)
            .unwrap_or(false);
        let is_hot = is_hot_gpu || is_hot_wall;

        let is_stable = wall_ratio < self.config.recovery_threshold
            && gpu_ratio
                .map(|r| r < self.config.recovery_threshold)
                .unwrap_or(true);

        if is_hot {
            state.consecutive_hot += 1;
            state.consecutive_stable = 0;
        } else if is_stable {
            state.consecutive_stable += 1;
            state.consecutive_hot = 0;
        } else {
            // Between thresholds — neither hot nor stable
            state.consecutive_hot = 0;
            // Don't reset stable count — allow gradual recovery
        }

        // --- Phase 3: Decision ---

        // Trigger: sustained drift
        if state.consecutive_hot >= self.config.hot_trigger_count {
            let step = Duration::from_millis(self.config.idle_step_ms);
            let max = Duration::from_millis(self.config.max_idle_ms);
            state.current_idle = (state.current_idle + step).min(max);
            state.consecutive_stable = 0;

            let level = if state.current_idle >= max {
                PaceLevel::Heavy
            } else if state.current_idle >= Duration::from_millis(self.config.max_idle_ms / 2) {
                PaceLevel::Moderate
            } else {
                PaceLevel::Light
            };

            let reason = if is_hot_gpu {
                PaceReason::GpuDrift {
                    consecutive_hot: state.consecutive_hot,
                    gpu_ratio: gpu_ratio.unwrap_or(0.0),
                }
            } else {
                PaceReason::WallDrift {
                    consecutive_hot: state.consecutive_hot,
                    wall_ratio,
                }
            };

            return PaceDecision {
                idle_for: state.current_idle,
                level,
                reason,
            };
        }

        // Recovery: sustained stability while pacing
        if !state.current_idle.is_zero()
            && state.consecutive_stable >= self.config.recovery_count
        {
            let step = Duration::from_millis(self.config.idle_step_ms);
            state.current_idle = state.current_idle.saturating_sub(step);
            state.consecutive_stable = 0;

            if state.current_idle.is_zero() {
                return PaceDecision::none();
            }

            return PaceDecision {
                idle_for: state.current_idle,
                level: PaceLevel::Light,
                reason: PaceReason::Recovery {
                    stable_count: self.config.recovery_count,
                },
            };
        }

        // Hold current idle if pacing but not yet recovered
        if !state.current_idle.is_zero() {
            let level = if state.current_idle >= Duration::from_millis(self.config.max_idle_ms) {
                PaceLevel::Heavy
            } else if state.current_idle >= Duration::from_millis(self.config.max_idle_ms / 2) {
                PaceLevel::Moderate
            } else {
                PaceLevel::Light
            };

            return PaceDecision {
                idle_for: state.current_idle,
                level,
                reason: PaceReason::Stable,
            };
        }

        PaceDecision::none()
    }

    /// Reset all state. Useful for benchmark mode transitions.
    pub fn reset(&mut self) {
        self.states.clear();
        self.last_thermal = None;
        self.last_recommended_idle_ns = 0;
        // Re-seed so the next observe() triggers an immediate poll.
        self.ns_since_thermal_poll = self.thermal_poll_interval_ns;
    }

    /// Reset state for a specific OpKey.
    pub fn reset_key(&mut self, key: &OpKey) {
        self.states.remove(key);
    }

    /// Get the current configuration.
    pub fn config(&self) -> &DriftPacerConfig {
        &self.config
    }

    /// Number of OpKeys currently being tracked.
    pub fn tracked_keys(&self) -> usize {
        self.states.len()
    }
}

impl std::fmt::Debug for DriftPacer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DriftPacer")
            .field("config", &self.config)
            .field("tracked_keys", &self.states.len())
            .field("has_thermal_source", &self.thermal_source.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn test_key() -> OpKey {
        OpKey {
            backend: "Vulkan".into(),
            platform_class: "AndroidNative".into(),
            gpu_family: "Adreno".into(),
            kernel_family: "stockham".into(),
            log_n: 20,
            direction: "Forward".into(),
        }
    }

    fn obs(key: &OpKey, wall_ns: u64) -> ExecutionObservation {
        ExecutionObservation {
            op_key: key.clone(),
            observed_wall_ns: wall_ns,
            gpu_total_ns: None,
            gpu_stage_ns: Vec::new(),
            bytes_moved: None,
            batch_size: 1,
        }
    }

    fn obs_with_gpu(key: &OpKey, wall_ns: u64, gpu_ns: u64) -> ExecutionObservation {
        ExecutionObservation {
            op_key: key.clone(),
            observed_wall_ns: wall_ns,
            gpu_total_ns: Some(gpu_ns),
            gpu_stage_ns: Vec::new(),
            bytes_moved: None,
            batch_size: 1,
        }
    }

    // === Baseline phase ===

    #[test]
    fn returns_baseline_during_warmup() {
        let mut pacer = DriftPacer::with_defaults();
        let key = test_key();

        for _ in 0..7 {
            let d = pacer.observe(&obs(&key, 10_000_000));
            assert_eq!(d.level, PaceLevel::None);
            assert!(matches!(d.reason, PaceReason::Baseline));
        }
    }

    #[test]
    fn completes_baseline_after_n_samples() {
        let config = DriftPacerConfig {
            baseline_count: 4,
            ..Default::default()
        };
        let mut pacer = DriftPacer::new(config);
        let key = test_key();

        for _ in 0..3 {
            let d = pacer.observe(&obs(&key, 10_000_000));
            assert!(matches!(d.reason, PaceReason::Baseline));
        }

        // 4th observation completes baseline and evaluates drift
        let d = pacer.observe(&obs(&key, 10_000_000));
        assert!(matches!(d.reason, PaceReason::Stable));
    }

    // === Stable operation ===

    #[test]
    fn no_pacing_when_stable() {
        let config = DriftPacerConfig {
            baseline_count: 4,
            ..Default::default()
        };
        let mut pacer = DriftPacer::new(config);
        let key = test_key();

        // Baseline
        for _ in 0..4 {
            pacer.observe(&obs(&key, 10_000_000));
        }

        // Stable iterations
        for _ in 0..20 {
            let d = pacer.observe(&obs(&key, 10_500_000)); // 5% above baseline
            assert_eq!(d.idle_for, Duration::ZERO);
            assert_eq!(d.level, PaceLevel::None);
        }
    }

    // === Drift detection ===

    #[test]
    fn triggers_pacing_after_sustained_drift() {
        let config = DriftPacerConfig {
            baseline_count: 4,
            wall_drift_threshold: 1.20,
            hot_trigger_count: 3,
            idle_step_ms: 5,
            ..Default::default()
        };
        let mut pacer = DriftPacer::new(config);
        let key = test_key();

        // Baseline at 10ms
        for _ in 0..4 {
            pacer.observe(&obs(&key, 10_000_000));
        }

        // 2 hot batches — not enough to trigger
        for _ in 0..2 {
            let d = pacer.observe(&obs(&key, 13_000_000)); // 30% above
            assert_eq!(d.idle_for, Duration::ZERO);
        }

        // 3rd hot batch — triggers pacing
        let d = pacer.observe(&obs(&key, 13_000_000));
        assert!(!d.idle_for.is_zero());
        assert_eq!(d.idle_for, Duration::from_millis(5));
        assert!(matches!(d.reason, PaceReason::WallDrift { .. }));
    }

    #[test]
    fn idle_ramps_up_with_continued_drift() {
        let config = DriftPacerConfig {
            baseline_count: 4,
            wall_drift_threshold: 1.20,
            hot_trigger_count: 3,
            idle_step_ms: 3,
            max_idle_ms: 15,
            ..Default::default()
        };
        let mut pacer = DriftPacer::new(config);
        let key = test_key();

        // Baseline at 10ms
        for _ in 0..4 {
            pacer.observe(&obs(&key, 10_000_000));
        }

        // Sustained drift
        let mut last_idle = Duration::ZERO;
        for i in 0..20 {
            let d = pacer.observe(&obs(&key, 15_000_000)); // 50% above
            if i >= 2 {
                // After 3rd hot batch, should be pacing
                assert!(d.idle_for >= last_idle || d.idle_for == Duration::from_millis(15));
                last_idle = d.idle_for;
            }
        }

        // Should have hit the cap
        assert_eq!(last_idle, Duration::from_millis(15));
    }

    #[test]
    fn intermittent_spikes_do_not_trigger_pacing() {
        let config = DriftPacerConfig {
            baseline_count: 4,
            wall_drift_threshold: 1.20,
            hot_trigger_count: 3,
            ..Default::default()
        };
        let mut pacer = DriftPacer::new(config);
        let key = test_key();

        // Baseline at 10ms
        for _ in 0..4 {
            pacer.observe(&obs(&key, 10_000_000));
        }

        // Alternating hot and normal — never 3 consecutive
        for _ in 0..20 {
            pacer.observe(&obs(&key, 15_000_000)); // hot
            pacer.observe(&obs(&key, 10_000_000)); // normal
        }

        // Should never have triggered
        let d = pacer.observe(&obs(&key, 10_000_000));
        assert_eq!(d.idle_for, Duration::ZERO);
    }

    // === Recovery ===

    #[test]
    fn recovers_after_sustained_stability() {
        let config = DriftPacerConfig {
            baseline_count: 4,
            wall_drift_threshold: 1.20,
            hot_trigger_count: 3,
            recovery_count: 5,
            recovery_threshold: 1.08,
            idle_step_ms: 5,
            max_idle_ms: 25,
            ..Default::default()
        };
        let mut pacer = DriftPacer::new(config);
        let key = test_key();

        // Baseline at 10ms
        for _ in 0..4 {
            pacer.observe(&obs(&key, 10_000_000));
        }

        // Drive into pacing
        for _ in 0..6 {
            pacer.observe(&obs(&key, 15_000_000)); // 50% hot
        }
        let d = pacer.observe(&obs(&key, 15_000_000));
        assert!(!d.idle_for.is_zero(), "should be pacing");
        let pacing_idle = d.idle_for;

        // Recover with stable batches
        for _ in 0..5 {
            pacer.observe(&obs(&key, 10_500_000)); // 5% — well below 1.08 threshold
        }
        let d = pacer.observe(&obs(&key, 10_500_000));

        // Should have reduced idle
        assert!(
            d.idle_for < pacing_idle || d.idle_for.is_zero(),
            "idle should decrease during recovery: was {:?}, now {:?}",
            pacing_idle,
            d.idle_for,
        );
    }

    // === GPU drift signal ===

    #[test]
    fn gpu_drift_triggers_pacing_when_wall_is_stable() {
        let config = DriftPacerConfig {
            baseline_count: 4,
            wall_drift_threshold: 1.20,
            gpu_drift_threshold: 1.15,
            hot_trigger_count: 3,
            idle_step_ms: 5,
            ..Default::default()
        };
        let mut pacer = DriftPacer::new(config);
        let key = test_key();

        // Baseline with GPU
        for _ in 0..4 {
            pacer.observe(&obs_with_gpu(&key, 10_000_000, 8_000_000));
        }

        // Wall is stable but GPU is drifting
        for _ in 0..3 {
            pacer.observe(&obs_with_gpu(&key, 10_500_000, 10_000_000)); // GPU 25% above
        }

        let d = pacer.observe(&obs_with_gpu(&key, 10_500_000, 10_000_000));
        assert!(
            !d.idle_for.is_zero(),
            "GPU drift should trigger pacing even when wall is stable"
        );
        assert!(matches!(d.reason, PaceReason::GpuDrift { .. }));
    }

    // === Multiple OpKeys ===

    #[test]
    fn tracks_multiple_op_keys_independently() {
        let config = DriftPacerConfig {
            baseline_count: 4,
            hot_trigger_count: 3,
            ..Default::default()
        };
        let mut pacer = DriftPacer::new(config);

        let key_a = OpKey {
            log_n: 20,
            ..test_key()
        };
        let key_b = OpKey {
            log_n: 14,
            ..test_key()
        };

        // Baseline both
        for _ in 0..4 {
            pacer.observe(&obs(&key_a, 10_000_000));
            pacer.observe(&obs(&key_b, 2_000_000));
        }

        // Drive key_a into pacing
        for _ in 0..5 {
            pacer.observe(&obs(&key_a, 15_000_000));
        }

        // key_a should be pacing
        let d_a = pacer.observe(&obs(&key_a, 15_000_000));
        assert!(!d_a.idle_for.is_zero());

        // key_b should still be stable
        let d_b = pacer.observe(&obs(&key_b, 2_100_000));
        assert_eq!(d_b.idle_for, Duration::ZERO);

        assert_eq!(pacer.tracked_keys(), 2);
    }

    // === Thermal hints ===

    struct MockThermal {
        hints: Vec<Option<ThermalHint>>,
        idx: usize,
    }

    impl ThermalHintSource for MockThermal {
        fn poll(&mut self) -> Option<ThermalHint> {
            if self.idx < self.hints.len() {
                let h = self.hints[self.idx].clone();
                self.idx += 1;
                h
            } else {
                None
            }
        }
        fn min_poll_interval(&self) -> Duration {
            // Zero so every observe() triggers a poll — lets the test
            // script control exactly which observation sees which hint.
            Duration::ZERO
        }
    }

    #[test]
    fn critical_thermal_hint_overrides_drift() {
        let config = DriftPacerConfig {
            baseline_count: 4,
            max_idle_ms: 25,
            ..Default::default()
        };
        let mut pacer = DriftPacer::new(config).with_thermal_source(Box::new(MockThermal {
            hints: vec![
                None,
                None,
                None,
                None,
                Some(ThermalHint {
                    severity: ThermalSeverity::Critical,
                    headroom: Some(0.05),
                }),
            ],
            idx: 0,
        }));
        let key = test_key();

        // Baseline
        for _ in 0..4 {
            pacer.observe(&obs(&key, 10_000_000));
        }

        // 5th observation triggers the Critical hint
        let d = pacer.observe(&obs(&key, 10_000_000));
        assert_eq!(d.idle_for, Duration::from_millis(25));
        assert_eq!(d.level, PaceLevel::Heavy);
        assert!(matches!(
            d.reason,
            PaceReason::ThermalHint {
                severity: ThermalSeverity::Critical
            }
        ));
    }

    // === Reset ===

    #[test]
    fn reset_clears_all_state() {
        let mut pacer = DriftPacer::with_defaults();
        let key = test_key();

        for _ in 0..10 {
            pacer.observe(&obs(&key, 10_000_000));
        }
        assert_eq!(pacer.tracked_keys(), 1);

        pacer.reset();
        assert_eq!(pacer.tracked_keys(), 0);

        // Should start baseline collection again
        let d = pacer.observe(&obs(&key, 10_000_000));
        assert!(matches!(d.reason, PaceReason::Baseline));
    }

    // === Thermal poll rate limiting ===

    struct CountingThermal {
        poll_count: std::rc::Rc<std::cell::Cell<u32>>,
    }

    impl ThermalHintSource for CountingThermal {
        fn poll(&mut self) -> Option<ThermalHint> {
            self.poll_count.set(self.poll_count.get() + 1);
            None
        }
        fn min_poll_interval(&self) -> Duration {
            Duration::from_millis(100)
        }
    }

    #[test]
    fn thermal_poll_respects_min_interval() {
        let poll_count = std::rc::Rc::new(std::cell::Cell::new(0u32));
        let counter = poll_count.clone();

        let config = DriftPacerConfig {
            baseline_count: 4,
            ..Default::default()
        };
        let mut pacer = DriftPacer::new(config)
            .with_thermal_source(Box::new(CountingThermal { poll_count: counter }));
        let key = test_key();

        // 30 observations at 10ms each = 300ms total wall time.
        // With 100ms poll interval we expect exactly 3 polls:
        //   1. Immediate (first observe — seeded at interval)
        //   2. After observations accumulate another 100ms (obs 11)
        //   3. After another 100ms (obs 21)
        // Without rate limiting this would be 30 polls.
        for _ in 0..30 {
            pacer.observe(&obs(&key, 10_000_000)); // 10ms wall
        }

        let count = poll_count.get();
        assert_eq!(count, 3, "expected 3 polls (initial + 2 periodic), got {count}");
    }

    #[test]
    fn thermal_poll_resets_with_pacer() {
        let poll_count = std::rc::Rc::new(std::cell::Cell::new(0u32));
        let counter = poll_count.clone();

        let config = DriftPacerConfig {
            baseline_count: 4,
            ..Default::default()
        };
        let mut pacer = DriftPacer::new(config)
            .with_thermal_source(Box::new(CountingThermal { poll_count: counter }));
        let key = test_key();

        // Drive past initial poll
        pacer.observe(&obs(&key, 10_000_000));
        assert_eq!(poll_count.get(), 1);

        pacer.reset();

        // After reset, next observe should poll immediately again
        pacer.observe(&obs(&key, 10_000_000));
        assert_eq!(poll_count.get(), 2, "reset should re-enable immediate poll");
    }

    #[test]
    fn thermal_poll_cadence_includes_recommended_idle() {
        let poll_count = std::rc::Rc::new(std::cell::Cell::new(0u32));
        let counter = poll_count.clone();

        let config = DriftPacerConfig {
            baseline_count: 4,
            wall_drift_threshold: 1.20,
            hot_trigger_count: 3,
            idle_step_ms: 40, // Large idle to make the effect obvious
            max_idle_ms: 40,
            ..Default::default()
        };
        let mut pacer = DriftPacer::new(config)
            .with_thermal_source(Box::new(CountingThermal { poll_count: counter }));
        let key = test_key();

        // Baseline: 4 × 10ms wall.
        // Obs 1 triggers the initial poll (seeded at interval), reset.
        // Obs 2–4 accumulate 10ms each = 30ms (below 100ms interval).
        for _ in 0..4 {
            pacer.observe(&obs(&key, 10_000_000));
        }

        // Trigger pacing: 3 × 15ms (50% above 10ms baseline).
        // After obs 7, pacer recommends 40ms idle.
        // Accumulator across obs 5–7: 30 + 15+15+15 = 75ms (no poll).
        for _ in 0..3 {
            pacer.observe(&obs(&key, 15_000_000));
        }

        // Reset counter to isolate the idle-aware phase.
        poll_count.set(0);

        // 3 more observations at 10ms wall. Each call now accumulates
        // wall_ns (10ms) + last_recommended_idle_ns (40ms) = 50ms.
        //
        // Obs 8: carried 75ms + 10ms + 40ms = 125ms ≥ 100ms → poll! (1)
        // Obs 9: 0 + 10ms + 40ms = 50ms → no poll
        // Obs 10: 50 + 10 + 40 = 100ms ≥ 100ms → poll! (2)
        //
        // Without idle accounting the cadence would only count wall_ns:
        // Obs 8: 75+10 = 85 → no. Obs 9: 95 → no. Obs 10: 105 → poll (1).
        // So 1 poll instead of 2.
        for _ in 0..3 {
            pacer.observe(&obs(&key, 10_000_000));
        }

        let count = poll_count.get();
        assert_eq!(
            count, 2,
            "expected 2 polls (idle-aware cadence), got {count}; \
             without idle accounting this would be 1"
        );
    }

    // === Dynamic interval adaptation ===

    /// Mock source that changes its poll interval after a set number of polls.
    /// Simulates ADPF headroom degradation (10s → 1s after broken HAL detected).
    struct DegradingThermal {
        poll_count: std::rc::Rc<std::cell::Cell<u32>>,
        degrade_after: u32,
    }

    impl ThermalHintSource for DegradingThermal {
        fn poll(&mut self) -> Option<ThermalHint> {
            self.poll_count.set(self.poll_count.get() + 1);
            None
        }
        fn min_poll_interval(&self) -> Duration {
            if self.poll_count.get() >= self.degrade_after {
                Duration::from_millis(50) // degraded: faster cadence
            } else {
                Duration::from_millis(200) // normal: slower cadence
            }
        }
    }

    #[test]
    fn pacer_adapts_to_changed_poll_interval() {
        let poll_count = std::rc::Rc::new(std::cell::Cell::new(0u32));
        let counter = poll_count.clone();

        let config = DriftPacerConfig {
            baseline_count: 4,
            ..Default::default()
        };
        // Source starts at 200ms interval, degrades to 50ms after 2 polls.
        let mut pacer = DriftPacer::new(config).with_thermal_source(Box::new(
            DegradingThermal {
                poll_count: counter,
                degrade_after: 2,
            },
        ));
        let key = test_key();

        // Phase 1: source at 200ms interval.
        // Obs 1 triggers the seeded initial poll.
        pacer.observe(&obs(&key, 10_000_000));
        assert_eq!(poll_count.get(), 1);

        // 20 more at 10ms = 200ms → triggers poll #2.
        for _ in 0..20 {
            pacer.observe(&obs(&key, 10_000_000));
        }
        assert_eq!(poll_count.get(), 2);

        // After poll #2, source degraded to 50ms. Pacer re-reads interval.
        // Don't reset poll_count — the DegradingThermal uses it for
        // its degradation check, and resetting would un-degrade it.
        let before = poll_count.get(); // 2

        // 10 observations at 10ms = 100ms.
        // At 50ms cadence: poll after obs 5 (50ms), poll after obs 10 (100ms).
        // Without interval re-read this would still be 200ms cadence → 0 polls.
        for _ in 0..10 {
            pacer.observe(&obs(&key, 10_000_000));
        }
        let phase3_polls = poll_count.get() - before;
        assert_eq!(
            phase3_polls, 2,
            "pacer should adapt to shorter interval after source degradation: got {phase3_polls}"
        );
    }

    // === best_half_median ===

    #[test]
    fn best_half_median_single_value() {
        assert_eq!(best_half_median(&[42]), 42);
    }

    #[test]
    fn best_half_median_favors_lower_half() {
        // [5, 10, 15, 20, 100, 200, 300, 400]
        // Lower half: [5, 10, 15, 20], median = best[2] = 15
        let samples = [100, 5, 200, 10, 300, 15, 400, 20];
        let result = best_half_median(&samples);
        assert_eq!(result, 15);
    }

    #[test]
    fn best_half_median_resists_outliers() {
        // Baseline with one huge outlier
        let samples = [10, 11, 10, 12, 10, 11, 10, 500];
        let result = best_half_median(&samples);
        // Lower half sorted: [10, 10, 10, 10], median = 10
        assert_eq!(result, 10);
    }
}
