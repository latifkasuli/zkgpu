use std::time::Duration;

use crate::observation::{ExecutionObservation, PaceDecision, PaceLevel, PaceReason};

use super::config::DriftPacerConfig;

/// Per-OpKey tracking state.
#[derive(Debug)]
pub(super) struct OpState {
    /// Baseline samples collected so far (wall_ns).
    baseline_wall: Vec<u64>,
    /// Baseline samples collected so far (gpu_ns).
    baseline_gpu: Vec<u64>,
    /// Computed baseline wall time (median of best samples). None during warmup.
    baseline_wall_ns: Option<u64>,
    /// Computed baseline GPU time. None if unprofiled or during warmup.
    baseline_gpu_ns: Option<u64>,
    /// Consecutive batches above drift threshold.
    pub(super) consecutive_hot: u32,
    /// Consecutive batches below recovery threshold (while pacing).
    pub(super) consecutive_stable: u32,
    /// Current idle duration being applied.
    pub(super) current_idle: Duration,
}

impl OpState {
    pub(super) fn new() -> Self {
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

    /// Try to collect a baseline sample. Returns `Some(PaceDecision)` if
    /// still in the warmup phase, `None` once baselines are ready.
    pub(super) fn try_collect_baseline(
        &mut self,
        config: &DriftPacerConfig,
        obs: &ExecutionObservation,
    ) -> Option<PaceDecision> {
        if self.baseline_wall_ns.is_some() {
            return None; // baselines already computed
        }

        self.baseline_wall.push(obs.observed_wall_ns);
        if let Some(gpu) = obs.gpu_total_ns {
            self.baseline_gpu.push(gpu);
        }

        if self.baseline_wall.len() >= config.baseline_count as usize {
            self.compute_baselines();
            None // baselines just became ready — caller should evaluate drift
        } else {
            Some(PaceDecision::baseline())
        }
    }

    /// Compute wall and GPU drift ratios against established baselines.
    pub(super) fn drift_ratios(&self, obs: &ExecutionObservation) -> (f64, Option<f64>) {
        let baseline_wall = self.baseline_wall_ns.unwrap();
        let wall_ratio = if baseline_wall > 0 {
            obs.observed_wall_ns as f64 / baseline_wall as f64
        } else {
            1.0
        };

        let gpu_ratio = match (obs.gpu_total_ns, self.baseline_gpu_ns) {
            (Some(gpu), Some(base)) if base > 0 => Some(gpu as f64 / base as f64),
            _ => None,
        };

        (wall_ratio, gpu_ratio)
    }

    /// Evaluate drift signals and update consecutive hot/stable counters.
    pub(super) fn update_drift_counters(
        &mut self,
        config: &DriftPacerConfig,
        wall_ratio: f64,
        gpu_ratio: Option<f64>,
    ) -> DriftSignal {
        let is_hot_wall = wall_ratio >= config.wall_drift_threshold;
        let is_hot_gpu = gpu_ratio
            .map(|r| r >= config.gpu_drift_threshold)
            .unwrap_or(false);
        let is_hot = is_hot_gpu || is_hot_wall;

        let is_stable = wall_ratio < config.recovery_threshold
            && gpu_ratio
                .map(|r| r < config.recovery_threshold)
                .unwrap_or(true);

        if is_hot {
            self.consecutive_hot += 1;
            self.consecutive_stable = 0;
        } else if is_stable {
            self.consecutive_stable += 1;
            self.consecutive_hot = 0;
        } else {
            // Between thresholds — neither hot nor stable
            self.consecutive_hot = 0;
            // Don't reset stable count — allow gradual recovery
        }

        DriftSignal {
            is_hot_gpu,
            wall_ratio,
            gpu_ratio,
        }
    }

    /// Produce a pacing decision based on current drift state.
    pub(super) fn make_decision(
        &mut self,
        config: &DriftPacerConfig,
        signal: &DriftSignal,
    ) -> PaceDecision {
        // Trigger: sustained drift
        if self.consecutive_hot >= config.hot_trigger_count {
            let step = Duration::from_millis(config.idle_step_ms);
            let max = Duration::from_millis(config.max_idle_ms);
            self.current_idle = (self.current_idle + step).min(max);
            self.consecutive_stable = 0;

            let level = if self.current_idle >= max {
                PaceLevel::Heavy
            } else if self.current_idle >= Duration::from_millis(config.max_idle_ms / 2) {
                PaceLevel::Moderate
            } else {
                PaceLevel::Light
            };

            let reason = if signal.is_hot_gpu {
                PaceReason::GpuDrift {
                    consecutive_hot: self.consecutive_hot,
                    gpu_ratio: signal.gpu_ratio.unwrap_or(0.0),
                }
            } else {
                PaceReason::WallDrift {
                    consecutive_hot: self.consecutive_hot,
                    wall_ratio: signal.wall_ratio,
                }
            };

            return PaceDecision {
                idle_for: self.current_idle,
                level,
                reason,
            };
        }

        // Recovery: sustained stability while pacing
        if !self.current_idle.is_zero()
            && self.consecutive_stable >= config.recovery_count
        {
            let step = Duration::from_millis(config.idle_step_ms);
            self.current_idle = self.current_idle.saturating_sub(step);
            self.consecutive_stable = 0;

            if self.current_idle.is_zero() {
                return PaceDecision::none();
            }

            return PaceDecision {
                idle_for: self.current_idle,
                level: PaceLevel::Light,
                reason: PaceReason::Recovery {
                    stable_count: config.recovery_count,
                },
            };
        }

        // Hold current idle if pacing but not yet recovered
        if !self.current_idle.is_zero() {
            let level = if self.current_idle >= Duration::from_millis(config.max_idle_ms) {
                PaceLevel::Heavy
            } else if self.current_idle >= Duration::from_millis(config.max_idle_ms / 2) {
                PaceLevel::Moderate
            } else {
                PaceLevel::Light
            };

            return PaceDecision {
                idle_for: self.current_idle,
                level,
                reason: PaceReason::Stable,
            };
        }

        PaceDecision::none()
    }
}

/// Intermediate drift evaluation result.
pub(super) struct DriftSignal {
    pub is_hot_gpu: bool,
    pub wall_ratio: f64,
    pub gpu_ratio: Option<f64>,
}

/// Median of the lower half of values — resistant to high outliers.
pub(super) fn best_half_median(samples: &[u64]) -> u64 {
    if samples.is_empty() {
        return 0;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let half = (sorted.len() / 2).max(1);
    let best = &sorted[..half];
    best[best.len() / 2]
}
