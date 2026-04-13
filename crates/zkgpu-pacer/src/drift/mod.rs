//! Adaptive drift-based pacer.
//!
//! Baselines on the first N homogeneous batches per [`OpKey`], then
//! triggers idle gaps after sustained timing drift. Recovers gradually
//! when drift subsides.

mod config;
mod state;
mod thermal;
#[cfg(test)]
mod tests;

pub use config::DriftPacerConfig;

use std::collections::HashMap;

use crate::observation::{
    ExecutionObservation, OpKey, PaceDecision, ThermalHintSource,
};

use state::OpState;
use thermal::ThermalPollState;

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
    thermal: ThermalPollState,
}

impl DriftPacer {
    /// Create a new pacer with the given configuration.
    pub fn new(config: DriftPacerConfig) -> Self {
        Self {
            config,
            states: HashMap::new(),
            thermal: ThermalPollState::new(),
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
        self.thermal.attach_source(source);
        self
    }

    /// Process one execution observation and return a pacing decision.
    ///
    /// Call this after each NTT batch completes.
    pub fn observe(&mut self, obs: &ExecutionObservation) -> PaceDecision {
        let decision = self.decide(obs);
        // Record the idle we recommended so the *next* call's cadence
        // accumulator accounts for the sleep the caller was told to apply.
        self.thermal.record_recommended_idle(decision.idle_for.as_nanos() as u64);
        decision
    }

    /// Internal: poll thermal source (rate-limited), evaluate drift,
    /// and produce a pacing decision.
    fn decide(&mut self, obs: &ExecutionObservation) -> PaceDecision {
        // Step 1: Poll thermal hint if due.
        self.thermal.poll_if_due(obs.observed_wall_ns);

        // Step 2: Get or create per-OpKey state.
        let state = self
            .states
            .entry(obs.op_key.clone())
            .or_insert_with(OpState::new);

        // Step 3: Baseline collection phase.
        if let Some(decision) = state.try_collect_baseline(&self.config, obs) {
            return decision;
        }

        // Step 4: Compute drift ratios.
        let (wall_ratio, gpu_ratio) = state.drift_ratios(obs);

        // Step 5: Check thermal override.
        if let Some(decision) = self.thermal.check_thermal_override(&self.config, state) {
            return decision;
        }

        // Step 6: Update drift counters and make decision.
        let signal = state.update_drift_counters(&self.config, wall_ratio, gpu_ratio);
        state.make_decision(&self.config, &signal)
    }

    /// Reset all state. Useful for benchmark mode transitions.
    pub fn reset(&mut self) {
        self.states.clear();
        self.thermal.reset();
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
            .field("has_thermal_source", &self.thermal.has_source())
            .finish()
    }
}
