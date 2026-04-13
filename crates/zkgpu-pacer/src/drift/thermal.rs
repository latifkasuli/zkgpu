use std::time::Duration;

use crate::observation::{PaceDecision, PaceLevel, PaceReason, ThermalHint, ThermalHintSource, ThermalSeverity};

use super::config::DriftPacerConfig;
use super::state::OpState;

/// Encapsulates thermal poll cadence bookkeeping.
///
/// Tracks the thermal hint source, rate-limits polling, and applies
/// thermal overrides to pacing decisions.
pub(super) struct ThermalPollState {
    source: Option<Box<dyn ThermalHintSource>>,
    last_thermal: Option<ThermalHint>,
    /// Nanoseconds accumulated since the last thermal poll.
    /// Initialized to `poll_interval_ns` so the first `observe()` call
    /// triggers an immediate poll.
    ns_since_poll: u64,
    /// Cached `min_poll_interval()` from the thermal source, in
    /// nanoseconds. Zero when no source is attached.
    poll_interval_ns: u64,
    /// Idle duration (ns) recommended by the previous `observe()` call.
    /// Added to the cadence accumulator on the next call so that the
    /// thermal poll tracks real elapsed time, not just batch execution
    /// time. Without this, once the pacer starts inserting sleeps the
    /// cadence undercounts and stale thermal overrides persist.
    last_recommended_idle_ns: u64,
}

impl ThermalPollState {
    pub(super) fn new() -> Self {
        Self {
            source: None,
            last_thermal: None,
            ns_since_poll: 0,
            poll_interval_ns: 0,
            last_recommended_idle_ns: 0,
        }
    }

    /// Attach a platform thermal hint source.
    pub(super) fn attach_source(&mut self, source: Box<dyn ThermalHintSource>) {
        self.poll_interval_ns = source.min_poll_interval().as_nanos() as u64;
        // Seed at the interval value so the first observe() immediately
        // crosses the threshold and triggers the initial poll.
        self.ns_since_poll = self.poll_interval_ns;
        self.source = Some(source);
    }

    /// Record the idle duration recommended by the last decision.
    pub(super) fn record_recommended_idle(&mut self, idle_ns: u64) {
        self.last_recommended_idle_ns = idle_ns;
    }

    /// Poll the thermal source if enough time has elapsed since the last poll.
    ///
    /// The cadence accumulates *estimated real elapsed time*: batch wall
    /// time + the idle the caller was told to apply after the previous
    /// observation.
    pub(super) fn poll_if_due(&mut self, observed_wall_ns: u64) {
        if let Some(ref mut source) = self.source {
            self.ns_since_poll = self
                .ns_since_poll
                .saturating_add(observed_wall_ns)
                .saturating_add(self.last_recommended_idle_ns);
            if self.ns_since_poll >= self.poll_interval_ns {
                self.ns_since_poll = 0;
                if let Some(hint) = source.poll() {
                    self.last_thermal = Some(hint);
                }
                // Re-read in case the source adjusted its interval
                // (e.g., ADPF headroom degraded to status-only mode).
                self.poll_interval_ns = source.min_poll_interval().as_nanos() as u64;
            }
        }
    }

    /// Check if the current thermal state warrants an override decision.
    ///
    /// Returns `Some(PaceDecision)` for Critical/Serious severity,
    /// `None` if no thermal override applies.
    pub(super) fn check_thermal_override(
        &self,
        config: &DriftPacerConfig,
        state: &mut OpState,
    ) -> Option<PaceDecision> {
        let hint = self.last_thermal.as_ref()?;

        if hint.severity == ThermalSeverity::Critical {
            let idle = Duration::from_millis(config.max_idle_ms);
            state.current_idle = idle;
            return Some(PaceDecision {
                idle_for: idle,
                level: PaceLevel::Heavy,
                reason: PaceReason::ThermalHint {
                    severity: ThermalSeverity::Critical,
                },
            });
        }

        if hint.severity == ThermalSeverity::Serious {
            let idle = Duration::from_millis(config.max_idle_ms * 2 / 3);
            state.current_idle = idle;
            return Some(PaceDecision {
                idle_for: idle,
                level: PaceLevel::Heavy,
                reason: PaceReason::ThermalHint {
                    severity: ThermalSeverity::Serious,
                },
            });
        }

        None
    }

    /// Reset thermal state. Re-seeds the poll cadence for immediate first poll.
    pub(super) fn reset(&mut self) {
        self.last_thermal = None;
        self.last_recommended_idle_ns = 0;
        // Re-seed so the next observe() triggers an immediate poll.
        self.ns_since_poll = self.poll_interval_ns;
    }

    /// Whether a thermal source is attached.
    pub(super) fn has_source(&self) -> bool {
        self.source.is_some()
    }
}
