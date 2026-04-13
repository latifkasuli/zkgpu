use super::*;
use super::state::best_half_median;
use std::time::Duration;

use crate::observation::{
    ExecutionObservation, OpKey, PaceLevel, PaceReason, ThermalHint,
    ThermalHintSource, ThermalSeverity,
};

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
