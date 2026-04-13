//! Types shared between the pacer and callers.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Identity key for per-operation baseline tracking.
///
/// The pacer maintains a separate baseline for each unique `OpKey`.
/// Using `(backend, gpu_family, kernel_family, log_n, direction)` means
/// a log₂₀ four-step forward NTT on Mali-G715 tracks independently
/// from a log₁₄ Stockham inverse on Adreno 830.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OpKey {
    pub backend: String,
    pub platform_class: String,
    pub gpu_family: String,
    pub kernel_family: String,
    pub log_n: u32,
    pub direction: String,
}

/// Per-stage GPU timing (optional, from hardware timestamp queries).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageTiming {
    pub label: String,
    pub duration_ns: u64,
}

/// A single observation from one NTT batch execution.
///
/// The caller fills this after each batch and passes it to
/// [`crate::DriftPacer::observe`]. The pacer uses `observed_wall_ns`
/// as the primary signal and `gpu_total_ns` as a secondary signal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionObservation {
    /// Which operation this observation is for.
    pub op_key: OpKey,
    /// Caller-measured wall-clock time for this batch, in nanoseconds.
    /// This is the primary pacing signal — measured *outside* the GPU
    /// execution path to include dispatch overhead, readback, etc.
    pub observed_wall_ns: u64,
    /// GPU-side total time in nanoseconds (from hardware timestamp
    /// queries). Secondary signal. `None` if unprofiled or unavailable.
    pub gpu_total_ns: Option<u64>,
    /// Per-stage GPU timings (optional, for diagnostics).
    pub gpu_stage_ns: Vec<StageTiming>,
    /// Bytes moved to/from GPU in this batch (optional, for throughput).
    pub bytes_moved: Option<u64>,
    /// Number of NTTs in this batch (usually 1).
    pub batch_size: u32,
}

/// How aggressively the pacer is throttling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PaceLevel {
    /// No throttling — running at full speed.
    None,
    /// Light throttling — small idle gaps to prevent thermal ramp.
    Light,
    /// Moderate throttling — sustained drift detected.
    Moderate,
    /// Heavy throttling — approaching thermal limit or OS hint says serious.
    Heavy,
}

/// Why the pacer made this decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PaceReason {
    /// No pacing needed — within baseline.
    Stable,
    /// Warming up — collecting baseline samples, no decision yet.
    Baseline,
    /// Wall time drift exceeded threshold for N consecutive batches.
    WallDrift {
        consecutive_hot: u32,
        wall_ratio: f64,
    },
    /// GPU time drift exceeded threshold for N consecutive batches.
    GpuDrift {
        consecutive_hot: u32,
        gpu_ratio: f64,
    },
    /// OS thermal hint triggered escalation.
    ThermalHint { severity: ThermalSeverity },
    /// Recovering — drift has subsided, reducing idle gradually.
    Recovery { stable_count: u32 },
}

/// The pacer's output — what the caller should do before the next batch.
///
/// The pacer never sleeps itself. The caller applies `idle_for` using
/// its own scheduling primitive:
/// - Native: `std::thread::sleep(decision.idle_for)`
/// - Browser: `setTimeout(next_batch, decision.idle_for.as_millis())`
/// - Async: `tokio::time::sleep(decision.idle_for).await`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaceDecision {
    /// How long to idle before the next batch. `Duration::ZERO` means
    /// no pause — run the next batch immediately.
    #[serde(with = "duration_ms")]
    pub idle_for: Duration,
    /// Current throttling level.
    pub level: PaceLevel,
    /// Why this decision was made.
    pub reason: PaceReason,
}

impl PaceDecision {
    /// No pacing — run immediately.
    pub fn none() -> Self {
        Self {
            idle_for: Duration::ZERO,
            level: PaceLevel::None,
            reason: PaceReason::Stable,
        }
    }

    /// Still collecting baseline — run immediately.
    pub fn baseline() -> Self {
        Self {
            idle_for: Duration::ZERO,
            level: PaceLevel::None,
            reason: PaceReason::Baseline,
        }
    }
}

/// Severity levels from OS thermal hint sources.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThermalSeverity {
    /// Within normal operating range.
    Nominal,
    /// Approaching thermal limit — light throttling recommended.
    Fair,
    /// At thermal limit — moderate throttling needed.
    Serious,
    /// Critical — reduce workload immediately.
    Critical,
}

/// A hint from an OS thermal API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalHint {
    pub severity: ThermalSeverity,
    /// Thermal headroom (0.0–1.0) if available. 0.0 = at thermal limit.
    pub headroom: Option<f64>,
}

/// Trait for platform-specific thermal hint providers.
///
/// Implementors:
/// - Android: ADPF `getThermalHeadroom()` + thermal status listener
/// - Apple: `ProcessInfo.thermalState` notifications
/// - Browser: not implemented (use drift-only)
pub trait ThermalHintSource {
    /// Poll for the latest thermal hint.
    ///
    /// Returns `None` if no new data is available or if the minimum
    /// poll interval has not elapsed.
    fn poll(&mut self) -> Option<ThermalHint>;

    /// Minimum interval between meaningful polls. Callers should not
    /// call `poll()` more frequently than this.
    ///
    /// Android ADPF: 10 seconds (documented limitation).
    /// Apple: 30 seconds (notification-based, not polling).
    fn min_poll_interval(&self) -> Duration;
}

// ---------------------------------------------------------------------------
// Serde helper: Duration as milliseconds
// ---------------------------------------------------------------------------

mod duration_ms {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S: Serializer>(d: &Duration, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u64(d.as_millis() as u64)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Duration, D::Error> {
        let ms = u64::deserialize(d)?;
        Ok(Duration::from_millis(ms))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pace_decision_none_is_zero_idle() {
        let d = PaceDecision::none();
        assert_eq!(d.idle_for, Duration::ZERO);
        assert_eq!(d.level, PaceLevel::None);
    }

    #[test]
    fn pace_decision_baseline_is_zero_idle() {
        let d = PaceDecision::baseline();
        assert_eq!(d.idle_for, Duration::ZERO);
        assert_eq!(d.level, PaceLevel::None);
    }

    #[test]
    fn op_key_equality() {
        let a = OpKey {
            backend: "Vulkan".into(),
            platform_class: "AndroidNative".into(),
            gpu_family: "Mali".into(),
            kernel_family: "stockham".into(),
            log_n: 20,
            direction: "Forward".into(),
        };
        let b = a.clone();
        assert_eq!(a, b);

        let c = OpKey {
            log_n: 18,
            ..a.clone()
        };
        assert_ne!(a, c);
    }

    #[test]
    fn pace_decision_roundtrips_json() {
        let d = PaceDecision {
            idle_for: Duration::from_millis(15),
            level: PaceLevel::Moderate,
            reason: PaceReason::WallDrift {
                consecutive_hot: 4,
                wall_ratio: 1.25,
            },
        };
        let json = serde_json::to_string(&d).unwrap();
        let parsed: PaceDecision = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.idle_for, Duration::from_millis(15));
        assert_eq!(parsed.level, PaceLevel::Moderate);
    }

    #[test]
    fn execution_observation_roundtrips_json() {
        let obs = ExecutionObservation {
            op_key: OpKey {
                backend: "Vulkan".into(),
                platform_class: "AndroidNative".into(),
                gpu_family: "Adreno".into(),
                kernel_family: "four-step".into(),
                log_n: 20,
                direction: "Forward".into(),
            },
            observed_wall_ns: 15_000_000,
            gpu_total_ns: Some(12_000_000),
            gpu_stage_ns: vec![StageTiming {
                label: "r4_pass".into(),
                duration_ns: 6_000_000,
            }],
            bytes_moved: Some(4_194_304),
            batch_size: 1,
        };
        let json = serde_json::to_string(&obs).unwrap();
        let parsed: ExecutionObservation = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.observed_wall_ns, 15_000_000);
        assert_eq!(parsed.gpu_total_ns, Some(12_000_000));
    }
}
