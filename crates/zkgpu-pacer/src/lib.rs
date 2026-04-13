//! Cross-platform adaptive pacing for sustained GPU compute workloads.
//!
//! This crate provides a timing-drift-based pacer that sits *above* the
//! NTT execution path. It does not sleep or block — it returns
//! [`PaceDecision`] values that the caller applies using its own
//! scheduler (`std::thread::sleep`, `setTimeout`, async sleep, etc.).
//!
//! # Architecture
//!
//! ```text
//! Proving loop
//!   └─ observe(wall, gpu) ──▶ DriftPacer ──▶ PaceDecision { idle_for }
//!                                  │
//!                      ThermalHintSource (optional)
//!                        Android ADPF / Apple thermalState
//! ```
//!
//! The pacer is keyed by [`OpKey`] so each `(backend, gpu_family,
//! kernel_family, log_n, direction)` combination tracks its own baseline.
//! The primary signal is caller-observed wall time; GPU timestamps are a
//! secondary signal when available.

mod drift;
mod observation;

// Compiled on Android for production use, and during `cargo test` on
// all platforms so the mapping-logic unit tests run on dev machines.
#[cfg(any(target_os = "android", test))]
mod android_adpf;

// Compiled on Apple platforms for production use, and during `cargo test`
// on all platforms so the mapping-logic unit tests run on dev machines.
#[cfg(any(target_vendor = "apple", test))]
mod apple_thermal;

pub use drift::{DriftPacer, DriftPacerConfig};
pub use observation::{
    ExecutionObservation, OpKey, PaceDecision, PaceLevel, PaceReason, StageTiming, ThermalHint,
    ThermalHintSource, ThermalSeverity,
};

#[cfg(target_os = "android")]
pub use android_adpf::AdpfThermalSource;

#[cfg(target_vendor = "apple")]
pub use apple_thermal::AppleThermalSource;
