//! Android ADPF thermal hint source.
//!
//! Uses the NDK `AThermalManager` API (`<android/thermal.h>`) to
//! provide thermal headroom and status to the drift pacer.
//!
//! # API level requirements
//!
//! | Function                            | NDK API level |
//! |-------------------------------------|---------------|
//! | `AThermal_acquireManager`           | 30            |
//! | `AThermal_getCurrentThermalStatus`  | 30            |
//! | `AThermal_getThermalHeadroom`       | 31            |
//!
//! All functions are resolved at runtime via `dlsym`. If the thermal
//! manager is unavailable (API < 30, emulator without thermal HAL),
//! [`AdpfThermalSource::new()`] returns `None` and the pacer falls
//! back to drift-only pacing.
//!
//! # Headroom semantics
//!
//! ADPF headroom: 0.0 (cold) to 1.0 (SEVERE threshold) to >1.0 (throttling).
//! This source converts to the crate's convention where 0.0 = at thermal
//! limit and 1.0 = maximum headroom.

use std::ffi::c_int;
use std::time::Duration;

use crate::observation::ThermalSeverity;

// ---------------------------------------------------------------------------
// Poll interval constants
// ---------------------------------------------------------------------------

/// Poll interval when only `AThermal_getCurrentThermalStatus` is
/// available (API 30). Status is a cheap getter with no documented
/// cadence restriction — 1 second is responsive without being wasteful.
const STATUS_ONLY_POLL_INTERVAL: Duration = Duration::from_secs(1);

/// Poll interval when `AThermal_getThermalHeadroom` is available
/// and returning usable data (API 31+). ADPF documentation says
/// headroom should not be polled more often than every 10 seconds —
/// returns NaN if violated.
const HEADROOM_POLL_INTERVAL: Duration = Duration::from_secs(10);

/// Consecutive NaN returns from `getThermalHeadroom` before the
/// source degrades to status-only mode. This detects devices where
/// the headroom symbol exists (API 31+) but the thermal HAL does not
/// actually support headroom queries.
const HEADROOM_NAN_THRESHOLD: u32 = 3;

/// Maximum valid forecast horizon, in seconds.
/// `PowerManager.getThermalHeadroom()` documents `forecastSeconds`
/// as valid in 0..=60. The NDK `AThermal_getThermalHeadroom` wraps
/// the same HAL call, so we honour the framework-documented range.
const MAX_FORECAST_SECONDS: i32 = 60;

// ---------------------------------------------------------------------------
// AThermalStatus constants (NDK API 30+)
// ---------------------------------------------------------------------------

const ATHERMAL_STATUS_ERROR: c_int = -1;
const ATHERMAL_STATUS_NONE: c_int = 0;
const ATHERMAL_STATUS_LIGHT: c_int = 1;
const ATHERMAL_STATUS_MODERATE: c_int = 2;
const ATHERMAL_STATUS_SEVERE: c_int = 3;
const ATHERMAL_STATUS_CRITICAL: c_int = 4;
const ATHERMAL_STATUS_EMERGENCY: c_int = 5;
const ATHERMAL_STATUS_SHUTDOWN: c_int = 6;

// ---------------------------------------------------------------------------
// Pure mapping helpers (no FFI, testable on all platforms)
// ---------------------------------------------------------------------------

/// Map `AThermalStatus` constant to our severity level.
fn status_to_severity(status: c_int) -> ThermalSeverity {
    match status {
        ATHERMAL_STATUS_NONE | ATHERMAL_STATUS_LIGHT => ThermalSeverity::Nominal,
        ATHERMAL_STATUS_MODERATE => ThermalSeverity::Fair,
        ATHERMAL_STATUS_SEVERE => ThermalSeverity::Serious,
        // CRITICAL (4), EMERGENCY (5), SHUTDOWN (6), and any future
        // values above CRITICAL are treated as Critical — new severity
        // levels added above SHUTDOWN are more likely to be escalations.
        s if s >= ATHERMAL_STATUS_CRITICAL => ThermalSeverity::Critical,
        // ATHERMAL_STATUS_ERROR (-1) or unknown negative values
        _ => ThermalSeverity::Nominal,
    }
}

/// Map ADPF headroom to our severity level.
///
/// ADPF headroom: 0.0 = cold, 1.0 = SEVERE threshold, >1.0 = throttling.
/// Returns `None` if the value is NaN or negative (API error / too frequent).
fn headroom_to_severity(adpf_headroom: f32) -> Option<ThermalSeverity> {
    if adpf_headroom.is_nan() || adpf_headroom < 0.0 {
        return None;
    }
    Some(if adpf_headroom >= 0.9 {
        ThermalSeverity::Critical
    } else if adpf_headroom >= 0.7 {
        ThermalSeverity::Serious
    } else if adpf_headroom >= 0.5 {
        ThermalSeverity::Fair
    } else {
        ThermalSeverity::Nominal
    })
}

/// Convert ADPF headroom (0 = cold, 1 = hot) to our convention (0 = hot, 1 = cold).
///
/// Returns `None` for NaN or negative values.
fn convert_headroom(adpf_headroom: f32) -> Option<f64> {
    if adpf_headroom.is_nan() || adpf_headroom < 0.0 {
        None
    } else {
        Some((1.0 - adpf_headroom as f64).clamp(0.0, 1.0))
    }
}

/// Return the more severe of two severity levels.
fn worse_severity(a: ThermalSeverity, b: ThermalSeverity) -> ThermalSeverity {
    let rank = |s: &ThermalSeverity| match s {
        ThermalSeverity::Nominal => 0,
        ThermalSeverity::Fair => 1,
        ThermalSeverity::Serious => 2,
        ThermalSeverity::Critical => 3,
    };
    if rank(&a) >= rank(&b) {
        a
    } else {
        b
    }
}

// ---------------------------------------------------------------------------
// Android FFI implementation (only compiled on Android)
// ---------------------------------------------------------------------------

#[cfg(target_os = "android")]
mod imp {
    use std::ffi::{c_char, c_float, c_int, c_void};
    use std::time::Duration;

    use crate::observation::{ThermalHint, ThermalHintSource};

    use super::{
        convert_headroom, headroom_to_severity, status_to_severity, worse_severity,
        ATHERMAL_STATUS_ERROR,
    };

    // --- dlopen / dlsym ---

    const RTLD_NOW: c_int = 2;

    extern "C" {
        fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        fn dlclose(handle: *mut c_void) -> c_int;
    }

    // --- NDK function pointer types ---

    type AcquireManagerFn = unsafe extern "C" fn() -> *mut c_void;
    type ReleaseManagerFn = unsafe extern "C" fn(manager: *mut c_void);
    type GetStatusFn = unsafe extern "C" fn(manager: *mut c_void) -> c_int;
    type GetHeadroomFn =
        unsafe extern "C" fn(manager: *mut c_void, forecast_seconds: c_int) -> c_float;

    // --- Resolved vtable ---

    /// Holds the dlopen handle, `AThermalManager*`, and resolved
    /// function pointers. Releases the manager and closes the
    /// library on drop.
    struct ThermalVtable {
        lib: *mut c_void,
        manager: *mut c_void,
        release_manager: ReleaseManagerFn,
        get_status: GetStatusFn,
        get_headroom: Option<GetHeadroomFn>,
    }

    impl Drop for ThermalVtable {
        fn drop(&mut self) {
            unsafe {
                (self.release_manager)(self.manager);
                dlclose(self.lib);
            }
        }
    }

    // SAFETY: AThermalManager is documented as thread-safe in the NDK.
    // The function pointers are stateless C functions with no thread affinity.
    unsafe impl Send for ThermalVtable {}

    // --- Public source ---

    /// Android ADPF thermal hint source.
    ///
    /// Wraps the NDK `AThermalManager` to provide thermal headroom and
    /// status to [`DriftPacer`](crate::DriftPacer).
    ///
    /// # Graceful degradation
    ///
    /// - API < 30: `new()` returns `None` (no thermal manager)
    /// - API 30: status only, 1 s poll cadence
    /// - API 31+ with working HAL: headroom + status, 10 s cadence
    /// - API 31+ with broken HAL: auto-degrades to status-only after
    ///   3 consecutive NaN returns from `getThermalHeadroom`
    ///
    /// # Example
    ///
    /// ```ignore
    /// use zkgpu_pacer::{DriftPacer, DriftPacerConfig};
    /// use zkgpu_pacer::AdpfThermalSource;
    ///
    /// let mut pacer = DriftPacer::new(DriftPacerConfig::default());
    /// if let Some(source) = AdpfThermalSource::new() {
    ///     pacer = pacer.with_thermal_source(Box::new(source));
    /// }
    /// // Drift-only on API < 30, ADPF-enhanced on API 30+.
    /// ```
    pub struct AdpfThermalSource {
        vtable: ThermalVtable,
        forecast_seconds: c_int,
        /// Consecutive NaN / negative returns from headroom. Once this
        /// reaches `HEADROOM_NAN_THRESHOLD` the source stops calling
        /// headroom and switches to the faster status-only cadence.
        consecutive_headroom_nan: u32,
    }

    impl AdpfThermalSource {
        /// Create an ADPF thermal source with the default 10-second forecast.
        ///
        /// Returns `None` if the NDK thermal API is unavailable (API < 30,
        /// emulator without thermal HAL, etc.).
        pub fn new() -> Option<Self> {
            Self::with_forecast(10)
        }

        /// Create an ADPF thermal source with a custom forecast horizon.
        ///
        /// `forecast_seconds` is clamped to 0-60 (the range documented by
        /// `PowerManager.getThermalHeadroom()`). Values outside this range
        /// are silently clamped rather than rejected, so a misconfigured
        /// caller still gets a working thermal source instead of silent
        /// degradation to drift-only.
        pub fn with_forecast(forecast_seconds: i32) -> Option<Self> {
            let vtable = unsafe { resolve_vtable()? };
            let clamped = forecast_seconds.clamp(0, super::MAX_FORECAST_SECONDS) as c_int;
            Some(Self {
                vtable,
                forecast_seconds: clamped,
                consecutive_headroom_nan: 0,
            })
        }

        /// Whether headroom queries are available and producing usable data.
        ///
        /// Returns `false` if the symbol was never resolved (API 30) or
        /// if the HAL has returned NaN for `HEADROOM_NAN_THRESHOLD`
        /// consecutive polls (broken HAL on API 31+).
        fn is_headroom_usable(&self) -> bool {
            self.vtable.get_headroom.is_some()
                && self.consecutive_headroom_nan < super::HEADROOM_NAN_THRESHOLD
        }
    }

    impl ThermalHintSource for AdpfThermalSource {
        fn poll(&mut self) -> Option<ThermalHint> {
            let status = unsafe { (self.vtable.get_status)(self.vtable.manager) };
            if status == ATHERMAL_STATUS_ERROR {
                return None;
            }

            // Only call headroom if the symbol exists AND it has been
            // returning usable data. After HEADROOM_NAN_THRESHOLD
            // consecutive NaN returns we stop calling it and fall back
            // to the faster status-only cadence.
            let raw_headroom = if self.is_headroom_usable() {
                self.vtable.get_headroom.map(|f| {
                    let h = unsafe { (f)(self.vtable.manager, self.forecast_seconds) };
                    if h.is_nan() || h < 0.0 {
                        self.consecutive_headroom_nan += 1;
                    } else {
                        self.consecutive_headroom_nan = 0;
                    }
                    h
                })
            } else {
                None
            };

            let headroom = raw_headroom.and_then(convert_headroom);
            let status_severity = status_to_severity(status);
            let hr_severity = raw_headroom.and_then(headroom_to_severity);

            let severity = match hr_severity {
                Some(hs) => worse_severity(status_severity, hs),
                None => status_severity,
            };

            // Always return Some when the API works — even for Nominal.
            // This overwrites stale Critical/Serious hints in the pacer
            // when the device cools down.
            Some(ThermalHint { severity, headroom })
        }

        fn min_poll_interval(&self) -> Duration {
            if self.is_headroom_usable() {
                // getThermalHeadroom must not be called more often
                // than every 10 seconds — returns NaN if violated.
                super::HEADROOM_POLL_INTERVAL
            } else {
                // Status-only path: either API 30 (no headroom symbol)
                // or API 31+ with a broken HAL (consecutive NaN).
                // getCurrentThermalStatus is cheap, no cadence restriction.
                super::STATUS_ONLY_POLL_INTERVAL
            }
        }
    }

    impl std::fmt::Debug for AdpfThermalSource {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("AdpfThermalSource")
                .field("forecast_seconds", &self.forecast_seconds)
                .field("headroom_usable", &self.is_headroom_usable())
                .finish()
        }
    }

    // --- dlsym resolution ---

    /// Resolve NDK thermal functions from `libandroid.so`.
    ///
    /// Returns `None` if the library can't be loaded, the base
    /// functions (API 30) aren't present, or the manager can't
    /// be acquired.
    unsafe fn resolve_vtable() -> Option<ThermalVtable> {
        let lib = dlopen(c"libandroid.so".as_ptr(), RTLD_NOW);
        if lib.is_null() {
            return None;
        }

        // Required: API 30+
        let acquire_ptr = dlsym(lib, c"AThermal_acquireManager".as_ptr());
        let release_ptr = dlsym(lib, c"AThermal_releaseManager".as_ptr());
        let status_ptr = dlsym(lib, c"AThermal_getCurrentThermalStatus".as_ptr());

        if acquire_ptr.is_null() || release_ptr.is_null() || status_ptr.is_null() {
            dlclose(lib);
            return None;
        }

        let acquire: AcquireManagerFn = std::mem::transmute(acquire_ptr);
        let release: ReleaseManagerFn = std::mem::transmute(release_ptr);
        let get_status: GetStatusFn = std::mem::transmute(status_ptr);

        let manager = acquire();
        if manager.is_null() {
            dlclose(lib);
            return None;
        }

        // Optional: API 31+
        let headroom_ptr = dlsym(lib, c"AThermal_getThermalHeadroom".as_ptr());
        let get_headroom: Option<GetHeadroomFn> = if headroom_ptr.is_null() {
            None
        } else {
            Some(std::mem::transmute(headroom_ptr))
        };

        Some(ThermalVtable {
            lib,
            manager,
            release_manager: release,
            get_status,
            get_headroom,
        })
    }
}

#[cfg(target_os = "android")]
pub use imp::AdpfThermalSource;

// ---------------------------------------------------------------------------
// Tests — mapping logic runs on all platforms during `cargo test`
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_mapping_covers_all_levels() {
        assert_eq!(status_to_severity(ATHERMAL_STATUS_NONE), ThermalSeverity::Nominal);
        assert_eq!(status_to_severity(ATHERMAL_STATUS_LIGHT), ThermalSeverity::Nominal);
        assert_eq!(status_to_severity(ATHERMAL_STATUS_MODERATE), ThermalSeverity::Fair);
        assert_eq!(status_to_severity(ATHERMAL_STATUS_SEVERE), ThermalSeverity::Serious);
        assert_eq!(status_to_severity(ATHERMAL_STATUS_CRITICAL), ThermalSeverity::Critical);
        assert_eq!(status_to_severity(ATHERMAL_STATUS_EMERGENCY), ThermalSeverity::Critical);
        assert_eq!(status_to_severity(ATHERMAL_STATUS_SHUTDOWN), ThermalSeverity::Critical);
    }

    #[test]
    fn status_mapping_handles_error_and_unknown() {
        // ATHERMAL_STATUS_ERROR (-1) falls through to nominal
        assert_eq!(status_to_severity(ATHERMAL_STATUS_ERROR), ThermalSeverity::Nominal);
        // Unknown negative
        assert_eq!(status_to_severity(-42), ThermalSeverity::Nominal);
        // Future values above SHUTDOWN should still map to Critical
        assert_eq!(status_to_severity(99), ThermalSeverity::Critical);
    }

    #[test]
    fn headroom_severity_thresholds() {
        // Cold — well below any threshold
        assert_eq!(headroom_to_severity(0.0), Some(ThermalSeverity::Nominal));
        assert_eq!(headroom_to_severity(0.49), Some(ThermalSeverity::Nominal));

        // Warming up
        assert_eq!(headroom_to_severity(0.5), Some(ThermalSeverity::Fair));
        assert_eq!(headroom_to_severity(0.69), Some(ThermalSeverity::Fair));

        // Approaching limit
        assert_eq!(headroom_to_severity(0.7), Some(ThermalSeverity::Serious));
        assert_eq!(headroom_to_severity(0.89), Some(ThermalSeverity::Serious));

        // At or past limit
        assert_eq!(headroom_to_severity(0.9), Some(ThermalSeverity::Critical));
        assert_eq!(headroom_to_severity(1.0), Some(ThermalSeverity::Critical));
        assert_eq!(headroom_to_severity(1.5), Some(ThermalSeverity::Critical));
    }

    #[test]
    fn headroom_severity_rejects_bad_values() {
        assert_eq!(headroom_to_severity(f32::NAN), None);
        assert_eq!(headroom_to_severity(-1.0), None);
        assert_eq!(headroom_to_severity(-0.001), None);
    }

    #[test]
    fn headroom_conversion_inverts_scale() {
        // ADPF 0.0 (cold) -> our 1.0 (full headroom)
        assert_eq!(convert_headroom(0.0), Some(1.0));
        // ADPF 1.0 (at limit) -> our 0.0 (no headroom)
        assert_eq!(convert_headroom(1.0), Some(0.0));
        // ADPF 0.7 -> our 0.3 (f32→f64 loses ~1e-8 precision)
        let h = convert_headroom(0.7).unwrap();
        assert!((h - 0.3).abs() < 1e-6);
        // ADPF > 1.0 (throttling) -> clamped to 0.0
        assert_eq!(convert_headroom(1.5), Some(0.0));
    }

    #[test]
    fn headroom_conversion_rejects_bad_values() {
        assert_eq!(convert_headroom(f32::NAN), None);
        assert_eq!(convert_headroom(-0.5), None);
    }

    #[test]
    fn worse_severity_picks_higher() {
        use ThermalSeverity::*;
        assert_eq!(worse_severity(Nominal, Fair), Fair);
        assert_eq!(worse_severity(Fair, Nominal), Fair);
        assert_eq!(worse_severity(Serious, Critical), Critical);
        assert_eq!(worse_severity(Critical, Nominal), Critical);
        assert_eq!(worse_severity(Serious, Serious), Serious);
    }

    #[test]
    fn combined_severity_takes_worst() {
        // Simulate the poll() logic: status says Fair, headroom says Serious
        let status_sev = status_to_severity(ATHERMAL_STATUS_MODERATE); // Fair
        let hr_sev = headroom_to_severity(0.75); // Serious
        let combined = worse_severity(status_sev, hr_sev.unwrap());
        assert_eq!(combined, ThermalSeverity::Serious);

        // Status says Critical, headroom says Fair
        let status_sev = status_to_severity(ATHERMAL_STATUS_CRITICAL);
        let hr_sev = headroom_to_severity(0.55);
        let combined = worse_severity(status_sev, hr_sev.unwrap());
        assert_eq!(combined, ThermalSeverity::Critical);
    }

    #[test]
    fn headroom_nan_falls_back_to_status_only() {
        // When headroom returns NaN (too frequent polling or unsupported),
        // severity should come from status alone
        let status_sev = status_to_severity(ATHERMAL_STATUS_SEVERE); // Serious
        let hr_sev = headroom_to_severity(f32::NAN); // None
        let severity = match hr_sev {
            Some(hs) => worse_severity(status_sev, hs),
            None => status_sev,
        };
        assert_eq!(severity, ThermalSeverity::Serious);
    }

    #[test]
    fn poll_interval_is_shorter_for_status_only() {
        // API 30 (status only): getCurrentThermalStatus has no cadence
        // restriction, so we poll every 1 second for fast reaction.
        // API 31+ (headroom): getThermalHeadroom returns NaN if polled
        // more often than every 10 seconds.
        assert_eq!(STATUS_ONLY_POLL_INTERVAL, Duration::from_secs(1));
        assert_eq!(HEADROOM_POLL_INTERVAL, Duration::from_secs(10));
        assert!(
            STATUS_ONLY_POLL_INTERVAL < HEADROOM_POLL_INTERVAL,
            "status-only devices should poll more frequently"
        );
    }

    #[test]
    fn headroom_nan_threshold_is_reasonable() {
        // 3 consecutive NaN returns before degradation — enough to
        // filter transient errors but fast enough to detect a broken HAL
        // within ~30 seconds at 10 s cadence.
        assert_eq!(HEADROOM_NAN_THRESHOLD, 3);
    }

    #[test]
    fn forecast_clamp_range() {
        // Matches PowerManager.getThermalHeadroom() documented range
        assert_eq!(MAX_FORECAST_SECONDS, 60);
    }
}
