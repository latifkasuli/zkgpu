//! Apple thermal-state hint source.
//!
//! Uses `NSProcessInfo.thermalState` via the Objective-C runtime to
//! provide thermal severity to the drift pacer on macOS and iOS.
//!
//! # Availability
//!
//! | Platform | Minimum version                    |
//! |----------|------------------------------------|
//! | macOS    | 10.10.3 (`thermalState` property)  |
//! | iOS      | 11.0                               |
//!
//! # Headroom
//!
//! Apple does not expose a continuous headroom value — only the
//! discrete four-level `thermalState` property. The source always
//! returns `headroom: None`.
//!
//! # Poll cadence
//!
//! `thermalState` is a cheap property read (no IPC, no kernel call).
//! The OS updates it based on `NSProcessInfoThermalStateDidChange`
//! notifications, so there is no penalty for reading it frequently.
//! We use a 1-second poll interval — responsive without wasting cycles.

use std::time::Duration;

use crate::observation::ThermalSeverity;

// ---------------------------------------------------------------------------
// Apple thermalState constants
// ---------------------------------------------------------------------------

/// `NSProcessInfoThermalStateNominal` (0)
const THERMAL_STATE_NOMINAL: isize = 0;
/// `NSProcessInfoThermalStateFair` (1)
const THERMAL_STATE_FAIR: isize = 1;
/// `NSProcessInfoThermalStateSerious` (2)
const THERMAL_STATE_SERIOUS: isize = 2;
/// `NSProcessInfoThermalStateCritical` (3)
const THERMAL_STATE_CRITICAL: isize = 3;

/// Poll interval for `thermalState`.
///
/// This is a cheap property read — the OS caches the current state
/// in-process and updates it via notification dispatch. 1 second
/// gives fast reaction without busy-polling.
const POLL_INTERVAL: Duration = Duration::from_secs(1);

// ---------------------------------------------------------------------------
// Pure mapping helper (no FFI, testable on all platforms)
// ---------------------------------------------------------------------------

/// Map Apple `NSProcessInfoThermalState` value to our severity level.
///
/// The four Apple states map 1:1 to our four severity levels:
/// - 0 (Nominal) → `ThermalSeverity::Nominal`
/// - 1 (Fair)     → `ThermalSeverity::Fair`
/// - 2 (Serious)  → `ThermalSeverity::Serious`
/// - 3 (Critical) → `ThermalSeverity::Critical`
///
/// Unknown values ≥ Critical are treated as Critical (future Apple
/// additions above Critical are more likely to be escalations).
/// Unknown negative values are treated as Nominal (defensive).
fn thermal_state_to_severity(state: isize) -> ThermalSeverity {
    match state {
        THERMAL_STATE_NOMINAL => ThermalSeverity::Nominal,
        THERMAL_STATE_FAIR => ThermalSeverity::Fair,
        THERMAL_STATE_SERIOUS => ThermalSeverity::Serious,
        s if s >= THERMAL_STATE_CRITICAL => ThermalSeverity::Critical,
        // Negative or unexpected — defensive fallback
        _ => ThermalSeverity::Nominal,
    }
}

// ---------------------------------------------------------------------------
// Apple FFI implementation (only compiled on Apple platforms)
// ---------------------------------------------------------------------------

#[cfg(target_vendor = "apple")]
mod imp {
    use std::ffi::c_void;
    use std::time::Duration;

    use crate::observation::{ThermalHint, ThermalHintSource};

    use super::thermal_state_to_severity;

    // --- Objective-C runtime types ---

    /// Opaque type for Objective-C class pointers.
    #[repr(C)]
    struct ObjcClass {
        _private: [u8; 0],
    }

    /// Opaque type for Objective-C selector pointers.
    #[repr(C)]
    struct ObjcSel {
        _private: [u8; 0],
    }

    // --- Objective-C runtime functions ---

    // Link Foundation so `objc_getClass("NSProcessInfo")` can find
    // the class. Without this, the class isn't loaded into the runtime.
    #[link(name = "Foundation", kind = "framework")]
    extern "C" {}

    #[link(name = "objc", kind = "dylib")]
    extern "C" {
        fn objc_getClass(name: *const u8) -> *const ObjcClass;
        fn sel_registerName(name: *const u8) -> *const ObjcSel;
        fn objc_msgSend(receiver: *const c_void, sel: *const ObjcSel, ...) -> *const c_void;
    }

    /// Cached Objective-C pointers for `[NSProcessInfo processInfo]`
    /// and the `thermalState` selector.
    struct ObjcVtable {
        /// The `[NSProcessInfo processInfo]` singleton. This never
        /// changes for the lifetime of the process — safe to cache.
        process_info: *const c_void,
        /// Selector for `thermalState` (returns `NSInteger`).
        thermal_state_sel: *const ObjcSel,
    }

    // SAFETY: NSProcessInfo is documented as thread-safe. The cached
    // process_info pointer is an immutable singleton. The selector is
    // a global interned pointer.
    unsafe impl Send for ObjcVtable {}

    /// Apple thermal hint source.
    ///
    /// Wraps `NSProcessInfo.thermalState` to provide thermal severity
    /// to [`DriftPacer`](crate::DriftPacer).
    ///
    /// # Example
    ///
    /// ```ignore
    /// use zkgpu_pacer::{DriftPacer, DriftPacerConfig};
    /// use zkgpu_pacer::AppleThermalSource;
    ///
    /// let mut pacer = DriftPacer::new(DriftPacerConfig::default());
    /// if let Some(source) = AppleThermalSource::new() {
    ///     pacer = pacer.with_thermal_source(Box::new(source));
    /// }
    /// // Drift-only if NSProcessInfo unavailable, thermal-enhanced otherwise.
    /// ```
    pub struct AppleThermalSource {
        vtable: ObjcVtable,
    }

    impl AppleThermalSource {
        /// Create an Apple thermal source.
        ///
        /// Returns `None` if `NSProcessInfo` or the `thermalState`
        /// selector can't be resolved (should not happen on any
        /// supported Apple platform, but we stay defensive).
        pub fn new() -> Option<Self> {
            let vtable = unsafe { resolve_vtable()? };
            Some(Self { vtable })
        }
    }

    impl ThermalHintSource for AppleThermalSource {
        fn poll(&mut self) -> Option<ThermalHint> {
            let state = unsafe {
                objc_msgSend(
                    self.vtable.process_info,
                    self.vtable.thermal_state_sel,
                ) as isize
            };
            let severity = thermal_state_to_severity(state);
            Some(ThermalHint {
                severity,
                headroom: None,
            })
        }

        fn min_poll_interval(&self) -> Duration {
            super::POLL_INTERVAL
        }
    }

    impl std::fmt::Debug for AppleThermalSource {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("AppleThermalSource").finish()
        }
    }

    // --- Objective-C resolution ---

    /// Resolve `[NSProcessInfo processInfo]` and `thermalState` selector.
    ///
    /// Returns `None` if the class or selector can't be found.
    unsafe fn resolve_vtable() -> Option<ObjcVtable> {
        let class = objc_getClass(b"NSProcessInfo\0".as_ptr());
        if class.is_null() {
            return None;
        }

        let process_info_sel = sel_registerName(b"processInfo\0".as_ptr());
        if process_info_sel.is_null() {
            return None;
        }

        let thermal_state_sel = sel_registerName(b"thermalState\0".as_ptr());
        if thermal_state_sel.is_null() {
            return None;
        }

        // [NSProcessInfo processInfo] — returns the shared singleton
        let process_info = objc_msgSend(class as *const c_void, process_info_sel);
        if process_info.is_null() {
            return None;
        }

        Some(ObjcVtable {
            process_info,
            thermal_state_sel,
        })
    }
}

#[cfg(target_vendor = "apple")]
pub use imp::AppleThermalSource;

// ---------------------------------------------------------------------------
// Tests — mapping logic runs on all platforms during `cargo test`
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_mapping_covers_all_levels() {
        assert_eq!(
            thermal_state_to_severity(THERMAL_STATE_NOMINAL),
            ThermalSeverity::Nominal
        );
        assert_eq!(
            thermal_state_to_severity(THERMAL_STATE_FAIR),
            ThermalSeverity::Fair
        );
        assert_eq!(
            thermal_state_to_severity(THERMAL_STATE_SERIOUS),
            ThermalSeverity::Serious
        );
        assert_eq!(
            thermal_state_to_severity(THERMAL_STATE_CRITICAL),
            ThermalSeverity::Critical
        );
    }

    #[test]
    fn state_mapping_future_values_are_critical() {
        // Future Apple additions above Critical should still map to Critical
        assert_eq!(thermal_state_to_severity(4), ThermalSeverity::Critical);
        assert_eq!(thermal_state_to_severity(99), ThermalSeverity::Critical);
    }

    #[test]
    fn state_mapping_negative_values_are_nominal() {
        // Defensive: unexpected negative values fall to Nominal
        assert_eq!(thermal_state_to_severity(-1), ThermalSeverity::Nominal);
        assert_eq!(thermal_state_to_severity(-42), ThermalSeverity::Nominal);
    }

    #[test]
    fn poll_interval_is_one_second() {
        assert_eq!(POLL_INTERVAL, Duration::from_secs(1));
    }

    #[test]
    fn no_headroom_in_apple_api() {
        // Apple thermalState is discrete — no continuous headroom value.
        // Verify the mapping helper only produces severity, and any
        // ThermalHint from this source would have headroom: None.
        // (The actual poll() returns headroom: None — this test
        // validates the design constraint at the mapping level.)
        let severity = thermal_state_to_severity(THERMAL_STATE_SERIOUS);
        assert_eq!(severity, ThermalSeverity::Serious);
        // Headroom is always None — confirmed by construction in poll().
    }

    // Integration test — only runs on Apple platforms where the
    // Objective-C runtime is available and we can actually call
    // NSProcessInfo.thermalState.
    #[cfg(target_vendor = "apple")]
    mod integration {
        use crate::observation::{ThermalHintSource, ThermalSeverity};

        use super::super::imp::AppleThermalSource;

        #[test]
        fn can_create_source() {
            let source = AppleThermalSource::new();
            assert!(
                source.is_some(),
                "AppleThermalSource::new() should succeed on Apple platforms"
            );
        }

        #[test]
        fn poll_returns_valid_hint() {
            let mut source = AppleThermalSource::new().unwrap();
            let hint = source.poll();
            assert!(hint.is_some(), "poll() should always return Some on Apple");
            let hint = hint.unwrap();
            // On a dev machine at idle, we expect Nominal — but any
            // valid severity is acceptable.
            assert!(
                matches!(
                    hint.severity,
                    ThermalSeverity::Nominal
                        | ThermalSeverity::Fair
                        | ThermalSeverity::Serious
                        | ThermalSeverity::Critical
                ),
                "severity should be a valid variant"
            );
            // Apple provides no headroom
            assert_eq!(hint.headroom, None);
        }

        #[test]
        fn poll_interval_matches_constant() {
            let source = AppleThermalSource::new().unwrap();
            assert_eq!(
                source.min_poll_interval(),
                std::time::Duration::from_secs(1)
            );
        }
    }
}
