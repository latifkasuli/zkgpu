//! Stockham tail-phase strategy.
//!
//! The Stockham hybrid plan splits a transform of size 2^log_n into:
//!   - A run of `log_n - LOG_BLOCK` global DIF stages.
//!   - A "tail" of the final `LOG_BLOCK` stages.
//!
//! Historically the tail always ran as a single workgroup-local fused R4
//! dispatch (`babybear_stockham_local_r4.wgsl`). On Mali-G715 and Xclipse 540
//! the local kernel's per-thread strided gather (`stride = N / BLOCK_SIZE`)
//! collapses coalescing once the working set exceeds L2:
//!
//! | log N | Xclipse 540 ns/elem | ratio/prev |
//! |-------|---------------------|-----------|
//! | 18    |  4.77               |   —       |
//! | 22    | 19.64               |  4.1×     |
//!
//! See `research/stockham-local-fused-rewrite.md` for the full diagnosis.
//!
//! This module exposes a small policy that picks one of two tail strategies
//! per (log_n, device-family, override) and reports *why* — the reason
//! string is plumbed into `CaseReport.stockham_tail_reason` for after-the-fact
//! audit when comparing benchmark runs.
//!
//! Pure data only — no GPU calls — so the heuristic is unit-testable.

use crate::caps::GpuFamily;

use super::constants::LOG_BLOCK;

/// Which kernel runs the final LOG_BLOCK stages of a Stockham plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StockhamTailStrategy {
    /// Single workgroup-local fused R4 dispatch in shared memory.
    /// Coalesced within a workgroup only when stride is small.
    LocalFusedR4,
    /// Extend the global R4 dispatch chain through end-of-transform.
    /// No local dispatch. Pays five extra global passes; wins whenever
    /// the local kernel's strided gather stops coalescing.
    GlobalOnlyR4,
}

impl StockhamTailStrategy {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::LocalFusedR4 => "LocalFusedR4",
            Self::GlobalOnlyR4 => "GlobalOnlyR4",
        }
    }
}

/// Why a tail strategy was chosen. Reported in `CaseReport.stockham_tail_reason`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StockhamTailReason {
    /// Xclipse at log_n ≥ 20 — local fused collapses (see scaling table).
    HeuristicXclipseLargeN,
    /// Mali at log_n ≥ 22 — local fused collapses one stage later than Xclipse.
    HeuristicMaliLargeN,
    /// BrowserWebGpu at log_n ≥ 20 — conservative; the browser hides the
    /// underlying silicon and we already know mobile-class devices behind
    /// the sandbox suffer the same gather pathology.
    HeuristicBrowserConservative,
    /// Default — local fused remains fastest at all tested sizes for this
    /// (backend, family) combination, or no caps were available.
    HeuristicDefaultLocal,
    /// Caller-forced via `PlannerPolicy::with_stockham_tail_override(Local)`
    /// or env var `ZKGPU_STOCKHAM_TAIL=local`.
    ForcedLocal,
    /// Caller-forced via `PlannerPolicy::with_stockham_tail_override(Global)`
    /// or env var `ZKGPU_STOCKHAM_TAIL=global`.
    ForcedGlobal,
}

impl StockhamTailReason {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::HeuristicXclipseLargeN => "HeuristicXclipseLargeN",
            Self::HeuristicMaliLargeN => "HeuristicMaliLargeN",
            Self::HeuristicBrowserConservative => "HeuristicBrowserConservative",
            Self::HeuristicDefaultLocal => "HeuristicDefaultLocal",
            Self::ForcedLocal => "ForcedLocal",
            Self::ForcedGlobal => "ForcedGlobal",
        }
    }
}

/// Bundled (strategy, reason). Returned by [`choose_stockham_tail`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TailDecision {
    pub strategy: StockhamTailStrategy,
    pub reason: StockhamTailReason,
}

/// Caller-supplied override — applied on top of the heuristic default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StockhamTailOverride {
    /// Use the heuristic default for this (log_n, backend, family).
    Auto,
    /// Force `LocalFusedR4` regardless of heuristic.
    Local,
    /// Force `GlobalOnlyR4` regardless of heuristic.
    Global,
}

impl Default for StockhamTailOverride {
    fn default() -> Self {
        Self::Auto
    }
}

/// A minimal capability summary for tail-policy decisions.
///
/// Kept separate from `CapabilityProfile` so this module stays free of
/// `wgpu::Adapter` introspection and is fully constructible in unit tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TailCapsHint {
    pub backend: wgpu::Backend,
    pub gpu_family: GpuFamily,
}

/// Decide the tail strategy for a Stockham plan of size `2^log_n`.
///
/// Returns `None` for `log_n < LOG_BLOCK`: there is no tail phase — the
/// whole transform fits in `log_n` global dispatches.
///
/// `caps` may be `None` for unit-test contexts that don't have a real device;
/// the heuristic falls back to `LocalFusedR4` (preserving legacy behavior).
pub(crate) fn choose_stockham_tail(
    log_n: u32,
    caps: Option<TailCapsHint>,
    override_: StockhamTailOverride,
) -> Option<TailDecision> {
    if log_n < LOG_BLOCK {
        return None;
    }

    // Compute the heuristic *first* so we always have a default reason on
    // record, even when an override fires.
    let (heuristic_strategy, heuristic_reason) = heuristic_default(log_n, caps);

    let decision = match override_ {
        StockhamTailOverride::Auto => TailDecision {
            strategy: heuristic_strategy,
            reason: heuristic_reason,
        },
        StockhamTailOverride::Local => TailDecision {
            strategy: StockhamTailStrategy::LocalFusedR4,
            reason: StockhamTailReason::ForcedLocal,
        },
        StockhamTailOverride::Global => TailDecision {
            strategy: StockhamTailStrategy::GlobalOnlyR4,
            reason: StockhamTailReason::ForcedGlobal,
        },
    };

    Some(decision)
}

fn heuristic_default(
    log_n: u32,
    caps: Option<TailCapsHint>,
) -> (StockhamTailStrategy, StockhamTailReason) {
    let Some(caps) = caps else {
        return (
            StockhamTailStrategy::LocalFusedR4,
            StockhamTailReason::HeuristicDefaultLocal,
        );
    };

    // Browser sandbox: assume the worst-case mobile-class hardware behind
    // the API. The Chrome/Android Vulkan backend has reproduced the same
    // gather collapse behind WebGPU.
    if matches!(caps.backend, wgpu::Backend::BrowserWebGpu) && log_n >= 20 {
        return (
            StockhamTailStrategy::GlobalOnlyR4,
            StockhamTailReason::HeuristicBrowserConservative,
        );
    }

    // Native Vulkan: family-driven. Other backends (Metal, DX12) keep the
    // local kernel — Apple shmem is large and we have no Metal regression.
    if matches!(caps.backend, wgpu::Backend::Vulkan) {
        match caps.gpu_family {
            // Xclipse 540: 4.1× ns/elem regression by log22; flips at log20.
            GpuFamily::Xclipse if log_n >= 20 => {
                return (
                    StockhamTailStrategy::GlobalOnlyR4,
                    StockhamTailReason::HeuristicXclipseLargeN,
                );
            }
            // Mali-G715: holds through log21, collapses at log22.
            GpuFamily::Mali if log_n >= 22 => {
                return (
                    StockhamTailStrategy::GlobalOnlyR4,
                    StockhamTailReason::HeuristicMaliLargeN,
                );
            }
            _ => {}
        }
    }

    (
        StockhamTailStrategy::LocalFusedR4,
        StockhamTailReason::HeuristicDefaultLocal,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vulkan(family: GpuFamily) -> TailCapsHint {
        TailCapsHint {
            backend: wgpu::Backend::Vulkan,
            gpu_family: family,
        }
    }

    fn metal() -> TailCapsHint {
        TailCapsHint {
            backend: wgpu::Backend::Metal,
            gpu_family: GpuFamily::Apple,
        }
    }

    fn browser() -> TailCapsHint {
        TailCapsHint {
            backend: wgpu::Backend::BrowserWebGpu,
            gpu_family: GpuFamily::Unknown,
        }
    }

    // -- log_n < LOG_BLOCK: no tail phase --

    #[test]
    fn no_tail_below_log_block() {
        assert!(choose_stockham_tail(1, None, StockhamTailOverride::Auto).is_none());
        assert!(choose_stockham_tail(9, Some(metal()), StockhamTailOverride::Auto).is_none());
        // Override does not invent a tail when there is none.
        assert!(choose_stockham_tail(9, Some(vulkan(GpuFamily::Mali)), StockhamTailOverride::Global).is_none());
    }

    // -- Heuristic defaults --

    #[test]
    fn xclipse_flips_at_log20() {
        let caps = Some(vulkan(GpuFamily::Xclipse));
        let d19 = choose_stockham_tail(19, caps, StockhamTailOverride::Auto).unwrap();
        assert_eq!(d19.strategy, StockhamTailStrategy::LocalFusedR4);
        assert_eq!(d19.reason, StockhamTailReason::HeuristicDefaultLocal);

        let d20 = choose_stockham_tail(20, caps, StockhamTailOverride::Auto).unwrap();
        assert_eq!(d20.strategy, StockhamTailStrategy::GlobalOnlyR4);
        assert_eq!(d20.reason, StockhamTailReason::HeuristicXclipseLargeN);

        let d22 = choose_stockham_tail(22, caps, StockhamTailOverride::Auto).unwrap();
        assert_eq!(d22.strategy, StockhamTailStrategy::GlobalOnlyR4);
    }

    #[test]
    fn mali_flips_at_log22() {
        let caps = Some(vulkan(GpuFamily::Mali));
        let d21 = choose_stockham_tail(21, caps, StockhamTailOverride::Auto).unwrap();
        assert_eq!(d21.strategy, StockhamTailStrategy::LocalFusedR4);

        let d22 = choose_stockham_tail(22, caps, StockhamTailOverride::Auto).unwrap();
        assert_eq!(d22.strategy, StockhamTailStrategy::GlobalOnlyR4);
        assert_eq!(d22.reason, StockhamTailReason::HeuristicMaliLargeN);
    }

    #[test]
    fn browser_flips_at_log20() {
        let caps = Some(browser());
        let d19 = choose_stockham_tail(19, caps, StockhamTailOverride::Auto).unwrap();
        assert_eq!(d19.strategy, StockhamTailStrategy::LocalFusedR4);

        let d20 = choose_stockham_tail(20, caps, StockhamTailOverride::Auto).unwrap();
        assert_eq!(d20.strategy, StockhamTailStrategy::GlobalOnlyR4);
        assert_eq!(d20.reason, StockhamTailReason::HeuristicBrowserConservative);
    }

    #[test]
    fn metal_keeps_local_at_all_sizes() {
        let caps = Some(metal());
        for log_n in [10, 18, 22, 24] {
            let d = choose_stockham_tail(log_n, caps, StockhamTailOverride::Auto).unwrap();
            assert_eq!(
                d.strategy,
                StockhamTailStrategy::LocalFusedR4,
                "Metal/Apple should keep local at log_n={log_n}"
            );
            assert_eq!(d.reason, StockhamTailReason::HeuristicDefaultLocal);
        }
    }

    #[test]
    fn adreno_keeps_local_at_all_sizes() {
        // Adreno's strided gather behaves on Android — four-step kicks in at
        // mobile threshold instead. Tail policy keeps local for whatever
        // Stockham work remains.
        let caps = Some(vulkan(GpuFamily::Adreno));
        for log_n in [10, 18, 22] {
            let d = choose_stockham_tail(log_n, caps, StockhamTailOverride::Auto).unwrap();
            assert_eq!(d.strategy, StockhamTailStrategy::LocalFusedR4);
        }
    }

    #[test]
    fn no_caps_falls_back_to_local() {
        let d = choose_stockham_tail(22, None, StockhamTailOverride::Auto).unwrap();
        assert_eq!(d.strategy, StockhamTailStrategy::LocalFusedR4);
        assert_eq!(d.reason, StockhamTailReason::HeuristicDefaultLocal);
    }

    // -- Overrides win over heuristic --

    #[test]
    fn forced_local_overrides_xclipse_heuristic() {
        let caps = Some(vulkan(GpuFamily::Xclipse));
        let d = choose_stockham_tail(22, caps, StockhamTailOverride::Local).unwrap();
        assert_eq!(d.strategy, StockhamTailStrategy::LocalFusedR4);
        assert_eq!(d.reason, StockhamTailReason::ForcedLocal);
    }

    #[test]
    fn forced_global_overrides_apple_heuristic() {
        let caps = Some(metal());
        let d = choose_stockham_tail(22, caps, StockhamTailOverride::Global).unwrap();
        assert_eq!(d.strategy, StockhamTailStrategy::GlobalOnlyR4);
        assert_eq!(d.reason, StockhamTailReason::ForcedGlobal);
    }
}
