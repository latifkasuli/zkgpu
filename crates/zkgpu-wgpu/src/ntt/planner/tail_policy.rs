//! Stockham tail-phase strategy.
//!
//! The Stockham hybrid plan splits a transform of size 2^log_n into:
//!   - A run of `log_n - LOG_BLOCK` global DIF stages.
//!   - A "tail" of the final `LOG_BLOCK` stages.
//!
//! Historically the tail always ran as a single workgroup-local fused R4
//! dispatch (`babybear_stockham_local_r4.wgsl`). The companion `GlobalOnlyR4`
//! strategy extends the global R4 dispatch chain through end-of-transform.
//!
//! ## Policy origin
//!
//! The original Xclipse 540 / Exynos 2200 measurement in
//! `research/stockham-local-fused-rewrite.md` showed a 4.1× ns/elem
//! regression for the local kernel by log22 — the working-set / L2 collapse
//! the strided gather (`stride = N / BLOCK_SIZE`) produces. PR 1 of the
//! tail-strategy refactor encoded that as
//! `Mali @ log_n ≥ 22 → GlobalOnlyR4` and `Xclipse @ log_n ≥ 20 → GlobalOnlyR4`.
//!
//! Phase-e (2026-04-15, 6-device FTL A/B —
//! `apps/android-harness/research/benchmarks/phase-e-tail-ab-2026-04-15/`)
//! falsified both rules:
//!   - Mali @ log22 across G715/G720 reduced to ±1% noise; the predicted
//!     collapse did not reproduce on current silicon / drivers.
//!   - Xclipse 940 (Exynos 2400) showed no collapse; the original
//!     measurement was Xclipse-540-specific.
//!
//! Phase-e *did* surface a separate small-N opportunity on Mali
//! (LocalFusedR4 launch+shmem-setup overhead dominating at log18-19, where
//! GlobalOnlyR4 wins by 30-50%) but with one mixed-signal device
//! (comet/Pixel 9 Pro Fold). Adding a small-N flip is deferred — see the
//! phase-e README's "Recommended planner change" section for the deferred
//! work.
//!
//! ## Current policy (PR 2 close-out)
//!
//! Native Vulkan: keep the legacy `LocalFusedR4` everywhere. The two
//! large-N flips that PR 1 inherited from the Xclipse 540 measurement are
//! gone. Browser stays at `log_n ≥ 20 → GlobalOnlyR4` — the sandbox hides
//! the underlying silicon and we have no field data to drop the rule.
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
///
/// `HeuristicXclipseLargeN` and `HeuristicMaliLargeN` were dropped in the
/// PR 2 close-out (2026-04-15). Phase-e A/B did not reproduce the collapses
/// those variants flipped on; their absence in current logs is intentional.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StockhamTailReason {
    /// BrowserWebGpu at log_n ≥ 20 — conservative; the browser hides the
    /// underlying silicon and we have no field data to drop this rule.
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

    // Native Vulkan: phase-e A/B (2026-04-15) falsified the two family-specific
    // large-N flips PR 1 inherited from the Xclipse-540 scaling table. Current
    // silicon (Xclipse 940, Mali-G715/G720) shows no collapse at log22 worth
    // flipping for. See:
    // `apps/android-harness/research/benchmarks/phase-e-tail-ab-2026-04-15/README.md`.
    //
    // The phase-e data did surface a small-N Mali win (log18-19, GlobalOnlyR4
    // +30-50%) but with one mixed-signal device (comet). Adding that rule is
    // deferred until the anomaly is explained.
    let _ = caps.gpu_family; // reserved for future family-driven rules.

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
    fn xclipse_keeps_local_at_all_sizes() {
        // PR 2 close-out (2026-04-15): the Xclipse log20 flip inherited from
        // the Xclipse-540 measurement was falsified by phase-e A/B on
        // Xclipse 940 (e1q/Galaxy S24). Large-N keeps LocalFusedR4.
        let caps = Some(vulkan(GpuFamily::Xclipse));
        for log_n in [18, 19, 20, 21, 22, 24] {
            let d = choose_stockham_tail(log_n, caps, StockhamTailOverride::Auto).unwrap();
            assert_eq!(
                d.strategy,
                StockhamTailStrategy::LocalFusedR4,
                "Xclipse should keep local at log_n={log_n} after PR 2",
            );
            assert_eq!(d.reason, StockhamTailReason::HeuristicDefaultLocal);
        }
    }

    #[test]
    fn mali_keeps_local_at_all_sizes() {
        // PR 2 close-out (2026-04-15): Mali @ log22 flip was falsified by
        // phase-e A/B across G715/G720 — the collapse did not reproduce.
        // A small-N Mali opportunity (log18-19) was surfaced but is deferred
        // (see phase-e README), so the current heuristic keeps local for all
        // log_n.
        let caps = Some(vulkan(GpuFamily::Mali));
        for log_n in [18, 19, 20, 21, 22, 24] {
            let d = choose_stockham_tail(log_n, caps, StockhamTailOverride::Auto).unwrap();
            assert_eq!(
                d.strategy,
                StockhamTailStrategy::LocalFusedR4,
                "Mali should keep local at log_n={log_n} after PR 2",
            );
            assert_eq!(d.reason, StockhamTailReason::HeuristicDefaultLocal);
        }
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
    fn forced_local_overrides_browser_heuristic() {
        // Browser @ log20 would default to GlobalOnlyR4 via
        // `HeuristicBrowserConservative`; ForcedLocal must win.
        let caps = Some(browser());
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
