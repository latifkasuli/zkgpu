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
//! ## Adreno collapse (PR 3, 2026-04-15)
//!
//! The Adreno generation-confirmation A/B
//! (`apps/android-harness/research/benchmarks/adreno-gen-confirm-2026-04-15/`)
//! discovered that three successive Adreno generations — 730 (b0q), 740
//! (dm3q), 750 (e3q) — reproduce the Xclipse-540 strided-gather collapse
//! pattern at every measured size (log18..=log22, +40% to +72% Global
//! wins, both directions). The 4th generation measured — Adreno 830
//! (pa3q) — was flatline in phase-e (±5% noise). The local S24 Ultra
//! (Adreno 750) run had already shown the production default picking a
//! ~3× slower path at log22 (23ms LocalFusedR4 vs 8ms GlobalOnlyR4).
//!
//! The collapse is wholesale (not log_n-gated) on the three older
//! generations, so the rule is unconditional on `GpuFamily::Adreno`. The
//! marginal pa3q regression is within phase-e's measured noise band and
//! acceptable against 40-70% wins on the older silicon.
//!
//! ## Mali collapse (G.1.4, 2026-04-16)
//!
//! G.1.1 (`apps/android-harness/research/benchmarks/mali-scope-match-2026-04-16/`)
//! re-ran phase-e's exact narrow forced-A/B scope on the same 3× Mali-G715
//! device cohort (husky/Pixel 8 Pro, komodo/Pixel 9 Pro XL, comet/Pixel 9
//! Pro Fold). Unlike phase-e's windowed shape, G.1.1 measured +50-85%
//! GlobalOnlyR4 wins across log 18..=22 on all 3 devices, forward and
//! inverse — matching phase-f's full-suite shape despite the cold device
//! and narrow scope. This falsified the "phase-f warm-up amplified the
//! win" thermal hypothesis: the signal is microarchitecture, not
//! thermal/test-order.
//!
//! G.1.3 (`apps/android-harness/research/benchmarks/mali-older-gen-2026-04-16/`)
//! extended coverage to two older Mali generations: oriole (Pixel 6,
//! Mali-G78 MP20, Tensor G1, Mali driver r38p1) and panther (Pixel 7,
//! Mali-G710 MC7, Tensor G2, also r38p1). Both devices reproduced the
//! unconditional shape with identical recommendation (`UNCONDITIONAL @
//! log21`, Global win +33-88%). Combined 5-device cohort spans 4 silicon
//! generations, 3 SoC generations, and 2 Mali driver major revisions
//! (r38 → r51). Weakest cell in the 50-cell combined matrix was panther
//! log 20 inverse at +33.0%, still well above the 20% decision-gate
//! threshold.
//!
//! Like Adreno, the collapse has no log_n floor within measurable range —
//! smallest test (log 18) shows +85-92% wins — so the rule is unconditional
//! once a tail phase exists. The capability-gate side finding: absolute
//! GlobalR4 timing at log 22 is within 1.2× of flagship G715 even on the
//! 2021 Tensor G1; all five devices are viable zk-proving targets.
//!
//! ## Xclipse collapse (G.2.3, 2026-04-16)
//!
//! G.2.2
//! (`apps/android-harness/research/benchmarks/browserstack-xclipse-cohort-2026-04-16/`)
//! sidestepped the FTL Exynos-pinning blocker via BrowserStack App Automate,
//! which exposes Exynos-pinned Samsung Galaxy axes (no region-split
//! lottery). The full `ZkgpuInstrumentedTest` class ran in parallel on
//! Galaxy S22 + S22 Ultra (Exynos 2200, Xclipse 920 — first-gen RDNA2
//! Xclipse, 2022), Galaxy S24 (Exynos 2400, Xclipse 940, 2024), and Galaxy
//! S26 (Exynos 2600, Xclipse 960, 2026). All four Xclipse devices reported
//! `UNCONDITIONAL @ log21` from `zkgpu-tail-analyze`: 38/40 cells
//! `global-big` (≥20% Global win), 2/40 cells `global-narrow` (S22
//! log21inv +17.8%, S24 log22inv +18.7%) — every cell a Global win, just
//! two slightly below the +20% "big" classification. Three driver major
//! revisions across the cohort (`a927fb4` / `3b10981` / `24.0.x` /
//! `25.2.x`) all show identical shape, reinforcing that the signal is
//! microarchitecture-level, not a driver regression.
//!
//! Cross-vendor confirmation: the same-day G.1.4 FTL lottery rolled e2s
//! (Galaxy S24+, Exynos 2400 variant) to Xclipse 940 and independently
//! measured `UNCONDITIONAL @ log21` (weakest cell log 22 inv +27.0%),
//! within measurement noise of the BrowserStack S24 result.
//!
//! As with Adreno and Mali, the collapse has no log_n floor within
//! measurable range — smallest test (log 18) shows +62-84% wins on every
//! Xclipse generation — so the rule is unconditional once a tail phase
//! exists. Xclipse 540 (Exynos 1580, Galaxy A56) is still queued on FTL
//! at this writing but not gating the rule decision: 3 Xclipse generations
//! × 4 devices × 40 cells exceeds the G.1.4 Mali evidence bar. If Xclipse
//! 540 fails to reproduce when it dequeues, we'd narrow the rule; absent
//! evidence to the contrary, unconditional stands.
//!
//! Absolute performance: Xclipse 960 at 9.16 ms log22 fwd GlobalR4
//! competes with Adreno 830 (6 ms) and Mali-G715 (15 ms); Xclipse 920 at
//! 14–17 ms peers with 2022-cohort Mali/Adreno. All 3 measured gens pass
//! the ≤17 ms capability gate established in G.1.3.
//!
//! ## Current policy
//!
//! * **Browser (any family):** `log_n ≥ 20 → GlobalOnlyR4` — conservative;
//!   the sandbox hides the silicon and we have no field data to drop it.
//! * **Native Adreno:** unconditional `GlobalOnlyR4` once a tail phase
//!   exists (`log_n ≥ LOG_BLOCK`). Reason: `HeuristicAdrenoCollapse`.
//!   Covers 730/740/750/830/840.
//! * **Native Mali:** unconditional `GlobalOnlyR4` once a tail phase
//!   exists. Reason: `HeuristicMaliCollapse`. Covers G78/G710/G715 and
//!   forward-extrapolates to G720/G725 (Valhall/Bifrost family, same
//!   gather-collapse pathology).
//! * **Native Xclipse:** unconditional `GlobalOnlyR4` once a tail phase
//!   exists. Reason: `HeuristicXclipseCollapse`. Covers 920 (RDNA2) /
//!   940 (RDNA3) / 960 (RDNA3+) measured, forward-extrapolates to 540
//!   (pending a56x FTL dequeue) and future Exynos RDNA variants.
//! * **Native Apple / everything else:** `LocalFusedR4`.
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
/// `HeuristicAdrenoCollapse` was added in PR 3 (2026-04-15) after the
/// generation-confirmation A/B on Adreno 730/740/750 reproduced the
/// Xclipse-540 pathology.
/// `HeuristicMaliCollapse` was added in G.1.4 (2026-04-16) after G.1.1 +
/// G.1.3 measured 5 Mali devices across 4 silicon generations (G78 / G710 /
/// G715×2, Tensor G1..G4) showing unconditional +33–88% GlobalOnlyR4 wins.
/// The earlier phase-e shape (windowed at log 18 only) was falsified by
/// G.1.1's scope-matched cold rerun.
/// `HeuristicXclipseCollapse` was added in G.2.3 (2026-04-16) after the
/// BrowserStack G.2.2 cohort (Xclipse 920/940/960 across 4 Samsung
/// Galaxies, 3 driver major revisions) + FTL e2s lottery (Xclipse 940
/// independent confirmation) reported unconditional +18–84% GlobalOnlyR4
/// wins at every measured cell. Closed out the Xclipse data gap that had
/// been open since PR 2's falsification of the original n=1
/// Xclipse-540-based rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StockhamTailReason {
    /// BrowserWebGpu at log_n ≥ 20 — conservative; the browser hides the
    /// underlying silicon and we have no field data to drop this rule.
    HeuristicBrowserConservative,
    /// Native Adreno — unconditional `GlobalOnlyR4` once a tail phase
    /// exists. Three generations (730/740/750) reproduce the gather
    /// collapse at every measured size; see the module doc and
    /// `apps/android-harness/research/benchmarks/adreno-gen-confirm-2026-04-15/`.
    HeuristicAdrenoCollapse,
    /// Native Mali — unconditional `GlobalOnlyR4` once a tail phase
    /// exists. Four silicon generations (G78 / G710 / G715×2 on Tensor
    /// G1..G4) reproduce the gather collapse at every measured size
    /// (log 18..=22, +33–88% GlobalOnlyR4 wins, both directions, two Mali
    /// driver major revisions r38 + r51). See the module doc and
    /// `apps/android-harness/research/benchmarks/mali-scope-match-2026-04-16/`
    /// + `mali-older-gen-2026-04-16/`.
    HeuristicMaliCollapse,
    /// Native Xclipse — unconditional `GlobalOnlyR4` once a tail phase
    /// exists. Three silicon generations measured (Xclipse 920 / 940 / 960
    /// on Exynos 2200 / 2400 / 2600) across 4 Samsung Galaxies and 3
    /// driver major revisions all report unconditional +18–84%
    /// GlobalOnlyR4 wins at every cell (log 18..=22, both directions).
    /// Xclipse 540 queued on FTL as confirmation of the 4th generation;
    /// not gating. See the module doc and
    /// `apps/android-harness/research/benchmarks/browserstack-xclipse-cohort-2026-04-16/`.
    HeuristicXclipseCollapse,
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
            Self::HeuristicAdrenoCollapse => "HeuristicAdrenoCollapse",
            Self::HeuristicMaliCollapse => "HeuristicMaliCollapse",
            Self::HeuristicXclipseCollapse => "HeuristicXclipseCollapse",
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

    // Native Vulkan Adreno: three generations (730/740/750) reproduce the
    // Xclipse-540 strided-gather collapse at every measured size (log18..=log22,
    // +40-72% GlobalOnlyR4 wins). Adreno 830 is flatline and absorbs a marginal
    // regression. See:
    // `apps/android-harness/research/benchmarks/adreno-gen-confirm-2026-04-15/README.md`.
    if matches!(caps.gpu_family, GpuFamily::Adreno) {
        return (
            StockhamTailStrategy::GlobalOnlyR4,
            StockhamTailReason::HeuristicAdrenoCollapse,
        );
    }

    // Native Vulkan Mali: G.1.1 (2026-04-16, scope-matched rerun on 3× G715)
    // falsified the earlier phase-e windowed shape — cold forced-A/B with no
    // preceding suite still produced +50-85% GlobalOnlyR4 wins across log
    // 18..=22, matching the phase-f warm-run shape. G.1.3 (2026-04-16)
    // confirmed the shape holds on older-gen Mali (G78 MP20 / G710 MC7) and
    // across the older Mali driver major revision (r38p1 vs r51p0 on G715).
    // Five devices × four silicon generations × two driver majors all report
    // `UNCONDITIONAL @ log21`, weakest cell +33%. Same treatment as Adreno:
    // unconditional flip once a tail phase exists — pathology is kernel-level
    // and present at the smallest size we can measure. See:
    // `apps/android-harness/research/benchmarks/mali-scope-match-2026-04-16/README.md`
    // and `mali-older-gen-2026-04-16/README.md`.
    if matches!(caps.gpu_family, GpuFamily::Mali) {
        return (
            StockhamTailStrategy::GlobalOnlyR4,
            StockhamTailReason::HeuristicMaliCollapse,
        );
    }

    // Native Vulkan Xclipse: G.2.2 (2026-04-16) ran the full
    // ZkgpuInstrumentedTest on 4 Exynos-pinned Samsung Galaxies via
    // BrowserStack App Automate (sidestepping the FTL region-split lottery):
    // S22 + S22 Ultra (Xclipse 920 / Exynos 2200 / RDNA2, 2022), S24
    // (Xclipse 940 / Exynos 2400, 2024), and S26 (Xclipse 960 / Exynos
    // 2600, 2026). All 4 devices × 40 cells report UNCONDITIONAL @ log21
    // (38/40 ≥+20% Global win, 2/40 `global-narrow` at +17.8% / +18.7%
    // still Global wins). Three driver major revisions across the cohort
    // (`a927fb4` / `3b10981` / `24.0.x` / `25.2.x`) all show identical
    // shape. Cross-vendor confirmation via FTL e2s lottery (Xclipse 940
    // Galaxy S24+) matched within noise. Same treatment as Adreno and
    // Mali: unconditional flip once a tail phase exists — the pathology
    // is kernel-level and present at the smallest tested size. See:
    // `apps/android-harness/research/benchmarks/browserstack-xclipse-cohort-2026-04-16/README.md`.
    if matches!(caps.gpu_family, GpuFamily::Xclipse) {
        return (
            StockhamTailStrategy::GlobalOnlyR4,
            StockhamTailReason::HeuristicXclipseCollapse,
        );
    }

    // Native Vulkan (other families / Apple / Unknown / Intel etc.):
    // LocalFusedR4 is the legacy default. No field data has shown a
    // collapse pattern on these families.
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
    fn xclipse_picks_global_tail_at_all_sizes() {
        // G.2.3 (2026-04-16): G.2.2 BrowserStack cohort measured 4 Exynos-
        // pinned Samsung Galaxies (S22 + S22 Ultra / Xclipse 920, S24 /
        // Xclipse 940, S26 / Xclipse 960) plus an FTL e2s cross-vendor
        // confirmation on Xclipse 940. All report `UNCONDITIONAL @ log21`
        // — 38/40 cells ≥+20% Global win, 2/40 narrow at +17.8% / +18.7%
        // (still Global wins). Three driver major revisions show identical
        // shape. Same treatment as Adreno / Mali: flip is unconditional
        // once a tail phase exists. See the module doc's
        // "Xclipse collapse (G.2.3)" section.
        //
        // This test supersedes the PR 2 close-out
        // `xclipse_keeps_local_at_all_sizes` assertion: that one was
        // defensive after phase-e falsified the original n=1 Xclipse-540
        // rule; G.2.2 supplied the multi-SKU evidence that was always
        // supposed to decide the rule shape.
        let caps = Some(vulkan(GpuFamily::Xclipse));
        for log_n in [10, 15, 18, 22, 24] {
            let d = choose_stockham_tail(log_n, caps, StockhamTailOverride::Auto).unwrap();
            assert_eq!(
                d.strategy,
                StockhamTailStrategy::GlobalOnlyR4,
                "Xclipse must pick GlobalOnlyR4 at log_n={log_n}",
            );
            assert_eq!(d.reason, StockhamTailReason::HeuristicXclipseCollapse);
        }
    }

    #[test]
    fn mali_picks_global_tail_at_all_sizes() {
        // G.1.4 (2026-04-16): G.1.1 + G.1.3 measured 5 Mali devices across
        // 4 silicon generations (G78 MP20 / G710 MC7 / G715 MC7 × 3) and 2
        // Mali driver major revisions (r38p1 + r51p0) at log 18..=22, both
        // directions. All 50 measured cells: GlobalOnlyR4 wins by +33% to
        // +88% (weakest cell: panther log 20 inv at +33.0%). Same treatment
        // as Adreno: flip is unconditional once a tail phase exists — the
        // pathology is kernel-level and present at the smallest tested size.
        // See the module doc's "Mali collapse (G.1.4)" section.
        let caps = Some(vulkan(GpuFamily::Mali));
        for log_n in [10, 15, 18, 22, 24] {
            let d = choose_stockham_tail(log_n, caps, StockhamTailOverride::Auto).unwrap();
            assert_eq!(
                d.strategy,
                StockhamTailStrategy::GlobalOnlyR4,
                "Mali must pick GlobalOnlyR4 at log_n={log_n}",
            );
            assert_eq!(d.reason, StockhamTailReason::HeuristicMaliCollapse);
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
    fn adreno_picks_global_tail_at_all_sizes() {
        // PR 3 (2026-04-15): Adreno 730/740/750 reproduce the Xclipse-540
        // gather collapse at every measured size (log18..=log22, +40-72%
        // GlobalOnlyR4 wins). Apply the flip unconditionally once a tail
        // phase exists — even below the measured range, since the pathology
        // is a kernel-level issue that the measurements show starts at the
        // smallest size we can test.
        let caps = Some(vulkan(GpuFamily::Adreno));
        for log_n in [10, 15, 18, 22, 24] {
            let d = choose_stockham_tail(log_n, caps, StockhamTailOverride::Auto).unwrap();
            assert_eq!(
                d.strategy,
                StockhamTailStrategy::GlobalOnlyR4,
                "Adreno must pick GlobalOnlyR4 at log_n={log_n}",
            );
            assert_eq!(d.reason, StockhamTailReason::HeuristicAdrenoCollapse);
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
