use crate::caps::{CapabilityProfile, GpuFamily, PlatformClass};

use super::constants::{DEFAULT_FOUR_STEP_THRESHOLD, MOBILE_UMA_FOUR_STEP_THRESHOLD};
use super::tail_policy::{StockhamTailOverride, TailCapsHint};

// ---------------------------------------------------------------------------
// Planner policy — capability-driven family selection
// ---------------------------------------------------------------------------

const ENV_STOCKHAM_TAIL: &str = "ZKGPU_STOCKHAM_TAIL";

/// Per-device crossover thresholds that control NTT family selection.
///
/// Derived from `CapabilityProfile` at plan construction time. The policy
/// layer keeps structural planning free of `wgpu` types so it remains
/// fully testable without a GPU.
///
/// Marked `#[non_exhaustive]` so downstream crates use the provided
/// constructors. New policy knobs can be added without semver breaks.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct PlannerPolicy {
    pub(super) four_step_threshold: Option<u32>,
    /// Caps subset used by `choose_stockham_tail` to decide tail strategy.
    /// `None` for capability-less constructors (`stockham_only`, etc.) —
    /// those fall back to the heuristic default of `LocalFusedR4`.
    pub(super) tail_caps_hint: Option<TailCapsHint>,
    /// Caller-supplied tail override. `Auto` means use the heuristic.
    pub(super) stockham_tail_override: StockhamTailOverride,
}

impl PlannerPolicy {
    /// Four-step disabled — always selects Stockham regardless of size.
    pub fn stockham_only() -> Self {
        Self {
            four_step_threshold: None,
            tail_caps_hint: None,
            stockham_tail_override: StockhamTailOverride::Auto,
        }
    }

    /// Four-step enabled at the given `log_n` threshold.
    pub fn with_four_step_threshold(threshold: u32) -> Self {
        Self {
            four_step_threshold: Some(threshold),
            tail_caps_hint: None,
            stockham_tail_override: StockhamTailOverride::Auto,
        }
    }

    /// Force four-step for all sizes (threshold = 1).
    pub fn force_four_step() -> Self {
        Self {
            four_step_threshold: Some(1),
            tail_caps_hint: None,
            stockham_tail_override: StockhamTailOverride::Auto,
        }
    }

    /// Returns the minimum `log_n` at which four-step replaces Stockham,
    /// or `None` if four-step is disabled.
    pub fn four_step_threshold(&self) -> Option<u32> {
        self.four_step_threshold
    }

    /// Per-(backend, family) cap on `log_leaf` where R8 leaves are used in
    /// Four-Step. Values above this cap fall back to R4+R2.
    ///
    /// NVIDIA scale-up T3.A investigation (2026-04-17) found R8 behavior
    /// varies by platform:
    ///
    /// - **Apple / Metal**: R8 wins consistently (M4 Pro measured 1.30×
    ///   faster than R4 at log 22/23, 5/5 trials stable). No regression
    ///   observed.
    /// - **NVIDIA / Vulkan**: Bimodal at log 23–24 due to GPU memory-system
    ///   regime (same pathology that affects Stockham at log 24). R4
    ///   Four-Step sometimes hits a "fast mode" at log 23 that R8 can't
    ///   reach; R8 wins median at log 24 but R4 wins median at log 23.
    ///   Conservative cap at 11 keeps R8 wins at log_leaf ≤ 11 without
    ///   risking the log-23 regression.
    /// - **Others (Mali/Adreno/Xclipse/AMD/Intel)**: Not A/B-tested at
    ///   log_leaf ≥ 12. Mobile workloads rarely exceed log 22 (log_leaf=11)
    ///   so the cap is effectively unlimited in practice. Correctness is
    ///   validated by the existing post-landing regression tests.
    ///
    /// Overridable via env var `ZKGPU_R8_MAX_LOG_LEAF` for investigation.
    pub(crate) fn r8_max_log_leaf(&self) -> u32 {
        // Env var override wins — used for A/B investigation (not a public
        // knob).
        if let Ok(s) = std::env::var("ZKGPU_R8_MAX_LOG_LEAF") {
            if let Ok(v) = s.parse::<u32>() {
                return v;
            }
        }
        let Some(hint) = self.tail_caps_hint else {
            // No device context (unit tests, fallback): conservative cap.
            return 11;
        };
        use crate::caps::GpuFamily;
        match (hint.backend, hint.gpu_family) {
            // Apple Metal — measured win, no regression.
            (wgpu::Backend::Metal, _) => u32::MAX,
            // NVIDIA Vulkan — bimodal at log 23 on RTX 4090. Keep conservative.
            (wgpu::Backend::Vulkan, GpuFamily::Nvidia)
            | (wgpu::Backend::Dx12, GpuFamily::Nvidia) => 11,
            // Everything else: ungate. Mobile families rarely reach
            // log_leaf ≥ 12 in practice; correctness verified via
            // post-landing tests.
            _ => u32::MAX,
        }
    }

    /// Disable the four-step path while preserving every other policy field.
    ///
    /// Use when a benchmark or test needs to force Stockham on a device
    /// whose natural policy would have selected four-step. Unlike
    /// [`stockham_only`](Self::stockham_only), this preserves the device's
    /// `tail_caps_hint`, so the Stockham tail heuristic still runs with
    /// full device context. Critical for forced-Stockham A/B comparisons
    /// on Xclipse / Mali / Browser, where the new `GlobalOnlyR4` strategy
    /// is exactly what we are trying to measure — `stockham_only()` would
    /// throw the caps hint away and silently fall back to `LocalFusedR4`.
    pub fn with_four_step_disabled(mut self) -> Self {
        self.four_step_threshold = None;
        self
    }

    /// Force the four-step path at every size while preserving every other
    /// policy field (notably `tail_caps_hint` and `stockham_tail_override`).
    ///
    /// Symmetric counterpart to [`with_four_step_disabled`](Self::with_four_step_disabled).
    pub fn with_force_four_step(mut self) -> Self {
        self.four_step_threshold = Some(1);
        self
    }

    /// Apply an explicit Stockham tail-phase override.
    ///
    /// `Auto` falls back to the heuristic. `Local` and `Global` force the
    /// corresponding strategy regardless of device or `log_n`. The override
    /// has no effect when the planner selects four-step.
    ///
    /// Visibility is `pub(crate)` because the override type is crate-internal;
    /// external callers reach this via `WgpuNttPlan`'s public wrapper.
    pub(crate) fn with_stockham_tail_override(mut self, ov: StockhamTailOverride) -> Self {
        self.stockham_tail_override = ov;
        self
    }

    /// Resolve the tail override from (caller value, env var).
    ///
    /// The caller value wins; env is consulted only when the caller passes
    /// `Auto`. Used by the runners so a benchmark spec's explicit override
    /// always takes precedence over a stale environment.
    pub(crate) fn with_stockham_tail_override_resolved(
        self,
        explicit: StockhamTailOverride,
    ) -> Self {
        let resolved = match explicit {
            StockhamTailOverride::Auto => parse_env_tail_override(),
            other => other,
        };
        self.with_stockham_tail_override(resolved)
    }

    /// Derive a planner policy from the device's capability profile.
    ///
    /// Dispatch is **backend-first**, then family-second only where the
    /// backend doesn't fully determine the strategy:
    ///
    /// ## Metal (always Apple silicon)
    /// Stockham only. Benchmarked on A12 (XS Max), A19 Pro (iPhone 17 Pro),
    /// and M4 Pro: Stockham beats four-step 1.2x–7.6x across 2^18–2^22.
    /// Apple's 60 KB threadgroup memory favours Stockham at every size.
    ///
    /// ## BrowserWebGpu (any physical GPU behind the sandbox)
    /// Stockham only. The browser caps capabilities regardless of
    /// physical hardware underneath.
    ///
    /// ## Vulkan (family matters — diverse hardware behind one API)
    /// - `Mali/Immortalis`: Stockham only. No dedicated on-chip shared
    ///   memory — `var<workgroup>` is backed by 16 KB L1 cache. Transpose
    ///   kernels would thrash it. ARM docs explicitly warn against copying
    ///   data to shared memory on ARM GPUs.
    /// - `Adreno` on Android: four-step at `log_n >= 18`. Benchmarked on
    ///   S24 Ultra (Adreno 750): four-step wins 2^18–2^21. Real 32 KB
    ///   GMEM scratchpad makes transpose viable.
    /// - `Adreno` off Android (Snapdragon X Elite): default threshold.
    /// - `Nvidia`: four-step at `log_n >= 21`. 48–100 KB shared memory
    ///   per SM; dispatch overhead penalises four-step's extra passes at
    ///   small N. Threshold dropped from 24 to 21 in the NVIDIA scale-up
    ///   Tier 1 work (2026-04-16) after G.0.4 ICICLE A/B on RTX 4090
    ///   showed zkgpu Four-Step beats ICICLE-Radix-2 at log 21.
    /// - `Amd`: four-step at `log_n >= 22`. 128 KB LDS per WGP.
    /// - `Intel`: four-step at `log_n >= 22`. Up to 128 KB SLM per Xe-core.
    /// - `PowerVrRogue` (old GE/GM/GX): blocked by `is_gpu_usable()` —
    ///   crashes on compute dispatch. If somehow reached, stockham only.
    /// - `PowerVrVolcanic` (modern DXT/CXT/BXT): also blocked by
    ///   `is_gpu_usable()` pending driver fixes. If allowlisted, default
    ///   threshold pending benchmarks.
    /// - `Xclipse`: default threshold. Samsung RDNA2 LDS is strong but
    ///   CU count is limited.
    /// - Unknown: default threshold as conservative fallback.
    ///
    /// ## DX12 (same hardware as Vulkan, Windows only)
    /// Same family-level dispatch as Vulkan.
    ///
    /// The Stockham tail-phase strategy is also derived from `caps` (see
    /// [`tail_policy`](super::tail_policy)). Override via
    /// [`with_stockham_tail_override`](Self::with_stockham_tail_override)
    /// or the `ZKGPU_STOCKHAM_TAIL` env var.
    pub fn from_caps(caps: &CapabilityProfile) -> Self {
        let mut policy = match caps.backend {
            // ----- Metal: always Apple, always stockham -----
            wgpu::Backend::Metal => Self::stockham_only(),

            // ----- Browser: portable baseline, no family dispatch -----
            wgpu::Backend::BrowserWebGpu => Self::stockham_only(),

            // ----- Vulkan / DX12: family dispatch -----
            wgpu::Backend::Vulkan | wgpu::Backend::Dx12 => Self::from_vulkan_family(caps),

            // ----- OpenGL / Empty / other: conservative fallback -----
            _ => Self::with_four_step_threshold(DEFAULT_FOUR_STEP_THRESHOLD),
        };
        policy.tail_caps_hint = Some(TailCapsHint {
            backend: caps.backend,
            gpu_family: caps.gpu_family,
        });
        policy
    }

    /// Family-level dispatch for Vulkan and DX12 backends where the same
    /// API surface covers wildly different GPU architectures.
    fn from_vulkan_family(caps: &CapabilityProfile) -> Self {
        match caps.gpu_family {
            // Apple via MoltenVK or similar — same strategy as Metal.
            GpuFamily::Apple => Self::stockham_only(),

            // Mali / Immortalis: no real shared memory (16 KB L1-backed).
            GpuFamily::Mali => Self::stockham_only(),

            // Adreno: real 32 KB GMEM. Four-step viable on Android.
            GpuFamily::Adreno => {
                if caps.platform_class == PlatformClass::AndroidNative {
                    Self::with_four_step_threshold(MOBILE_UMA_FOUR_STEP_THRESHOLD)
                } else {
                    Self::with_four_step_threshold(DEFAULT_FOUR_STEP_THRESHOLD)
                }
            }

            // Nvidia: large shared mem (48–100 KB), high dispatch cost.
            //
            // NVIDIA scale-up Tier 1 (2026-04-16, from G.0.4 ICICLE A/B on
            // RTX 4090 at `research/benchmarks/foundation-audit-2026-04-15/
            // g04-icicle-comparison/rtx4090_vastai/README.md`): the old
            // log_n >= 24 threshold was pathological. At log 21, zkgpu
            // Four-Step *beats* ICICLE Radix-2 (0.75×); at log 22 forward
            // it's still within 3.35× of ICICLE-Radix-2 on its own family.
            // Staying Stockham through log 23 surrendered the 17–21×
            // DEFAULT gap the G.0.4 doc credits as the single biggest
            // pathological default-pick regression. Drop the threshold
            // to 21 — the lowest log_n where Four-Step wins on measured
            // hardware. Projected geomean: DEFAULT 21.0× → ORACLE 10.5×
            // per `research/benchmarks/foundation-audit-2026-04-15/
            // nvidia-scale-up-roadmap.md` Tier 1 table.
            GpuFamily::Nvidia => Self::with_four_step_threshold(21),

            // AMD: 128 KB LDS per WGP.
            GpuFamily::Amd => Self::with_four_step_threshold(22),

            // Intel: up to 128 KB SLM per Xe-core.
            GpuFamily::IntelIntegrated | GpuFamily::IntelDiscrete => {
                Self::with_four_step_threshold(22)
            }

            // PowerVR Rogue: should be blocked by is_gpu_usable() before
            // we get here. If somehow reached, stockham only — these GPUs
            // have no real compute capability.
            GpuFamily::PowerVrRogue => Self::stockham_only(),

            // PowerVR Volcanic: should be blocked by is_gpu_usable() until
            // drivers mature. If allowlisted, conservative default pending
            // benchmarks. Pixel 10 DXT-48 would go here once fixed.
            GpuFamily::PowerVrVolcanic => {
                Self::with_four_step_threshold(DEFAULT_FOUR_STEP_THRESHOLD)
            }

            // Xclipse (Samsung RDNA2 LDS): conservative default.
            GpuFamily::Xclipse => {
                Self::with_four_step_threshold(DEFAULT_FOUR_STEP_THRESHOLD)
            }

            // Unknown family: conservative default.
            GpuFamily::Unknown => {
                Self::with_four_step_threshold(DEFAULT_FOUR_STEP_THRESHOLD)
            }
        }
    }
}

/// Parse `ZKGPU_STOCKHAM_TAIL` into a tail override.
///
/// Recognised values: `auto` (default), `local`, `global`. Unrecognised
/// values log a warning and resolve to `Auto`.
fn parse_env_tail_override() -> StockhamTailOverride {
    let Ok(val) = std::env::var(ENV_STOCKHAM_TAIL) else {
        return StockhamTailOverride::Auto;
    };
    match val.trim().to_ascii_lowercase().as_str() {
        "" | "auto" => StockhamTailOverride::Auto,
        "local" => StockhamTailOverride::Local,
        "global" => StockhamTailOverride::Global,
        other => {
            log::warn!(
                "{ENV_STOCKHAM_TAIL}={other:?} not recognised \
                 (expected auto|local|global); using auto"
            );
            StockhamTailOverride::Auto
        }
    }
}
