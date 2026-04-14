use crate::caps::{CapabilityProfile, GpuFamily, PlatformClass};

use super::constants::{DEFAULT_FOUR_STEP_THRESHOLD, MOBILE_UMA_FOUR_STEP_THRESHOLD};

// ---------------------------------------------------------------------------
// Planner policy — capability-driven family selection
// ---------------------------------------------------------------------------

/// Hint for which local kernel variant to use in the Stockham tail.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LocalKernelHint {
    /// Use device capabilities to decide (subgroup if available, else R4).
    #[default]
    Auto,
    /// Force the subgroup-accelerated DIT kernel (requires SUBGROUP feature).
    ForceSubgroup,
    /// Force the portable R4 DIF kernel (ignores subgroup capability).
    ForcePortable,
}

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
    pub(super) local_kernel_hint: LocalKernelHint,
}

impl PlannerPolicy {
    /// Four-step disabled — always selects Stockham regardless of size.
    pub fn stockham_only() -> Self {
        Self {
            four_step_threshold: None,
            local_kernel_hint: LocalKernelHint::Auto,
        }
    }

    /// Four-step enabled at the given `log_n` threshold.
    pub fn with_four_step_threshold(threshold: u32) -> Self {
        Self {
            four_step_threshold: Some(threshold),
            local_kernel_hint: LocalKernelHint::Auto,
        }
    }

    /// Force four-step for all sizes (threshold = 1).
    pub fn force_four_step() -> Self {
        Self {
            four_step_threshold: Some(1),
            local_kernel_hint: LocalKernelHint::Auto,
        }
    }

    /// Returns the minimum `log_n` at which four-step replaces Stockham,
    /// or `None` if four-step is disabled.
    pub fn four_step_threshold(&self) -> Option<u32> {
        self.four_step_threshold
    }

    /// Returns the local kernel hint.
    pub fn local_kernel_hint(&self) -> LocalKernelHint {
        self.local_kernel_hint
    }

    /// Return a copy of this policy with the local kernel hint overridden.
    pub fn with_local_kernel_hint(mut self, hint: LocalKernelHint) -> Self {
        self.local_kernel_hint = hint;
        self
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
    /// Stockham only, portable local kernel, no subgroup. The browser caps
    /// capabilities regardless of physical hardware underneath.
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
    /// - `Nvidia`: four-step at `log_n >= 24`. 48–100 KB shared memory
    ///   per SM; dispatch overhead penalises four-step's extra passes.
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
    pub fn from_caps(caps: &CapabilityProfile) -> Self {
        let base = match caps.backend {
            // ----- Metal: always Apple, always stockham -----
            wgpu::Backend::Metal => Self::stockham_only(),

            // ----- Browser: portable baseline, no family dispatch -----
            wgpu::Backend::BrowserWebGpu => Self {
                four_step_threshold: None,
                local_kernel_hint: LocalKernelHint::ForcePortable,
            },

            // ----- Vulkan / DX12: family dispatch -----
            wgpu::Backend::Vulkan | wgpu::Backend::Dx12 => Self::from_vulkan_family(caps),

            // ----- OpenGL / Empty / other: conservative fallback -----
            _ => Self::with_four_step_threshold(DEFAULT_FOUR_STEP_THRESHOLD),
        };

        // Force portable local kernel on ALL backends: Naga's WGSL
        // frontend does not yet support `enable subgroups;`
        // (gfx-rs/wgpu#5555). Every native backend routes WGSL through
        // Naga, so the subgroup-accelerated DIT shader cannot compile
        // on any current wgpu backend. Browser WebGPU already forces
        // portable above.
        //
        // Metal currently dodges this because Apple reports
        // min_subgroup_size=4, which fails the >=32 check in Auto
        // resolution — but that is accidental, not safe.
        //
        // Remove this blanket override once Naga lands the subgroups
        // enable-extension, or when a SPIR-V subgroup path is added
        // that bypasses the WGSL frontend.
        base.with_local_kernel_hint(LocalKernelHint::ForcePortable)
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
            GpuFamily::Nvidia => Self::with_four_step_threshold(24),

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
