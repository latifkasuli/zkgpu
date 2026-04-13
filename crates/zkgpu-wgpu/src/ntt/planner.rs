use zkgpu_core::ZkGpuError;

use crate::caps::{CapabilityProfile, GpuFamily, PlatformClass};

pub(crate) const WORKGROUP_SIZE: u32 = 256;

/// Block size for the workgroup-local Stockham kernel.
/// Each R4 butterfly touches 4 elements, so BLOCK_SIZE = 4 * WORKGROUP_SIZE
/// ensures all threads are active during every R4 stage pair.
pub(crate) const BLOCK_SIZE: u32 = 4 * WORKGROUP_SIZE;
pub(crate) const LOG_BLOCK: u32 = 10; // log2(1024)

/// Tile dimension for four-step transpose.
pub(crate) const TRANSPOSE_TILE: u32 = 16;

/// Maximum `log_n` the planner accepts (u32 shift safety).
const MAX_LOG_N: u32 = 31;

/// Maximum `log_n` for BabyBear transforms (limited by 2-adicity).
pub(crate) const MAX_BABYBEAR_LOG_N: u32 = 27;

/// Default four-step crossover for native tiers, revisable with benchmark data.
pub(crate) const DEFAULT_FOUR_STEP_THRESHOLD: u32 = 20;

/// Four-step crossover for mobile UMA (Adreno on Android).
///
/// Benchmarked on Samsung S24 Ultra (Adreno 750, Vulkan): four-step wins
/// from 2^18 through 2^21 (0.73x–0.96x GPU ratio), tied at 2^22.
/// The dominant Stockham cost centre on this class of device is the
/// shared-memory `local fused` kernel.
pub(crate) const MOBILE_UMA_FOUR_STEP_THRESHOLD: u32 = 18;

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
    four_step_threshold: Option<u32>,
    local_kernel_hint: LocalKernelHint,
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
        match caps.backend {
            // ----- Metal: always Apple, always stockham -----
            wgpu::Backend::Metal => Self::stockham_only(),

            // ----- Browser: portable baseline, no family dispatch -----
            wgpu::Backend::BrowserWebGpu => Self {
                four_step_threshold: None,
                local_kernel_hint: LocalKernelHint::ForcePortable,
            },

            // ----- Vulkan / DX12: family dispatch -----
            wgpu::Backend::Vulkan | wgpu::Backend::Dx12 => {
                Self::from_vulkan_family(caps)
            }

            // ----- OpenGL / Empty / other: conservative fallback -----
            _ => Self::with_four_step_threshold(DEFAULT_FOUR_STEP_THRESHOLD),
        }
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

// ---------------------------------------------------------------------------
// NTT family selection
// ---------------------------------------------------------------------------

/// Top-level planner decision: which family and config to use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PlannedNtt {
    Stockham(StockhamPlanConfig),
    FourStep(FourStepPlanConfig),
}

/// Choose the best NTT family for a given size and device policy.
pub(crate) fn plan_ntt(log_n: u32, policy: &PlannerPolicy) -> Result<PlannedNtt, ZkGpuError> {
    if log_n == 0 || log_n > MAX_LOG_N {
        return Err(ZkGpuError::InvalidNttSize(format!(
            "log_n={log_n} out of range (must be 1..={MAX_LOG_N})"
        )));
    }

    if let Some(threshold) = policy.four_step_threshold {
        if log_n >= threshold {
            return Ok(PlannedNtt::FourStep(FourStepPlanConfig::new(log_n)?));
        }
    }

    Ok(PlannedNtt::Stockham(StockhamPlanConfig::new(log_n)?))
}

// ---------------------------------------------------------------------------
// Stockham planner — the original hybrid R2 family
// ---------------------------------------------------------------------------

/// Uniform parameters for a single global Stockham DIF stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GlobalStageParams {
    pub n: u32,
    pub s: u32,
    pub m: u32,
    pub twiddle_offset: u32,
}

/// Parameters for a radix-4 global Stockham DIF dispatch (combines 2 stages).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct R4StageParams {
    pub n: u32,
    pub s: u32,
    pub m4: u32,
    pub twiddle_offset: u32,
}

/// Structural decisions for a Stockham hybrid NTT execution plan.
///
/// All fields are derived from `log_n` and the compile-time constants
/// `WORKGROUP_SIZE`, `BLOCK_SIZE`, and `LOG_BLOCK`. No GPU interaction,
/// no field-specific arithmetic — fully testable on any host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StockhamPlanConfig {
    pub log_n: u32,
    pub n: u32,
    pub use_local_kernel: bool,
    pub num_global_stages: u32,
    pub global_workgroups: u32,
    pub local_workgroups: u32,
    pub local_stride: u32,
    pub result_in_scratch: bool,
    pub global_stage_params: Vec<GlobalStageParams>,
    /// Radix-4 stage params (each combines 2 logical stages into 1 dispatch).
    pub r4_stage_params: Vec<R4StageParams>,
    /// (h, m4) pairs for twiddle generation, one per R4 dispatch.
    pub r4_twiddle_spec: Vec<(u32, u32)>,
    /// Number of global dispatches: r4_count + r2_count.
    pub num_global_dispatches: u32,
}

impl StockhamPlanConfig {
    /// Plan a Stockham NTT for a transform of size `2^log_n`.
    ///
    /// Returns `Err` if `log_n` is 0 or exceeds `MAX_LOG_N`.
    pub fn new(log_n: u32) -> Result<Self, ZkGpuError> {
        if log_n == 0 || log_n > MAX_LOG_N {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "log_n={log_n} out of range (must be 1..={MAX_LOG_N})"
            )));
        }
        let n = 1u32 << log_n;

        let use_local_kernel = log_n >= LOG_BLOCK;
        let num_global_stages = if use_local_kernel {
            log_n - LOG_BLOCK
        } else {
            log_n
        };

        let num_butterflies = n / 2;
        let global_workgroups = num_butterflies.div_ceil(WORKGROUP_SIZE);

        let local_workgroups = if use_local_kernel { n / BLOCK_SIZE } else { 0 };
        let local_stride = if use_local_kernel { n / BLOCK_SIZE } else { 1 };

        // Build R4 pairs and R2 remainder for global stages.
        let num_r4 = num_global_stages / 2;
        let has_r2_remainder = num_global_stages % 2 == 1;

        let mut r4_stage_params = Vec::with_capacity(num_r4 as usize);
        let mut r4_twiddle_spec = Vec::with_capacity(num_r4 as usize);
        let mut r4_twiddle_offset = 0u32;

        for i in 0..num_r4 {
            let h = i * 2;
            let s = 1u32 << h;
            let m4 = n / (4 * s);
            r4_stage_params.push(R4StageParams {
                n,
                s,
                m4,
                twiddle_offset: r4_twiddle_offset,
            });
            r4_twiddle_spec.push((h, m4));
            r4_twiddle_offset += 3 * m4;
        }

        // R2 remainder stage (if odd number of global stages)
        let mut global_stage_params = Vec::new();
        if has_r2_remainder {
            let h = num_r4 * 2;
            let s = 1u32 << h;
            let m = n >> (h + 1);
            global_stage_params.push(GlobalStageParams {
                n,
                s,
                m,
                twiddle_offset: 0,
            });
        }

        let num_global_dispatches = num_r4 + u32::from(has_r2_remainder);
        let total_swaps = num_global_dispatches + u32::from(use_local_kernel);
        let result_in_scratch = total_swaps % 2 == 1;

        Ok(Self {
            log_n,
            n,
            use_local_kernel,
            num_global_stages,
            global_workgroups,
            local_workgroups,
            local_stride,
            result_in_scratch,
            global_stage_params,
            r4_stage_params,
            r4_twiddle_spec,
            num_global_dispatches,
        })
    }

    /// Plan a Stockham NTT that uses only global DIF stages (no local kernel).
    ///
    /// Needed for four-step batched leaves where the local kernel's strided
    /// gather/scatter pattern doesn't match the contiguous batch layout.
    /// Uses radix-4 pairing for global stages (same as the main planner).
    pub fn new_global_only(log_n: u32) -> Result<Self, ZkGpuError> {
        if log_n == 0 || log_n > MAX_LOG_N {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "log_n={log_n} out of range (must be 1..={MAX_LOG_N})"
            )));
        }
        let n = 1u32 << log_n;
        let num_global_stages = log_n;
        let num_butterflies = n / 2;
        let global_workgroups = num_butterflies.div_ceil(WORKGROUP_SIZE);

        let num_r4 = num_global_stages / 2;
        let has_r2_remainder = num_global_stages % 2 == 1;

        let mut r4_stage_params = Vec::with_capacity(num_r4 as usize);
        let mut r4_twiddle_spec = Vec::with_capacity(num_r4 as usize);
        let mut r4_twiddle_offset = 0u32;

        for i in 0..num_r4 {
            let h = i * 2;
            let s = 1u32 << h;
            let m4 = n / (4 * s);
            r4_stage_params.push(R4StageParams {
                n,
                s,
                m4,
                twiddle_offset: r4_twiddle_offset,
            });
            r4_twiddle_spec.push((h, m4));
            r4_twiddle_offset += 3 * m4;
        }

        let mut global_stage_params = Vec::new();
        if has_r2_remainder {
            let h = num_r4 * 2;
            let s = 1u32 << h;
            let m = n >> (h + 1);
            global_stage_params.push(GlobalStageParams {
                n,
                s,
                m,
                twiddle_offset: 0,
            });
        }

        let num_global_dispatches = num_r4 + u32::from(has_r2_remainder);
        let result_in_scratch = num_global_dispatches % 2 == 1;

        Ok(Self {
            log_n,
            n,
            use_local_kernel: false,
            num_global_stages,
            global_workgroups,
            local_workgroups: 0,
            local_stride: 1,
            result_in_scratch,
            global_stage_params,
            r4_stage_params,
            r4_twiddle_spec,
            num_global_dispatches,
        })
    }

    /// Number of NTT stage dispatches (R4 + R2 global dispatches + optional local).
    pub fn ntt_dispatches(&self) -> u32 {
        self.num_global_dispatches + u32::from(self.use_local_kernel)
    }
}

// ---------------------------------------------------------------------------
// Four-step planner
// ---------------------------------------------------------------------------

/// Structural decisions for a four-step decomposition NTT.
///
/// Factorizes N = rows * cols and plans leaf transforms plus
/// transpose/twiddle dispatches. Leaf NTTs reuse stockham plans.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FourStepPlanConfig {
    pub log_n: u32,
    pub n: u32,
    pub row_log_n: u32,
    pub col_log_n: u32,
    pub rows: u32,
    pub cols: u32,
    pub transpose_tile: u32,
    pub transpose_workgroups_x: u32,
    pub transpose_workgroups_y: u32,
    pub row_leaf: StockhamPlanConfig,
    pub col_leaf: StockhamPlanConfig,
}

impl FourStepPlanConfig {
    /// Plan a four-step NTT of size `2^log_n`.
    ///
    /// Uses a balanced factorization: rows = 2^floor(log_n/2), cols = 2^ceil(log_n/2).
    pub fn new(log_n: u32) -> Result<Self, ZkGpuError> {
        if log_n == 0 || log_n > MAX_LOG_N {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "log_n={log_n} out of range (must be 1..={MAX_LOG_N})"
            )));
        }
        let n = 1u32 << log_n;

        let row_log_n = log_n / 2;
        let col_log_n = log_n - row_log_n;
        let rows = 1u32 << row_log_n;
        let cols = 1u32 << col_log_n;

        let row_leaf = StockhamPlanConfig::new_global_only(col_log_n)?;
        let col_leaf = StockhamPlanConfig::new_global_only(row_log_n)?;

        let transpose_tile = TRANSPOSE_TILE;
        let transpose_workgroups_x = cols.div_ceil(transpose_tile);
        let transpose_workgroups_y = rows.div_ceil(transpose_tile);

        Ok(Self {
            log_n,
            n,
            row_log_n,
            col_log_n,
            rows,
            cols,
            transpose_tile,
            transpose_workgroups_x,
            transpose_workgroups_y,
            row_leaf,
            col_leaf,
        })
    }

    /// Total dispatches across all six phases.
    pub fn total_dispatches(&self) -> u32 {
        1 // Phase 1: transpose R×C → C×R
            + self.col_leaf.ntt_dispatches() // Phase 2: R-point batched NTTs
            + 1 // Phase 3: twiddle multiply
            + 1 // Phase 4: transpose C×R → R×C
            + self.row_leaf.ntt_dispatches() // Phase 5: C-point batched NTTs
            + 1 // Phase 6: transpose R×C → C×R (output)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caps::{DetectionSource, DeviceTier, MemoryModel};

    fn stockham(log_n: u32) -> StockhamPlanConfig {
        StockhamPlanConfig::new(log_n).expect("valid log_n")
    }

    #[test]
    fn planner_log1_global_only() {
        let c = stockham(1);
        assert_eq!(c.n, 2);
        assert!(!c.use_local_kernel);
        assert_eq!(c.num_global_stages, 1);
        assert_eq!(c.ntt_dispatches(), 1);
        assert_eq!(c.global_workgroups, 1);
        assert_eq!(c.local_workgroups, 0);
        assert!(c.result_in_scratch);
    }

    #[test]
    fn planner_log4_global_only_even() {
        let c = stockham(4);
        assert_eq!(c.n, 16);
        assert!(!c.use_local_kernel);
        assert_eq!(c.num_global_stages, 4);
        assert_eq!(c.r4_stage_params.len(), 2);
        assert!(c.global_stage_params.is_empty());
        assert_eq!(c.ntt_dispatches(), 2);
        assert!(!c.result_in_scratch);
    }

    #[test]
    fn planner_log5_global_only_odd() {
        let c = stockham(5);
        assert_eq!(c.n, 32);
        assert!(!c.use_local_kernel);
        assert_eq!(c.num_global_stages, 5);
        assert_eq!(c.r4_stage_params.len(), 2);
        assert_eq!(c.global_stage_params.len(), 1);
        assert_eq!(c.ntt_dispatches(), 3);
        assert!(c.result_in_scratch);
    }

    #[test]
    fn planner_log8_global_only() {
        let c = stockham(8);
        assert!(!c.use_local_kernel);
        assert_eq!(c.num_global_stages, 8);
        assert_eq!(c.r4_stage_params.len(), 4);
        assert!(c.global_stage_params.is_empty());
        assert_eq!(c.ntt_dispatches(), 4);
        assert!(!c.result_in_scratch);
    }

    #[test]
    fn planner_log9_global_only_boundary() {
        // log_n=9 < LOG_BLOCK=10, so entirely global stages
        let c = stockham(9);
        assert_eq!(c.n, 512);
        assert!(!c.use_local_kernel);
        assert_eq!(c.num_global_stages, 9);
        assert_eq!(c.r4_stage_params.len(), 4);
        assert_eq!(c.global_stage_params.len(), 1);
        assert_eq!(c.ntt_dispatches(), 5); // 4 R4 + 1 R2
        assert!(c.result_in_scratch);
    }

    #[test]
    fn planner_log10_local_only() {
        // log_n=10 = LOG_BLOCK, so 0 global stages + 1 local dispatch
        let c = stockham(10);
        assert_eq!(c.n, 1024);
        assert!(c.use_local_kernel);
        assert_eq!(c.num_global_stages, 0);
        assert!(c.r4_stage_params.is_empty());
        assert!(c.global_stage_params.is_empty());
        assert_eq!(c.ntt_dispatches(), 1);
        assert_eq!(c.local_workgroups, 1);
        assert_eq!(c.local_stride, 1);
        assert!(c.result_in_scratch); // 1 swap, odd
    }

    #[test]
    fn planner_log11_hybrid_odd() {
        let c = stockham(11);
        assert_eq!(c.n, 2048);
        assert!(c.use_local_kernel);
        assert_eq!(c.num_global_stages, 1);
        assert!(c.r4_stage_params.is_empty());
        assert_eq!(c.global_stage_params.len(), 1);
        assert_eq!(c.ntt_dispatches(), 2); // 1 R2 + 1 local
        assert_eq!(c.local_workgroups, 2);
        assert_eq!(c.local_stride, 2);
        assert!(!c.result_in_scratch); // 2 swaps, even
    }

    #[test]
    fn planner_log20_large() {
        let c = stockham(20);
        assert_eq!(c.n, 1 << 20);
        assert!(c.use_local_kernel);
        assert_eq!(c.num_global_stages, 10);
        assert_eq!(c.r4_stage_params.len(), 5);
        assert!(c.global_stage_params.is_empty());
        assert_eq!(c.ntt_dispatches(), 6); // 5 R4 + 1 local
        assert_eq!(c.local_workgroups, (1 << 20) / BLOCK_SIZE);
        assert!(!c.result_in_scratch); // 6 swaps, even
    }

    #[test]
    fn planner_global_stage_params_log4() {
        let c = stockham(4);
        assert!(c.global_stage_params.is_empty());
        assert_eq!(c.r4_stage_params.len(), 2);
        assert_eq!(c.r4_stage_params[0], R4StageParams {
            n: 16, s: 1, m4: 4, twiddle_offset: 0,
        });
        assert_eq!(c.r4_stage_params[1], R4StageParams {
            n: 16, s: 4, m4: 1, twiddle_offset: 12,
        });
    }

    #[test]
    fn planner_global_stage_params_log10() {
        // log_n=10 = LOG_BLOCK → 0 global stages, local kernel handles all
        let c = stockham(10);
        assert!(c.r4_stage_params.is_empty());
        assert!(c.global_stage_params.is_empty());
    }

    #[test]
    fn planner_twiddle_offsets_sum() {
        for log_n in 1..=20 {
            let c = stockham(log_n);
            // Verify R4 offsets are contiguous
            let mut prev_end = 0u32;
            for sp in &c.r4_stage_params {
                assert_eq!(sp.twiddle_offset, prev_end,
                    "R4 twiddle offset gap at log_n={log_n}");
                prev_end += 3 * sp.m4;
            }
            // Verify R2 offsets are contiguous
            prev_end = 0;
            for sp in &c.global_stage_params {
                assert_eq!(sp.twiddle_offset, prev_end,
                    "R2 twiddle offset gap at log_n={log_n}");
                prev_end += sp.m;
            }
        }
    }

    #[test]
    fn planner_result_in_scratch_pattern() {
        for log_n in 1..=20 {
            let c = stockham(log_n);
            let total_swaps = c.ntt_dispatches();
            assert_eq!(
                c.result_in_scratch,
                total_swaps % 2 == 1,
                "ping-pong mismatch at log_n={log_n}"
            );
        }
    }

    #[test]
    fn planner_rejects_log0() {
        assert!(StockhamPlanConfig::new(0).is_err());
    }

    #[test]
    fn planner_rejects_log32() {
        assert!(StockhamPlanConfig::new(32).is_err());
    }

    #[test]
    fn planner_accepts_log31() {
        assert!(StockhamPlanConfig::new(31).is_ok());
    }

    // --- Four-step planner tests ---

    #[test]
    fn four_step_log20_balanced() {
        let c = FourStepPlanConfig::new(20).unwrap();
        assert_eq!(c.n, 1 << 20);
        assert_eq!(c.row_log_n, 10);
        assert_eq!(c.col_log_n, 10);
        assert_eq!(c.rows, 1024);
        assert_eq!(c.cols, 1024);
        assert_eq!(c.row_leaf.log_n, 10);
        assert_eq!(c.col_leaf.log_n, 10);
    }

    #[test]
    fn four_step_log21_unbalanced() {
        let c = FourStepPlanConfig::new(21).unwrap();
        assert_eq!(c.n, 1 << 21);
        assert_eq!(c.row_log_n, 10);
        assert_eq!(c.col_log_n, 11);
        assert_eq!(c.rows, 1024);
        assert_eq!(c.cols, 2048);
    }

    #[test]
    fn four_step_dispatch_count() {
        let c = FourStepPlanConfig::new(20).unwrap();
        // 1 transpose + 5 R4 leaf + 1 twiddle + 1 transpose + 5 R4 leaf + 1 transpose = 14
        assert_eq!(c.total_dispatches(), 14);
    }

    // --- Policy-driven family selection tests ---

    fn native_policy() -> PlannerPolicy {
        PlannerPolicy {
            four_step_threshold: Some(DEFAULT_FOUR_STEP_THRESHOLD),
            local_kernel_hint: LocalKernelHint::Auto,
        }
    }

    fn web_policy() -> PlannerPolicy {
        PlannerPolicy {
            four_step_threshold: None,
            local_kernel_hint: LocalKernelHint::Auto,
        }
    }

    #[test]
    fn plan_ntt_selects_stockham_below_threshold() {
        let p = plan_ntt(14, &native_policy()).unwrap();
        assert!(matches!(p, PlannedNtt::Stockham(_)));
    }

    #[test]
    fn plan_ntt_selects_four_step_at_threshold() {
        let p = plan_ntt(20, &native_policy()).unwrap();
        assert!(matches!(p, PlannedNtt::FourStep(_)));
    }

    #[test]
    fn plan_ntt_selects_four_step_above_threshold() {
        let p = plan_ntt(22, &native_policy()).unwrap();
        assert!(matches!(p, PlannedNtt::FourStep(_)));
    }

    #[test]
    fn portable_web_always_stockham() {
        let policy = web_policy();
        for log_n in [10, 14, 18, 20, 22, 24] {
            let p = plan_ntt(log_n, &policy).unwrap();
            assert!(
                matches!(p, PlannedNtt::Stockham(_)),
                "PortableWeb should always select Stockham, got FourStep at log_n={log_n}"
            );
        }
    }

    #[test]
    fn native_tiers_use_four_step_at_threshold() {
        let policy = native_policy();
        for log_n in [20, 21, 24, 27] {
            let p = plan_ntt(log_n, &policy).unwrap();
            assert!(
                matches!(p, PlannedNtt::FourStep(_)),
                "native tier should select FourStep at log_n={log_n}"
            );
        }
    }

    #[test]
    fn native_tiers_use_stockham_below_threshold() {
        let policy = native_policy();
        for log_n in [1, 4, 9, 14, 18, 19] {
            let p = plan_ntt(log_n, &policy).unwrap();
            assert!(
                matches!(p, PlannedNtt::Stockham(_)),
                "native tier should select Stockham at log_n={log_n}"
            );
        }
    }

    #[test]
    fn custom_threshold_overrides_default() {
        let low_threshold = PlannerPolicy {
            four_step_threshold: Some(14),
            local_kernel_hint: LocalKernelHint::Auto,
        };
        let p14 = plan_ntt(14, &low_threshold).unwrap();
        assert!(matches!(p14, PlannedNtt::FourStep(_)));

        let p13 = plan_ntt(13, &low_threshold).unwrap();
        assert!(matches!(p13, PlannedNtt::Stockham(_)));
    }

    #[test]
    fn disabled_four_step_ignores_threshold() {
        let disabled = PlannerPolicy {
            four_step_threshold: None,
            local_kernel_hint: LocalKernelHint::Auto,
        };
        let p = plan_ntt(25, &disabled).unwrap();
        assert!(matches!(p, PlannedNtt::Stockham(_)));
    }

    // --- from_caps identity-driven policy tests ---

    fn mock_caps_identity(
        gpu_family: GpuFamily,
        platform_class: PlatformClass,
    ) -> CapabilityProfile {
        let memory_model = match platform_class {
            PlatformClass::DesktopDiscrete => MemoryModel::Discrete,
            PlatformClass::Browser => MemoryModel::Unknown,
            _ => MemoryModel::Unified,
        };
        let backend = match platform_class {
            PlatformClass::AppleNative => wgpu::Backend::Metal,
            PlatformClass::Browser => wgpu::Backend::BrowserWebGpu,
            _ => wgpu::Backend::Vulkan,
        };
        let device_type = match platform_class {
            PlatformClass::DesktopDiscrete => wgpu::DeviceType::DiscreteGpu,
            _ => wgpu::DeviceType::IntegratedGpu,
        };
        CapabilityProfile {
            tier: DeviceTier::NativeBasic,
            backend,
            device_type,
            vendor_id: 0,
            device_id: 0,
            device_name: String::new(),
            driver: String::new(),
            driver_info: String::new(),
            gpu_family,
            detection_source: DetectionSource::VendorId,
            platform_class,
            memory_model,
            has_subgroup: false,
            min_subgroup_size: 0,
            max_subgroup_size: 0,
            has_timestamp_query: false,
            has_timestamp_query_inside_passes: false,
            has_mappable_primary_buffers: false,
            has_pipeline_cache: false,
            max_buffer_size: 0,
            max_storage_buffer_binding_size: 0,
            max_compute_workgroup_size_x: 0,
            max_compute_workgroup_size_y: 0,
            max_compute_workgroup_size_z: 0,
            max_compute_invocations_per_workgroup: 0,
            max_compute_workgroups_per_dimension: 0,
        }
    }

    #[test]
    fn from_caps_browser_disables_four_step() {
        let caps = mock_caps_identity(GpuFamily::Unknown, PlatformClass::Browser);
        let policy = PlannerPolicy::from_caps(&caps);
        assert_eq!(policy.four_step_threshold(), None);
    }

    #[test]
    fn from_caps_browser_forces_portable_local() {
        let caps = mock_caps_identity(GpuFamily::Unknown, PlatformClass::Browser);
        let policy = PlannerPolicy::from_caps(&caps);
        assert_eq!(policy.local_kernel_hint(), LocalKernelHint::ForcePortable);
    }

    #[test]
    fn from_caps_apple_native_disables_four_step() {
        let caps = mock_caps_identity(GpuFamily::Apple, PlatformClass::AppleNative);
        let policy = PlannerPolicy::from_caps(&caps);
        assert_eq!(policy.four_step_threshold(), None);
    }

    #[test]
    fn from_caps_adreno_android_uses_mobile_threshold() {
        let caps = mock_caps_identity(GpuFamily::Adreno, PlatformClass::AndroidNative);
        let policy = PlannerPolicy::from_caps(&caps);
        assert_eq!(
            policy.four_step_threshold(),
            Some(MOBILE_UMA_FOUR_STEP_THRESHOLD)
        );
    }

    #[test]
    fn from_caps_mali_android_uses_stockham_only() {
        let caps = mock_caps_identity(GpuFamily::Mali, PlatformClass::AndroidNative);
        let policy = PlannerPolicy::from_caps(&caps);
        assert_eq!(policy.four_step_threshold(), None);
    }

    #[test]
    fn from_caps_powervr_rogue_uses_stockham_only() {
        let caps = mock_caps_identity(GpuFamily::PowerVrRogue, PlatformClass::AndroidNative);
        let policy = PlannerPolicy::from_caps(&caps);
        assert_eq!(policy.four_step_threshold(), None);
    }

    #[test]
    fn from_caps_powervr_volcanic_uses_default_threshold() {
        let caps = mock_caps_identity(GpuFamily::PowerVrVolcanic, PlatformClass::AndroidNative);
        let policy = PlannerPolicy::from_caps(&caps);
        assert_eq!(
            policy.four_step_threshold(),
            Some(DEFAULT_FOUR_STEP_THRESHOLD)
        );
    }

    #[test]
    fn from_caps_xclipse_android_uses_default_threshold() {
        let caps = mock_caps_identity(GpuFamily::Xclipse, PlatformClass::AndroidNative);
        let policy = PlannerPolicy::from_caps(&caps);
        assert_eq!(
            policy.four_step_threshold(),
            Some(DEFAULT_FOUR_STEP_THRESHOLD)
        );
    }

    #[test]
    fn from_caps_adreno_non_android_uses_default_threshold() {
        let caps = mock_caps_identity(GpuFamily::Adreno, PlatformClass::DesktopIntegrated);
        let policy = PlannerPolicy::from_caps(&caps);
        assert_eq!(
            policy.four_step_threshold(),
            Some(DEFAULT_FOUR_STEP_THRESHOLD)
        );
    }

    #[test]
    fn from_caps_mali_non_android_uses_stockham_only() {
        let caps = mock_caps_identity(GpuFamily::Mali, PlatformClass::UnknownNative);
        let policy = PlannerPolicy::from_caps(&caps);
        assert_eq!(policy.four_step_threshold(), None);
    }

    #[test]
    fn from_caps_intel_integrated_uses_raised_threshold() {
        let caps = mock_caps_identity(
            GpuFamily::IntelIntegrated,
            PlatformClass::DesktopIntegrated,
        );
        let policy = PlannerPolicy::from_caps(&caps);
        assert_eq!(policy.four_step_threshold(), Some(22));
    }

    #[test]
    fn from_caps_nvidia_discrete_uses_raised_threshold() {
        let caps = mock_caps_identity(GpuFamily::Nvidia, PlatformClass::DesktopDiscrete);
        let policy = PlannerPolicy::from_caps(&caps);
        assert_eq!(policy.four_step_threshold(), Some(24));
    }

    #[test]
    fn from_caps_amd_discrete_uses_raised_threshold() {
        let caps = mock_caps_identity(GpuFamily::Amd, PlatformClass::DesktopDiscrete);
        let policy = PlannerPolicy::from_caps(&caps);
        assert_eq!(policy.four_step_threshold(), Some(22));
    }

    #[test]
    fn from_caps_unknown_native_uses_default_threshold() {
        let caps = mock_caps_identity(GpuFamily::Unknown, PlatformClass::UnknownNative);
        let policy = PlannerPolicy::from_caps(&caps);
        assert_eq!(
            policy.four_step_threshold(),
            Some(DEFAULT_FOUR_STEP_THRESHOLD)
        );
    }

    #[test]
    fn apple_always_stockham_regardless_of_size() {
        let caps = mock_caps_identity(GpuFamily::Apple, PlatformClass::AppleNative);
        let policy = PlannerPolicy::from_caps(&caps);
        for log_n in [10, 14, 18, 20, 22, 24] {
            let p = plan_ntt(log_n, &policy).unwrap();
            assert!(
                matches!(p, PlannedNtt::Stockham(_)),
                "Apple should select Stockham at log_n={log_n}"
            );
        }
    }

    #[test]
    fn mobile_android_selects_four_step_at_mobile_threshold() {
        let caps = mock_caps_identity(GpuFamily::Adreno, PlatformClass::AndroidNative);
        let policy = PlannerPolicy::from_caps(&caps);
        assert!(matches!(
            plan_ntt(17, &policy).unwrap(),
            PlannedNtt::Stockham(_)
        ));
        assert!(matches!(
            plan_ntt(18, &policy).unwrap(),
            PlannedNtt::FourStep(_)
        ));
        assert!(matches!(
            plan_ntt(20, &policy).unwrap(),
            PlannedNtt::FourStep(_)
        ));
    }

    #[test]
    fn mobile_non_android_selects_four_step_at_default_threshold() {
        let caps = mock_caps_identity(GpuFamily::Adreno, PlatformClass::DesktopIntegrated);
        let policy = PlannerPolicy::from_caps(&caps);
        assert!(matches!(
            plan_ntt(18, &policy).unwrap(),
            PlannedNtt::Stockham(_)
        ));
        assert!(matches!(
            plan_ntt(19, &policy).unwrap(),
            PlannedNtt::Stockham(_)
        ));
        assert!(matches!(
            plan_ntt(20, &policy).unwrap(),
            PlannedNtt::FourStep(_)
        ));
    }

    #[test]
    fn desktop_discrete_selects_four_step_at_raised_threshold() {
        let caps = mock_caps_identity(GpuFamily::Nvidia, PlatformClass::DesktopDiscrete);
        let policy = PlannerPolicy::from_caps(&caps);
        assert!(matches!(
            plan_ntt(23, &policy).unwrap(),
            PlannedNtt::Stockham(_)
        ));
        assert!(matches!(
            plan_ntt(24, &policy).unwrap(),
            PlannedNtt::FourStep(_)
        ));
    }

    // --- Constructor + accessor tests ---

    #[test]
    fn stockham_only_disables_four_step() {
        let policy = PlannerPolicy::stockham_only();
        assert_eq!(policy.four_step_threshold(), None);
    }

    #[test]
    fn force_four_step_sets_threshold_to_1() {
        let policy = PlannerPolicy::force_four_step();
        assert_eq!(policy.four_step_threshold(), Some(1));
    }

    #[test]
    fn with_four_step_threshold_sets_value() {
        let policy = PlannerPolicy::with_four_step_threshold(14);
        assert_eq!(policy.four_step_threshold(), Some(14));
    }
}
