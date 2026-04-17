use super::*;
use super::constants::{BLOCK_SIZE, DEFAULT_FOUR_STEP_THRESHOLD, MOBILE_UMA_FOUR_STEP_THRESHOLD, LOG_BLOCK};
use super::stockham_config::R4StageParams;
use super::tail_policy::{
    StockhamTailOverride, StockhamTailReason, StockhamTailStrategy, TailDecision,
};
use crate::caps::{DetectionSource, DeviceTier, GpuFamily, MemoryModel, PlatformClass};

/// Test helper: plan a Stockham config with the canonical LocalFusedR4 tail
/// above `LOG_BLOCK`, and no tail below. Mirrors what the planner chooses
/// by default when no override is present and the caps hint is unknown.
fn stockham(log_n: u32) -> StockhamPlanConfig {
    let tail = (log_n >= LOG_BLOCK).then_some(TailDecision {
        strategy: StockhamTailStrategy::LocalFusedR4,
        reason: StockhamTailReason::HeuristicDefaultLocal,
    });
    StockhamPlanConfig::new(log_n, tail).expect("valid log_n")
}

/// Test helper: plan a Stockham config that explicitly takes the
/// `GlobalOnlyR4` tail strategy for log_n >= LOG_BLOCK.
fn stockham_global_tail(log_n: u32) -> StockhamPlanConfig {
    StockhamPlanConfig::new(
        log_n,
        Some(TailDecision {
            strategy: StockhamTailStrategy::GlobalOnlyR4,
            reason: StockhamTailReason::ForcedGlobal,
        }),
    )
    .expect("valid log_n")
}

#[test]
fn planner_log1_global_only() {
    let c = stockham(1);
    assert_eq!(c.n, 2);
    assert!(!c.use_local_kernel());
    assert!(c.tail.is_none(), "log_n < LOG_BLOCK should have no tail");
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
    assert!(!c.use_local_kernel());
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
    assert!(!c.use_local_kernel());
    assert_eq!(c.num_global_stages, 5);
    assert_eq!(c.r4_stage_params.len(), 2);
    assert_eq!(c.global_stage_params.len(), 1);
    assert_eq!(c.ntt_dispatches(), 3);
    assert!(c.result_in_scratch);
}

#[test]
fn planner_log8_global_only() {
    let c = stockham(8);
    assert!(!c.use_local_kernel());
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
    assert!(!c.use_local_kernel());
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
    assert!(c.use_local_kernel());
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
    assert!(c.use_local_kernel());
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
    assert!(c.use_local_kernel());
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
    assert!(StockhamPlanConfig::new(0, None).is_err());
}

#[test]
fn planner_rejects_log32() {
    assert!(StockhamPlanConfig::new(32, None).is_err());
}

#[test]
fn planner_accepts_log31() {
    // log_n=31 with the canonical tail, matching how the real planner builds it.
    let tail = Some(TailDecision {
        strategy: StockhamTailStrategy::LocalFusedR4,
        reason: StockhamTailReason::HeuristicDefaultLocal,
    });
    assert!(StockhamPlanConfig::new(31, tail).is_ok());
}

#[test]
fn planner_rejects_tail_below_log_block() {
    // Asking for a tail phase below LOG_BLOCK is nonsensical — the constructor
    // must reject rather than silently ignore the decision.
    let tail = Some(TailDecision {
        strategy: StockhamTailStrategy::LocalFusedR4,
        reason: StockhamTailReason::HeuristicDefaultLocal,
    });
    assert!(StockhamPlanConfig::new(LOG_BLOCK - 1, tail).is_err());
}

#[test]
fn planner_global_only_tail_matches_global_only_shape() {
    // Pre-T3.A (2026-04-16 and earlier): `new_global_only` and
    // `new(log_n, Some(GlobalOnlyR4))` produced the same dispatch shape,
    // differing only in metadata (the `tail` field).
    //
    // Post-T3.A (2026-04-17): `new_global_only` now uses R8/R4/R2 greedy
    // factoring for four-step leaves (where the R8 kernel is wired).
    // Top-level Stockham (via `new` with tail) keeps R4-only factoring
    // because the top-level R8 kernel isn't wired. The two plans can
    // legitimately have different R4/R8 stage breakdowns now.
    //
    // Invariants that still hold:
    //   - Same `num_global_stages` (both cover all log_n stages).
    //   - Neither uses the local kernel.
    //   - Tail metadata still distinguishes them.
    for log_n in [LOG_BLOCK, 12, 15, 20, 22] {
        // T3.A (2026-04-17): `new_global_only` takes `r8_max_log_leaf`.
        // Tests use `u32::MAX` to exercise the R8-always path.
        let global_only = StockhamPlanConfig::new_global_only(log_n, u32::MAX).unwrap();
        let global_tail = stockham_global_tail(log_n);
        assert_eq!(global_only.num_global_stages, global_tail.num_global_stages);
        assert!(!global_tail.use_local_kernel());
        assert!(global_tail.tail.is_some(), "GlobalOnlyR4 must record a tail decision");
        assert!(global_only.tail.is_none(), "new_global_only leaves tail unset");
        // R4 stage params differ by design: new_global_only may have
        // consumed some stages with R8 dispatches.
        assert!(global_tail.r8_stage_params.is_empty(), "top-level Stockham has no R8");
    }
}

#[test]
fn planner_tail_stride_bytes_only_set_for_local_fused() {
    // tail_stride_bytes is the coalescing-pressure signal: it must be Some
    // iff the plan actually performs the strided gather in the local kernel.
    let local = stockham(20);
    assert!(local.use_local_kernel());
    let stride = local.tail_stride_bytes();
    let expected_stride = ((1u64 << 20) / BLOCK_SIZE as u64) * std::mem::size_of::<u32>() as u64;
    assert_eq!(stride, Some(expected_stride));

    let global_tail = stockham_global_tail(20);
    assert!(!global_tail.use_local_kernel());
    assert_eq!(
        global_tail.tail_stride_bytes(),
        None,
        "GlobalOnlyR4 has no strided gather, so no stride to report"
    );

    let tiny = stockham(8);
    assert!(!tiny.use_local_kernel());
    assert_eq!(tiny.tail_stride_bytes(), None);
}

// --- Four-step planner tests ---

#[test]
fn four_step_log20_balanced() {
    let c = FourStepPlanConfig::new(20, u32::MAX).unwrap();
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
    let c = FourStepPlanConfig::new(21, u32::MAX).unwrap();
    assert_eq!(c.n, 1 << 21);
    assert_eq!(c.row_log_n, 10);
    assert_eq!(c.col_log_n, 11);
    assert_eq!(c.rows, 1024);
    assert_eq!(c.cols, 2048);
}

#[test]
fn four_step_dispatch_count() {
    let c = FourStepPlanConfig::new(20, u32::MAX).unwrap();
    // Post-T3.A (2026-04-17) with R8/R4/R2 factoring for leaves of log 10:
    //   log 10 = 3 R8 (9 stages) + 1 R2 (1 stage) = 4 dispatches per leaf
    //   (pre-T3.A was 5 R4 = 5 dispatches per leaf)
    // Total: 1 transpose (phase 1) + 4 leaf dispatches (phase 2) + 1 twiddle +
    //        1 transpose (phase 4) + 4 leaf dispatches (phase 5) +
    //        1 transpose (phase 6) = 12 dispatches.
    // (Tier 2B Option A removed the separate Phase-7 inverse-scale dispatch.)
    assert_eq!(c.total_dispatches(), 12);
}

// --- Policy-driven family selection tests ---

fn native_policy() -> PlannerPolicy {
    PlannerPolicy::with_four_step_threshold(DEFAULT_FOUR_STEP_THRESHOLD)
}

fn web_policy() -> PlannerPolicy {
    PlannerPolicy::stockham_only()
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
    let low_threshold = PlannerPolicy::with_four_step_threshold(14);
    let p14 = plan_ntt(14, &low_threshold).unwrap();
    assert!(matches!(p14, PlannedNtt::FourStep(_)));

    let p13 = plan_ntt(13, &low_threshold).unwrap();
    assert!(matches!(p13, PlannedNtt::Stockham(_)));
}

#[test]
fn disabled_four_step_ignores_threshold() {
    let disabled = PlannerPolicy::stockham_only();
    let p = plan_ntt(25, &disabled).unwrap();
    assert!(matches!(p, PlannedNtt::Stockham(_)));
}

// --- from_caps identity-driven policy tests ---

fn mock_caps_identity(
    gpu_family: GpuFamily,
    platform_class: PlatformClass,
) -> crate::caps::CapabilityProfile {
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
    crate::caps::CapabilityProfile {
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
        transient_saves_memory: false,
        max_buffer_size: 0,
        max_storage_buffer_binding_size: 0,
        max_compute_workgroup_size_x: 0,
        max_compute_workgroup_size_y: 0,
        max_compute_workgroup_size_z: 0,
        max_compute_invocations_per_workgroup: 0,
        max_compute_workgroups_per_dimension: 0,
        max_compute_workgroup_storage_size: 0,
    }
}

#[test]
fn from_caps_browser_disables_four_step() {
    let caps = mock_caps_identity(GpuFamily::Unknown, PlatformClass::Browser);
    let policy = PlannerPolicy::from_caps(&caps);
    assert_eq!(policy.four_step_threshold(), None);
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
    // NVIDIA scale-up Tier 1 (2026-04-16): dropped from 24 to 21 after
    // G.0.4 ICICLE A/B on RTX 4090 showed zkgpu Four-Step beats ICICLE
    // Radix-2 at log 21 (0.75×). The old log_n >= 24 threshold left the
    // pathological 17–21× DEFAULT regression on the table. See:
    // `research/benchmarks/foundation-audit-2026-04-15/nvidia-scale-up-roadmap.md`
    // §Tier 1, and the matching comment in `policy.rs::from_vulkan_family`.
    let caps = mock_caps_identity(GpuFamily::Nvidia, PlatformClass::DesktopDiscrete);
    let policy = PlannerPolicy::from_caps(&caps);
    assert_eq!(policy.four_step_threshold(), Some(21));
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
    // NVIDIA scale-up Tier 1 (2026-04-16): threshold moved from 24 to 21.
    // log 20 still Stockham (below threshold); log 21+ is Four-Step.
    let caps = mock_caps_identity(GpuFamily::Nvidia, PlatformClass::DesktopDiscrete);
    let policy = PlannerPolicy::from_caps(&caps);
    assert!(matches!(
        plan_ntt(20, &policy).unwrap(),
        PlannedNtt::Stockham(_)
    ));
    assert!(matches!(
        plan_ntt(21, &policy).unwrap(),
        PlannedNtt::FourStep(_)
    ));
    assert!(matches!(
        plan_ntt(23, &policy).unwrap(),
        PlannedNtt::FourStep(_)
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

// --- Tail-strategy integration tests (policy + plan_ntt) ---

/// Plan a Stockham config via `plan_ntt` and return its tail strategy.
/// Panics if the planner selected four-step.
fn stockham_tail_from_plan(
    log_n: u32,
    policy: &PlannerPolicy,
) -> Option<StockhamTailStrategy> {
    match plan_ntt(log_n, policy).unwrap() {
        PlannedNtt::Stockham(cfg) => cfg.tail.map(|t| t.strategy),
        PlannedNtt::FourStep(_) => panic!("plan_ntt selected four-step, expected Stockham"),
    }
}

#[test]
fn xclipse_picks_global_tail_at_all_tail_sizes() {
    // G.2.3 (2026-04-16): G.2.2 BrowserStack App Automate cohort measured
    // 4 Exynos-pinned Samsung Galaxies — S22 + S22 Ultra (Xclipse 920 /
    // Exynos 2200 / first-gen RDNA2, 2022), S24 (Xclipse 940 / Exynos
    // 2400, 2024), S26 (Xclipse 960 / Exynos 2600, 2026) — plus an FTL
    // e2s cross-vendor confirmation on Xclipse 940 Galaxy S24+. All report
    // `UNCONDITIONAL @ log21`: 38/40 cells ≥+20% Global win, 2/40 narrow
    // at +17.8% / +18.7% (still Global wins), three driver major revisions
    // all showing identical shape. See:
    // `apps/android-harness/research/benchmarks/browserstack-xclipse-cohort-2026-04-16/`.
    //
    // This test supersedes the PR-2-era `xclipse_keeps_local_tail_across_all_sizes`
    // defensive assertion. PR 2 dropped the n=1 Xclipse-540 rule; G.2.2
    // supplied the multi-SKU evidence that was always supposed to decide
    // the shape.
    //
    // Four-step disabled so we observe the tail decision in isolation;
    // Xclipse's default four-step threshold would otherwise flip at log_n=20.
    let caps = mock_caps_identity(GpuFamily::Xclipse, PlatformClass::AndroidNative);
    let mut policy = PlannerPolicy::from_caps(&caps);
    policy.four_step_threshold = None;
    for log_n in [10, 15, 18, 20, 22] {
        assert_eq!(
            stockham_tail_from_plan(log_n, &policy),
            Some(StockhamTailStrategy::GlobalOnlyR4),
            "Xclipse must pick GlobalOnlyR4 at log_n={log_n}",
        );
    }
}

#[test]
fn mali_picks_global_tail_at_all_tail_sizes() {
    // G.1.4 (2026-04-16): G.1.1 + G.1.3 measured 5 Mali devices (oriole
    // Mali-G78 MP20, panther Mali-G710 MC7, husky/komodo/comet Mali-G715
    // MC7) across 4 silicon generations, 3 SoC generations (Tensor G1..G4),
    // and 2 Mali driver major revisions (r38p1 + r51p0). All cells at log
    // 18..=22, both directions, report +33% to +88% GlobalOnlyR4 wins —
    // `UNCONDITIONAL @ log21` from `zkgpu-tail-analyze` on every device.
    // See:
    // - `apps/android-harness/research/benchmarks/mali-scope-match-2026-04-16/`
    // - `apps/android-harness/research/benchmarks/mali-older-gen-2026-04-16/`
    let caps = mock_caps_identity(GpuFamily::Mali, PlatformClass::AndroidNative);
    let policy = PlannerPolicy::from_caps(&caps);
    for log_n in [10, 15, 18, 20, 22] {
        assert_eq!(
            stockham_tail_from_plan(log_n, &policy),
            Some(StockhamTailStrategy::GlobalOnlyR4),
            "Mali must pick GlobalOnlyR4 at log_n={log_n}",
        );
    }
}

#[test]
fn adreno_picks_global_tail_at_all_tail_sizes() {
    // PR 3 (2026-04-15): the Adreno generation-confirmation A/B
    // (`apps/android-harness/research/benchmarks/adreno-gen-confirm-2026-04-15/`)
    // measured Adreno 730 (b0q), 740 (dm3q), 750 (e3q) at log_n 18..=22,
    // both directions, and saw +40-72% GlobalOnlyR4 wins at every cell.
    //
    // Four-step disabled so we observe the tail decision in isolation;
    // Adreno's mobile four-step threshold would otherwise flip at log_n=18.
    let caps = mock_caps_identity(GpuFamily::Adreno, PlatformClass::AndroidNative);
    let mut policy = PlannerPolicy::from_caps(&caps);
    policy.four_step_threshold = None;
    for log_n in [10, 15, 18, 20, 22] {
        assert_eq!(
            stockham_tail_from_plan(log_n, &policy),
            Some(StockhamTailStrategy::GlobalOnlyR4),
            "Adreno must pick GlobalOnlyR4 at log_n={log_n}",
        );
    }
}

#[test]
fn browser_large_n_picks_global_tail_by_default() {
    // Browsers sit behind an unpredictable driver stack; at log_n >= 20
    // the conservative choice is the global-only tail.
    let caps = mock_caps_identity(GpuFamily::Unknown, PlatformClass::Browser);
    let policy = PlannerPolicy::from_caps(&caps);
    assert_eq!(
        stockham_tail_from_plan(20, &policy),
        Some(StockhamTailStrategy::GlobalOnlyR4),
    );
    assert_eq!(
        stockham_tail_from_plan(18, &policy),
        Some(StockhamTailStrategy::LocalFusedR4),
    );
}

#[test]
fn default_apple_picks_local_tail() {
    // Apple's big shared memory makes LocalFusedR4 the right default.
    let caps = mock_caps_identity(GpuFamily::Apple, PlatformClass::AppleNative);
    let policy = PlannerPolicy::from_caps(&caps);
    for log_n in [10, 15, 20, 22] {
        assert_eq!(
            stockham_tail_from_plan(log_n, &policy),
            Some(StockhamTailStrategy::LocalFusedR4),
            "Apple should choose LocalFusedR4 at log_n={log_n}"
        );
    }
}

#[test]
fn explicit_local_override_wins_over_heuristic() {
    // If the heuristic says GlobalOnlyR4 (Browser @ log_n>=20 —
    // HeuristicBrowserConservative) but the caller passes `Local`, the
    // override must win and the plan uses LocalFusedR4.
    //
    // Four-step disabled so we exercise the Stockham path in isolation.
    let caps = mock_caps_identity(GpuFamily::Unknown, PlatformClass::Browser);
    let mut policy =
        PlannerPolicy::from_caps(&caps).with_stockham_tail_override(StockhamTailOverride::Local);
    policy.four_step_threshold = None;
    assert_eq!(
        stockham_tail_from_plan(20, &policy),
        Some(StockhamTailStrategy::LocalFusedR4),
    );
}

#[test]
fn explicit_global_override_wins_over_heuristic() {
    // Conversely, an Apple device would default to LocalFusedR4, but a
    // caller investigating the global-only path should be able to force it.
    let caps = mock_caps_identity(GpuFamily::Apple, PlatformClass::AppleNative);
    let policy =
        PlannerPolicy::from_caps(&caps).with_stockham_tail_override(StockhamTailOverride::Global);
    assert_eq!(
        stockham_tail_from_plan(20, &policy),
        Some(StockhamTailStrategy::GlobalOnlyR4),
    );
}

#[test]
fn with_four_step_disabled_preserves_caps_tail_heuristic() {
    // Regression for Codex review P2#1: forced-Stockham runs were calling
    // `PlannerPolicy::stockham_only()` which dropped the device caps hint
    // and silently fell back to LocalFusedR4 on every device. The fix is
    // `with_four_step_disabled()`, which preserves the caps hint so the
    // browser tail heuristic still triggers.
    //
    // After PR 2 close-out (2026-04-15) the only surviving heuristic flip
    // is `BrowserWebGpu @ log_n >= 20 → GlobalOnlyR4`, so this test uses
    // the Browser platform class to distinguish caps-aware from caps-less
    // policy construction.
    let caps = mock_caps_identity(GpuFamily::Unknown, PlatformClass::Browser);
    let policy = PlannerPolicy::from_caps(&caps).with_four_step_disabled();
    assert_eq!(policy.four_step_threshold(), None);
    assert_eq!(
        stockham_tail_from_plan(20, &policy),
        Some(StockhamTailStrategy::GlobalOnlyR4),
        "with_four_step_disabled must keep the caps-driven tail decision"
    );

    // Counter-test: the legacy `stockham_only()` constructor explicitly
    // throws caps away, so it falls back to the default local heuristic.
    // We keep this assertion to document the difference.
    let legacy = PlannerPolicy::stockham_only();
    assert_eq!(
        stockham_tail_from_plan(20, &legacy),
        Some(StockhamTailStrategy::LocalFusedR4),
    );
}

#[test]
fn with_force_four_step_preserves_caps_and_overrides() {
    // Symmetric regression: forcing four-step must also preserve the caps
    // hint and tail override so that, if the four-step plan is later
    // swapped back to Stockham (e.g. the threshold is undone via a
    // different code path), the tail strategy still reflects the device.
    let caps = mock_caps_identity(GpuFamily::Xclipse, PlatformClass::AndroidNative);
    let policy = PlannerPolicy::from_caps(&caps)
        .with_stockham_tail_override(StockhamTailOverride::Local)
        .with_force_four_step();
    assert_eq!(policy.four_step_threshold(), Some(1));
    // Force-four-step always returns four-step, so we don't go through
    // stockham_tail_from_plan; assert directly that the override survived.
    assert!(matches!(
        plan_ntt(20, &policy).unwrap(),
        PlannedNtt::FourStep(_)
    ));
    // And: undo the four-step force, observe the tail override still wins.
    let policy = policy.with_four_step_disabled();
    assert_eq!(
        stockham_tail_from_plan(20, &policy),
        Some(StockhamTailStrategy::LocalFusedR4),
        "tail override must survive the four-step → Stockham toggle"
    );
}

#[test]
fn tail_strategy_below_log_block_is_none() {
    // Below LOG_BLOCK there is no tail phase at all — neither heuristic
    // nor override should synthesise one.
    let caps = mock_caps_identity(GpuFamily::Xclipse, PlatformClass::AndroidNative);
    let base = PlannerPolicy::from_caps(&caps);
    let forced_local = base
        .clone()
        .with_stockham_tail_override(StockhamTailOverride::Local);
    let forced_global = base
        .clone()
        .with_stockham_tail_override(StockhamTailOverride::Global);
    for policy in [&base, &forced_local, &forced_global] {
        assert_eq!(
            stockham_tail_from_plan(LOG_BLOCK - 1, policy),
            None,
            "no tail should be planned below LOG_BLOCK"
        );
    }
}
