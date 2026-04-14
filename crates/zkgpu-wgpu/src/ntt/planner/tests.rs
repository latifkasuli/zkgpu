use super::*;
use super::constants::{BLOCK_SIZE, DEFAULT_FOUR_STEP_THRESHOLD, MOBILE_UMA_FOUR_STEP_THRESHOLD};
use super::stockham_config::R4StageParams;
use crate::caps::{DetectionSource, DeviceTier, GpuFamily, MemoryModel, PlatformClass};

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
