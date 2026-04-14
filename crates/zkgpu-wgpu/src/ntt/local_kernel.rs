//! Local kernel resolution — maps `LocalKernelHint` to a concrete
//! `ResolvedLocalKernel` based on device capabilities, cargo features,
//! and experimental flags.
//!
//! The separation between *hint* and *resolved kernel* exists because
//! the planner policy (`LocalKernelHint`) is set from device capabilities
//! alone, while the final kernel choice also depends on the build
//! toolchain (cargo feature, SPIR-V availability) and runtime gates
//! (experimental env flag). Merging both concerns into the hint enum
//! would leak toolchain details into the policy layer.

use zkgpu_core::ZkGpuError;

use crate::caps::CapabilityProfile;
use super::planner::LocalKernelHint;

/// The local kernel actually selected for workgroup-local Stockham stages.
///
/// `LocalKernelHint` is intent (what the planner policy requests).
/// `ResolvedLocalKernel` is the concrete shader artifact to load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedLocalKernel {
    /// Portable radix-4 DIF kernel (WGSL, works on every backend).
    PortableR4,
    /// Subgroup-accelerated DIT kernel loaded from pre-compiled SPIR-V.
    ///
    /// Only available on Vulkan when the `subgroup-vulkan-spirv` cargo
    /// feature is enabled, the device advertises `SUBGROUP` with
    /// `min_subgroup_size >= 32`, and the `ZKGPU_EXPERIMENTAL_VK_SUBGROUP=1`
    /// environment variable is set (for `Auto` hint) or the hint is
    /// `ForceSubgroup`.
    SubgroupSpirV,
}

/// Reason the subgroup kernel was not selected when it could have been.
///
/// Returned alongside `ResolvedLocalKernel::PortableR4` to explain why
/// a potentially faster path was skipped. Useful for diagnostics and
/// for Phase C reporting.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)] // variants constructed only with subgroup-vulkan-spirv feature
pub enum SubgroupUnavailableReason {
    /// Policy explicitly forced the portable kernel.
    ForcedPortable,
    /// The `subgroup-vulkan-spirv` cargo feature is not enabled.
    FeatureNotEnabled,
    /// Backend is not Vulkan (SPIR-V path is Vulkan-only).
    NotVulkan,
    /// Device does not advertise `wgpu::Features::SUBGROUP`.
    NoSubgroupFeature,
    /// Device `min_subgroup_size` is below the required minimum of 32.
    SubgroupSizeTooSmall { min_size: u32 },
    /// `ZKGPU_EXPERIMENTAL_VK_SUBGROUP=1` environment variable not set.
    /// Only checked in `Auto` mode; `ForceSubgroup` bypasses this gate.
    ExperimentalFlagNotSet,
}

/// Resolve a `LocalKernelHint` into a concrete kernel choice.
///
/// Returns the resolved kernel and, when the result is `PortableR4`,
/// an optional reason explaining why the subgroup path was not used.
///
/// When `hint` is `ForceSubgroup`, validation is strict: returns
/// `Err(ZkGpuError)` if the subgroup path cannot be satisfied.
/// `ForceSubgroup` bypasses the experimental env-var gate (the caller
/// is explicitly opting in).
///
/// When `hint` is `Auto`, all gates are checked and the function
/// silently falls back to `PortableR4` with a reason.
pub fn resolve_local_kernel(
    hint: LocalKernelHint,
    caps: &CapabilityProfile,
) -> Result<(ResolvedLocalKernel, Option<SubgroupUnavailableReason>), ZkGpuError> {
    match hint {
        LocalKernelHint::ForcePortable => Ok((
            ResolvedLocalKernel::PortableR4,
            Some(SubgroupUnavailableReason::ForcedPortable),
        )),
        LocalKernelHint::ForceSubgroup => try_subgroup(caps, true),
        LocalKernelHint::Auto => try_subgroup(caps, false),
    }
}

/// Attempt to select the subgroup SPIR-V kernel.
///
/// `strict`: when true (ForceSubgroup), failures return `Err`.
///           when false (Auto), failures return `Ok(PortableR4, reason)`.
fn try_subgroup(
    caps: &CapabilityProfile,
    strict: bool,
) -> Result<(ResolvedLocalKernel, Option<SubgroupUnavailableReason>), ZkGpuError> {
    // `caps` is only used in the feature-gated block below.
    let _ = &caps;
    // Gate 1: cargo feature
    #[cfg(not(feature = "subgroup-vulkan-spirv"))]
    {
        let reason = SubgroupUnavailableReason::FeatureNotEnabled;
        return if strict {
            Err(ZkGpuError::GpuValidation(
                "ForceSubgroup requires the `subgroup-vulkan-spirv` cargo feature".into(),
            ))
        } else {
            Ok((ResolvedLocalKernel::PortableR4, Some(reason)))
        };
    }

    #[cfg(feature = "subgroup-vulkan-spirv")]
    {
        // Gate 2: Vulkan backend
        if caps.backend != wgpu::Backend::Vulkan {
            let reason = SubgroupUnavailableReason::NotVulkan;
            return if strict {
                Err(ZkGpuError::GpuValidation(
                    "ForceSubgroup SPIR-V path requires Vulkan backend".into(),
                ))
            } else {
                Ok((ResolvedLocalKernel::PortableR4, Some(reason)))
            };
        }

        // Gate 3: SUBGROUP feature
        if !caps.has_subgroup {
            let reason = SubgroupUnavailableReason::NoSubgroupFeature;
            return if strict {
                Err(ZkGpuError::GpuValidation(
                    "ForceSubgroup requires wgpu::Features::SUBGROUP".into(),
                ))
            } else {
                Ok((ResolvedLocalKernel::PortableR4, Some(reason)))
            };
        }

        // Gate 4: min_subgroup_size >= 32
        if caps.min_subgroup_size < 32 {
            let reason = SubgroupUnavailableReason::SubgroupSizeTooSmall {
                min_size: caps.min_subgroup_size,
            };
            return if strict {
                Err(ZkGpuError::GpuValidation(format!(
                    "ForceSubgroup requires min_subgroup_size >= 32, got {}",
                    caps.min_subgroup_size,
                )))
            } else {
                Ok((ResolvedLocalKernel::PortableR4, Some(reason)))
            };
        }

        // Gate 5: experimental env flag (Auto only; ForceSubgroup bypasses)
        if !strict {
            if std::env::var("ZKGPU_EXPERIMENTAL_VK_SUBGROUP").as_deref() != Ok("1") {
                return Ok((
                    ResolvedLocalKernel::PortableR4,
                    Some(SubgroupUnavailableReason::ExperimentalFlagNotSet),
                ));
            }
        }

        log::info!(
            "zkgpu: subgroup SPIR-V local kernel activated (subgroup_size={}-{})",
            caps.min_subgroup_size,
            caps.max_subgroup_size,
        );

        Ok((ResolvedLocalKernel::SubgroupSpirV, None))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caps::types::*;

    /// Minimal mock caps for resolver tests.
    fn mock_caps(
        backend: wgpu::Backend,
        has_subgroup: bool,
        min_subgroup_size: u32,
    ) -> CapabilityProfile {
        CapabilityProfile {
            tier: DeviceTier::NativeBasic,
            backend,
            device_type: wgpu::DeviceType::DiscreteGpu,
            vendor_id: 0,
            device_id: 0,
            device_name: String::new(),
            driver: String::new(),
            driver_info: String::new(),
            gpu_family: GpuFamily::Unknown,
            detection_source: DetectionSource::Unknown,
            platform_class: PlatformClass::UnknownNative,
            memory_model: MemoryModel::Discrete,
            has_subgroup,
            min_subgroup_size,
            max_subgroup_size: min_subgroup_size,
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
    fn force_portable_always_returns_portable() {
        let caps = mock_caps(wgpu::Backend::Vulkan, true, 32);
        let (resolved, reason) = resolve_local_kernel(LocalKernelHint::ForcePortable, &caps)
            .expect("should not error");
        assert_eq!(resolved, ResolvedLocalKernel::PortableR4);
        assert_eq!(reason, Some(SubgroupUnavailableReason::ForcedPortable));
    }

    // --- Feature-gated tests ---
    // When `subgroup-vulkan-spirv` is NOT enabled:

    #[cfg(not(feature = "subgroup-vulkan-spirv"))]
    mod without_feature {
        use super::*;

        #[test]
        fn auto_returns_portable_feature_not_enabled() {
            let caps = mock_caps(wgpu::Backend::Vulkan, true, 32);
            let (resolved, reason) =
                resolve_local_kernel(LocalKernelHint::Auto, &caps).unwrap();
            assert_eq!(resolved, ResolvedLocalKernel::PortableR4);
            assert_eq!(reason, Some(SubgroupUnavailableReason::FeatureNotEnabled));
        }

        #[test]
        fn force_subgroup_errors_without_feature() {
            let caps = mock_caps(wgpu::Backend::Vulkan, true, 32);
            let result = resolve_local_kernel(LocalKernelHint::ForceSubgroup, &caps);
            assert!(result.is_err());
        }
    }

    // When `subgroup-vulkan-spirv` IS enabled:

    #[cfg(feature = "subgroup-vulkan-spirv")]
    mod with_feature {
        use super::*;

        #[test]
        fn auto_returns_portable_on_metal() {
            let caps = mock_caps(wgpu::Backend::Metal, true, 32);
            let (resolved, reason) =
                resolve_local_kernel(LocalKernelHint::Auto, &caps).unwrap();
            assert_eq!(resolved, ResolvedLocalKernel::PortableR4);
            assert_eq!(reason, Some(SubgroupUnavailableReason::NotVulkan));
        }

        #[test]
        fn auto_returns_portable_no_subgroup_feature() {
            let caps = mock_caps(wgpu::Backend::Vulkan, false, 0);
            let (resolved, reason) =
                resolve_local_kernel(LocalKernelHint::Auto, &caps).unwrap();
            assert_eq!(resolved, ResolvedLocalKernel::PortableR4);
            assert_eq!(reason, Some(SubgroupUnavailableReason::NoSubgroupFeature));
        }

        #[test]
        fn auto_returns_portable_subgroup_too_small() {
            let caps = mock_caps(wgpu::Backend::Vulkan, true, 4);
            let (resolved, reason) =
                resolve_local_kernel(LocalKernelHint::Auto, &caps).unwrap();
            assert_eq!(resolved, ResolvedLocalKernel::PortableR4);
            assert_eq!(
                reason,
                Some(SubgroupUnavailableReason::SubgroupSizeTooSmall { min_size: 4 })
            );
        }

        #[test]
        fn auto_returns_portable_without_env_flag() {
            // Clear the flag to ensure test isolation
            std::env::remove_var("ZKGPU_EXPERIMENTAL_VK_SUBGROUP");
            let caps = mock_caps(wgpu::Backend::Vulkan, true, 32);
            let (resolved, reason) =
                resolve_local_kernel(LocalKernelHint::Auto, &caps).unwrap();
            assert_eq!(resolved, ResolvedLocalKernel::PortableR4);
            assert_eq!(
                reason,
                Some(SubgroupUnavailableReason::ExperimentalFlagNotSet)
            );
        }

        #[test]
        fn auto_selects_spirv_when_all_gates_pass() {
            std::env::set_var("ZKGPU_EXPERIMENTAL_VK_SUBGROUP", "1");
            let caps = mock_caps(wgpu::Backend::Vulkan, true, 32);
            let (resolved, reason) =
                resolve_local_kernel(LocalKernelHint::Auto, &caps).unwrap();
            assert_eq!(resolved, ResolvedLocalKernel::SubgroupSpirV);
            assert_eq!(reason, None);
            std::env::remove_var("ZKGPU_EXPERIMENTAL_VK_SUBGROUP");
        }

        #[test]
        fn force_subgroup_bypasses_env_flag() {
            std::env::remove_var("ZKGPU_EXPERIMENTAL_VK_SUBGROUP");
            let caps = mock_caps(wgpu::Backend::Vulkan, true, 32);
            let (resolved, reason) =
                resolve_local_kernel(LocalKernelHint::ForceSubgroup, &caps).unwrap();
            assert_eq!(resolved, ResolvedLocalKernel::SubgroupSpirV);
            assert_eq!(reason, None);
        }

        #[test]
        fn force_subgroup_errors_on_metal() {
            let caps = mock_caps(wgpu::Backend::Metal, true, 32);
            let result = resolve_local_kernel(LocalKernelHint::ForceSubgroup, &caps);
            assert!(result.is_err());
        }

        #[test]
        fn force_subgroup_errors_subgroup_too_small() {
            let caps = mock_caps(wgpu::Backend::Vulkan, true, 4);
            let result = resolve_local_kernel(LocalKernelHint::ForceSubgroup, &caps);
            assert!(result.is_err());
        }
    }
}
