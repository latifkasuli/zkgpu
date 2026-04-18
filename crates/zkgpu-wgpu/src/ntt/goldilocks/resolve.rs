//! Goldilocks kernel-variant resolver.
//!
//! One decision, taken once at plan-construction time: given
//! (target backend, adapter features, Cargo feature flags, caller
//! override) → which Goldilocks kernel variant does this plan use?
//!
//! Goals:
//!
//! - Never silently fall back from a requested native path; unknown
//!   or unsupported override → explicit error.
//! - `Auto` defaults to `PortableU32x2` **unconditionally** until a
//!   field allowlist of proven `(backend, gpu_family, driver)`
//!   fingerprints lands. Availability ≠ fastest path; a native int64
//!   kernel on a driver we haven't measured is not the right default.
//! - Browser / wasm path always resolves to portable regardless of
//!   override value, because WGSL on the web has no `u64`.
//!
//! # Phase status
//!
//! **Phase A (this commit): enums + doc-only scaffolding.**
//! `resolve_variant` is defined but has no call sites — Phase B will
//! wire it into a new `WgpuGoldilocksNttPlan` constructor. Landing the
//! enums first lets report plumbing and CLI flags reference concrete
//! types without Phase B's kernel churn.

use crate::caps::CapabilityProfile;
use crate::field_codec::FieldStorageAbi;

/// Caller-supplied override for Goldilocks kernel selection.
///
/// Mirrors the shape of `StockhamTailOverride` — `Auto` defers to the
/// resolver's default policy; the explicit variants force a specific
/// path and error if it can't be honoured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // `Portable` / `NativeVulkan` are wired into the CLI in Phase B.
pub enum GoldilocksKernelOverride {
    /// Let the resolver pick based on target + caps + policy.
    Auto,
    /// Force the portable `u32x2` path. Works on every backend
    /// including browser-WebGPU.
    Portable,
    /// Force the native-int64 Vulkan path. Requires the
    /// `goldilocks-vulkan-int64` Cargo feature, a Vulkan backend, and
    /// an adapter exposing `wgpu::Features::SHADER_INT64`. Errors if
    /// any of those don't hold — no silent fallback.
    NativeVulkan,
}

impl Default for GoldilocksKernelOverride {
    fn default() -> Self {
        Self::Auto
    }
}

/// The concrete kernel variant a Goldilocks plan actually uses.
///
/// Distinct from [`GoldilocksKernelOverride`] because `Auto` is a
/// caller intent, not an executable variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GoldilocksKernelVariant {
    /// WGSL kernels that treat each field element as a `vec2<u32>`
    /// limb pair. Mandatory on browser / wasm, available on every
    /// native backend.
    PortableU32x2,
    /// Vulkan-specific kernels built on native `u64` shader types.
    /// Requires `SHADER_INT64`; not shippable until Phase D.
    #[allow(dead_code)]
    NativeInt64Vulkan,
}

impl GoldilocksKernelVariant {
    /// Stable tag for report JSON.
    ///
    /// Unused in Phase A; Phase B populates `CaseReport::field_kernel_variant`.
    #[allow(dead_code)]
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::PortableU32x2 => "PortableU32x2",
            Self::NativeInt64Vulkan => "NativeInt64Vulkan",
        }
    }

    /// Storage ABI implied by this variant.
    ///
    /// Unused in Phase A; Phase B's plan constructor consumes this.
    #[allow(dead_code)]
    pub(crate) const fn storage_abi(self) -> FieldStorageAbi {
        match self {
            Self::PortableU32x2 => FieldStorageAbi::Limb32x2Le,
            Self::NativeInt64Vulkan => FieldStorageAbi::NativeU64,
        }
    }
}

/// Why the resolver picked the variant it did.
///
/// One enum value per logical resolution path, so downstream reports
/// can distinguish "fell back because feature was missing" from
/// "caller forced portable" from "allowlist matched native". This
/// mirrors the tail-strategy `reason` observability pattern already in
/// the Stockham planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GoldilocksKernelReason {
    /// Running under wasm / browser-WebGPU; WGSL has no `u64`.
    BrowserWgslNoInt64,
    /// `Auto` + native backend, but adapter doesn't expose SHADER_INT64.
    /// (Only reachable once allowlist matching is implemented; Phase A
    /// resolves `Auto` → portable unconditionally, so this variant is
    /// reserved.)
    #[allow(dead_code)]
    NativeFeatureUnavailable,
    /// `Auto` + wrong backend for native int64 (e.g. Metal, DX12).
    #[allow(dead_code)]
    NativeBackendUnsupported,
    /// Native variant is disabled at build time (Cargo feature off).
    #[allow(dead_code)]
    NativeVariantDisabledByPolicy,
    /// Caller passed `Portable` override.
    ForcedPortable,
    /// Caller passed `NativeVulkan` override and all preconditions held.
    #[allow(dead_code)]
    ForcedNativeVulkan,
    /// `Auto` resolved to native via allowlist match on
    /// (backend, gpu_family, driver). Phase A never reaches this — the
    /// allowlist doesn't exist yet — but the variant exists so the
    /// report schema is stable before the allowlist lands.
    #[allow(dead_code)]
    AllowlistedNativeVulkan,
    /// Default Phase-A policy: `Auto` → `PortableU32x2` regardless of
    /// backend or feature availability, until benchmark data proves a
    /// per-device native int64 allowlist.
    AutoPortableDefault,
}

impl GoldilocksKernelReason {
    /// Unused in Phase A; Phase E (harness integration) will thread
    /// this into `CaseReport::field_kernel_reason`.
    #[allow(dead_code)]
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::BrowserWgslNoInt64 => "BrowserWgslNoInt64",
            Self::NativeFeatureUnavailable => "NativeFeatureUnavailable",
            Self::NativeBackendUnsupported => "NativeBackendUnsupported",
            Self::NativeVariantDisabledByPolicy => "NativeVariantDisabledByPolicy",
            Self::ForcedPortable => "ForcedPortable",
            Self::ForcedNativeVulkan => "ForcedNativeVulkan",
            Self::AllowlistedNativeVulkan => "AllowlistedNativeVulkan",
            Self::AutoPortableDefault => "AutoPortableDefault",
        }
    }
}

/// Fully-resolved Goldilocks kernel choice carried by the plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResolvedGoldilocksKernel {
    pub(crate) variant: GoldilocksKernelVariant,
    pub(crate) storage_abi: FieldStorageAbi,
    pub(crate) reason: GoldilocksKernelReason,
}

/// Error returned when a forced override cannot be honoured.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum GoldilocksResolveError {
    /// Caller forced `NativeVulkan` but one or more preconditions
    /// failed. The `reason` field explains which.
    NativeVulkanUnavailable { reason: GoldilocksKernelReason },
}

impl std::fmt::Display for GoldilocksResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NativeVulkanUnavailable { reason } => {
                write!(
                    f,
                    "Goldilocks native-Vulkan kernel requested but unavailable: {}",
                    reason.label()
                )
            }
        }
    }
}

impl std::error::Error for GoldilocksResolveError {}

/// Run the Phase-A resolution policy.
///
/// See module docs for the rule set. Note that `Auto` currently always
/// returns `PortableU32x2` with reason `AutoPortableDefault`; the
/// per-device native allowlist will be added in a later phase without
/// changing this signature.
#[allow(dead_code)] // Wired up in Phase B.
pub(crate) fn resolve_variant(
    override_choice: GoldilocksKernelOverride,
    caps: &CapabilityProfile,
) -> Result<ResolvedGoldilocksKernel, GoldilocksResolveError> {
    // Browser / wasm always resolves to portable, regardless of override.
    // WGSL has no standard u64 on the web.
    let on_browser = matches!(caps.backend, wgpu::Backend::BrowserWebGpu)
        || cfg!(target_arch = "wasm32");

    match override_choice {
        GoldilocksKernelOverride::Portable => Ok(ResolvedGoldilocksKernel {
            variant: GoldilocksKernelVariant::PortableU32x2,
            storage_abi: FieldStorageAbi::Limb32x2Le,
            reason: if on_browser {
                GoldilocksKernelReason::BrowserWgslNoInt64
            } else {
                GoldilocksKernelReason::ForcedPortable
            },
        }),

        GoldilocksKernelOverride::NativeVulkan => {
            if on_browser {
                return Err(GoldilocksResolveError::NativeVulkanUnavailable {
                    reason: GoldilocksKernelReason::BrowserWgslNoInt64,
                });
            }
            // Without the Cargo feature there are literally no SPIR-V
            // modules compiled in, so the request can't be honoured.
            if !native_int64_compiled_in() {
                return Err(GoldilocksResolveError::NativeVulkanUnavailable {
                    reason: GoldilocksKernelReason::NativeVariantDisabledByPolicy,
                });
            }
            if caps.backend != wgpu::Backend::Vulkan {
                return Err(GoldilocksResolveError::NativeVulkanUnavailable {
                    reason: GoldilocksKernelReason::NativeBackendUnsupported,
                });
            }
            if !caps.has_shader_int64 {
                return Err(GoldilocksResolveError::NativeVulkanUnavailable {
                    reason: GoldilocksKernelReason::NativeFeatureUnavailable,
                });
            }
            Ok(ResolvedGoldilocksKernel {
                variant: GoldilocksKernelVariant::NativeInt64Vulkan,
                storage_abi: FieldStorageAbi::NativeU64,
                reason: GoldilocksKernelReason::ForcedNativeVulkan,
            })
        }

        GoldilocksKernelOverride::Auto => {
            // Phase A: Auto is always portable. The per-device allowlist
            // that will later promote Auto to native-int64 on proven
            // fingerprints has not been built yet, and shipping an
            // on-by-default native path without that data is exactly
            // the "availability ≠ fastest" trap the spec warns against.
            Ok(ResolvedGoldilocksKernel {
                variant: GoldilocksKernelVariant::PortableU32x2,
                storage_abi: FieldStorageAbi::Limb32x2Le,
                reason: if on_browser {
                    GoldilocksKernelReason::BrowserWgslNoInt64
                } else {
                    GoldilocksKernelReason::AutoPortableDefault
                },
            })
        }
    }
}

/// `true` when the `goldilocks-vulkan-int64` Cargo feature is on —
/// i.e. SPIR-V modules for the native path are actually compiled into
/// this binary.
#[inline]
const fn native_int64_compiled_in() -> bool {
    cfg!(feature = "goldilocks-vulkan-int64")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caps::{DetectionSource, DeviceTier, GpuFamily, MemoryModel, PlatformClass};

    fn mock_caps(
        backend: wgpu::Backend,
        has_shader_int64: bool,
    ) -> CapabilityProfile {
        CapabilityProfile {
            tier: DeviceTier::PortableWeb,
            backend,
            device_type: wgpu::DeviceType::Other,
            vendor_id: 0,
            device_id: 0,
            device_name: String::new(),
            driver: String::new(),
            driver_info: String::new(),
            gpu_family: GpuFamily::Unknown,
            detection_source: DetectionSource::Unknown,
            platform_class: PlatformClass::UnknownNative,
            memory_model: MemoryModel::Discrete,
            has_subgroup: false,
            min_subgroup_size: 0,
            max_subgroup_size: 0,
            has_timestamp_query: false,
            has_timestamp_query_inside_passes: false,
            has_mappable_primary_buffers: false,
            has_pipeline_cache: false,
            has_shader_int64,
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
    fn auto_on_browser_resolves_to_portable_with_browser_reason() {
        let caps = mock_caps(wgpu::Backend::BrowserWebGpu, false);
        let r = resolve_variant(GoldilocksKernelOverride::Auto, &caps).unwrap();
        assert_eq!(r.variant, GoldilocksKernelVariant::PortableU32x2);
        assert_eq!(r.storage_abi, FieldStorageAbi::Limb32x2Le);
        assert_eq!(r.reason, GoldilocksKernelReason::BrowserWgslNoInt64);
    }

    #[test]
    fn auto_on_native_without_allowlist_is_portable_by_default() {
        let caps = mock_caps(wgpu::Backend::Vulkan, true);
        let r = resolve_variant(GoldilocksKernelOverride::Auto, &caps).unwrap();
        assert_eq!(r.variant, GoldilocksKernelVariant::PortableU32x2);
        assert_eq!(r.reason, GoldilocksKernelReason::AutoPortableDefault);
    }

    #[test]
    fn forced_portable_succeeds_on_any_backend() {
        for backend in [
            wgpu::Backend::Vulkan,
            wgpu::Backend::Dx12,
            wgpu::Backend::Metal,
            wgpu::Backend::BrowserWebGpu,
        ] {
            let caps = mock_caps(backend, false);
            let r = resolve_variant(GoldilocksKernelOverride::Portable, &caps).unwrap();
            assert_eq!(r.variant, GoldilocksKernelVariant::PortableU32x2);
        }
    }

    #[test]
    fn forced_native_vulkan_errors_when_feature_flag_off() {
        // This test is only meaningful when the Cargo feature isn't set
        // — which is the default. If someone flips the feature on in
        // CI, the resolver should still catch wrong-backend and
        // missing-caps cases below; this one becomes trivially passing
        // via the wrong-backend path.
        let caps = mock_caps(wgpu::Backend::Vulkan, true);
        let err = resolve_variant(GoldilocksKernelOverride::NativeVulkan, &caps);
        assert!(err.is_err());
    }

    #[test]
    fn forced_native_vulkan_errors_on_non_vulkan_backend() {
        let caps = mock_caps(wgpu::Backend::Metal, true);
        let err = resolve_variant(GoldilocksKernelOverride::NativeVulkan, &caps);
        match err {
            Err(GoldilocksResolveError::NativeVulkanUnavailable { reason }) => {
                // Either a wrong-backend reason or a missing-feature-flag
                // reason is acceptable — the precise ordering depends on
                // Cargo feature state, but both are legitimate rejections.
                assert!(matches!(
                    reason,
                    GoldilocksKernelReason::NativeBackendUnsupported
                        | GoldilocksKernelReason::NativeVariantDisabledByPolicy
                ));
            }
            Ok(_) => panic!("expected NativeVulkanUnavailable on Metal backend"),
        }
    }

    #[test]
    fn forced_native_vulkan_errors_on_browser() {
        let caps = mock_caps(wgpu::Backend::BrowserWebGpu, false);
        let err = resolve_variant(GoldilocksKernelOverride::NativeVulkan, &caps);
        match err {
            Err(GoldilocksResolveError::NativeVulkanUnavailable { reason }) => {
                assert_eq!(reason, GoldilocksKernelReason::BrowserWgslNoInt64);
            }
            Ok(_) => panic!("expected NativeVulkanUnavailable on browser"),
        }
    }

    #[test]
    fn resolved_kernel_storage_abi_matches_variant() {
        for (v, expected) in [
            (
                GoldilocksKernelVariant::PortableU32x2,
                FieldStorageAbi::Limb32x2Le,
            ),
            (
                GoldilocksKernelVariant::NativeInt64Vulkan,
                FieldStorageAbi::NativeU64,
            ),
        ] {
            assert_eq!(v.storage_abi(), expected);
        }
    }
}
