//! Coarse classification of device tier, platform context, and memory model.
//!
//! These classifiers map known adapter signals (backend, device type, features)
//! to broad capability buckets. They do not inspect device names — that is
//! handled by [`detect`](super::detect).

use super::types::{DeviceTier, MemoryModel, PlatformClass};

pub(crate) fn classify_tier(
    backend: wgpu::Backend,
    device_type: wgpu::DeviceType,
    has_subgroup: bool,
    has_mappable: bool,
) -> DeviceTier {
    if backend == wgpu::Backend::BrowserWebGpu {
        return DeviceTier::PortableWeb;
    }

    if has_mappable && device_type == wgpu::DeviceType::IntegratedGpu {
        return DeviceTier::UnifiedMemoryNative;
    }

    if has_subgroup {
        return DeviceTier::NativeSubgroup;
    }

    DeviceTier::NativeBasic
}

/// Classify the deployment context from backend, device type, and build
/// target.
///
/// Uses `#[cfg(target_os)]` to distinguish Android from desktop Vulkan,
/// so an Adreno on a Snapdragon X Windows laptop correctly gets
/// `DesktopIntegrated` instead of `AndroidNative`.
pub(crate) fn classify_platform_class(
    backend: wgpu::Backend,
    #[allow(unused_variables)] device_type: wgpu::DeviceType,
) -> PlatformClass {
    if backend == wgpu::Backend::BrowserWebGpu {
        return PlatformClass::Browser;
    }

    if backend == wgpu::Backend::Metal {
        return PlatformClass::AppleNative;
    }

    #[cfg(target_os = "android")]
    {
        return PlatformClass::AndroidNative;
    }

    #[cfg(not(target_os = "android"))]
    match device_type {
        wgpu::DeviceType::IntegratedGpu => PlatformClass::DesktopIntegrated,
        wgpu::DeviceType::DiscreteGpu => PlatformClass::DesktopDiscrete,
        _ => PlatformClass::UnknownNative,
    }
}

pub(crate) fn classify_memory_model(
    device_type: wgpu::DeviceType,
    has_mappable: bool,
) -> MemoryModel {
    if device_type == wgpu::DeviceType::IntegratedGpu && has_mappable {
        return MemoryModel::Unified;
    }
    if device_type == wgpu::DeviceType::DiscreteGpu {
        return MemoryModel::Discrete;
    }
    MemoryModel::Unknown
}

#[cfg(test)]
mod tests {
    use super::*;

    // === classify_platform_class ===

    #[test]
    fn platform_class_browser() {
        assert_eq!(
            classify_platform_class(
                wgpu::Backend::BrowserWebGpu,
                wgpu::DeviceType::Other,
            ),
            PlatformClass::Browser
        );
    }

    #[test]
    fn platform_class_metal_is_apple_native() {
        assert_eq!(
            classify_platform_class(
                wgpu::Backend::Metal,
                wgpu::DeviceType::IntegratedGpu,
            ),
            PlatformClass::AppleNative
        );
    }

    // On non-Android build targets, Adreno on Vulkan should be desktop,
    // not AndroidNative. This is the Snapdragon X fix.
    #[cfg(not(target_os = "android"))]
    #[test]
    fn platform_class_adreno_vulkan_on_desktop_is_not_android() {
        // Adreno on Vulkan on a non-Android host (Snapdragon X laptop)
        assert_eq!(
            classify_platform_class(
                wgpu::Backend::Vulkan,
                wgpu::DeviceType::IntegratedGpu,
            ),
            PlatformClass::DesktopIntegrated
        );
    }

    #[cfg(not(target_os = "android"))]
    #[test]
    fn platform_class_discrete_nvidia() {
        assert_eq!(
            classify_platform_class(
                wgpu::Backend::Vulkan,
                wgpu::DeviceType::DiscreteGpu,
            ),
            PlatformClass::DesktopDiscrete
        );
    }

    #[cfg(not(target_os = "android"))]
    #[test]
    fn platform_class_integrated_intel() {
        assert_eq!(
            classify_platform_class(
                wgpu::Backend::Vulkan,
                wgpu::DeviceType::IntegratedGpu,
            ),
            PlatformClass::DesktopIntegrated
        );
    }

    // === classify_memory_model ===

    #[test]
    fn memory_model_unified_for_integrated_mappable() {
        assert_eq!(
            classify_memory_model(wgpu::DeviceType::IntegratedGpu, true),
            MemoryModel::Unified
        );
    }

    #[test]
    fn memory_model_discrete_for_discrete_gpu() {
        assert_eq!(
            classify_memory_model(wgpu::DeviceType::DiscreteGpu, false),
            MemoryModel::Discrete
        );
    }

    #[test]
    fn memory_model_unknown_for_other() {
        assert_eq!(
            classify_memory_model(wgpu::DeviceType::Other, false),
            MemoryModel::Unknown
        );
    }
}
