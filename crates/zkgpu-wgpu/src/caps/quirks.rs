//! Driver safety quirks — blocklist known-broken GPU/driver combos.
//!
//! The canary calls these checks *before* attempting any `vkQueueSubmit`
//! to avoid process-fatal SIGSEGV on Android.

use super::profile::CapabilityProfile;
use super::types::{DriverQuirks, GpuFamily, PlatformClass};

/// Derive driver quirks from the capability profile.
///
/// Scoped to the observed failure signature: PowerVR on Android Vulkan.
/// Browser WebGPU and non-Android platforms are not blocked because:
/// - Browsers provide their own crash isolation (process sandbox)
/// - No crash evidence exists outside the Android Vulkan path
///
/// Known-bad driver fingerprints (tested via ADB + Firebase Test Lab):
///
/// | Device               | GPU              | Driver info      | Failure mode                         |
/// |----------------------|------------------|------------------|--------------------------------------|
/// | Xiaomi Redmi (klein) | Rogue GE8322     | 24.2@6643903     | SIGSEGV on 2nd vkQueueSubmit + silent arith corruption |
/// | Pixel 10 Pro Fold    | Volcanic DXT-48  | 24.3@6660496     | SIGSEGV in IMG_vkQueueSubmit+1524    |
///
/// As Imagination ships driver fixes, add allowlist entries keyed on
/// `driver_info` version strings inside the Volcanic arm.
pub fn driver_quirks(caps: &CapabilityProfile) -> DriverQuirks {
    // Only block on the backend + platform combination where crashes
    // were actually observed: Vulkan on Android.
    let is_android_vulkan = caps.backend == wgpu::Backend::Vulkan
        && caps.platform_class == PlatformClass::AndroidNative;

    match caps.gpu_family {
        // Rogue on Android Vulkan: universally broken compute.
        // Tested: GE8322 on Xiaomi Redmi (Unisoc SP9863A) — SIGSEGV null-
        // deref in vulkan.sp9863a.so on 2nd vkQueueSubmit, plus silent
        // arithmetic corruption (wrong BabyBear modular results).
        // Driver: "PowerVR Rogue Vulkan Driver", info: "24.2@6643903".
        // No working compute drivers exist for Rogue-era hardware.
        GpuFamily::PowerVrRogue if is_android_vulkan => DriverQuirks {
            may_crash_on_compute: true,
        },

        // Volcanic on Android Vulkan: block by default, allowlist known-
        // good driver versions.
        // Tested: DXT-48-1536 MC1 on Pixel 10 Pro Fold (Tensor G5) —
        // SIGSEGV in IMG_vkQueueSubmit+1524 inside vulkan.powervr.so.
        // Driver: "PowerVR D-Series Vulkan Driver", info: "24.3@6660496".
        GpuFamily::PowerVrVolcanic if is_android_vulkan => {
            // Allowlist: add known-good driver versions here as they are
            // confirmed on Firebase. Example:
            // if caps.driver_info.starts_with("25.") {
            //     return DriverQuirks::default();
            // }
            DriverQuirks {
                may_crash_on_compute: true,
            }
        }

        _ => DriverQuirks::default(),
    }
}

/// Pre-flight check: is this GPU safe to use for compute dispatch?
///
/// Must be called **before** the canary dispatch. On Android, a SIGSEGV
/// kills the entire process with no recovery path, so we must refuse to
/// dispatch rather than let the canary crash.
///
/// Only blocks the specific backend + platform combination where crashes
/// were observed (Android Vulkan). Browser WebGPU and desktop platforms
/// are not blocked — the browser provides crash isolation, and there is
/// no crash evidence outside the Android Vulkan path.
///
/// Returns `Ok(())` if the GPU is safe, or `Err(GpuComputeUnsupported)`
/// with a human-readable reason if the GPU/driver combo is known-broken.
pub fn is_gpu_usable(caps: &CapabilityProfile) -> Result<(), zkgpu_core::ZkGpuError> {
    let quirks = driver_quirks(caps);
    if quirks.may_crash_on_compute {
        let family = format!("{:?}", caps.gpu_family);
        return Err(zkgpu_core::ZkGpuError::GpuComputeUnsupported(format!(
            "{} GPU '{}' has a known driver bug that crashes on compute dispatch \
             (backend: {:?}, platform: {:?}, driver: '{}', info: '{}'); \
             refusing to use GPU to avoid process-fatal SIGSEGV",
            family, caps.device_name, caps.backend, caps.platform_class,
            caps.driver, caps.driver_info,
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::types::*;

    fn mock_caps_with_family(gpu_family: GpuFamily) -> CapabilityProfile {
        mock_caps(gpu_family, wgpu::Backend::Vulkan, PlatformClass::AndroidNative)
    }

    fn mock_caps(
        gpu_family: GpuFamily,
        backend: wgpu::Backend,
        platform_class: PlatformClass,
    ) -> CapabilityProfile {
        CapabilityProfile {
            tier: DeviceTier::NativeBasic,
            backend,
            device_type: wgpu::DeviceType::IntegratedGpu,
            vendor_id: 0,
            device_id: 0,
            device_name: String::new(),
            driver: String::new(),
            driver_info: String::new(),
            gpu_family,
            detection_source: DetectionSource::VendorId,
            platform_class,
            memory_model: MemoryModel::Unified,
            has_subgroup: false,
            min_subgroup_size: 0,
            max_subgroup_size: 0,
            has_timestamp_query: false,
            has_timestamp_query_inside_passes: false,
            has_mappable_primary_buffers: false,
            has_pipeline_cache: false,
            has_shader_int64: false,
            transient_saves_memory: false,
            max_buffer_size: 0,
            max_storage_buffer_binding_size: 0u64,
            max_compute_workgroup_size_x: 0,
            max_compute_workgroup_size_y: 0,
            max_compute_workgroup_size_z: 0,
            max_compute_invocations_per_workgroup: 0,
            max_compute_workgroups_per_dimension: 0,
            max_compute_workgroup_storage_size: 0,
        }
    }

    // --- Android Vulkan: PowerVR is blocked ---

    #[test]
    fn driver_quirks_rogue_android_vulkan_crashes() {
        let caps = mock_caps(GpuFamily::PowerVrRogue, wgpu::Backend::Vulkan, PlatformClass::AndroidNative);
        assert!(driver_quirks(&caps).may_crash_on_compute);
    }

    #[test]
    fn driver_quirks_volcanic_android_vulkan_crashes() {
        let caps = mock_caps(GpuFamily::PowerVrVolcanic, wgpu::Backend::Vulkan, PlatformClass::AndroidNative);
        assert!(driver_quirks(&caps).may_crash_on_compute);
    }

    // --- Browser WebGPU: PowerVR is NOT blocked (browser provides isolation) ---

    #[test]
    fn driver_quirks_rogue_browser_webgpu_ok() {
        let caps = mock_caps(GpuFamily::PowerVrRogue, wgpu::Backend::BrowserWebGpu, PlatformClass::Browser);
        assert!(!driver_quirks(&caps).may_crash_on_compute);
    }

    #[test]
    fn driver_quirks_volcanic_browser_webgpu_ok() {
        let caps = mock_caps(GpuFamily::PowerVrVolcanic, wgpu::Backend::BrowserWebGpu, PlatformClass::Browser);
        assert!(!driver_quirks(&caps).may_crash_on_compute);
    }

    // --- Desktop / non-Android native: PowerVR is NOT blocked ---

    #[test]
    fn driver_quirks_volcanic_desktop_vulkan_ok() {
        let caps = mock_caps(GpuFamily::PowerVrVolcanic, wgpu::Backend::Vulkan, PlatformClass::DesktopIntegrated);
        assert!(!driver_quirks(&caps).may_crash_on_compute);
    }

    #[test]
    fn driver_quirks_rogue_desktop_vulkan_ok() {
        let caps = mock_caps(GpuFamily::PowerVrRogue, wgpu::Backend::Vulkan, PlatformClass::DesktopIntegrated);
        assert!(!driver_quirks(&caps).may_crash_on_compute);
    }

    // --- Non-PowerVR families are never blocked ---

    #[test]
    fn driver_quirks_adreno_ok() {
        let caps = mock_caps_with_family(GpuFamily::Adreno);
        assert!(!driver_quirks(&caps).may_crash_on_compute);
    }

    #[test]
    fn driver_quirks_mali_ok() {
        let caps = mock_caps_with_family(GpuFamily::Mali);
        assert!(!driver_quirks(&caps).may_crash_on_compute);
    }

    #[test]
    fn driver_quirks_nvidia_ok() {
        let caps = mock_caps_with_family(GpuFamily::Nvidia);
        assert!(!driver_quirks(&caps).may_crash_on_compute);
    }

    // === is_gpu_usable ===

    #[test]
    fn is_gpu_usable_blocks_rogue_android_vulkan() {
        let caps = mock_caps(GpuFamily::PowerVrRogue, wgpu::Backend::Vulkan, PlatformClass::AndroidNative);
        assert!(is_gpu_usable(&caps).is_err());
    }

    #[test]
    fn is_gpu_usable_blocks_volcanic_android_vulkan() {
        let caps = mock_caps(GpuFamily::PowerVrVolcanic, wgpu::Backend::Vulkan, PlatformClass::AndroidNative);
        assert!(is_gpu_usable(&caps).is_err());
    }

    #[test]
    fn is_gpu_usable_allows_volcanic_browser() {
        let caps = mock_caps(GpuFamily::PowerVrVolcanic, wgpu::Backend::BrowserWebGpu, PlatformClass::Browser);
        assert!(is_gpu_usable(&caps).is_ok());
    }

    #[test]
    fn is_gpu_usable_allows_rogue_browser() {
        let caps = mock_caps(GpuFamily::PowerVrRogue, wgpu::Backend::BrowserWebGpu, PlatformClass::Browser);
        assert!(is_gpu_usable(&caps).is_ok());
    }

    #[test]
    fn is_gpu_usable_allows_volcanic_desktop() {
        let caps = mock_caps(GpuFamily::PowerVrVolcanic, wgpu::Backend::Vulkan, PlatformClass::DesktopIntegrated);
        assert!(is_gpu_usable(&caps).is_ok());
    }

    #[test]
    fn is_gpu_usable_allows_adreno() {
        let caps = mock_caps_with_family(GpuFamily::Adreno);
        assert!(is_gpu_usable(&caps).is_ok());
    }

    #[test]
    fn is_gpu_usable_allows_mali() {
        let caps = mock_caps_with_family(GpuFamily::Mali);
        assert!(is_gpu_usable(&caps).is_ok());
    }

    #[test]
    fn is_gpu_usable_allows_unknown() {
        let caps = mock_caps_with_family(GpuFamily::Unknown);
        assert!(is_gpu_usable(&caps).is_ok());
    }
}
