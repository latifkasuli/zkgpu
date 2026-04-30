//! Adapter scoring — ranks enumerated adapters for compute suitability.
//!
//! Produces an integer score for each candidate adapter so the device
//! constructor can pick the best GPU without relying on wgpu's opaque
//! `request_adapter(HighPerformance)` heuristic.
//!
//! Scoring criteria (descending priority):
//!
//! 1. **Device type** — discrete GPUs have dedicated VRAM and higher
//!    throughput than integrated GPUs for large NTTs.
//! 2. **Backend** — on systems where a GPU is reachable via multiple
//!    APIs (e.g. Vulkan *and* DX12 on Windows), prefer the backend
//!    with better wgpu compute support.
//! 3. **Compute features** — subgroup ops and timestamp queries are
//!    useful but not decisive.
//! 4. **Buffer capacity** — larger max buffer means larger NTTs.

use super::profile::CapabilityProfile;
use super::quirks;

/// Score an adapter for compute suitability.
///
/// Higher is better. Returns `None` if the adapter is known-broken
/// (e.g. PowerVR Rogue on Android Vulkan) and must not be used.
pub(crate) fn score_adapter(caps: &CapabilityProfile, adapter_limits: &wgpu::Limits) -> Option<i64> {
    // Reject known-broken drivers outright — no score assigned.
    if quirks::is_gpu_usable(caps).is_err() {
        return None;
    }

    let mut score: i64 = 0;

    // --- Device type (dominant factor) ---
    //
    // Discrete GPUs have dedicated high-bandwidth VRAM and many more
    // compute units than integrated GPUs. For large NTTs (2^20+) they
    // are substantially faster.
    match caps.device_type {
        wgpu::DeviceType::DiscreteGpu => score += 10_000,
        wgpu::DeviceType::IntegratedGpu => score += 5_000,
        wgpu::DeviceType::VirtualGpu => score += 2_000,
        _ => score += 1_000,
    }

    // --- Backend preference ---
    //
    // When the same physical GPU is visible via multiple APIs, prefer
    // the backend with better wgpu compute support:
    // - Metal: best on Apple; the only option in practice.
    // - Vulkan: mature compute path in wgpu, widest feature coverage.
    // - DX12: functional but fewer wgpu optimisations than Vulkan.
    // - BrowserWebGpu: limited capabilities, no enumerate_adapters.
    match caps.backend {
        wgpu::Backend::Metal => score += 500,
        wgpu::Backend::Vulkan => score += 400,
        wgpu::Backend::Dx12 => score += 300,
        wgpu::Backend::BrowserWebGpu => score += 200,
        _ => score += 100,
    }

    // --- Compute features (tiebreakers) ---
    if caps.has_subgroup {
        score += 100;
    }
    if caps.has_timestamp_query {
        score += 25;
    }

    // --- Buffer capacity ---
    //
    // Larger max_storage_buffer_binding_size enables larger NTTs.
    // Award 1 point per 16 MB, capped at 256 points (4 GB).
    let buf_mb = adapter_limits.max_storage_buffer_binding_size / (1024 * 1024);
    score += (buf_mb / 16).min(256) as i64;

    Some(score)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caps::types::*;

    fn mock_caps(
        device_type: wgpu::DeviceType,
        backend: wgpu::Backend,
        gpu_family: GpuFamily,
        platform_class: PlatformClass,
    ) -> CapabilityProfile {
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
            detection_source: DetectionSource::Unknown,
            platform_class,
            memory_model: MemoryModel::Unknown,
            has_subgroup: false,
            min_subgroup_size: 0,
            max_subgroup_size: 0,
            has_timestamp_query: false,
            has_timestamp_query_inside_passes: false,
            has_mappable_primary_buffers: false,
            has_pipeline_cache: false,
            has_immediates: false,
            has_shader_int64: false,
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

    fn default_limits() -> wgpu::Limits {
        wgpu::Limits {
            max_storage_buffer_binding_size: 128 * 1024 * 1024,
            ..wgpu::Limits::default()
        }
    }

    #[test]
    fn discrete_beats_integrated() {
        let discrete = mock_caps(
            wgpu::DeviceType::DiscreteGpu,
            wgpu::Backend::Vulkan,
            GpuFamily::Nvidia,
            PlatformClass::DesktopDiscrete,
        );
        let integrated = mock_caps(
            wgpu::DeviceType::IntegratedGpu,
            wgpu::Backend::Vulkan,
            GpuFamily::IntelIntegrated,
            PlatformClass::DesktopIntegrated,
        );
        let lim = default_limits();
        assert!(score_adapter(&discrete, &lim).unwrap() > score_adapter(&integrated, &lim).unwrap());
    }

    #[test]
    fn vulkan_beats_dx12_same_gpu() {
        let vulkan = mock_caps(
            wgpu::DeviceType::DiscreteGpu,
            wgpu::Backend::Vulkan,
            GpuFamily::Nvidia,
            PlatformClass::DesktopDiscrete,
        );
        let dx12 = mock_caps(
            wgpu::DeviceType::DiscreteGpu,
            wgpu::Backend::Dx12,
            GpuFamily::Nvidia,
            PlatformClass::DesktopDiscrete,
        );
        let lim = default_limits();
        assert!(score_adapter(&vulkan, &lim).unwrap() > score_adapter(&dx12, &lim).unwrap());
    }

    #[test]
    fn known_broken_driver_returns_none() {
        let caps = mock_caps(
            wgpu::DeviceType::IntegratedGpu,
            wgpu::Backend::Vulkan,
            GpuFamily::PowerVrRogue,
            PlatformClass::AndroidNative,
        );
        let lim = default_limits();
        assert!(score_adapter(&caps, &lim).is_none());
    }

    #[test]
    fn subgroup_adds_score() {
        let mut with = mock_caps(
            wgpu::DeviceType::IntegratedGpu,
            wgpu::Backend::Vulkan,
            GpuFamily::Adreno,
            PlatformClass::AndroidNative,
        );
        with.has_subgroup = true;
        let without = mock_caps(
            wgpu::DeviceType::IntegratedGpu,
            wgpu::Backend::Vulkan,
            GpuFamily::Adreno,
            PlatformClass::AndroidNative,
        );
        let lim = default_limits();
        assert!(score_adapter(&with, &lim).unwrap() > score_adapter(&without, &lim).unwrap());
    }

    #[test]
    fn larger_buffer_adds_score() {
        let caps = mock_caps(
            wgpu::DeviceType::DiscreteGpu,
            wgpu::Backend::Vulkan,
            GpuFamily::Nvidia,
            PlatformClass::DesktopDiscrete,
        );
        let small = wgpu::Limits {
            max_storage_buffer_binding_size: 128 * 1024 * 1024,
            ..wgpu::Limits::default()
        };
        let large = wgpu::Limits {
            max_storage_buffer_binding_size: 512 * 1024 * 1024,
            ..wgpu::Limits::default()
        };
        assert!(
            score_adapter(&caps, &large).unwrap() > score_adapter(&caps, &small).unwrap()
        );
    }
}
