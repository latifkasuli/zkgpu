use zkgpu_wgpu::{CapabilityProfile, WgpuDevice};

use crate::report::DeviceReport;

pub fn build_device_report(device: &WgpuDevice) -> DeviceReport {
    let caps: &CapabilityProfile = device.caps();
    let mut feature_flags = Vec::new();
    if caps.has_subgroup {
        feature_flags.push("subgroup".to_string());
    }
    if caps.has_timestamp_query {
        feature_flags.push("timestamp_query".to_string());
    }
    if caps.has_timestamp_query_inside_passes {
        feature_flags.push("timestamp_query_inside_passes".to_string());
    }
    if caps.has_mappable_primary_buffers {
        feature_flags.push("mappable_primary_buffers".to_string());
    }
    if caps.has_pipeline_cache {
        feature_flags.push("pipeline_cache".to_string());
    }

    DeviceReport {
        name: caps.device_name.clone(),
        backend: format!("{:?}", caps.backend),
        tier: format!("{:?}", caps.tier),
        gpu_family: format!("{:?}", caps.gpu_family),
        detection_source: format!("{:?}", caps.detection_source),
        platform_class: format!("{:?}", caps.platform_class),
        memory_model: format!("{:?}", caps.memory_model),
        driver: caps.driver.clone(),
        driver_info: caps.driver_info.clone(),
        max_buffer_size_bytes: caps.max_buffer_size,
        max_workgroup_size_x: caps.max_compute_workgroup_size_x,
        max_compute_invocations: caps.max_compute_invocations_per_workgroup,
        feature_flags,
    }
}
