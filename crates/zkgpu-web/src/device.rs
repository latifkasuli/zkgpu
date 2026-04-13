//! Global device state for the web harness.
//!
//! The device is initialized once via `init_device()` and stored in a
//! thread-local `Rc`. All subsequent operations clone the `Rc` handle
//! so they hold an owned reference that survives across await points.
//! If a second `gpu_init()` replaces the device, outstanding `Rc`
//! handles keep the old device alive — no dangling references.

use std::cell::RefCell;
use std::rc::Rc;

use zkgpu_report::DeviceReport;
use zkgpu_wgpu::{CapabilityProfile, WgpuDevice};

thread_local! {
    static DEVICE: RefCell<Option<Rc<WgpuDevice>>> = const { RefCell::new(None) };
}

/// Initialize the global device. Returns JSON device info.
pub(crate) async fn init_device() -> Result<String, String> {
    let device = WgpuDevice::new_async()
        .await
        .map_err(|e| format!("failed to create GPU device: {e}"))?;

    let report = build_device_report(&device);
    let json =
        serde_json::to_string(&report).map_err(|e| format!("failed to serialize device info: {e}"))?;

    let rc = Rc::new(device);
    DEVICE.with(|cell| {
        *cell.borrow_mut() = Some(rc);
    });

    Ok(json)
}

/// Get cached device info as JSON. Fails if `init_device` was not called.
pub(crate) fn get_device_info_json() -> Result<String, String> {
    DEVICE.with(|cell| {
        let borrow = cell.borrow();
        let device = borrow
            .as_ref()
            .ok_or("GPU device not initialized — call gpu_init() first")?;
        let report = build_device_report(device);
        serde_json::to_string(&report).map_err(|e| format!("failed to serialize: {e}"))
    })
}

/// Clone the `Rc<WgpuDevice>` out of thread-local storage.
///
/// The returned handle is an owned `Rc` that keeps the device alive
/// even if a subsequent `init_device()` replaces the thread-local.
/// This is the safe alternative to raw pointers across await points.
pub(crate) fn clone_device() -> Result<Rc<WgpuDevice>, String> {
    DEVICE.with(|cell| {
        cell.borrow()
            .as_ref()
            .cloned()
            .ok_or_else(|| "GPU device not initialized — call gpu_init() first".to_string())
    })
}

fn build_device_report(device: &WgpuDevice) -> DeviceReport {
    let caps: &CapabilityProfile = device.caps();
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
        feature_flags: caps.feature_flags_flat(),
    }
}
