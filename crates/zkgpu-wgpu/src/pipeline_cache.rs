use std::path::PathBuf;

use crate::caps::CapabilityProfile;

/// Load a persistent pipeline cache from disk if available.
///
/// The cache is keyed by a fingerprint of the GPU device (backend, vendor,
/// device name) to prevent cross-device cache corruption.
///
/// Returns `None` if the device doesn't support pipeline caching (i.e.
/// non-Vulkan backends where the parameter is ignored anyway).
pub(crate) fn load_cache(
    device: &wgpu::Device,
    caps: &CapabilityProfile,
) -> Option<wgpu::PipelineCache> {
    // Pipeline caching only benefits Vulkan on Android; desktop drivers
    // cache internally. We still create the cache on all Vulkan backends
    // since it's harmless and helps CI parity.
    if caps.backend != wgpu::Backend::Vulkan {
        return None;
    }

    let path = cache_path(caps)?;
    let data = std::fs::read(&path).ok();
    // SAFETY: `fallback: true` ensures the driver validates the cache data
    // and silently discards it if corrupt or incompatible, so providing
    // potentially stale data from disk cannot cause undefined behaviour.
    let cache = unsafe {
        device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
            label: Some("zkgpu"),
            data: data.as_deref(),
            fallback: true,
        })
    };
    Some(cache)
}

/// Persist the pipeline cache to disk for faster startup on subsequent runs.
///
/// Silently ignores write failures (e.g. no write permission, read-only FS).
pub(crate) fn save_cache(cache: &wgpu::PipelineCache, caps: &CapabilityProfile) {
    if let Some(data) = cache.get_data() {
        if let Some(path) = cache_path(caps) {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let _ = std::fs::write(&path, &data);
            log::debug!("zkgpu: saved pipeline cache to {}", path.display());
        }
    }
}

/// Compute the disk path for the pipeline cache file.
///
/// Returns `None` if no suitable cache directory can be determined.
fn cache_path(caps: &CapabilityProfile) -> Option<PathBuf> {
    // Build a device fingerprint from backend + vendor + device name.
    // This is intentionally coarse: driver version changes should not
    // invalidate the cache (Vulkan validates cache headers internally).
    let fingerprint = format!(
        "{:?}-{:04x}-{}",
        caps.backend, caps.vendor_id, caps.device_name
    );
    // Simple hash to keep filenames short and filesystem-safe.
    let hash = fingerprint
        .bytes()
        .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));

    // Try platform cache dir, fall back to current directory.
    let base = dirs_fallback_cache_dir();
    let dir = base.join("zkgpu").join("pipeline-cache");
    Some(dir.join(format!("{hash:016x}.bin")))
}

/// Best-effort cache directory without pulling in the `dirs` crate.
fn dirs_fallback_cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("ZKGPU_CACHE_DIR") {
        return PathBuf::from(dir);
    }
    #[cfg(target_os = "android")]
    {
        // Android: TMPDIR is set by the runtime to the app's cache dir.
        // Failing that, try the app's private files directory via HOME
        // (set by some Android runtimes), then /data/local/tmp for dev.
        if let Ok(tmp) = std::env::var("TMPDIR") {
            return PathBuf::from(tmp);
        }
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join("cache");
        }
        return PathBuf::from("/data/local/tmp");
    }
    #[cfg(target_os = "macos")]
    {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join("Library").join("Caches");
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
            return PathBuf::from(xdg);
        }
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join(".cache");
        }
    }
    #[cfg(target_os = "windows")]
    {
        if let Ok(local) = std::env::var("LOCALAPPDATA") {
            return PathBuf::from(local);
        }
    }
    PathBuf::from(".")
}
