/// Hardware capability tier — determines which kernel variants are available.
///
/// Tiers form a capability ladder. Higher tiers imply all capabilities
/// of lower tiers plus additional features. Kernel selection and buffer
/// strategy are driven by the tier assigned at device initialization.
///
/// This is a coarse capability bucket, not the full planner identity.
/// For planner decisions, use `GpuFamily` and `PlatformClass` alongside
/// the tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DeviceTier {
    /// Browser WebGPU or minimal native. WGSL portable baseline only.
    PortableWeb,
    /// Native wgpu without advanced compute features.
    NativeBasic,
    /// Native with subgroup operations (wave/SIMD intrinsics).
    NativeSubgroup,
    /// Native on unified memory architecture (Apple Silicon, some integrated GPUs).
    UnifiedMemoryNative,
}

/// Normalized GPU silicon family, derived primarily from vendor ID.
///
/// Used by the planner to apply per-family crossover thresholds.
/// Device-name matching is a fallback, not the primary identity key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuFamily {
    Apple,
    Adreno,
    Mali,
    /// Imagination Rogue architecture (GE/GM/GX series).
    /// Old, budget-tier GPUs found in Unisoc/MediaTek SoCs.
    /// Tested: GE8322 on Xiaomi Redmi (SP9863A) — SIGSEGV + silent corruption.
    PowerVrRogue,
    /// Imagination Volcanic+ architecture (BXE/BXM/BXT/CXT/DXT/AXE/AXM/AXT).
    /// Modern GPUs used in Google Tensor G5 (Pixel 10).
    /// Tested: DXT-48 on Pixel 10 Pro Fold — SIGSEGV in vkQueueSubmit.
    PowerVrVolcanic,
    Xclipse,
    IntelIntegrated,
    IntelDiscrete,
    Amd,
    Nvidia,
    Unknown,
}

/// How the GPU family was determined.
///
/// Included in device reports for field diagnostics. Not used by
/// the planner — family identity drives policy, not detection method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DetectionSource {
    /// Matched a well-known PCI vendor ID (most reliable).
    VendorId,
    /// Vendor ID unknown or zero; matched from device name string.
    NameFallback,
    /// Metal backend with unrecognised name; defaulted to Apple.
    MetalDefault,
    /// No signal matched.
    Unknown,
}

/// Deployment context that affects runtime behavior.
///
/// The same GPU silicon can behave differently depending on whether it
/// is in a phone, a desktop, or a browser sandbox.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlatformClass {
    Browser,
    AppleNative,
    AndroidNative,
    DesktopIntegrated,
    DesktopDiscrete,
    UnknownNative,
}

/// Known driver-level quirks that affect whether compute dispatch is safe.
///
/// Populated by [`driver_quirks`] from the GPU family, backend, platform,
/// and driver strings. The canary checks these *before* attempting any
/// `vkQueueSubmit` to avoid process-fatal SIGSEGV on broken drivers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct DriverQuirks {
    /// Driver is known to crash (SIGSEGV / null-deref) on compute dispatch.
    ///
    /// When true, the GPU must not be used at all — even a trivial canary
    /// dispatch will kill the process on Android (no SIGSEGV recovery).
    ///
    /// Scoped to the failure signature actually observed on real hardware:
    /// - Rogue GE8322 on Xiaomi Redmi (24.2@6643903): SIGSEGV on 2nd
    ///   vkQueueSubmit + silent arithmetic corruption
    /// - Volcanic DXT-48 on Pixel 10 Pro Fold (24.3@6660496): SIGSEGV
    ///   in IMG_vkQueueSubmit+1524
    ///
    /// Only set for Android Vulkan. Not set for Browser WebGPU (browser
    /// provides crash isolation) or desktop (no crash evidence).
    pub may_crash_on_compute: bool,
}

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

/// Memory architecture model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryModel {
    Unified,
    Discrete,
    Unknown,
}

/// Snapshot of a GPU adapter's capabilities, captured once at device init.
///
/// Drives kernel selection, buffer strategy, profiling mode, and limit
/// validation. All fields are derived from `wgpu::Adapter` introspection.
#[derive(Debug, Clone)]
pub struct CapabilityProfile {
    pub tier: DeviceTier,
    pub backend: wgpu::Backend,
    pub device_type: wgpu::DeviceType,

    pub vendor_id: u32,
    pub device_id: u32,
    pub device_name: String,
    pub driver: String,
    pub driver_info: String,

    pub gpu_family: GpuFamily,
    pub detection_source: DetectionSource,
    pub platform_class: PlatformClass,
    pub memory_model: MemoryModel,

    pub has_subgroup: bool,
    pub min_subgroup_size: u32,
    pub max_subgroup_size: u32,
    pub has_timestamp_query: bool,
    pub has_timestamp_query_inside_passes: bool,
    pub has_mappable_primary_buffers: bool,
    pub has_pipeline_cache: bool,

    pub max_buffer_size: u64,
    pub max_storage_buffer_binding_size: u32,
    pub max_compute_workgroup_size_x: u32,
    pub max_compute_workgroup_size_y: u32,
    pub max_compute_workgroup_size_z: u32,
    pub max_compute_invocations_per_workgroup: u32,
    pub max_compute_workgroups_per_dimension: u32,
}

impl CapabilityProfile {
    pub fn from_adapter(adapter: &wgpu::Adapter) -> Self {
        let info = adapter.get_info();
        let features = adapter.features();

        let has_subgroup = features.contains(wgpu::Features::SUBGROUP);
        let has_timestamp_query = features.contains(wgpu::Features::TIMESTAMP_QUERY);
        let has_timestamp_inside =
            features.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES);
        let has_mappable = features.contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS);
        let has_pipeline_cache = features.contains(wgpu::Features::PIPELINE_CACHE);

        let tier = classify_tier(
            info.backend,
            info.device_type,
            has_subgroup,
            has_mappable,
        );

        let (gpu_family, detection_source) =
            classify_gpu_family(info.vendor, info.backend, &info.name);
        let platform_class = classify_platform_class(
            info.backend,
            info.device_type,
        );
        let memory_model = classify_memory_model(info.device_type, has_mappable);

        Self {
            tier,
            backend: info.backend,
            device_type: info.device_type,
            vendor_id: info.vendor,
            device_id: info.device,
            device_name: info.name.clone(),
            driver: info.driver.clone(),
            driver_info: info.driver_info.clone(),
            gpu_family,
            detection_source,
            platform_class,
            memory_model,
            has_subgroup,
            min_subgroup_size: 0,
            max_subgroup_size: 0,
            has_timestamp_query,
            has_timestamp_query_inside_passes: has_timestamp_inside,
            has_mappable_primary_buffers: has_mappable,
            has_pipeline_cache,
            max_buffer_size: 0,
            max_storage_buffer_binding_size: 0,
            max_compute_workgroup_size_x: 0,
            max_compute_workgroup_size_y: 0,
            max_compute_workgroup_size_z: 0,
            max_compute_invocations_per_workgroup: 0,
            max_compute_workgroups_per_dimension: 0,
        }
    }

    /// Populate limit fields from the device that was actually created.
    ///
    /// `request_device` may grant limits equal to or above the requested
    /// floor, but never above the adapter maximum. These are the limits
    /// that wgpu will enforce at runtime, so all buffer-size validation
    /// and `DeviceInfo` reporting must use these, not the adapter maxima.
    pub fn apply_device_limits(&mut self, limits: &wgpu::Limits) {
        self.max_buffer_size = limits.max_buffer_size;
        self.max_storage_buffer_binding_size = limits.max_storage_buffer_binding_size;
        self.max_compute_workgroup_size_x = limits.max_compute_workgroup_size_x;
        self.max_compute_workgroup_size_y = limits.max_compute_workgroup_size_y;
        self.max_compute_workgroup_size_z = limits.max_compute_workgroup_size_z;
        self.max_compute_invocations_per_workgroup = limits.max_compute_invocations_per_workgroup;
        self.max_compute_workgroups_per_dimension = limits.max_compute_workgroups_per_dimension;
        self.min_subgroup_size = limits.min_subgroup_size;
        self.max_subgroup_size = limits.max_subgroup_size;
    }

    /// The set of wgpu features this runtime should request from the adapter.
    ///
    /// Only requests features that the adapter actually supports AND that
    /// the runtime knows how to use. Does not request features that would
    /// change kernel correctness requirements (e.g., SHADER_INT64 is reserved
    /// for when we have kernel variants that use it).
    pub fn required_features(&self) -> wgpu::Features {
        let mut f = wgpu::Features::empty();
        if self.has_subgroup {
            f |= wgpu::Features::SUBGROUP;
        }
        if self.has_timestamp_query {
            f |= wgpu::Features::TIMESTAMP_QUERY;
        }
        if self.has_timestamp_query_inside_passes {
            f |= wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES;
        }
        if self.has_pipeline_cache {
            f |= wgpu::Features::PIPELINE_CACHE;
        }
        f
    }

    /// The minimum wgpu limits this runtime actually needs, clamped to
    /// what the adapter can provide.
    ///
    /// Requests up to 256 MB for storage buffers (NTT up to 2^26
    /// BabyBear elements), but clamps to the adapter maximum on devices
    /// with tighter limits (e.g. Adreno at 128 MB).
    ///
    /// Callers that need larger buffers (e.g. 2^20+ element NTTs) can
    /// check `CapabilityProfile` fields directly and fail gracefully.
    pub fn required_limits(&self, adapter_limits: &wgpu::Limits) -> wgpu::Limits {
        let defaults = wgpu::Limits::downlevel_defaults();
        let desired_buffer: u32 = 256 * 1024 * 1024;
        wgpu::Limits {
            max_storage_buffer_binding_size: desired_buffer
                .min(adapter_limits.max_storage_buffer_binding_size),
            max_buffer_size: (desired_buffer as u64).min(adapter_limits.max_buffer_size),
            max_compute_workgroup_size_x: 256,
            max_compute_invocations_per_workgroup: 256,
            ..defaults
        }
    }

    /// The effective maximum size (in bytes) for a single storage buffer
    /// that will be bound via `as_entire_binding()`.
    ///
    /// This is `min(max_buffer_size, max_storage_buffer_binding_size)` —
    /// a buffer can be created up to `max_buffer_size`, but binding it as
    /// a full storage buffer is capped at `max_storage_buffer_binding_size`.
    pub fn max_storage_buffer_size(&self) -> u64 {
        self.max_buffer_size
            .min(self.max_storage_buffer_binding_size as u64)
    }

    fn feature_summary(&self) -> String {
        let mut flags = Vec::new();
        if self.has_subgroup {
            flags.push(format!(
                "subgroup({}-{})",
                self.min_subgroup_size, self.max_subgroup_size
            ));
        }
        if self.has_timestamp_query {
            flags.push("timestamp".to_string());
        }
        if self.has_mappable_primary_buffers {
            flags.push("mappable".to_string());
        }
        if self.has_pipeline_cache {
            flags.push("pipeline_cache".to_string());
        }
        if flags.is_empty() {
            "none".to_string()
        } else {
            flags.join(", ")
        }
    }
}

impl std::fmt::Display for CapabilityProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} ({:?}/{:?}) tier={:?} family={:?} platform={:?} memory={:?} \
             features=[{}] buffer={}MB workgroup={}",
            self.device_name,
            self.backend,
            self.device_type,
            self.tier,
            self.gpu_family,
            self.platform_class,
            self.memory_model,
            self.feature_summary(),
            self.max_buffer_size / (1024 * 1024),
            self.max_compute_workgroup_size_x,
        )
    }
}

fn classify_tier(
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

// Well-known PCI vendor IDs.
const VENDOR_APPLE: u32 = 0x106B;
const VENDOR_QUALCOMM: u32 = 0x5143;
const VENDOR_ARM: u32 = 0x13B5;
const VENDOR_IMAGINATION: u32 = 0x1010;
const VENDOR_SAMSUNG: u32 = 0x144D;
const VENDOR_INTEL: u32 = 0x8086;
const VENDOR_AMD: u32 = 0x1002;
const VENDOR_NVIDIA: u32 = 0x10DE;

fn classify_gpu_family(
    vendor_id: u32,
    backend: wgpu::Backend,
    device_name: &str,
) -> (GpuFamily, DetectionSource) {
    match vendor_id {
        VENDOR_APPLE => (GpuFamily::Apple, DetectionSource::VendorId),
        VENDOR_QUALCOMM => (GpuFamily::Adreno, DetectionSource::VendorId),
        VENDOR_ARM => (GpuFamily::Mali, DetectionSource::VendorId),
        VENDOR_IMAGINATION => {
            // Vendor ID confirms Imagination; inspect name for Rogue vs Volcanic.
            let sub = classify_img_subfamily(device_name);
            (sub, DetectionSource::VendorId)
        }
        VENDOR_SAMSUNG => (GpuFamily::Xclipse, DetectionSource::VendorId),
        VENDOR_NVIDIA => (GpuFamily::Nvidia, DetectionSource::VendorId),
        VENDOR_AMD => (GpuFamily::Amd, DetectionSource::VendorId),
        VENDOR_INTEL => {
            let name = device_name.to_lowercase();
            if name.contains(" arc ") || name.contains(" arc(") || name.ends_with(" arc") {
                (GpuFamily::IntelDiscrete, DetectionSource::VendorId)
            } else {
                (GpuFamily::IntelIntegrated, DetectionSource::VendorId)
            }
        }
        _ => {
            let by_name = classify_gpu_family_by_name(device_name);
            if by_name != GpuFamily::Unknown {
                (by_name, DetectionSource::NameFallback)
            } else if backend == wgpu::Backend::Metal {
                (GpuFamily::Apple, DetectionSource::MetalDefault)
            } else {
                (GpuFamily::Unknown, DetectionSource::Unknown)
            }
        }
    }
}

/// Last-resort fallback: match GPU family from device name when vendor ID
/// is missing or non-standard (e.g. some Android Vulkan drivers report 0).
///
/// Uses word-boundary-aware matching so short tokens like "arc" or "img"
/// don't produce false positives from substrings. Imagination model
/// prefixes (GE, BXM, DXT, etc.) are matched when followed by a digit,
/// catching bare model numbers like "GE8322" that lack a "PowerVR" prefix.
fn classify_gpu_family_by_name(device_name: &str) -> GpuFamily {
    let name = device_name.to_lowercase();

    // --- Brand-name matches (unambiguous, checked first) ---
    if name.contains("adreno") {
        return GpuFamily::Adreno;
    }
    if name.contains("mali") || name.contains("immortalis") {
        return GpuFamily::Mali;
    }
    if name.contains("xclipse") {
        return GpuFamily::Xclipse;
    }
    if name.contains("apple") {
        return GpuFamily::Apple;
    }

    // --- Imagination / PowerVR ---
    // Any Imagination-related keyword or model prefix → classify subfamily.
    if name.contains("powervr") || name.starts_with("img ") || name.contains(" img ")
        || has_img_model_prefix(&name)
    {
        return classify_img_subfamily(device_name);
    }

    // --- Desktop GPU brands ---
    if name.contains("nvidia") || name.contains("geforce") || has_word(&name, "rtx") || has_word(&name, "gtx") || name.contains("quadro") || name.contains("tesla") {
        return GpuFamily::Nvidia;
    }
    if name.contains("radeon") || name.contains("amd ") || name.starts_with("amd ") {
        return GpuFamily::Amd;
    }

    // --- Intel: "arc" requires "intel" context to avoid false positives ---
    if name.contains("intel") {
        if has_word(&name, "arc") {
            return GpuFamily::IntelDiscrete;
        }
        return GpuFamily::IntelIntegrated;
    }

    GpuFamily::Unknown
}

/// Check if `name` contains a word `w` at a word boundary.
///
/// A "word boundary" is start/end of string or any non-alphanumeric char.
/// This prevents "arc" matching inside "search" or "rtx" inside "cortx".
fn has_word(name: &str, w: &str) -> bool {
    for (i, _) in name.match_indices(w) {
        let before_ok = i == 0 || !name.as_bytes()[i - 1].is_ascii_alphanumeric();
        let end = i + w.len();
        let after_ok = end == name.len() || !name.as_bytes()[end].is_ascii_alphanumeric();
        if before_ok && after_ok {
            return true;
        }
    }
    false
}

/// Check if `name` contains an Imagination Technologies model prefix
/// followed by a digit (e.g. "ge8322", "bxm8", "dxt48").
///
/// Covers all known Imagination GPU model families:
/// - Rogue:   GE, GM, GX
/// - Volcanic / IMG-C / IMG-D / IMG-A:
///            BXE, BXM, BXS, BXT, CXT, CXTP, DXT, DXTP, DXS, AXE, AXM, AXT
fn has_img_model_prefix(name: &str) -> bool {
    // Sorted longest-first so "cxtp" is tried before "cxt".
    const PREFIXES: &[&str] = &[
        "cxtp", "dxtp", "bxe", "bxm", "bxs", "bxt", "cxt", "dxt", "dxs",
        "axe", "axm", "axt", "ge", "gm", "gx",
    ];
    let bytes = name.as_bytes();
    for prefix in PREFIXES {
        for (i, _) in name.match_indices(prefix) {
            // Must be at word boundary before the prefix
            let before_ok = i == 0 || !bytes[i - 1].is_ascii_alphanumeric();
            if !before_ok {
                continue;
            }
            // Must be followed by a digit, either directly (GE8322)
            // or after a hyphen separator (BXE-2-32, BXM-8-256).
            let end = i + prefix.len();
            if end < bytes.len() && bytes[end].is_ascii_digit() {
                return true;
            }
            if end + 1 < bytes.len() && bytes[end] == b'-' && bytes[end + 1].is_ascii_digit() {
                return true;
            }
        }
    }
    false
}

/// Rogue-era Imagination model prefixes (followed by digit).
const ROGUE_PREFIXES: &[&str] = &["ge", "gm", "gx"];

/// Distinguish Imagination Rogue vs Volcanic from the device name.
///
/// - Explicit "rogue" keyword → `PowerVrRogue`
/// - Rogue model prefix (GE/GM/GX + digit) → `PowerVrRogue`
/// - Anything else (Volcanic model prefix, "powervr" without "rogue",
///   "IMG" prefix) → `PowerVrVolcanic` (modern default)
fn classify_img_subfamily(device_name: &str) -> GpuFamily {
    let name = device_name.to_lowercase();

    // Explicit "rogue" keyword in the name
    if name.contains("rogue") {
        return GpuFamily::PowerVrRogue;
    }

    // Check for Rogue-era model prefixes (GE/GM/GX + digit)
    let bytes = name.as_bytes();
    for prefix in ROGUE_PREFIXES {
        for (i, _) in name.match_indices(prefix) {
            let before_ok = i == 0 || !bytes[i - 1].is_ascii_alphanumeric();
            if !before_ok {
                continue;
            }
            let end = i + prefix.len();
            if end < bytes.len() && bytes[end].is_ascii_digit() {
                return GpuFamily::PowerVrRogue;
            }
            if end + 1 < bytes.len() && bytes[end] == b'-' && bytes[end + 1].is_ascii_digit() {
                return GpuFamily::PowerVrRogue;
            }
        }
    }

    // Everything else: Volcanic / modern Imagination
    GpuFamily::PowerVrVolcanic
}

/// Classify the deployment context from backend, device type, and build
/// target.
///
/// Uses `#[cfg(target_os)]` to distinguish Android from desktop Vulkan,
/// so an Adreno on a Snapdragon X Windows laptop correctly gets
/// `DesktopIntegrated` instead of `AndroidNative`.
fn classify_platform_class(
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

fn classify_memory_model(
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

    // === classify_gpu_family: vendor-ID path ===

    #[test]
    fn gpu_family_from_qualcomm_vendor_id() {
        assert_eq!(
            classify_gpu_family(VENDOR_QUALCOMM, wgpu::Backend::Vulkan, "Adreno (TM) 750"),
            (GpuFamily::Adreno, DetectionSource::VendorId)
        );
    }

    #[test]
    fn gpu_family_from_arm_vendor_id() {
        assert_eq!(
            classify_gpu_family(VENDOR_ARM, wgpu::Backend::Vulkan, "Mali-G720"),
            (GpuFamily::Mali, DetectionSource::VendorId)
        );
    }

    #[test]
    fn gpu_family_from_imagination_vendor_id_volcanic() {
        // Generic "PowerVR" without "Rogue" → Volcanic (modern default)
        assert_eq!(
            classify_gpu_family(VENDOR_IMAGINATION, wgpu::Backend::Vulkan, "PowerVR"),
            (GpuFamily::PowerVrVolcanic, DetectionSource::VendorId)
        );
    }

    #[test]
    fn gpu_family_from_imagination_vendor_id_rogue() {
        assert_eq!(
            classify_gpu_family(VENDOR_IMAGINATION, wgpu::Backend::Vulkan, "PowerVR Rogue GE8320"),
            (GpuFamily::PowerVrRogue, DetectionSource::VendorId)
        );
    }

    #[test]
    fn gpu_family_from_imagination_vendor_id_dxt() {
        // Pixel 10 DXT-48 with vendor ID
        assert_eq!(
            classify_gpu_family(VENDOR_IMAGINATION, wgpu::Backend::Vulkan, "DXT-48-1536"),
            (GpuFamily::PowerVrVolcanic, DetectionSource::VendorId)
        );
    }

    #[test]
    fn gpu_family_from_samsung_vendor_id() {
        assert_eq!(
            classify_gpu_family(VENDOR_SAMSUNG, wgpu::Backend::Vulkan, "Xclipse 940"),
            (GpuFamily::Xclipse, DetectionSource::VendorId)
        );
    }

    #[test]
    fn gpu_family_apple_vendor_id_is_apple() {
        assert_eq!(
            classify_gpu_family(VENDOR_APPLE, wgpu::Backend::Vulkan, "Apple M4 Pro"),
            (GpuFamily::Apple, DetectionSource::VendorId)
        );
    }

    #[test]
    fn gpu_family_nvidia_vendor_id() {
        assert_eq!(
            classify_gpu_family(VENDOR_NVIDIA, wgpu::Backend::Vulkan, "NVIDIA RTX 4090"),
            (GpuFamily::Nvidia, DetectionSource::VendorId)
        );
    }

    #[test]
    fn gpu_family_amd_vendor_id() {
        assert_eq!(
            classify_gpu_family(VENDOR_AMD, wgpu::Backend::Vulkan, "Radeon RX 7900"),
            (GpuFamily::Amd, DetectionSource::VendorId)
        );
    }

    #[test]
    fn gpu_family_intel_integrated() {
        assert_eq!(
            classify_gpu_family(VENDOR_INTEL, wgpu::Backend::Vulkan, "Intel Iris Xe"),
            (GpuFamily::IntelIntegrated, DetectionSource::VendorId)
        );
    }

    #[test]
    fn gpu_family_intel_arc_discrete() {
        assert_eq!(
            classify_gpu_family(VENDOR_INTEL, wgpu::Backend::Vulkan, "Intel Arc A770"),
            (GpuFamily::IntelDiscrete, DetectionSource::VendorId)
        );
        assert_eq!(
            classify_gpu_family(VENDOR_INTEL, wgpu::Backend::Vulkan, "Intel(R) Arc(TM) A580"),
            (GpuFamily::IntelDiscrete, DetectionSource::VendorId)
        );
    }

    #[test]
    fn gpu_family_intel_arc_no_false_positive() {
        // "search" contains "arc" as substring — must not match
        assert_eq!(
            classify_gpu_family(VENDOR_INTEL, wgpu::Backend::Vulkan, "Intel UHD search mode"),
            (GpuFamily::IntelIntegrated, DetectionSource::VendorId)
        );
    }

    // === classify_gpu_family: Metal fallback ===

    #[test]
    fn gpu_family_metal_backend_is_apple() {
        assert_eq!(
            classify_gpu_family(0, wgpu::Backend::Metal, "Apple M4 Pro"),
            (GpuFamily::Apple, DetectionSource::NameFallback)
        );
    }

    #[test]
    fn gpu_family_metal_backend_unknown_name_is_apple() {
        assert_eq!(
            classify_gpu_family(0, wgpu::Backend::Metal, "Some Future GPU"),
            (GpuFamily::Apple, DetectionSource::MetalDefault)
        );
    }

    #[test]
    fn gpu_family_metal_backend_non_apple_name_uses_name() {
        assert_eq!(
            classify_gpu_family(0, wgpu::Backend::Metal, "AMD Radeon RX 6900"),
            (GpuFamily::Amd, DetectionSource::NameFallback)
        );
        assert_eq!(
            classify_gpu_family(0, wgpu::Backend::Metal, "Intel UHD 770"),
            (GpuFamily::IntelIntegrated, DetectionSource::NameFallback)
        );
    }

    #[test]
    fn gpu_family_amd_on_metal_is_amd() {
        assert_eq!(
            classify_gpu_family(VENDOR_AMD, wgpu::Backend::Metal, "AMD Radeon Pro 5500M"),
            (GpuFamily::Amd, DetectionSource::VendorId)
        );
    }

    #[test]
    fn gpu_family_intel_on_metal_is_intel() {
        assert_eq!(
            classify_gpu_family(VENDOR_INTEL, wgpu::Backend::Metal, "Intel UHD Graphics 630"),
            (GpuFamily::IntelIntegrated, DetectionSource::VendorId)
        );
    }

    // === classify_gpu_family: unknown vendor fallback ===

    #[test]
    fn gpu_family_unknown_vendor_falls_back_to_name() {
        assert_eq!(
            classify_gpu_family(0x9999, wgpu::Backend::Vulkan, "Adreno 640"),
            (GpuFamily::Adreno, DetectionSource::NameFallback)
        );
    }

    #[test]
    fn gpu_family_unknown_vendor_unknown_name() {
        assert_eq!(
            classify_gpu_family(0x9999, wgpu::Backend::Vulkan, "mystery device"),
            (GpuFamily::Unknown, DetectionSource::Unknown)
        );
    }

    // === classify_gpu_family_by_name: brand names ===

    #[test]
    fn name_fallback_adreno() {
        assert_eq!(classify_gpu_family_by_name("Adreno (TM) 750"), GpuFamily::Adreno);
        assert_eq!(classify_gpu_family_by_name("adreno 640"), GpuFamily::Adreno);
    }

    #[test]
    fn name_fallback_mali() {
        assert_eq!(classify_gpu_family_by_name("Mali-G720"), GpuFamily::Mali);
        assert_eq!(classify_gpu_family_by_name("ARM Mali-G78"), GpuFamily::Mali);
    }

    #[test]
    fn name_fallback_immortalis() {
        assert_eq!(classify_gpu_family_by_name("Immortalis-G925"), GpuFamily::Mali);
        assert_eq!(classify_gpu_family_by_name("ARM Immortalis-G720"), GpuFamily::Mali);
    }

    #[test]
    fn name_fallback_xclipse() {
        assert_eq!(classify_gpu_family_by_name("Samsung Xclipse 940"), GpuFamily::Xclipse);
    }

    // === classify_gpu_family_by_name: PowerVR / Imagination ===

    #[test]
    fn name_fallback_powervr_rogue_brand() {
        assert_eq!(classify_gpu_family_by_name("PowerVR Rogue GE8320"), GpuFamily::PowerVrRogue);
    }

    #[test]
    fn name_fallback_powervr_generic_is_volcanic() {
        // "PowerVR" without "Rogue" → Volcanic (modern default)
        assert_eq!(classify_gpu_family_by_name("PowerVR BXT-32-1024"), GpuFamily::PowerVrVolcanic);
    }

    #[test]
    fn name_fallback_img_prefix_is_volcanic() {
        // IMG branding is used for Volcanic-era parts
        assert_eq!(classify_gpu_family_by_name("IMG BXM-8-256"), GpuFamily::PowerVrVolcanic);
        assert_eq!(classify_gpu_family_by_name("img cxt48"), GpuFamily::PowerVrVolcanic);
    }

    #[test]
    fn name_fallback_imagination_rogue_model_numbers() {
        // Bare Rogue model numbers (no "PowerVR" prefix)
        assert_eq!(classify_gpu_family_by_name("GE8322"), GpuFamily::PowerVrRogue);
        assert_eq!(classify_gpu_family_by_name("GE8320"), GpuFamily::PowerVrRogue);
        assert_eq!(classify_gpu_family_by_name("GM9446"), GpuFamily::PowerVrRogue);
        assert_eq!(classify_gpu_family_by_name("GX6250"), GpuFamily::PowerVrRogue);
    }

    #[test]
    fn name_fallback_imagination_volcanic_model_numbers() {
        // Volcanic / IMG-C / IMG-D / IMG-A model numbers
        assert_eq!(classify_gpu_family_by_name("BXE-2-32"), GpuFamily::PowerVrVolcanic);
        assert_eq!(classify_gpu_family_by_name("BXM-8-256"), GpuFamily::PowerVrVolcanic);
        assert_eq!(classify_gpu_family_by_name("BXT-32-1024"), GpuFamily::PowerVrVolcanic);
        assert_eq!(classify_gpu_family_by_name("CXT48"), GpuFamily::PowerVrVolcanic);
        assert_eq!(classify_gpu_family_by_name("CXTP48"), GpuFamily::PowerVrVolcanic);
        assert_eq!(classify_gpu_family_by_name("DXT48"), GpuFamily::PowerVrVolcanic);
        assert_eq!(classify_gpu_family_by_name("DXTP48"), GpuFamily::PowerVrVolcanic);
        assert_eq!(classify_gpu_family_by_name("DXS36"), GpuFamily::PowerVrVolcanic);
        assert_eq!(classify_gpu_family_by_name("AXE16"), GpuFamily::PowerVrVolcanic);
        assert_eq!(classify_gpu_family_by_name("AXM8"), GpuFamily::PowerVrVolcanic);
        assert_eq!(classify_gpu_family_by_name("AXT48"), GpuFamily::PowerVrVolcanic);
    }

    #[test]
    fn name_fallback_imagination_model_no_false_positive() {
        // "gems" contains "ge" but not followed by digit
        assert_eq!(classify_gpu_family_by_name("gems rendering engine"), GpuFamily::Unknown);
        // "age" ends with "ge" — not at word boundary
        assert_eq!(classify_gpu_family_by_name("average gpu 3000"), GpuFamily::Unknown);
        // "gx" in the middle of a word
        assert_eq!(classify_gpu_family_by_name("nexgen hybrid"), GpuFamily::Unknown);
    }

    // === classify_gpu_family_by_name: desktop GPUs ===

    #[test]
    fn name_fallback_nvidia() {
        assert_eq!(classify_gpu_family_by_name("NVIDIA GeForce RTX 4090"), GpuFamily::Nvidia);
        assert_eq!(classify_gpu_family_by_name("nvidia rtx 5090"), GpuFamily::Nvidia);
        assert_eq!(classify_gpu_family_by_name("Quadro RTX 8000"), GpuFamily::Nvidia);
    }

    #[test]
    fn name_fallback_nvidia_rtx_word_boundary() {
        // "rtx" must be a word — "cortxl" should not match
        assert_eq!(classify_gpu_family_by_name("cortxl accelerator"), GpuFamily::Unknown);
    }

    #[test]
    fn name_fallback_amd() {
        assert_eq!(classify_gpu_family_by_name("AMD Radeon Graphics"), GpuFamily::Amd);
        assert_eq!(classify_gpu_family_by_name("AMD Custom GPU 0405"), GpuFamily::Amd);
    }

    #[test]
    fn name_fallback_intel_integrated() {
        assert_eq!(classify_gpu_family_by_name("Intel(R) Iris(R) Xe Graphics"), GpuFamily::IntelIntegrated);
    }

    #[test]
    fn name_fallback_intel_arc_is_discrete() {
        assert_eq!(classify_gpu_family_by_name("Intel Arc A770"), GpuFamily::IntelDiscrete);
        assert_eq!(classify_gpu_family_by_name("Intel(R) Arc(TM) A580"), GpuFamily::IntelDiscrete);
    }

    #[test]
    fn name_fallback_intel_arc_no_false_positive() {
        // "arc" without "intel" context should not match Intel
        assert_eq!(classify_gpu_family_by_name("ArcGPU 3000"), GpuFamily::Unknown);
    }

    // === classify_gpu_family_by_name: unknown ===

    #[test]
    fn name_fallback_unknown() {
        assert_eq!(classify_gpu_family_by_name("mock"), GpuFamily::Unknown);
        assert_eq!(classify_gpu_family_by_name("mystery device"), GpuFamily::Unknown);
    }

    // === classify_gpu_family: zero-vendor-ID integration ===

    #[test]
    fn name_fallback_immortalis_with_zero_vendor_id() {
        assert_eq!(
            classify_gpu_family(0, wgpu::Backend::Vulkan, "Immortalis-G925"),
            (GpuFamily::Mali, DetectionSource::NameFallback)
        );
    }

    #[test]
    fn name_fallback_intel_arc_with_zero_vendor_id() {
        assert_eq!(
            classify_gpu_family(0, wgpu::Backend::Vulkan, "Intel Arc A770"),
            (GpuFamily::IntelDiscrete, DetectionSource::NameFallback)
        );
    }

    #[test]
    fn name_fallback_bare_rogue_model_with_zero_vendor_id() {
        assert_eq!(
            classify_gpu_family(0, wgpu::Backend::Vulkan, "GE8322"),
            (GpuFamily::PowerVrRogue, DetectionSource::NameFallback)
        );
    }

    #[test]
    fn name_fallback_bare_volcanic_model_with_zero_vendor_id() {
        assert_eq!(
            classify_gpu_family(0, wgpu::Backend::Vulkan, "DXT48"),
            (GpuFamily::PowerVrVolcanic, DetectionSource::NameFallback)
        );
    }

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

    // === classify_img_subfamily ===

    #[test]
    fn img_subfamily_rogue_keyword() {
        assert_eq!(classify_img_subfamily("PowerVR Rogue GE8320"), GpuFamily::PowerVrRogue);
    }

    #[test]
    fn img_subfamily_rogue_bare_model() {
        assert_eq!(classify_img_subfamily("GE8322"), GpuFamily::PowerVrRogue);
        assert_eq!(classify_img_subfamily("GM9446"), GpuFamily::PowerVrRogue);
        assert_eq!(classify_img_subfamily("GX6250"), GpuFamily::PowerVrRogue);
    }

    #[test]
    fn img_subfamily_volcanic_model() {
        assert_eq!(classify_img_subfamily("DXT48"), GpuFamily::PowerVrVolcanic);
        assert_eq!(classify_img_subfamily("BXM-8-256"), GpuFamily::PowerVrVolcanic);
        assert_eq!(classify_img_subfamily("CXT48"), GpuFamily::PowerVrVolcanic);
        assert_eq!(classify_img_subfamily("AXT48"), GpuFamily::PowerVrVolcanic);
    }

    #[test]
    fn img_subfamily_generic_powervr_defaults_to_volcanic() {
        // No "Rogue" keyword, no Rogue prefix → modern default
        assert_eq!(classify_img_subfamily("PowerVR"), GpuFamily::PowerVrVolcanic);
        assert_eq!(classify_img_subfamily("IMG BXM-8-256"), GpuFamily::PowerVrVolcanic);
    }

    // === driver_quirks ===

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
            max_buffer_size: 0,
            max_storage_buffer_binding_size: 0,
            max_compute_workgroup_size_x: 0,
            max_compute_workgroup_size_y: 0,
            max_compute_workgroup_size_z: 0,
            max_compute_invocations_per_workgroup: 0,
            max_compute_workgroups_per_dimension: 0,
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
