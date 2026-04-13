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
/// Populated by [`driver_quirks`](super::driver_quirks) from the GPU
/// family, backend, platform, and driver strings. The canary checks
/// these *before* attempting any `vkQueueSubmit` to avoid process-fatal
/// SIGSEGV on broken drivers.
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

/// Memory architecture model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryModel {
    Unified,
    Discrete,
    Unknown,
}
