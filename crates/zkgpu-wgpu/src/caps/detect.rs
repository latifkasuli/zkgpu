//! GPU family detection from vendor IDs and device name strings.
//!
//! Primary signal: PCI vendor ID (reliable, deterministic).
//! Fallback: device name heuristics (handles zero-vendor-ID drivers).

use super::types::{DetectionSource, GpuFamily};

// Well-known PCI vendor IDs.
pub(crate) const VENDOR_APPLE: u32 = 0x106B;
pub(crate) const VENDOR_QUALCOMM: u32 = 0x5143;
pub(crate) const VENDOR_ARM: u32 = 0x13B5;
pub(crate) const VENDOR_IMAGINATION: u32 = 0x1010;
pub(crate) const VENDOR_SAMSUNG: u32 = 0x144D;
pub(crate) const VENDOR_INTEL: u32 = 0x8086;
pub(crate) const VENDOR_AMD: u32 = 0x1002;
pub(crate) const VENDOR_NVIDIA: u32 = 0x10DE;

pub(crate) fn classify_gpu_family(
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
pub(crate) fn classify_img_subfamily(device_name: &str) -> GpuFamily {
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
}
