//! `CapabilityProfile` — the single source of truth for what this GPU can do.
//!
//! Populated once at device init from `wgpu::Adapter` introspection.
//! Drives kernel selection, buffer strategy, profiling mode, and limit
//! validation throughout the runtime.

use super::classify::{classify_memory_model, classify_platform_class, classify_tier};
use super::detect::classify_gpu_family;
use super::types::*;

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

    /// Whether the adapter advertises `wgpu::Features::SHADER_INT64`.
    ///
    /// **Detection only in Phase A** — this bool is populated from
    /// adapter introspection but never causes the feature to be
    /// requested at device creation. Phase D (native Goldilocks
    /// kernels) owns the device-construction path that actually
    /// requests the feature, so no code path in Phase A's default
    /// build changes behaviour based on it.
    ///
    /// Per the March 2026 WGSL CRD and current WebGPU REC this flag is
    /// always `false` on browser / wasm targets — there is no
    /// `shader-int64` feature in the web spec.
    pub has_shader_int64: bool,

    /// Whether adding `TextureUsages::TRANSIENT` to render attachments
    /// reduces memory usage on this adapter.
    ///
    /// `true` on TBDR GPUs (Mali, Adreno, PowerVR, Apple Silicon) where
    /// the Vulkan backend supports `VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT`
    /// or the Metal backend supports `MTLStorageMode.memoryless`.
    ///
    /// Currently captured for future use — zkgpu has no texture/render-pass
    /// path yet. When one is added (e.g. texture-backed transpose), use
    /// [`render_attachment_policy`](Self::render_attachment_policy) to apply
    /// the hint correctly.
    pub transient_saves_memory: bool,

    pub max_buffer_size: u64,
    pub max_storage_buffer_binding_size: u64,
    pub max_compute_workgroup_size_x: u32,
    pub max_compute_workgroup_size_y: u32,
    pub max_compute_workgroup_size_z: u32,
    pub max_compute_invocations_per_workgroup: u32,
    pub max_compute_workgroups_per_dimension: u32,
    /// Size in bytes of shared-memory (`var<workgroup>`) budget per workgroup.
    ///
    /// On Vulkan this is `maxComputeSharedMemorySize`; on Metal the threadgroup
    /// memory limit; on DX12 the TGSM limit. Needed to reason about Stockham
    /// local-stage depth: the kernel uses `BLOCK_SIZE * 4` bytes of workgroup
    /// storage, so a device reporting ≥32 KiB can hold 8192-element blocks
    /// while a 16 KiB device caps at 4096. Drives future `LOG_BLOCK` tuning
    /// and the decision of whether to extend the local-fused stage.
    pub max_compute_workgroup_storage_size: u32,
}

/// Policy for creating a render attachment texture.
///
/// On TBDR GPUs that support transient memory, ephemeral attachments
/// (depth/stencil, MSAA resolve) can use `TextureUsages::TRANSIENT` +
/// `StoreOp::Discard` to avoid backing the texture with physical DRAM.
pub struct RenderAttachmentPolicy {
    /// Usage flags to pass to `TextureDescriptor::usage`.
    pub usage: wgpu::TextureUsages,
    /// Store operation for the render pass that writes to this attachment.
    pub store_op: wgpu::StoreOp,
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
        // SHADER_INT64 is native-only (Vulkan / DX12-DXC / Metal 2.3+)
        // per the wgpu docs and always absent on browser-WebGPU. We
        // only *detect* here; actual feature requesting is gated by
        // the `goldilocks-vulkan-int64` Cargo feature in
        // `required_features`.
        let has_shader_int64 = features.contains(wgpu::Features::SHADER_INT64);

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
            min_subgroup_size: info.subgroup_min_size,
            max_subgroup_size: info.subgroup_max_size,
            has_timestamp_query,
            has_timestamp_query_inside_passes: has_timestamp_inside,
            has_mappable_primary_buffers: has_mappable,
            has_pipeline_cache,
            has_shader_int64,
            transient_saves_memory: info.transient_saves_memory,
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
        self.max_compute_workgroup_storage_size = limits.max_compute_workgroup_storage_size;
    }

    /// Whether readback should map compute buffers directly (UMA fast path).
    ///
    /// Returns `true` only when **both** conditions hold:
    /// 1. The adapter exposes `MAPPABLE_PRIMARY_BUFFERS`.
    /// 2. The memory model is `Unified` (CPU and GPU share physical DRAM).
    ///
    /// `MAPPABLE_PRIMARY_BUFFERS` on a discrete GPU would silently move
    /// storage buffers into PCI-visible host memory, severely hurting
    /// compute throughput. By requiring unified memory we ensure the
    /// optimization only fires on hardware where it is always beneficial
    /// (Apple Silicon, Adreno, integrated Intel/AMD).
    pub fn use_direct_map_readback(&self) -> bool {
        self.has_mappable_primary_buffers && self.memory_model == MemoryModel::Unified
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
        // TIMESTAMP_QUERY_INSIDE_PASSES is intentionally NOT requested.
        // The profiler only uses begin/end-of-pass timestamps via
        // ComputePassTimestampWrites, which requires only TIMESTAMP_QUERY.
        // INSIDE_PASSES enables mid-pass write_timestamp() calls that
        // zkgpu does not use. Still detected and reported for diagnostics.
        // Only request MPB when we will actually use it (unified memory).
        // Requesting it on discrete GPUs would let wgpu place storage
        // buffers in host-visible VRAM, penalising compute throughput.
        if self.use_direct_map_readback() {
            f |= wgpu::Features::MAPPABLE_PRIMARY_BUFFERS;
        }
        if self.has_pipeline_cache {
            f |= wgpu::Features::PIPELINE_CACHE;
        }
        // SHADER_INT64 is **never** requested here, even when
        // `goldilocks-vulkan-int64` is enabled. Phase A is detection-
        // only; requesting the feature at device creation would change
        // behaviour for every `WgpuDevice::new()` caller in an
        // experimental build, including BabyBear workloads that have
        // nothing to do with Goldilocks. Phase D (native Goldilocks
        // kernels) is where this feature actually becomes required,
        // and it will own the device-construction path for plans that
        // elect the native variant — either a fresh device with
        // SHADER_INT64 requested, or an explicit constructor.
        f
    }

    /// The minimum wgpu limits this runtime actually needs, clamped to
    /// what the adapter can provide.
    ///
    /// **Mobile / integrated** (AndroidNative / DesktopIntegrated /
    /// Browser / AppleNative / Unknown): requests up to 256 MB for storage
    /// buffers (NTT up to 2^26 BabyBear elements), clamped to the adapter
    /// maximum. This matches mobile device caps (Adreno 128 MB, Mali
    /// typically 2 GB, Apple 2 GB).
    ///
    /// **Desktop discrete** (DesktopDiscrete): requests the adapter's full
    /// reported `max_storage_buffer_binding_size` / `max_buffer_size`. On
    /// RTX 4090 that's ~2 GB, unlocking log 24+ working buffers.
    ///
    /// NVIDIA scale-up Tier 1 (2026-04-16): this is the "Gap 2" from
    /// G.0.2 (`research/benchmarks/foundation-audit-2026-04-15/
    /// g02-desktop-webgpu/README.md` §"Gap 2 — wgpu adapter requested
    /// with default (256 MB) buffer limits"). Stockham at log 25 wants
    /// 512 MB and Four-Step at log 24 wants 335 MB — both panic with
    /// `BufferSize { requested: X, limit: 268435456 }` against the old
    /// 256 MB ceiling. Bumping the request to the adapter max unlocks
    /// log 24+ on desktop silicon without affecting any mobile path.
    ///
    /// Callers that need larger buffers (e.g. 2^20+ element NTTs) can
    /// check `CapabilityProfile` fields directly and fail gracefully.
    pub fn required_limits(&self, adapter_limits: &wgpu::Limits) -> wgpu::Limits {
        let defaults = wgpu::Limits::downlevel_defaults();
        const MOBILE_DESIRED_BUFFER: u64 = 256 * 1024 * 1024;
        let desired_buffer: u64 = match self.platform_class {
            // Desktop discrete (NVIDIA / AMD / Intel Arc): request the
            // adapter's full reported ceiling. These GPUs report 2 GB
            // (Vulkan's i32 cap) and have the VRAM to back it.
            PlatformClass::DesktopDiscrete => adapter_limits
                .max_storage_buffer_binding_size
                .max(MOBILE_DESIRED_BUFFER),
            // Mobile / integrated / browser / Apple / unknown: keep the
            // conservative 256 MB ceiling. Mobile devices can report
            // smaller maxima (Adreno ~128 MB); the adapter clamp below
            // still handles that correctly.
            PlatformClass::AndroidNative
            | PlatformClass::DesktopIntegrated
            | PlatformClass::Browser
            | PlatformClass::AppleNative
            | PlatformClass::UnknownNative => MOBILE_DESIRED_BUFFER,
        };
        wgpu::Limits {
            max_storage_buffer_binding_size: desired_buffer
                .min(adapter_limits.max_storage_buffer_binding_size),
            max_buffer_size: desired_buffer.min(adapter_limits.max_buffer_size),
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
            .min(self.max_storage_buffer_binding_size)
    }

    /// Determine the correct usage flags and store op for a render attachment.
    ///
    /// When `ephemeral` is `true` and the adapter supports transient memory,
    /// returns `RENDER_ATTACHMENT | TRANSIENT` with `StoreOp::Discard` —
    /// the texture may live entirely in tile memory on TBDR GPUs and never
    /// be backed by physical DRAM.
    ///
    /// When `ephemeral` is `false` or the adapter does not benefit from
    /// transient textures, returns plain `RENDER_ATTACHMENT` with
    /// `StoreOp::Store`.
    ///
    /// Only use for attachments that are truly single-pass throwaways and
    /// are never rebound as sampled/storage/copy resources.
    pub fn render_attachment_policy(&self, ephemeral: bool) -> RenderAttachmentPolicy {
        if ephemeral && self.transient_saves_memory {
            RenderAttachmentPolicy {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TRANSIENT,
                store_op: wgpu::StoreOp::Discard,
            }
        } else {
            RenderAttachmentPolicy {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                store_op: wgpu::StoreOp::Store,
            }
        }
    }

    /// Structured feature flag summary for device reports.
    ///
    /// Returns a list of `(name, value)` pairs suitable for serialization
    /// into JSON reports, CLI output, or web diagnostics. Centralizes the
    /// feature flag list so CLI/web/testkit don't duplicate it.
    pub fn feature_flags(&self) -> Vec<(&'static str, String)> {
        let mut flags = Vec::new();
        if self.has_subgroup {
            flags.push(("subgroup", format!("{}-{}", self.min_subgroup_size, self.max_subgroup_size)));
        }
        if self.has_timestamp_query {
            flags.push(("timestamp_query", "true".to_string()));
        }
        if self.has_timestamp_query_inside_passes {
            flags.push(("timestamp_query_inside_passes", "true".to_string()));
        }
        if self.has_mappable_primary_buffers {
            flags.push(("mappable_primary_buffers", "true".to_string()));
        }
        if self.has_pipeline_cache {
            flags.push(("pipeline_cache", "true".to_string()));
        }
        if self.transient_saves_memory {
            flags.push(("transient_memory", "true".to_string()));
        }
        flags
    }

    /// Flattened feature flag list for `DeviceReport::feature_flags`.
    ///
    /// Produces the canonical `Vec<String>` consumed by `zkgpu_report::DeviceReport`.
    /// Boolean flags emit their name; range flags (e.g. subgroup) emit
    /// `"name(min-max)"`. This is the **single source of truth** — testkit,
    /// web, and CLI builders should all call this instead of hand-assembling
    /// flag lists.
    pub fn feature_flags_flat(&self) -> Vec<String> {
        self.feature_flags()
            .into_iter()
            .map(|(name, val)| {
                if val == "true" {
                    name.to_string()
                } else {
                    format!("{name}({val})")
                }
            })
            .collect()
    }

    fn feature_summary(&self) -> String {
        let flat = self.feature_flags_flat();
        if flat.is_empty() {
            "none".to_string()
        } else {
            flat.join(", ")
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal `CapabilityProfile` with the two knobs that matter for
    /// readback policy. Everything else is zeroed / defaulted.
    fn mock_readback_caps(has_mpb: bool, memory: MemoryModel) -> CapabilityProfile {
        CapabilityProfile {
            tier: DeviceTier::NativeBasic,
            backend: wgpu::Backend::Vulkan,
            device_type: wgpu::DeviceType::IntegratedGpu,
            vendor_id: 0,
            device_id: 0,
            device_name: String::new(),
            driver: String::new(),
            driver_info: String::new(),
            gpu_family: GpuFamily::Unknown,
            detection_source: DetectionSource::Unknown,
            platform_class: PlatformClass::UnknownNative,
            memory_model: memory,
            has_subgroup: false,
            min_subgroup_size: 0,
            max_subgroup_size: 0,
            has_timestamp_query: false,
            has_timestamp_query_inside_passes: false,
            has_mappable_primary_buffers: has_mpb,
            has_pipeline_cache: false,
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

    // --- use_direct_map_readback policy ---

    #[test]
    fn direct_map_enabled_when_mpb_and_unified() {
        let caps = mock_readback_caps(true, MemoryModel::Unified);
        assert!(caps.use_direct_map_readback());
    }

    #[test]
    fn direct_map_disabled_when_mpb_but_discrete() {
        let caps = mock_readback_caps(true, MemoryModel::Discrete);
        assert!(!caps.use_direct_map_readback());
    }

    #[test]
    fn direct_map_disabled_when_mpb_but_unknown_memory() {
        let caps = mock_readback_caps(true, MemoryModel::Unknown);
        assert!(!caps.use_direct_map_readback());
    }

    #[test]
    fn direct_map_disabled_when_no_mpb_even_unified() {
        let caps = mock_readback_caps(false, MemoryModel::Unified);
        assert!(!caps.use_direct_map_readback());
    }

    // --- required_features respects the same policy ---

    #[test]
    fn required_features_includes_mpb_when_policy_allows() {
        let caps = mock_readback_caps(true, MemoryModel::Unified);
        assert!(caps.required_features().contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS));
    }

    #[test]
    fn required_features_excludes_mpb_on_discrete() {
        let caps = mock_readback_caps(true, MemoryModel::Discrete);
        assert!(!caps.required_features().contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS));
    }
}
