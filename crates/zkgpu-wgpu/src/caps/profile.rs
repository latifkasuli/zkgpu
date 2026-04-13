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
            transient_saves_memory: info.transient_saves_memory,
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
        let desired_buffer: u64 = 256 * 1024 * 1024;
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
