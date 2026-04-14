use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Device-scoped registry for reusing compiled compute pipelines.
///
/// All WGSL shader sources are `include_str!`'d into the binary, so
/// pointer equality on the source `&'static str` is a valid identity key.
/// Forward + inverse NTT plans share the same shaders and pipeline layouts
/// (they differ only in twiddle values and scale params), so reusing
/// pipelines across plans eliminates redundant shader compilation.
///
/// On desktop drivers this saves ~50ms per duplicate pipeline. On mobile
/// Vulkan without a driver cache, it saves ~200ms per pipeline.
pub(crate) struct PipelineRegistry {
    modules: Mutex<HashMap<ShaderKey, Arc<wgpu::ShaderModule>>>,
    pipelines: Mutex<HashMap<PipelineKey, Arc<wgpu::ComputePipeline>>>,
    bgls: Mutex<HashMap<BglKey, Arc<wgpu::BindGroupLayout>>>,
}

/// Key for shader modules: pointer identity on the source string.
#[derive(Hash, Eq, PartialEq, Clone)]
struct ShaderKey {
    /// Pointer to the `include_str!`'d source. Same source → same pointer.
    source_ptr: usize,
}

/// Key for compute pipelines: shader + entry point + layout fingerprint.
#[derive(Hash, Eq, PartialEq, Clone)]
struct PipelineKey {
    source_ptr: usize,
    entry_point: &'static str,
    layout_key: BglKey,
}

/// Key for bind group layouts, identified by label.
#[derive(Hash, Eq, PartialEq, Clone)]
struct BglKey {
    label: &'static str,
}

impl PipelineRegistry {
    pub(crate) fn new() -> Self {
        Self {
            modules: Mutex::new(HashMap::new()),
            pipelines: Mutex::new(HashMap::new()),
            bgls: Mutex::new(HashMap::new()),
        }
    }

    /// Get or create a shader module from an `include_str!` source.
    pub(crate) fn get_or_create_module(
        &self,
        device: &wgpu::Device,
        source: &'static str,
        label: &'static str,
    ) -> Arc<wgpu::ShaderModule> {
        let key = ShaderKey {
            source_ptr: source.as_ptr() as usize,
        };
        let mut modules = self.modules.lock().expect("pipeline registry lock");
        modules
            .entry(key)
            .or_insert_with(|| {
                Arc::new(device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(label),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                }))
            })
            .clone()
    }

    /// Get or create a bind group layout.
    pub(crate) fn get_or_create_bgl(
        &self,
        device: &wgpu::Device,
        label: &'static str,
        entries: &[wgpu::BindGroupLayoutEntry],
    ) -> Arc<wgpu::BindGroupLayout> {
        let key = BglKey { label };
        let mut bgls = self.bgls.lock().expect("pipeline registry lock");
        bgls.entry(key)
            .or_insert_with(|| {
                Arc::new(
                    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some(label),
                        entries,
                    }),
                )
            })
            .clone()
    }

    /// Get or create a compute pipeline.
    ///
    /// When a `cache` is provided (Vulkan only), the driver can skip shader
    /// compilation on subsequent runs by reusing serialised microcode.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn get_or_create_pipeline(
        &self,
        device: &wgpu::Device,
        source: &'static str,
        entry_point: &'static str,
        bgl_label: &'static str,
        layout: &wgpu::PipelineLayout,
        module: &wgpu::ShaderModule,
        cache: Option<&wgpu::PipelineCache>,
    ) -> Arc<wgpu::ComputePipeline> {
        let key = PipelineKey {
            source_ptr: source.as_ptr() as usize,
            entry_point,
            layout_key: BglKey { label: bgl_label },
        };
        let mut pipelines = self.pipelines.lock().expect("pipeline registry lock");
        pipelines
            .entry(key)
            .or_insert_with(|| {
                Arc::new(
                    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(entry_point),
                        layout: Some(layout),
                        module,
                        entry_point: Some(entry_point),
                        compilation_options: Default::default(),
                        cache,
                    }),
                )
            })
            .clone()
    }

    /// Get or create a shader module from pre-compiled SPIR-V bytes.
    ///
    /// The SPIR-V binary is embedded via `include_bytes!` at build time.
    /// Keyed by the byte slice pointer, analogous to `get_or_create_module`
    /// keying on `include_str!` source pointers.
    ///
    /// Only available when the `subgroup-vulkan-spirv` feature is enabled
    /// (which activates `wgpu/spirv` for `ShaderSource::SpirV`).
    #[cfg(feature = "subgroup-vulkan-spirv")]
    pub(crate) fn get_or_create_module_spirv(
        &self,
        device: &wgpu::Device,
        spirv_bytes: &'static [u8],
        label: &'static str,
    ) -> Arc<wgpu::ShaderModule> {
        let key = ShaderKey {
            source_ptr: spirv_bytes.as_ptr() as usize,
        };
        let mut modules = self.modules.lock().expect("pipeline registry lock");
        modules
            .entry(key)
            .or_insert_with(|| {
                let spirv_words = spirv_bytes_to_words(spirv_bytes);
                Arc::new(device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(label),
                    source: wgpu::ShaderSource::SpirV(spirv_words.into()),
                }))
            })
            .clone()
    }

    /// Get or create a compute pipeline, keyed by an arbitrary source
    /// identifier rather than a WGSL source string pointer.
    ///
    /// Used by the SPIR-V path where the key is the `include_bytes!`
    /// pointer. WGSL callers should continue using `get_or_create_pipeline`.
    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)] // used only with subgroup-vulkan-spirv feature
    pub(crate) fn get_or_create_pipeline_keyed(
        &self,
        device: &wgpu::Device,
        source_key: usize,
        entry_point: &'static str,
        bgl_label: &'static str,
        layout: &wgpu::PipelineLayout,
        module: &wgpu::ShaderModule,
        cache: Option<&wgpu::PipelineCache>,
    ) -> Arc<wgpu::ComputePipeline> {
        let key = PipelineKey {
            source_ptr: source_key,
            entry_point,
            layout_key: BglKey { label: bgl_label },
        };
        let mut pipelines = self.pipelines.lock().expect("pipeline registry lock");
        pipelines
            .entry(key)
            .or_insert_with(|| {
                Arc::new(
                    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(entry_point),
                        layout: Some(layout),
                        module,
                        entry_point: Some(entry_point),
                        compilation_options: Default::default(),
                        cache,
                    }),
                )
            })
            .clone()
    }

    /// Number of cached pipelines (for testing / diagnostics).
    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn pipeline_count(&self) -> usize {
        self.pipelines.lock().expect("lock").len()
    }
}

/// Convert a SPIR-V byte slice to a word slice.
///
/// SPIR-V modules are always 4-byte aligned and `include_bytes!` may
/// not guarantee alignment, so we parse words explicitly.
#[cfg(feature = "subgroup-vulkan-spirv")]
pub(crate) fn spirv_bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    assert!(
        bytes.len() % 4 == 0,
        "SPIR-V binary size must be a multiple of 4 bytes"
    );
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
