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

    /// Number of cached pipelines (for testing / diagnostics).
    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn pipeline_count(&self) -> usize {
        self.pipelines.lock().expect("lock").len()
    }
}
