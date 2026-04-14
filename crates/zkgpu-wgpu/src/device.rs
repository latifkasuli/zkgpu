use std::sync::Arc;

use wgpu::{self, util::DeviceExt};
use zkgpu_core::{DeviceInfo, GpuDevice, GpuField, ZkGpuError};

use crate::buffer::WgpuBuffer;
use crate::canary;
use crate::caps::scoring::score_adapter;
use crate::caps::CapabilityProfile;
use crate::pipeline_cache;
use crate::pipeline_registry::PipelineRegistry;

pub struct WgpuDevice {
    pub(crate) device: Arc<wgpu::Device>,
    pub(crate) queue: Arc<wgpu::Queue>,
    pub(crate) caps: CapabilityProfile,
    pub(crate) pipelines: PipelineRegistry,
    pub(crate) pipeline_cache: Option<wgpu::PipelineCache>,
    info: DeviceInfo,
}

impl WgpuDevice {
    /// Request the best available GPU device (blocking).
    ///
    /// Uses `Backends::PRIMARY` (Vulkan, Metal, DX12, BrowserWebGPU) and
    /// prefers high-performance adapters. Probes the adapter for a
    /// `CapabilityProfile` that drives kernel selection and feature requests.
    ///
    /// **Native only.** On WebAssembly, use [`new_async`](Self::new_async).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new() -> Result<Self, ZkGpuError> {
        pollster::block_on(Self::new_async())
    }

    pub fn caps(&self) -> &CapabilityProfile {
        &self.caps
    }

    /// Request the best available GPU device (async).
    ///
    /// Enumerates all adapters on `Backends::PRIMARY` (Vulkan, Metal,
    /// DX12, BrowserWebGPU), scores each for compute suitability, and
    /// picks the highest-scoring usable adapter. Falls back to wgpu's
    /// `request_adapter(HighPerformance)` if enumeration yields no
    /// usable candidates (e.g. on WebGPU where enumeration may return
    /// an empty list).
    ///
    /// On native, the blocking [`new`](Self::new) wrapper calls this
    /// internally.
    pub async fn new_async() -> Result<Self, ZkGpuError> {
        let backends = wgpu::Backends::PRIMARY;
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });

        let adapter = Self::select_best_adapter(&instance, backends).await?;
        let mut caps = CapabilityProfile::from_adapter(&adapter);

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("zkgpu"),
                    required_features: caps.required_features(),
                    required_limits: caps.required_limits(&adapter.limits()),
                    experimental_features: wgpu::ExperimentalFeatures::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                    trace: wgpu::Trace::Off,
                },
            )
            .await
            .map_err(|e| ZkGpuError::BackendError(Box::new(e)))?;

        caps.apply_device_limits(&device.limits());

        let pipeline_cache = pipeline_cache::load_cache(&device, &caps);

        device.on_uncaptured_error(Arc::new(|error| {
            log::error!("zkgpu uncaptured device error: {error}");
        }));

        // Canary dispatch: first checks is_gpu_usable() to block known-broken
        // drivers (PowerVR Rogue/Volcanic), then verifies the driver can
        // execute a trivial compute shader. Catches both known-bad drivers
        // (clean error) and unknown-bad drivers (crash at init, not mid-workload).
        canary::canary_dispatch(&device, &queue, pipeline_cache.as_ref(), &caps).await?;

        let info = DeviceInfo {
            name: caps.device_name.clone(),
            backend: format!("{:?}", caps.backend),
            max_buffer_size: caps.max_buffer_size,
            max_workgroup_size: caps.max_compute_workgroup_size_x,
            max_compute_invocations: caps.max_compute_invocations_per_workgroup,
        };

        log::info!("zkgpu: {caps}");

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            pipelines: PipelineRegistry::new(),
            pipeline_cache,
            caps,
            info,
        })
    }

    /// Enumerate all adapters, score each for compute suitability, and
    /// return the highest-scoring usable adapter.
    ///
    /// Known-broken adapters (e.g. PowerVR Rogue on Android Vulkan) are
    /// filtered out by the scoring function before they can be selected.
    ///
    /// Falls back to `request_adapter(HighPerformance)` **only** when
    /// enumeration is unavailable or returns an empty list (the browser
    /// WebGPU case). If enumeration returned adapters and every one was
    /// rejected, this returns `GpuComputeUnsupported` — the fallback
    /// must not silently reselect an adapter the scorer just rejected.
    async fn select_best_adapter(
        instance: &wgpu::Instance,
        backends: wgpu::Backends,
    ) -> Result<wgpu::Adapter, ZkGpuError> {
        let adapters = instance.enumerate_adapters(backends).await;

        if adapters.is_empty() {
            // Enumeration unavailable or empty (browser WebGPU).
            // Fall back to wgpu's opaque request_adapter heuristic.
            log::info!(
                "zkgpu: enumerate_adapters returned empty list, \
                 falling back to request_adapter"
            );
            return instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .map_err(|e| ZkGpuError::BackendError(Box::new(e)));
        }

        // Score each adapter and pick the best usable one.
        let adapter_count = adapters.len();
        let mut rejected_count = 0usize;
        let mut best: Option<(wgpu::Adapter, i64)> = None;

        for adapter in adapters {
            let caps = CapabilityProfile::from_adapter(&adapter);
            let limits = adapter.limits();
            if let Some(score) = score_adapter(&caps, &limits) {
                log::debug!(
                    "zkgpu adapter candidate: {} ({:?}/{:?}) score={score}",
                    caps.device_name,
                    caps.backend,
                    caps.device_type,
                );
                if best.as_ref().map_or(true, |(_, s)| score > *s) {
                    best = Some((adapter, score));
                }
            } else {
                rejected_count += 1;
                log::debug!(
                    "zkgpu adapter rejected (known-broken): {} ({:?})",
                    caps.device_name,
                    caps.gpu_family,
                );
            }
        }

        if let Some((adapter, score)) = best {
            let info = adapter.get_info();
            log::info!(
                "zkgpu selected adapter: {} ({:?}/{:?}) score={score}",
                info.name,
                info.backend,
                info.device_type,
            );
            return Ok(adapter);
        }

        // Enumeration found adapters but all were rejected.
        Err(ZkGpuError::GpuComputeUnsupported(format!(
            "all {adapter_count} enumerated GPU adapter(s) were rejected \
             ({rejected_count} known-broken); no usable GPU available"
        )))
    }
}

impl WgpuDevice {
    /// Persist the Vulkan pipeline cache to disk.
    ///
    /// Called after each NTT plan is created so compiled shader microcode
    /// for all pipeline families (Stockham, Four-Step) reaches disk.
    /// The wgpu PipelineCache object accumulates data from every pipeline
    /// compiled against it, so later calls capture a superset of earlier
    /// ones. No-op on non-Vulkan backends.
    pub(crate) fn save_pipeline_cache(&self) {
        if let Some(ref cache) = self.pipeline_cache {
            pipeline_cache::save_cache(cache, &self.caps);
        }
    }

    /// Begin capturing GPU validation errors and return a scope guard.
    ///
    /// The guard must be consumed via `pop_validation_scope` or
    /// `pop_validation_scope_async` to collect the result.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub(crate) fn push_validation_scope(&self) -> wgpu::ErrorScopeGuard {
        self.device
            .push_error_scope(wgpu::ErrorFilter::Validation)
    }

    /// Pop the validation error scope (blocking).
    ///
    /// Returns `Ok(())` if no validation error was captured, or
    /// `Err(ZkGpuError::GpuValidation(...))` with the backend message.
    ///
    /// **Native only.** On WebAssembly, use
    /// [`pop_validation_scope_async`](Self::pop_validation_scope_async).
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn pop_validation_scope(
        &self,
        scope: wgpu::ErrorScopeGuard,
        context: &str,
    ) -> Result<(), ZkGpuError> {
        match pollster::block_on(scope.pop()) {
            Some(err) => Err(ZkGpuError::GpuValidation(format!("{context}: {err}"))),
            None => Ok(()),
        }
    }

    /// Pop the validation error scope (async, browser-safe).
    #[allow(dead_code)] // kept for cross-target parity; wasm uses async, native uses blocking
    pub(crate) async fn pop_validation_scope_async(
        &self,
        scope: wgpu::ErrorScopeGuard,
        context: &str,
    ) -> Result<(), ZkGpuError> {
        match scope.pop().await {
            Some(err) => Err(ZkGpuError::GpuValidation(format!("{context}: {err}"))),
            None => Ok(()),
        }
    }
}

impl GpuDevice for WgpuDevice {
    type Buffer<F: GpuField> = WgpuBuffer<F>;

    fn info(&self) -> &DeviceInfo {
        &self.info
    }

    fn upload<F: GpuField>(&self, data: &[F]) -> Result<WgpuBuffer<F>, ZkGpuError> {
        let byte_size = (data.len() * std::mem::size_of::<F::Repr>()) as u64;
        let limit = self.caps.max_storage_buffer_size();
        if byte_size > limit {
            return Err(ZkGpuError::BufferSize {
                requested: byte_size,
                limit,
            });
        }

        let mappable = self.caps.use_direct_map_readback();
        let mut usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        if mappable {
            usage |= wgpu::BufferUsages::MAP_READ;
        }

        let bytes = bytemuck::cast_slice(data);
        let buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("zkgpu upload"),
                contents: bytes,
                usage,
            });
        Ok(WgpuBuffer::new(
            buf,
            data.len(),
            self.device.clone(),
            self.queue.clone(),
            mappable,
        ))
    }

    fn alloc_zeros<F: GpuField>(&self, len: usize) -> Result<WgpuBuffer<F>, ZkGpuError> {
        let size = (len * std::mem::size_of::<F::Repr>()) as u64;
        let limit = self.caps.max_storage_buffer_size();
        if size > limit {
            return Err(ZkGpuError::BufferSize {
                requested: size,
                limit,
            });
        }

        let mappable = self.caps.use_direct_map_readback();
        let mut usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        if mappable {
            usage |= wgpu::BufferUsages::MAP_READ;
        }

        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("zkgpu zeros"),
            size,
            usage,
            mapped_at_creation: false,
        });
        Ok(WgpuBuffer::new(
            buf,
            len,
            self.device.clone(),
            self.queue.clone(),
            mappable,
        ))
    }
}
