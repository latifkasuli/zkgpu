use std::marker::PhantomData;
use std::sync::Arc;

use zkgpu_core::{GpuBuffer, GpuField, ZkGpuError};

use crate::async_util;

pub struct WgpuBuffer<F: GpuField> {
    pub(crate) inner: wgpu::Buffer,
    len: usize,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    /// `true` when the primary buffer was created with `MAP_READ` (UMA
    /// fast path). Readback can map `inner` directly, skipping the
    /// staging-copy round-trip.
    mappable: bool,
    _marker: PhantomData<F>,
}

impl<F: GpuField> WgpuBuffer<F> {
    pub(crate) fn new(
        inner: wgpu::Buffer,
        len: usize,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        mappable: bool,
    ) -> Self {
        Self {
            inner,
            len,
            device,
            queue,
            mappable,
            _marker: PhantomData,
        }
    }

    pub(crate) fn byte_size(&self) -> u64 {
        (self.len * std::mem::size_of::<F::Repr>()) as u64
    }

    /// Asynchronously read the buffer contents back to the host.
    ///
    /// This is the browser-safe equivalent of [`GpuBuffer::read_to_vec`].
    /// On native, it is functionally identical. On WebGPU (browser),
    /// it properly awaits the mapping callback via the event loop
    /// instead of blocking with `device.poll(Wait)`.
    ///
    /// When the buffer lives on a UMA device with `MAPPABLE_PRIMARY_BUFFERS`,
    /// the primary buffer is mapped directly — no staging copy needed.
    pub async fn read_to_vec_async(&self) -> Result<Vec<F>, ZkGpuError> {
        if self.mappable {
            return self.read_direct_async().await;
        }
        self.read_staged_async().await
    }

    /// Direct-map fast path (UMA). Maps the primary compute buffer for
    /// reading without allocating a staging buffer or issuing a GPU copy.
    async fn read_direct_async(&self) -> Result<Vec<F>, ZkGpuError> {
        let slice = self.inner.slice(..);
        async_util::map_buffer_read(&self.device, slice).await?;

        let mapped = slice.get_mapped_range();
        let repr_slice: &[F::Repr] = bytemuck::cast_slice(&mapped);
        let result: Vec<F> = repr_slice.iter().map(|r| F::from_repr(*r)).collect();

        drop(mapped);
        self.inner.unmap();

        Ok(result)
    }

    /// Staging-copy readback (discrete GPU or any device without MPB).
    ///
    /// Uses `map_buffer_on_submit` (wgpu v29) to schedule the staging
    /// buffer mapping as a deferred action at encode time. The mapping
    /// fires automatically when `queue.submit()` processes the command
    /// buffer, eliminating the separate `map_async` call.
    async fn read_staged_async(&self) -> Result<Vec<F>, ZkGpuError> {
        let size = self.byte_size();

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("zkgpu readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("zkgpu readback encoder"),
            });
        encoder.copy_buffer_to_buffer(&self.inner, 0, &staging, 0, size);

        // Schedule deferred mapping — fires when submit completes.
        let (sender, future) = async_util::CallbackFuture::new();
        encoder.map_buffer_on_submit(&staging, wgpu::MapMode::Read, .., move |result| {
            sender.send(result);
        });

        self.queue.submit(Some(encoder.finish()));

        #[cfg(not(target_arch = "wasm32"))]
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| ZkGpuError::BackendError(Box::new(e)))?;

        future
            .await
            .map_err(|e| ZkGpuError::BackendError(Box::new(e)))?;

        let slice = staging.slice(..);
        let mapped = slice.get_mapped_range();
        let repr_slice: &[F::Repr] = bytemuck::cast_slice(&mapped);
        let result: Vec<F> = repr_slice.iter().map(|r| F::from_repr(*r)).collect();

        drop(mapped);
        staging.unmap();

        Ok(result)
    }
}

impl<F: GpuField> GpuBuffer<F> for WgpuBuffer<F> {
    fn len(&self) -> usize {
        self.len
    }

    fn read_to_vec(&self) -> Result<Vec<F>, ZkGpuError> {
        if self.mappable {
            return self.read_direct_blocking();
        }
        self.read_staged_blocking()
    }
}

impl<F: GpuField> WgpuBuffer<F> {
    /// Direct-map fast path (UMA, blocking).
    fn read_direct_blocking(&self) -> Result<Vec<F>, ZkGpuError> {
        let slice = self.inner.slice(..);
        async_util::map_buffer_read_blocking(&self.device, slice)?;

        let mapped = slice.get_mapped_range();
        let repr_slice: &[F::Repr] = bytemuck::cast_slice(&mapped);
        let result: Vec<F> = repr_slice.iter().map(|r| F::from_repr(*r)).collect();

        drop(mapped);
        self.inner.unmap();

        Ok(result)
    }

    /// Staging-copy readback (blocking).
    ///
    /// Uses `map_buffer_on_submit` (wgpu v29) to schedule the staging
    /// buffer mapping as a deferred action at encode time.
    fn read_staged_blocking(&self) -> Result<Vec<F>, ZkGpuError> {
        let size = self.byte_size();

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("zkgpu readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("zkgpu readback encoder"),
            });
        encoder.copy_buffer_to_buffer(&self.inner, 0, &staging, 0, size);

        // Schedule deferred mapping — fires when submit completes.
        let (tx, rx) = std::sync::mpsc::channel();
        encoder.map_buffer_on_submit(&staging, wgpu::MapMode::Read, .., move |result| {
            let _ = tx.send(result);
        });

        self.queue.submit(Some(encoder.finish()));
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| ZkGpuError::BackendError(Box::new(e)))?;
        rx.recv()
            .map_err(|e| ZkGpuError::BackendError(Box::new(e)))?
            .map_err(|e| ZkGpuError::BackendError(Box::new(e)))?;

        let slice = staging.slice(..);
        let mapped = slice.get_mapped_range();
        let repr_slice: &[F::Repr] = bytemuck::cast_slice(&mapped);
        let result: Vec<F> = repr_slice.iter().map(|r| F::from_repr(*r)).collect();

        drop(mapped);
        staging.unmap();

        Ok(result)
    }
}
