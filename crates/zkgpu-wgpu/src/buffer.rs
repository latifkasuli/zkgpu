use std::marker::PhantomData;
use std::sync::Arc;

use zkgpu_core::{GpuBuffer, GpuField, ZkGpuError};

use crate::async_util;

pub struct WgpuBuffer<F: GpuField> {
    pub(crate) inner: wgpu::Buffer,
    len: usize,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    _marker: PhantomData<F>,
}

impl<F: GpuField> WgpuBuffer<F> {
    pub(crate) fn new(
        inner: wgpu::Buffer,
        len: usize,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
    ) -> Self {
        Self {
            inner,
            len,
            device,
            queue,
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
    pub async fn read_to_vec_async(&self) -> Result<Vec<F>, ZkGpuError> {
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
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        async_util::map_buffer_read(&self.device, slice).await?;

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
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        async_util::map_buffer_read_blocking(&self.device, slice)?;

        let mapped = slice.get_mapped_range();
        let repr_slice: &[F::Repr] = bytemuck::cast_slice(&mapped);
        let result: Vec<F> = repr_slice.iter().map(|r| F::from_repr(*r)).collect();

        drop(mapped);
        staging.unmap();

        Ok(result)
    }
}
