use zkgpu_core::ZkGpuError;

use crate::async_util;
use crate::caps::CapabilityProfile;

/// GPU-side timestamp profiling via wgpu timestamp queries.
///
/// Only active when the device supports `TIMESTAMP_QUERY`. Collects
/// begin/end timestamps for compute passes and converts them to
/// nanosecond durations using `Queue::get_timestamp_period()`.
///
/// Created per profiling session (e.g., one NTT execution), not shared.
pub struct GpuProfiler {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    timestamp_period: f32,
    max_queries: u32,
    next_query: u32,
}

/// Indices into the profiler's query set for a begin/end pair.
pub struct TimestampSpan {
    pub begin_query: u32,
    pub end_query: u32,
}

/// A resolved GPU timing measurement.
#[derive(Debug, Clone)]
pub struct GpuTiming {
    pub label: String,
    pub duration_ns: f64,
}

impl GpuProfiler {
    /// Create a profiling session if the device supports timestamp queries.
    ///
    /// Returns `None` if `TIMESTAMP_QUERY` is not available.
    /// `max_queries` should be at least 2x the number of spans you intend
    /// to record (each span uses a begin + end query).
    pub fn new_if_supported(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        caps: &CapabilityProfile,
        max_queries: u32,
    ) -> Option<Self> {
        if !caps.has_timestamp_query {
            return None;
        }

        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("zkgpu profiler"),
            ty: wgpu::QueryType::Timestamp,
            count: max_queries,
        });

        let resolve_size = (max_queries as u64) * 8;
        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("zkgpu profiler resolve"),
            size: resolve_size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("zkgpu profiler readback"),
            size: resolve_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let timestamp_period = queue.get_timestamp_period();

        Some(Self {
            query_set,
            resolve_buffer,
            readback_buffer,
            timestamp_period,
            max_queries,
            next_query: 0,
        })
    }

    /// Reserve a pair of query indices for a begin/end timestamp span.
    ///
    /// Returns `None` if the query set is full.
    pub fn begin_span(&mut self) -> Option<TimestampSpan> {
        if self.next_query + 2 > self.max_queries {
            return None;
        }
        let begin_idx = self.next_query;
        self.next_query += 2;
        Some(TimestampSpan {
            begin_query: begin_idx,
            end_query: begin_idx + 1,
        })
    }

    /// Build the `TimestampWrites` descriptor for a compute pass.
    pub fn pass_timestamp_writes(
        &self,
        span: &TimestampSpan,
    ) -> wgpu::ComputePassTimestampWrites<'_> {
        wgpu::ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(span.begin_query),
            end_of_pass_write_index: Some(span.end_query),
        }
    }

    /// Append resolve + copy commands to the encoder. Call before submitting.
    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.resolve_query_set(&self.query_set, 0..self.next_query, &self.resolve_buffer, 0);
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.readback_buffer,
            0,
            (self.next_query as u64) * 8,
        );
    }

    /// After submission + poll(Wait), read back the resolved timestamps.
    pub fn read_timestamps(&self, device: &wgpu::Device) -> Result<Vec<u64>, ZkGpuError> {
        let slice = self.readback_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| ZkGpuError::BackendError(Box::new(e)))?
            .map_err(|e| ZkGpuError::BackendError(Box::new(e)))?;

        let mapped = slice.get_mapped_range();
        let timestamps: Vec<u64> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        self.readback_buffer.unmap();

        Ok(timestamps)
    }

    /// Convert a span's raw timestamps to a duration in nanoseconds.
    ///
    /// Returns `None` if the span indices are out of bounds for the
    /// provided timestamp slice (e.g. a span from a different session).
    pub fn span_duration_ns(&self, timestamps: &[u64], span: &TimestampSpan) -> Option<f64> {
        let begin = *timestamps.get(span.begin_query as usize)?;
        let end = *timestamps.get(span.end_query as usize)?;
        Some(end.wrapping_sub(begin) as f64 * self.timestamp_period as f64)
    }

    /// Asynchronous variant of [`read_timestamps`](Self::read_timestamps).
    ///
    /// Browser-safe: uses the event-loop-driven mapping callback instead of
    /// `device.poll(Wait)`.
    pub async fn read_timestamps_async(
        &self,
        device: &wgpu::Device,
    ) -> Result<Vec<u64>, ZkGpuError> {
        let slice = self.readback_buffer.slice(..);
        async_util::map_buffer_read(device, slice).await?;

        let mapped = slice.get_mapped_range();
        let timestamps: Vec<u64> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        self.readback_buffer.unmap();

        Ok(timestamps)
    }

    /// Number of queries recorded so far.
    pub fn query_count(&self) -> u32 {
        self.next_query
    }
}
