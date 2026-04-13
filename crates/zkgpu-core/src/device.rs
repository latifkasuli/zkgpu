use crate::error::ZkGpuError;
use crate::field::GpuField;
use crate::thread_safety::MaybeSendSync;

/// Metadata about a GPU device.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub backend: String,
    pub max_buffer_size: u64,
    pub max_workgroup_size: u32,
    pub max_compute_invocations: u32,
}

/// An opaque handle to a GPU buffer holding field elements.
///
/// The buffer lives on the device and can be read back to the host.
/// Implementations must ensure the buffer contents are valid field elements
/// in `F::Repr` layout.
pub trait GpuBuffer<F: GpuField>: MaybeSendSync {
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read the buffer contents back to the host.
    fn read_to_vec(&self) -> Result<Vec<F>, ZkGpuError>;
}

/// A GPU device capable of allocating buffers and dispatching compute work.
///
/// This is the primary abstraction that backend crates implement.
/// It is intentionally minimal: each operation (NTT, MSM, etc.) defines
/// its own extension trait on top of `GpuDevice`.
pub trait GpuDevice: MaybeSendSync {
    type Buffer<F: GpuField>: GpuBuffer<F>;

    fn info(&self) -> &DeviceInfo;

    /// Upload host data to a new GPU buffer.
    fn upload<F: GpuField>(&self, data: &[F]) -> Result<Self::Buffer<F>, ZkGpuError>;

    /// Allocate a zero-initialized GPU buffer of `len` elements.
    fn alloc_zeros<F: GpuField>(&self, len: usize) -> Result<Self::Buffer<F>, ZkGpuError>;
}
