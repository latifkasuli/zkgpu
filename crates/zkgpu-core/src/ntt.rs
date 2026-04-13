use crate::device::GpuDevice;
use crate::error::ZkGpuError;
use crate::field::GpuField;
use crate::thread_safety::MaybeSendSync;

/// Direction of the NTT transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NttDirection {
    Forward,
    Inverse,
}

/// A prepared NTT execution plan.
///
/// Plans are created once for a given size and direction, then reused
/// across multiple invocations. The plan pre-uploads twiddle factors
/// and computes the dispatch layout.
///
/// Execution takes `&mut self` because plans own internal scratch
/// resources (GPU buffers) that are aliased across stages. Concurrent
/// submissions on the same plan would corrupt intermediate data.
/// Create separate plans for concurrent use.
pub trait NttPlan<F: GpuField, D: GpuDevice>: MaybeSendSync {
    /// Execute the NTT in-place on the given buffer.
    fn execute(&mut self, device: &D, buf: &mut D::Buffer<F>) -> Result<(), ZkGpuError>;

    /// The log2 of the transform size.
    fn log_n(&self) -> u32;

    /// The transform direction.
    fn direction(&self) -> NttDirection;
}
