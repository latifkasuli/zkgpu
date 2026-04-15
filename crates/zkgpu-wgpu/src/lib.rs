pub mod caps;
pub mod profiling;

mod async_util;
mod buffer;
mod canary;
mod device;
mod dispatch;
mod ntt;
mod pipeline_cache;
mod pipeline_registry;

pub use buffer::WgpuBuffer;
pub use caps::{CapabilityProfile, DetectionSource, DeviceTier, DriverQuirks, GpuFamily, MemoryModel, PlatformClass, driver_quirks, is_gpu_usable};
pub use device::WgpuDevice;
pub use ntt::{NttTimings, PlannerPolicy, StockhamTailOverride, WgpuNttPlan};
pub use profiling::{GpuProfiler, GpuTiming, TimestampSpan};
