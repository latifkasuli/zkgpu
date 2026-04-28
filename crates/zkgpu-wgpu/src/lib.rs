pub mod caps;
pub mod profiling;

mod async_util;
mod buffer;
mod canary;
mod device;
mod dispatch;
mod field_codec;
mod ntt;
mod pipeline_cache;
mod pipeline_registry;
mod poseidon2;

pub use buffer::WgpuBuffer;
pub use caps::{CapabilityProfile, DetectionSource, DeviceTier, DriverQuirks, GpuFamily, MemoryModel, PlatformClass, driver_quirks, is_gpu_usable};
pub use device::WgpuDevice;
pub use ntt::goldilocks::{WgpuGoldilocksNttPlan, MAX_GOLDILOCKS_LOG_N};
pub use ntt::{NttTimings, PlannerPolicy, StockhamTailOverride, WgpuBatchedNttPlan, WgpuNttPlan};
pub use poseidon2::{
    commit_mixed_height_with_w16_leaf, commit_mixed_height_with_w24_leaf,
    open_batch_mixed_height, root_from_retained, MERKLE_DIGEST_LEN,
    MixedHeightMatrixInput, MixedHeightOpening, RetainedLayersHost,
    WgpuBabyBearPoseidon2Plan, WgpuBabyBearPoseidon2PlonkyW16Plan,
    WgpuBabyBearPoseidon2PlonkyW24Plan, WgpuGoldilocksPoseidon2Plan,
    WgpuPoseidon2InterleavePairsPlan, WgpuPoseidon2MerkleCommit,
    WgpuPoseidon2MerkleCompressPlan, WgpuPoseidon2MerkleLeafPlan,
    WgpuPoseidon2MerkleLeafW16R8Plan,
};
pub use profiling::{GpuProfiler, GpuTiming, TimestampSpan};
