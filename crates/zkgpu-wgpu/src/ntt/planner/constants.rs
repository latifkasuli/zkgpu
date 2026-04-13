pub(crate) const WORKGROUP_SIZE: u32 = 256;

/// Block size for the workgroup-local Stockham kernel.
/// Each R4 butterfly touches 4 elements, so BLOCK_SIZE = 4 * WORKGROUP_SIZE
/// ensures all threads are active during every R4 stage pair.
pub(crate) const BLOCK_SIZE: u32 = 4 * WORKGROUP_SIZE;
pub(crate) const LOG_BLOCK: u32 = 10; // log2(1024)

/// Tile dimension for four-step transpose.
pub(crate) const TRANSPOSE_TILE: u32 = 16;

/// Maximum `log_n` the planner accepts (u32 shift safety).
pub(super) const MAX_LOG_N: u32 = 31;

/// Maximum `log_n` for BabyBear transforms (limited by 2-adicity).
pub(crate) const MAX_BABYBEAR_LOG_N: u32 = 27;

/// Default four-step crossover for native tiers, revisable with benchmark data.
pub(crate) const DEFAULT_FOUR_STEP_THRESHOLD: u32 = 20;

/// Four-step crossover for mobile UMA (Adreno on Android).
///
/// Benchmarked on Samsung S24 Ultra (Adreno 750, Vulkan): four-step wins
/// from 2^18 through 2^21 (0.73x–0.96x GPU ratio), tied at 2^22.
/// The dominant Stockham cost centre on this class of device is the
/// shared-memory `local fused` kernel.
pub(crate) const MOBILE_UMA_FOUR_STEP_THRESHOLD: u32 = 18;
