mod batched;
mod common;
mod four_step;
mod planner;
pub(crate) mod stockham;
// Renamed from `twiddles` as part of the Goldilocks Phase-A scaffolding —
// the existing precomputation module is BabyBear-only. A sibling
// `goldilocks_twiddles` module will land in Phase B with the portable
// u32x2 kernels. See `crates/zkgpu-wgpu/src/ntt/goldilocks/` for the
// Goldilocks-side planning / resolver surface.
pub(crate) mod babybear_twiddles;
pub mod goldilocks;

pub use batched::WgpuBatchedNttPlan;

use zkgpu_babybear::BabyBear;
use zkgpu_core::{NttDirection, NttPlan, ZkGpuError};

use crate::buffer::WgpuBuffer;
use crate::device::WgpuDevice;

pub use planner::PlannerPolicy;
use planner::{plan_ntt, PlannedNtt, StockhamTailOverride as PlannerTailOverride, MAX_BABYBEAR_LOG_N};

pub use stockham::NttTimings;
pub use stockham::R4ParamMode;

/// Public mirror of the planner's tail override, re-exported for callers
/// that construct `PlannerPolicy` directly (testkit, web, ffi runners).
///
/// Kept distinct from `zkgpu_report::StockhamTailOverride` so the runner
/// crates don't have to depend on `zkgpu-report` just to set this knob.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StockhamTailOverride {
    Auto,
    Local,
    Global,
}

impl Default for StockhamTailOverride {
    fn default() -> Self {
        Self::Auto
    }
}

impl StockhamTailOverride {
    fn into_planner(self) -> PlannerTailOverride {
        match self {
            Self::Auto => PlannerTailOverride::Auto,
            Self::Local => PlannerTailOverride::Local,
            Self::Global => PlannerTailOverride::Global,
        }
    }
}

impl PlannerPolicy {
    /// Apply a public `StockhamTailOverride` to this policy.
    ///
    /// Convenience wrapper so runners don't have to import the planner-internal
    /// override type. `Auto` resolves the env var `ZKGPU_STOCKHAM_TAIL`; an
    /// explicit `Local`/`Global` always wins.
    pub fn with_public_tail_override(self, ov: StockhamTailOverride) -> Self {
        self.with_stockham_tail_override_resolved(ov.into_planner())
    }
}

/// GPU NTT execution plan for BabyBear fields.
///
/// Family selection is driven by `GpuFamily` and `PlatformClass` from
/// the device's `CapabilityProfile`:
///
/// - `Browser`: Stockham only (transpose not validated in browser).
/// - `Apple` (any platform): Stockham only — benchmarked on M4 Pro,
///   four-step does not outperform at any tested size.
/// - Mobile families (`Adreno`, `Mali`, `PowerVrVolcanic`, `Xclipse`) on
///   `AndroidNative`: four-step for `log_n >= 18` — benchmarked on
///   Adreno 750, four-step wins from 2^18 through 2^21.
/// - Mobile families on non-Android platforms: four-step for `log_n >= 20`
///   (provisional — S24 Ultra data not assumed to generalize beyond Android).
/// - All other families (Intel, Nvidia, AMD, unknown):
///   four-step for `log_n >= 20` (provisional, pending discrete GPU benchmarks).
///
/// Thresholds are derived from `PlannerPolicy` and revisable with benchmark data.
pub struct WgpuNttPlan {
    inner: PlanImpl,
    direction: NttDirection,
    log_n: u32,
}

/// Item #3 (immediates): default R4 param mode.
///
/// **Currently `Storage` even when `caps.has_immediates`** — pilot
/// shape mirroring item #6's. The Immediate path is implemented and
/// validated for parity, but the default doesn't switch until per-host
/// bench data confirms a clean win across the consumer hot-path log_n
/// range (typically 18). M4 Pro / Metal data (commit ac379b4) shows
/// Immediate +2-5% slower at log_n ∈ {10, 14, 18} and -11.6% faster
/// at log_n=20; NVIDIA Vulkan A/B is pending vast.ai availability.
/// Once a uniform-win or thresholded-win story is confirmed, this
/// function flips to return Immediate (or a thresholded mix) and the
/// bench feeds back into the gate decision in the speed-opportunities
/// doc.
///
/// Callers wanting to opt in explicitly use
/// `WgpuNttPlan::new_with_r4_param_mode`.
fn default_r4_param_mode(_device: &WgpuDevice) -> stockham::R4ParamMode {
    stockham::R4ParamMode::Storage
}

enum PlanImpl {
    Stockham(stockham::StockhamPlan),
    FourStep(four_step::FourStepPlan),
}

impl WgpuNttPlan {
    /// Create a new NTT plan for BabyBear transforms of size `2^log_n`.
    ///
    /// `log_n` must be in the range `1..=27` (BabyBear 2-adicity bound).
    /// The planner automatically selects the best family based on device
    /// capabilities.
    pub fn new(
        device: &WgpuDevice,
        log_n: u32,
        direction: NttDirection,
    ) -> Result<Self, ZkGpuError> {
        // The convenience constructor honours the `ZKGPU_STOCKHAM_TAIL` env
        // var. `new_with_policy` does not — callers passing an explicit
        // policy are presumed to have already resolved their override.
        let policy = PlannerPolicy::from_caps(device.caps())
            .with_public_tail_override(StockhamTailOverride::Auto);
        Self::new_with_policy(device, log_n, direction, &policy)
    }

    /// Create a plan with an explicit R4 param-mode override.
    ///
    /// Item #3 (immediates) bench-only knob. Production callers go
    /// through [`Self::new`], which auto-detects from
    /// `device.caps.has_immediates`. This constructor exists so the
    /// `ntt_param_mode` bench can A/B `Storage` vs `Immediate` on the
    /// same hardware. Building `R4ParamMode::Immediate` on a device
    /// that doesn't advertise `Features::IMMEDIATES` returns
    /// `ZkGpuError::InvalidNttSize`.
    pub fn new_with_r4_param_mode(
        device: &WgpuDevice,
        log_n: u32,
        direction: NttDirection,
        r4_param_mode: stockham::R4ParamMode,
    ) -> Result<Self, ZkGpuError> {
        let policy = PlannerPolicy::from_caps(device.caps())
            .with_public_tail_override(StockhamTailOverride::Auto);
        Self::new_with_options(device, log_n, direction, &policy, r4_param_mode)
    }

    /// Create a plan with an explicit planner policy.
    ///
    /// Useful for benchmarks that need to force a specific family, or for
    /// integration tests that must exercise the four-step path on hardware
    /// where it would not normally be selected.
    pub fn new_with_policy(
        device: &WgpuDevice,
        log_n: u32,
        direction: NttDirection,
        policy: &PlannerPolicy,
    ) -> Result<Self, ZkGpuError> {
        Self::new_with_options(
            device,
            log_n,
            direction,
            policy,
            default_r4_param_mode(device),
        )
    }

    /// Internal constructor wired by all the public entry points.
    fn new_with_options(
        device: &WgpuDevice,
        log_n: u32,
        direction: NttDirection,
        policy: &PlannerPolicy,
        r4_param_mode: stockham::R4ParamMode,
    ) -> Result<Self, ZkGpuError> {
        if log_n > MAX_BABYBEAR_LOG_N {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "log_n={log_n} exceeds BabyBear 2-adicity (max {MAX_BABYBEAR_LOG_N})"
            )));
        }

        // Preflight: verify the device can hold N-element storage buffers.
        // Both Stockham and Four-Step allocate scratch buffers of size N×4
        // that are bound as full storage buffers via as_entire_binding().
        let n_bytes = (1u64 << log_n) * std::mem::size_of::<u32>() as u64;
        let storage_limit = device.caps.max_storage_buffer_size();
        if n_bytes > storage_limit {
            return Err(ZkGpuError::BufferSize {
                requested: n_bytes,
                limit: storage_limit,
            });
        }

        let planned = plan_ntt(log_n, policy)?;

        // Total-memory preflight: check aggregate N-sized buffer footprint.
        //
        // Stockham allocates 1 N-sized scratch buffer; the user provides
        // 1 N-sized data buffer → 2 × N×4 total.
        //
        // Four-Step allocates 2 N-sized scratch buffers + 2 N-sized diagonal
        // twiddle tables; plus the user's data buffer → 5 × N×4 total.
        //
        // We check against max_buffer_size as a proxy for practical device
        // memory capacity. wgpu does not expose total GPU memory, but
        // max_buffer_size reflects what the driver considers feasible for
        // a single allocation and is a reasonable aggregate ceiling on
        // mobile UMA devices.
        let n_buffer_count: u64 = match &planned {
            PlannedNtt::Stockham(_) => 2, // 1 scratch + 1 user data
            PlannedNtt::FourStep(_) => 5, // 2 scratch + 2 twiddle + 1 user data
        };
        let total_n_bytes = n_buffer_count * n_bytes;
        if total_n_bytes > device.caps.max_buffer_size {
            return Err(ZkGpuError::BufferSize {
                requested: total_n_bytes,
                limit: device.caps.max_buffer_size,
            });
        }

        let inner = match planned {
            PlannedNtt::Stockham(config) => {
                PlanImpl::Stockham(stockham::StockhamPlan::new(
                    device,
                    config,
                    direction,
                    r4_param_mode,
                )?)
            }
            PlannedNtt::FourStep(config) => {
                PlanImpl::FourStep(four_step::FourStepPlan::new(device, config, direction)?)
            }
        };

        device.save_pipeline_cache();

        Ok(Self {
            inner,
            direction,
            log_n,
        })
    }

    /// Total number of GPU dispatches (NTT stages + optional scaling).
    pub fn num_dispatches(&self) -> u32 {
        match &self.inner {
            PlanImpl::Stockham(p) => p.num_dispatches(),
            PlanImpl::FourStep(p) => p.num_dispatches(),
        }
    }

    /// Execute the full NTT in a single command submission.
    pub fn execute_kernels(
        &mut self,
        device: &WgpuDevice,
        buf: &mut WgpuBuffer<BabyBear>,
    ) -> Result<(), ZkGpuError> {
        match &mut self.inner {
            PlanImpl::Stockham(p) => p.execute_kernels(device, buf),
            PlanImpl::FourStep(p) => p.execute_kernels(device, buf),
        }
    }

    /// Execute the full NTT with GPU timestamp profiling.
    pub fn execute_kernels_profiled(
        &mut self,
        device: &WgpuDevice,
        buf: &mut WgpuBuffer<BabyBear>,
    ) -> Result<Option<NttTimings>, ZkGpuError> {
        match &mut self.inner {
            PlanImpl::Stockham(p) => p.execute_kernels_profiled(device, buf),
            PlanImpl::FourStep(p) => p.execute_kernels_profiled(device, buf),
        }
    }

    /// Async variant of [`execute_kernels`](Self::execute_kernels). Browser-safe.
    pub async fn execute_kernels_async(
        &mut self,
        device: &WgpuDevice,
        buf: &mut WgpuBuffer<BabyBear>,
    ) -> Result<(), ZkGpuError> {
        match &mut self.inner {
            PlanImpl::Stockham(p) => p.execute_kernels_async(device, buf).await,
            PlanImpl::FourStep(p) => p.execute_kernels_async(device, buf).await,
        }
    }

    /// Async variant of [`execute_kernels_profiled`](Self::execute_kernels_profiled).
    /// Browser-safe. Uses `web_time::Instant` for wall-clock timing.
    pub async fn execute_kernels_profiled_async(
        &mut self,
        device: &WgpuDevice,
        buf: &mut WgpuBuffer<BabyBear>,
    ) -> Result<Option<NttTimings>, ZkGpuError> {
        match &mut self.inner {
            PlanImpl::Stockham(p) => p.execute_kernels_profiled_async(device, buf).await,
            PlanImpl::FourStep(p) => p.execute_kernels_profiled_async(device, buf).await,
        }
    }

    /// Human-readable name for the selected kernel family.
    pub fn family_name(&self) -> &'static str {
        match &self.inner {
            PlanImpl::Stockham(_) => "stockham",
            PlanImpl::FourStep(_) => "four-step",
        }
    }

    /// Tail strategy chosen for this Stockham plan, e.g. `"LocalFusedR4"` or
    /// `"GlobalOnlyR4"`. Returns `None` for Four-Step plans and for Stockham
    /// plans below `LOG_BLOCK` where there is no tail phase.
    pub fn stockham_tail_strategy(&self) -> Option<&'static str> {
        match &self.inner {
            PlanImpl::Stockham(p) => p.tail_strategy_label(),
            PlanImpl::FourStep(_) => None,
        }
    }

    /// Reason that tail strategy was chosen — heuristic name or forced override.
    pub fn stockham_tail_reason(&self) -> Option<&'static str> {
        match &self.inner {
            PlanImpl::Stockham(p) => p.tail_reason_label(),
            PlanImpl::FourStep(_) => None,
        }
    }

    /// Per-thread gather stride in bytes for the local-fused tail.
    /// `None` for Four-Step, for `GlobalOnlyR4` tails, and for plans
    /// without a tail phase.
    pub fn tail_stride_bytes(&self) -> Option<u64> {
        match &self.inner {
            PlanImpl::Stockham(p) => p.tail_stride_bytes(),
            PlanImpl::FourStep(_) => None,
        }
    }
}

impl NttPlan<BabyBear, WgpuDevice> for WgpuNttPlan {
    fn execute(
        &mut self,
        device: &WgpuDevice,
        buf: &mut WgpuBuffer<BabyBear>,
    ) -> Result<(), ZkGpuError> {
        self.execute_kernels(device, buf)
    }

    fn log_n(&self) -> u32 {
        self.log_n
    }

    fn direction(&self) -> NttDirection {
        self.direction
    }
}
