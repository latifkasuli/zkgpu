use zkgpu_core::ZkGpuError;

use super::constants::{BLOCK_SIZE, LOG_BLOCK, MAX_LOG_N, WORKGROUP_SIZE};
use super::tail_policy::{StockhamTailReason, StockhamTailStrategy, TailDecision};

// ---------------------------------------------------------------------------
// Stockham planner — the original hybrid R2 family
// ---------------------------------------------------------------------------

/// Uniform parameters for a single global Stockham DIF stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GlobalStageParams {
    pub n: u32,
    pub s: u32,
    pub m: u32,
    pub twiddle_offset: u32,
}

/// Parameters for a radix-4 global Stockham DIF dispatch (combines 2 stages).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct R4StageParams {
    pub n: u32,
    pub s: u32,
    pub m4: u32,
    pub twiddle_offset: u32,
}

/// Records *which* tail strategy a Stockham plan committed to and *why*.
///
/// `None` on `StockhamPlanConfig.tail` means there is no tail phase — the
/// transform fits entirely in `num_global_stages` global dispatches. This is
/// the case both for tiny transforms (`log_n < LOG_BLOCK`) and for four-step
/// batched leaves that opted out of the tail entirely via `new_global_only`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StockhamTail {
    pub strategy: StockhamTailStrategy,
    pub reason: StockhamTailReason,
}

/// Structural decisions for a Stockham hybrid NTT execution plan.
///
/// All fields are derived from `log_n`, the requested tail decision, and the
/// compile-time constants `WORKGROUP_SIZE`, `BLOCK_SIZE`, and `LOG_BLOCK`.
/// No GPU interaction, no field-specific arithmetic — fully testable on any host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StockhamPlanConfig {
    pub log_n: u32,
    pub n: u32,
    /// The tail-phase decision. `None` for `log_n < LOG_BLOCK` and for the
    /// `new_global_only` path (four-step batched leaves).
    pub tail: Option<StockhamTail>,
    pub num_global_stages: u32,
    pub global_workgroups: u32,
    pub local_workgroups: u32,
    pub local_stride: u32,
    pub result_in_scratch: bool,
    pub global_stage_params: Vec<GlobalStageParams>,
    /// Radix-4 stage params (each combines 2 logical stages into 1 dispatch).
    pub r4_stage_params: Vec<R4StageParams>,
    /// (h, m4) pairs for twiddle generation, one per R4 dispatch.
    pub r4_twiddle_spec: Vec<(u32, u32)>,
    /// Number of global dispatches: r4_count + r2_count.
    pub num_global_dispatches: u32,
}

impl StockhamPlanConfig {
    /// Plan a Stockham NTT for a transform of size `2^log_n`.
    ///
    /// `tail` controls the final `LOG_BLOCK` stages:
    /// - `None`: no tail phase (tiny transforms or four-step leaves).
    /// - `Some(LocalFusedR4)`: emit `log_n - LOG_BLOCK` global stages plus
    ///   one workgroup-local dispatch.
    /// - `Some(GlobalOnlyR4)`: emit all `log_n` stages as global dispatches,
    ///   no local dispatch. Same kernel shape as `new_global_only`, but the
    ///   `tail` field records the decision so reporting can distinguish
    ///   "globally planned because too small for a tail" from "globally
    ///   planned because the tail heuristic said so".
    ///
    /// Returns `Err` if `log_n` is 0 or exceeds `MAX_LOG_N`, or if `tail` is
    /// requested for a `log_n` below `LOG_BLOCK`.
    pub fn new(log_n: u32, tail: Option<TailDecision>) -> Result<Self, ZkGpuError> {
        if log_n == 0 || log_n > MAX_LOG_N {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "log_n={log_n} out of range (must be 1..={MAX_LOG_N})"
            )));
        }
        if tail.is_some() && log_n < LOG_BLOCK {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "log_n={log_n} below LOG_BLOCK={LOG_BLOCK}; cannot have a tail phase"
            )));
        }

        let n = 1u32 << log_n;

        // Decide how many global stages run, based on the tail decision.
        let (use_local_dispatch, num_global_stages) = match tail {
            Some(TailDecision { strategy: StockhamTailStrategy::LocalFusedR4, .. }) => {
                (true, log_n - LOG_BLOCK)
            }
            Some(TailDecision { strategy: StockhamTailStrategy::GlobalOnlyR4, .. }) => {
                (false, log_n)
            }
            None => (false, log_n),
        };

        let num_butterflies = n / 2;
        let global_workgroups = num_butterflies.div_ceil(WORKGROUP_SIZE);

        let local_workgroups = if use_local_dispatch { n / BLOCK_SIZE } else { 0 };
        let local_stride = if use_local_dispatch { n / BLOCK_SIZE } else { 1 };

        // Build R4 pairs and R2 remainder for global stages.
        let num_r4 = num_global_stages / 2;
        let has_r2_remainder = num_global_stages % 2 == 1;

        let mut r4_stage_params = Vec::with_capacity(num_r4 as usize);
        let mut r4_twiddle_spec = Vec::with_capacity(num_r4 as usize);
        let mut r4_twiddle_offset = 0u32;

        for i in 0..num_r4 {
            let h = i * 2;
            let s = 1u32 << h;
            let m4 = n / (4 * s);
            r4_stage_params.push(R4StageParams {
                n,
                s,
                m4,
                twiddle_offset: r4_twiddle_offset,
            });
            r4_twiddle_spec.push((h, m4));
            r4_twiddle_offset += 3 * m4;
        }

        // R2 remainder stage (if odd number of global stages)
        let mut global_stage_params = Vec::new();
        if has_r2_remainder {
            let h = num_r4 * 2;
            let s = 1u32 << h;
            let m = n >> (h + 1);
            global_stage_params.push(GlobalStageParams {
                n,
                s,
                m,
                twiddle_offset: 0,
            });
        }

        let num_global_dispatches = num_r4 + u32::from(has_r2_remainder);
        let total_swaps = num_global_dispatches + u32::from(use_local_dispatch);
        let result_in_scratch = total_swaps % 2 == 1;

        let tail_record = tail.map(|d| StockhamTail {
            strategy: d.strategy,
            reason: d.reason,
        });

        Ok(Self {
            log_n,
            n,
            tail: tail_record,
            num_global_stages,
            global_workgroups,
            local_workgroups,
            local_stride,
            result_in_scratch,
            global_stage_params,
            r4_stage_params,
            r4_twiddle_spec,
            num_global_dispatches,
        })
    }

    /// Plan a Stockham NTT that uses only global DIF stages (no tail phase).
    ///
    /// Used by four-step batched leaves where the local kernel's strided
    /// gather doesn't match the contiguous batch layout. Equivalent to
    /// `Self::new(log_n, None)` but kept as a named entry point because
    /// four-step's call sites read more clearly.
    pub fn new_global_only(log_n: u32) -> Result<Self, ZkGpuError> {
        Self::new(log_n, None)
    }

    /// Whether this plan emits a workgroup-local fused dispatch.
    ///
    /// True iff `tail.strategy == LocalFusedR4`. False for tiny transforms,
    /// for four-step leaves, and for the new `GlobalOnlyR4` tail strategy.
    pub fn use_local_kernel(&self) -> bool {
        matches!(
            self.tail.as_ref().map(|t| t.strategy),
            Some(StockhamTailStrategy::LocalFusedR4)
        )
    }

    /// Number of NTT stage dispatches (R4 + R2 global dispatches + optional local).
    pub fn ntt_dispatches(&self) -> u32 {
        self.num_global_dispatches + u32::from(self.use_local_kernel())
    }

    /// Per-thread gather stride in bytes for the local-fused tail.
    ///
    /// `Some(N / BLOCK_SIZE * 4)` when the tail uses `LocalFusedR4`;
    /// `None` for `GlobalOnlyR4` and for plans without a tail phase.
    /// Reported in `CaseReport.tail_stride_bytes` so coalescing pressure
    /// is visible alongside benchmark numbers.
    pub fn tail_stride_bytes(&self) -> Option<u64> {
        if self.use_local_kernel() {
            Some((self.n as u64 / BLOCK_SIZE as u64) * std::mem::size_of::<u32>() as u64)
        } else {
            None
        }
    }
}
