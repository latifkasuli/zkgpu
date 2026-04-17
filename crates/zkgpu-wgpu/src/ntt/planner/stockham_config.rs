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

/// Parameters for a radix-8 global Stockham DIF dispatch (combines 3 stages).
///
/// NVIDIA scale-up Tier 3 Option A (T3.A, 2026-04-17): R8 leaves halve the
/// memory round-trips per 3 stages vs R4. At log-22 Four-Step this drops
/// the leaf dispatch count from 6 (5 R4 + 1 R2) to 4 (3 R8 + 1 R4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct R8StageParams {
    pub n: u32,
    pub s: u32,
    pub m8: u32,
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
    /// Radix-8 stage params (each combines 3 logical stages into 1 dispatch).
    /// NVIDIA scale-up T3.A: populated only by `new_global_only` (used in
    /// four-step leaves where the R8 kernel is wired). The `new` path with
    /// a tail decision leaves this empty and uses R4-only factoring.
    pub r8_stage_params: Vec<R8StageParams>,
    /// (h, m8) pairs for R8 twiddle generation, one per R8 dispatch.
    pub r8_twiddle_spec: Vec<(u32, u32)>,
    /// Number of global dispatches: r8_count + r4_count + r2_count.
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
            // Top-level Stockham does not wire an R8 kernel (T3.A is Four-Step
            // leaves only for now). Leave R8 fields empty.
            r8_stage_params: Vec::new(),
            r8_twiddle_spec: Vec::new(),
            num_global_dispatches,
        })
    }

    /// Plan a Stockham NTT for four-step leaves with R8/R4/R2 factoring
    /// (NVIDIA scale-up T3.A, 2026-04-17).
    ///
    /// Used by four-step batched leaves — the R8 kernel is a separate
    /// pipeline (`babybear_fourstep_leaf_r8.wgsl`) only built into
    /// `FourStepPlan`. Top-level Stockham still uses R4-only factoring via
    /// `Self::new(log_n, None)`.
    ///
    /// `r8_max_log_leaf` caps the leaf size at which R8 is used. See
    /// [`PlannerPolicy::r8_max_log_leaf`](super::policy::PlannerPolicy::r8_max_log_leaf)
    /// for the per-family decision table.
    ///
    /// Greedy factoring:
    ///   num_r8 = num_stages / 3
    ///   remainder = num_stages % 3
    ///   num_r4 = remainder / 2
    ///   has_r2 = remainder % 2 == 1
    ///
    /// Dispatch ordering: R8 stages run first (s = 1, 8, 64, …), then R4
    /// stages (s = 8^num_r8, …), then optional R2 residue.
    pub fn new_global_only(log_n: u32, r8_max_log_leaf: u32) -> Result<Self, ZkGpuError> {
        if log_n == 0 || log_n > MAX_LOG_N {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "log_n={log_n} out of range (must be 1..={MAX_LOG_N})"
            )));
        }

        let n = 1u32 << log_n;
        let num_global_stages = log_n;

        let num_butterflies = n / 2;
        let global_workgroups = num_butterflies.div_ceil(WORKGROUP_SIZE);

        // T3.A gate (2026-04-17): passed in per-family from PlannerPolicy.
        // See `PlannerPolicy::r8_max_log_leaf` for the per-(backend, family)
        // decision matrix and evidence base (M4 Pro, RTX 4090 5-trial
        // medians).
        let use_r8 = num_global_stages <= r8_max_log_leaf;

        // Greedy R8 → R4 → R2 factoring.
        let num_r8 = if use_r8 { num_global_stages / 3 } else { 0 };
        let stages_after_r8 = num_global_stages - 3 * num_r8;
        let num_r4 = stages_after_r8 / 2;
        let has_r2_remainder = stages_after_r8 % 2 == 1;

        // Build R8 stage params: stages (0, 1, 2), (3, 4, 5), (6, 7, 8), …
        let mut r8_stage_params = Vec::with_capacity(num_r8 as usize);
        let mut r8_twiddle_spec = Vec::with_capacity(num_r8 as usize);
        let mut r8_twiddle_offset = 0u32;
        for i in 0..num_r8 {
            let h = i * 3;
            let s = 1u32 << h;
            let m8 = n / (8 * s);
            r8_stage_params.push(R8StageParams {
                n,
                s,
                m8,
                twiddle_offset: r8_twiddle_offset,
            });
            r8_twiddle_spec.push((h, m8));
            r8_twiddle_offset += 7 * m8; // 7 twiddles per butterfly position
        }

        // Build R4 stage params: start after the R8 stages.
        let r4_start_h = 3 * num_r8;
        let mut r4_stage_params = Vec::with_capacity(num_r4 as usize);
        let mut r4_twiddle_spec = Vec::with_capacity(num_r4 as usize);
        let mut r4_twiddle_offset = 0u32;
        for i in 0..num_r4 {
            let h = r4_start_h + i * 2;
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

        // R2 residue (at most one dispatch).
        let mut global_stage_params = Vec::new();
        if has_r2_remainder {
            let h = r4_start_h + 2 * num_r4;
            let s = 1u32 << h;
            let m = n >> (h + 1);
            global_stage_params.push(GlobalStageParams {
                n,
                s,
                m,
                twiddle_offset: 0,
            });
        }

        let num_global_dispatches = num_r8 + num_r4 + u32::from(has_r2_remainder);
        let total_swaps = num_global_dispatches;
        let result_in_scratch = total_swaps % 2 == 1;

        Ok(Self {
            log_n,
            n,
            tail: None,
            num_global_stages,
            global_workgroups,
            local_workgroups: 0,
            local_stride: 1,
            result_in_scratch,
            global_stage_params,
            r4_stage_params,
            r4_twiddle_spec,
            r8_stage_params,
            r8_twiddle_spec,
            num_global_dispatches,
        })
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
