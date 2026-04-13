use zkgpu_core::ZkGpuError;

use super::constants::{BLOCK_SIZE, LOG_BLOCK, MAX_LOG_N, WORKGROUP_SIZE};

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

/// Structural decisions for a Stockham hybrid NTT execution plan.
///
/// All fields are derived from `log_n` and the compile-time constants
/// `WORKGROUP_SIZE`, `BLOCK_SIZE`, and `LOG_BLOCK`. No GPU interaction,
/// no field-specific arithmetic — fully testable on any host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StockhamPlanConfig {
    pub log_n: u32,
    pub n: u32,
    pub use_local_kernel: bool,
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
    /// Returns `Err` if `log_n` is 0 or exceeds `MAX_LOG_N`.
    pub fn new(log_n: u32) -> Result<Self, ZkGpuError> {
        if log_n == 0 || log_n > MAX_LOG_N {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "log_n={log_n} out of range (must be 1..={MAX_LOG_N})"
            )));
        }
        let n = 1u32 << log_n;

        let use_local_kernel = log_n >= LOG_BLOCK;
        let num_global_stages = if use_local_kernel {
            log_n - LOG_BLOCK
        } else {
            log_n
        };

        let num_butterflies = n / 2;
        let global_workgroups = num_butterflies.div_ceil(WORKGROUP_SIZE);

        let local_workgroups = if use_local_kernel { n / BLOCK_SIZE } else { 0 };
        let local_stride = if use_local_kernel { n / BLOCK_SIZE } else { 1 };

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
        let total_swaps = num_global_dispatches + u32::from(use_local_kernel);
        let result_in_scratch = total_swaps % 2 == 1;

        Ok(Self {
            log_n,
            n,
            use_local_kernel,
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

    /// Plan a Stockham NTT that uses only global DIF stages (no local kernel).
    ///
    /// Needed for four-step batched leaves where the local kernel's strided
    /// gather/scatter pattern doesn't match the contiguous batch layout.
    /// Uses radix-4 pairing for global stages (same as the main planner).
    pub fn new_global_only(log_n: u32) -> Result<Self, ZkGpuError> {
        if log_n == 0 || log_n > MAX_LOG_N {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "log_n={log_n} out of range (must be 1..={MAX_LOG_N})"
            )));
        }
        let n = 1u32 << log_n;
        let num_global_stages = log_n;
        let num_butterflies = n / 2;
        let global_workgroups = num_butterflies.div_ceil(WORKGROUP_SIZE);

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
        let result_in_scratch = num_global_dispatches % 2 == 1;

        Ok(Self {
            log_n,
            n,
            use_local_kernel: false,
            num_global_stages,
            global_workgroups,
            local_workgroups: 0,
            local_stride: 1,
            result_in_scratch,
            global_stage_params,
            r4_stage_params,
            r4_twiddle_spec,
            num_global_dispatches,
        })
    }

    /// Number of NTT stage dispatches (R4 + R2 global dispatches + optional local).
    pub fn ntt_dispatches(&self) -> u32 {
        self.num_global_dispatches + u32::from(self.use_local_kernel)
    }
}
