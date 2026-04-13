use zkgpu_core::ZkGpuError;

use super::constants::{MAX_LOG_N, TRANSPOSE_TILE};
use super::stockham_config::StockhamPlanConfig;

// ---------------------------------------------------------------------------
// Four-step planner
// ---------------------------------------------------------------------------

/// Structural decisions for a four-step decomposition NTT.
///
/// Factorizes N = rows * cols and plans leaf transforms plus
/// transpose/twiddle dispatches. Leaf NTTs reuse stockham plans.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FourStepPlanConfig {
    pub log_n: u32,
    pub n: u32,
    pub row_log_n: u32,
    pub col_log_n: u32,
    pub rows: u32,
    pub cols: u32,
    pub transpose_tile: u32,
    pub transpose_workgroups_x: u32,
    pub transpose_workgroups_y: u32,
    pub row_leaf: StockhamPlanConfig,
    pub col_leaf: StockhamPlanConfig,
}

impl FourStepPlanConfig {
    /// Plan a four-step NTT of size `2^log_n`.
    ///
    /// Uses a balanced factorization: rows = 2^floor(log_n/2), cols = 2^ceil(log_n/2).
    pub fn new(log_n: u32) -> Result<Self, ZkGpuError> {
        if log_n == 0 || log_n > MAX_LOG_N {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "log_n={log_n} out of range (must be 1..={MAX_LOG_N})"
            )));
        }
        let n = 1u32 << log_n;

        let row_log_n = log_n / 2;
        let col_log_n = log_n - row_log_n;
        let rows = 1u32 << row_log_n;
        let cols = 1u32 << col_log_n;

        let row_leaf = StockhamPlanConfig::new_global_only(col_log_n)?;
        let col_leaf = StockhamPlanConfig::new_global_only(row_log_n)?;

        let transpose_tile = TRANSPOSE_TILE;
        let transpose_workgroups_x = cols.div_ceil(transpose_tile);
        let transpose_workgroups_y = rows.div_ceil(transpose_tile);

        Ok(Self {
            log_n,
            n,
            row_log_n,
            col_log_n,
            rows,
            cols,
            transpose_tile,
            transpose_workgroups_x,
            transpose_workgroups_y,
            row_leaf,
            col_leaf,
        })
    }

    /// Total dispatches across all six phases.
    pub fn total_dispatches(&self) -> u32 {
        1 // Phase 1: transpose R×C → C×R
            + self.col_leaf.ntt_dispatches() // Phase 2: R-point batched NTTs
            + 1 // Phase 3: twiddle multiply
            + 1 // Phase 4: transpose C×R → R×C
            + self.row_leaf.ntt_dispatches() // Phase 5: C-point batched NTTs
            + 1 // Phase 6: transpose R×C → C×R (output)
    }
}
