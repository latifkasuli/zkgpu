mod constants;
mod four_step_config;
mod policy;
mod stockham_config;
#[cfg(test)]
mod tests;

pub(crate) use constants::{BLOCK_SIZE, LOG_BLOCK, MAX_BABYBEAR_LOG_N, WORKGROUP_SIZE};
pub use policy::PlannerPolicy;
pub(crate) use four_step_config::FourStepPlanConfig;
pub(crate) use stockham_config::StockhamPlanConfig;


use constants::MAX_LOG_N;
use zkgpu_core::ZkGpuError;

// ---------------------------------------------------------------------------
// NTT family selection
// ---------------------------------------------------------------------------

/// Top-level planner decision: which family and config to use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PlannedNtt {
    Stockham(StockhamPlanConfig),
    FourStep(FourStepPlanConfig),
}

/// Choose the best NTT family for a given size and device policy.
pub(crate) fn plan_ntt(log_n: u32, policy: &PlannerPolicy) -> Result<PlannedNtt, ZkGpuError> {
    if log_n == 0 || log_n > MAX_LOG_N {
        return Err(ZkGpuError::InvalidNttSize(format!(
            "log_n={log_n} out of range (must be 1..={MAX_LOG_N})"
        )));
    }

    if let Some(threshold) = policy.four_step_threshold {
        if log_n >= threshold {
            return Ok(PlannedNtt::FourStep(FourStepPlanConfig::new(log_n)?));
        }
    }

    Ok(PlannedNtt::Stockham(StockhamPlanConfig::new(log_n)?))
}
