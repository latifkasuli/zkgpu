use zkgpu_core::ZkGpuError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LinearDispatch {
    pub x: u32,
    pub y: u32,
    /// Number of workgroups laid out across the x dimension before wrapping to y.
    pub groups_per_row: u32,
}

/// Plan a 1D logical dispatch over `total_elements` while respecting
/// `max_compute_workgroups_per_dimension`.
///
/// The resulting grid uses `x` workgroups across and wraps additional groups
/// into the `y` dimension. Kernels can reconstruct a flat index with:
///
/// `idx = gid.x + gid.y * groups_per_row * workgroup_size`
pub(crate) fn plan_linear_dispatch(
    total_elements: u32,
    workgroup_size: u32,
    max_workgroups_per_dimension: u32,
) -> Result<LinearDispatch, ZkGpuError> {
    debug_assert!(workgroup_size > 0);
    debug_assert!(max_workgroups_per_dimension > 0);

    let total_groups = (total_elements as u64).div_ceil(workgroup_size as u64);
    let max_dim = max_workgroups_per_dimension as u64;
    let x = total_groups.min(max_dim);
    let y = total_groups.div_ceil(x);

    if y > max_dim {
        return Err(ZkGpuError::GpuValidation(format!(
            "dispatch grid for {total_elements} elements with workgroup_size={workgroup_size} \
             exceeds device limits ({max_workgroups_per_dimension} x {max_workgroups_per_dimension})"
        )));
    }

    Ok(LinearDispatch {
        x: x as u32,
        y: y as u32,
        groups_per_row: x as u32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_dispatch_stays_1d_below_limit() {
        let d = plan_linear_dispatch(1024, 256, 65_535).unwrap();
        assert_eq!(d.x, 4);
        assert_eq!(d.y, 1);
        assert_eq!(d.groups_per_row, 4);
    }

    #[test]
    fn linear_dispatch_wraps_at_limit_boundary() {
        let elems = 65_536u32 * 256u32;
        let d = plan_linear_dispatch(elems, 256, 65_535).unwrap();
        assert_eq!(d.x, 65_535);
        assert_eq!(d.y, 2);
        assert_eq!(d.groups_per_row, 65_535);
    }

    #[test]
    fn linear_dispatch_handles_babybear_max_size() {
        let elems = 1u32 << 27;
        let d = plan_linear_dispatch(elems, 256, 65_535).unwrap();
        assert_eq!(d.x, 65_535);
        assert_eq!(d.y, 9);
    }
}
