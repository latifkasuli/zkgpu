use wgpu::util::DeviceExt;

use super::super::planner::StockhamPlanConfig;

// NVIDIA scale-up Tier 1 Fix 2b (2026-04-16): added `groups_per_row`
// to the batched R4 + R2 leaf uniform params (slot 7 / slot 5
// respectively — what the WGSL side called `_pad0`). Every stage
// within a phase shares the same dispatch count
// (`batch_count * leaf_n / radix`), so every stage param buffer
// gets the same `groups_per_row` value computed by the caller via
// `plan_linear_dispatch`. Workgroups wrap into `gid.y` once
// `batch_count * leaf_n / (4 * 256)` exceeds the wgpu per-dimension
// limit (65535), i.e. once log_n ≥ 26 for BabyBear. See matching
// WGSL comments in `babybear_fourstep_leaf_{r4,r2}.wgsl`.
pub(super) fn build_batched_r4_stage_params(
    device: &wgpu::Device,
    leaf: &StockhamPlanConfig,
    batch_count: u32,
    omega4: u32,
    omega4_prime: u32,
    groups_per_row: u32,
) -> Vec<wgpu::Buffer> {
    leaf.r4_stage_params
        .iter()
        .map(|sp| {
            let params = [
                sp.n,
                sp.s,
                sp.m4,
                sp.twiddle_offset,
                batch_count,
                omega4,
                omega4_prime,
                groups_per_row,
            ];
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Batched R4 leaf stage params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            })
        })
        .collect()
}

pub(super) fn build_batched_r2_stage_params(
    device: &wgpu::Device,
    leaf: &StockhamPlanConfig,
    batch_count: u32,
    groups_per_row: u32,
) -> Vec<wgpu::Buffer> {
    leaf.global_stage_params
        .iter()
        .map(|sp| {
            let params = [
                sp.n,
                sp.s,
                sp.m,
                sp.twiddle_offset,
                batch_count,
                groups_per_row,
                0u32,
                0u32,
            ];
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Batched R2 leaf stage params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            })
        })
        .collect()
}

/// NVIDIA scale-up T3.A (2026-04-17): R8 leaf stage params. Layout
/// matches `BatchedR8Params` in `babybear_fourstep_leaf_r8.wgsl`:
///   { leaf_n, s, m8, twiddle_offset,
///     batch_count, omega8, omega8_prime,
///     omega4, omega4_prime, omega8_cubed, omega8_cubed_prime,
///     groups_per_row }
/// = 12 u32s = 48 bytes (multiple of 16, uniform-buffer-safe).
#[allow(clippy::too_many_arguments)]
pub(super) fn build_batched_r8_stage_params(
    device: &wgpu::Device,
    leaf: &StockhamPlanConfig,
    batch_count: u32,
    omega8: u32,
    omega8_prime: u32,
    omega4: u32,
    omega4_prime: u32,
    omega8_cubed: u32,
    omega8_cubed_prime: u32,
    groups_per_row: u32,
) -> Vec<wgpu::Buffer> {
    leaf.r8_stage_params
        .iter()
        .map(|sp| {
            let params = [
                sp.n,
                sp.s,
                sp.m8,
                sp.twiddle_offset,
                batch_count,
                omega8,
                omega8_prime,
                omega4,
                omega4_prime,
                omega8_cubed,
                omega8_cubed_prime,
                groups_per_row,
            ];
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Batched R8 leaf stage params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            })
        })
        .collect()
}
