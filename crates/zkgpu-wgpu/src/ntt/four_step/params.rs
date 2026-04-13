use wgpu::util::DeviceExt;

use super::super::planner::StockhamPlanConfig;

pub(super) fn build_batched_r4_stage_params(
    device: &wgpu::Device,
    leaf: &StockhamPlanConfig,
    batch_count: u32,
    omega4: u32,
    omega4_prime: u32,
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
                0u32,
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
                0u32,
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
