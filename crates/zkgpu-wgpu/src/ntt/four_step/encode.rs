use zkgpu_babybear::BabyBear;

use crate::buffer::WgpuBuffer;

use super::FourStepPlan;
use super::super::planner::{StockhamPlanConfig, WORKGROUP_SIZE};

impl FourStepPlan {
    /// Encode all six phases of the four-step NTT.
    pub(super) fn encode_all(
        &self,
        wgpu_device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        buf: &WgpuBuffer<BabyBear>,
        ts_writes: &[Option<wgpu::ComputePassTimestampWrites<'_>>],
    ) {
        let buf_size = (self.config.n as u64) * std::mem::size_of::<u32>() as u64;
        let mut dispatch_idx = 0usize;

        // Phase 1: Transpose R×C → C×R (buf → transpose_scratch → buf)
        dispatch_idx = self.encode_transpose(
            wgpu_device,
            encoder,
            &buf.inner,
            &self.transpose_scratch_buffer,
            &self.transpose_rc_to_cr_params,
            self.config.transpose_workgroups_x,
            self.config.transpose_workgroups_y,
            ts_writes,
            dispatch_idx,
        );
        encoder.copy_buffer_to_buffer(&self.transpose_scratch_buffer, 0, &buf.inner, 0, buf_size);

        // Phase 2: R-point batched row DFTs (C batches)
        // Data is now C×R in buf. C independent R-point NTTs on contiguous rows.
        dispatch_idx = self.encode_batched_leaf_r4(
            wgpu_device,
            encoder,
            buf,
            &self.config.col_leaf,
            self.config.cols,
            &self.phase2_r4_twiddle_buffer,
            &self.phase2_r4_twiddle_prime_buffer,
            &self.phase2_r4_stage_param_buffers,
            &self.phase2_r2_twiddle_buffer,
            &self.phase2_r2_twiddle_prime_buffer,
            &self.phase2_r2_stage_param_buffers,
            ts_writes,
            dispatch_idx,
        );

        // Phase 3: Twiddle multiply on C×R data (in-place on buf)
        {
            let bind_group = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.twiddle_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.inner.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.twiddle_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.twiddle_param_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.twiddle_prime_buffer.as_entire_binding(),
                    },
                ],
            });

            let ts = ts_writes.get(dispatch_idx).and_then(|t| t.clone());
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: ts,
                });
                pass.set_pipeline(&self.twiddle_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(self.twiddle_dispatch.x, self.twiddle_dispatch.y, 1);
            }
            dispatch_idx += 1;
        }

        // Phase 4: Transpose C×R → R×C (buf → transpose_scratch → buf)
        dispatch_idx = self.encode_transpose(
            wgpu_device,
            encoder,
            &buf.inner,
            &self.transpose_scratch_buffer,
            &self.transpose_cr_to_rc_params,
            self.config.transpose_workgroups_y, // C×R: x-tiles = R/tile, y-tiles = C/tile
            self.config.transpose_workgroups_x,
            ts_writes,
            dispatch_idx,
        );
        encoder.copy_buffer_to_buffer(&self.transpose_scratch_buffer, 0, &buf.inner, 0, buf_size);

        // Phase 5: C-point batched row DFTs (R batches)
        // Data is now R×C in buf. R independent C-point NTTs on contiguous rows.
        dispatch_idx = self.encode_batched_leaf_r4(
            wgpu_device,
            encoder,
            buf,
            &self.config.row_leaf,
            self.config.rows,
            &self.phase5_r4_twiddle_buffer,
            &self.phase5_r4_twiddle_prime_buffer,
            &self.phase5_r4_stage_param_buffers,
            &self.phase5_r2_twiddle_buffer,
            &self.phase5_r2_twiddle_prime_buffer,
            &self.phase5_r2_stage_param_buffers,
            ts_writes,
            dispatch_idx,
        );

        // Phase 6: Transpose R×C → C×R (final output reordering)
        dispatch_idx = self.encode_transpose(
            wgpu_device,
            encoder,
            &buf.inner,
            &self.transpose_scratch_buffer,
            &self.transpose_rc_to_cr_params,
            self.config.transpose_workgroups_x,
            self.config.transpose_workgroups_y,
            ts_writes,
            dispatch_idx,
        );
        encoder.copy_buffer_to_buffer(&self.transpose_scratch_buffer, 0, &buf.inner, 0, buf_size);

        // Phase 7: Inverse scale
        if let Some(ref param_buf) = self.scale_param_buffer {
            let bind_group = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.scale_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.inner.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: param_buf.as_entire_binding(),
                    },
                ],
            });

            let ts = ts_writes.get(dispatch_idx).and_then(|t| t.clone());
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: ts,
                });
                pass.set_pipeline(&self.scale_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(self.scale_dispatch.x, self.scale_dispatch.y, 1);
            }
        }
    }

    /// Encode a single transpose dispatch (src → dst).
    #[allow(clippy::too_many_arguments)]
    fn encode_transpose(
        &self,
        wgpu_device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::Buffer,
        dst: &wgpu::Buffer,
        param_buffer: &wgpu::Buffer,
        workgroups_x: u32,
        workgroups_y: u32,
        ts_writes: &[Option<wgpu::ComputePassTimestampWrites<'_>>],
        dispatch_idx: usize,
    ) -> usize {
        let bind_group = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.transpose_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: src.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dst.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: param_buffer.as_entire_binding(),
                },
            ],
        });

        let ts = ts_writes.get(dispatch_idx).and_then(|t| t.clone());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: ts,
            });
            pass.set_pipeline(&self.transpose_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }
        dispatch_idx + 1
    }

    /// Encode batched R4 + R2 global Stockham stages for leaf NTTs.
    ///
    /// Dispatches R4 stages first using `leaf_r4_pipeline`, then any R2
    /// remainder stages using `leaf_global_pipeline`. The src/dst ping-pong
    /// continues across both R4 and R2 dispatches.
    #[allow(clippy::too_many_arguments)]
    fn encode_batched_leaf_r4(
        &self,
        wgpu_device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        buf: &WgpuBuffer<BabyBear>,
        leaf_config: &StockhamPlanConfig,
        batch_count: u32,
        r4_twiddle_buffer: &wgpu::Buffer,
        r4_twiddle_prime_buffer: &wgpu::Buffer,
        r4_stage_param_buffers: &[wgpu::Buffer],
        r2_twiddle_buffer: &wgpu::Buffer,
        r2_twiddle_prime_buffer: &wgpu::Buffer,
        r2_stage_param_buffers: &[wgpu::Buffer],
        ts_writes: &[Option<wgpu::ComputePassTimestampWrites<'_>>],
        start_dispatch: usize,
    ) -> usize {
        let mut dispatch_idx = start_dispatch;
        let mut parity = 0usize;

        // R4 dispatches: each R4 butterfly processes 4 elements → leaf_n/4 butterflies per batch
        for param_buffer in r4_stage_param_buffers {
            let (src_buf, dst_buf) = if parity % 2 == 0 {
                (&buf.inner, &self.scratch_buffer)
            } else {
                (&self.scratch_buffer, &buf.inner)
            };

            let bind_group = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.leaf_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: src_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: dst_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: r4_twiddle_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: param_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: r4_twiddle_prime_buffer.as_entire_binding(),
                    },
                ],
            });

            let total_r4_butterflies = batch_count * (leaf_config.n / 4);
            let workgroups = total_r4_butterflies.div_ceil(WORKGROUP_SIZE);

            let ts = ts_writes.get(dispatch_idx).and_then(|t| t.clone());
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: ts,
                });
                pass.set_pipeline(&self.leaf_r4_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            dispatch_idx += 1;
            parity += 1;
        }

        // R2 remainder dispatches
        for param_buffer in r2_stage_param_buffers {
            let (src_buf, dst_buf) = if parity % 2 == 0 {
                (&buf.inner, &self.scratch_buffer)
            } else {
                (&self.scratch_buffer, &buf.inner)
            };

            let bind_group = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.leaf_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: src_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: dst_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: r2_twiddle_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: param_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: r2_twiddle_prime_buffer.as_entire_binding(),
                    },
                ],
            });

            let total_butterflies = batch_count * (leaf_config.n / 2);
            let workgroups = total_butterflies.div_ceil(WORKGROUP_SIZE);

            let ts = ts_writes.get(dispatch_idx).and_then(|t| t.clone());
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: ts,
                });
                pass.set_pipeline(&self.leaf_global_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            dispatch_idx += 1;
            parity += 1;
        }

        // Copy result back to buf if it ended in scratch
        if leaf_config.result_in_scratch {
            let size = (self.config.n as u64) * std::mem::size_of::<u32>() as u64;
            encoder.copy_buffer_to_buffer(&self.scratch_buffer, 0, &buf.inner, 0, size);
        }

        dispatch_idx
    }
}
