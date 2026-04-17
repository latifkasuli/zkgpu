use zkgpu_babybear::BabyBear;

use crate::buffer::WgpuBuffer;

use super::FourStepPlan;
use super::super::planner::StockhamPlanConfig;

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

        // Workgroup counts scale with the selected transpose variant's tile size.
        // The `FourStepPlanConfig` fields use the legacy Tile16 sizing; when
        // the tuned Tile32 kernel is active we recompute here so the dispatch
        // matches the shader's workgroup layout.
        let tile = self.transpose_variant.tile_size();
        let (rc_wgx, rc_wgy) = (self.config.cols.div_ceil(tile), self.config.rows.div_ceil(tile));
        let (cr_wgx, cr_wgy) = (self.config.rows.div_ceil(tile), self.config.cols.div_ceil(tile));

        // Phase 1: Transpose R×C → C×R (buf → transpose_scratch → buf)
        dispatch_idx = self.encode_transpose(
            wgpu_device,
            encoder,
            &buf.inner,
            &self.transpose_scratch_buffer,
            &self.transpose_rc_to_cr_params,
            rc_wgx,
            rc_wgy,
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
            &self.phase2_r8_twiddle_buffer,
            &self.phase2_r8_twiddle_prime_buffer,
            &self.phase2_r8_stage_param_buffers,
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
        //
        // NVIDIA scale-up Tier 2A Option A (2026-04-16): binding 3
        // (twiddle_prime) dropped — see `babybear_fourstep_twiddle.wgsl`
        // comment header for rationale.
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
            cr_wgx, // C×R: x-tiles = R/tile, y-tiles = C/tile
            cr_wgy,
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
            &self.phase5_r8_twiddle_buffer,
            &self.phase5_r8_twiddle_prime_buffer,
            &self.phase5_r8_stage_param_buffers,
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
            rc_wgx,
            rc_wgy,
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

    /// Encode batched R8 + R4 + R2 global Stockham stages for leaf NTTs.
    ///
    /// Dispatches R8 stages first (consume 3 logical stages each,
    /// introduced in T3.A 2026-04-17), then R4 stages (2 stages each),
    /// then the R2 residue (1 stage). The src/dst ping-pong continues
    /// across all radix bands because each dispatch swaps.
    #[allow(clippy::too_many_arguments)]
    fn encode_batched_leaf_r4(
        &self,
        wgpu_device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        buf: &WgpuBuffer<BabyBear>,
        leaf_config: &StockhamPlanConfig,
        // `batch_count` (= `cols` for Phase 2, `rows` for Phase 5) is no
        // longer used at encode time after Fix 2b moved the dispatch
        // computation to plan-build time, but the callers still pass it
        // and it documents the encode-site invariant.
        _batch_count: u32,
        r8_twiddle_buffer: &wgpu::Buffer,
        r8_twiddle_prime_buffer: &wgpu::Buffer,
        r8_stage_param_buffers: &[wgpu::Buffer],
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

        // NVIDIA scale-up T3.A (2026-04-17): R8 dispatches run first
        // (smallest `s` per `StockhamPlanConfig::new_global_only` greedy
        // factoring). Each R8 butterfly covers 3 logical Stockham stages
        // and processes 8 elements. Total workgroup coverage = n/8.
        for param_buffer in r8_stage_param_buffers {
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
                        resource: r8_twiddle_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: param_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: r8_twiddle_prime_buffer.as_entire_binding(),
                    },
                ],
            });

            let ts = ts_writes.get(dispatch_idx).and_then(|t| t.clone());
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: ts,
                });
                pass.set_pipeline(&self.leaf_r8_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                // 2D-folded dispatch: leaf_r8_dispatch covers n/8 butterflies.
                pass.dispatch_workgroups(
                    self.leaf_r8_dispatch.x,
                    self.leaf_r8_dispatch.y,
                    1,
                );
            }
            dispatch_idx += 1;
            parity += 1;
        }

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

            let ts = ts_writes.get(dispatch_idx).and_then(|t| t.clone());
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: ts,
                });
                pass.set_pipeline(&self.leaf_r4_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                // Tier 1 Fix 2b (2026-04-16): 2D-folded dispatch.
                // `leaf_r4_dispatch.{x,y}` together cover
                // `batch_count * leaf_config.n / 4` butterflies; 2D
                // grid avoids the wgpu 65535 per-dim limit at log ≥ 26.
                // See matching `tid = gid.x + gid.y * groups_per_row * 256`
                // in `babybear_fourstep_leaf_r4.wgsl`.
                pass.dispatch_workgroups(
                    self.leaf_r4_dispatch.x,
                    self.leaf_r4_dispatch.y,
                    1,
                );
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

            let ts = ts_writes.get(dispatch_idx).and_then(|t| t.clone());
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: ts,
                });
                pass.set_pipeline(&self.leaf_global_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                // Tier 1 Fix 2b (2026-04-16): 2D-folded dispatch —
                // see R4 site above for rationale.
                pass.dispatch_workgroups(
                    self.leaf_r2_dispatch.x,
                    self.leaf_r2_dispatch.y,
                    1,
                );
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
