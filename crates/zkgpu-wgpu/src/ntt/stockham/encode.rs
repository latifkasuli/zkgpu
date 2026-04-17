use zkgpu_babybear::BabyBear;

use crate::buffer::WgpuBuffer;

use super::StockhamPlan;

impl StockhamPlan {
    /// Encode NTT stages into the given command encoder.
    ///
    /// Dispatch order: R4 global stages -> R2 global stages -> local fused.
    pub(super) fn encode_ntt_stages(
        &self,
        wgpu_device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        buf: &WgpuBuffer<BabyBear>,
        ts_writes: &[Option<wgpu::ComputePassTimestampWrites<'_>>],
    ) {
        let mut dispatch_idx: usize = 0;

        // Phase 1a: Radix-4 global dispatches
        for param_buffer in self.r4_stage_param_buffers.iter() {
            let (src_buf, dst_buf) = if dispatch_idx % 2 == 0 {
                (&buf.inner, &self.scratch_buffer)
            } else {
                (&self.scratch_buffer, &buf.inner)
            };

            let bind_group = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.ntt_bind_group_layout,
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
                        resource: self.r4_twiddle_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: param_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.r4_twiddle_prime_buffer.as_entire_binding(),
                    },
                ],
            });

            let ts = ts_writes.get(dispatch_idx).and_then(|t| t.clone());

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: ts,
                });
                pass.set_pipeline(&self.r4_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                // Tier 1 Fix 2 (2026-04-16): 2D-folded dispatch. Total
                // workgroups = r4_dispatch.x * r4_dispatch.y covers
                // `config.n / 4` butterflies; log_n ≥ 25 workloads wrap
                // into the y-dimension instead of hitting wgpu's
                // per-dimension limit. See the WGSL kernel for the
                // matching `tid = gid.x + gid.y * groups_per_row * 256`
                // reconstruction.
                pass.dispatch_workgroups(self.r4_dispatch.x, self.r4_dispatch.y, 1);
            }

            dispatch_idx += 1;
        }

        // Phase 1b: Radix-2 remainder global dispatches
        for param_buffer in &self.global_stage_param_buffers {
            let (src_buf, dst_buf) = if dispatch_idx % 2 == 0 {
                (&buf.inner, &self.scratch_buffer)
            } else {
                (&self.scratch_buffer, &buf.inner)
            };

            let bind_group = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.ntt_bind_group_layout,
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
                        resource: self.global_twiddle_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: param_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.global_twiddle_prime_buffer.as_entire_binding(),
                    },
                ],
            });

            let ts = ts_writes.get(dispatch_idx).and_then(|t| t.clone());

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: ts,
                });
                pass.set_pipeline(&self.global_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                // Tier 1 Fix 2 (2026-04-16): 2D-folded dispatch — see
                // R4 site above for the rationale.
                pass.dispatch_workgroups(self.r2_dispatch.x, self.r2_dispatch.y, 1);
            }

            dispatch_idx += 1;
        }

        // Phase 2: workgroup-local fused dispatch
        // Only emitted when tail.strategy == LocalFusedR4. The GlobalOnlyR4
        // tail strategy fuses these stages into the global R4 chain above,
        // so this branch is skipped entirely.
        if self.config.use_local_kernel() {
            let (src_buf, dst_buf) = if dispatch_idx % 2 == 0 {
                (&buf.inner, &self.scratch_buffer)
            } else {
                (&self.scratch_buffer, &buf.inner)
            };

            let bind_group = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.ntt_bind_group_layout,
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
                        resource: self.local_twiddle_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.local_param_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.local_twiddle_prime_buffer.as_entire_binding(),
                    },
                ],
            });

            let ts = ts_writes.get(dispatch_idx).and_then(|t| t.clone());

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: ts,
                });
                pass.set_pipeline(&self.local_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                // Tier 1 Fix 2 (2026-04-16): 2D-folded workgroup grid.
                // Local kernel uses workgroup_id (not global_invocation_id)
                // so `block_id = wg_id.x + wg_id.y * groups_per_row` in
                // the WGSL. log_n ≥ 25 ⇒ local_workgroups > 65535.
                pass.dispatch_workgroups(self.local_dispatch.x, self.local_dispatch.y, 1);
            }
        }

        // Copy-back if the final result landed in the scratch buffer
        if self.config.result_in_scratch {
            let size = (self.config.n as u64) * std::mem::size_of::<u32>() as u64;
            encoder.copy_buffer_to_buffer(&self.scratch_buffer, 0, &buf.inner, 0, size);
        }
    }

    /// Encode the inverse scaling dispatch if this is an inverse plan.
    pub(super) fn encode_scale(
        &self,
        wgpu_device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        buf: &WgpuBuffer<BabyBear>,
        ts: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        let Some(ref param_buf) = self.scale_param_buffer else {
            return;
        };

        let bind_group = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.scale_bind_group_layout,
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
