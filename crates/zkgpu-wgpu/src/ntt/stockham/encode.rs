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
        // Gate 2 item #4 (multi-dispatch per pass): the previous version
        // of this function began a separate `wgpu::ComputePass` per stage
        // (one per R4, one per R2, one for the local fused kernel) — 5+
        // passes per NTT call at typical log_n. Each `begin_compute_pass`
        // pays driver-side overhead that's a meaningful fraction of small-
        // log_n wall time on integrated / mobile / browser. The fold
        // collapses them into ONE pass with `set_pipeline` switches and
        // `set_bind_group` swaps between dispatches inside it. Memory
        // barriers on the ping-pong (`buf` ↔ `scratch`) buffers still
        // get emitted by the driver (correctness invariant — see
        // wgpu issue #5766) so we don't lose synchronization, just the
        // per-pass setup cost.
        //
        // Trade-off: per-stage `ComputePassTimestampWrites` collapses to
        // a single beginning/end timestamp pair on the outer pass.
        // Profiled benches that need per-stage breakdown should use the
        // sibling `profiled.rs` path which retains separate passes for
        // measurement granularity.
        //
        // Bind groups are created upfront in a Vec because each one must
        // outlive its `set_bind_group` call inside the pass — wgpu's pass
        // recorder retains references to bind groups until the pass ends.

        // Pre-build all bind groups with their associated pipelines.
        enum Stage<'a> {
            R4(wgpu::BindGroup),
            R2Global(wgpu::BindGroup),
            Local(wgpu::BindGroup),
            #[allow(dead_code)]
            _Phantom(std::marker::PhantomData<&'a ()>),
        }

        let mut stages: Vec<Stage<'_>> = Vec::with_capacity(
            self.r4_stage_param_buffers.len()
                + self.global_stage_param_buffers.len()
                + 1,
        );
        let mut dispatch_idx: usize = 0;

        // Phase 1a: Radix-4 global stage bind groups
        for param_buffer in self.r4_stage_param_buffers.iter() {
            let (src_buf, dst_buf) = if dispatch_idx % 2 == 0 {
                (&buf.inner, &self.scratch_buffer)
            } else {
                (&self.scratch_buffer, &buf.inner)
            };
            let bg = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
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
            stages.push(Stage::R4(bg));
            dispatch_idx += 1;
        }

        // Phase 1b: Radix-2 remainder global stage bind groups
        for param_buffer in &self.global_stage_param_buffers {
            let (src_buf, dst_buf) = if dispatch_idx % 2 == 0 {
                (&buf.inner, &self.scratch_buffer)
            } else {
                (&self.scratch_buffer, &buf.inner)
            };
            let bg = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
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
            stages.push(Stage::R2Global(bg));
            dispatch_idx += 1;
        }

        // Phase 2: workgroup-local fused dispatch (skipped under
        // GlobalOnlyR4 tail strategy). KEPT IN SEPARATE PASS because
        // experiments with the local kernel folded into the same pass
        // as the global stages broke parity at log_n ≥ 11 (where both
        // global and local dispatches coexist). The hypothesized cause
        // is wgpu's automatic barrier emission not correctly handling
        // a workgroup-memory-using pipeline switching in after non-
        // workgroup-memory pipelines within the same pass — bears
        // investigation, but for now the safe fold is "global stages
        // together; local in its own pass." That still saves
        // (num_global_stages - 1) pass-begin/end pairs per NTT, which
        // is the dominant share at log_n ≥ 14.
        let local_bg = if self.config.use_local_kernel() {
            let (src_buf, dst_buf) = if dispatch_idx % 2 == 0 {
                (&buf.inner, &self.scratch_buffer)
            } else {
                (&self.scratch_buffer, &buf.inner)
            };
            Some(wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
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
            }))
        } else {
            None
        };

        // Pick the first non-None timestamp config (if any) for the
        // outer pass. Per-stage breakdown is lost; total
        // global-stages time still measurable.
        let outer_ts = ts_writes
            .iter()
            .find_map(|t| t.as_ref().cloned());

        // Pass 1: all global stages (R4 + R2) folded together.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: outer_ts,
            });

            for stage in &stages {
                let (pipeline, bind_group, dispatch) = match stage {
                    Stage::R4(bg) => (&self.r4_pipeline, bg, &self.r4_dispatch),
                    Stage::R2Global(bg) => {
                        (&self.global_pipeline, bg, &self.r2_dispatch)
                    }
                    Stage::Local(_) => unreachable!(
                        "Local stage handled separately below"
                    ),
                    Stage::_Phantom(_) => unreachable!(),
                };

                // Set pipeline + bind group before every dispatch.
                // Skipping redundant set_pipeline calls between
                // dispatches with identical pipelines was tried and
                // broke parity — the redundant call is cheap and the
                // skip optimization isn't worth correctness risk.
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(dispatch.x, dispatch.y, 1);
            }
        }

        // Pass 2: local fused dispatch (own pass — see comment above).
        if let Some(bg) = local_bg.as_ref() {
            let local_ts = ts_writes
                .get(dispatch_idx)
                .and_then(|t| t.clone());
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: local_ts,
            });
            pass.set_pipeline(&self.local_pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(self.local_dispatch.x, self.local_dispatch.y, 1);
        }

        // Copy-back if the final result landed in the scratch buffer.
        // Outside the pass (it's a buffer copy, not a compute dispatch).
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
