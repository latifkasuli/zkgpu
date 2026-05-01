use zkgpu_babybear::BabyBear;

use crate::buffer::WgpuBuffer;

use super::{R4ParamSource, StockhamPlan};

/// Selects how `encode_ntt_stages` lays out compute passes.
///
/// `Folded` is the production path (Gate 2 item #4): all global R4+R2
/// stages share a single `wgpu::ComputePass`, with `set_pipeline` and
/// `set_bind_group` switches between dispatches. The local-fused
/// kernel keeps its own pass — folding it broke parity at log_n ≥ 11
/// (hypothesized wgpu barrier-emission interaction with workgroup-
/// memory-using pipelines mid-pass; see comment below).
///
/// `PerPass` is the profiling path. Each logical dispatch gets its
/// own `ComputePass` so the per-stage `ComputePassTimestampWrites`
/// allocated in `profiled.rs` actually fire and report accurate
/// per-stage GPU durations. Folding the profiled path would collapse
/// every per-stage span into a single outer-pass span, making the
/// stage labels in `NttTimings::gpu_stage_ns` lie about durations.
#[derive(Clone, Copy)]
pub(super) enum NttEncodeMode {
    Folded,
    // The integration-test build (`cargo test --test gpu_ntt_validation
    // --no-run`) flags `PerPass` as never-constructed even though
    // `profiled.rs` references it from `execute_kernels_profiled`.
    // The lib-only `cargo check -p zkgpu-wgpu` is clean, so this is a
    // dead-code-analyzer artefact specific to how the integration-test
    // target sees the lib (the public NTT API at `ntt/mod.rs` only
    // reaches `PerPass` via the profiled trampoline; the analyzer
    // apparently doesn't trace far enough). The variant IS used at
    // runtime — every profiled bench/test exercises it. Allow the
    // dead-code lint locally so the test matrix stays warning-clean.
    #[allow(dead_code)]
    PerPass,
}

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
        mode: NttEncodeMode,
    ) {
        // Gate 2 item #4 (multi-dispatch per pass): the previous version
        // of this function began a separate `wgpu::ComputePass` per stage
        // (one per R4, one per R2, one for the local fused kernel) — 5+
        // passes per NTT call at typical log_n. Each `begin_compute_pass`
        // pays driver-side overhead that's a meaningful fraction of small-
        // log_n wall time on integrated / mobile / browser. The Folded
        // mode collapses them into ONE pass with `set_pipeline` switches
        // and `set_bind_group` swaps between dispatches inside it. Memory
        // barriers on the ping-pong (`buf` ↔ `scratch`) buffers still
        // get emitted by the driver (correctness invariant — see
        // wgpu issue #5766) so we don't lose synchronization, just the
        // per-pass setup cost.
        //
        // PerPass mode preserves the original one-pass-per-stage layout
        // for the profiled execution path, so each per-dispatch
        // timestamp span (allocated in `execute_kernels_profiled`) maps
        // 1:1 to a real pass and reports an accurate stage duration.
        //
        // Bind groups are created upfront in a Vec because each one must
        // outlive its `set_bind_group` call inside the pass — wgpu's pass
        // recorder retains references to bind groups until the pass ends.

        // Pre-build all bind groups with their associated pipelines.
        // R4 stage bind groups have either 5 entries (Storage mode) or
        // 4 entries (Immediate mode — slot 3 is gone). The Stage enum
        // carries the optional immediate-payload reference so the
        // dispatch site can call `pass.set_immediates(0, ...)` before
        // each R4 dispatch in Immediate mode.
        enum Stage<'p> {
            R4 {
                bind_group: wgpu::BindGroup,
                /// `Some(&[u32; 8])` when the plan is in Immediate
                /// mode; `None` when the params live in the bind
                /// group at slot 3 (Storage mode).
                immediate: Option<&'p [u32; 8]>,
            },
            R2Global(wgpu::BindGroup),
        }

        let mut stages: Vec<Stage<'_>> = Vec::with_capacity(
            self.r4_param_source.len()
                + self.global_stage_param_buffers.len(),
        );
        let mut dispatch_idx: usize = 0;

        // Phase 1a: Radix-4 global stage bind groups (mode-dependent).
        match &self.r4_param_source {
            R4ParamSource::Storage(buffers) => {
                for param_buffer in buffers.iter() {
                    let (src_buf, dst_buf) = if dispatch_idx % 2 == 0 {
                        (&buf.inner, &self.scratch_buffer)
                    } else {
                        (&self.scratch_buffer, &buf.inner)
                    };
                    let bg = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.r4_bind_group_layout,
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
                    stages.push(Stage::R4 {
                        bind_group: bg,
                        immediate: None,
                    });
                    dispatch_idx += 1;
                }
            }
            R4ParamSource::Immediate(param_blocks) => {
                // Item #5 (bind-group reuse) win specific to the
                // Immediate path: every R4 stage in this branch shares
                // the SAME static bindings (twiddles at slot 2,
                // twiddles_prime at slot 4) and the SAME ping-pong
                // src/dst pair (slots 0/1), with only the parity
                // alternating. The per-stage immediate payload arrives
                // via `pass.set_immediates(0, ...)` later, NOT through
                // the bind group. So instead of building one bind
                // group per stage like the Storage branch must (its
                // slot-3 param_buffer differs per stage), we build
                // exactly TWO here — one per parity — and clone the
                // appropriate one into each stage. wgpu::BindGroup
                // is internally Arc-managed, so `clone()` is a cheap
                // ref-count bump, not a fresh allocation. Cuts the
                // per-encode bind-group create cost in Immediate
                // mode from `O(num_r4_stages)` to `O(1)`.
                let make_bg = |parity: usize| -> wgpu::BindGroup {
                    let (src_buf, dst_buf) = if parity == 0 {
                        (&buf.inner, &self.scratch_buffer)
                    } else {
                        (&self.scratch_buffer, &buf.inner)
                    };
                    wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.r4_bind_group_layout,
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
                                binding: 4,
                                resource: self.r4_twiddle_prime_buffer.as_entire_binding(),
                            },
                        ],
                    })
                };
                // Only build the parities we'll actually use. Empty
                // R4 stage list means no bind groups; one stage means
                // only parity 0; two-or-more covers both parities.
                let need_parity_0 = !param_blocks.is_empty();
                let need_parity_1 = param_blocks.len() >= 2;
                let bg_0 = if need_parity_0 { Some(make_bg(0)) } else { None };
                let bg_1 = if need_parity_1 { Some(make_bg(1)) } else { None };
                for params in param_blocks.iter() {
                    let parity = dispatch_idx % 2;
                    let shared = if parity == 0 {
                        bg_0.as_ref().expect("parity 0 bind group built when needed")
                    } else {
                        bg_1.as_ref().expect("parity 1 bind group built when needed")
                    };
                    stages.push(Stage::R4 {
                        bind_group: shared.clone(),
                        immediate: Some(params),
                    });
                    dispatch_idx += 1;
                }
            }
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
        // GlobalOnlyR4 tail strategy). KEPT IN SEPARATE PASS in Folded
        // mode because experiments with the local kernel folded into
        // the same pass as the global stages broke parity at log_n ≥ 11
        // (where both global and local dispatches coexist). The
        // hypothesized cause is wgpu's automatic barrier emission not
        // correctly handling a workgroup-memory-using pipeline switching
        // in after non-workgroup-memory pipelines within the same pass —
        // bears investigation, but for now the safe fold is "global
        // stages together; local in its own pass." That still saves
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

        match mode {
            NttEncodeMode::Folded => {
                // Pick the first non-None timestamp config (if any) for the
                // outer pass. Per-stage breakdown is lost in this mode by
                // design; the production hot path doesn't request per-stage
                // timestamps. Profiling callers must use PerPass mode.
                let outer_ts = ts_writes.iter().find_map(|t| t.as_ref().cloned());

                // Pass 1: all global stages (R4 + R2) folded together.
                // Skipped entirely when there are no global stages
                // (local-only plans at small log_n) — beginning an empty
                // compute pass is wasteful and can confuse some
                // backends' validation layers.
                if !stages.is_empty() {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: outer_ts,
                    });

                    for stage in &stages {
                        let (pipeline, bind_group, dispatch, immediate) = match stage {
                            Stage::R4 { bind_group, immediate } => (
                                &self.r4_pipeline,
                                bind_group,
                                &self.r4_dispatch,
                                *immediate,
                            ),
                            Stage::R2Global(bg) => (
                                &self.global_pipeline,
                                bg,
                                &self.r2_dispatch,
                                None,
                            ),
                        };

                        // Set pipeline + bind group before every dispatch.
                        // Skipping redundant set_pipeline calls between
                        // dispatches with identical pipelines was tried and
                        // broke parity — the redundant call is cheap and the
                        // skip optimization isn't worth correctness risk.
                        pass.set_pipeline(pipeline);
                        pass.set_bind_group(0, bind_group, &[]);
                        // Item #3 immediate path: write the per-stage
                        // param block to register-resident bytes. The
                        // wgpu API requires immediates be set AFTER the
                        // pipeline (which carries the layout's
                        // `immediate_size`) and BEFORE the dispatch.
                        if let Some(params) = immediate {
                            pass.set_immediates(0, bytemuck::cast_slice(params));
                        }
                        pass.dispatch_workgroups(dispatch.x, dispatch.y, 1);
                    }
                }

                // Pass 2: local fused dispatch (own pass — see comment above).
                if let Some(bg) = local_bg.as_ref() {
                    let local_ts = ts_writes.get(dispatch_idx).and_then(|t| t.clone());
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: local_ts,
                    });
                    pass.set_pipeline(&self.local_pipeline);
                    pass.set_bind_group(0, bg, &[]);
                    pass.dispatch_workgroups(self.local_dispatch.x, self.local_dispatch.y, 1);
                }
            }
            NttEncodeMode::PerPass => {
                // One pass per logical dispatch — preserves the 1:1 mapping
                // from `ts_writes[i]` to a measurable pass duration that
                // `execute_kernels_profiled` relies on for per-stage
                // labelling in `NttTimings::gpu_stage_ns`.
                for (i, stage) in stages.iter().enumerate() {
                    let (pipeline, bind_group, dispatch, immediate) = match stage {
                        Stage::R4 { bind_group, immediate } => (
                            &self.r4_pipeline,
                            bind_group,
                            &self.r4_dispatch,
                            *immediate,
                        ),
                        Stage::R2Global(bg) => {
                            (&self.global_pipeline, bg, &self.r2_dispatch, None)
                        }
                    };
                    let ts = ts_writes.get(i).and_then(|t| t.clone());
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: ts,
                    });
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, bind_group, &[]);
                    if let Some(params) = immediate {
                        pass.set_immediates(0, bytemuck::cast_slice(params));
                    }
                    pass.dispatch_workgroups(dispatch.x, dispatch.y, 1);
                }

                if let Some(bg) = local_bg.as_ref() {
                    let local_ts = ts_writes.get(dispatch_idx).and_then(|t| t.clone());
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: local_ts,
                    });
                    pass.set_pipeline(&self.local_pipeline);
                    pass.set_bind_group(0, bg, &[]);
                    pass.dispatch_workgroups(self.local_dispatch.x, self.local_dispatch.y, 1);
                }
            }
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
