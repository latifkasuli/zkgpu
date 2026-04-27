# `zkgpu-wgpu` speed opportunities

This note records the current judgment about where more speed is likely to come from inside `zkgpu-wgpu`, based on the code that is already in-tree and the current `wgpu` v29 surface.

## Thesis

The right model for `zkgpu-wgpu` remains:

```text
portable WGSL baseline
+ capability / family planner
+ measured native wgpu features
+ no CUDA inside this crate
```

That philosophy already shows up in the current code:

- capability and family detection live in [`crates/zkgpu-wgpu/src/caps/profile.rs`](../../crates/zkgpu-wgpu/src/caps/profile.rs) and drive planner policy in [`crates/zkgpu-wgpu/src/ntt/planner/policy.rs`](../../crates/zkgpu-wgpu/src/ntt/planner/policy.rs)
- mobile tail heuristics are explicit and data-backed in [`crates/zkgpu-wgpu/src/ntt/planner/tail_policy.rs`](../../crates/zkgpu-wgpu/src/ntt/planner/tail_policy.rs)
- 2D dispatch folding is already used in [`crates/zkgpu-wgpu/src/ntt/stockham/encode.rs`](../../crates/zkgpu-wgpu/src/ntt/stockham/encode.rs)
- pipeline / shader / BGL reuse already exists in [`crates/zkgpu-wgpu/src/pipeline_registry.rs`](../../crates/zkgpu-wgpu/src/pipeline_registry.rs)
- the shared mixed-height Poseidon2 DAG engine already has a same-height fast path in [`crates/zkgpu-wgpu/src/poseidon2/merkle_commit_dag.rs`](../../crates/zkgpu-wgpu/src/poseidon2/merkle_commit_dag.rs)

The next good work is the kind that fits that structure cleanly. The main trap now would be skipping straight to backend-specific shader complexity before exhausting the high-leverage `wgpu`-native options.

## Current baseline

`zkgpu-wgpu` is not missing the obvious first-order performance work anymore.

What is already in good shape:

- Per-family NTT policy. Apple and browser stay Stockham; NVIDIA switches to four-step at `log_n >= 21`; AMD / Intel at `log_n >= 22`; Android Adreno at `log_n >= 18`.
- Mobile tail policy. Adreno / Mali / Xclipse switch to `GlobalOnlyR4` once a tail exists, avoiding the local strided-gather collapse.
- 2D dispatch folding. Large NTTs no longer hit WebGPU's per-dimension dispatch ceiling.
- Pipeline reuse. Duplicate plans do not pay duplicate shader compilation.
- Same-height MMCS fast path. Same-height commits now avoid the general mixed-height DAG round-trip pattern.

That means the remaining wins are not "turn on the obvious thing" work. They are targeted planner and execution-path improvements.

## Scope for this phase

This document lists more candidates than should be worked on in one go. The intended cadence is:

- **Do now (this phase):** items in the *Do now* section below — currently five items, scoped to ~2-3 weeks total. Items #1-2 are independently publishable; items #3-5 cluster on the same encoder-side hot paths and should land as one batch. After items #1-2 are measured and merged, a `v0.2` perf update note can ship alongside the existing `docs/two-consumers.md` headline; items #3-5 can land in the same window or as a `v0.3` follow-on.
- **Conditional / future:** the items in *Conditional / future* are real but should not be picked up speculatively. Each lists the concrete trigger that should justify scheduling it. Without that trigger, scheduling them ahead of demand is a higher cost than letting the published artifact compound external signal.

Reading this document as a fixed queue and grinding through every item would commit the project to a long internal optimization phase before any external feedback on the recently published two-consumers note. That is still the wrong cadence. Pick the *Do now* items, ship, see what bites, then revisit the *Conditional / future* tier with fresh information.

## Pipeline cache-key correctness — blocking precondition

Before any work that touches `PipelineCompilationOptions` **or** changes pipeline layouts (push-constant ranges, BGL contents), the `PipelineKey` in [`crates/zkgpu-wgpu/src/pipeline_registry.rs`](../../crates/zkgpu-wgpu/src/pipeline_registry.rs) must be expanded. Today the key is only `{source_ptr, entry_point, layout_key}` and `layout_key` is just a BGL label. Two failure modes:

- Two specializations of the same shader via `compilation_options` (e.g. `zero_initialize_workgroup_memory = true` vs `false`, or different override constants) collide in the cache and one is served silently for both call sites.
- Two pipelines with the same BGL but different push-constant ranges also collide, since `layout_key` does not capture push-constant ranges.

Either case is a correctness bug, not a performance bug. This is treated as a precondition for *Do now* items #2 (`PipelineCompilationOptions`), #3 (push constants), and #4 (multi-dispatch fold — incidentally changes BGL contents in some encode sites) below — not a parallel cleanup. **Land the cache-key expansion as the first commit of this phase**, before any of those items.

## Do now

### 1. GPU-resident mixed-height Poseidon2 injection

This is the biggest remaining MMCS-specific speed opportunity, and it has the clearest cost model.

The same-height fast path already fixes the obvious degenerate case. The general mixed-height path still does host interleave work inside [`crates/zkgpu-wgpu/src/poseidon2/merkle_commit_dag.rs`](../../crates/zkgpu-wgpu/src/poseidon2/merkle_commit_dag.rs):

```text
compress on GPU
download temp digests
hash injected group
host interleave
upload interleaved rows
compress again
```

That is fine for correctness and for getting the shared backend landed, but it is still the main place where mixed-height commits pay host-device traffic that same-height commits no longer pay. The same-height fast path proved this style of optimization recovers the ratio cleanly; mixed-height is the same playbook applied to the remaining hot path.

Why it is the highest-ROI next item:

- clear cost model (`O(log h_max)` PCIe round-trips per commit at injection levels, exactly the pattern the same-height fast path eliminated)
- direct OpenVM benefit on the headline mixed-height shape; indirect Plonky3 benefit on any future mixed-height workload
- matches the real `compress_and_inject` semantics OpenVM documents publicly
- result is publishable on its own as a `v0.2` perf row

Source: [OpenVM Poseidon2 mixed-height spec](https://github.com/openvm-org/openvm/blob/main/extensions/native/circuit/src/poseidon2/README.md)

Estimated effort: 3-5 days end-to-end including parity validation across both adapters and benches on at least one NVIDIA host.

### 2. `PipelineCompilationOptions` in `PipelineRegistry`, narrow scope

Today every compute pipeline is created with `compilation_options: Default::default()` in [`crates/zkgpu-wgpu/src/pipeline_registry.rs`](../../crates/zkgpu-wgpu/src/pipeline_registry.rs).

`wgpu` v29 exposes two levers:

- pipeline override constants
- `zero_initialize_workgroup_memory`

Source: [wgpu `PipelineCompilationOptions`](https://wgpu.rs/doc/wgpu/struct.PipelineCompilationOptions.html)

This phase's scope should be deliberately narrow:

- **First, expand the pipeline cache key** (see the precondition section above). This is permanent infrastructure either way; do it before any specialization opt-in to avoid silent cache-collision bugs.
- **Then, opt into `zero_initialize_workgroup_memory = false` on one kernel** where full workgroup-memory initialization is easy to prove safe — probably the W16 or W24 leaf sponge, since those are the most-audited Poseidon2 kernels in the tree.
- **Do not** start on override constants in this phase. Override constants are mostly setup for *Conditional / future* item #2 (workgroup-size tuning); without that follow-on, override constants are foundation without payoff. Land them when (or if) workgroup-size tuning becomes scheduled.

This stays inside the existing architecture, ships a measurable native-backend speedup, and leaves the foundation in place for a future workgroup-size experiment without committing to it now.

Estimated effort: 2-3 days for the cache-key expansion plus one-kernel zero-init opt-in plus parity / regression validation.

> **Cluster note for items #3-5.** The next three items all attack the same category — **encoder-side per-call CPU overhead** — and all touch the same source files (`crates/zkgpu-wgpu/src/ntt/stockham/encode.rs`, `crates/zkgpu-wgpu/src/ntt/four_step/encode.rs`, the Poseidon2 leaf/compress encode paths). They should be planned as one batch, even if each item is described separately below. The combined refactor lets each piece pay off without incremental rework.
>
> Why this category matters even though current headline numbers (15.07× single-matrix `fri_commit @ log_h=18` on 5090) live in kernel time: the encoder cost dominates **small `log_n` on mobile / browser**, **long prover loops with many small dispatches** (FRI fold rounds with `num_queries=40` are exactly this shape), and **integrated GPUs / browsers** where every Rust↔driver call has marshaling cost. zkgpu's existing benches don't cover small-`log_n` mobile shapes prominently, so this category is currently under-measured rather than known-unimportant.

### 3. Push constants for small per-dispatch params

Today every NTT and Poseidon2 stage allocates a small `param_buffer` (e.g. `(stage_idx, log_stride, n)` for R4 stages) and binds it as a uniform/storage buffer at binding 3 in the bind group. See `crates/zkgpu-wgpu/src/ntt/stockham/encode.rs` lines 41-47 (R4), 101-104 (R2), 156-159 (local), and equivalent sites in the Poseidon2 plans.

`wgpu::Features::PUSH_CONSTANTS` is a native-only feature flag (Vulkan / Metal / DX12 / GL — not browser). It exposes a register-sized parameter block (typical limit 128 bytes) that flows through the command stream as direct register writes rather than via a buffer-bound binding.

Concrete benefits per dispatch:

- One fewer `wgpu::Buffer` allocated at plan-build time.
- One fewer bind-group entry (the param binding goes away), reducing BGL fingerprint size.
- No per-stage uniform-buffer-update cost.

Browser builds keep the current uniform-buffer path; native builds switch to push constants. The capability gate matches the existing pattern used for `MAPPABLE_PRIMARY_BUFFERS` in `crates/zkgpu-wgpu/src/caps/profile.rs`.

Implementation surface:

- Detect / request `Features::PUSH_CONSTANTS` in `caps/profile.rs`.
- Add a parallel WGSL stage variant (or `#[cfg]`-gated WGSL string) that declares the params via `var<push_constant>` instead of `@binding(3) var<uniform>`.
- Update the encode sites to call `pass.set_push_constants(...)` instead of binding the param buffer.
- Two pipeline variants per kernel (push-constant native + uniform-buffer browser), both routed through the `PipelineRegistry` with the expanded cache key.

Estimated effort: 2 days, including one bench rerun on a native host to confirm the encoder-time win.

Source: [wgpu Features (PUSH_CONSTANTS)](https://wgpu.rs/doc/wgpu/struct.Features.html), [wgpu-py backends documentation](https://wgpu-py.readthedocs.io/en/latest/backends.html), [Vulkan push constants tutorial](https://kylemayes.github.io/vulkanalia/dynamic/push_constants.html)

### 4. Multi-dispatch per compute pass

Today each NTT call begins **4-5+ separate compute passes** — one per R4 stage, one per R2 stage, one for the local-fused kernel, one for the scale dispatch. See the four `encoder.begin_compute_pass(...)` blocks in `crates/zkgpu-wgpu/src/ntt/stockham/encode.rs` (lines 58, 115, 170, 219). Same pattern in `four_step/encode.rs` and the Poseidon2 plans.

The standard wgpu pattern is **one compute pass with multiple `dispatch_workgroups` calls inside it**, swapping bind groups via `set_bind_group` between dispatches. Per [Toji's bind-group best practices](https://toji.dev/webgpu-best-practices/bind-groups.html):

> "If you need to apply multiple compute passes to the same data, chain them together in a single command encoder submission rather than reading back and re-uploading between passes."

What this saves: per-pass driver overhead (each `begin_compute_pass`/`end_compute_pass` boundary is non-trivial on Vulkan and Metal). What it does **not** save: memory barriers between dispatches that read+write the same buffer (the GPU correctness invariants still need them). See [wgpu issue #5766](https://github.com/gfx-rs/wgpu/issues/5766) for confirmation that the barriers stay even when passes fold.

So the gain is purely encoder/driver-side overhead reduction, not GPU-time reduction. For long prover loops with many small commits, that overhead amortizes badly today.

Implementation surface:

- In each `encode.rs` site, replace the multiple `{ let mut pass = encoder.begin_compute_pass(...); ... }` blocks with one outer `let mut pass = encoder.begin_compute_pass(...)`, then issue the R4 / R2 / local / scale dispatches inside it with `set_bind_group` swaps.
- Timestamp queries currently attach to per-pass `timestamp_writes`; when folding into one pass, switch to per-dispatch `write_timestamp` calls inside the pass (requires `TIMESTAMP_QUERY_INSIDE_PASSES`, which we already detect in `caps/profile.rs:105`).
- Stockham first, since it's the hot path; then four-step; then Poseidon2 plans.

Estimated effort: 1-2 days. Pure encoder refactor, no shader changes.

### 5. Bind-group reuse — pre-built at plan time

Today the encode functions call `device.create_bind_group(...)` **inside the dispatch loop**, on every NTT / Poseidon2 call. See `stockham/encode.rs:28`, `:85`, `:140`, `:203` and the equivalent sites in `four_step/encode.rs` and the Poseidon2 plans. The bind-group entries are mostly stable across calls — `twiddle_buffer`, `param_buffer`, `twiddle_prime_buffer` are owned by the plan and never change between calls; only `src_buf` and `dst_buf` ping-pong, and even that is between two known buffers.

The standard wgpu pattern (per [Toji's bind-group best practices](https://toji.dev/webgpu-best-practices/bind-groups.html) and the [WebGPU optimization guide](https://webgpufundamentals.org/webgpu/lessons/webgpu-optimization.html)):

> "If the buffer bindings do not change between dispatches, reuse the bind group object rather than recreating it every iteration."
> "Recreating pipelines inside a tight loop is one of the most common performance mistakes seen in early wgpu compute code. This extends naturally to bind groups."

Implementation surface:

- For each NTT stage, pre-build **two** bind groups at plan-build time — one for the `buf → scratch` ping-pong direction, one for the `scratch → buf` direction. Pick the right one per dispatch based on the parity of `dispatch_idx % 2`, exactly the same parity check `encode.rs` already uses.
- Same pattern for the Poseidon2 leaf and compress plans (one bind group per fixed buffer pair, picked at encode time).
- Pre-built bind groups own an `Arc<wgpu::BindGroup>` clone; encode-time call collapses to `pass.set_bind_group(0, &self.bind_groups[parity], &[])`.

Estimated effort: 1 day. Mechanical refactor.

This was previously listed as Conditional / future #3 with a "needs a microbench" gate. The external research is unambiguous enough that the gate is not worth the wait — the change is cheap, the pattern is canonical, and the cost on every NTT call is real and unmeasured-because-not-instrumented. Promote and land.

## Conditional / future

These are real candidates, not a polite "no". Each one needs a concrete trigger before it should be scheduled — listed inline. Note: bind-group reuse (formerly conditional #3) was promoted to *Do now* item #5 once external research confirmed the pattern is canonical and the cost is real on every call.

### 6. Planner-selected workgroup size via shader overrides

Bigger refactor than it first appears.

[`crates/zkgpu-wgpu/src/ntt/planner/constants.rs`](../../crates/zkgpu-wgpu/src/ntt/planner/constants.rs) hardcodes `WORKGROUP_SIZE = 256`, which feeds `BLOCK_SIZE = 4 * WORKGROUP_SIZE` and `LOG_BLOCK = 10`. Those in turn affect planner math, tail existence, twiddle generation, local dispatch sizing, and scratch sizing. [`crates/zkgpu-wgpu/src/caps/profile.rs`](../../crates/zkgpu-wgpu/src/caps/profile.rs) also currently requests `max_compute_invocations_per_workgroup = 256`, so anything above that is blocked at device-limit-request time before shader specialization even matters.

Expected upside: 5-15% class improvement, not a structural 2x.

**Trigger:** a measured per-platform cliff that 128-thread or 64-thread workgroups would unblock — e.g. an Adreno or Mali bench showing occupancy / register-pressure data that argues for a smaller workgroup. Without that measurement, the planner refactor cost is too high relative to the speculative gain.

### 7. BN254 / MSM portable WGSL baseline

Strategically important, but this is a new workload family — not a `zkgpu-wgpu` cleanup. The right pattern is the one already used for Goldilocks: portable WGSL first, optional native fast paths later if the measurements justify them.

**Trigger:** demand for an MSM-using consumer adapter (e.g. a Halo2 or Groth16 acceleration ask), or a strategic decision to expand into pairing-based stacks. Should not be scheduled as "next zkgpu-wgpu speed work."

### 8. Subgroup variants for specific kernels

`CapabilityProfile` already detects subgroup support and requests `SUBGROUP` when available. `wgpu` also exposes `SUBGROUP_BARRIER` and native-only `SHADER_INT64`.

Source: [wgpu `Features`](https://wgpu.rs/doc/wgpu/struct.Features.html)

The measured BabyBear NTT work already closed most of the practical CUDA gap without subgroup specialization (RTX 4090 Week-8 scale-up data).

**Trigger:** a specific hot kernel where profiling shows warp/subgroup-shuffle would unblock progress — likely candidates are MSM bucket reductions or BN254 limb arithmetic, neither of which exist yet. Not a candidate for blanket rewrite of existing NTT kernels.

### 9. SPIR-V / Metal passthrough

Last.

**Trigger:** items 1-5 (the *Do now* tier) are landed and measured, items 6-8 (the remaining Conditional / future tier) have been triggered and landed, and there is still meaningful performance on the table that the WGSL-via-`wgpu` path cannot reach. Native shader ingestion is the highest-complexity option in this list and should not be the first response to any speed gap.

## Constraints to keep

These should remain design constraints, not just current preferences:

- No CUDA inside `zkgpu-wgpu`
- Browser / WebGPU remains the portability floor
- Native-only features are opt-in and capability-gated
- Planner choices stay measured and per-family, not theoretical
- Shared backend wins should benefit both consumer adapters automatically

## Practical stop/go rule

For **this phase** (the *Do now* tier, ~2-3 weeks total):

Land in the following order:

1. **Cache-key expansion first** (~half-day). The blocking precondition. Reflect compilation options + push-constant ranges in the `PipelineKey`. No behavior change yet, but unblocks items #2-4 safely.
2. **GPU-resident mixed-height injection** (~3-5 days). Algorithm-level, biggest standalone gain, independently publishable. Run targeted benches on at least one NVIDIA host post-fix.
3. **`PipelineCompilationOptions` zero-init opt-in on one kernel** (~1-2 days). Foundation now safe to opt into; narrow scope.

→ **First publish gate:** a small `v0.2` perf update note here, alongside the existing `docs/two-consumers.md` headline. Items #1 and #2 above are both bench-publishable; the encoder cluster (next) is bench-publishable as a follow-on.

4. **Encoder-side cluster** (~3-5 days, items #3 push constants + #4 multi-dispatch fold + #5 bind-group reuse). All three touch the same encode.rs sites; plan as one batch refactor. Run a small encoder-time microbench at small `log_n` (e.g. log_n ∈ {12, 14, 16}) to capture the per-call CPU overhead reduction — this is where the wins live, not at the existing log_h=18 headline.

→ **Second publish gate:** `v0.3` perf update note, focused on small-`log_n` and prover-loop scenarios where the cluster shows up.

For **future phases** (the *Conditional / future* tier):

- Schedule each item only when its listed trigger has fired.
- Reorder if a trigger fires for a lower-listed item before a higher-listed one — this list is by current likelihood of being triggered, not a fixed queue.

**Stop rules within this phase:**

- After the cache-key expansion (step 1), if any landed parity test regresses, stop and investigate before proceeding. The cache key is correctness-critical.
- After mixed-height injection (step 2), if NVIDIA `commit_open_40q` doesn't show a clear ratio improvement on the mixed-height shape, stop and check the cost model — something else is dominating.
- After the encoder cluster (step 4), if small-`log_n` microbench numbers don't show measurable encoder-time reduction, the cluster's impact may have been overestimated. Don't keep grinding. Publish what's true and move on.

Constraint that overrides everything: preserve the strongest current property of `zkgpu-wgpu` — one portable WGSL codebase that gets faster by making better planning decisions, not by fragmenting into per-backend products. Native-only items (#3 push constants, plus the existing #2 zero-init opt-in) must be capability-gated, not unconditional.
