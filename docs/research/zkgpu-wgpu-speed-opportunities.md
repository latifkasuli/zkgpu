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

- **Do now (this phase):** items in the *Do now* section below — currently two items, scoped to ~1-2 weeks total. Land them, then publish a `v0.2` perf update note alongside the existing `docs/two-consumers.md` headline.
- **Conditional / future:** the items in *Conditional / future* are real but should not be picked up speculatively. Each lists the concrete trigger that should justify scheduling it. Without that trigger, scheduling them ahead of demand is a higher cost than letting the published artifact compound external signal.

Reading this document as a 7-item queue would commit the project to ~2-3 weeks of internal optimization before any external feedback on the recently published two-consumers note. That is the wrong cadence for this phase. Pick the *Do now* items, ship, see what bites, then revisit the *Conditional / future* tier with fresh information.

## Pipeline cache-key correctness — blocking precondition

Before any work that touches `PipelineCompilationOptions`, the `PipelineKey` in [`crates/zkgpu-wgpu/src/pipeline_registry.rs`](../../crates/zkgpu-wgpu/src/pipeline_registry.rs) must be expanded to include compilation options. Today the key is only `{source_ptr, entry_point, layout_key}`. Two specializations of the same shader (e.g. `zero_initialize_workgroup_memory = true` vs `false`, or different override constants) would collide in the cache and one would be served silently for both call sites — a correctness bug, not a performance bug. This is treated as a precondition for *Do now* item #2 below, not a parallel cleanup.

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

## Conditional / future

These are real candidates, not a polite "no". Each one needs a concrete trigger before it should be scheduled — listed inline.

### 3. Bind group reuse / execution objects

The code already caches pipelines, shader modules, and bind-group layouts, but hot encode paths still create bind groups per call:

- [`crates/zkgpu-wgpu/src/ntt/stockham/encode.rs`](../../crates/zkgpu-wgpu/src/ntt/stockham/encode.rs)
- [`crates/zkgpu-wgpu/src/ntt/four_step/encode.rs`](../../crates/zkgpu-wgpu/src/ntt/four_step/encode.rs)
- Poseidon2 leaf / compress plans in `crates/zkgpu-wgpu/src/poseidon2/*`

The shape would be an "execution object" or "prepared invocation" API for repeated prover loops with stable buffer identities.

**Trigger:** a microbench (or a real prover-loop profile) showing bind-group creation is a meaningful fraction of small-NTT or per-Poseidon2-call wall time on at least one platform. Expected to matter more on browser / mobile / integrated than on large desktop kernels — but currently no measurement quantifies the share. Without that share, ranking it is guesswork.

### 4. Planner-selected workgroup size via shader overrides

Bigger refactor than it first appears.

[`crates/zkgpu-wgpu/src/ntt/planner/constants.rs`](../../crates/zkgpu-wgpu/src/ntt/planner/constants.rs) hardcodes `WORKGROUP_SIZE = 256`, which feeds `BLOCK_SIZE = 4 * WORKGROUP_SIZE` and `LOG_BLOCK = 10`. Those in turn affect planner math, tail existence, twiddle generation, local dispatch sizing, and scratch sizing. [`crates/zkgpu-wgpu/src/caps/profile.rs`](../../crates/zkgpu-wgpu/src/caps/profile.rs) also currently requests `max_compute_invocations_per_workgroup = 256`, so anything above that is blocked at device-limit-request time before shader specialization even matters.

Expected upside: 5-15% class improvement, not a structural 2x.

**Trigger:** a measured per-platform cliff that 128-thread or 64-thread workgroups would unblock — e.g. an Adreno or Mali bench showing occupancy / register-pressure data that argues for a smaller workgroup. Without that measurement, the planner refactor cost is too high relative to the speculative gain.

### 5. BN254 / MSM portable WGSL baseline

Strategically important, but this is a new workload family — not a `zkgpu-wgpu` cleanup. The right pattern is the one already used for Goldilocks: portable WGSL first, optional native fast paths later if the measurements justify them.

**Trigger:** demand for an MSM-using consumer adapter (e.g. a Halo2 or Groth16 acceleration ask), or a strategic decision to expand into pairing-based stacks. Should not be scheduled as "next zkgpu-wgpu speed work."

### 6. Subgroup variants for specific kernels

`CapabilityProfile` already detects subgroup support and requests `SUBGROUP` when available. `wgpu` also exposes `SUBGROUP_BARRIER` and native-only `SHADER_INT64`.

Source: [wgpu `Features`](https://wgpu.rs/doc/wgpu/struct.Features.html)

The measured BabyBear NTT work already closed most of the practical CUDA gap without subgroup specialization (RTX 4090 Week-8 scale-up data).

**Trigger:** a specific hot kernel where profiling shows warp/subgroup-shuffle would unblock progress — likely candidates are MSM bucket reductions or BN254 limb arithmetic, neither of which exist yet. Not a candidate for blanket rewrite of existing NTT kernels.

### 7. SPIR-V / Metal passthrough

Last.

**Trigger:** items 1-2 above are landed and measured, items 3-4 have been triggered and landed, and there is still meaningful performance on the table that the WGSL-via-`wgpu` path cannot reach. Native shader ingestion is the highest-complexity option in this list and should not be the first response to any speed gap.

## Constraints to keep

These should remain design constraints, not just current preferences:

- No CUDA inside `zkgpu-wgpu`
- Browser / WebGPU remains the portability floor
- Native-only features are opt-in and capability-gated
- Planner choices stay measured and per-family, not theoretical
- Shared backend wins should benefit both consumer adapters automatically

## Practical stop/go rule

For **this phase** (the *Do now* tier):

- Time-box to ~1-2 weeks total across both items.
- Land item 1 (mixed-height GPU injection) first because the cost model is concrete and the result is independently publishable as a `v0.2` perf row.
- Land item 2 (`PipelineCompilationOptions` with cache-key expansion + one-kernel zero-init opt-in) second. Stop after the one-kernel opt-in unless a measured signal from item 1's bench rerun says more is justified.
- Publish a small `v0.2` perf update note alongside the published two-consumers headline. Do not bundle a multi-week internal optimization phase with no external touchpoint.

For **future phases** (the *Conditional / future* tier):

- Schedule each item only when its listed trigger has fired.
- Reorder if a trigger fires for a lower-listed item before a higher-listed one — this list is by current likelihood of being triggered, not a fixed queue.

Constraint that overrides everything: preserve the strongest current property of `zkgpu-wgpu` — one portable WGSL codebase that gets faster by making better planning decisions, not by fragmenting into per-backend products.
