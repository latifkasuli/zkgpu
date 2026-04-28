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

- **Do now (this phase):** items in the *Do now* section below — currently eight items, scoped to ~3-4 weeks total. The list grew when reviewer-led research against wgpu v29 docs surfaced corrections to two prior items (push constants was actually `Features::IMMEDIATES` in v29; the zero-init kernel target was wrong) and three additional candidates (Poseidon constants migration, trusted shader modules, cross-operation command batching). To avoid re-committing to the whole budget, the *Do now* tier is split into three publish gates and you can stop at any of them.
- **Conditional / future:** the items in *Conditional / future* are real but should not be picked up speculatively. Each lists the concrete trigger that should justify scheduling it. Without that trigger, scheduling them ahead of demand is a higher cost than letting the published artifact compound external signal.

The three publish gates inside *Do now*:

- **Gate 1 (`v0.2`, ~1-1.5 weeks):** items #1-3 — algorithmic mixed-height fix + WGSL/pipeline-side specialization (zero-init + immediates). Bench and publish before continuing.
- **Gate 2 (`v0.3`, +1 week):** items #4-6 — encoder-side cluster (multi-dispatch fold + bind-group reuse + constants migration). All touch the same encode.rs hot paths; land as one batch.
- **Gate 3 (`v0.4`, +1-1.5 weeks):** items #7-8 — advanced native paths (trusted modules + cross-op batching). Bigger architectural moves; only worth it if items #1-6 still leave meaningful gap.

You can stop at gate 1 if external signal arrives, or stop at gate 2 if the encoder-cluster wins are smaller than expected. Don't grind through all 8 by default.

## Pipeline cache-key correctness — blocking precondition

Before any work in the *Do now* tier below lands, the `PipelineKey` in [`crates/zkgpu-wgpu/src/pipeline_registry.rs`](../../crates/zkgpu-wgpu/src/pipeline_registry.rs) must be expanded. Today the key is only `{source_ptr, entry_point, layout_key}` and `layout_key` is just a BGL label. Several upcoming items will silently produce wrong-pipeline-served-from-cache bugs against this shape:

- Two specializations of the same shader via `compilation_options` (e.g. `zero_initialize_workgroup_memory = true` vs `false`, or different override constants) collide.
- Two pipelines with the same BGL but different `immediate_size` values (item #3) also collide.
- Two pipelines created with `create_shader_module_trusted` vs `create_shader_module` from the same source (item #7) silently share a module entry under the current `source_ptr` key.
- Generated WGSL variants (item #6, if pursued) need content-hash identity, not just source-pointer identity, since they're built at runtime.

The expanded `PipelineKey` should include, at minimum:

- shader content hash (or content-stable identity, replacing the current `source_ptr`-only fingerprint)
- BGL structural fingerprint (not just label)
- `immediate_size`
- `PipelineCompilationOptions` snapshot (zero-init flag + override-constants map)
- runtime-check mode (safe / trusted / passthrough)
- relevant capability/feature bits that affect codegen (e.g. `IMMEDIATES`, `SUBGROUP`)

Each failure mode above is a correctness bug, not a performance bug. **Land the cache-key expansion as the first commit of this phase**, before any other *Do now* item. The current narrow key is fine for the production path that ships today; it is the next round's items that surface the gap.

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

### 2. `PipelineCompilationOptions` zero-init opt-in — local Stockham + transpose tile kernels

Today every compute pipeline is created with `compilation_options: Default::default()` in [`crates/zkgpu-wgpu/src/pipeline_registry.rs`](../../crates/zkgpu-wgpu/src/pipeline_registry.rs).

`wgpu` v29 exposes two levers:

- pipeline override constants
- `zero_initialize_workgroup_memory`

Source: [wgpu `PipelineCompilationOptions`](https://wgpu.rs/doc/wgpu/struct.PipelineCompilationOptions.html)

This phase's scope is deliberately narrow:

- **The cache-key expansion** (precondition section above) lands first.
- **Opt into `zero_initialize_workgroup_memory = false`** on the kernels that actually allocate workgroup memory. The previous version of this doc wrongly listed the W16/W24 Poseidon leaf sponges as candidates — neither has `var<workgroup>` declarations, so disabling zero-init on them is a no-op. The actual targets are the kernels with real workgroup arrays:

  | Kernel | `var<workgroup>` declared |
  |---|---|
  | `babybear_stockham_local_r4.wgsl` | `shmem_a/b: array<u32, 1056>` |
  | `babybear_stockham_local.wgsl` (R2) | `shmem_a/b: array<u32, 528>` |
  | `babybear_fourstep_transpose_tiled32.wgsl` | `tile: array<u32, 1056>` |
  | `babybear_fourstep_transpose.wgsl` | `tile: array<u32, 272>` |

  All four kernels write every reachable workgroup-memory slot before any read in their butterfly / tile loops, which is the soundness condition for disabling zero-init. Audit the WGSL once per kernel; opt in as a separate compilation-options profile per kernel.

- **Do not** start on override constants in this phase. Override constants are mostly setup for *Conditional / future* item #9 (workgroup-size tuning); without that follow-on, override constants are foundation without payoff.

Estimated effort: 2-3 days for the cache-key expansion + per-kernel zero-init audit + opt-in + parity / regression validation across all four kernels.

> **Encoder-side category note (items #3 + #4-6).** Items #3 through #6 all reduce **encoder-side per-call CPU overhead**, but they split across two publish gates by scope. Item #3 (immediates) is plan-build + per-dispatch and ships in Gate 1 alongside #1 and #2. Items #4-6 (multi-dispatch fold + bind-group reuse + constants migration) all touch the same encode.rs hot paths (`crates/zkgpu-wgpu/src/ntt/stockham/encode.rs`, `crates/zkgpu-wgpu/src/ntt/four_step/encode.rs`, the Poseidon2 leaf/compress encode paths) and should land as one batch in Gate 2.
>
> Why this category matters even though current headline numbers (15.07× single-matrix `fri_commit @ log_h=18` on 5090) live in kernel time: the encoder cost dominates **small `log_n` on mobile / browser**, **long prover loops with many small dispatches** (FRI fold rounds with `num_queries=40` are exactly this shape), and **integrated GPUs / browsers** where every Rust↔driver call has marshaling cost. zkgpu's existing benches don't cover small-`log_n` mobile shapes prominently, so this category is currently under-measured rather than known-unimportant. Item #8 (cross-op command batching) extends this category one level higher — across primitives, not just within an NTT — and is the largest single CPU-side win candidate for mobile / browser, but lands as Gate 3 because of its bigger API-surface change.

### 3. Immediates for small per-dispatch params (`Features::IMMEDIATES`)

Today every NTT and Poseidon2 stage allocates a small `param_buffer` (e.g. `(stage_idx, log_stride, n)` for R4 stages) and binds it as a uniform buffer at binding 3 in the bind group. See `crates/zkgpu-wgpu/src/ntt/stockham/encode.rs` lines 41-47 (R4), 101-104 (R2), 156-159 (local), and the equivalent param-uniform sites in `crates/zkgpu-wgpu/src/poseidon2/merkle_leaf_w16.rs:143` and `crates/zkgpu-wgpu/src/poseidon2/merkle_compress.rs:120`.

`wgpu::Features::IMMEDIATES` is a wgpu v29 feature classified as **"Web and native"** — DX12, Vulkan, Metal, OpenGL (emulated as uniforms), and WebGPU. It exposes a register-sized parameter block (`PipelineLayoutDescriptor::immediate_size`, WGSL `var<immediate>`, `ComputePass::set_immediates`) that flows through the command stream as direct register writes rather than via a buffer-bound binding.

> **Naming correction:** an earlier version of this doc called this "push constants" with `Features::PUSH_CONSTANTS`. That feature flag does not exist in wgpu v29. The API was redesigned around the IMMEDIATES name (matching WebGPU spec terminology). The semantics are nearly identical to the older push-constants surface from a caller perspective.

Concrete benefits per dispatch:

- One fewer `wgpu::Buffer` allocated at plan-build time (the params uniform).
- One fewer bind-group entry (the param binding goes away), reducing BGL fingerprint size.
- No per-stage uniform-buffer-update cost.

Portability note (correcting the previous version of this doc): IMMEDIATES is **not native-only**. The wgpu doc lists WebGPU as supported, with OpenGL explicitly emulated via uniforms. Browser perf may not move much (likely emulated), but the call site stays unified and there is no need for a native vs browser code split — just one path that uses immediates everywhere they're available.

Implementation surface:

- Detect / request `Features::IMMEDIATES` in `caps/profile.rs`.
- For each param-bearing kernel, declare params via `var<immediate>` in WGSL (or generate the variant per kernel).
- Update encode sites to call `pass.set_immediates(...)` instead of binding the param buffer.
- Drop the param-uniform allocation from the plan builders.
- Pipeline cache key must include `immediate_size` (see precondition).

Estimated effort: 2-3 days, including a bench rerun on at least one native host to confirm the encoder-time win.

Source: [wgpu `Features` (IMMEDIATES)](https://wgpu.rs/doc/wgpu/struct.Features.html), [wgpu `ComputePass::set_immediates`](https://wgpu.rs/doc/wgpu/struct.ComputePass.html)

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

### 6. Poseidon2 constants migration: storage → uniform / generated module constants

> **Status (2026-04-28, pilot landed + benched).** The safer **Uniform** path is implemented as a pilot on the standalone `WgpuBabyBearPoseidon2Plan` only, opt-in via `Poseidon2ConstantsSource::Uniform` (default stays `Storage`). A/B benched on three hosts:
>
> | Host | Δ at batch=64 | Δ at 4096 | Δ at 65536 |
> |---|---:|---:|---:|
> | M4 Pro / Metal | +3.9% | +2.6% | +2.2% |
> | RTX 4090 / Vulkan | +0.5% | −0.9% | +0.4% |
> | RTX 5090 / Vulkan | **−4.1%** | **−5.0%** | −2.1% |
>
> Metal regresses by 2-4%, the 4090 is null, only the 5090 shows a clean win. **The Metal-regression gate criterion fails, so the production kernels (`merkle_leaf`, `merkle_leaf_w16`, `merkle_compress`, `plonky3_w16`, `plonky3_w24`) keep the Storage path.** The Uniform pilot stays accessible behind the explicit constructor for callers who want to opt in on 5090-class hardware. The faster generated-module-constants variant remains unimplemented; if it's pursued later, it's a different mechanism (WGSL `const` array inlined at plan-build) and gets its own A/B against this baseline. Full verdict + reproducer: `research/benchmarks/poseidon2-uniform-pilot-2026-04-28/verdict.md` (gitignored).

Today the Poseidon2 round constants are uploaded as **storage buffers** and read via `var<storage, read>` at shader binding 2. See:

```text
babybear_poseidon2_merkle_leaf_w16.wgsl:158
  @group(0) @binding(2) var<storage, read> poseidon_constants: array<u32>;
babybear_poseidon2_merkle_compress.wgsl:158
  @group(0) @binding(2) var<storage, read> poseidon_constants: array<u32>;
```

These constants are immutable per plan but currently use the storage path — costing a binding slot, one buffer alloc per plan, and storage-cache pressure on every kernel invocation. Two experiments are worth running side-by-side:

- **Safer portable path: constants in a uniform buffer.** Move binding 2 from `var<storage, read>` to `var<uniform>` (subject to the WebGPU 64 KB uniform-buffer ceiling — for BabyBear Plonky3 W24 the constants are well under that; W16 is even smaller). Same plumbing, different binding type. Browser-safe.
- **Faster specialized path: generated WGSL with module-level constants.** At plan build time, generate the WGSL string with the round constants baked into a `const POSEIDON_CONSTANTS: array<u32, N> = array<u32, N>(...)` block. Drops the binding entirely; lets the WGSL compiler hoist the constants into registers / immediate operands; pairs naturally with item #3 (immediates) since both shrink the bind-group fingerprint. Native-friendly; works in browser too but at the cost of per-plan shader compilation (no shader-module reuse across plans with different constants).

Implementation surface:

- For each Poseidon2 plan (`merkle_leaf.rs`, `merkle_leaf_w16.rs`, `merkle_compress.rs`, the various NTT-side Poseidon2 plans), add a constants source enum: `Storage` (today), `Uniform` (safer experiment), `ModuleConstant` (faster experiment).
- Pipeline cache key must include the constants source variant (different shader content → different module identity).
- Bench both variants on at least one mobile (M4 Pro Metal) and one discrete (RTX 5090 Vulkan) host. Adopt whichever wins on each platform per capability/family.

Estimated effort: 2-3 days, dominated by the WGSL generation path and per-platform bench validation.

### 7. Trusted shader modules for audited native kernels

`wgpu::Device::create_shader_module_trusted` (`unsafe`) lets the caller customize which runtime checks are skipped via a `ShaderRuntimeChecks` parameter. Source: [wgpu `Device` docs](https://docs.rs/wgpu/latest/wgpu/struct.Device.html). Today every shader module in zkgpu-wgpu goes through the safe `create_shader_module` path (via `PipelineRegistry::get_or_create_module`).

For kernels with **planner-determined dispatch geometry and zero caller-supplied indices into storage buffers**, runtime bounds-check elision can move per-thread storage-buffer access to lower latency. The benefit shows up on storage-pressure-heavy kernels — exactly the NTT and Poseidon2 hot path on integrated and mobile GPUs.

Why this is gated tightly:

- **Unsafe.** Out-of-bounds storage access on a trusted module is undefined behavior at the GPU driver level (potentially: silent corruption, memory disclosure across processes, driver crash). Not "slow" or "wrong output" — actual UB.
- **Per-kernel audit required.** A kernel qualifies only if every storage-buffer access uses an index that is either a constant or a linear function of `global_invocation_id` / `workgroup_id` / planner-fed param values, with no user-supplied arrays-of-indices and no data-dependent indexing.
- **Native-only.** WebGPU has no equivalent surface; browser keeps the safe path.

Concretely, the kernels that pass this bar today (after audit) would be the same ones that benefit from item #2's zero-init opt-in: local Stockham R2/R4 and the four-step transpose tiles. Their dispatch shapes are entirely planner-determined.

Implementation surface:

- Extend `PipelineRegistry` to track per-module trust mode (must be in the cache key).
- Capability-gate at `caps/profile.rs` — only enable the trusted path on native backends.
- Per-kernel audit checklist + opt-in flag in the plan builder. Default remains the safe path; trusted is per-kernel opt-in by name.
- Comprehensive parity tests against the safe-mode shader output on the same input across all bench hosts before any opt-in lands.

Estimated effort: 3-5 days, dominated by the audit and parity validation, not the wgpu API surface change.

### 8. Cross-operation command batching — single encoder for multi-primitive sequences

Today every primitive submits and polls independently:

```text
crates/zkgpu-wgpu/src/ntt/stockham/mod.rs:104  // create encoder, dispatch NTT, submit
crates/zkgpu-wgpu/src/poseidon2/merkle_leaf_w16.rs:302  // separate encoder, dispatch leaf, submit
crates/zkgpu-wgpu/src/poseidon2/merkle_compress.rs:284  // separate encoder per compression level
```

For prover workloads — especially the FRI commit phase, where each commit calls leaf hash → compression chain × log_h levels → root readback — every `submit + poll(wait)` boundary is per-primitive driver overhead. On integrated / mobile GPUs and in browser, this overhead can dominate the kernel time itself for small `log_n`.

Item #4 above (multi-dispatch per pass) reduces the within-NTT pass count. This item goes one level higher: **let the caller compose a multi-primitive command stream into one encoder + one submit**.

The shape would be a lower-level `encode_*` API on each primitive:

```rust
impl WgpuPoseidon2MerkleLeafW16R8Plan {
    pub fn encode_hash_rows(
        &mut self,
        device: &WgpuDevice,
        encoder: &mut wgpu::CommandEncoder,
        compute_pass: &mut wgpu::ComputePass<'_>,  // shared pass!
        input: &WgpuBuffer<BabyBear>,
        output: &mut WgpuBuffer<BabyBear>,
        ...
    ) -> Result<(), ZkGpuError>;
}
```

The current `commit_*`-style methods that own their encoder become thin wrappers over the new `encode_*` methods, for backwards-compat. New consumers (and the mixed-height DAG engine after item #1) build their own encoder, compose primitives into it, and submit once.

Concrete impact target: a FRI commit at `log_h=18` today goes through ~20 separate submits (1 leaf hash + 18 compression levels + readbacks). Folding into one submit removes 19 driver-side `submit + poll` boundaries. On mobile / browser this is potentially the largest single CPU-side win in the doc.

Implementation surface:

- Add `encode_*` methods to each `WgpuPoseidon2*` plan and to the NTT plans.
- Update the mixed-height DAG engine (item #1's home) to use a single shared encoder for the leaf-hash + compression-chain.
- Update zkgpu-plonky3 and zkgpu-openvm adapters to optionally build cross-primitive encoders for their hot paths.
- Audit barrier semantics across primitive boundaries — the wgpu compute-pass barrier rules still apply, but they're cheaper than driver-level submit boundaries.

Estimated effort: 4-6 days. Largest single item in the *Do now* tier; only proceed after items #1-6 land and benches show whether per-primitive submit boundaries are actually a meaningful share of small-`log_n` time.

## Conditional / future

These are real candidates, not a polite "no". Each one needs a concrete trigger before it should be scheduled — listed inline. Note: bind-group reuse (formerly conditional #3) was promoted to *Do now* item #5 once external research confirmed the pattern is canonical and the cost is real on every call.

### 9. Planner-selected workgroup size via shader overrides

Bigger refactor than it first appears.

[`crates/zkgpu-wgpu/src/ntt/planner/constants.rs`](../../crates/zkgpu-wgpu/src/ntt/planner/constants.rs) hardcodes `WORKGROUP_SIZE = 256`, which feeds `BLOCK_SIZE = 4 * WORKGROUP_SIZE` and `LOG_BLOCK = 10`. Those in turn affect planner math, tail existence, twiddle generation, local dispatch sizing, and scratch sizing. [`crates/zkgpu-wgpu/src/caps/profile.rs`](../../crates/zkgpu-wgpu/src/caps/profile.rs) also currently requests `max_compute_invocations_per_workgroup = 256`, so anything above that is blocked at device-limit-request time before shader specialization even matters.

Expected upside: 5-15% class improvement, not a structural 2x.

**Trigger:** a measured per-platform cliff that 128-thread or 64-thread workgroups would unblock — e.g. an Adreno or Mali bench showing occupancy / register-pressure data that argues for a smaller workgroup. Without that measurement, the planner refactor cost is too high relative to the speculative gain.

### 10. BN254 / MSM portable WGSL baseline

Strategically important, but this is a new workload family — not a `zkgpu-wgpu` cleanup. The right pattern is the one already used for Goldilocks: portable WGSL first, optional native fast paths later if the measurements justify them.

**Trigger:** demand for an MSM-using consumer adapter (e.g. a Halo2 or Groth16 acceleration ask), or a strategic decision to expand into pairing-based stacks. Should not be scheduled as "next zkgpu-wgpu speed work."

### 11. Subgroup variants for specific kernels

`CapabilityProfile` already detects subgroup support and requests `SUBGROUP` when available. `wgpu` also exposes `SUBGROUP_BARRIER` and native-only `SHADER_INT64`.

Source: [wgpu `Features`](https://wgpu.rs/doc/wgpu/struct.Features.html)

The measured BabyBear NTT work already closed most of the practical CUDA gap without subgroup specialization (RTX 4090 Week-8 scale-up data).

**Trigger:** a specific hot kernel where profiling shows warp/subgroup-shuffle would unblock progress — likely candidates are MSM bucket reductions or BN254 limb arithmetic, neither of which exist yet. Not a candidate for blanket rewrite of existing NTT kernels.

### 12. SPIR-V / Metal passthrough

Last.

**Trigger:** items 1-8 (the *Do now* tier) are landed and measured, items 9-11 (the remaining Conditional / future tier) have been triggered and landed, and there is still meaningful performance on the table that the WGSL-via-`wgpu` path cannot reach. Native shader ingestion is the highest-complexity option in this list and should not be the first response to any speed gap.

## Constraints to keep

These should remain design constraints, not just current preferences:

- No CUDA inside `zkgpu-wgpu`
- Browser / WebGPU remains the portability floor
- Native-only features are opt-in and capability-gated
- Planner choices stay measured and per-family, not theoretical
- Shared backend wins should benefit both consumer adapters automatically

## Practical stop/go rule

For **this phase** (the *Do now* tier, ~3-4 weeks total split across three publish gates):

### Foundation commit (~half-day)

**Cache-key expansion** in `PipelineRegistry`. Reflect content hash + BGL fingerprint + `immediate_size` + `PipelineCompilationOptions` snapshot + runtime-check mode + relevant feature bits. No behavior change yet, but unblocks items #2-7 safely. **First commit of the phase.**

### Gate 1 (`v0.2`, ~1-1.5 weeks): WGSL/pipeline-side specialization

- Item #1: GPU-resident mixed-height Poseidon2 injection (~3-5 days). Algorithm-level, biggest standalone gain.
- Item #2: zero-init opt-in on local Stockham + transpose tile kernels (~2-3 days, includes per-kernel audit).
- Item #3: Immediates for small per-dispatch params (~2-3 days).

→ **Publish gate:** `v0.2` perf update note alongside `docs/two-consumers.md`. Item #1 carries the headline; items #2-3 are bench-confirmable but smaller. Bench targets: NVIDIA `commit_open_40q` for #1, encoder-time microbench for #3.

### Gate 2 (`v0.3`, ~+1-1.5 weeks): encode.rs hot-path cluster

- Item #4: Multi-dispatch per compute pass (~1-2 days).
- Item #5: Bind-group reuse pre-built at plan time (~1 day).
- Item #6: Poseidon2 constants migration storage → uniform / generated module constants (~2-3 days).

All three touch the same encode.rs and plan-build sites. Land as one refactor batch. Run a small-`log_n` microbench (e.g. `log_n ∈ {12, 14, 16}`) on at least one mobile host (Adreno or M4 Pro Metal) — that's where the per-call CPU overhead category actually shows up, not the log_h=18 desktop headline.

→ **Publish gate:** `v0.3` perf update note, focused on small-`log_n` and prover-loop scenarios.

### Gate 3 (`v0.4`, ~+1-1.5 weeks): advanced native paths

- Item #7: Trusted shader modules for audited native kernels (~3-5 days, dominated by audit).
- Item #8: Cross-operation command batching (~4-6 days, largest API-surface change).

Only proceed to Gate 3 if Gates 1-2 leave meaningful gap on the targets that matter (mobile / browser / integrated, plus prover-loop cumulative cost). Item #7 is `unsafe` and per-kernel audited; item #8 is the largest CPU-side mobile/browser candidate but also the biggest refactor.

### Stop rules across gates

- **After foundation commit:** if any parity test regresses, stop and investigate. Cache-key is correctness-critical.
- **After Gate 1:** if mixed-height injection (item #1) doesn't show a clear ratio improvement on NVIDIA `commit_open_40q` mixed-height, stop and check the cost model — something else is dominating. Items #2-3 can land independently of #1's bench result.
- **After Gate 2:** if the small-`log_n` microbench doesn't show measurable encoder-time reduction, the encoder-side category's impact may have been overestimated. Publish what's true and skip Gate 3.
- **External-signal override:** if the `v0.2` publish (Gate 1) generates external engagement (Plonky3 forum / Mopro / OpenVM consumers), pause Gate 2/3 and respond to that signal first.

### For *Conditional / future* (items #9-12)

- Schedule each item only when its listed trigger has fired.
- Reorder if a trigger fires for a lower-listed item before a higher-listed one — this list is by current likelihood of being triggered, not a fixed queue.

### Architectural constraint that overrides everything

Preserve the strongest current property of `zkgpu-wgpu`: one portable WGSL codebase that gets faster by making better planning decisions, not by fragmenting into per-backend products. The native-only items (#7 trusted modules, the trusted variants of #6 generated constants) must be **per-kernel opt-in with explicit audit**, not blanket. Native fast paths and portable browser paths share the same `PipelineRegistry` cache machinery and the same algorithmic primitives — they differ only at the per-pipeline opt-in points the cache key now distinguishes.
