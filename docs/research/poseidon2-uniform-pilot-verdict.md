# Item #6 pilot verdict — Poseidon2 constants storage vs uniform

**Date:** 2026-04-28
**Pilot commit:** `898b5bd` (kernel + plan), `05b9f56` (bench)
**Bench reproducer:**

```bash
git clone https://github.com/latifkasuli/zkgpu.git && cd zkgpu
cargo bench -p zkgpu-wgpu --bench poseidon2_constants_source -- \
    --warm-up-time 2 --measurement-time 5
```

**Scope:** standalone `WgpuBabyBearPoseidon2Plan` permutation throughput, BabyBear W16, three batch sizes (64, 4096, 65536). Upload, plan-build, and readback are all outside the timing loop; only `plan.execute(&device, &mut buf)` is measured.

## Headline numbers

Each cell is the Criterion median across 100 samples (5s measurement window after a 2s warmup). "Δ" is `(uniform − storage) / storage`. Negative = uniform faster.

| Host | Batch | Storage median | Uniform median | Δ | Verdict |
|------|------:|---------------:|---------------:|----:|---------|
| **M4 Pro / Metal** | 64    | 1.4461 ms | 1.5032 ms | **+3.9%** | uniform slower |
| **M4 Pro / Metal** | 4096  | 1.3738 ms | 1.4093 ms | **+2.6%** | uniform slower |
| **M4 Pro / Metal** | 65536 | 5.2643 ms | 5.3821 ms | **+2.2%** | uniform slower |
| **RTX 4090 / Vulkan** | 64    | 269.95 µs | 271.22 µs | +0.5% | within noise |
| **RTX 4090 / Vulkan** | 4096  | 270.90 µs | 268.41 µs | −0.9% | within noise |
| **RTX 4090 / Vulkan** | 65536 | 434.50 µs | 436.16 µs | +0.4% | within noise |
| **RTX 5090 / Vulkan** | 64    | 271.27 µs | 260.27 µs | **−4.1%** | uniform faster |
| **RTX 5090 / Vulkan** | 4096  | 273.55 µs | 259.94 µs | **−5.0%** | uniform faster |
| **RTX 5090 / Vulkan** | 65536 | 324.86 µs | 318.03 µs | −2.1% | uniform faster (small) |

CIs were tight on every host (`high - low` typically ±0.3% on NVIDIA, ±2-3% on M4 Pro). M4 Pro and 5090 deltas are outside the 95% CI band; 4090 is squarely inside it.

## Hypothesis vs. result

The pilot's hypothesis was that uniforms route through dedicated constant-cache hardware (Metal `constant` address space, NVIDIA cmem) and therefore beat storage on the broadcast access pattern Poseidon2 round constants exhibit (every thread in a warp reads the same `constants[i]` at the same time).

The hypothesis is **partially falsified**:

- **Apple Silicon regresses.** M4 Pro's unified-memory architecture appears to serve small storage-buffer reads through the same caching tier uniforms would route through. The Uniform variant adds the cost of `vec4<u32>` lane-select packing without any compensating cache-tier gain, so it lands net-negative.
- **NVIDIA 4090 is null.** The cmem path exists and is used, but the working-set delta (704 B for the Uniform packed struct vs ~640 B across three storage buffers) is small enough that L1-resident storage reads are indistinguishable from constant-cache reads at this kernel size.
- **NVIDIA 5090 wins, modestly.** Same architecture family as the 4090 but newer silicon and driver. The 2-5% uniform win is reproducible across all three batch sizes with tight CIs. Most plausibly attributable to the 5090's reorganized constant-cache path or a Vulkan-driver-side bind-group fast path for the smaller (3-entry vs 5-entry) bind group the Uniform variant uses on every `execute` call.

The "fixed-overhead" interpretation (that the win is bind-group-allocation cost, not kernel cache) doesn't hold: the 4090 has the same bind-group difference and shows null. So the 5090 win is most likely a real, architecture-specific cache-path effect that just doesn't generalize.

## Decision against the documented gate

The pilot's pre-registered gate criterion in commit `898b5bd`: *"If Metal shows ≥ 5% win, propagate to the four production kernels (merkle_leaf, merkle_leaf_w16, merkle_compress, plonky3_w16, plonky3_w24)."*

Metal shows a **regression**, not a 5% win. **Gate not met. Do not propagate.**

A secondary criterion of "any host ≥ 5%" would catch the 5090, but propagating with a clear Metal regression would damage the portability story (Apple is the consumer the v0.2/v0.3 narrative hinges on for the cross-platform claim) and the 5090 win is small enough that an end-to-end MMCS bench would likely have it inside the discrete-GPU clocking-noise floor anyway.

## What ships

- **Production kernels keep the Storage path.** No change to `merkle_leaf`, `merkle_leaf_w16`, `merkle_compress`, `plonky3_w16`, `plonky3_w24`.
- **Pilot path stays.** `Poseidon2ConstantsSource::Uniform` remains accessible via `WgpuBabyBearPoseidon2Plan::new_with_constants_source` for callers who want to opt in on 5090-class hardware. Tests + bench stay green.
- **Validation scope stays.** The `push_validation_scope` / `pop_validation_scope` wrap on plan construction is permanent — it caught a real silent-failure mode (`external` / `internal` are reserved WGSL keywords, and Naga's tombstone pipeline only manifests as zero-valued GPU output). Native-only — `pop_validation_scope` is gated `#[cfg(not(target_arch = "wasm32"))]` to mirror the NTT builder pattern; the wasm/WebGPU build is not affected by this addition.

## Future re-evaluation triggers

The Uniform path becomes worth revisiting if any of:

- Apple's Naga / Metal codegen for `array<vec4<u32>>` lane-select improves measurably (current Naga master may already differ from the wgpu v29 version we're pinned to).
- A future Poseidon2 variant lands with a much larger constant footprint (e.g. wider state, more partial rounds) where the working set crosses the L1-resident threshold and the constant-cache path actually buys cache-tier separation.
- The "generated module-constant" variant of item #6 (the *faster* of the two paths the speed-opportunities doc lists) is implemented — the constants get inlined as WGSL `const` arrays, which is a different mechanism entirely from the uniform-buffer path piloted here.

## Reproducer

On any host with a wgpu-supported GPU:

```bash
git clone https://github.com/latifkasuli/zkgpu.git && cd zkgpu
cargo bench -p zkgpu-wgpu --bench poseidon2_constants_source -- \
    --warm-up-time 2 --measurement-time 5
```

The two Criterion groups (`poseidon2_storage` and `poseidon2_uniform`) run back-to-back; eyeballing the median delta at each batch size gives the per-host signal.
