# Item #5 verdict — Bind-group reuse, Stockham R4 Immediate

**Date:** 2026-05-01
**Pilot commit:** `f23a7db` (collapse + WgpuBuffer.id field)
**Status:** narrow scope shipped; broader cross-execute cache deferred.
**Bench reproducer:**

```bash
cargo bench -p zkgpu-wgpu --bench ntt_r4_param_mode -- \
    --warm-up-time 2 --measurement-time 5
```

## What shipped

`encode_ntt_stages` in the `R4ParamSource::Immediate` branch now builds **2** bind groups per encode (one per parity) instead of one per stage. wgpu's `BindGroup` is internally Arc-managed, so cloning the appropriate bind group into each stage is a cheap ref-count bump rather than a fresh allocation. Cuts the per-encode R4 bind-group create cost in Immediate mode from `O(num_r4_stages)` to `O(1)`.

Storage mode is unchanged: each stage's `param_buffer` lives at bind-group entry 3 (different per stage), so each stage requires its own bind group. R2 / local / Poseidon2 / Four-Step plans are unchanged for the same reason.

The `WgpuBuffer<F>` struct gained a process-unique `id: u64` field assigned via atomic counter at construction. Reserved for the deferred cross-execute bind-group cache; not yet wired in any plan.

## Why narrow scope

The doc's full scope ("pre-build two bind groups at plan-build time") assumes an API where the user buffer is known at plan time. The current API has the user buffer as a per-execute parameter (`fn execute(&mut self, buf: &mut WgpuBuffer)`). To deliver the doc's full scope without changing that API:

1. **Cache bind groups across execute() calls keyed by buf identity.**
2. Adapter pipelines (Plonky3, OpenVM) pass different buffers per call (different matrix rows, different leaves), so the cache hit rate is workload-dependent and uncertain a priori.
3. The cache itself adds a Mutex lookup per execute — small but non-zero overhead on the miss path.

Without evidence that consumer pipelines reuse the same buffer often enough for the cache to pay for itself, building the cache speculatively is the kind of "scaffolding without signal" the project's pilot pattern is designed to avoid (see items #6 and #3 verdicts). The buffer-id infrastructure is in place; the cache itself waits for a workload that demonstrates the hit rate.

## Bench data

### RTX 5090 / Vulkan (same-session A/B vs 2026-05-01 baseline)

| log_n | Immediate before | Immediate after | Δ | p |
|---|---:|---:|---:|---:|
| 10 | 41.457 µs | 41.124 µs | −0.8% | 0.76 |
| 14 | 57.807 µs | 57.838 µs | +0.05% | 0.29 |
| 18 | 81.954 µs | 82.029 µs | +0.09% | 0.60 |
| 20 | 111.03 µs | 107.94 µs | **−2.8%** | **0.01** |

Storage path changed by < 1% across all log_n with p ≫ 0.05 — confirming that the variance we do see is session-to-session noise rather than the collapse leaking into the unchanged path. The only stat-significant Immediate result is log_n=20 (−2.8%, p=0.01); CIs are just barely disjoint (Storage 20 lower bound 109.18 µs, Immediate 20 upper bound 109.07 µs).

### M4 Pro / Metal

The same A/B on M4 Pro showed both Storage and Immediate dropping by 20-33% from the 2026-04-30 baseline at log_n=20. Storage's code didn't change between commits, so this is system-side variance (thermal / load) between sessions, not a trustworthy collapse-isolation signal. Apple Silicon variability needs a controlled-thermal-state test rig to extract the collapse delta cleanly; not run here.

## Read

The collapse is logically sound (N create_bind_group calls → 2 is unambiguously cheaper for the GPU runtime) and benches cleanly net-positive on NVIDIA at log_n=20 — the regime where many R4 stages run and the bind-group cost stacks up. At smaller log_n, kernel time dominates and the collapse is below the noise floor.

This is consistent with item #3's pattern: encoder-side optimizations help most at the large-log_n end where many dispatches accumulate, and are below noise at small log_n. Item #3's R4 Immediate path becomes incrementally more attractive at log_n=20 with item #5 stacked on top, but Apple Silicon's small-log_n regression from item #3 is unchanged here (this commit doesn't touch Storage on Apple).

## Decision

- **Item #5 narrow scope shipped.** The Immediate path collapse is committed; consumers who opt into `R4ParamMode::Immediate` get the win automatically.
- **Auto-default does not change.** Per the item #3 verdict, default stays `Storage`; this commit doesn't shift that calculus.
- **Cross-execute cache deferred.** The `WgpuBuffer.id` infrastructure is in place. Wiring the cache into any plan waits for evidence that consumer hot paths reuse the same buffer often enough — possibly an instrumentation pass on Plonky3 fri_commit and OpenVM mixed-height to count distinct buf-ids per session.

## Re-evaluation triggers

The cross-execute cache becomes worth implementing if any of:

- Instrumentation shows >50% buf-id reuse rate in the Plonky3 / OpenVM hot paths.
- A new consumer workload arrives that explicitly reuses buffers (e.g. a multi-pass FRI variant operating on the same data).
- The R4 Immediate auto-default flips per item #3's re-evaluation triggers, expanding the surface that benefits from the within-encode collapse.

## Reproducer

```bash
git clone https://github.com/latifkasuli/zkgpu.git && cd zkgpu
cargo bench -p zkgpu-wgpu --bench ntt_r4_param_mode -- \
    --warm-up-time 2 --measurement-time 5
```

The collapse is on `f23a7db^..f23a7db`; baseline at `f23a7db^` (the merge of items #3 + #4 + #6) gives the comparison point.
