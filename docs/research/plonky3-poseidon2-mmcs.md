# GPU Poseidon2 MMCS for Plonky3

This note captures the current claim that `zkgpu` can support and reproduce from the code in this repository.

> **Sibling note:** [GPU Poseidon2 MMCS for OpenVM](openvm-poseidon2-mmcs.md) — the **same shared backend**, a different consumer (OpenVM's Plonky3 0.4.1 W16-leaf MMCS), portability-first framing.
>
> **v0.2 update (2026-04-28).** The shared backend now runs the mixed-height commit DAG entirely device-resident (item #1 of the speed-opportunities note). The Plonky3-side headline numbers in this note are single-matrix shapes; the same-height fast path was unchanged in v0.2, so the headline ratios below remain accurate as of the same vast.ai host configuration that produced them. The mixed-height parity tests added at adapter convergence (4 tests in `poseidon2_mmcs_gpu.rs`) continue to pass under the new GPU-resident path; per-shape mixed-height timings on the Plonky3 W24 leaf are still not bench-headlined here. See the OpenVM note's v0.2 section for the measured mixed-height GPU-time recovery; that result lands on the shared engine that both adapters drive, so Plonky3 mixed-height consumers benefit without any Plonky3-specific code change.

## Claim

For a Plonky3 STARK stack using:

- `BabyBear`
- `TwoAdicFriPcs`
- Poseidon2 MMCS (W24 leaf + W16 binary compression)
- `cap_height = 0`

the `zkgpu-plonky3::gpu_mmcs::GpuPoseidon2Mmcs` adapter produces bit-identical commitments, openings, and verification behavior to Plonky3's CPU `MerkleTreeMmcs` — across single-matrix, same-height multi-matrix, **and mixed-height multi-matrix** (`compress_and_inject` DAG) shapes. It materially reduces prover wall time on discrete NVIDIA hardware.

The mixed-height path routes through the shared backend's `commit_mixed_height_with_w24_leaf` engine — the same engine the sibling [`zkgpu-openvm`](openvm-poseidon2-mmcs.md) adapter uses with its W16 leaf variant. **Two consumer adapters; one shared mixed-height MMCS backend; parity-pinned end-to-end.**

Measured wall-clock wins on two matched consumer-flagship pairs:

| Host | `fri_commit @ log_h=18, w=8` | `prove+verify @ FibAir, log_h=18` |
|---|---:|---:|
| RTX 4090 + Ryzen 9 7950X (Zen 4) | **11.46x** | **3.98x** |
| RTX 5090 + Ryzen 9 9950X (Zen 5) | **15.07x** | **4.76x** |

On the Blackwell / Zen 5 pair the CPU baseline is ~1.22x faster than on Ada / Zen 4 (native AVX-512 helping Plonky3's `BabyBear::Packing`), and the GPU side is faster too, so the MMCS ratio still grows on newer silicon. These numbers are commit-only on the same-height shape; the mixed-height path is parity-validated separately (see below).

**On the mixed-height path:** semantics are parity-validated for Plonky3 (4 mixed-height parity tests, byte-identical commit roots and `BatchOpening` against Plonky3 0.5.x's CPU `MerkleTreeMmcs`). Mixed-height **performance** has not been benchmarked separately on the Plonky3 config — the OpenVM note's mixed-height numbers (e.g. 20.07× on 5090 at `log_h_max=18` mixed-height) cover the same DAG engine but a different leaf shape (OpenVM uses W16/RATE=8, Plonky3 uses W24/RATE=16). The leaf-cost asymmetry already shows up in the `target_stack/commit` data inside the OpenVM note (single-matrix W16 at log_h=18 hits 18.26× on the 5090 vs Plonky3's W24 at 15.07× on the same shape), so OpenVM's mixed-height numbers are evidence the shared engine works, not a Plonky3-config performance claim. A dedicated mixed-height bench at the Plonky3 W24 leaf shape is straightforward future work; not in scope today.

### Convergence + same-height fast path (2026-04-25)

When `GpuPoseidon2Mmcs` was first migrated onto the shared mixed-height DAG engine (so both consumer adapters share one backend), the same-height commit-only ratio temporarily regressed on NVIDIA. Root cause: the general DAG path downloads each level's digests to host between compression levels (so it can interleave injection digests at injection levels) — for same-height inputs no injection ever happens, so those `log2(h_max)` PCIe round-trips are pure overhead.

A same-height fast path was added to the shared backend
([`crates/zkgpu-wgpu/src/poseidon2/merkle_commit_dag.rs`](../../crates/zkgpu-wgpu/src/poseidon2/merkle_commit_dag.rs))
that detects "every input matrix has one height" early in
`commit_mixed_height_internal` and routes to a GPU-resident
pipeline (pre-allocate every retained-layer buffer, run the leaf
sponge into level 0, run binary compression entirely
device-resident, batch-download every layer once at the end). For
the single-matrix case it also skips an unnecessary host concat
copy and uploads the input slice directly. Both consumer adapters
benefit — the fix lives in the shared backend, not in either
adapter, so no re-fork.

Recovery progression at `fri_commit @ log_h=18, w=8`:

| Host | Pre-convergence | Post-convergence (no fast path) | **Post fast path (current)** | Δ vs pre |
|---|---:|---:|---:|:---:|
| RTX 4090 + 7950X | 9.78× | 8.09× (-17%) | **11.46×** | +17% (improved) |
| RTX 5090 + 9950X | 16.20× | 11.41× (-30%) | **15.07×** | -7% (within stop rule) |

Why the 4090 ratio actually improved past the pre-convergence baseline: the fast path's batched end-of-pipeline download appears to be more efficient than the old `commit_with_retained_layers` per-buffer download pattern. The 5090's residual gap (~7%) is small enough to land cleanly; further optimization would need profiling to identify the remaining cost. Prove ratios are essentially unchanged on both hosts.

The fast path itself is an unconditional optimization — same-height shapes always take it, mixed-height shapes always take the general DAG path. No public API change; no consumer adapter change.

## Scope

This is a focused engineering result, not a universal GPU-prover claim.

It covers:

- the Plonky3 Poseidon2-MMCS stack
- `BabyBear`
- `TwoAdicFriPcs`
- single-matrix trace commit
- same-height multi-matrix quotient chunks
- **mixed-height multi-matrix** via the `compress_and_inject` DAG engine
- full `prove + verify` on the in-tree `FibAir` benchmark

It does not claim:

- "GPU always wins"
- support for `cap_height > 0`
- a result for Keccak-MMCS stacks
- a result for non-Plonky3 proving systems
- a result for every GPU/CPU pair

## Code Surface

The result is backed by these tracked components:

- `zkgpu-plonky3` GPU MMCS adapter:
  - [`crates/zkgpu-plonky3/src/gpu_mmcs.rs`](../../crates/zkgpu-plonky3/src/gpu_mmcs.rs) — `GpuPoseidon2Mmcs`, routing through the shared mixed-height DAG engine
- Shared mixed-height DAG backend (zkgpu-wgpu):
  - [`crates/zkgpu-wgpu/src/poseidon2/merkle_commit_dag.rs`](../../crates/zkgpu-wgpu/src/poseidon2/merkle_commit_dag.rs) — `commit_mixed_height_with_w24_leaf`, `open_batch_mixed_height` — the same engine the OpenVM adapter uses (with `_w16_leaf`)
  - [`crates/zkgpu-wgpu/src/poseidon2/merkle_leaf.rs`](../../crates/zkgpu-wgpu/src/poseidon2/merkle_leaf.rs) — W24/RATE=16 leaf sponge plan
  - [`crates/zkgpu-wgpu/src/poseidon2/merkle_compress.rs`](../../crates/zkgpu-wgpu/src/poseidon2/merkle_compress.rs) — W16 binary compression plan
  - [`crates/zkgpu-wgpu/src/poseidon2/plonky3_plan.rs`](../../crates/zkgpu-wgpu/src/poseidon2/plonky3_plan.rs) — Plonky3 Poseidon2 permutation plan
- Correctness tests:
  - [`crates/zkgpu-plonky3/tests/poseidon2_mmcs_gpu.rs`](../../crates/zkgpu-plonky3/tests/poseidon2_mmcs_gpu.rs) — 12 tests: single, same-height multi, mixed-height (2-level / 3-level / every-level / multi-at-non-max-height), open + verify_batch parity, cross-verifier roundtrip, cap_height guard
  - [`crates/zkgpu-plonky3/tests/poseidon2_bridge_gpu.rs`](../../crates/zkgpu-plonky3/tests/poseidon2_bridge_gpu.rs)
  - [`crates/zkgpu-plonky3/tests/poseidon2_merkle_commit_dag_gpu.rs`](../../crates/zkgpu-plonky3/tests/poseidon2_merkle_commit_dag_gpu.rs) — backend mixed-height commit parity (W24 leaf)
  - [`crates/zkgpu-plonky3/tests/poseidon2_merkle_open_dag_gpu.rs`](../../crates/zkgpu-plonky3/tests/poseidon2_merkle_open_dag_gpu.rs) — backend mixed-height open parity (both W24 and W16 leaf)
- Benchmark harness:
  - [`crates/zkgpu-plonky3/benches/prover_hot_path.rs`](../../crates/zkgpu-plonky3/benches/prover_hot_path.rs)

## Benchmark Envelope

The in-tree benchmark harness pins the target stack as follows:

- field: `BabyBear`
- DFT variants:
  - CPU: `Radix2DitParallel`
  - GPU: `GpuDft::strict_gpu()`
- MMCS variants:
  - CPU: Plonky3 `MerkleTreeMmcs`
  - GPU: `GpuPoseidon2Mmcs`
- FRI params:
  - `log_blowup = 1`
  - `log_final_poly_len = 0`
  - `max_log_arity = 2`
  - `num_queries = 40`
  - `commit_proof_of_work_bits = 0`
  - `query_proof_of_work_bits = 8`
- commitment cap:
  - `cap_height = 0`

The benchmark groups are:

- `target_stack/coset_lde_batch`
- `target_stack/fri_commit`
- `target_stack/prove`

The prove benchmark uses `FibAir` with width `2`, and measures full `prove` followed by `verify`.

## Results

### FRI Commit

Measured on RTX 4090 + Ryzen 9 7950X.

Target shape:

- `target_stack/fri_commit`
- `log_h = 18`
- `w = 8`

| Variant | Median time | Ratio vs baseline |
|---|---:|---:|
| `cpu_dft_cpu_mmcs` | 1.081 s | 1.00x |
| `gpu_dft_cpu_mmcs` | 1.096 s | 0.99x |
| `cpu_dft_gpu_mmcs` | 110.5 ms | 9.78x |
| `gpu_dft_gpu_mmcs` | 111.8 ms | 9.67x |

Interpretation:

- Swapping only the DFT does essentially nothing for `fri_commit`.
- Swapping only the MMCS collapses commit time by almost an order of magnitude.
- The bottleneck at this shape is Poseidon2 leaf hashing + tree compression, not the DFT.

### Full Prove + Verify

Measured on RTX 4090 + Ryzen 9 7950X.

Target group:

- `target_stack/prove`
- `FibAir`
- 10-sample median

| `log_h` | `cpu_dft_cpu_mmcs` | `gpu_dft_cpu_mmcs` | `cpu_dft_gpu_mmcs` | `gpu_dft_gpu_mmcs` |
|---:|---:|---:|---:|---:|
| 10 | 12.35 ms | 14.57 ms (0.85x) | 12.41 ms (0.99x) | 13.14 ms (0.94x) |
| 14 | 161.1 ms | 166.9 ms (0.97x) | 58.2 ms (2.77x) | 58.7 ms (2.74x) |
| 16 | 640.2 ms | 645.6 ms (0.99x) | 217.6 ms (2.94x) | 217.3 ms (2.95x) |
| 18 | 2559.9 ms | 2591.3 ms (0.99x) | 648.1 ms (3.95x) | 657.2 ms (3.89x) |

### Second host: RTX 5090 + Ryzen 9 9950X

Measured on a vast.ai box with a **next-generation matched pair** of consumer flagships:

- GPU: NVIDIA RTX 5090 (Blackwell, 32 GiB GDDR7, driver 590.48.01)
- CPU: AMD Ryzen 9 9950X (Zen 5, 16-core / 32-thread, native AVX-512)
- RAM: 186 GiB
- OS: Ubuntu 22.04, wgpu Vulkan backend (Vulkan 1.3.275)

Same benchmark harness, same code, same commits. Both GPUs available in the instance; all numbers below are **single-GPU** (wgpu defaults to device 0) to match the 4090 methodology.

#### `target_stack/fri_commit`

| `log_h` | `w` | `cpu_dft_cpu_mmcs` | `gpu_dft_cpu_mmcs` | `cpu_dft_gpu_mmcs` | `gpu_dft_gpu_mmcs` |
|---:|---:|---:|---:|---:|---:|
| 14 | 1 | 53.17 ms | 53.51 ms (0.99x) | 5.47 ms (9.72x) | 5.30 ms (**10.03x**) |
| 14 | 8 | 54.61 ms | 55.12 ms (0.99x) | 6.98 ms (7.82x) | 7.19 ms (7.60x) |
| 16 | 1 | 212.80 ms | 215.11 ms (0.99x) | 10.15 ms (20.97x) | 8.20 ms (**25.95x**) |
| 16 | 8 | 219.64 ms | 218.00 ms (1.01x) | 16.77 ms (13.10x) | 14.12 ms (**15.55x**) |
| 18 | 1 | 856.26 ms | 844.19 ms (1.01x) | 28.60 ms (29.94x) | 18.78 ms (**45.60x**) |
| 18 | 8 | 889.50 ms | 876.03 ms (1.02x) | 68.95 ms (12.90x) | 54.91 ms (**16.20x**) |
| 20 | 1 | 3.457 s | 3.415 s (1.01x) | 135.56 ms (25.50x) | 89.15 ms (**38.78x**) |
| 20 | 8 | 3.621 s | 3.621 s (1.00x) | 331.23 ms (10.93x) | 301.85 ms (**12.00x**) |

#### `target_stack/prove` (FibAir, prove + verify)

| `log_h` | `cpu_dft_cpu_mmcs` | `gpu_dft_cpu_mmcs` | `cpu_dft_gpu_mmcs` | `gpu_dft_gpu_mmcs` |
|---:|---:|---:|---:|---:|
| 10 | 10.17 ms | 11.27 ms (0.90x) | 10.03 ms (1.01x) | 10.70 ms (0.95x) |
| 14 | 132.82 ms | 133.39 ms (1.00x) | 36.97 ms (3.59x) | 37.87 ms (3.51x) |
| 16 | 525.14 ms | 526.12 ms (1.00x) | 118.33 ms (4.44x) | 114.47 ms (**4.59x**) |
| 18 | 2.123 s | 2.105 s (1.01x) | 466.0 ms (4.56x) | 458.1 ms (**4.63x**) |

#### Host-to-host comparison at the target shape (log_h=18, w=8 for fri_commit; log_h=18 for prove)

| | 4090 + 7950X | 5090 + 9950X | Change |
|---|---:|---:|---|
| `fri_commit` baseline (cpu/cpu) | 1.081 s | 889.5 ms | CPU 1.22x faster on 9950X |
| `fri_commit` best GPU (gpu/gpu) | 110.5 ms | 54.9 ms | GPU 2.01x faster on 5090 |
| **`fri_commit` ratio** | **9.78x** | **16.20x** | **+66%** |
| `prove+verify` baseline | 2.56 s | 2.12 s | CPU 1.21x faster on 9950X |
| `prove+verify` best GPU | 648 ms | 458 ms | GPU 1.41x faster on 5090 |
| **`prove+verify` ratio** | **3.95x** | **4.63x** | **+17%** |

#### Interpretation of the second-host data

- The 9950X strengthens the CPU baseline (native AVX-512 helps Plonky3's `BabyBear::Packing`), as predicted. CPU is ~1.22x faster for `fri_commit` and ~1.21x faster for `prove`.
- The 5090 strengthens the GPU side by more (~2.0x on `fri_commit`, ~1.41x on `prove`), so **both ratios grow** on the newer matched pair rather than shrinking.
- The w=1 narrow-matrix case is dramatic: 45.60x on `fri_commit` at log_h=18. Narrow matrices are dominated by per-row hash overhead, and the 5090 absorbs that workload especially well.
- The envelope now extends cleanly to log_h=20 (~1M rows) — the `fri_commit` ratio at log_h=20, w=1 is 38.78x; at log_h=20, w=8 it is 12.00x. Within the envelope of real zkVM trace sizes.
- `gpu_dft_cpu_mmcs` still hugs the CPU baseline across every row of both groups (~0.99-1.02x of `cpu_dft_cpu_mmcs`) on the 5090 too. The "DFT swap alone does nothing for commit" finding from the 4090 data holds on Blackwell.

Interpretation:

- At small sizes, GPU overhead dominates and the CPU still wins.
- As `log_h` grows, the commit phase becomes a larger share of end-to-end proving time.
- The GPU MMCS path is what changes the curve materially; GPU DFT alone does not.

## Correctness Evidence

The narrow claim is backed by tracked tests that lock GPU behavior to Plonky3's CPU reference:

- Poseidon2 parameter bridge differential tests for widths 16 and 24
- GPU permutation differential tests for widths 16 and 24
- GPU leaf sponge differential tests
- GPU Merkle commit differential tests
- GPU MMCS commit and `open_batch` differential tests
- full `verify` round-trip tests through the MMCS adapter

In particular, the GPU MMCS test surface locks:

- single-matrix trace commit
- same-height multi-matrix quotient opens
- mixed-height multi-matrix DAG (`compress_and_inject`-style trees, every shape Plonky3's `MerkleTreeMmcs` validly accepts at `cap_height=0`)
- `Mmcs::verify_batch` parity against the CPU reference path
- cross-verifier roundtrip: GPU openings verified by both the adapter's own `verify_batch` and a freshly-built CPU `MerkleTreeMmcs::verify_batch` over the GPU commit

## Why This Matters

This result is the first tracked point in the repository where `zkgpu` is not only accelerating an isolated primitive, but producing a measured win on a real prover workload.

The main takeaway from the measured data is:

- `GpuDft` is not the story for this stack's prover wall time
- `GpuPoseidon2Mmcs` is

That observation changes the roadmap. For this target stack, GPU MMCS was the pitch-defining step; GPU-resident LDE handoff is now secondary pipeline cleanup rather than the main performance lever.

## Current Limitations

- `cap_height > 0` is rejected
- the adapter targets the Poseidon2-MMCS stack, not Keccak-MMCS
- numerical results are measured on discrete NVIDIA hardware (RTX 4090 and RTX 5090). The backend is portable via wgpu (Metal, Vulkan, DX12, WebGPU); other targets have functional tests but no published benchmark numbers yet
- the full claim is benchmark-backed for the in-tree `FibAir` proving workload, not every AIR or every proof system

## Reproducing

Compile the benchmark harness:

```bash
cargo bench -p zkgpu-plonky3 --bench prover_hot_path --no-run
```

Run the benchmark:

```bash
cargo bench -p zkgpu-plonky3 --bench prover_hot_path
```

Run the GPU MMCS parity tests:

```bash
cargo test -p zkgpu-plonky3 --test poseidon2_mmcs_gpu -- --nocapture
```

Run the workspace check:

```bash
cargo check --workspace
```

## Follow-On Work

The next plausible branches are:

1. Mixed-height multi-matrix support for broader Plonky3 `Mmcs` coverage
2. GPU-resident `coset_lde_batch` handoff cleanup
3. A second consumer adapter to prove `zkgpu` is a reusable platform, not only a Plonky3 optimization

If this note is used externally, the safest summary sentences are:

> On a Plonky3 BabyBear + Poseidon2-MMCS + `TwoAdicFriPcs` stack, `zkgpu`'s GPU MMCS path matches the CPU reference exactly and reduces FRI commit time by about 9.8x and full `prove + verify` time by up to 3.95x on RTX 4090 + Ryzen 9 7950X.

> On a next-generation RTX 5090 + Ryzen 9 9950X matched pair (Blackwell GPU, Zen 5 CPU with native AVX-512), the same code reduces FRI commit time by about 16.2x at log_h=18, w=8 (up to 45.6x at log_h=18, w=1) and full `prove + verify` by about 4.63x at log_h=18. Both the CPU baseline and the GPU path got faster on the newer hardware; the GPU path got faster by more, so the ratio grows rather than shrinks.
