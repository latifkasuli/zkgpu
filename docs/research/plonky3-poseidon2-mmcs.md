# GPU Poseidon2 MMCS for Plonky3

This note captures the current narrow claim that `zkgpu` can support and reproduce from the code in this repository.

> **Sibling note:** [GPU Poseidon2 MMCS for OpenVM](openvm-poseidon2-mmcs.md) — the same shared backend, a different consumer (OpenVM's Plonky3 0.4.1 W16-leaf MMCS), portability-first framing.

## Claim

For a Plonky3 STARK stack using:

- `BabyBear`
- `TwoAdicFriPcs`
- Poseidon2 MMCS
- `cap_height = 0`

the `zkgpu-plonky3::gpu_mmcs::GpuPoseidon2Mmcs` adapter produces bit-identical commitments, openings, and verification behavior to Plonky3's CPU `MerkleTreeMmcs`, and materially reduces prover wall time on discrete NVIDIA hardware.

Measured wall-clock wins on two matched consumer-flagship pairs:

| Host | `fri_commit @ log_h=18, w=8` | `prove+verify @ FibAir, log_h=18` |
|---|---:|---:|
| RTX 4090 + Ryzen 9 7950X (Zen 4) | **9.78x** | **3.95x** |
| RTX 5090 + Ryzen 9 9950X (Zen 5) | **16.20x** | **4.63x** |

On the Blackwell / Zen 5 pair the CPU baseline is ~1.22x faster than on Ada / Zen 4 (native AVX-512 helping Plonky3's `BabyBear::Packing`), and the GPU side is ~1.41-2.01x faster, so the MMCS ratio grows on newer silicon rather than shrinking. The envelope extends cleanly to log_h=20 (~1M rows).

## Scope

This is a narrow engineering result, not a universal GPU-prover claim.

It covers:

- the Plonky3 Poseidon2-MMCS stack
- `BabyBear`
- `TwoAdicFriPcs`
- single-matrix trace commit
- same-height multi-matrix quotient chunks
- full `prove + verify` on the in-tree `FibAir` benchmark

It does not claim:

- "GPU always wins"
- a drop-in `Mmcs` for arbitrary mixed-height `compress_and_inject`
- support for `cap_height > 0`
- a result for Keccak-MMCS stacks
- a result for non-Plonky3 proving systems
- a result for every GPU/CPU pair

## Code Surface

The result is backed by these tracked components:

- `zkgpu-plonky3` GPU MMCS adapter:
  - [`crates/zkgpu-plonky3/src/gpu_mmcs.rs`](../../crates/zkgpu-plonky3/src/gpu_mmcs.rs)
- GPU Poseidon2 Merkle commit backend:
  - [`crates/zkgpu-wgpu/src/poseidon2/merkle_commit.rs`](../../crates/zkgpu-wgpu/src/poseidon2/merkle_commit.rs)
  - [`crates/zkgpu-wgpu/src/poseidon2/merkle_leaf.rs`](../../crates/zkgpu-wgpu/src/poseidon2/merkle_leaf.rs)
  - [`crates/zkgpu-wgpu/src/poseidon2/plonky3_plan.rs`](../../crates/zkgpu-wgpu/src/poseidon2/plonky3_plan.rs)
- Correctness tests:
  - [`crates/zkgpu-plonky3/tests/poseidon2_mmcs_gpu.rs`](../../crates/zkgpu-plonky3/tests/poseidon2_mmcs_gpu.rs)
  - [`crates/zkgpu-plonky3/tests/poseidon2_bridge_gpu.rs`](../../crates/zkgpu-plonky3/tests/poseidon2_bridge_gpu.rs)
  - [`crates/zkgpu-plonky3/tests/poseidon2_merkle_commit_gpu.rs`](../../crates/zkgpu-plonky3/tests/poseidon2_merkle_commit_gpu.rs)
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
- `Mmcs::verify_batch` parity against the CPU reference path

## Why This Matters

This result is the first tracked point in the repository where `zkgpu` is not only accelerating an isolated primitive, but producing a measured win on a real prover workload.

The main takeaway from the measured data is:

- `GpuDft` is not the story for this stack's prover wall time
- `GpuPoseidon2Mmcs` is

That observation changes the roadmap. For this target stack, GPU MMCS was the pitch-defining step; GPU-resident LDE handoff is now secondary pipeline cleanup rather than the main performance lever.

## Current Limitations

- `cap_height > 0` is rejected
- arbitrary mixed-height multi-matrix `compress_and_inject` is not implemented
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
