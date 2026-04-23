# GPU Poseidon2 MMCS for OpenVM

This note captures the current narrow claim that `zkgpu` can support and reproduce from the code in this repository for the **OpenVM** consumer stack.

Sibling to [`docs/research/plonky3-poseidon2-mmcs.md`](plonky3-poseidon2-mmcs.md). Same shared GPU backend, different consumer adapter, different Plonky3 version pin, different Poseidon2 shape.

## Claim

For an OpenVM STARK stack using:

- `BabyBear`
- `MerkleTreeMmcs` with OpenVM's BabyBear Poseidon2 config
  (`PaddingFreeSponge<Perm16, 16, 8, 8>` leaf + `TruncatedPermutation<Perm16, 2, 8, 16>` compression)
- binary N=2 tree
- `cap_height = 0`

the `zkgpu_openvm::OpenVmGpuMmcs` adapter produces bit-identical commitments, openings, and verifier-roundtrip behavior to OpenVM's CPU reference (Plonky3 0.4.1 `MerkleTreeMmcs` under the same config), and materially reduces prover-side commit wall time on a representative range of hosts — including **non-NVIDIA hardware, where OpenVM's own CUDA backend cannot run**.

### Headline numbers — `target_stack/commit`, `log_h_max = 18`, trace + 4 quotient chunks

| Host | CPU | GPU | Ratio |
|---|---:|---:|---:|
| Apple M4 Pro / Metal | 110.62 ms | **101.46 ms** | **1.09×** |
| RTX 4090 + Ryzen 9 7950X | 605.14 ms | **59.66 ms** | **10.14×** |
| RTX 5090 + Ryzen 9 9950X | 481.50 ms | **23.99 ms** | **20.07×** |

## Scope

Narrow on purpose.

Covers:

- OpenVM BabyBear Poseidon2 config (W16/RATE=8/DIGEST=8)
- Plonky3 0.4.1 surface (the version OpenVM's own `stark-backend` workspace pins)
- binary `MerkleTreeMmcs`, `cap_height = 0`
- mixed-height commits — single matrix, same-height multi-matrix, **and** `compress_and_inject`-style mixed-height DAG (OpenVM's `VERIFY_BATCH` shape)
- `commit`, `open_batch`, `verify_batch` at the `p3_commit::Mmcs<BabyBear>` trait surface

Does not claim:

- "GPU always wins" — at `log_h ≤ 16` on Apple Silicon, CPU still wins because launch overhead dominates.
- A replacement for OpenVM's own CUDA backend on NVIDIA. OpenVM ships `openvm-cuda-backend` for that. zkgpu-openvm is the **portable** path — Metal, Vulkan (AMD / mobile / server non-CUDA), DX12, WebGPU — that the CUDA backend can't serve.
- Correctness for `cap_height > 0`. The adapter's `new()` rejects it.
- A result for non-Plonky3 proving systems, or other OpenVM components (AIR evaluation, FRI fold commits, etc.).
- Performance on GPUs that failed `zkgpu` adapter init (driver blocklist, etc.).

## Positioning relative to OpenVM's CUDA backend

OpenVM ships a first-party `openvm-cuda-backend` crate that accelerates the same Poseidon2 STARK config on NVIDIA GPUs. zkgpu-openvm is **not** trying to beat that on CUDA:

- On a 5090, `openvm-cuda-backend` has native CUDA kernels specialized for NVIDIA hardware. zkgpu-openvm routes through wgpu → Vulkan, which has ~10–20% overhead vs native CUDA in general. Not the lane to compete.
- zkgpu-openvm's lane is **every GPU that isn't NVIDIA**:
  - Apple M-series via Metal (the blocking Gate 4a below)
  - AMD GPUs via Vulkan
  - Intel / DX12 Windows hosts
  - WebGPU in browsers (experimental)

The value proposition is breadth, not peak NVIDIA throughput. The NVIDIA numbers in this note are reported informationally to show the DAG-level acceleration works on those hosts too, not as a CUDA-beating claim.

## Code Surface

Backed by these tracked components:

- `zkgpu-openvm` adapter:
  - [crates/zkgpu-openvm/src/gpu_mmcs.rs](../../crates/zkgpu-openvm/src/gpu_mmcs.rs)
  - [crates/zkgpu-openvm/src/config.rs](../../crates/zkgpu-openvm/src/config.rs) — local rebuild of OpenVM's BabyBear Poseidon2 type aliases
  - [crates/zkgpu-openvm/src/bridge.rs](../../crates/zkgpu-openvm/src/bridge.rs) — Plonky3 0.4.1 → zkgpu `Poseidon2Params` bridge
- Shared mixed-height DAG backend (zkgpu-wgpu):
  - [crates/zkgpu-wgpu/src/poseidon2/merkle_commit_dag.rs](../../crates/zkgpu-wgpu/src/poseidon2/merkle_commit_dag.rs) — commit + open engine
  - [crates/zkgpu-wgpu/src/poseidon2/merkle_leaf_w16.rs](../../crates/zkgpu-wgpu/src/poseidon2/merkle_leaf_w16.rs) — W16/RATE=8 leaf sponge plan
  - [crates/zkgpu-wgpu/src/poseidon2/merkle_compress.rs](../../crates/zkgpu-wgpu/src/poseidon2/merkle_compress.rs) — W16 compression plan
- Correctness tests:
  - [crates/zkgpu-openvm/tests/commit_parity.rs](../../crates/zkgpu-openvm/tests/commit_parity.rs) — root parity vs CPU `MerkleTreeMmcs`
  - [crates/zkgpu-openvm/tests/open_verify_parity.rs](../../crates/zkgpu-openvm/tests/open_verify_parity.rs) — `(opened_values, proof)` parity + cross-verifier roundtrip
  - [crates/zkgpu-plonky3/tests/poseidon2_merkle_commit_dag_w16_gpu.rs](../../crates/zkgpu-plonky3/tests/poseidon2_merkle_commit_dag_w16_gpu.rs) — backend mixed-height commit parity (W16 leaf)
  - [crates/zkgpu-plonky3/tests/poseidon2_merkle_leaf_w16_gpu.rs](../../crates/zkgpu-plonky3/tests/poseidon2_merkle_leaf_w16_gpu.rs) — backend W16/R8 leaf sponge parity
- Benchmark harness:
  - [crates/zkgpu-openvm/benches/openvm_commit.rs](../../crates/zkgpu-openvm/benches/openvm_commit.rs)

## Benchmark envelope

The in-tree bench harness pins the OpenVM shape as follows:

- Field: `BabyBear`
- Poseidon2: W16/RATE=8, single `Perm16` for leaf and compression
- Plonky3 version: `=0.4.1` (matching OpenVM's own pin)
- MMCS: `MerkleTreeMmcs<Packing, Packing, Hash, Compress, 8>`
- Cap height: `0`
- Methodology: Criterion, 15 samples per bench, 10s target measurement time — matches `zkgpu-plonky3`'s `target_stack/fri_commit` block

Benchmark groups:

- `target_stack/commit` — primary metric, commit-only.
- `target_stack/commit_open_40q` — commit + 40 consecutive `open_batch` queries. Mirrors the FRI-side per-commit workload.

`verify_batch` is intentionally not in any timed group — the adapter delegates verification to CPU `MerkleTreeMmcs` by design, so timing it would measure the CPU delegate, not the GPU adapter.

Shapes:

- **Single-matrix** at `log_h ∈ {14, 16, 18, 20}`, `w = 8` — sanity rows, direct comparison to the W24-leaf `zkgpu-plonky3` bench at the same nominal heights.
- **Mixed-height** at `log_h_max ∈ {14, 16, 18}` — trace at `h_max` with `w = 8` plus 4 quotient-chunk-shaped matrices at `h_max / 2` with `w = 2`. Mirrors OpenVM's commit_quotient-style injection pattern. **Headline shape**.

## Results

### `target_stack/commit` — primary gate

#### M4 Pro / Metal (Gate 4a — blocking)

| Shape | log_h / log_h_max | CPU | GPU | Ratio |
|---|---:|---:|---:|---:|
| single | 14 | 4.65 ms | 24.11 ms | 0.19× |
| single | 16 | 18.30 ms | 35.42 ms | 0.52× |
| single | 18 | 73.21 ms | 75.37 ms | 0.97× |
| **single** | **20** | **292.58 ms** | **203.78 ms** | **1.44×** |
| mixed | 14 | 6.98 ms | 31.79 ms | 0.22× |
| mixed | 16 | 27.63 ms | 46.18 ms | 0.60× |
| **mixed** | **18** | **110.62 ms** | **101.46 ms** | **1.09×** |

Gate 4a outcome: **cleared.** The portability claim holds on Apple Silicon at the headline mixed-height shape (1.09× at log_h_max=18) and widens at larger sizes (1.44× at single log_h=20). At `log_h ≤ 16` the GPU loses — launch overhead dominates; this is expected and documented.

#### RTX 4090 + Ryzen 9 7950X (Gate 4b — informational)

| Shape | log_h / log_h_max | CPU | GPU | Ratio |
|---|---:|---:|---:|---:|
| single | 14 | 25.07 ms | 5.33 ms | 4.71× |
| single | 16 | 102.55 ms | 10.21 ms | 10.04× |
| single | 18 | 411.64 ms | 44.78 ms | 9.19× |
| single | 20 | 1644.0 ms | 231.00 ms | 7.12× |
| mixed | 14 | 37.97 ms | 6.33 ms | 6.00× |
| mixed | 16 | 151.36 ms | 9.59 ms | **15.78×** |
| mixed | 18 | 605.14 ms | 59.66 ms | **10.14×** |

#### RTX 5090 + Ryzen 9 9950X (Gate 4b — informational)

| Shape | log_h / log_h_max | CPU | GPU | Ratio |
|---|---:|---:|---:|---:|
| single | 14 | 19.94 ms | 4.36 ms | 4.57× |
| single | 16 | 79.95 ms | 6.74 ms | 11.86× |
| single | 18 | 321.63 ms | 17.61 ms | 18.26× |
| single | 20 | 1288.1 ms | 74.74 ms | 17.23× |
| mixed | 14 | 30.00 ms | 5.14 ms | 5.84× |
| mixed | 16 | 120.18 ms | 7.61 ms | **15.78×** |
| **mixed** | **18** | **481.50 ms** | **23.99 ms** | **🟢 20.07×** |

### `target_stack/commit_open_40q` — FRI-workload shape

Numbers nearly identical to `commit` across all three hosts — the 40 host-side opens (retained-layer indexing + Plonky3 Monty conversion) cost is small relative to the GPU commit. Full tables in the raw bench logs.

### Host-to-host comparison at the headline shape

Mixed-height `commit` at `log_h_max = 18`:

| | CPU (ms) | GPU (ms) | ratio |
|---|---:|---:|---:|
| M4 Pro | 110.62 | 101.46 | 1.09× |
| RTX 4090 + 7950X | 605.14 | 59.66 | 10.14× |
| RTX 5090 + 9950X | 481.50 | 23.99 | 20.07× |

Observations:

- **Apple Silicon's CPU is unusually strong per-core.** M4 Pro CPU is faster than RTX 4090 host's 7950X at this shape (110 ms vs 605 ms), despite having fewer cores — the Apple performance cores run Poseidon2 at near-AVX-512 throughput. That's why the Metal win is modest: Apple's integrated GPU is competitive with Apple's own CPU, not with a 32-core Zen 4/5 server chip.
- **NVIDIA wins get larger with newer silicon.** 4090 → 5090 takes the ratio from 10.14× to 20.07× — the Blackwell GPU widens the gap faster than the Zen 5 CPU closes it. Consistent with the trend seen in [`plonky3-poseidon2-mmcs.md`](plonky3-poseidon2-mmcs.md).
- **The mixed-height shape benefits more than single-matrix** on NVIDIA (15.78× at mixed log_h=16 vs 10.04× at single log_h=16). Injection levels add CPU work (extra leaf hashes + second-compression passes) but GPU absorbs them cheaply via the existing leaf-sponge + compress kernels.

## Delta vs the Plonky3 consumer's claim

The sibling note ([`plonky3-poseidon2-mmcs.md`](plonky3-poseidon2-mmcs.md)) reports for Plonky3's canonical W24 leaf + W16 compress shape at RTX 5090 + 9950X:

- `fri_commit @ log_h=18, w=8`: 16.20× (commit only, single matrix)
- `prove+verify @ FibAir log_h=18`: 4.63× (end-to-end prove)

This note's comparable number is **20.07×** at `log_h_max=18` mixed-height on the same 5090+9950X host. The lift comes from:

1. **Mixed-height DAG exercises more Poseidon2 per commit** than single-matrix. Injection levels require an extra row hash per injected matrix + a second compression per output — all GPU-cheap, all CPU-expensive.
2. **OpenVM's W16 leaf is less GPU-friendly per row than Plonky3's W24 leaf** — fewer elements absorbed per permutation, so the GPU's parallelism advantage is smaller. Net positive on this shape because the mixed-height extra-work effect dominates.

Neither ratio is directly better than the other; they measure different consumers' hot paths.

## Correctness evidence

The numbers above are only meaningful because the adapter's commit/open/verify semantics are bit-identical to Plonky3's CPU reference. Locked by:

- **Commit parity** ([commit_parity.rs](../../crates/zkgpu-openvm/tests/commit_parity.rs), 9 tests): GPU commit roots match CPU `MerkleTreeMmcs::commit` across single-matrix at 6 heights, same-height multi-matrix, mixed-height 2/3/5-level, multi-matrices-at-non-max-height, and bench-shape proxy.
- **Open + verify parity** ([open_verify_parity.rs](../../crates/zkgpu-openvm/tests/open_verify_parity.rs), 11 tests): per-index `(opened_values, opening_proof)` match CPU bit-for-bit, AND GPU openings pass through both the adapter's own `verify_batch` and CPU `MerkleTreeMmcs::verify_batch` (cross-verifier roundtrip). Plus negative tests for tampered openings and wrong-index verification.
- **Backend-layer parity** (in the zkgpu-plonky3 test tree, because that's where the Plonky3 0.5.x reference is convenient): the mixed-height DAG engine is parity-pinned through both leaf shapes (W24 and W16), 28 tests total.

## Current limitations

- `cap_height > 0` rejected at `OpenVmGpuMmcs::new`.
- Not yet integrated into a full OpenVM prove/verify end-to-end test. The `Mmcs` trait surface is complete at Plonky3 0.4.1; plugging into OpenVM's own `StarkConfig` / `TwoAdicFriPcs` pipelines is straightforward but hasn't been exercised in-tree.
- `verify_batch` delegates to CPU `MerkleTreeMmcs`. This is correct (the commitment + proof types are identical to the CPU adapter's output) but not GPU-accelerated.
- Numerical results measured on three GPUs: Apple M4 Pro (Metal), NVIDIA RTX 4090 (Vulkan via wgpu), NVIDIA RTX 5090 (Vulkan via wgpu). The backend is portable to AMD / DX12 / WebGPU; no published benchmark numbers on those targets yet.

## Reproducing

```bash
# On any host with a GPU + wgpu-compatible backend:
cd /workspace/zkgpu
cargo test -p zkgpu-openvm --tests                                # 20 parity tests
cargo bench -p zkgpu-openvm --bench openvm_commit                 # ~10-20 min
```

Parity suites should print `test result: ok. N passed` for each of `commit_parity` (9 tests) and `open_verify_parity` (11 tests). Bench median times should agree with the tables above to within ±5% on the same hardware class.

## External summary

> `zkgpu_openvm::OpenVmGpuMmcs` is a portable GPU backend for OpenVM's BabyBear Poseidon2 MMCS, bit-compatible with OpenVM's CPU reference at every public MMCS operation. On Apple Silicon (Metal), it clears the portability gate with a 1.09× commit-time speedup at `log_h_max = 18` on the mixed-height trace+quotient shape — a configuration OpenVM's own CUDA backend cannot run. On discrete NVIDIA (RTX 5090 + Ryzen 9 9950X), the same adapter reaches 20.07× at the same shape via Vulkan through wgpu, informationally alongside what OpenVM's first-party CUDA backend offers on NVIDIA-only.

## Cross-references

- Sibling consumer note: [`plonky3-poseidon2-mmcs.md`](plonky3-poseidon2-mmcs.md)
- OpenVM canonical Poseidon2 config: [`baby_bear_poseidon2.rs`](https://github.com/openvm-org/stark-backend/blob/main/crates/stark-sdk/src/config/baby_bear_poseidon2.rs)
- OpenVM `VERIFY_BATCH` mixed-height MMCS spec: [native Poseidon2 README](https://github.com/openvm-org/openvm/blob/main/extensions/native/circuit/src/poseidon2/README.md)
- OpenVM's CUDA backend (non-portable alternative): [`cuda-backend`](https://github.com/openvm-org/stark-backend/tree/main/crates/cuda-backend)
