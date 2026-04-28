# Two consumers on one GPU backend

> **Status: pre-alpha.** As of 2026-04-28 (v0.2).
>
> **v0.2 update (2026-04-28).** The mixed-height commit DAG now runs entirely device-resident — the previous host-bouncing pattern at injection levels (download intermediate digests, host-side interleave, re-upload, second compress) is gone. On a same-host A/B (RTX 5090 + Ryzen 9 9900X), the GPU mixed-height commit time at `log_h_max=18` reduced from 25.33 ms to 14.47 ms (-43%). Per-consumer details and the methodology note about the discrete-GPU clocking floor are in [`docs/research/openvm-poseidon2-mmcs.md`'s v0.2 section](research/openvm-poseidon2-mmcs.md#v02-2026-04-28--gpu-resident-mixed-height-injection). Parity validated on both Metal and Vulkan/NVIDIA. The headline-ratio table below is from the v0.1 measurement window and uses different vast.ai host configurations than the v0.2 numbers; we're keeping it as a historical reference rather than rewriting it under shifting host baselines.

## Claim

Two production STARK consumer codebases — **Plonky3** (BabyBear / Poseidon2 MMCS at Plonky3 0.5.x) and **OpenVM** (the same MMCS shape pinned at Plonky3 0.4.1, W16 leaf) — both run their Merkle commit phase on a single shared GPU backend through this repository. The same engine handles every shape either consumer's `Mmcs::commit` validly accepts at `cap_height = 0`: single-matrix, same-height multi-matrix, and mixed-height DAG (`compress_and_inject`). Output is bit-identical to each consumer's own CPU `MerkleTreeMmcs` reference under the same Poseidon2 constants.

Speedup at the trace-commit shape each consumer actually uses:

| Host | Plonky3 `fri_commit` log_h=18 | Plonky3 `prove+verify` log_h=18 | OpenVM `commit` mixed-height log_h_max=18 |
|---|---:|---:|---:|
| Apple M4 Pro / Metal | — | — | **1.09×** |
| RTX 4090 + Ryzen 9 7950X | **11.46×** | **3.98×** | **10.14×** |
| RTX 5090 + Ryzen 9 9950X | **15.07×** | **4.76×** | **20.07×** |

Plonky3 numbers measured on commit `26646fc`. OpenVM numbers from the OpenVM research note.

## Why this matters

The MMCS-layer wins above are real and measured, but they aren't the point on their own. The point is that **the same GPU backend code is the bottom of both adapters**:

```text
        ┌──────────────────────┐    ┌──────────────────────┐
        │   zkgpu-plonky3      │    │    zkgpu-openvm      │
        │  Plonky3 0.5.x       │    │   Plonky3 0.4.1      │
        │  W24 leaf sponge     │    │   W16 leaf sponge    │
        └──────────┬───────────┘    └───────────┬──────────┘
                   │                            │
                   ▼                            ▼
        ┌──────────────────────────────────────────────────┐
        │  zkgpu-wgpu  —  mixed-height DAG engine          │
        │  commit_mixed_height_with_{w24,w16}_leaf         │
        │  open_batch_mixed_height                         │
        │  + same-height fast path (no host round-trips)   │
        └────────────────────────┬─────────────────────────┘
                                 ▼
        ┌──────────────────────────────────────────────────┐
        │  wgpu  →  Metal / Vulkan / DX12 / WebGPU         │
        └──────────────────────────────────────────────────┘
```

The two version lanes (Plonky3 0.5.x and 0.4.1) coexist as separate adapter crates because OpenVM's workspace pins Plonky3 at exactly 0.4.1; forcing them to share one pin would push one ecosystem off its current dependency surface. Cargo resolves both trees side-by-side. The shared backend itself is Plonky3-version-independent — it consumes flattened `Vec<BabyBear>` matrices in canonical form, not Plonky3 types.

## What's portable

`zkgpu-wgpu` is GPU-agnostic via wgpu. The same source compiles for Apple Silicon (Metal), discrete AMD / Intel / NVIDIA via Vulkan, Windows via DX12, and browsers via WebGPU (experimental). This is particularly relevant for OpenVM: the official `openvm-cuda-backend` accelerates the same MMCS on NVIDIA via native CUDA. `zkgpu-openvm` is **not** competing with that on NVIDIA — the NVIDIA numbers are reported informationally. It's the **portable** path — Apple, AMD, mobile, browser — for users whose hardware can't run CUDA. The Metal `1.09×` row is the portability gate.

## What's parity-pinned

- Commit roots match each consumer's CPU `MerkleTreeMmcs::commit` byte-for-byte.
- Opening proofs match the CPU reference at every supplied index, across single-matrix, same-height multi-matrix, and mixed-height DAG shapes.
- GPU-produced openings verify through both the adapter's own `verify_batch` and a freshly-built CPU `MerkleTreeMmcs::verify_batch` over the GPU commit (cross-verifier roundtrip).
- End-to-end `prove_then_verify` on OpenVM's own dummy Fibonacci AIR runs through both a CPU control engine and a GPU-swapped engine, with the GPU proof round-tripping cleanly.

Total test surface: **~138 tests** across the two consumer crates (~116 on `zkgpu-plonky3`, 22 on `zkgpu-openvm`), plus the backend-level mixed-height commit + open parity suites covering both leaf shapes.

## What we don't claim

- *"GPU always wins."* At small `log_h` on Apple Silicon, launch overhead dominates and the CPU is faster. Documented per-host in the detail notes.
- A drop-in for `cap_height > 0`. Both adapters reject it at construction.
- Acceleration of the verifier path. `verify_batch` delegates to CPU `MerkleTreeMmcs`.
- A result for non-Plonky3-shape proving systems (Keccak MMCS, sum-check zkVMs, Halo2/Groth16, etc.). Those would be additional adapter crates, not this one.
- Mixed-height **performance** for Plonky3 specifically. Mixed-height *semantics* are parity-validated for Plonky3 (4 dedicated tests at differing-height shapes); the 20.07× mixed-height number above is OpenVM's W16 measurement and shouldn't be borrowed across leaf shapes — per-row Poseidon2 cost differs.

## Reproducing

```bash
git clone https://github.com/latifkasuli/zkgpu && cd zkgpu

# Parity tests (no GPU available = GPU bodies skip with a notice)
cargo test -p zkgpu-plonky3 --tests
cargo test -p zkgpu-openvm  --tests

# Plonky3 prover hot path
cargo bench -p zkgpu-plonky3 --bench prover_hot_path

# OpenVM MMCS-layer + end-to-end prove benches
cargo bench -p zkgpu-openvm --bench openvm_commit
cargo bench -p zkgpu-openvm --bench openvm_prove
```

## Detailed reading

- [`docs/research/plonky3-poseidon2-mmcs.md`](research/plonky3-poseidon2-mmcs.md) — full Plonky3 narrow + mixed-height claim, the NVIDIA scale-up history, and the post-convergence recovery story (regression + same-height fast path that closed it).
- [`docs/research/openvm-poseidon2-mmcs.md`](research/openvm-poseidon2-mmcs.md) — full OpenVM mixed-height headline claim, the three-host portability table, and the end-to-end `prove_then_verify` integration through OpenVM's `StarkConfig`.
