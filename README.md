# zkgpu

Open, cross-platform GPU-accelerated cryptographic primitives for zero-knowledge proofs.

## Status

**Pre-alpha.** Not ready for production use.

## What this is

A Rust library that runs finite field arithmetic, NTT, Poseidon2, and Merkle commits on client-side GPUs via [wgpu](https://github.com/gfx-rs/wgpu). A single codebase targets:

- **Metal** (macOS / iOS)
- **Vulkan** (Linux / Android)
- **DX12** (Windows)
- **WebGPU** (browsers, experimental)

## Why

There is no open, standard, broadly adopted client-side GPU crypto library for ZK. Existing solutions are either closed-source / commercially licensed for production, or tightly coupled to a single proof system. zkgpu aims to be the reusable base layer that any proving stack can depend on.

## Crates

| Crate | Description |
|---|---|
| `zkgpu-core` | Backend-agnostic traits and types |
| `zkgpu-wgpu` | wgpu backend (Metal / Vulkan / DX12 / WebGPU) |
| `zkgpu-babybear` | BabyBear field (p = 2^31 − 2^27 + 1) |
| `zkgpu-koalabear` | KoalaBear field (p = 2^31 − 2^24 + 1) |
| `zkgpu-goldilocks` | Goldilocks field (p = 2^64 − 2^32 + 1) |
| `zkgpu-ntt` | Number Theoretic Transform |
| `zkgpu-poseidon2` | Poseidon2 permutation (field-parametric reference) |
| `zkgpu-plonky3` | Plonky3 adapter (`TwoAdicSubgroupDft` + Poseidon2 MMCS) |
| `zkgpu-testkit` | Shared validation + benchmark suite |
| `zkgpu-report` | Report / spec types shared by native + browser harnesses |
| `zkgpu-pacer` | Adaptive pacing for sustained GPU workloads |
| `zkgpu-web` | WebAssembly bindings for the browser harness |
| `zkgpu-ffi` | C ABI shim for external harnesses |
| `zkgpu-tail-analyze` | Logcat parser / threshold tool for Stockham tail A/B |
| `zkgpu-cli` | Benchmark and debug utility |

## Quick start

```bash
# Run the benchmark CLI
cargo run -p zkgpu-cli --release

# Run tests
cargo test --workspace

# Run BabyBear field tests
cargo test -p zkgpu-babybear
```

## Roadmap

- [x] Project structure and core traits
- [x] BabyBear field with CPU reference
- [x] KoalaBear field
- [x] Goldilocks field
- [x] WGSL compute shader for NTT butterfly
- [x] GPU NTT validated against CPU reference (forward, inverse, roundtrip)
- [x] Batched command submission (one encoder, one submit for all stages)
- [x] Capability-tiered runtime (`CapabilityProfile`, `DeviceTier`, driver quirks)
- [x] GPU timestamp profiling
- [x] Stockham autosort NTT (radix-2 DIF, no CPU bit-reversal)
- [x] Workgroup-local blocked Stockham NTT (stages fused in shared memory)
- [x] Four-Step NTT for large `log_n` on discrete GPU
- [x] Poseidon2 permutation (BabyBear widths 16 and 24, Goldilocks)
- [x] GPU Poseidon2 Merkle commit (leaf sponge, tree compression, retained layers for openings)
- [x] Benchmark harness against Plonky3 CPU NTT and CPU Poseidon2 MMCS
- [x] Plonky3 adapter: `GpuDft<BabyBear>` and `GpuPoseidon2Mmcs`
- [x] FFI and WebAssembly entry points for external harnesses
- [ ] MSM (multi-scalar multiplication)
- [ ] BN254 scalar field
- [ ] Stable browser / WebGPU support (experimental today)

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.
