# zkgpu

Open, cross-platform GPU-accelerated cryptographic primitives for zero-knowledge proofs.

## Status

**Pre-alpha.** Not ready for production use.

## What this is

A Rust library that runs finite field arithmetic, NTT, and (soon) MSM on client-side GPUs via [wgpu](https://github.com/gfx-rs/wgpu). A single codebase targets:

- **Metal** (macOS / iOS)
- **Vulkan** (Linux / Android)
- **DX12** (Windows)
- **WebGPU** (browsers, experimental)

## Why

There is no open, standard, broadly adopted client-side GPU crypto library for ZK. Existing solutions are either closed-source/commercially licensed for production, or tightly coupled to a single proof system. zkgpu aims to be the reusable base layer that any proving stack can depend on.

## Crates

| Crate | Description |
|---|---|
| `zkgpu-core` | Backend-agnostic traits and types |
| `zkgpu-wgpu` | wgpu backend (Metal, Vulkan, DX12, WebGPU) |
| `zkgpu-babybear` | BabyBear field (p = 2^31 - 2^27 + 1) |
| `zkgpu-ntt` | Number Theoretic Transform |
| `zkgpu-plonky3` | Plonky3 adapter (planned) |
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
- [x] WGSL compute shader for BabyBear NTT butterfly
- [x] GPU NTT validated against CPU reference (forward, inverse, roundtrip)
- [x] Batched command submission (one encoder, one submit for all stages)
- [x] Capability-tiered runtime (CapabilityProfile, DeviceTier)
- [x] GPU timestamp profiling
- [x] Stockham autosort NTT (radix-2 DIF, no CPU bit-reversal)
- [x] Workgroup-local blocked Stockham NTT (9 stages fused in shared memory)
- [ ] Benchmark against Plonky3 CPU NTT
- [ ] MSM (multi-scalar multiplication)
- [ ] Goldilocks field
- [ ] BN254 scalar field
- [ ] Plonky3 DFT adapter
- [ ] Browser/WASM support

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.
