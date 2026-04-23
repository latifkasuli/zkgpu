//! # zkgpu-openvm — GPU Poseidon2 MMCS adapter for OpenVM
//!
//! Second consumer adapter for the zkgpu shared Merkle commit
//! backend, targeting [OpenVM](https://github.com/openvm-org/openvm)'s
//! BabyBear Poseidon2 MMCS. Sibling to `zkgpu-plonky3`; both drive
//! the same GPU primitive layer but are pinned to different
//! Plonky3 versions (OpenVM: `0.4.1`, zkgpu-plonky3: `0.5.x`).
//!
//! ## Scope
//!
//! * Field: BabyBear
//! * Poseidon2 shape: `WIDTH = 16`, `RATE = 8`, `DIGEST_WIDTH = 8`
//!   (OpenVM's canonical config — single `Perm16` instance for
//!   both leaf sponge and tree compression)
//! * MMCS: `MerkleTreeMmcs<Packing, Packing, PaddingFreeSponge<…>,
//!   TruncatedPermutation<…>, 2, 8>` (N=2 binary tree)
//! * Cap height: `0` (commitment is a single root digest)
//! * Mixed-height commits supported from day one via the shared
//!   `compress_and_inject`-style DAG engine in
//!   [`zkgpu_wgpu::commit_mixed_height_with_w16_leaf`].
//!
//! ## Positioning relative to OpenVM's own CUDA backend
//!
//! OpenVM ships a first-party `openvm-cuda-backend` crate that
//! accelerates the same Poseidon2 STARK config on NVIDIA GPUs. This
//! adapter is **not** trying to outperform that on CUDA — it's the
//! **portable** GPU path (Metal, Vulkan, DX12, WebGPU via wgpu) for
//! OpenVM users who can't use CUDA: Apple silicon, AMD, mobile,
//! browser. The value prop is breadth, not peak NVIDIA throughput.
//!
//! ## Dependency strategy
//!
//! The crate pins Plonky3 at `=0.4.1` to match OpenVM's own
//! workspace pin. The library itself (`src/*`) rebuilds OpenVM's
//! config aliases locally from Plonky3 0.4.1 primitives (see
//! [`config`]) rather than depending on `openvm-stark-backend` —
//! the aliases are ~30 lines.
//!
//! `openvm-stark-sdk` + `openvm-stark-backend` are pulled in as
//! **dev-dependencies** only (for the end-to-end prove/verify test
//! and bench). The library API surface has no OpenVM-crate
//! dependency, so downstream consumers who only want the MMCS
//! adapter don't pay the full OpenVM dependency-tree cost.
//!
//! ## What's shipped
//!
//! * `commit`, `open_batch`, `verify_batch` — full Plonky3 0.4.1
//!   `Mmcs` trait impl plus direct inherent methods.
//! * Mixed-height commits (single / same-height multi / mixed-height
//!   DAG) through the shared `commit_mixed_height_with_w16_leaf`
//!   engine.
//! * Parity tests: 9 commit-root parity cases, 11 open/verify parity
//!   cases, 2 end-to-end prove/verify cases (CPU control +
//!   GPU-swapped OpenVM engine).
//! * Benches: MMCS-layer `target_stack/commit` +
//!   `target_stack/commit_open_40q`, end-to-end
//!   `target_stack/prove/fib_air` on OpenVM's dummy Fibonacci AIR.
//!
//! See [`docs/research/openvm-poseidon2-mmcs.md`](../../docs/research/openvm-poseidon2-mmcs.md)
//! for the numerical results and positioning.

pub mod bridge;
pub mod config;
pub mod gpu_mmcs;

pub use bridge::{
    babybear_openvm_params, p3_array_to_zkgpu, p3_to_zkgpu, zkgpu_array_to_p3, zkgpu_to_p3,
};
pub use config::{
    Commitment, Compress, Error, LeafHash, Perm, Proof, Val, DIGEST_WIDTH, RATE,
    SUPPORTED_CAP_HEIGHT, WIDTH,
};
pub use gpu_mmcs::{GpuProverData, OpenVmGpuMmcs};
