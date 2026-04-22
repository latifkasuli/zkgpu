//! GPU Poseidon2 permutation plans.
//!
//! Phase F.1 deliverable: the first non-NTT primitive GPU kernel in
//! zkgpu. Follows the [`crate::ntt`] module shape — a per-field plan
//! uploads round constants + internal diagonal once at construction
//! and reuses them across `execute` calls.
//!
//! # Status
//!
//! - [`WgpuBabyBearPoseidon2Plan`] — portable WGSL kernel, one thread
//!   per permutation instance, width hardcoded to
//!   [`zkgpu_poseidon2::WIDTH`] (= 16). Validated against the CPU
//!   reference in [`zkgpu_poseidon2::Poseidon2`]: bit-parity on the
//!   `babybear_regression_state_0001` anchor plus batch
//!   differential tests over random inputs.
//! - [`WgpuGoldilocksPoseidon2Plan`] — Phase F.2 portable u32x2
//!   Goldilocks twin. Same structure, `vec2<u32>` limbs via
//!   `goldilocks_arith_helpers.wgsl` prelude; pinned against
//!   `goldilocks_regression_state_0001`.
//! - [`WgpuBabyBearPoseidon2PlonkyW16Plan`],
//!   [`WgpuBabyBearPoseidon2PlonkyW24Plan`] — Phase 7 Step 1.5b
//!   Plonky3-variant twins (M_4 = circ(2,3,1,1)). Consumed by Plonky3's
//!   `Poseidon2MerkleMmcs` via the `zkgpu-plonky3::poseidon2_bridge`
//!   adapter.
//! - [`WgpuPoseidon2MerkleLeafPlan`] — Phase 7 Step 3.a GPU leaf
//!   sponge: runs `PaddingFreeSponge<Perm24, 24, 16, 8>` over each row
//!   of a matrix in one dispatch. Packed-constants BGL (3 storage + 1
//!   uniform) to fit the WebGPU baseline 4-storage cap.
//! - [`WgpuPoseidon2MerkleCompressPlan`],
//!   [`WgpuPoseidon2MerkleCommit`] — Phase 7 Step 3.b tree-compression
//!   kernel (width-16 `TruncatedPermutation`) and the commit
//!   orchestrator that chains leaf-sponge → `log₂(h)` compression
//!   levels → 8-element root. Single-matrix, power-of-two-h scope
//!   (Plonky3's `TwoAdicFriPcs::commit` shape).
//!
//! Not yet:
//! - Multi-matrix / non-power-of-two-h merkle commits (Plonky3's
//!   `compress_and_inject`; not needed for the Step 3 bench gate).
//! - Testkit / CLI / web harness wiring for Poseidon2 suites
//!   (Phase F.3).
//!
//! # Batch model
//!
//! The kernel processes a flat `Vec<BabyBear>` of length
//! `num_permutations * WIDTH`, where each `WIDTH`-element run is one
//! independent permutation instance. One GPU thread owns one
//! instance: loads its 16 state slots, runs all rounds, writes back.
//! No inter-thread synchronisation — the permutation is a pure
//! per-thread loop. Dispatch grid is 2D-folded via
//! [`crate::dispatch::plan_linear_dispatch`] so very large batches
//! (≥ 65535 × 64 = ~4.2M on WebGPU baseline) respect
//! `max_compute_workgroups_per_dimension`.

mod goldilocks_plan;
mod merkle_commit;
mod merkle_compress;
mod merkle_leaf;
mod plan;
mod plonky3_plan;

pub use goldilocks_plan::WgpuGoldilocksPoseidon2Plan;
pub use merkle_commit::WgpuPoseidon2MerkleCommit;
pub use merkle_compress::WgpuPoseidon2MerkleCompressPlan;
pub use merkle_leaf::{WgpuPoseidon2MerkleLeafPlan, DIGEST_LEN as MERKLE_DIGEST_LEN};
pub use plan::WgpuBabyBearPoseidon2Plan;
pub use plonky3_plan::{
    WgpuBabyBearPoseidon2PlonkyW16Plan, WgpuBabyBearPoseidon2PlonkyW24Plan,
};
