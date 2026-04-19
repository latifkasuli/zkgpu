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
//!
//! Not yet:
//! - Goldilocks GPU Poseidon2 (Phase F.2 — portable u32x2 limb
//!   variant, same shape as the NTT `WgpuGoldilocksNttPlan`).
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

mod plan;

pub use plan::WgpuBabyBearPoseidon2Plan;
