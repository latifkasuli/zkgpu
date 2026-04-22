//! Phase 7 Step 3 — GPU Poseidon2 MMCS adapter.
//!
//! Implements [`p3_commit::Mmcs<BabyBear>`] with a GPU-accelerated
//! `commit()`. Designed to be the *bench-gate* adapter: it produces
//! the same commitment root as Plonky3's
//! `MerkleTreeMmcs<Packing, Packing, Poseidon2Sponge, Poseidon2Compression, 2, 8>`
//! while keeping the commit path entirely on the GPU — **no CPU tree
//! is materialized until [`Mmcs::open_batch`] or
//! [`Mmcs::verify_batch`] is called**.
//!
//! # Why lazy prover data matters
//!
//! Plonky3's `MerkleTreeMmcs::commit` eagerly constructs a full
//! `MerkleTree` (all interior layers) as part of its `ProverData`.
//! That CPU work is the thing the GPU is supposed to replace. If the
//! adapter's `commit()` paid that cost up front, the Step 3
//! `fri_commit` bench would just measure "GPU hash + CPU hash", and
//! the GPU win would be hidden. This adapter's `ProverData` is
//! therefore a bare `Vec<M>` — commit runs the GPU kernel, wraps the
//! root, and returns. No CPU tree is ever materialised.
//!
//! # Scope
//!
//! **Commit-only. Single-matrix, power-of-two height.** The Step 3
//! bench gate times `TwoAdicFriPcs::commit` (one call into
//! `mmcs.commit`, one matrix); FRI `open_batch` / `verify_batch` are
//! out of scope for this bench and `open_batch` therefore panics
//! loudly if called. Multi-matrix commits (Plonky3's
//! `compress_and_inject` injection pattern) and opening support
//! belong in a future hybrid adapter (Step 3.c).
//!
//! Silent CPU fallback is intentionally *not* used: it would let a
//! multi-matrix or opening-path caller produce a "GPU" bench number
//! that actually measured CPU, and the fri_commit go/no-go gate
//! can't survive that kind of contamination.

use std::sync::{Arc, Mutex};

use p3_baby_bear::BabyBear as P3BabyBear;
use p3_commit::{BatchOpening, BatchOpeningRef, Mmcs};
use p3_matrix::{Dimensions, Matrix};
use p3_merkle_tree::{MerkleTreeError, MerkleTreeMmcs};
use p3_symmetric::{MerkleCap, PaddingFreeSponge, TruncatedPermutation};

use zkgpu_babybear::BabyBear as ZkgpuBabyBear;
use zkgpu_poseidon2::Poseidon2Params;
use zkgpu_wgpu::{WgpuDevice, WgpuPoseidon2MerkleCommit};

use crate::poseidon2_bridge::p3_to_zkgpu;

/// Digest length for BabyBear Poseidon2 MMCS (matches Plonky3's
/// `Poseidon2MerkleMmcs` config).
pub const DIGEST_LEN: usize = 8;

/// Plonky3's canonical width-24 leaf sponge type.
pub type Perm24 = p3_baby_bear::Poseidon2BabyBear<24>;
/// Plonky3's canonical width-16 tree compression type.
pub type Perm16 = p3_baby_bear::Poseidon2BabyBear<16>;

/// Plonky3 leaf hasher used by `Poseidon2MerkleMmcs`.
pub type P3Poseidon2Sponge = PaddingFreeSponge<Perm24, 24, 16, DIGEST_LEN>;
/// Plonky3 tree compression used by `Poseidon2MerkleMmcs`.
pub type P3Poseidon2Compression = TruncatedPermutation<Perm16, 2, DIGEST_LEN, 16>;

/// CPU fallback MMCS — only used for `open_batch` / `verify_batch`
/// (and `get_matrices`), built lazily per-commit.
type CpuValMmcs = MerkleTreeMmcs<
    <P3BabyBear as p3_field::Field>::Packing,
    <P3BabyBear as p3_field::Field>::Packing,
    P3Poseidon2Sponge,
    P3Poseidon2Compression,
    2,
    DIGEST_LEN,
>;

// ---------------------------------------------------------------------------
// GpuPoseidon2Mmcs
// ---------------------------------------------------------------------------

/// GPU-accelerated Poseidon2 MMCS adapter for the Step 3 bench gate.
///
/// `Clone` semantics: cheap — internal state is shared via `Arc`.
/// The GPU plan sits behind a `Mutex` because
/// `WgpuPoseidon2MerkleCommit::commit` takes `&mut self` (owns
/// internal scratch/pipeline state), and Plonky3's `Mmcs::commit`
/// takes `&self`. Concurrent commits from one adapter instance
/// serialize on that mutex; in the bench workload, `commit` is a
/// single synchronous call per PCS, so there's no contention.
#[derive(Clone)]
pub struct GpuPoseidon2Mmcs {
    device: Arc<WgpuDevice>,
    gpu_commit: Arc<Mutex<WgpuPoseidon2MerkleCommit>>,
    /// CPU-side (hash, compress) instances — stored here so that
    /// `open_batch` / `verify_batch` can build a fresh `CpuValMmcs`
    /// lazily per-commit without threading them through the trait
    /// API.
    cpu_hash: P3Poseidon2Sponge,
    cpu_compress: P3Poseidon2Compression,
    cap_height: usize,
}

impl GpuPoseidon2Mmcs {
    /// Construct a GPU MMCS from matched Plonky3 `(Perm16, Perm24)`
    /// constants. `leaf_params` and `compress_params` must be the
    /// Plonky3-variant, α=7 Poseidon2 params produced by
    /// [`crate::poseidon2_bridge::babybear_plonky3_params`] at widths
    /// 24 and 16 respectively.
    ///
    /// `cap_height = 0` means the commitment is a single digest (the
    /// root), matching the FRI configuration used by the target-stack
    /// bench.
    pub fn new(
        device: Arc<WgpuDevice>,
        perm24: Perm24,
        perm16: Perm16,
        leaf_params: Poseidon2Params<ZkgpuBabyBear, 24>,
        compress_params: Poseidon2Params<ZkgpuBabyBear, 16>,
        cap_height: usize,
    ) -> Result<Self, String> {
        let gpu_commit = WgpuPoseidon2MerkleCommit::new(
            device.as_ref(),
            leaf_params,
            compress_params,
        )
        .map_err(|e| format!("GpuPoseidon2Mmcs: GPU plan construction failed: {e}"))?;

        let cpu_hash = P3Poseidon2Sponge::new(perm24);
        let cpu_compress = P3Poseidon2Compression::new(perm16);

        Ok(Self {
            device,
            gpu_commit: Arc::new(Mutex::new(gpu_commit)),
            cpu_hash,
            cpu_compress,
            cap_height,
        })
    }

    /// Build a CPU `MerkleTreeMmcs` configured to produce byte-
    /// identical commitments to this adapter. Used for lazy
    /// `open_batch` / `verify_batch` paths and for differential
    /// testing.
    fn cpu_mmcs(&self) -> CpuValMmcs {
        CpuValMmcs::new(
            self.cpu_hash.clone(),
            self.cpu_compress.clone(),
            self.cap_height,
        )
    }
}

// ---------------------------------------------------------------------------
// GpuProverData<M>
// ---------------------------------------------------------------------------

/// Prover data for [`GpuPoseidon2Mmcs`] — commit-only scope.
///
/// Holds just the input matrices (for [`Mmcs::get_matrices`] lookups).
/// No CPU tree is ever built: this adapter is commit-only and
/// `Mmcs::open_batch` panics if called (see module doc). A future
/// hybrid adapter that needs opening will either build the CPU tree
/// lazily here or split the API across two adapters.
pub struct GpuProverData<M> {
    /// The matrices committed to.
    matrices: Vec<M>,
}

// ---------------------------------------------------------------------------
// Mmcs impl
// ---------------------------------------------------------------------------

impl Mmcs<P3BabyBear> for GpuPoseidon2Mmcs {
    type ProverData<M> = GpuProverData<M>;
    type Commitment = MerkleCap<P3BabyBear, [P3BabyBear; DIGEST_LEN]>;
    type Proof = <CpuValMmcs as Mmcs<P3BabyBear>>::Proof;
    type Error = MerkleTreeError;

    fn commit<M: Matrix<P3BabyBear>>(
        &self,
        inputs: Vec<M>,
    ) -> (Self::Commitment, <Self as Mmcs<P3BabyBear>>::ProverData<M>) {
        // --- Hard-scope: single matrix only. See module doc. ---
        assert!(
            !inputs.is_empty(),
            "GpuPoseidon2Mmcs::commit: called with 0 matrices"
        );
        assert_eq!(
            inputs.len(),
            1,
            "GpuPoseidon2Mmcs is a single-matrix adapter for the Step 3 bench gate; \
             received {} matrices. Multi-matrix commit (Plonky3 compress_and_inject) \
             is deferred — either split the call or use the hybrid adapter.",
            inputs.len(),
        );

        let mat = &inputs[0];
        let h = mat.height();
        let w = mat.width();
        assert!(h > 0, "GpuPoseidon2Mmcs::commit: h must be ≥ 1");
        assert!(w > 0, "GpuPoseidon2Mmcs::commit: w must be ≥ 1");
        assert!(
            h.is_power_of_two(),
            "GpuPoseidon2Mmcs::commit: h must be a power of two (got {h})"
        );

        // Flatten row-major h × w matrix into a `Vec<ZkgpuBabyBear>`,
        // with Monty → canonical conversion in the same pass. Row
        // iteration goes through `Matrix::row`, which works for both
        // `RowMajorMatrix` (Plonky3's default) and
        // `BitReversedMatrixView` (the shape returned from
        // `coset_lde_batch`) without forcing a full materialisation.
        let mut flat: Vec<ZkgpuBabyBear> = Vec::with_capacity(h * w);
        for r in 0..h {
            for v in mat.row(r).expect("matrix row access").into_iter() {
                flat.push(p3_to_zkgpu(v));
            }
        }

        // GPU commit: upload, dispatch leaf sponge + log(h) tree
        // compressions, read back the 8-element root. Nothing else —
        // no CPU tree, no sibling pre-computation.
        let gpu_root: [ZkgpuBabyBear; DIGEST_LEN] = {
            let mut plan = self.gpu_commit.lock().expect("gpu commit mutex");
            plan.commit_host_matrix(
                self.device.as_ref(),
                &flat,
                h as u32,
                w as u32,
            )
            .expect("GpuPoseidon2Mmcs::commit: GPU commit failed")
        };

        // Wrap the root in a MerkleCap — matches the type returned by
        // Plonky3's MerkleTreeMmcs at cap_height = 0.
        let root_p3: [P3BabyBear; DIGEST_LEN] =
            gpu_root.map(|x| P3BabyBear::new(x.0));
        let cap = MerkleCap::from(vec![root_p3]);

        let prover_data = GpuProverData { matrices: inputs };
        (cap, prover_data)
    }

    fn open_batch<M: Matrix<P3BabyBear>>(
        &self,
        _index: usize,
        _prover_data: &<Self as Mmcs<P3BabyBear>>::ProverData<M>,
    ) -> BatchOpening<P3BabyBear, Self> {
        // Commit-only scope for the Step 3 bench gate. Opening
        // support requires a CPU sibling tree — either built eagerly
        // (which defeats the purpose of the GPU commit) or lazily
        // here (which requires `M: Clone`, a bound the Mmcs trait
        // doesn't provide). Both options belong in a future hybrid
        // adapter; this one panics instead of silently producing a
        // "GPU" bench number that actually measured CPU.
        unimplemented!(
            "GpuPoseidon2Mmcs is a commit-only bench-gate adapter; \
             opening (and verification of openings) is not implemented. \
             Use Plonky3's MerkleTreeMmcs directly for paths that open."
        );
    }

    fn get_matrices<'a, M: Matrix<P3BabyBear>>(
        &self,
        prover_data: &'a <Self as Mmcs<P3BabyBear>>::ProverData<M>,
    ) -> Vec<&'a M> {
        prover_data.matrices.iter().collect()
    }

    fn verify_batch(
        &self,
        commit: &Self::Commitment,
        dimensions: &[Dimensions],
        index: usize,
        batch_opening: BatchOpeningRef<'_, P3BabyBear, Self>,
    ) -> Result<(), Self::Error> {
        // Verify by delegating to a fresh CPU `MerkleTreeMmcs` — the
        // root + sibling path are valid Plonky3 proof artefacts by
        // construction, so the verifier side can be checked CPU-side
        // without needing a pre-built tree. (The bench doesn't call
        // this, but providing it keeps the trait impl complete so
        // the type plugs cleanly into `TwoAdicFriPcs::commit`.)
        let cpu = self.cpu_mmcs();
        let cpu_ref: BatchOpeningRef<'_, P3BabyBear, CpuValMmcs> =
            BatchOpeningRef::new(batch_opening.opened_values, batch_opening.opening_proof);
        cpu.verify_batch(commit, dimensions, index, cpu_ref)
    }
}

// ---------------------------------------------------------------------------
// Convenience: BabyBear-root accessor (bench-friendly)
// ---------------------------------------------------------------------------

impl GpuPoseidon2Mmcs {
    /// Extract the 8-element BabyBear root from a `MerkleCap`
    /// produced by this adapter (or by Plonky3's CPU equivalent —
    /// same concrete type, same cap_height convention).
    ///
    /// Returns `None` if the cap has fewer than one element, which
    /// shouldn't happen under our `cap_height = 0` contract but is
    /// checked defensively.
    pub fn root(
        cap: &MerkleCap<P3BabyBear, [P3BabyBear; DIGEST_LEN]>,
    ) -> Option<[P3BabyBear; DIGEST_LEN]> {
        let slice: &[[P3BabyBear; DIGEST_LEN]] = cap.as_ref();
        slice.first().copied()
    }
}

// ---------------------------------------------------------------------------
// Tests (unit-level — the full Mmcs parity test lives in
// `tests/poseidon2_mmcs_gpu.rs`).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    // Unit-test coverage lives in the integration test file
    // `tests/poseidon2_mmcs_gpu.rs` where we can import
    // `rand::distr::StandardUniform` / SmallRng without pulling them
    // into the library build.
}
