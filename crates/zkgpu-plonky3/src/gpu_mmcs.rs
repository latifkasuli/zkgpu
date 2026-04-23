//! Phase 7 Step 3 — GPU Poseidon2 MMCS adapter.
//!
//! Implements [`p3_commit::Mmcs<BabyBear>`] with a GPU-accelerated
//! `commit()` and opening path that matches Plonky3's
//! `MerkleTreeMmcs<Packing, Packing, Poseidon2Sponge, Poseidon2Compression, 2, 8>`
//! bit-for-bit at `cap_height = 0`.
//!
//! # Scope (Step 3.c)
//!
//! **Binary N=2 tree, same-height inputs, power-of-two height, cap_height=0.**
//! Supports the two matrix-count shapes that show up in the target
//! stack's full prove path:
//!
//! * **Single matrix** — the trace commit
//!   (`uni-stark::prover::prove` → `pcs.commit(vec![(domain, trace)])`).
//! * **N same-height matrices** — the quotient-chunk batch
//!   (`uni-stark::prover::commit_quotient` splits the quotient into
//!   `k` equal-height matrices and commits them all together).
//!
//! Plonky3's `first_digest_layer` hashes row *i* of each tallest
//! matrix into one leaf digest via `flat_map` (see
//! `p3_merkle_tree::merkle_tree.rs` line 300). The adapter mirrors
//! that by flattening row-*i* of every input matrix into a single
//! wide row and running the existing single-matrix GPU leaf sponge —
//! no new kernel required.
//!
//! # Out of scope
//!
//! * Multi-matrix with **differing** heights → needs
//!   `compress_and_inject` (DAG-shaped tree with injection at
//!   height-specific levels). Not needed for the target stack; a
//!   future adapter can layer it on top.
//! * `cap_height > 0` → this adapter returns a single-digest cap
//!   shaped for the root-only convention; see `SUPPORTED_CAP_HEIGHT`.
//!
//! # Opening path — GPU-retained layers, no CPU tree rebuild
//!
//! The reviewer's Step 3.c direction (option 3): keep the digest
//! layers the GPU commit already produced and serve openings by
//! indexing into them, instead of materialising a CPU
//! `MerkleTreeMmcs` prover tree on demand. Every `commit()` now calls
//! `WgpuPoseidon2MerkleCommit::commit_host_matrix_with_layers`, which
//! writes per-level buffers on the GPU and downloads them once.
//! `open_batch` then walks those layers (log₂(h) sibling lookups) and
//! reads opened row values directly from the stored input matrices —
//! no CPU hash is ever computed.
//!
//! VRAM cost per commit: `(2h - 1) * DIGEST_LEN * 4` bytes. With
//! `DIGEST_LEN = 8` that's `≈ 64·h` bytes — about **16 MiB at
//! h = 2¹⁸**. Host cost is the same (we download the layers).
//! ICICLE-style "retain upper levels, recompute lower layers on
//! demand" belongs in a follow-up — for the initial multi-query
//! claim at target-stack sizes this is fine, but larger shapes
//! (e.g. h = 2²², ≈ 256 MiB retained) would motivate the cutoff.

use std::sync::{Arc, Mutex};

use p3_baby_bear::BabyBear as P3BabyBear;
use p3_commit::{BatchOpening, BatchOpeningRef, Mmcs};
use p3_matrix::{Dimensions, Matrix};
use p3_merkle_tree::{MerkleTreeError, MerkleTreeMmcs};
use p3_symmetric::{MerkleCap, PaddingFreeSponge, TruncatedPermutation};

use zkgpu_babybear::BabyBear as ZkgpuBabyBear;
use zkgpu_poseidon2::Poseidon2Params;
use zkgpu_wgpu::{RetainedLayersHost, WgpuDevice, WgpuPoseidon2MerkleCommit};

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

/// CPU MMCS type whose commit/open_batch this adapter produces
/// bit-identical output for. Kept here for `verify_batch` delegation
/// and for tests.
type CpuValMmcs = MerkleTreeMmcs<
    <P3BabyBear as p3_field::Field>::Packing,
    <P3BabyBear as p3_field::Field>::Packing,
    P3Poseidon2Sponge,
    P3Poseidon2Compression,
    2,
    DIGEST_LEN,
>;

/// Cap height supported by this adapter.
///
/// `commit()` wraps the GPU-computed root as a single-digest
/// `MerkleCap`, which only matches Plonky3's `MerkleTreeMmcs` output
/// at `cap_height = 0`. Any other value would ask for `2^cap_height`
/// sibling digests one layer below the root — which the GPU pipeline
/// doesn't surface in this adapter. A future wider-cap adapter would
/// change the commit shape anyway.
const SUPPORTED_CAP_HEIGHT: usize = 0;

// ---------------------------------------------------------------------------
// GpuPoseidon2Mmcs
// ---------------------------------------------------------------------------

/// GPU-accelerated Poseidon2 MMCS. See module doc for scope.
///
/// `Clone` is cheap — internal state is shared via `Arc`. The GPU
/// plan sits behind a `Mutex` because `WgpuPoseidon2MerkleCommit::
/// commit_with_retained_layers` takes `&mut self`, and
/// `Mmcs::commit` takes `&self`. Concurrent commits on one adapter
/// instance serialise on the mutex; the target-stack prover commits
/// the trace and the quotient back-to-back on a single thread, so
/// there's no contention.
#[derive(Clone)]
pub struct GpuPoseidon2Mmcs {
    device: Arc<WgpuDevice>,
    gpu_commit: Arc<Mutex<WgpuPoseidon2MerkleCommit>>,
    cpu_hash: P3Poseidon2Sponge,
    cpu_compress: P3Poseidon2Compression,
}

impl GpuPoseidon2Mmcs {
    /// Construct from matched Plonky3 `(Perm16, Perm24)` constants.
    ///
    /// * `cap_height` must be `0`. See `SUPPORTED_CAP_HEIGHT`.
    pub fn new(
        device: Arc<WgpuDevice>,
        perm24: Perm24,
        perm16: Perm16,
        leaf_params: Poseidon2Params<ZkgpuBabyBear, 24>,
        compress_params: Poseidon2Params<ZkgpuBabyBear, 16>,
        cap_height: usize,
    ) -> Result<Self, String> {
        if cap_height != SUPPORTED_CAP_HEIGHT {
            return Err(format!(
                "GpuPoseidon2Mmcs: cap_height={cap_height} not supported; \
                 this adapter only implements cap_height={SUPPORTED_CAP_HEIGHT}"
            ));
        }

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
        })
    }

    fn cpu_mmcs(&self) -> CpuValMmcs {
        CpuValMmcs::new(
            self.cpu_hash.clone(),
            self.cpu_compress.clone(),
            SUPPORTED_CAP_HEIGHT,
        )
    }

    /// Extract the 8-element root from a cap produced by this adapter
    /// (or by Plonky3's CPU equivalent with cap_height=0).
    pub fn root(
        cap: &MerkleCap<P3BabyBear, [P3BabyBear; DIGEST_LEN]>,
    ) -> Option<[P3BabyBear; DIGEST_LEN]> {
        let slice: &[[P3BabyBear; DIGEST_LEN]] = cap.as_ref();
        slice.first().copied()
    }
}

// ---------------------------------------------------------------------------
// GpuProverData<M>
// ---------------------------------------------------------------------------

/// Prover data produced by [`GpuPoseidon2Mmcs::commit`]. Holds the
/// input matrices (for row-level opened-values lookups) and the
/// per-level digest stack the GPU retained during commit, so
/// [`Mmcs::open_batch`] can answer openings without touching a CPU
/// hash.
///
/// Memory: `matrices` cost is caller-defined; retained layers cost
/// `≈ 64·h` bytes on host (≈16 MiB at h=2¹⁸ with `DIGEST_LEN = 8`).
pub struct GpuProverData<M> {
    matrices: Vec<M>,
    layers: RetainedLayersHost,
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
    ) -> (Self::Commitment, Self::ProverData<M>) {
        assert!(
            !inputs.is_empty(),
            "GpuPoseidon2Mmcs::commit: called with 0 matrices"
        );

        // All inputs must be same height + same power-of-two height,
        // and each must have width ≥ 1. Mixed-height batches need
        // Plonky3's `compress_and_inject` injection DAG, which this
        // adapter doesn't implement.
        let h = inputs[0].height();
        let mut total_width: usize = 0;
        for (i, mat) in inputs.iter().enumerate() {
            let mh = mat.height();
            let mw = mat.width();
            assert_eq!(
                mh, h,
                "GpuPoseidon2Mmcs::commit: matrix {i} has height {mh}, expected {h} \
                 (this adapter only supports same-height batches; mixed-height \
                 injection is not implemented)"
            );
            assert!(
                mw > 0,
                "GpuPoseidon2Mmcs::commit: matrix {i} has width 0"
            );
            total_width = total_width
                .checked_add(mw)
                .expect("total width overflow");
        }
        assert!(h > 0, "GpuPoseidon2Mmcs::commit: h must be ≥ 1");
        assert!(
            h.is_power_of_two(),
            "GpuPoseidon2Mmcs::commit: h must be a power of two (got {h})"
        );

        // Flatten: one logical row of the "joint" matrix is the
        // concatenation of the same row across every input, in input
        // order — exactly what Plonky3's `first_digest_layer` feeds
        // the leaf hasher for same-height tallest matrices.
        let mut flat: Vec<ZkgpuBabyBear> = Vec::with_capacity(h * total_width);
        for r in 0..h {
            for mat in &inputs {
                for v in mat.row(r).expect("matrix row access").into_iter() {
                    flat.push(p3_to_zkgpu(v));
                }
            }
        }

        let layers: RetainedLayersHost = {
            let mut plan = self.gpu_commit.lock().expect("gpu commit mutex");
            plan.commit_host_matrix_with_layers(
                self.device.as_ref(),
                &flat,
                h as u32,
                total_width as u32,
            )
            .expect("GpuPoseidon2Mmcs::commit: GPU commit failed")
        };

        // Convert the root from zkgpu BabyBear (canonical u32) to p3
        // BabyBear (Monty) and wrap as a 1-element MerkleCap.
        let gpu_root = layers.root();
        let root_p3: [P3BabyBear; DIGEST_LEN] =
            gpu_root.map(|x| P3BabyBear::new(x.0));
        let cap = MerkleCap::from(vec![root_p3]);

        let prover_data = GpuProverData {
            matrices: inputs,
            layers,
        };
        (cap, prover_data)
    }

    fn open_batch<M: Matrix<P3BabyBear>>(
        &self,
        index: usize,
        prover_data: &<Self as Mmcs<P3BabyBear>>::ProverData<M>,
    ) -> BatchOpening<P3BabyBear, Self> {
        // All matrices are same-height by construction (enforced in
        // commit), so `max_height == matrix.height()` and the
        // Plonky3 `bits_reduced = log_max_height - log_height` is 0
        // for every matrix. `opened_values[i] = matrices[i].row(index)`.
        let h = prover_data.layers.num_leaves();
        assert!(
            index < h,
            "GpuPoseidon2Mmcs::open_batch: index {index} out of bounds (h={h})"
        );

        let opened_values: Vec<Vec<P3BabyBear>> = prover_data
            .matrices
            .iter()
            .map(|m| {
                m.row(index)
                    .expect("row access at valid index")
                    .into_iter()
                    .collect()
            })
            .collect();

        // Sibling path: for each tree level `k` in `0..log₂(h)`,
        // push the sibling of the node at `index >> k`, i.e. index
        // `(index >> k) ^ 1` of `layers[k]`. Bottom-up order matches
        // Plonky3's `MerkleTreeMmcs::open_batch`.
        let log_h = h.trailing_zeros() as usize;
        let mut proof: Vec<[P3BabyBear; DIGEST_LEN]> = Vec::with_capacity(log_h);
        let mut idx = index;
        for layer_idx in 0..log_h {
            let sibling_idx = idx ^ 1;
            let sib_zkgpu = prover_data
                .layers
                .digest_at(layer_idx, sibling_idx)
                .expect("sibling digest in retained layers");
            let sib_p3: [P3BabyBear; DIGEST_LEN] =
                sib_zkgpu.map(|x| P3BabyBear::new(x.0));
            proof.push(sib_p3);
            idx >>= 1;
        }

        BatchOpening::new(opened_values, proof)
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
        // Delegate verification to a fresh CPU `MerkleTreeMmcs`. The
        // root + sibling chain are valid Plonky3 proof artefacts by
        // construction (matched constants, same tree shape, same
        // compression), so a verifier built from the same Poseidon2
        // params accepts them. This keeps verify_batch CPU-side,
        // which is correct for the prove path where the verifier is
        // itself CPU.
        let cpu = self.cpu_mmcs();
        let cpu_ref: BatchOpeningRef<'_, P3BabyBear, CpuValMmcs> =
            BatchOpeningRef::new(batch_opening.opened_values, batch_opening.opening_proof);
        cpu.verify_batch(commit, dimensions, index, cpu_ref)
    }
}
