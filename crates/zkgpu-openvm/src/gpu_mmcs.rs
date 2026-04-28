//! GPU-accelerated OpenVM Poseidon2 MMCS adapter.
//!
//! Targets OpenVM's canonical `MerkleTreeMmcs<_, _, PaddingFreeSponge<
//! Perm16, 16, 8, 8>, TruncatedPermutation<Perm16, 2, 8, 16>, 8>` at
//! Plonky3 version 0.4.1, `cap_height = 0`. Bit-compatible commit
//! roots, openings, and verifier roundtrips.
//!
//! # Surface
//!
//! Two layers that share logic:
//!
//! 1. **Inherent methods** on [`OpenVmGpuMmcs`]:
//!    [`OpenVmGpuMmcs::commit`], [`OpenVmGpuMmcs::get_matrices`],
//!    [`OpenVmGpuMmcs::open_batch_inherent`],
//!    [`OpenVmGpuMmcs::verify_batch_inherent`]. Direct, no trait
//!    dispatch.
//! 2. **`impl Mmcs<P3BabyBear>`**: full Plonky3 0.4.1 trait impl
//!    (Stage 2b). Each method delegates to its inherent
//!    counterpart. Lets the adapter plug into anything generic
//!    over `Mmcs<BabyBear>` — `TwoAdicFriPcs`, `StarkConfig`,
//!    downstream consumers.
//!
//! Both layers are parity-pinned against Plonky3's CPU
//! `MerkleTreeMmcs` under the exact OpenVM config (see
//! `tests/commit_parity.rs` for commit root parity;
//! `tests/open_verify_parity.rs` for opening parity + verifier
//! roundtrip).
//!
//! # Commit path
//!
//! Routes through [`zkgpu_wgpu::commit_mixed_height_with_w16_leaf`].
//! Supports the full mixed-height topology OpenVM's `VERIFY_BATCH`
//! spec describes — single-matrix, same-height batches, and
//! mixed-height injection DAG — in one entry point.
//!
//! # Open path
//!
//! Routes through [`zkgpu_wgpu::open_batch_mixed_height`]. Pure
//! host-side work: extract the row at the local-shifted index
//! from each stored matrix, walk retained layers bottom-up for the
//! sibling chain. No GPU dispatch on the open path itself — all
//! the GPU work was done at commit time and cached in the
//! retained-layer stack.
//!
//! # Verify path
//!
//! Delegates to a freshly-constructed CPU `MerkleTreeMmcs` with
//! the same Poseidon2 constants as the GPU plans. The commitment
//! type (`Hash<Val, Val, DIGEST_WIDTH>`) and proof type
//! (`Vec<[Val; DIGEST_WIDTH]>`) are identical on both sides, so a
//! `BatchOpeningRef` retyping is a no-op at runtime.
//!
//! # Out of scope
//!
//! * `cap_height > 0` — rejected in `new()`.
//! * Non-BabyBear fields.
//! * Proof-of-work / challenger integration — above the MMCS layer.

use std::convert::TryInto;
use std::sync::{Arc, Mutex};

use p3_baby_bear::BabyBear as P3BabyBear;
use p3_matrix::Matrix;
use p3_symmetric::Hash;

use zkgpu_babybear::BabyBear as ZkgpuBabyBear;
use zkgpu_poseidon2::Poseidon2Params;
use zkgpu_wgpu::{
    commit_mixed_height_with_w16_leaf, MixedHeightMatrixInput, RetainedLayersHost,
    WgpuDevice, WgpuPoseidon2InterleavePairsPlan, WgpuPoseidon2MerkleCompressPlan,
    WgpuPoseidon2MerkleLeafW16R8Plan,
};

use crate::config::{
    self, Commitment, Compress, LeafHash, Perm, Proof, SUPPORTED_CAP_HEIGHT, Val,
    DIGEST_WIDTH, WIDTH,
};

/// GPU-accelerated OpenVM Poseidon2 MMCS.
///
/// `Clone` semantics: cheap — all shared state is behind `Arc`. The
/// GPU plans sit behind `Mutex` because their backend methods take
/// `&mut self` (they own per-instance pipeline cache + scratch
/// state), and `Mmcs::commit` takes `&self`. In practice the prover
/// commits matrices serially from one thread, so there's no
/// contention.
#[derive(Clone)]
pub struct OpenVmGpuMmcs {
    device: Arc<WgpuDevice>,
    leaf: Arc<Mutex<WgpuPoseidon2MerkleLeafW16R8Plan>>,
    compress: Arc<Mutex<WgpuPoseidon2MerkleCompressPlan>>,
    /// GPU pair-interleave plan — used at injection levels of the
    /// mixed-height commit DAG (item #1 of speed-opportunities).
    /// One plan constructed per adapter instance, reused across every
    /// commit; same-height commits never invoke it.
    interleave: Arc<Mutex<WgpuPoseidon2InterleavePairsPlan>>,
    /// CPU leaf hasher, same constants as `leaf`. Used by
    /// `verify_batch` (Stage 2b) and by any consumer that needs a
    /// host-side reference hash.
    cpu_hash: LeafHash,
    /// CPU compression function, same constants as `compress`.
    cpu_compress: Compress,
}

impl OpenVmGpuMmcs {
    /// Construct from matched Plonky3 `(Perm16)` constants.
    ///
    /// `perm16` drives both the CPU sponge/compression (for
    /// `verify_batch` and the fallback path) and is the identity
    /// used to configure the GPU `leaf` + `compress` plans via the
    /// zkgpu Poseidon2 bridge. OpenVM's canonical config uses a
    /// single `Perm16` instance for both leaf and compression —
    /// we mirror that exactly.
    ///
    /// `leaf_params` and `compress_params` must be the same
    /// `Poseidon2Params<BabyBear, 16>` built via the Step-1.5a
    /// bridge from the same `(ext, int)` constants that generated
    /// `perm16`. The adapter's correctness depends on this; we
    /// can't verify the relationship at runtime without
    /// re-canonicalising the permutation round constants, but
    /// callers who build both via the same bridge function get it
    /// right by construction.
    ///
    /// # `cap_height`
    ///
    /// Must be `0`. See [`crate::config::SUPPORTED_CAP_HEIGHT`].
    pub fn new(
        device: Arc<WgpuDevice>,
        perm16: Perm,
        leaf_params: Poseidon2Params<ZkgpuBabyBear, WIDTH>,
        compress_params: Poseidon2Params<ZkgpuBabyBear, WIDTH>,
        cap_height: usize,
    ) -> Result<Self, String> {
        if cap_height != SUPPORTED_CAP_HEIGHT {
            return Err(format!(
                "OpenVmGpuMmcs: cap_height={cap_height} not supported; \
                 this adapter only implements cap_height={SUPPORTED_CAP_HEIGHT}"
            ));
        }

        let leaf = WgpuPoseidon2MerkleLeafW16R8Plan::new(device.as_ref(), leaf_params)
            .map_err(|e| {
                format!("OpenVmGpuMmcs: GPU leaf plan construction failed: {e}")
            })?;
        let compress =
            WgpuPoseidon2MerkleCompressPlan::new(device.as_ref(), compress_params)
                .map_err(|e| {
                    format!(
                        "OpenVmGpuMmcs: GPU compress plan construction failed: {e}"
                    )
                })?;
        let interleave = WgpuPoseidon2InterleavePairsPlan::new(device.as_ref())
            .map_err(|e| {
                format!(
                    "OpenVmGpuMmcs: GPU interleave plan construction failed: {e}"
                )
            })?;

        let cpu_hash = LeafHash::new(perm16.clone());
        let cpu_compress = Compress::new(perm16);

        Ok(Self {
            device,
            leaf: Arc::new(Mutex::new(leaf)),
            compress: Arc::new(Mutex::new(compress)),
            interleave: Arc::new(Mutex::new(interleave)),
            cpu_hash,
            cpu_compress,
        })
    }

    /// Rebuild the CPU-side `MerkleTreeMmcs` on demand — used by the
    /// Stage 2b `verify_batch` delegation path. Cheap: no tree is
    /// materialised, it's just the config wrapper.
    ///
    /// Note: Plonky3 0.4.1's `MerkleTreeMmcs::new` takes just
    /// `(hash, compress)` — the `cap_height` parameter was added in
    /// 0.5.x. [`SUPPORTED_CAP_HEIGHT`] is enforced at
    /// [`Self::new`] time and isn't needed here.
    #[allow(dead_code)] // Stage 2b consumer
    pub(crate) fn cpu_mmcs(&self) -> config::ValMmcs {
        config::ValMmcs::new(self.cpu_hash.clone(), self.cpu_compress.clone())
    }

    /// Extract the 8-element BabyBear root from a commitment
    /// produced by this adapter (or by OpenVM's CPU
    /// `MerkleTreeMmcs::commit`).
    ///
    /// Plonky3 0.4.1's `Commitment = Hash<Val, Val, DIGEST_WIDTH>`
    /// is a phantom-typed wrapper over `[Val; DIGEST_WIDTH]`; this
    /// just unwraps it. Unlike the 0.5.x `MerkleCap` case,
    /// extraction can't fail at this layer — the type itself
    /// guarantees a single digest.
    pub fn root(commitment: &Commitment) -> [Val; DIGEST_WIDTH] {
        let arr: &[Val; DIGEST_WIDTH] = commitment.as_ref();
        *arr
    }
}

// ---------------------------------------------------------------------------
// GpuProverData<M>
// ---------------------------------------------------------------------------

/// Prover data for [`OpenVmGpuMmcs`].
///
/// Holds the input matrices (for `get_matrices` and for
/// Stage-2b-era row lookups in `open_batch`) plus the GPU-computed
/// retained layers (for Stage-2b sibling walks).
///
/// Memory: matrix cost is caller-defined; retained layers cost
/// `≈ 64·h_max` bytes (`(2·h_max − 1) · DIGEST_WIDTH · 4` — with
/// `DIGEST_WIDTH = 8` that's `≈ 64·h_max`). About 16 MiB at
/// `h_max = 2¹⁸`.
pub struct GpuProverData<M> {
    pub(crate) matrices: Vec<M>,
    /// Retained layer stack from `commit_mixed_height_with_w16_leaf`.
    /// Consumed by `open_batch` through `open_batch_mixed_height`
    /// to extract the sibling chain for a given row index.
    pub(crate) layers: RetainedLayersHost,
    /// Per-matrix row-major flats in zkgpu canonical form, cached
    /// from the commit path so `open_batch` doesn't re-flatten on
    /// every call. For a FRI commit phase with `num_queries = 40`,
    /// this saves 40 host-side re-flattens per committed matrix
    /// batch; memory cost is one extra copy of each matrix (same
    /// order as `matrices`, same memory budget).
    pub(crate) flats: Vec<(Vec<ZkgpuBabyBear>, u32, u32)>,
}

// ---------------------------------------------------------------------------
// Internal helper: convert Plonky3 matrices → MixedHeightMatrixInput
// ---------------------------------------------------------------------------

/// Flatten each `Matrix<Val>` into a host `Vec<ZkgpuBabyBear>` in
/// row-major order, returning the concrete storage so the adapter's
/// `MixedHeightMatrixInput<'_>` slices can reference it during the
/// commit call.
fn flatten_matrices_for_gpu<M: Matrix<Val>>(
    inputs: &[M],
) -> Vec<(Vec<ZkgpuBabyBear>, u32, u32)> {
    use p3_field::PrimeField32;
    inputs
        .iter()
        .map(|m| {
            let h = m.height();
            let w = m.width();
            let mut flat: Vec<ZkgpuBabyBear> = Vec::with_capacity(h * w);
            for r in 0..h {
                for v in m.row(r).expect("Matrix::row at valid index") {
                    flat.push(ZkgpuBabyBear(v.as_canonical_u32()));
                }
            }
            (flat, h as u32, w as u32)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Inherent methods (Stage 2a surface)
// ---------------------------------------------------------------------------
//
// Stage 2a deliberately does NOT `impl Mmcs<BabyBear> for
// OpenVmGpuMmcs` yet. Only `commit` and `get_matrices` are
// implemented here; the full `Mmcs` trait also requires
// `open_batch` and `verify_batch`, which route through the shared
// backend's `open_batch_mixed_height` and a CPU delegation,
// respectively. Those land in Stage 2b.
//
// Publishing a half-implemented `Mmcs` with `todo!()` placeholders
// would let downstream callers compile against a `Mmcs`-shaped
// type whose opening paths panic at runtime — the P2 review
// finding. Exposing only inherent `commit`/`get_matrices` methods
// keeps the Stage 2a surface honest: what's callable works, and
// what's not callable literally isn't on the type. Stage 2b adds
// the trait impl as a strictly additive change (existing callers
// that use the inherent methods continue to work; new callers can
// additionally route through the trait).

impl OpenVmGpuMmcs {
    /// Commit a batch of matrices. Shape matches
    /// `p3_commit::Mmcs::commit`: `Vec<M: Matrix<BabyBear>>` →
    /// `(Commitment, GpuProverData<M>)`. Trait impl lands in Stage
    /// 2b; this inherent method is forward-compatible with it.
    ///
    /// Supports single, same-height multi, and mixed-height
    /// batches via the shared-backend mixed-height DAG engine.
    ///
    /// Panics if `inputs` is empty (matches Plonky3's
    /// `MerkleTreeMmcs::commit` behavior).
    pub fn commit<M: Matrix<P3BabyBear>>(
        &self,
        inputs: Vec<M>,
    ) -> (Commitment, GpuProverData<M>) {
        assert!(
            !inputs.is_empty(),
            "OpenVmGpuMmcs::commit: called with 0 matrices"
        );

        let flats = flatten_matrices_for_gpu(&inputs);
        let gpu_inputs: Vec<MixedHeightMatrixInput<'_>> = flats
            .iter()
            .map(|(flat, h, w)| MixedHeightMatrixInput {
                flat: flat.as_slice(),
                height: *h,
                width: *w,
            })
            .collect();

        let layers = {
            let mut leaf = self.leaf.lock().expect("gpu leaf mutex");
            let mut compress = self.compress.lock().expect("gpu compress mutex");
            let mut interleave =
                self.interleave.lock().expect("gpu interleave mutex");
            commit_mixed_height_with_w16_leaf(
                self.device.as_ref(),
                &mut *leaf,
                &mut *compress,
                &mut *interleave,
                &gpu_inputs,
            )
            .expect("OpenVmGpuMmcs::commit: GPU commit failed")
        };

        let top = layers
            .layers
            .last()
            .expect("retained layers non-empty (backend guarantee)");
        assert_eq!(
            top.len(),
            DIGEST_WIDTH,
            "OpenVmGpuMmcs::commit: top retained layer should hold exactly \
             DIGEST_WIDTH elements; got {}",
            top.len()
        );
        let root_zkgpu: [ZkgpuBabyBear; DIGEST_WIDTH] = top[..DIGEST_WIDTH]
            .try_into()
            .expect("slice length checked above");
        let root_p3: [Val; DIGEST_WIDTH] = root_zkgpu.map(|x| Val::new(x.0));
        let cap: Commitment = Hash::from(root_p3);

        let prover_data = GpuProverData {
            matrices: inputs,
            layers,
            flats,
        };
        (cap, prover_data)
    }

    /// Return references to the matrices that were passed to
    /// [`Self::commit`]. Shape matches `p3_commit::Mmcs::
    /// get_matrices`.
    pub fn get_matrices<'a, M>(
        &self,
        prover_data: &'a GpuProverData<M>,
    ) -> Vec<&'a M> {
        prover_data.matrices.iter().collect()
    }

    /// Open the row at global `index` against a previously-committed
    /// batch. Shape matches `p3_commit::Mmcs::open_batch`: returns
    /// `(opened_values, opening_proof)` where
    /// `opened_values[i]` is matrix `i`'s row at its local index
    /// (global `index` shifted for height-relative indexing), and
    /// `opening_proof` is a `log2(h_max)`-long sibling chain
    /// bottom-up from the retained layers.
    ///
    /// Routes through the shared backend's
    /// [`zkgpu_wgpu::open_batch_mixed_height`], which is already
    /// parity-pinned against Plonky3's `MerkleTreeMmcs::open_batch`
    /// (both leaf shapes, both single- and mixed-height) in
    /// `zkgpu-plonky3/tests/poseidon2_merkle_open_dag_gpu.rs`.
    /// Result is converted from zkgpu canonical BabyBear to
    /// Plonky3 Monty-form `P3BabyBear` at the boundary.
    pub fn open_batch_inherent<M>(
        &self,
        index: usize,
        prover_data: &GpuProverData<M>,
    ) -> (Vec<Vec<Val>>, Proof) {
        let gpu_inputs: Vec<MixedHeightMatrixInput<'_>> = prover_data
            .flats
            .iter()
            .map(|(flat, h, w)| MixedHeightMatrixInput {
                flat: flat.as_slice(),
                height: *h,
                width: *w,
            })
            .collect();
        let idx_u32 = u32::try_from(index)
            .expect("OpenVmGpuMmcs::open_batch_inherent: index exceeds u32");
        let opening = zkgpu_wgpu::open_batch_mixed_height(
            &gpu_inputs,
            &prover_data.layers,
            idx_u32,
        )
        .expect("OpenVmGpuMmcs::open_batch_inherent: backend open failed");

        // Convert zkgpu canonical → Plonky3 Monty-form at the
        // boundary. Shape is preserved: one row Vec per matrix,
        // and one 8-element sibling digest per tree level.
        let opened_values: Vec<Vec<Val>> = opening
            .opened_values
            .into_iter()
            .map(|row| row.into_iter().map(|v| Val::new(v.0)).collect())
            .collect();
        let proof: Proof = opening
            .opening_proof
            .into_iter()
            .map(|d| d.map(|v| Val::new(v.0)))
            .collect();
        (opened_values, proof)
    }

    /// Verify a batch opening against a commitment. Shape matches
    /// `p3_commit::Mmcs::verify_batch`.
    ///
    /// Delegates to a fresh CPU `MerkleTreeMmcs` built from the
    /// same Poseidon2 constants as the GPU plans. Safe because
    /// Plonky3 0.4.1's `MerkleTreeMmcs::Commitment` and `Proof`
    /// types are the same concrete types the GPU path produces:
    /// `Hash<Val, Val, DIGEST_WIDTH>` and `Vec<[Val; DIGEST_WIDTH]>`
    /// respectively. A `BatchOpeningRef` built for one impl can be
    /// retyped to the other at the trait-generic layer.
    pub fn verify_batch_inherent(
        &self,
        commit: &Commitment,
        dimensions: &[p3_matrix::Dimensions],
        index: usize,
        opened_values: &[Vec<Val>],
        opening_proof: &Proof,
    ) -> Result<(), crate::config::Error> {
        use p3_commit::{BatchOpeningRef, Mmcs};
        let cpu = self.cpu_mmcs();
        let cpu_ref: BatchOpeningRef<'_, P3BabyBear, crate::config::ValMmcs> =
            BatchOpeningRef::new(opened_values, opening_proof);
        cpu.verify_batch(commit, dimensions, index, cpu_ref)
    }
}

// ---------------------------------------------------------------------------
// Mmcs<P3BabyBear> impl
// ---------------------------------------------------------------------------
//
// Stage 2b landing: full Plonky3 0.4.1 `Mmcs` trait impl. Each
// trait method delegates to the inherent method of the same shape.
// This addition is strictly additive over Stage 2a's inherent API —
// existing callers using the inherent methods continue to work
// unchanged.
//
// Trait dispatch at the 0.4.1 surface wires `OpenVmGpuMmcs` into
// anything that generically consumes `Mmcs<BabyBear>`: e.g.
// OpenVM's `StarkConfig` / `TwoAdicFriPcs` pipelines, and any
// future downstream consumer writing `fn commit<M: Mmcs<_>>(...)`.
// Correctness is backed by both the Stage 2a commit parity suite
// and Stage 2b's new open/verify parity suite.

use p3_commit::{BatchOpening, BatchOpeningRef, Mmcs};
use p3_matrix::Dimensions;

impl Mmcs<P3BabyBear> for OpenVmGpuMmcs {
    type ProverData<M> = GpuProverData<M>;
    type Commitment = Commitment;
    type Proof = crate::config::Proof;
    type Error = crate::config::Error;

    fn commit<M: Matrix<P3BabyBear>>(
        &self,
        inputs: Vec<M>,
    ) -> (Self::Commitment, Self::ProverData<M>) {
        // Delegate to the inherent method to keep the commit-path
        // logic in one place. Any future specialisation (e.g.
        // GPU-resident inputs) changes only the inherent method.
        OpenVmGpuMmcs::commit(self, inputs)
    }

    fn open_batch<M: Matrix<P3BabyBear>>(
        &self,
        index: usize,
        prover_data: &Self::ProverData<M>,
    ) -> BatchOpening<P3BabyBear, Self> {
        let (opened_values, opening_proof) =
            self.open_batch_inherent(index, prover_data);
        BatchOpening::new(opened_values, opening_proof)
    }

    fn get_matrices<'a, M: Matrix<P3BabyBear>>(
        &self,
        prover_data: &'a Self::ProverData<M>,
    ) -> Vec<&'a M> {
        OpenVmGpuMmcs::get_matrices(self, prover_data)
    }

    fn verify_batch(
        &self,
        commit: &Self::Commitment,
        dimensions: &[Dimensions],
        index: usize,
        batch_opening: BatchOpeningRef<'_, P3BabyBear, Self>,
    ) -> Result<(), Self::Error> {
        // Unpack the trait-generic BatchOpeningRef into its
        // underlying slices, then hand to the inherent verify
        // helper. Proof type is the same concrete
        // `Vec<[Val; DIGEST_WIDTH]>` on both the GPU and CPU
        // adapter sides, so no conversion is needed here.
        let (opened_values, opening_proof) = batch_opening.unpack();
        self.verify_batch_inherent(
            commit,
            dimensions,
            index,
            opened_values,
            opening_proof,
        )
    }
}
