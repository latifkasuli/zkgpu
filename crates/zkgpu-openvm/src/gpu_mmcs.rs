//! GPU-accelerated OpenVM Poseidon2 MMCS adapter.
//!
//! Implements `p3_commit::Mmcs<BabyBear>` at Plonky3 version 0.4.1
//! (OpenVM's pin) with a GPU-accelerated `commit()`. Bit-compatible
//! with OpenVM's CPU `MerkleTreeMmcs<_, _, PaddingFreeSponge<Perm16,
//! 16, 8, 8>, TruncatedPermutation<Perm16, 2, 8, 16>, 2, 8>` at
//! `cap_height = 0`.
//!
//! # Scope (Stage 2a)
//!
//! * `Mmcs::commit` — mixed-height commits via
//!   [`zkgpu_wgpu::commit_mixed_height_with_w16_leaf`]. Supports the
//!   full mixed-height topology OpenVM's `VERIFY_BATCH` spec
//!   describes (single-matrix, same-height batches, and mixed-height
//!   injection DAG), all in one entry point.
//! * `Mmcs::get_matrices` — returns refs to the stored input
//!   matrices; zero allocation.
//! * `Mmcs::open_batch` + `Mmcs::verify_batch` — Stage 2b. Currently
//!   return `todo!("Stage 2b")` panics so the type checks-out as an
//!   `Mmcs<BabyBear>` impl without shipping unverified opening logic
//!   yet. The shared backend already provides
//!   `open_batch_mixed_height` with parity + verifier-roundtrip
//!   coverage (see `zkgpu-plonky3/tests/
//!   poseidon2_merkle_open_dag_gpu.rs`), so 2b is plumbing rather
//!   than new correctness work.
//!
//! # Out of scope
//!
//! * `cap_height > 0` — rejected in `new()`.
//! * Non-BabyBear fields.
//! * Proof-of-work / challenger integration — that lives above the
//!   MMCS layer, not here.

use std::convert::TryInto;
use std::sync::{Arc, Mutex};

use p3_baby_bear::BabyBear as P3BabyBear;
use p3_commit::{BatchOpening, BatchOpeningRef, Mmcs};
use p3_matrix::{Dimensions, Matrix};
use p3_symmetric::Hash;

use zkgpu_babybear::BabyBear as ZkgpuBabyBear;
use zkgpu_poseidon2::Poseidon2Params;
use zkgpu_wgpu::{
    commit_mixed_height_with_w16_leaf, MixedHeightMatrixInput, RetainedLayersHost,
    WgpuDevice, WgpuPoseidon2MerkleCompressPlan, WgpuPoseidon2MerkleLeafW16R8Plan,
};

use crate::config::{
    self, Commitment, Compress, Error, LeafHash, Perm, Proof, SUPPORTED_CAP_HEIGHT, Val,
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

        let cpu_hash = LeafHash::new(perm16.clone());
        let cpu_compress = Compress::new(perm16);

        Ok(Self {
            device,
            leaf: Arc::new(Mutex::new(leaf)),
            compress: Arc::new(Mutex::new(compress)),
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
    /// Consumed by Stage 2b's `open_batch` via
    /// `open_batch_mixed_height`; unused in Stage 2a (commit-only
    /// path) beyond the commit call itself.
    #[allow(dead_code)] // Stage 2b consumer
    pub(crate) layers: RetainedLayersHost,
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
// Mmcs impl
// ---------------------------------------------------------------------------

impl Mmcs<P3BabyBear> for OpenVmGpuMmcs {
    type ProverData<M> = GpuProverData<M>;
    type Commitment = Commitment;
    type Proof = Proof;
    type Error = Error;

    fn commit<M: Matrix<P3BabyBear>>(
        &self,
        inputs: Vec<M>,
    ) -> (Self::Commitment, Self::ProverData<M>) {
        assert!(
            !inputs.is_empty(),
            "OpenVmGpuMmcs::commit: called with 0 matrices"
        );

        // Flatten each matrix to a host-side row-major
        // `Vec<ZkgpuBabyBear>`. We do this before building the
        // `MixedHeightMatrixInput` slices so they can borrow into
        // the owned flats.
        let flats = flatten_matrices_for_gpu(&inputs);

        // Build the adapter-shape input refs.
        let gpu_inputs: Vec<MixedHeightMatrixInput<'_>> = flats
            .iter()
            .map(|(flat, h, w)| MixedHeightMatrixInput {
                flat: flat.as_slice(),
                height: *h,
                width: *w,
            })
            .collect();

        // Shared backend does the mixed-height commit (leaf sponge
        // on tallest group → per-level binary compression → inject
        // at matching-height levels → retained layers).
        let layers = {
            let mut leaf = self.leaf.lock().expect("gpu leaf mutex");
            let mut compress = self.compress.lock().expect("gpu compress mutex");
            commit_mixed_height_with_w16_leaf(
                self.device.as_ref(),
                &mut *leaf,
                &mut *compress,
                &gpu_inputs,
            )
            .expect("OpenVmGpuMmcs::commit: GPU commit failed")
        };

        // Root = top layer of the retained stack, in canonical
        // `ZkgpuBabyBear` form. Convert to Monty-form `P3BabyBear`
        // for the Plonky3-facing commitment.
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
        // Plonky3 0.4.1's `MerkleTreeMmcs::Commitment` is
        // `Hash<F, W, DIGEST>` — a phantom-typed wrapper over the
        // digest array. Direct `From<[Val; DIGEST_WIDTH]>`.
        let cap: Commitment = Hash::from(root_p3);

        let prover_data = GpuProverData {
            matrices: inputs,
            layers,
        };
        (cap, prover_data)
    }

    fn open_batch<M: Matrix<P3BabyBear>>(
        &self,
        _index: usize,
        _prover_data: &Self::ProverData<M>,
    ) -> BatchOpening<P3BabyBear, Self> {
        // Stage 2b plumbing. The shared backend's
        // `open_batch_mixed_height` already has parity + verify
        // roundtrip coverage (see
        // `zkgpu-plonky3/tests/poseidon2_merkle_open_dag_gpu.rs`);
        // Stage 2b wires it through here and maps to
        // `BatchOpening<P3BabyBear, Self>`.
        todo!("Stage 2b: open_batch via open_batch_mixed_height")
    }

    fn get_matrices<'a, M: Matrix<P3BabyBear>>(
        &self,
        prover_data: &'a Self::ProverData<M>,
    ) -> Vec<&'a M> {
        prover_data.matrices.iter().collect()
    }

    fn verify_batch(
        &self,
        _commit: &Self::Commitment,
        _dimensions: &[Dimensions],
        _index: usize,
        _batch_opening: BatchOpeningRef<'_, P3BabyBear, Self>,
    ) -> Result<(), Self::Error> {
        // Stage 2b: delegate to the CPU `ValMmcs` built on the same
        // constants. Safe because the commitment types are identical
        // (`MerkleCap<Val, [Val; DIGEST_WIDTH]>`) and the proof
        // format is the same concrete `Vec<[Val; DIGEST_WIDTH]>`.
        todo!("Stage 2b: verify_batch via CPU MerkleTreeMmcs delegation")
    }
}
