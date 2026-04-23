//! GPU-accelerated OpenVM Poseidon2 MMCS adapter.
//!
//! Targets OpenVM's canonical `MerkleTreeMmcs<_, _, PaddingFreeSponge<
//! Perm16, 16, 8, 8>, TruncatedPermutation<Perm16, 2, 8, 16>, 8>` at
//! Plonky3 version 0.4.1, `cap_height = 0`. Bit-compatible commit
//! roots; opens and verify roundtrip in Stage 2b.
//!
//! # Surface (Stage 2a)
//!
//! The type exposes two **inherent** methods matching the
//! `p3_commit::Mmcs` trait's `commit` and `get_matrices` shapes:
//!
//! * [`OpenVmGpuMmcs::commit`] — mixed-height commits via
//!   [`zkgpu_wgpu::commit_mixed_height_with_w16_leaf`]. Supports the
//!   full mixed-height topology OpenVM's `VERIFY_BATCH` spec
//!   describes (single-matrix, same-height batches, and mixed-height
//!   injection DAG), all in one entry point.
//! * [`OpenVmGpuMmcs::get_matrices`] — returns refs to the stored
//!   input matrices; zero allocation.
//!
//! # Why no `impl Mmcs<BabyBear>` yet
//!
//! Stage 2a deliberately does **not** expose a `Mmcs` trait impl.
//! The trait requires `open_batch` and `verify_batch`, which would
//! have to be `todo!()` placeholders in Stage 2a and would let
//! downstream callers compile against a `Mmcs`-shaped type whose
//! opening paths panic at runtime. Instead, the inherent methods
//! above carry Stage 2a's complete, tested surface; Stage 2b adds
//! the full `Mmcs` trait impl as a **strictly additive** change
//! (existing callers keep working; new callers additionally get
//! trait dispatch).
//!
//! The shared backend already has parity-pinned implementations of
//! the opening path (see `zkgpu_wgpu::open_batch_mixed_height` and
//! the suite in `zkgpu-plonky3/tests/poseidon2_merkle_open_dag_gpu.rs`)
//! — Stage 2b is plumbing, not new correctness work.
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
use p3_matrix::Matrix;
use p3_symmetric::Hash;

use zkgpu_babybear::BabyBear as ZkgpuBabyBear;
use zkgpu_poseidon2::Poseidon2Params;
use zkgpu_wgpu::{
    commit_mixed_height_with_w16_leaf, MixedHeightMatrixInput, RetainedLayersHost,
    WgpuDevice, WgpuPoseidon2MerkleCompressPlan, WgpuPoseidon2MerkleLeafW16R8Plan,
};

use crate::config::{
    self, Commitment, Compress, LeafHash, Perm, SUPPORTED_CAP_HEIGHT, Val,
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
            commit_mixed_height_with_w16_leaf(
                self.device.as_ref(),
                &mut *leaf,
                &mut *compress,
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
        };
        (cap, prover_data)
    }

    /// Return references to the matrices that were passed to
    /// [`Self::commit`]. Shape matches `p3_commit::Mmcs::
    /// get_matrices`; forward-compatible with the Stage 2b trait
    /// impl.
    pub fn get_matrices<'a, M>(
        &self,
        prover_data: &'a GpuProverData<M>,
    ) -> Vec<&'a M> {
        prover_data.matrices.iter().collect()
    }
}
