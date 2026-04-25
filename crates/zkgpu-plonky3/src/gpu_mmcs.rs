//! GPU Poseidon2 MMCS adapter — mixed-height capable.
//!
//! Implements [`p3_commit::Mmcs<BabyBear>`] with a GPU-accelerated
//! `commit()` / `open_batch()` path that matches Plonky3's
//! `MerkleTreeMmcs<Packing, Packing, Poseidon2Sponge, Poseidon2Compression, 2, 8>`
//! bit-for-bit at `cap_height = 0`.
//!
//! # Scope
//!
//! **Binary N=2 tree, power-of-two heights, cap_height=0.** Supports
//! every shape Plonky3's `MerkleTreeMmcs::commit` validly accepts at
//! that cap height:
//!
//! * **Single matrix** — the trace commit
//!   (`uni-stark::prover::prove` → `pcs.commit(vec![(domain, trace)])`).
//! * **N same-height matrices** — the quotient-chunk batch
//!   (`uni-stark::prover::commit_quotient` splits the quotient into
//!   `k` equal-height matrices and commits them all together).
//! * **Mixed-height multi-matrix** — `compress_and_inject`-style DAG
//!   tree where matrices at different heights are injected at the
//!   tree levels matching their power-of-two height. Required for
//!   any Plonky3 consumer that commits trace + preprocessing /
//!   fixed / random matrices alongside it at differing heights.
//!
//! Mixed-height routing goes through the shared backend's
//! [`zkgpu_wgpu::commit_mixed_height_with_w24_leaf`] — the same
//! engine the sibling `zkgpu-openvm` adapter uses (with the W16
//! leaf variant). This is the post-convergence shape: the two
//! consumer adapters share one backend, parity-pinned across both
//! leaf shapes.
//!
//! # Out of scope
//!
//! * `cap_height > 0` → the adapter returns a single-digest cap
//!   shaped for the root-only convention; see `SUPPORTED_CAP_HEIGHT`.
//!   Wider caps would change the commit shape and require a backend
//!   change to retain `2^cap_height` digests at the appropriate
//!   level.
//!
//! # Opening path — GPU-retained layers, no CPU tree rebuild
//!
//! Every `commit()` calls `commit_mixed_height_with_w24_leaf`, which
//! writes per-level digest buffers on the GPU and downloads them
//! once. `open_batch` then routes through
//! [`zkgpu_wgpu::open_batch_mixed_height`], which walks the variable-
//! arity DAG bottom-up and produces the sibling chain Plonky3's
//! verifier expects. No CPU hash is ever computed during the open
//! path.
//!
//! VRAM cost per commit: `(2·h_max − 1) · DIGEST_LEN · 4` bytes for
//! the all-same-height shape, slightly less for mixed-height (some
//! levels collapse). With `DIGEST_LEN = 8` that's ≈64·h_max bytes —
//! about **16 MiB at h_max = 2¹⁸**. Host cost is the same; we
//! download the retained layers once at commit time and serve
//! openings from them. ICICLE-style "retain upper levels, recompute
//! lower on demand" belongs in a follow-up — relevant around
//! h_max = 2²² (≈256 MiB retained).

use std::sync::{Arc, Mutex};

use p3_baby_bear::BabyBear as P3BabyBear;
use p3_commit::{BatchOpening, BatchOpeningRef, Mmcs};
use p3_field::PrimeField32;
use p3_matrix::{Dimensions, Matrix};
use p3_merkle_tree::{MerkleTreeError, MerkleTreeMmcs};
use p3_symmetric::{MerkleCap, PaddingFreeSponge, TruncatedPermutation};

use zkgpu_babybear::BabyBear as ZkgpuBabyBear;
use zkgpu_poseidon2::Poseidon2Params;
use zkgpu_wgpu::{
    commit_mixed_height_with_w24_leaf, open_batch_mixed_height, MixedHeightMatrixInput,
    RetainedLayersHost, WgpuDevice, WgpuPoseidon2MerkleCompressPlan,
    WgpuPoseidon2MerkleLeafPlan,
};

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
    /// W24/RATE=16 leaf sponge plan — Plonky3 canonical.
    leaf: Arc<Mutex<WgpuPoseidon2MerkleLeafPlan>>,
    /// W16 binary compression plan — shared with OpenVM's adapter.
    compress: Arc<Mutex<WgpuPoseidon2MerkleCompressPlan>>,
    /// CPU leaf hasher, same constants as `leaf`. Used by
    /// `verify_batch` and any consumer that needs a host-side
    /// reference hash.
    cpu_hash: P3Poseidon2Sponge,
    /// CPU compression function, same constants as `compress`.
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

        let leaf = WgpuPoseidon2MerkleLeafPlan::new(device.as_ref(), leaf_params)
            .map_err(|e| {
                format!("GpuPoseidon2Mmcs: GPU leaf plan construction failed: {e}")
            })?;
        let compress =
            WgpuPoseidon2MerkleCompressPlan::new(device.as_ref(), compress_params)
                .map_err(|e| {
                    format!(
                        "GpuPoseidon2Mmcs: GPU compress plan construction failed: {e}"
                    )
                })?;

        let cpu_hash = P3Poseidon2Sponge::new(perm24);
        let cpu_compress = P3Poseidon2Compression::new(perm16);

        Ok(Self {
            device,
            leaf: Arc::new(Mutex::new(leaf)),
            compress: Arc::new(Mutex::new(compress)),
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

/// Prover data produced by [`GpuPoseidon2Mmcs::commit`].
///
/// Holds the input matrices (for `get_matrices` and row-level
/// opened-values lookups), the GPU-retained per-level digest stack
/// (so [`Mmcs::open_batch`] never touches a CPU hash), and a cache of
/// the row-major flattened matrices in zkgpu canonical form (so
/// `open_batch` doesn't re-flatten on every query — saves
/// `num_queries` flattens per FRI commit).
///
/// Memory: `matrices` cost is caller-defined; retained layers cost
/// ≈ `64·h_max` bytes (≈16 MiB at `h_max = 2¹⁸` with
/// `DIGEST_LEN = 8`); `flats` is one extra copy of each input matrix
/// in zkgpu canonical form (same order as the original).
pub struct GpuProverData<M> {
    matrices: Vec<M>,
    /// Retained layer stack from `commit_mixed_height_with_w24_leaf`.
    /// Consumed by `open_batch` through `open_batch_mixed_height` to
    /// extract the sibling chain for a given row index.
    layers: RetainedLayersHost,
    /// Per-matrix row-major flats in zkgpu canonical form, cached
    /// from the commit path so `open_batch` doesn't re-flatten on
    /// every call.
    flats: Vec<(Vec<ZkgpuBabyBear>, u32, u32)>,
}

// Internal helper: convert Plonky3 matrices → row-major flats in
// zkgpu canonical form. Mirrors the same pattern in
// `zkgpu-openvm/src/gpu_mmcs.rs`.
fn flatten_matrices_for_gpu<M: Matrix<P3BabyBear>>(
    inputs: &[M],
) -> Vec<(Vec<ZkgpuBabyBear>, u32, u32)> {
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

        // Mixed-height is fully supported via the shared backend's
        // `compress_and_inject`-style DAG engine. Single-matrix and
        // same-height multi-matrix shapes still flow through the
        // same path — they're degenerate cases of the mixed-height
        // engine where every matrix lands at the same level.
        let flats = flatten_matrices_for_gpu(&inputs);
        let gpu_inputs: Vec<MixedHeightMatrixInput<'_>> = flats
            .iter()
            .map(|(flat, h, w)| MixedHeightMatrixInput {
                flat: flat.as_slice(),
                height: *h,
                width: *w,
            })
            .collect();

        let layers: RetainedLayersHost = {
            let mut leaf = self.leaf.lock().expect("gpu leaf mutex");
            let mut compress = self.compress.lock().expect("gpu compress mutex");
            commit_mixed_height_with_w24_leaf(
                self.device.as_ref(),
                &mut *leaf,
                &mut *compress,
                &gpu_inputs,
            )
            .expect("GpuPoseidon2Mmcs::commit: GPU commit failed")
        };

        // Convert the root from zkgpu BabyBear (canonical u32) to
        // Plonky3 BabyBear (Monty form) and wrap as a 1-element
        // MerkleCap (matches Plonky3's MerkleTreeMmcs output at
        // cap_height = 0).
        let gpu_root = layers.root();
        let root_p3: [P3BabyBear; DIGEST_LEN] =
            gpu_root.map(|x| P3BabyBear::new(x.0));
        let cap = MerkleCap::from(vec![root_p3]);

        let prover_data = GpuProverData {
            matrices: inputs,
            layers,
            flats,
        };
        (cap, prover_data)
    }

    fn open_batch<M: Matrix<P3BabyBear>>(
        &self,
        index: usize,
        prover_data: &<Self as Mmcs<P3BabyBear>>::ProverData<M>,
    ) -> BatchOpening<P3BabyBear, Self> {
        // Routes through the shared backend's
        // `open_batch_mixed_height`, which is parity-pinned against
        // Plonky3's `MerkleTreeMmcs::open_batch` for both same-height
        // and mixed-height shapes (see
        // `tests/poseidon2_merkle_open_dag_gpu.rs`). The retained
        // layers + cached flats live in `prover_data` from commit
        // time — no GPU work happens here, just a host-side DAG
        // walk + boundary type conversion.
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
            .expect("GpuPoseidon2Mmcs::open_batch: index exceeds u32");
        let opening = open_batch_mixed_height(
            &gpu_inputs,
            &prover_data.layers,
            idx_u32,
        )
        .expect("GpuPoseidon2Mmcs::open_batch: backend open failed");

        // Convert zkgpu canonical → Plonky3 Monty-form at the
        // boundary. Shape preserved: one row Vec per matrix (in
        // input order, not height-sorted), and one 8-element
        // sibling digest per tree level bottom-up.
        let opened_values: Vec<Vec<P3BabyBear>> = opening
            .opened_values
            .into_iter()
            .map(|row| row.into_iter().map(|v| P3BabyBear::new(v.0)).collect())
            .collect();
        let proof: Vec<[P3BabyBear; DIGEST_LEN]> = opening
            .opening_proof
            .into_iter()
            .map(|d| d.map(|v| P3BabyBear::new(v.0)))
            .collect();

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
