//! Phase 3.d Stage 1b — mixed-height Merkle commit DAG engine.
//!
//! Generalises the same-height `WgpuPoseidon2MerkleCommit` path in
//! [`super::merkle_commit`] to Plonky3's `MerkleTreeMmcs`
//! `compress_and_inject` semantics: a batch of matrices at varying
//! power-of-two heights, hashed and merged into a single root.
//!
//! Mirrors the algorithm in `p3-merkle-tree-0.5.2/src/merkle_tree.rs::
//! compress_and_inject` for the N=2 binary case (OpenVM's and
//! Plonky3's default arity). Same-height inputs are a degenerate case
//! of this engine — one height group, no injection.
//!
//! # Abstraction
//!
//! The engine is parameterised over a [`GpuLeafSponge`] trait so
//! either of the two in-tree shapes works:
//!
//! * [`super::merkle_leaf::WgpuPoseidon2MerkleLeafPlan`] — W24/RATE=16
//!   (Plonky3 canonical `PaddingFreeSponge<Perm24, 24, 16, 8>`).
//! * [`super::merkle_leaf_w16::WgpuPoseidon2MerkleLeafW16R8Plan`] —
//!   W16/RATE=8 (OpenVM `PaddingFreeSponge<Perm16, 16, 8, 8>`).
//!
//! The compression step always uses
//! [`super::merkle_compress::WgpuPoseidon2MerkleCompressPlan`]
//! (W16 `TruncatedPermutation<Perm16, 2, 8, 16>`) because both
//! consumer configs use it.
//!
//! # Algorithm (N=2 binary, all heights power of two)
//!
//! ```text
//! 1. Sort matrices by height descending; group by equal height.
//! 2. Let h_max = max height. Concatenate rows of all matrices in
//!    the tallest group horizontally → one row vector of length
//!    sum(widths in group). Hash row-by-row with the leaf sponge:
//!        layers[0] = hash_rows(concat_rows, h_max, total_width_tallest)
//! 3. For each level k in 1..=log2(h_max):
//!       h_k  = h_max / 2^k
//!       prev = layers[k-1]
//!
//!       # First: binary compression of prev_layer pairs.
//!       temp[i] = compress(prev[2i], prev[2i+1])  for i in 0..h_k
//!
//!       # Inject if a matrix group exists at height h_k.
//!       if any matrix has height == h_k:
//!         inj_group = matrices with height == h_k
//!         inj_hash[i] = leaf.hash_rows of concatenated rows at i
//!         layers[k][i] = compress(temp[i], inj_hash[i])  for i in 0..h_k
//!       else:
//!         layers[k] = temp
//! ```
//!
//! # Buffer model (Stage 1b)
//!
//! Intermediate interleaving at injection levels happens **host-
//! side**. Each level's output is already downloaded to host to
//! populate `RetainedLayersHost` for future opens, so CPU interleave
//! piggybacks on that existing readback — no new dispatch on the
//! critical path. A GPU-resident injection kernel can land later if
//! bench data motivates it; Stage 1b's priority is correctness.

use std::collections::BTreeMap;
use std::convert::TryInto;

use zkgpu_babybear::BabyBear;
use zkgpu_core::{GpuBuffer, GpuDevice, ZkGpuError};

use crate::buffer::WgpuBuffer;
use crate::device::WgpuDevice;

use super::interleave_pairs::WgpuPoseidon2InterleavePairsPlan;
use super::merkle_commit::RetainedLayersHost;
use super::merkle_compress::WgpuPoseidon2MerkleCompressPlan;
use super::merkle_leaf::{WgpuPoseidon2MerkleLeafPlan, DIGEST_LEN};
use super::merkle_leaf_w16::WgpuPoseidon2MerkleLeafW16R8Plan;

/// Crate-private seam for the mixed-height DAG engine's leaf-hash
/// step. Deliberately not `pub` — the DAG is currently consumed only
/// by the two in-tree leaf shapes, and external callers use the
/// named entry points below (`..._with_w24_leaf`, `..._with_w16_leaf`)
/// rather than implementing the trait themselves. Keeping this
/// internal preserves the freedom to refactor the mixed-height
/// backend (e.g. fold a GPU-side injection kernel in) without a
/// public API break.
///
/// Implemented for both in-tree shapes:
/// * W24/RATE=16 (Plonky3 canonical)
/// * W16/RATE=8 (OpenVM)
///
/// Callers must produce output in the same layout the compression
/// kernel expects: a flat `num_leaves * DIGEST_LEN` `u32` buffer,
/// digest `i` at offset `i * DIGEST_LEN`.
pub(crate) trait GpuLeafSponge {
    fn hash_rows(
        &mut self,
        device: &WgpuDevice,
        input: &WgpuBuffer<BabyBear>,
        output: &mut WgpuBuffer<BabyBear>,
        num_leaves: u32,
        row_width: u32,
    ) -> Result<(), ZkGpuError>;
}

impl GpuLeafSponge for WgpuPoseidon2MerkleLeafPlan {
    fn hash_rows(
        &mut self,
        device: &WgpuDevice,
        input: &WgpuBuffer<BabyBear>,
        output: &mut WgpuBuffer<BabyBear>,
        num_leaves: u32,
        row_width: u32,
    ) -> Result<(), ZkGpuError> {
        self.hash_rows(device, input, output, num_leaves, row_width)
    }
}

impl GpuLeafSponge for WgpuPoseidon2MerkleLeafW16R8Plan {
    fn hash_rows(
        &mut self,
        device: &WgpuDevice,
        input: &WgpuBuffer<BabyBear>,
        output: &mut WgpuBuffer<BabyBear>,
        num_leaves: u32,
        row_width: u32,
    ) -> Result<(), ZkGpuError> {
        self.hash_rows(device, input, output, num_leaves, row_width)
    }
}

/// One matrix in a mixed-height commit batch.
///
/// `flat` is a host-side row-major slice of length `height * width`.
/// The engine validates this and uploads each height group into a
/// single GPU buffer.
#[derive(Clone, Copy, Debug)]
pub struct MixedHeightMatrixInput<'a> {
    pub flat: &'a [BabyBear],
    pub height: u32,
    pub width: u32,
}

/// Mixed-height Poseidon2 Merkle commit with full retained layers,
/// driven by the **W24/RATE=16 leaf sponge** (Plonky3 canonical
/// `PaddingFreeSponge<Perm24, 24, 16, 8>`). Thin wrapper over
/// [`commit_mixed_height_internal`]; concrete-typed so the generic
/// over the crate-private [`GpuLeafSponge`] trait stays internal.
///
/// See [`commit_mixed_height_internal`] for the full contract;
/// semantics are identical.
///
/// `interleave` is the GPU-resident pair-interleave plan used at
/// injection levels of the DAG. Same-height inputs (no injection)
/// take the fast path and never invoke `interleave`, but the
/// parameter is still required so the function signature stays
/// uniform across both shapes — callers construct one
/// [`WgpuPoseidon2InterleavePairsPlan`] per device at startup and
/// reuse it across every commit.
pub fn commit_mixed_height_with_w24_leaf(
    device: &WgpuDevice,
    leaf: &mut WgpuPoseidon2MerkleLeafPlan,
    compress: &mut WgpuPoseidon2MerkleCompressPlan,
    interleave: &mut WgpuPoseidon2InterleavePairsPlan,
    matrices: &[MixedHeightMatrixInput<'_>],
) -> Result<RetainedLayersHost, ZkGpuError> {
    commit_mixed_height_internal(device, leaf, compress, interleave, matrices)
}

/// Mixed-height Poseidon2 Merkle commit with full retained layers,
/// driven by the **W16/RATE=8 leaf sponge** (OpenVM
/// `PaddingFreeSponge<Perm16, 16, 8, 8>`). Thin wrapper over
/// [`commit_mixed_height_internal`].
pub fn commit_mixed_height_with_w16_leaf(
    device: &WgpuDevice,
    leaf: &mut WgpuPoseidon2MerkleLeafW16R8Plan,
    compress: &mut WgpuPoseidon2MerkleCompressPlan,
    interleave: &mut WgpuPoseidon2InterleavePairsPlan,
    matrices: &[MixedHeightMatrixInput<'_>],
) -> Result<RetainedLayersHost, ZkGpuError> {
    commit_mixed_height_internal(device, leaf, compress, interleave, matrices)
}

/// Generic mixed-height Poseidon2 Merkle commit implementation.
/// Crate-internal because the [`GpuLeafSponge`] trait it's generic
/// over is crate-private. Public access goes through the two
/// concrete-typed wrappers above.
///
/// Returns `layers[0..=log2(h_max)]` where
/// * `layers[0]` has `h_max * DIGEST_LEN` elements (leaf digests),
/// * `layers[k]` has `(h_max >> k) * DIGEST_LEN` elements, and
/// * `layers.last()` has exactly `DIGEST_LEN` elements (the root).
///
/// At levels where a matrix group of matching height is injected,
/// the retained layer is the **final** (post-inject) digest — the
/// intermediate pairwise-compression output is ephemeral. Opener
/// side (Stage 1c) needs the injection schedule to reconstruct the
/// tree; this engine returns the same flat structure as
/// [`super::merkle_commit::WgpuPoseidon2MerkleCommit::
/// commit_host_matrix_with_layers`] for compatibility.
///
/// # Preconditions
/// * `matrices.len() >= 1`
/// * every `height` is a power of two
/// * every `width >= 1`
/// * `flat.len() == height * width`
/// * matrices with the same `height` are legal (same-height group)
///
/// # Scope
/// Binary tree (`N = 2`), BabyBear, Plonky3-variant Poseidon2
/// constants, power-of-two heights. Matches the configured shape of
/// both Plonky3's canonical `Poseidon2MerkleMmcs` and OpenVM's
/// `baby_bear_poseidon2` (when paired with the matching leaf sponge).
pub(crate) fn commit_mixed_height_internal<L: GpuLeafSponge>(
    device: &WgpuDevice,
    leaf: &mut L,
    compress: &mut WgpuPoseidon2MerkleCompressPlan,
    interleave: &mut WgpuPoseidon2InterleavePairsPlan,
    matrices: &[MixedHeightMatrixInput<'_>],
) -> Result<RetainedLayersHost, ZkGpuError> {
    // --- Validation ---
    if matrices.is_empty() {
        return Err(ZkGpuError::InvalidNttSize(
            "mixed-height commit: matrices cannot be empty".into(),
        ));
    }
    for (i, m) in matrices.iter().enumerate() {
        if m.height == 0 {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "mixed-height commit: matrix {i} has height 0"
            )));
        }
        if !m.height.is_power_of_two() {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "mixed-height commit: matrix {i} has non-power-of-two height {}",
                m.height
            )));
        }
        if m.width == 0 {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "mixed-height commit: matrix {i} has width 0"
            )));
        }
        let expected = (m.height as usize)
            .checked_mul(m.width as usize)
            .ok_or_else(|| {
                ZkGpuError::InvalidNttSize(format!(
                    "mixed-height commit: matrix {i} shape overflow {}*{}",
                    m.height, m.width
                ))
            })?;
        if m.flat.len() != expected {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "mixed-height commit: matrix {i} len {} != height*width ({}*{}={})",
                m.flat.len(),
                m.height,
                m.width,
                expected
            )));
        }
    }

    // --- Group matrices by height (descending iteration via BTreeMap::rev). ---
    let mut by_height: BTreeMap<u32, Vec<&MixedHeightMatrixInput<'_>>> = BTreeMap::new();
    for m in matrices {
        by_height.entry(m.height).or_default().push(m);
    }

    // --- Same-height fast path ---
    //
    // When every matrix shares one height, `compress_and_inject` has
    // no injection levels to handle — the DAG degenerates into the
    // "leaf sponge once, then binary compress all the way to the
    // root" shape. The general path below correctly produces the
    // same output for this case, but pays a per-level host
    // round-trip (download `temp_gpu`, re-upload as `prev_gpu` next
    // iteration) so it can interleave injection digests host-side at
    // injection levels. With no injections that round-trip is pure
    // overhead — `log2(h_max)` PCIe pings of latency on a discrete
    // GPU, which is exactly where we measured a 17–30% commit-only
    // ratio regression on RTX 4090 / 5090 after Plonky3 adapter
    // convergence.
    //
    // The fast path mirrors the pre-convergence
    // `WgpuPoseidon2MerkleCommit::commit_with_retained_layers` shape:
    // pre-allocate every level's GPU buffer up front, run the leaf
    // sponge into level 0, run compression `level k → level k+1`
    // entirely GPU-resident, then download all layers in one batch
    // at the end. Output (`RetainedLayersHost`) shape is identical
    // to the general path, so opening / verify / parity tests don't
    // care which path produced it.
    if by_height.len() == 1 {
        let height = matrices[0].height;
        return commit_same_height_fast_path(
            device, leaf, compress, matrices, height,
        );
    }

    let heights_desc: Vec<u32> = by_height.keys().rev().copied().collect();
    let max_height = heights_desc[0];
    // Max-height tree depth. `log2(max_height) + 1` retained layers.
    let log_h_max = max_height.trailing_zeros() as usize;

    // --- GPU-resident path (item #1 of speed-opportunities) ---
    //
    // Pre-allocate every level's GPU buffer up front. Run leaf sponge
    // for the tallest group into level 0 directly. For each subsequent
    // level, run pairwise compression on GPU; at injection levels, run
    // injection-leaf-hash on GPU + GPU pair-interleave + second
    // compression — all without bouncing through host memory. Then
    // batch-download every retained layer in one final pass at the end.
    //
    // Pre-item-#1 the host-bouncing path paid `2 * log_h_max` PCIe
    // round-trips per commit at minimum (download `temp_gpu` + upload
    // `prev_gpu` per level), plus 3 extra round-trips at each injection
    // level (download temp + upload interleaved + download merged). On
    // discrete GPUs that's the dominant cost of mixed-height commits.

    let mut gpu_layers: Vec<WgpuBuffer<BabyBear>> = Vec::with_capacity(log_h_max + 1);
    let mut level_size = max_height;
    for _ in 0..=log_h_max {
        gpu_layers.push(
            device.alloc_zeros::<BabyBear>((level_size as usize) * DIGEST_LEN)?,
        );
        level_size /= 2;
    }

    // --- Level 0: hash tallest group → gpu_layers[0] ---
    let tallest = by_height
        .get(&max_height)
        .expect("max_height is a key we iterated");
    hash_group_gpu(device, leaf, tallest, max_height, &mut gpu_layers[0])?;

    // --- Compression chain, GPU-resident ---
    let mut current_height = max_height;
    for k in 1..=log_h_max {
        let next_height = current_height / 2;

        // Borrow `prev` (immutable) and `next` (mutable) from the same
        // Vec via `split_at_mut`, the standard pattern.
        let (prev_slice, next_slice) = gpu_layers.split_at_mut(k);
        let prev = &prev_slice[k - 1];
        let next = &mut next_slice[0];

        if let Some(inj_group) = by_height.get(&next_height) {
            // Injection level. Three GPU scratch buffers are needed:
            //   * `temp_gpu`   — pairwise compression of prev (N digests)
            //   * `inj_gpu`    — leaf-hash of injected group (N digests)
            //   * `interleaved_gpu` — pair-interleaved (temp, inj) (2N digests)
            // Then a second compression: interleaved_gpu → next.
            let n_digests = next_height as usize;
            let mut temp_gpu = device.alloc_zeros::<BabyBear>(n_digests * DIGEST_LEN)?;
            compress.compress_level(device, prev, &mut temp_gpu, next_height)?;

            let mut inj_gpu = device.alloc_zeros::<BabyBear>(n_digests * DIGEST_LEN)?;
            hash_group_gpu(device, leaf, inj_group, next_height, &mut inj_gpu)?;

            let mut interleaved_gpu =
                device.alloc_zeros::<BabyBear>(2 * n_digests * DIGEST_LEN)?;
            interleave.interleave(
                device,
                &temp_gpu,
                &inj_gpu,
                &mut interleaved_gpu,
                next_height,
            )?;

            compress.compress_level(device, &interleaved_gpu, next, next_height)?;
        } else {
            // No injection: pairwise-compress prev → next directly.
            compress.compress_level(device, prev, next, next_height)?;
        }

        current_height = next_height;
    }

    // --- Batch download every retained layer ---
    //
    // Single host-side phase at the end. The open path (`open_batch_mixed_height`)
    // walks these host-side layers; we have to materialize them
    // anyway, but doing all the downloads at once lets the driver
    // pipeline them rather than serializing across `log_h_max + 1`
    // separate `submit + poll(wait)` boundaries the way the host-
    // bouncing path did.
    let mut layers: Vec<Vec<BabyBear>> = Vec::with_capacity(gpu_layers.len());
    for buf in &gpu_layers {
        layers.push(buf.read_to_vec()?);
    }

    debug_assert_eq!(layers.len(), log_h_max + 1);
    debug_assert_eq!(
        layers.last().map(|l| l.len()).unwrap_or(0),
        DIGEST_LEN,
        "root layer must be exactly DIGEST_LEN elements"
    );
    Ok(RetainedLayersHost { layers })
}

/// Same-height fast path: pre-allocates every retained-layer GPU
/// buffer up front, runs the leaf sponge once into layer 0, runs the
/// binary compression chain entirely GPU-resident with no host
/// round-trip between levels, then downloads every layer in a single
/// batch at the end.
///
/// Output is byte-identical to the general
/// [`commit_mixed_height_internal`] path for inputs where every
/// matrix shares one height. Parity-pinned by the existing same-
/// height shape entries in
/// `tests/poseidon2_merkle_commit_dag_w24_gpu.rs` and
/// `tests/poseidon2_merkle_commit_dag_w16_gpu.rs`, plus the
/// adapter-level commit-parity suites in `zkgpu-plonky3` and
/// `zkgpu-openvm`.
///
/// Why it lives in the DAG engine rather than as a separate public
/// API: keeping the fast path **inside** `commit_mixed_height_*` lets
/// both consumer adapters benefit from the optimization without
/// either of them needing to special-case "is this a same-height
/// commit" — they always call the same `commit_mixed_height_with_*`
/// entry, the engine picks the right path internally.
fn commit_same_height_fast_path<L: GpuLeafSponge>(
    device: &WgpuDevice,
    leaf: &mut L,
    compress: &mut WgpuPoseidon2MerkleCompressPlan,
    matrices: &[MixedHeightMatrixInput<'_>],
    height: u32,
) -> Result<RetainedLayersHost, ZkGpuError> {
    // Validation already done by the caller; debug-assert the contract.
    debug_assert!(
        matrices.iter().all(|m| m.height == height),
        "fast path called with mixed heights"
    );

    // Row-major concat: row i of the joint matrix is row i of m0 then
    // m1 then ... — Plonky3's `first_digest_layer` semantics. For a
    // single-matrix input this still copies once, but that copy was
    // already required by the upload step in the general path.
    let total_width: u32 = matrices
        .iter()
        .map(|m| m.width)
        .try_fold(0u32, u32::checked_add)
        .ok_or_else(|| {
            ZkGpuError::InvalidNttSize(
                "same-height fast path: total width overflow".into(),
            )
        })?;

    let h = height as usize;
    let tw = total_width as usize;

    // Single upload — no further host round-trips until the final
    // batch download. Two cases:
    //
    // * **Single matrix:** the joint matrix IS the input, no concat
    //   needed. Upload `matrices[0].flat` directly. This matches the
    //   pre-convergence path's single bulk upload — skipping the
    //   concat saves an 8 MiB host-memcpy at h=2¹⁸ / w=8 and is the
    //   biggest contributor to closing the post-convergence
    //   commit-only ratio gap on faster GPUs (5090 measured).
    //
    // * **Multi-matrix same-height:** Plonky3's row-major joint-row
    //   semantics require a row-by-row interleave (row i = m0_row_i
    //   then m1_row_i then ...), which can't be done by a single
    //   bulk copy. Falls back to the per-row loop.
    let matrix_buf = if matrices.len() == 1 {
        device.upload::<BabyBear>(matrices[0].flat)?
    } else {
        let mut concat: Vec<BabyBear> = Vec::with_capacity(h * tw);
        for row in 0..h {
            for m in matrices {
                let w = m.width as usize;
                let start = row * w;
                concat.extend_from_slice(&m.flat[start..start + w]);
            }
        }
        device.upload::<BabyBear>(&concat)?
    };

    // Pre-allocate every layer's GPU buffer (level 0 = h * DIGEST_LEN
    // down to root = 1 * DIGEST_LEN).
    let log_h = height.trailing_zeros() as usize;
    let mut gpu_layers: Vec<WgpuBuffer<BabyBear>> = Vec::with_capacity(log_h + 1);
    let mut level_size = h;
    for _ in 0..=log_h {
        gpu_layers.push(device.alloc_zeros::<BabyBear>(level_size * DIGEST_LEN)?);
        level_size /= 2;
    }

    // Leaf sponge: matrix_buf → gpu_layers[0]
    leaf.hash_rows(device, &matrix_buf, &mut gpu_layers[0], height, total_width)?;

    // Compression chain, GPU-resident: gpu_layers[k] → gpu_layers[k+1]
    let mut num_outputs = height / 2;
    for k in 0..log_h {
        let (left, right) = gpu_layers.split_at_mut(k + 1);
        let input = &left[k];
        let output = &mut right[0];
        compress.compress_level(device, input, output, num_outputs)?;
        num_outputs /= 2;
    }

    // Single batch download at the end. This is the only host
    // transfer beyond the initial `concat` upload — vs the general
    // path's per-level download/upload pair, the fast path saves
    // `log_h` host round-trips.
    let mut layers: Vec<Vec<BabyBear>> = Vec::with_capacity(gpu_layers.len());
    for buf in &gpu_layers {
        layers.push(buf.read_to_vec()?);
    }

    debug_assert_eq!(layers.len(), log_h + 1);
    debug_assert_eq!(
        layers.last().map(|l| l.len()).unwrap_or(0),
        DIGEST_LEN,
        "fast path: root layer must be exactly DIGEST_LEN elements"
    );
    Ok(RetainedLayersHost { layers })
}

/// A row-opening produced by [`open_batch_mixed_height`].
///
/// Shape matches Plonky3's `BatchOpening` for binary `MerkleTreeMmcs`
/// at `cap_height = 0`:
///
/// * `opened_values[i]` is the row of matrix `i` (same order as the
///   `matrices` slice passed to the original commit call) at local
///   index `index >> (log2(h_max) - log2(matrices[i].height))`.
/// * `opening_proof` is `log2(h_max)` sibling digests, bottom-up
///   (level 0 first, root level last excluded).
///
/// Element order: BabyBear elements are in canonical form (not
/// Montgomery). Adapter-side (e.g. `zkgpu-openvm`) conversion to p3
/// BabyBear happens at the API boundary.
#[derive(Clone, Debug)]
pub struct MixedHeightOpening {
    pub opened_values: Vec<Vec<BabyBear>>,
    pub opening_proof: Vec<[BabyBear; DIGEST_LEN]>,
}

/// Produce a row-opening from a mixed-height commit's retained layers.
///
/// Consumers are the Stage-2 `zkgpu-openvm` adapter (and any future
/// consumer that needs Plonky3-compatible openings over the shared
/// backend).
///
/// # Arguments
/// * `matrices` — the **exact slice** passed to the original
///   [`commit_mixed_height_with_w24_leaf`] /
///   [`commit_mixed_height_with_w16_leaf`] call. Same length, same
///   order, same heights/widths. The function validates consistency
///   against the retained layers' inferred `h_max`.
/// * `retained` — the [`RetainedLayersHost`] returned by the commit.
/// * `index` — global row index in `0..h_max` where `h_max` is the
///   tallest matrix's height (== `retained.num_leaves()`).
///
/// # Errors
/// * empty matrices slice, or inconsistent with retained
/// * `index >= h_max`
/// * retained layers structurally invalid (wrong layer lengths,
///   missing layers)
///
/// # Proof shape
/// Matches Plonky3's `MerkleTreeMmcs::open_batch` bit-for-bit for
/// the binary N=2 / `cap_height=0` / power-of-two heights case.
/// `opening_proof.len() == log2(h_max)`; for `h_max == 1` the proof
/// is empty.
pub fn open_batch_mixed_height(
    matrices: &[MixedHeightMatrixInput<'_>],
    retained: &RetainedLayersHost,
    index: u32,
) -> Result<MixedHeightOpening, ZkGpuError> {
    if matrices.is_empty() {
        return Err(ZkGpuError::InvalidNttSize(
            "open_batch_mixed_height: matrices cannot be empty".into(),
        ));
    }
    if retained.layers.is_empty() {
        return Err(ZkGpuError::InvalidNttSize(
            "open_batch_mixed_height: retained layers cannot be empty".into(),
        ));
    }
    let h_max_layer = retained.layers[0].len() / DIGEST_LEN;
    if h_max_layer == 0 || !h_max_layer.is_power_of_two() {
        return Err(ZkGpuError::InvalidNttSize(format!(
            "open_batch_mixed_height: retained layer 0 has {} digests, \
             expected a nonzero power of two",
            h_max_layer
        )));
    }
    let h_max = h_max_layer as u32;
    if index >= h_max {
        return Err(ZkGpuError::InvalidNttSize(format!(
            "open_batch_mixed_height: index {index} out of bounds (h_max={h_max})"
        )));
    }
    let log_h_max = h_max.trailing_zeros() as usize;
    if retained.layers.len() != log_h_max + 1 {
        return Err(ZkGpuError::InvalidNttSize(format!(
            "open_batch_mixed_height: retained has {} layers, expected {}",
            retained.layers.len(),
            log_h_max + 1
        )));
    }

    // --- Opened values: one row per matrix, shifted to the matrix's
    //     local coordinate via `index >> (log_h_max - log_h_i)`. ---
    let mut opened_values: Vec<Vec<BabyBear>> = Vec::with_capacity(matrices.len());
    for (i, m) in matrices.iter().enumerate() {
        if m.height == 0 || !m.height.is_power_of_two() {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "open_batch_mixed_height: matrix {i} has invalid height {}",
                m.height
            )));
        }
        if m.height > h_max {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "open_batch_mixed_height: matrix {i} height {} > h_max {h_max} \
                 (commit input/retained mismatch)",
                m.height
            )));
        }
        let log_h = m.height.trailing_zeros() as usize;
        let bits_reduced = log_h_max - log_h;
        let local_idx = (index >> bits_reduced) as usize;
        let w = m.width as usize;
        let start = local_idx
            .checked_mul(w)
            .ok_or_else(|| {
                ZkGpuError::InvalidNttSize(format!(
                    "open_batch_mixed_height: matrix {i} row offset overflow"
                ))
            })?;
        let end = start.checked_add(w).ok_or_else(|| {
            ZkGpuError::InvalidNttSize(format!(
                "open_batch_mixed_height: matrix {i} row end overflow"
            ))
        })?;
        if end > m.flat.len() {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "open_batch_mixed_height: matrix {i} row [{start}, {end}) \
                 out of bounds (flat.len={})",
                m.flat.len()
            )));
        }
        opened_values.push(m.flat[start..end].to_vec());
    }

    // --- Proof: bottom-up sibling chain through retained layers.
    //     At level k, sibling = layers[k][(idx >> k) ^ 1]. ---
    let mut proof: Vec<[BabyBear; DIGEST_LEN]> = Vec::with_capacity(log_h_max);
    let mut idx = index as usize;
    for k in 0..log_h_max {
        let sibling_pos = idx ^ 1;
        let layer = &retained.layers[k];
        let expected_layer_len = (h_max_layer >> k) * DIGEST_LEN;
        if layer.len() != expected_layer_len {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "open_batch_mixed_height: retained layer {k} has {} entries, \
                 expected {} ({} digests × {DIGEST_LEN})",
                layer.len(),
                expected_layer_len,
                h_max_layer >> k
            )));
        }
        let start = sibling_pos * DIGEST_LEN;
        let end = start + DIGEST_LEN;
        let sibling: [BabyBear; DIGEST_LEN] = layer[start..end]
            .try_into()
            .expect("slice length checked against expected_layer_len");
        proof.push(sibling);
        idx >>= 1;
    }

    Ok(MixedHeightOpening {
        opened_values,
        opening_proof: proof,
    })
}

/// Extract the root (top layer) from a `RetainedLayersHost` as an
/// 8-element `[BabyBear; DIGEST_LEN]`.
///
/// `RetainedLayersHost` is publicly constructible (`layers` is a
/// `pub Vec<Vec<BabyBear>>`), so callers can in principle feed in
/// malformed data. This helper requires `top.len() == DIGEST_LEN`
/// exactly — a "top" of 16 or 24 elements would silently slice the
/// first 8 under a looser `>= DIGEST_LEN` check and hand back a
/// plausible-but-wrong root. Exact-length catches corruption and
/// caller bugs at the boundary instead.
pub fn root_from_retained(
    retained: &RetainedLayersHost,
) -> Result<[BabyBear; DIGEST_LEN], ZkGpuError> {
    let top = retained
        .layers
        .last()
        .ok_or_else(|| ZkGpuError::InvalidNttSize("empty retained layers".into()))?;
    if top.len() != DIGEST_LEN {
        return Err(ZkGpuError::InvalidNttSize(format!(
            "retained layers top has {} entries, expected exactly {DIGEST_LEN} \
             (a malformed top layer would silently slice to a plausible-but-wrong root)",
            top.len()
        )));
    }
    let root: [BabyBear; DIGEST_LEN] = top[..DIGEST_LEN]
        .try_into()
        .expect("slice length checked above");
    Ok(root)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Hash one height group, writing digests to a caller-provided GPU
/// buffer. Matches Plonky3's `first_digest_layer` / `compress_and_inject`
/// row-hashing: for each row index `i`, the hash input is
/// `flat_map(matrix.row(i) for matrix in group)`.
///
/// Performs the row concat host-side (cheap — we already hold the
/// per-matrix `flat: &[BabyBear]` slices) and uploads once, then runs
/// the leaf sponge dispatch into the caller's `output` buffer with no
/// download. Replaces the older `hash_group → Vec<BabyBear>` shape
/// that bounced every level's hashes through host memory; this
/// version is the GPU-resident counterpart used by the mixed-height
/// commit path landed in item #1 of the speed-opportunities phase.
fn hash_group_gpu<L: GpuLeafSponge>(
    device: &WgpuDevice,
    leaf: &mut L,
    group: &[&MixedHeightMatrixInput<'_>],
    group_height: u32,
    output: &mut WgpuBuffer<BabyBear>,
) -> Result<(), ZkGpuError> {
    debug_assert!(
        group.iter().all(|m| m.height == group_height),
        "hash_group_gpu: group must be uniform-height"
    );
    let total_width: u32 = group
        .iter()
        .map(|m| m.width)
        .try_fold(0u32, u32::checked_add)
        .ok_or_else(|| {
            ZkGpuError::InvalidNttSize(
                "mixed-height commit: group total width overflow".into(),
            )
        })?;

    let required_out = (group_height as usize) * DIGEST_LEN;
    if output.len() < required_out {
        return Err(ZkGpuError::InvalidNttSize(format!(
            "hash_group_gpu output length {} < group_height*DIGEST_LEN = {}",
            output.len(),
            required_out,
        )));
    }

    // Row-major concat: row i of the concat is row i of m0, then row i
    // of m1, etc. This is the Plonky3 semantics. Single-matrix
    // shortcut bypasses the row loop and copies the input slice
    // directly (identical to the same-height fast path's optimization).
    let input_gpu = if group.len() == 1 {
        device.upload::<BabyBear>(group[0].flat)?
    } else {
        let mut concat: Vec<BabyBear> =
            Vec::with_capacity((group_height as usize) * (total_width as usize));
        for row in 0..group_height as usize {
            for m in group {
                let w = m.width as usize;
                let start = row * w;
                concat.extend_from_slice(&m.flat[start..start + w]);
            }
        }
        device.upload::<BabyBear>(&concat)?
    };

    leaf.hash_rows(device, &input_gpu, output, group_height, total_width)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Unit coverage (shape checks only; parity lives in the integration suite).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Note: the previous `interleave_pairs_host_shape` test was
    // removed when item #1 (GPU-resident mixed-height injection)
    // landed — the host-side interleave function it tested no longer
    // exists. The GPU equivalent
    // (`WgpuPoseidon2InterleavePairsPlan::interleave`) is covered by
    // the existing mixed-height integration parity tests in
    // `zkgpu-plonky3/tests/poseidon2_merkle_commit_dag_*.rs` and
    // `zkgpu-plonky3/tests/poseidon2_merkle_open_dag_gpu.rs`, which
    // would catch any divergence at the byte level against Plonky3's
    // CPU `MerkleTreeMmcs`.

    #[test]
    fn root_from_retained_rejects_empty_layers() {
        let retained = RetainedLayersHost { layers: Vec::new() };
        let err = root_from_retained(&retained);
        assert!(err.is_err(), "empty layers must reject");
    }

    #[test]
    fn root_from_retained_rejects_undersized_top() {
        // Top has fewer than DIGEST_LEN elements.
        let retained = RetainedLayersHost {
            layers: vec![(0..4u32).map(BabyBear::new).collect()],
        };
        let err = root_from_retained(&retained);
        assert!(err.is_err(), "top layer of len < DIGEST_LEN must reject");
    }

    #[test]
    fn root_from_retained_rejects_oversized_top() {
        // Regression guard for the P3 review finding: a top layer of
        // 16 or 24 elements would silently slice the first 8 under a
        // `>=` check and hand back a plausible-but-wrong root. Now
        // requires exact `== DIGEST_LEN`.
        for bogus_len in [9u32, 16, 24, 32] {
            let retained = RetainedLayersHost {
                layers: vec![(0..bogus_len).map(BabyBear::new).collect()],
            };
            let err = root_from_retained(&retained);
            assert!(
                err.is_err(),
                "top layer of len {bogus_len} (> DIGEST_LEN) must reject, not silently slice"
            );
        }
    }

    #[test]
    fn root_from_retained_accepts_exact_length() {
        let retained = RetainedLayersHost {
            layers: vec![(42..50u32).map(BabyBear::new).collect()],
        };
        let root = root_from_retained(&retained).expect("valid top must succeed");
        for i in 0..DIGEST_LEN {
            assert_eq!(root[i].0, 42 + i as u32);
        }
    }
}
