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

use super::merkle_commit::RetainedLayersHost;
use super::merkle_compress::WgpuPoseidon2MerkleCompressPlan;
use super::merkle_leaf::{WgpuPoseidon2MerkleLeafPlan, DIGEST_LEN};
use super::merkle_leaf_w16::WgpuPoseidon2MerkleLeafW16R8Plan;

/// Shared seam for the mixed-height DAG engine's leaf-hash step.
///
/// Implemented for both in-tree shapes:
/// * W24/RATE=16 (Plonky3 canonical)
/// * W16/RATE=8 (OpenVM)
///
/// Caller-supplied sponges must produce output in the same layout
/// the compression kernel expects: a flat `num_leaves * DIGEST_LEN`
/// `u32` buffer, digest `i` at offset `i * DIGEST_LEN`.
pub trait GpuLeafSponge {
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

/// Mixed-height Poseidon2 Merkle commit with full retained layers.
///
/// Returns `layers[0..=log2(h_max)]` where
/// * `layers[0]` has `h_max * DIGEST_LEN` elements (leaf digests),
/// * `layers[k]` has `(h_max >> k) * DIGEST_LEN` elements, and
/// * `layers.last()` has `DIGEST_LEN` elements (the root).
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
/// * matrices with the same `height` are legal (same-height group);
///   differing heights must not round up to the same power of two
///   (which is vacuous when all heights are already powers of two —
///   so distinct heights are all admissible)
///
/// # Scope
/// Binary tree (`N = 2`), BabyBear, Plonky3-variant Poseidon2
/// constants, power-of-two heights. Matches the configured shape of
/// both Plonky3's canonical `Poseidon2MerkleMmcs` and OpenVM's
/// `baby_bear_poseidon2` (when paired with the matching leaf sponge).
pub fn commit_mixed_height_host_matrices_with_retained_layers<L: GpuLeafSponge>(
    device: &WgpuDevice,
    leaf: &mut L,
    compress: &mut WgpuPoseidon2MerkleCompressPlan,
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
    let heights_desc: Vec<u32> = by_height.keys().rev().copied().collect();
    let max_height = heights_desc[0];
    // Max-height tree depth. `log2(max_height) + 1` retained layers.
    let log_h_max = max_height.trailing_zeros() as usize;

    // --- Level 0: hash the tallest group ---
    let tallest = by_height
        .get(&max_height)
        .expect("max_height is a key we iterated");
    let layer0_host = hash_group(device, leaf, tallest, max_height)?;
    let mut layers: Vec<Vec<BabyBear>> = Vec::with_capacity(log_h_max + 1);
    layers.push(layer0_host);

    // --- Compression levels ---
    // At each level, the input is `layers[k-1]` (already on host). We
    // upload it to GPU, run compression, download temp; if the level
    // matches an injected group's height, we hash the injected rows,
    // interleave host-side, run a second compression, download final.
    let mut current_height = max_height;
    for _k in 1..=log_h_max {
        let next_height = current_height / 2;
        let prev_host = layers.last().expect("layers non-empty").clone();

        // Step 1: standard pairwise compression of prev_layer.
        let prev_gpu = device.upload::<BabyBear>(&prev_host)?;
        let mut temp_gpu =
            device.alloc_zeros::<BabyBear>((next_height as usize) * DIGEST_LEN)?;
        compress.compress_level(device, &prev_gpu, &mut temp_gpu, next_height)?;

        // Step 2: if a matrix group exists at this height, inject.
        let next_host = if let Some(inj_group) = by_height.get(&next_height) {
            let inj_hash_host = hash_group(device, leaf, inj_group, next_height)?;
            let temp_host = temp_gpu.read_to_vec()?;
            let interleaved =
                interleave_pairs_host(&temp_host, &inj_hash_host, next_height);
            let interleaved_gpu = device.upload::<BabyBear>(&interleaved)?;
            let mut merged_gpu =
                device.alloc_zeros::<BabyBear>((next_height as usize) * DIGEST_LEN)?;
            compress.compress_level(
                device,
                &interleaved_gpu,
                &mut merged_gpu,
                next_height,
            )?;
            merged_gpu.read_to_vec()?
        } else {
            temp_gpu.read_to_vec()?
        };

        layers.push(next_host);
        current_height = next_height;
    }

    debug_assert_eq!(layers.len(), log_h_max + 1);
    debug_assert_eq!(
        layers.last().map(|l| l.len()).unwrap_or(0),
        DIGEST_LEN,
        "root layer must be exactly DIGEST_LEN elements"
    );
    Ok(RetainedLayersHost { layers })
}

/// Extract the root (top layer) from a `RetainedLayersHost` as an
/// 8-element `[BabyBear; DIGEST_LEN]`. Re-exported here for
/// convenience with the DAG engine; the root is just
/// `layers.last().try_into().unwrap()` by construction.
pub fn root_from_retained(
    retained: &RetainedLayersHost,
) -> Result<[BabyBear; DIGEST_LEN], ZkGpuError> {
    let top = retained
        .layers
        .last()
        .ok_or_else(|| ZkGpuError::InvalidNttSize("empty retained layers".into()))?;
    if top.len() < DIGEST_LEN {
        return Err(ZkGpuError::InvalidNttSize(format!(
            "retained layers top has {} < {DIGEST_LEN} entries",
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

/// Hash one height group. Matches Plonky3's `first_digest_layer` /
/// `compress_and_inject` row-hashing: for each row index `i`, the
/// hash input is `flat_map(matrix.row(i) for matrix in group)`.
///
/// Performs the flattening host-side — conceptually cheap (we're
/// already holding `flat: &[BabyBear]` per matrix), and avoids a new
/// GPU gather kernel. Uploads the concatenated matrix as one buffer,
/// runs the leaf sponge in a single dispatch, downloads the digests.
fn hash_group<L: GpuLeafSponge>(
    device: &WgpuDevice,
    leaf: &mut L,
    group: &[&MixedHeightMatrixInput<'_>],
    group_height: u32,
) -> Result<Vec<BabyBear>, ZkGpuError> {
    debug_assert!(
        group.iter().all(|m| m.height == group_height),
        "hash_group: group must be uniform-height"
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

    // Row-major concat: row i of the concat is row i of m0, then row i
    // of m1, etc. This is the Plonky3 semantics.
    let mut concat: Vec<BabyBear> =
        Vec::with_capacity((group_height as usize) * (total_width as usize));
    for row in 0..group_height as usize {
        for m in group {
            let w = m.width as usize;
            let start = row * w;
            concat.extend_from_slice(&m.flat[start..start + w]);
        }
    }

    let input_gpu = device.upload::<BabyBear>(&concat)?;
    let mut digest_gpu =
        device.alloc_zeros::<BabyBear>((group_height as usize) * DIGEST_LEN)?;
    leaf.hash_rows(device, &input_gpu, &mut digest_gpu, group_height, total_width)?;
    digest_gpu.read_to_vec()
}

/// Host-side pair interleave: takes two size-`n` digest arrays and
/// produces a size-`2n` digest array where pair `i` is
/// `(left[i], right[i])`. Used at injection levels so the existing
/// pair-sibling compression kernel can consume the result directly
/// without needing a two-input compress variant.
fn interleave_pairs_host(left: &[BabyBear], right: &[BabyBear], n: u32) -> Vec<BabyBear> {
    let n = n as usize;
    debug_assert_eq!(left.len(), n * DIGEST_LEN);
    debug_assert_eq!(right.len(), n * DIGEST_LEN);
    let mut out = Vec::with_capacity(2 * n * DIGEST_LEN);
    for i in 0..n {
        let base = i * DIGEST_LEN;
        out.extend_from_slice(&left[base..base + DIGEST_LEN]);
        out.extend_from_slice(&right[base..base + DIGEST_LEN]);
    }
    out
}

// ---------------------------------------------------------------------------
// Unit coverage (shape checks only; parity lives in the integration suite).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interleave_pairs_host_shape() {
        let left: Vec<BabyBear> =
            (0..16u32).map(BabyBear::new).collect(); // 2 digests (n=2, DIGEST_LEN=8)
        let right: Vec<BabyBear> =
            (100..116u32).map(BabyBear::new).collect();
        let out = interleave_pairs_host(&left, &right, 2);
        assert_eq!(out.len(), 32);
        // Pair 0: left[0..8] then right[0..8].
        for k in 0..8u32 {
            assert_eq!(out[k as usize].0, k);
            assert_eq!(out[8 + k as usize].0, 100 + k);
        }
        // Pair 1: left[8..16] then right[8..16].
        for k in 0..8u32 {
            assert_eq!(out[16 + k as usize].0, 8 + k);
            assert_eq!(out[24 + k as usize].0, 108 + k);
        }
    }
}
