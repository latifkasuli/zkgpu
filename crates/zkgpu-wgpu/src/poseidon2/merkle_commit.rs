//! Phase 7 Step 3.b — GPU Poseidon2 Merkle commit orchestrator.
//!
//! Composes [`super::merkle_leaf::WgpuPoseidon2MerkleLeafPlan`] and
//! [`super::merkle_compress::WgpuPoseidon2MerkleCompressPlan`] into the
//! full Plonky3 `Poseidon2MerkleMmcs::commit` pipeline, staying
//! GPU-resident throughout:
//!
//! 1. Leaf sponge: `h × w` matrix  →  `h × DIGEST_LEN` digests
//!    (one `PaddingFreeSponge<Perm24, 24, 16, 8>` per row).
//! 2. Tree compression: `log₂(h)` width-16
//!    `TruncatedPermutation<Perm16, 2, 8, 16>` dispatches, each halving
//!    the previous layer.
//! 3. Read back the 8-element root as `[BabyBear; DIGEST_LEN]`.
//!
//! # Scope
//!
//! This first GPU commit path covers the **single-matrix,
//! power-of-two-height** case, which is the shape Plonky3's
//! `TwoAdicFriPcs::commit` feeds into FRI's commit phase in normal
//! operation (coset-LDE output is always power-of-two-high). Multi-
//! matrix commits and non-power-of-two heights would need Plonky3's
//! `compress_and_inject` injection logic from `p3_merkle_tree`; not
//! needed for the Step 3 bench gate.
//!
//! # Buffer model
//!
//! Tree compression ping-pongs between two scratch buffers of size
//! `h/2 * DIGEST_LEN` (each level's output fits). After `log₂(h)`
//! compression passes, the final root lives at offset 0 of one of the
//! two scratch buffers — [`Self::commit`] reads the 8-element slice
//! from there.

use std::convert::TryInto;

use zkgpu_babybear::BabyBear;
use zkgpu_core::{GpuBuffer, GpuDevice, ZkGpuError};
use zkgpu_poseidon2::Poseidon2Params;

use crate::buffer::WgpuBuffer;
use crate::device::WgpuDevice;

use super::merkle_compress::WgpuPoseidon2MerkleCompressPlan;
use super::merkle_leaf::{WgpuPoseidon2MerkleLeafPlan, DIGEST_LEN};

/// GPU-resident Plonky3 Poseidon2 Merkle commit.
pub struct WgpuPoseidon2MerkleCommit {
    leaf: WgpuPoseidon2MerkleLeafPlan,
    compress: WgpuPoseidon2MerkleCompressPlan,
}

/// All digest layers retained from a single commit. Index `k` holds the
/// digests at tree level `k`:
///
/// * `layers[0]` — `h` leaf digests produced by the Poseidon2 leaf sponge.
/// * `layers[k]` for `0 < k < log2(h)` — `h / 2^k` compression outputs.
/// * `layers[log2(h)]` — the single-element root.
///
/// Each `Vec<BabyBear>` is a flat `num_digests_at_level * DIGEST_LEN`
/// buffer; digest `i` at level `k` is `layers[k][i*DIGEST_LEN .. (i+1)*DIGEST_LEN]`.
///
/// This is the Step 3.c opening-friendly prover data: `open_batch(idx)`
/// walks `layers[0..log2(h)]` extracting `layers[k][(idx >> k) ^ 1]` as
/// the sibling at level `k`, no CPU tree rebuild. Total host storage is
/// `(2h - 1) * DIGEST_LEN * 4` bytes. With `DIGEST_LEN = 8` that's
/// `≈ 64·h` bytes — about **16 MiB at h=2¹⁸**.
#[derive(Debug, Clone)]
pub struct RetainedLayersHost {
    pub layers: Vec<Vec<BabyBear>>,
}

impl RetainedLayersHost {
    /// The Merkle root (tree's top layer, always 1 digest).
    pub fn root(&self) -> [BabyBear; DIGEST_LEN] {
        let top = self.layers.last().expect("retained layers: at least one level");
        top[..DIGEST_LEN]
            .try_into()
            .expect("retained layers: top layer has at least DIGEST_LEN entries")
    }

    /// Number of leaf digests (`h`).
    pub fn num_leaves(&self) -> usize {
        if self.layers.is_empty() {
            0
        } else {
            self.layers[0].len() / DIGEST_LEN
        }
    }

    /// Read the digest at level `level`, index `idx` (0-based). Returns
    /// `None` if either is out of bounds.
    pub fn digest_at(&self, level: usize, idx: usize) -> Option<[BabyBear; DIGEST_LEN]> {
        let layer = self.layers.get(level)?;
        let start = idx.checked_mul(DIGEST_LEN)?;
        let end = start.checked_add(DIGEST_LEN)?;
        if end > layer.len() {
            return None;
        }
        Some(layer[start..end].try_into().expect("slice len == DIGEST_LEN"))
    }
}

impl WgpuPoseidon2MerkleCommit {
    /// Build a commit orchestrator from matched Plonky3 Poseidon2 params:
    /// * `leaf_params` — width-24 Poseidon2 for the leaf sponge
    ///   (`PaddingFreeSponge<Perm24, 24, 16, 8>`).
    /// * `compress_params` — width-16 Poseidon2 for the tree
    ///   compression (`TruncatedPermutation<Perm16, 2, 8, 16>`).
    ///
    /// Both must be Plonky3-variant (M_4 = circ(2, 3, 1, 1)) and α = 7
    /// — the sub-plans enforce this.
    pub fn new(
        device: &WgpuDevice,
        leaf_params: Poseidon2Params<BabyBear, 24>,
        compress_params: Poseidon2Params<BabyBear, 16>,
    ) -> Result<Self, ZkGpuError> {
        let leaf = WgpuPoseidon2MerkleLeafPlan::new(device, leaf_params)?;
        let compress = WgpuPoseidon2MerkleCompressPlan::new(device, compress_params)?;
        Ok(Self { leaf, compress })
    }

    /// Digest length produced per leaf and per tree node.
    pub const fn digest_len(&self) -> usize {
        DIGEST_LEN
    }

    /// Commit a GPU-resident `h × w` row-major matrix and return the
    /// 8-element Merkle root.
    ///
    /// This is the production seam — Step 2's future GPU-resident
    /// `coset_lde_batch` will feed its output directly into here, and
    /// the Plonky3 `Mmcs` adapter routes through this path. All shape
    /// guards live here (not just in [`Self::commit_host_matrix`]) so
    /// that GPU-resident callers can't bypass them.
    ///
    /// # Preconditions
    /// * `h` must be `≥ 1` and a power of two (Step 3.b scope).
    ///   `h == 1` is valid — no compression runs and the leaf-sponge
    ///   output IS the root.
    /// * `w` must be `≥ 1`. Plonky3's `MerkleTreeMmcs::commit` panics
    ///   on an `h × 0` matrix (a row with no field elements has no
    ///   meaningful digest in the target stack); we reject it up front
    ///   rather than letting the kernel produce an arbitrary constant
    ///   digest that no CPU reference would agree with.
    /// * `matrix.len() == h * w`.
    pub fn commit(
        &mut self,
        device: &WgpuDevice,
        matrix: &WgpuBuffer<BabyBear>,
        h: u32,
        w: u32,
    ) -> Result<[BabyBear; DIGEST_LEN], ZkGpuError> {
        if h == 0 {
            return Err(ZkGpuError::InvalidNttSize(
                "Merkle commit: h must be ≥ 1 (empty matrix not supported)".into(),
            ));
        }
        if w == 0 {
            return Err(ZkGpuError::InvalidNttSize(
                "Merkle commit: w must be ≥ 1 (zero-width rows have no \
                 defined digest in the Plonky3 target stack)"
                    .into(),
            ));
        }
        if !h.is_power_of_two() {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "Merkle commit: h must be a power of two (got {h}); \
                 non-power-of-two support is deferred to multi-matrix Step 3.c"
            )));
        }
        let expected_input_len = (h as usize) * (w as usize);
        if matrix.len() != expected_input_len {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "Merkle commit input length {} != h*w ({}*{}={})",
                matrix.len(),
                h,
                w,
                expected_input_len,
            )));
        }

        // --- 1. Leaf sponge ---
        // Leaf output = h digests of DIGEST_LEN elements.
        let leaf_buf_len = (h as usize) * DIGEST_LEN;
        let mut leaf_out = device.alloc_zeros::<BabyBear>(leaf_buf_len)?;
        self.leaf.hash_rows(device, matrix, &mut leaf_out, h, w)?;

        // --- 2. Tree compression (log₂(h) ping-pong passes) ---
        if h == 1 {
            return read_root(&leaf_out);
        }

        // Ping-pong buffers. Each has capacity for h/2 digests (the
        // largest intermediate level). The leaf output `leaf_out` is
        // consumed by the first compression pass, then alternates
        // between `pong` and a reused `ping` (reallocated to the right
        // size below).
        let max_level_len = (h as usize / 2) * DIGEST_LEN;
        let mut ping = device.alloc_zeros::<BabyBear>(max_level_len)?;
        let mut pong = device.alloc_zeros::<BabyBear>(max_level_len)?;

        // First pass: leaf_out (h digests) → ping (h/2 digests).
        let mut num_outputs = h / 2;
        self.compress.compress_level(
            device,
            &leaf_out,
            &mut ping,
            num_outputs,
        )?;

        // Remaining passes alternate ping ↔ pong.
        let mut read_from_ping = true;
        while num_outputs > 1 {
            let next_num = num_outputs / 2;
            if read_from_ping {
                self.compress.compress_level(device, &ping, &mut pong, next_num)?;
            } else {
                self.compress.compress_level(device, &pong, &mut ping, next_num)?;
            }
            read_from_ping = !read_from_ping;
            num_outputs = next_num;
        }

        // After the final pass, the single root digest is in whichever
        // buffer was the WRITE target of that pass.
        //
        // Track: after the first pass we wrote to `ping` → read_from_ping
        // = true. Each subsequent iteration flips `read_from_ping`
        // AFTER writing. So `read_from_ping` after the loop points to
        // the READ buffer for the NEXT (non-existent) pass — which
        // means the most recent WRITE went to the OTHER buffer.
        if read_from_ping {
            read_root(&ping)
        } else {
            read_root(&pong)
        }
    }

    /// Convenience: upload a host matrix, commit, return the root.
    ///
    /// Prefer [`Self::commit`] with GPU-resident buffers in production
    /// (e.g. fed by a GPU coset-LDE in Step 2) to avoid the round-trip.
    ///
    /// Shape contract is whatever [`Self::commit`] enforces — this
    /// wrapper only performs pre-upload validation so we don't feed
    /// wgpu's `create_buffer_init` a zero-length slice when `h` or `w`
    /// is zero (wgpu rejects that with a panic deep in the buffer
    /// init path, which is a worse failure mode than our explicit
    /// error). All other guards (power-of-two, `matrix.len == h*w`)
    /// fire inside `commit()` itself.
    pub fn commit_host_matrix(
        &mut self,
        device: &WgpuDevice,
        matrix: &[BabyBear],
        h: u32,
        w: u32,
    ) -> Result<[BabyBear; DIGEST_LEN], ZkGpuError> {
        // Pre-upload shape check: `h * w` might overflow u32, and
        // upload on an empty slice panics inside wgpu. Both are
        // recoverable as `InvalidNttSize` errors; surface them
        // ourselves so the caller sees a clean error rather than a
        // panic.
        let expected_len = (h as usize)
            .checked_mul(w as usize)
            .ok_or_else(|| {
                ZkGpuError::InvalidNttSize(format!(
                    "Merkle commit host matrix shape overflow: {h} * {w}",
                ))
            })?;
        if matrix.len() != expected_len {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "Merkle commit host matrix length {} != h*w ({}*{}={})",
                matrix.len(),
                h,
                w,
                expected_len,
            )));
        }
        // Delegate the h==0 / w==0 / power-of-two rejections to
        // `commit()` so there's exactly one source of truth for the
        // shape contract. Special-case only the upload step: wgpu's
        // `create_buffer_init` panics on an empty slice, and
        // `expected_len == 0` is one of the two zero-shape inputs
        // `commit()` rejects — short-circuit into `commit()` with a
        // dummy 1-element buffer so `commit()` errors cleanly without
        // tripping the wgpu panic on the way in.
        if expected_len == 0 {
            let dummy = device.upload::<BabyBear>(&[BabyBear::new(0)])?;
            return self.commit(device, &dummy, h, w);
        }
        let matrix_buf = device.upload::<BabyBear>(matrix)?;
        self.commit(device, &matrix_buf, h, w)
    }

    // ----- Step 3.c: retained-layers commit --------------------------------
    //
    // Used by the `GpuPoseidon2Mmcs` Plonky3 adapter to serve openings
    // without rebuilding a CPU tree. The root-only `commit` above stays as
    // the lightweight production path (bench gate, coset_lde Step 2 future
    // plumbing); `commit_with_retained_layers` is strictly additive.
    //
    // Backend structure:
    //   * replaces the 2-buffer ping-pong with log2(h)+1 per-level
    //     allocations (one buffer per tree level). Total VRAM is
    //     (2h - 1) * DIGEST_LEN * 4 bytes; with DIGEST_LEN = 8 that's
    //     ≈64·h bytes — about 16 MiB at h=2^18. Still trivial on any
    //     discrete GPU; the next budget threshold worth thinking about
    //     is h=2^22 (≈256 MiB), where an ICICLE-style retain-upper /
    //     recompute-lower cutoff starts making sense.
    //   * each compression writes into a fresh buffer, so nothing is
    //     clobbered; the full tree is available afterward.
    //   * `commit_host_matrix_with_layers` downloads every layer to host
    //     in one pass so the adapter can index openings without GPU
    //     round-trips.

    /// GPU-resident commit that retains every digest layer. Same shape
    /// contract as [`Self::commit`] (h ≥ 1, h power of two, w ≥ 1,
    /// `matrix.len() == h*w`). Returns the full layer stack as GPU
    /// buffers, bottom-up: `out[0]` holds the `h` leaf digests,
    /// `out[log₂(h)]` holds the 1-element root.
    ///
    /// Step 3.c uses this path. Downstream code can either keep the
    /// layers GPU-resident for future openings or download them via
    /// [`Self::commit_host_matrix_with_layers`].
    pub fn commit_with_retained_layers(
        &mut self,
        device: &WgpuDevice,
        matrix: &WgpuBuffer<BabyBear>,
        h: u32,
        w: u32,
    ) -> Result<Vec<WgpuBuffer<BabyBear>>, ZkGpuError> {
        if h == 0 {
            return Err(ZkGpuError::InvalidNttSize(
                "Merkle commit_with_retained_layers: h must be ≥ 1".into(),
            ));
        }
        if w == 0 {
            return Err(ZkGpuError::InvalidNttSize(
                "Merkle commit_with_retained_layers: w must be ≥ 1".into(),
            ));
        }
        if !h.is_power_of_two() {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "Merkle commit_with_retained_layers: h must be a power of two (got {h})"
            )));
        }
        let expected_input_len = (h as usize) * (w as usize);
        if matrix.len() != expected_input_len {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "Merkle commit_with_retained_layers input length {} != h*w ({}*{}={})",
                matrix.len(),
                h,
                w,
                expected_input_len,
            )));
        }

        // Allocate all layer buffers up front. `layers[0]` size = h,
        // then halving per level until `layers[log_h]` size = 1.
        let log_h = h.trailing_zeros() as usize;
        let mut layers: Vec<WgpuBuffer<BabyBear>> = Vec::with_capacity(log_h + 1);
        let mut level_h = h as usize;
        for _ in 0..=log_h {
            layers.push(device.alloc_zeros::<BabyBear>(level_h * DIGEST_LEN)?);
            level_h /= 2;
        }

        // Leaf sponge: matrix → layers[0].
        self.leaf.hash_rows(device, matrix, &mut layers[0], h, w)?;

        // Tree compression: layers[k] → layers[k+1].
        // Splitting a Vec into two mutable references to adjacent
        // elements requires `split_at_mut`; that keeps the borrow
        // checker happy across the per-level loop.
        let mut num_outputs = h / 2;
        for k in 0..log_h {
            let (left, right) = layers.split_at_mut(k + 1);
            let input = &left[k];
            let output = &mut right[0];
            self.compress.compress_level(device, input, output, num_outputs)?;
            num_outputs /= 2;
        }

        Ok(layers)
    }

    /// Host-fed version of [`Self::commit_with_retained_layers`]: uploads
    /// the matrix, runs the retained-layer commit, and downloads every
    /// layer to a `RetainedLayersHost` ready for opening-path indexing.
    pub fn commit_host_matrix_with_layers(
        &mut self,
        device: &WgpuDevice,
        matrix: &[BabyBear],
        h: u32,
        w: u32,
    ) -> Result<RetainedLayersHost, ZkGpuError> {
        let expected_len = (h as usize)
            .checked_mul(w as usize)
            .ok_or_else(|| {
                ZkGpuError::InvalidNttSize(format!(
                    "Merkle commit host matrix shape overflow: {h} * {w}"
                ))
            })?;
        if matrix.len() != expected_len {
            return Err(ZkGpuError::InvalidNttSize(format!(
                "Merkle commit host matrix length {} != h*w ({}*{}={})",
                matrix.len(),
                h,
                w,
                expected_len,
            )));
        }
        if expected_len == 0 {
            // `commit_with_retained_layers` would reject on h==0 / w==0
            // (and we need a non-empty buffer to hand to upload anyway);
            // route through the guard with a placeholder so the caller
            // sees the same error shape.
            let dummy = device.upload::<BabyBear>(&[BabyBear::new(0)])?;
            let _ = self.commit_with_retained_layers(device, &dummy, h, w)?;
            unreachable!(
                "commit_with_retained_layers rejects h==0/w==0 before returning Ok"
            );
        }
        let matrix_buf = device.upload::<BabyBear>(matrix)?;
        let gpu_layers = self.commit_with_retained_layers(device, &matrix_buf, h, w)?;
        let mut layers: Vec<Vec<BabyBear>> = Vec::with_capacity(gpu_layers.len());
        for buf in &gpu_layers {
            layers.push(buf.read_to_vec()?);
        }
        Ok(RetainedLayersHost { layers })
    }
}

fn read_root(buf: &WgpuBuffer<BabyBear>) -> Result<[BabyBear; DIGEST_LEN], ZkGpuError> {
    let flat = buf.read_to_vec()?;
    if flat.len() < DIGEST_LEN {
        return Err(ZkGpuError::InvalidNttSize(format!(
            "Merkle commit: root buffer too short ({} < {})",
            flat.len(),
            DIGEST_LEN,
        )));
    }
    let root: [BabyBear; DIGEST_LEN] = flat[..DIGEST_LEN]
        .try_into()
        .expect("slice length checked above");
    Ok(root)
}
