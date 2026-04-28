//! Phase 3.d Stage 1c — mixed-height `open_batch` parity.
//!
//! Pins [`zkgpu_wgpu::open_batch_mixed_height`] against Plonky3's
//! `MerkleTreeMmcs::open_batch` for the N=2 binary / `cap_height=0`
//! / power-of-two-heights case. Both in-tree leaf shapes are
//! exercised (W24/RATE=16 Plonky3 canonical, W16/RATE=8 OpenVM
//! shape), using the matching CPU `MerkleTreeMmcs` as oracle.
//!
//! Proof shape for binary mixed-height is identical to single-matrix
//! (`log2(h_max)` sibling digests, bottom-up). Injection awareness
//! lives on the verifier side — this suite also verifies that GPU
//! openings pass through CPU `verify_batch`, exercising the full
//! prover → verifier round trip across mixed-height batches.

use std::sync::Arc;

use p3_baby_bear::{BabyBear as P3BabyBear, Poseidon2BabyBear};
use p3_commit::{BatchOpeningRef, Mmcs};
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Dimensions;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_poseidon2::ExternalLayerConstants;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::distr::StandardUniform;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use zkgpu_babybear::BabyBear as ZkgpuBabyBear;
use zkgpu_plonky3::poseidon2_bridge::{babybear_plonky3_params, p3_to_zkgpu};
use zkgpu_wgpu::{
    commit_mixed_height_with_w16_leaf, commit_mixed_height_with_w24_leaf,
    WgpuPoseidon2InterleavePairsPlan,
    open_batch_mixed_height, MixedHeightMatrixInput, MixedHeightOpening,
    WgpuDevice, WgpuPoseidon2MerkleCompressPlan, WgpuPoseidon2MerkleLeafPlan,
    WgpuPoseidon2MerkleLeafW16R8Plan,
};

const ROUNDS_F: usize = 8;
const DIGEST_LEN: usize = 8;

type Val = P3BabyBear;
type Perm16 = Poseidon2BabyBear<16>;
type Perm24 = Poseidon2BabyBear<24>;

// --- W24 (Plonky3 canonical) oracle config ---
type W24Sponge = PaddingFreeSponge<Perm24, 24, 16, DIGEST_LEN>;
type W24Compression = TruncatedPermutation<Perm16, 2, DIGEST_LEN, 16>;
type W24Mmcs = MerkleTreeMmcs<
    <Val as p3_field::Field>::Packing,
    <Val as p3_field::Field>::Packing,
    W24Sponge,
    W24Compression,
    2,
    DIGEST_LEN,
>;

// --- W16 (OpenVM shape) oracle config ---
type W16Sponge = PaddingFreeSponge<Perm16, 16, 8, DIGEST_LEN>;
type W16Compression = TruncatedPermutation<Perm16, 2, DIGEST_LEN, 16>;
type W16Mmcs = MerkleTreeMmcs<
    <Val as p3_field::Field>::Packing,
    <Val as p3_field::Field>::Packing,
    W16Sponge,
    W16Compression,
    2,
    DIGEST_LEN,
>;

fn try_device() -> Option<Arc<WgpuDevice>> {
    WgpuDevice::new().ok().map(Arc::new)
}

fn random_matrix(h: usize, w: usize, seed: u64) -> (RowMajorMatrix<P3BabyBear>, Vec<ZkgpuBabyBear>) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let p3_values: Vec<P3BabyBear> =
        (&mut rng).sample_iter(StandardUniform).take(h * w).collect();
    let gpu_flat: Vec<ZkgpuBabyBear> =
        p3_values.iter().map(|e| p3_to_zkgpu(*e)).collect();
    (RowMajorMatrix::new(p3_values, w), gpu_flat)
}

/// Assert two opened-values shapes match element-for-element.
fn assert_opened_values_match(
    cpu_opened: &[Vec<P3BabyBear>],
    gpu_opened: &[Vec<ZkgpuBabyBear>],
    ctx: &str,
) {
    assert_eq!(
        cpu_opened.len(),
        gpu_opened.len(),
        "{ctx}: opened_values count"
    );
    for (i, (c, g)) in cpu_opened.iter().zip(gpu_opened.iter()).enumerate() {
        assert_eq!(c.len(), g.len(), "{ctx}: matrix {i} row length");
        for (j, (cv, gv)) in c.iter().zip(g.iter()).enumerate() {
            assert_eq!(
                cv.as_canonical_u32(),
                gv.0,
                "{ctx}: matrix {i} col {j}"
            );
        }
    }
}

/// Assert sibling-chain proofs match element-for-element.
fn assert_proofs_match(
    cpu_proof: &[[P3BabyBear; DIGEST_LEN]],
    gpu_proof: &[[ZkgpuBabyBear; DIGEST_LEN]],
    ctx: &str,
) {
    assert_eq!(cpu_proof.len(), gpu_proof.len(), "{ctx}: proof length");
    for (level, (c, g)) in cpu_proof.iter().zip(gpu_proof.iter()).enumerate() {
        for k in 0..DIGEST_LEN {
            assert_eq!(
                c[k].as_canonical_u32(),
                g[k].0,
                "{ctx}: proof level {level} slot {k}"
            );
        }
    }
}

// ==========================================================================
// W24/RATE=16 leaf path — mixed-height opens
// ==========================================================================

fn build_w24_matched(
    device: Arc<WgpuDevice>,
    seed: u64,
) -> (
    W24Mmcs,
    WgpuPoseidon2MerkleLeafPlan,
    WgpuPoseidon2MerkleCompressPlan,
    WgpuPoseidon2InterleavePairsPlan,
) {
    let mut rng16 = SmallRng::seed_from_u64(seed ^ 0xA11_1600_u64);
    let ext16: ExternalLayerConstants<P3BabyBear, 16> =
        ExternalLayerConstants::new_from_rng(ROUNDS_F, &mut rng16);
    let int16: Vec<P3BabyBear> =
        (&mut rng16).sample_iter(StandardUniform).take(13).collect();
    let perm16: Perm16 = Perm16::new(ext16.clone(), int16.clone());
    let zkgpu_params16 = babybear_plonky3_params::<16>(&ext16, &int16);

    let mut rng24 = SmallRng::seed_from_u64(seed ^ 0xA11_2400_u64);
    let ext24: ExternalLayerConstants<P3BabyBear, 24> =
        ExternalLayerConstants::new_from_rng(ROUNDS_F, &mut rng24);
    let int24: Vec<P3BabyBear> =
        (&mut rng24).sample_iter(StandardUniform).take(21).collect();
    let perm24: Perm24 = Perm24::new(ext24.clone(), int24.clone());
    let zkgpu_params24 = babybear_plonky3_params::<24>(&ext24, &int24);

    let sponge = PaddingFreeSponge::new(perm24);
    let compression = TruncatedPermutation::new(perm16);
    let cpu_mmcs = W24Mmcs::new(sponge, compression, 0);
    let leaf = WgpuPoseidon2MerkleLeafPlan::new(device.as_ref(), zkgpu_params24).unwrap();
    let compress =
        WgpuPoseidon2MerkleCompressPlan::new(device.as_ref(), zkgpu_params16).unwrap();
    let interleave = WgpuPoseidon2InterleavePairsPlan::new(device.as_ref()).unwrap();
    (cpu_mmcs, leaf, compress, interleave)
}

/// Shared driver for W24 open parity. Runs CPU + GPU commit, opens
/// at each supplied index, asserts both opened_values and proof
/// match. Then round-trips the GPU opening through CPU verify_batch
/// so correctness of the sibling chain is backed by the verifier,
/// not just structural comparison.
fn run_w24_open_parity(
    device: &WgpuDevice,
    cpu: &W24Mmcs,
    leaf: &mut WgpuPoseidon2MerkleLeafPlan,
    compress: &mut WgpuPoseidon2MerkleCompressPlan,
    interleave: &mut WgpuPoseidon2InterleavePairsPlan,
    shapes: &[(usize, usize)],
    indices: &[u32],
    seed_base: u64,
) {
    let ctx_base = format!("W24 shapes={shapes:?}");

    // CPU side
    let (cpu_matrices, gpu_inputs_storage) = gen_matrices(shapes, seed_base);
    let (cap, cpu_pd) = cpu.commit(cpu_matrices);

    // GPU side
    let gpu_inputs: Vec<MixedHeightMatrixInput<'_>> = gpu_inputs_storage
        .iter()
        .map(|(flat, h, w)| MixedHeightMatrixInput {
            flat: flat.as_slice(),
            height: *h,
            width: *w,
        })
        .collect();
    let retained = commit_mixed_height_with_w24_leaf(
        device, leaf, compress, interleave, &gpu_inputs,
    )
    .unwrap();

    let dims: Vec<Dimensions> = shapes
        .iter()
        .map(|&(h, w)| Dimensions { width: w, height: h })
        .collect();

    for &idx in indices {
        let ctx = format!("{ctx_base} idx={idx}");
        let cpu_open = cpu.open_batch(idx as usize, &cpu_pd);
        let gpu_open: MixedHeightOpening =
            open_batch_mixed_height(&gpu_inputs, &retained, idx).unwrap();
        assert_opened_values_match(&cpu_open.opened_values, &gpu_open.opened_values, &ctx);
        assert_proofs_match(&cpu_open.opening_proof, &gpu_open.opening_proof, &ctx);

        // Round-trip: CPU verify_batch must accept the GPU proof.
        // Convert GPU opening (zkgpu BabyBear) → p3 BabyBear for the
        // verifier call.
        let gpu_opened_p3: Vec<Vec<P3BabyBear>> = gpu_open
            .opened_values
            .iter()
            .map(|row| row.iter().map(|v| P3BabyBear::new(v.0)).collect())
            .collect();
        let gpu_proof_p3: Vec<[P3BabyBear; DIGEST_LEN]> = gpu_open
            .opening_proof
            .iter()
            .map(|d| d.map(|v| P3BabyBear::new(v.0)))
            .collect();
        let opening_ref: BatchOpeningRef<'_, P3BabyBear, W24Mmcs> =
            BatchOpeningRef::new(&gpu_opened_p3, &gpu_proof_p3);
        cpu.verify_batch(&cap, &dims, idx as usize, opening_ref)
            .unwrap_or_else(|e| panic!("{ctx}: CPU verify_batch rejected GPU proof: {e:?}"));
    }
}

fn gen_matrices(
    shapes: &[(usize, usize)],
    seed_base: u64,
) -> (Vec<RowMajorMatrix<P3BabyBear>>, Vec<(Vec<ZkgpuBabyBear>, u32, u32)>) {
    let mut cpu_matrices: Vec<RowMajorMatrix<P3BabyBear>> = Vec::with_capacity(shapes.len());
    let mut gpu_storage: Vec<(Vec<ZkgpuBabyBear>, u32, u32)> = Vec::with_capacity(shapes.len());
    for (i, &(h, w)) in shapes.iter().enumerate() {
        let (cpu_m, gpu_flat) = random_matrix(h, w, seed_base ^ (i as u64) * 0xBADC0DE);
        cpu_matrices.push(cpu_m);
        gpu_storage.push((gpu_flat, h as u32, w as u32));
    }
    (cpu_matrices, gpu_storage)
}

#[test]
fn w24_open_single_matrix_matches_plonky3() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, mut leaf, mut compress, mut interleave) = build_w24_matched(device.clone(), 0x_0BE1_0001_u64);
    run_w24_open_parity(
        &device,
        &cpu,
        &mut leaf,
        &mut compress,
        &mut interleave,
        &[(64, 8)],
        &[0, 1, 7, 31, 32, 63],
        0x_F20_0000_u64,
    );
}

#[test]
fn w24_open_same_height_multi_matrix_matches_plonky3() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, mut leaf, mut compress, mut interleave) = build_w24_matched(device.clone(), 0x_0BE1_0002_u64);
    run_w24_open_parity(
        &device,
        &cpu,
        &mut leaf,
        &mut compress,
        &mut interleave,
        &[(32, 4), (32, 3)],
        &[0, 5, 15, 31],
        0x_F20_0010_u64,
    );
}

#[test]
fn w24_open_mixed_height_two_levels_matches_plonky3() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, mut leaf, mut compress, mut interleave) = build_w24_matched(device.clone(), 0x_0BE1_0003_u64);
    // h_max=16, one injection at height 4
    run_w24_open_parity(
        &device,
        &cpu,
        &mut leaf,
        &mut compress,
        &mut interleave,
        &[(16, 4), (4, 2)],
        &[0, 3, 7, 15],
        0x_F20_0020_u64,
    );
}

#[test]
fn w24_open_mixed_height_three_levels_matches_plonky3() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, mut leaf, mut compress, mut interleave) = build_w24_matched(device.clone(), 0x_0BE1_0004_u64);
    // h_max=32, injection at heights 16 and 4
    run_w24_open_parity(
        &device,
        &cpu,
        &mut leaf,
        &mut compress,
        &mut interleave,
        &[(32, 6), (16, 2), (4, 5)],
        &[0, 1, 5, 12, 20, 31],
        0x_F20_0030_u64,
    );
}

#[test]
fn w24_open_mixed_height_every_level_matches_plonky3() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, mut leaf, mut compress, mut interleave) = build_w24_matched(device.clone(), 0x_0BE1_0005_u64);
    // Injection at every level
    let shapes: Vec<(usize, usize)> =
        [16usize, 8, 4, 2, 1].iter().map(|&h| (h, 3usize)).collect();
    run_w24_open_parity(
        &device,
        &cpu,
        &mut leaf,
        &mut compress,
        &mut interleave,
        &shapes,
        &[0, 5, 10, 15],
        0x_F20_0040_u64,
    );
}

#[test]
fn w24_open_single_row_matrix_matches_plonky3() {
    // Edge case: h=1 → no compression, root IS the leaf hash.
    // Opening at index=0 has an empty proof.
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, mut leaf, mut compress, mut interleave) = build_w24_matched(device.clone(), 0x_0BE1_0006_u64);
    run_w24_open_parity(
        &device,
        &cpu,
        &mut leaf,
        &mut compress,
        &mut interleave,
        &[(1, 8)],
        &[0],
        0x_F20_0050_u64,
    );
}

// ==========================================================================
// W16/RATE=8 leaf path — mixed-height opens (OpenVM shape)
// ==========================================================================

fn build_w16_matched(
    device: Arc<WgpuDevice>,
    seed: u64,
) -> (
    W16Mmcs,
    WgpuPoseidon2MerkleLeafW16R8Plan,
    WgpuPoseidon2MerkleCompressPlan,
    WgpuPoseidon2InterleavePairsPlan,
) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let ext: ExternalLayerConstants<P3BabyBear, 16> =
        ExternalLayerConstants::new_from_rng(ROUNDS_F, &mut rng);
    let int: Vec<P3BabyBear> =
        (&mut rng).sample_iter(StandardUniform).take(13).collect();
    let perm16: Perm16 = Perm16::new(ext.clone(), int.clone());
    let zkgpu_params = babybear_plonky3_params::<16>(&ext, &int);

    let sponge = PaddingFreeSponge::new(perm16.clone());
    let compression = TruncatedPermutation::new(perm16);
    let cpu_mmcs = W16Mmcs::new(sponge, compression, 0);

    let leaf = WgpuPoseidon2MerkleLeafW16R8Plan::new(
        device.as_ref(),
        zkgpu_params.clone(),
    )
    .unwrap();
    let compress =
        WgpuPoseidon2MerkleCompressPlan::new(device.as_ref(), zkgpu_params).unwrap();
    let interleave = WgpuPoseidon2InterleavePairsPlan::new(device.as_ref()).unwrap();
    (cpu_mmcs, leaf, compress, interleave)
}

fn run_w16_open_parity(
    device: &WgpuDevice,
    cpu: &W16Mmcs,
    leaf: &mut WgpuPoseidon2MerkleLeafW16R8Plan,
    compress: &mut WgpuPoseidon2MerkleCompressPlan,
    interleave: &mut WgpuPoseidon2InterleavePairsPlan,
    shapes: &[(usize, usize)],
    indices: &[u32],
    seed_base: u64,
) {
    let ctx_base = format!("W16 shapes={shapes:?}");
    let (cpu_matrices, gpu_inputs_storage) = gen_matrices(shapes, seed_base);
    let (cap, cpu_pd) = cpu.commit(cpu_matrices);

    let gpu_inputs: Vec<MixedHeightMatrixInput<'_>> = gpu_inputs_storage
        .iter()
        .map(|(flat, h, w)| MixedHeightMatrixInput {
            flat: flat.as_slice(),
            height: *h,
            width: *w,
        })
        .collect();
    let retained = commit_mixed_height_with_w16_leaf(
        device, leaf, compress, interleave, &gpu_inputs,
    )
    .unwrap();

    let dims: Vec<Dimensions> = shapes
        .iter()
        .map(|&(h, w)| Dimensions { width: w, height: h })
        .collect();

    for &idx in indices {
        let ctx = format!("{ctx_base} idx={idx}");
        let cpu_open = cpu.open_batch(idx as usize, &cpu_pd);
        let gpu_open = open_batch_mixed_height(&gpu_inputs, &retained, idx).unwrap();
        assert_opened_values_match(&cpu_open.opened_values, &gpu_open.opened_values, &ctx);
        assert_proofs_match(&cpu_open.opening_proof, &gpu_open.opening_proof, &ctx);

        let gpu_opened_p3: Vec<Vec<P3BabyBear>> = gpu_open
            .opened_values
            .iter()
            .map(|row| row.iter().map(|v| P3BabyBear::new(v.0)).collect())
            .collect();
        let gpu_proof_p3: Vec<[P3BabyBear; DIGEST_LEN]> = gpu_open
            .opening_proof
            .iter()
            .map(|d| d.map(|v| P3BabyBear::new(v.0)))
            .collect();
        let opening_ref: BatchOpeningRef<'_, P3BabyBear, W16Mmcs> =
            BatchOpeningRef::new(&gpu_opened_p3, &gpu_proof_p3);
        cpu.verify_batch(&cap, &dims, idx as usize, opening_ref)
            .unwrap_or_else(|e| panic!("{ctx}: CPU verify_batch rejected GPU proof: {e:?}"));
    }
}

#[test]
fn w16_open_single_matrix_matches_plonky3() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, mut leaf, mut compress, mut interleave) = build_w16_matched(device.clone(), 0x_0BE1_1001_u64);
    run_w16_open_parity(
        &device,
        &cpu,
        &mut leaf,
        &mut compress,
        &mut interleave,
        &[(64, 8)],
        &[0, 1, 7, 31, 32, 63],
        0x_F21_0000_u64,
    );
}

#[test]
fn w16_open_same_height_multi_matrix_matches_plonky3() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, mut leaf, mut compress, mut interleave) = build_w16_matched(device.clone(), 0x_0BE1_1002_u64);
    run_w16_open_parity(
        &device,
        &cpu,
        &mut leaf,
        &mut compress,
        &mut interleave,
        &[(32, 4), (32, 3)],
        &[0, 5, 15, 31],
        0x_F21_0010_u64,
    );
}

#[test]
fn w16_open_mixed_height_three_levels_matches_plonky3() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, mut leaf, mut compress, mut interleave) = build_w16_matched(device.clone(), 0x_0BE1_1003_u64);
    run_w16_open_parity(
        &device,
        &cpu,
        &mut leaf,
        &mut compress,
        &mut interleave,
        &[(32, 6), (16, 2), (4, 5)],
        &[0, 1, 5, 12, 20, 31],
        0x_F21_0030_u64,
    );
}

#[test]
fn w16_open_mixed_height_every_level_matches_plonky3() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, mut leaf, mut compress, mut interleave) = build_w16_matched(device.clone(), 0x_0BE1_1004_u64);
    let shapes: Vec<(usize, usize)> =
        [16usize, 8, 4, 2, 1].iter().map(|&h| (h, 3usize)).collect();
    run_w16_open_parity(
        &device,
        &cpu,
        &mut leaf,
        &mut compress,
        &mut interleave,
        &shapes,
        &[0, 5, 10, 15],
        0x_F21_0040_u64,
    );
}

// ==========================================================================
// Guards (backend-side, input validation)
// ==========================================================================

#[test]
fn open_rejects_index_out_of_bounds() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (_cpu, mut leaf, mut compress, mut interleave) = build_w24_matched(device.clone(), 0x_0BE1_9001_u64);
    let (_, gpu_storage) = gen_matrices(&[(8, 4)], 0x_F29_0000_u64);
    let inputs: Vec<MixedHeightMatrixInput<'_>> = gpu_storage
        .iter()
        .map(|(flat, h, w)| MixedHeightMatrixInput { flat, height: *h, width: *w })
        .collect();
    let retained =
        commit_mixed_height_with_w24_leaf(&device, &mut leaf, &mut compress, &mut interleave, &inputs)
            .unwrap();
    let err = open_batch_mixed_height(&inputs, &retained, 8); // h_max=8, valid: 0..8
    assert!(err.is_err(), "open must reject index >= h_max");
}

#[test]
fn open_rejects_empty_matrices() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (_cpu, mut leaf, mut compress, mut interleave) = build_w24_matched(device.clone(), 0x_0BE1_9002_u64);
    let (_, gpu_storage) = gen_matrices(&[(8, 4)], 0x_F29_0010_u64);
    let inputs: Vec<MixedHeightMatrixInput<'_>> = gpu_storage
        .iter()
        .map(|(flat, h, w)| MixedHeightMatrixInput { flat, height: *h, width: *w })
        .collect();
    let retained =
        commit_mixed_height_with_w24_leaf(&device, &mut leaf, &mut compress, &mut interleave, &inputs)
            .unwrap();
    let err = open_batch_mixed_height(&[], &retained, 0);
    assert!(err.is_err(), "open must reject empty matrices slice");
}
