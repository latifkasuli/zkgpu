//! Phase 3.d Stage 1b addendum — mixed-height DAG parity with the
//! **W16/RATE=8 leaf sponge** (OpenVM-shape leaf path through the
//! shared backend).
//!
//! Sibling to `poseidon2_merkle_commit_dag_gpu.rs` which pins the
//! same DAG engine with the W24/RATE=16 leaf path. Together the two
//! suites prove the mixed-height Merkle DAG orchestration is
//! byte-compatible with Plonky3's `MerkleTreeMmcs::commit` under
//! **both** in-tree leaf shapes — closing the residual-risk the
//! reviewer flagged on Stage 1b (DAG orchestration was validated
//! through W24 only at that point).
//!
//! Oracle config: `MerkleTreeMmcs<_, _, PaddingFreeSponge<Perm16, 16, 8, 8>,
//! TruncatedPermutation<Perm16, 2, 8, 16>, 2, 8>` — the OpenVM
//! BabyBear Poseidon2 MMCS shape from
//! `openvm-org/stark-backend/crates/stark-sdk/src/config/baby_bear_poseidon2.rs`.

use std::sync::Arc;

use p3_baby_bear::{BabyBear as P3BabyBear, Poseidon2BabyBear};
use p3_commit::Mmcs;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_poseidon2::ExternalLayerConstants;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::distr::StandardUniform;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use zkgpu_babybear::BabyBear as ZkgpuBabyBear;
use zkgpu_plonky3::poseidon2_bridge::{babybear_plonky3_params, p3_to_zkgpu};
use zkgpu_wgpu::{
    commit_mixed_height_with_w16_leaf, root_from_retained, MixedHeightMatrixInput,
    WgpuDevice, WgpuPoseidon2InterleavePairsPlan, WgpuPoseidon2MerkleCompressPlan,
    WgpuPoseidon2MerkleLeafW16R8Plan,
};

const ROUNDS_F: usize = 8;
const DIGEST_LEN: usize = 8;

type Val = P3BabyBear;
type Perm16 = Poseidon2BabyBear<16>;
// W16/RATE=8 leaf sponge — OpenVM's config.
type Poseidon2W16Sponge = PaddingFreeSponge<Perm16, 16, 8, DIGEST_LEN>;
// Same W16 truncated-permutation compression as the W24-leaf suite.
type Poseidon2W16Compression = TruncatedPermutation<Perm16, 2, DIGEST_LEN, 16>;
type CpuValMmcsW16 = MerkleTreeMmcs<
    <Val as p3_field::Field>::Packing,
    <Val as p3_field::Field>::Packing,
    Poseidon2W16Sponge,
    Poseidon2W16Compression,
    2,
    DIGEST_LEN,
>;

fn try_device() -> Option<Arc<WgpuDevice>> {
    WgpuDevice::new().ok().map(Arc::new)
}

/// Build a matched (CPU MerkleTreeMmcs, GPU W16 leaf plan, GPU W16
/// compress plan) triple driven by identical Poseidon2 constants.
/// Both leaf and compression use the same `Perm16` instance here —
/// matches OpenVM's `baby_bear_poseidon2` config where
/// `type Hash<P> = PaddingFreeSponge<P, WIDTH, RATE, DIGEST_WIDTH>`
/// and `type Compress<P> = TruncatedPermutation<P, 2, DIGEST_WIDTH, WIDTH>`
/// share a single `P = Perm16`.
fn build_matched(
    device: Arc<WgpuDevice>,
    seed: u64,
) -> (
    CpuValMmcsW16,
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
    let cpu_mmcs = CpuValMmcsW16::new(sponge, compression, 0);

    let leaf =
        WgpuPoseidon2MerkleLeafW16R8Plan::new(device.as_ref(), zkgpu_params.clone())
            .unwrap();
    let compress =
        WgpuPoseidon2MerkleCompressPlan::new(device.as_ref(), zkgpu_params).unwrap();
    let interleave = WgpuPoseidon2InterleavePairsPlan::new(device.as_ref()).unwrap();

    (cpu_mmcs, leaf, compress, interleave)
}

fn random_matrix(h: usize, w: usize, seed: u64) -> (RowMajorMatrix<P3BabyBear>, Vec<ZkgpuBabyBear>) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let p3_values: Vec<P3BabyBear> =
        (&mut rng).sample_iter(StandardUniform).take(h * w).collect();
    let gpu_flat: Vec<ZkgpuBabyBear> =
        p3_values.iter().map(|e| p3_to_zkgpu(*e)).collect();
    (RowMajorMatrix::new(p3_values, w), gpu_flat)
}

fn assert_root_matches(
    cpu_cap: &p3_symmetric::MerkleCap<P3BabyBear, [P3BabyBear; DIGEST_LEN]>,
    gpu_root: [ZkgpuBabyBear; DIGEST_LEN],
    ctx: &str,
) {
    let cpu_root: [P3BabyBear; DIGEST_LEN] = cpu_cap.as_ref()[0];
    for i in 0..DIGEST_LEN {
        assert_eq!(
            cpu_root[i].as_canonical_u32(),
            gpu_root[i].0,
            "{ctx}: root slot {i}"
        );
    }
}

fn run_mixed_shapes(
    device: &WgpuDevice,
    cpu: &CpuValMmcsW16,
    leaf: &mut WgpuPoseidon2MerkleLeafW16R8Plan,
    compress: &mut WgpuPoseidon2MerkleCompressPlan,
    interleave: &mut WgpuPoseidon2InterleavePairsPlan,
    shapes: &[(usize, usize)],
    seed_base: u64,
) {
    let ctx = format!("W16 shapes={shapes:?}");

    let mut cpu_matrices: Vec<RowMajorMatrix<P3BabyBear>> = Vec::with_capacity(shapes.len());
    let mut gpu_inputs_storage: Vec<(Vec<ZkgpuBabyBear>, u32, u32)> =
        Vec::with_capacity(shapes.len());
    for (i, &(h, w)) in shapes.iter().enumerate() {
        let (cpu_m, gpu_flat) = random_matrix(h, w, seed_base ^ (i as u64) * 0xBADC0DE);
        cpu_matrices.push(cpu_m);
        gpu_inputs_storage.push((gpu_flat, h as u32, w as u32));
    }
    let (cpu_cap, _cpu_pd) = cpu.commit(cpu_matrices);

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
    let gpu_root = root_from_retained(&retained).unwrap();

    assert_root_matches(&cpu_cap, gpu_root, &ctx);
}

// ==========================================================================
// Parity — same-height cases through the W16 leaf path
// ==========================================================================

#[test]
fn w16_dag_single_matrix_matches_plonky3() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, mut leaf, mut compress, mut interleave) = build_matched(device.clone(), 0x_D51_0001_u64);
    for &(h, w) in &[(1usize, 8usize), (2, 8), (4, 8), (16, 8), (64, 3), (1024, 8)] {
        run_mixed_shapes(&device, &cpu, &mut leaf, &mut compress, &mut interleave, &[(h, w)], 0x_F10_0000_u64);
    }
}

#[test]
fn w16_dag_same_height_multi_matrix_matches_plonky3() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, mut leaf, mut compress, mut interleave) = build_matched(device.clone(), 0x_D51_0002_u64);
    for shapes in &[
        vec![(16usize, 4usize), (16, 4)],
        vec![(8, 3), (8, 5), (8, 1)],
        vec![(64, 8); 4],
    ] {
        run_mixed_shapes(&device, &cpu, &mut leaf, &mut compress, &mut interleave, shapes, 0x_F10_0010_u64);
    }
}

// ==========================================================================
// Parity — true mixed-height batches through the W16 leaf path
// ==========================================================================

#[test]
fn w16_dag_mixed_height_two_levels_matches_plonky3() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, mut leaf, mut compress, mut interleave) = build_matched(device.clone(), 0x_D51_0003_u64);
    for shapes in &[
        vec![(8usize, 4usize), (4, 2)],
        vec![(16, 8), (8, 4)],
        vec![(32, 3), (16, 7)],
    ] {
        run_mixed_shapes(&device, &cpu, &mut leaf, &mut compress, &mut interleave, shapes, 0x_F10_0020_u64);
    }
}

#[test]
fn w16_dag_mixed_height_three_levels_matches_plonky3() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, mut leaf, mut compress, mut interleave) = build_matched(device.clone(), 0x_D51_0004_u64);
    for shapes in &[
        vec![(16usize, 4usize), (8, 3), (2, 1)],
        vec![(32, 6), (16, 2), (4, 5)],
        vec![(64, 8), (32, 4), (4, 2)],
    ] {
        run_mixed_shapes(&device, &cpu, &mut leaf, &mut compress, &mut interleave, shapes, 0x_F10_0030_u64);
    }
}

#[test]
fn w16_dag_mixed_height_every_level_matches_plonky3() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, mut leaf, mut compress, mut interleave) = build_matched(device.clone(), 0x_D51_0005_u64);
    let shapes: Vec<(usize, usize)> =
        [16usize, 8, 4, 2, 1].iter().map(|&h| (h, 3usize)).collect();
    run_mixed_shapes(&device, &cpu, &mut leaf, &mut compress, &mut interleave, &shapes, 0x_F10_0040_u64);
}

#[test]
fn w16_dag_bench_shape_proxy_matches_plonky3() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, mut leaf, mut compress, mut interleave) = build_matched(device.clone(), 0x_D51_0006_u64);
    let shapes = [(1024usize, 8usize), (512, 4)];
    run_mixed_shapes(&device, &cpu, &mut leaf, &mut compress, &mut interleave, &shapes, 0x_F10_0060_u64);
}
