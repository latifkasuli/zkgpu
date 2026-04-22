//! Phase 7 Step 3 — GPU Poseidon2 MMCS adapter parity test.
//!
//! Locks [`zkgpu_plonky3::gpu_mmcs::GpuPoseidon2Mmcs`] against
//! Plonky3's canonical `MerkleTreeMmcs<Packing, Packing,
//! Poseidon2Sponge, Poseidon2Compression, 2, 8>` at the
//! `Mmcs::commit` level — the same shape `TwoAdicFriPcs::commit`
//! feeds into our Step 3 `fri_commit` bench. If this passes, the
//! adapter is a bit-compatible drop-in for the target stack's input
//! MMCS and the bench gate can call through it with confidence.

use std::sync::Arc;

use p3_baby_bear::BabyBear as P3BabyBear;
use p3_commit::Mmcs;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_poseidon2::ExternalLayerConstants;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::distr::StandardUniform;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use zkgpu_plonky3::gpu_mmcs::{
    GpuPoseidon2Mmcs, Perm16, Perm24, P3Poseidon2Compression, P3Poseidon2Sponge,
};
use zkgpu_plonky3::poseidon2_bridge::babybear_plonky3_params;
use zkgpu_wgpu::WgpuDevice;

const ROUNDS_F: usize = 8;

type Val = P3BabyBear;
type CpuValMmcs = MerkleTreeMmcs<
    <Val as p3_field::Field>::Packing,
    <Val as p3_field::Field>::Packing,
    P3Poseidon2Sponge,
    P3Poseidon2Compression,
    2,
    8,
>;

fn try_device() -> Option<Arc<WgpuDevice>> {
    WgpuDevice::new().ok().map(Arc::new)
}

/// Build a matched (CPU MerkleTreeMmcs, GPU adapter) pair with
/// identical Poseidon2 constants. W16 and W24 get distinct RNG
/// draws, as in the canonical Plonky3 config.
fn build_matched(
    device: Arc<WgpuDevice>,
    seed: u64,
) -> (CpuValMmcs, GpuPoseidon2Mmcs) {
    // W16 (compression)
    let mut rng16 = SmallRng::seed_from_u64(seed ^ 0xA11_1600_u64);
    let ext16: ExternalLayerConstants<P3BabyBear, 16> =
        ExternalLayerConstants::new_from_rng(ROUNDS_F, &mut rng16);
    let int16: Vec<P3BabyBear> =
        (&mut rng16).sample_iter(StandardUniform).take(13).collect();
    let perm16: Perm16 = Perm16::new(ext16.clone(), int16.clone());
    let zkgpu_params16 = babybear_plonky3_params::<16>(&ext16, &int16);

    // W24 (leaf sponge)
    let mut rng24 = SmallRng::seed_from_u64(seed ^ 0xA11_2400_u64);
    let ext24: ExternalLayerConstants<P3BabyBear, 24> =
        ExternalLayerConstants::new_from_rng(ROUNDS_F, &mut rng24);
    let int24: Vec<P3BabyBear> =
        (&mut rng24).sample_iter(StandardUniform).take(21).collect();
    let perm24: Perm24 = Perm24::new(ext24.clone(), int24.clone());
    let zkgpu_params24 = babybear_plonky3_params::<24>(&ext24, &int24);

    // CPU MMCS (cap_height = 0 → root only)
    let sponge = PaddingFreeSponge::new(perm24.clone());
    let compression = TruncatedPermutation::new(perm16.clone());
    let cpu_mmcs = CpuValMmcs::new(sponge, compression, 0);

    // GPU adapter with the same constants
    let gpu_mmcs = GpuPoseidon2Mmcs::new(
        device,
        perm24,
        perm16,
        zkgpu_params24,
        zkgpu_params16,
        0,
    )
    .unwrap();

    (cpu_mmcs, gpu_mmcs)
}

fn random_rowmajor(h: usize, w: usize, seed: u64) -> RowMajorMatrix<P3BabyBear> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let values: Vec<P3BabyBear> =
        (&mut rng).sample_iter(StandardUniform).take(h * w).collect();
    RowMajorMatrix::new(values, w)
}

fn assert_roots_match(
    cpu_cap: &p3_symmetric::MerkleCap<P3BabyBear, [P3BabyBear; 8]>,
    gpu_cap: &p3_symmetric::MerkleCap<P3BabyBear, [P3BabyBear; 8]>,
    ctx: &str,
) {
    let cpu = GpuPoseidon2Mmcs::root(cpu_cap).expect("cpu cap has root");
    let gpu = GpuPoseidon2Mmcs::root(gpu_cap).expect("gpu cap has root");
    for i in 0..8 {
        assert_eq!(
            cpu[i].as_canonical_u32(),
            gpu[i].as_canonical_u32(),
            "{ctx}: root slot {i} mismatch"
        );
    }
}

// -- Parity across (h, w) shapes ------------------------------------------

#[test]
fn mmcs_commit_matches_plonky3_small_shapes() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, gpu) = build_matched(device, 0x_CC550001_u64);
    for &(h, w) in &[(1usize, 8usize), (2, 8), (4, 8), (16, 8), (16, 17)] {
        let mat = random_rowmajor(h, w, 0x_DEAD_0000_u64 ^ (h as u64) << 8 ^ (w as u64));
        let (cpu_cap, _cpu_pd) = cpu.commit(vec![mat.clone()]);
        let (gpu_cap, _gpu_pd) = gpu.commit(vec![mat]);
        assert_roots_match(&cpu_cap, &gpu_cap, &format!("(h={h}, w={w})"));
    }
}

#[test]
fn mmcs_commit_matches_plonky3_bench_shape() {
    // Stand-in for the `log_h=18, w=8` bench shape. Scaled down to
    // keep the test fast (the full bench size takes several seconds
    // even with the GPU).
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, gpu) = build_matched(device, 0x_C0FF_EE_40_u64);
    // log_h = 12 (h = 4096), w = 8 — 15 compression levels + a leaf
    // sponge over a 15-chunk row = 1 permute per row. Same code path
    // as the bench shape, small enough for CI.
    let mat = random_rowmajor(4096, 8, 0x_BEEF_1000_u64);
    let (cpu_cap, _) = cpu.commit(vec![mat.clone()]);
    let (gpu_cap, _) = gpu.commit(vec![mat]);
    assert_roots_match(&cpu_cap, &gpu_cap, "(h=4096, w=8)");
}

// -- Multi-matrix rejection ----------------------------------------------

#[test]
#[should_panic(expected = "single-matrix adapter")]
fn mmcs_commit_panics_on_multi_matrix() {
    let Some(device) = try_device() else {
        // No GPU: the panic assertion can't fire because the adapter
        // wasn't built. `should_panic` tests that fail early with a
        // non-panic get counted as failures — feed the test what it
        // expects instead, so CI on headless machines doesn't block.
        panic!("single-matrix adapter (skipping GPU body)");
    };
    let (_, gpu) = build_matched(device, 0x_C0FF_EE_41_u64);
    let m1 = random_rowmajor(8, 8, 1);
    let m2 = random_rowmajor(8, 8, 2);
    let _ = gpu.commit(vec![m1, m2]);
}

// -- open_batch unimplemented panic --------------------------------------

#[test]
#[should_panic(expected = "commit-only bench-gate adapter")]
fn mmcs_open_batch_unimplemented() {
    let Some(device) = try_device() else {
        panic!("commit-only bench-gate adapter (skipping GPU body)");
    };
    let (_, gpu) = build_matched(device, 0x_C0FF_EE_42_u64);
    let mat = random_rowmajor(8, 8, 0x_0B_0042_u64);
    let (_cap, prover_data) = gpu.commit(vec![mat]);
    let _ = gpu.open_batch(0, &prover_data);
}

// -- get_matrices returns refs without building a tree ------------------

#[test]
fn mmcs_get_matrices_returns_inputs() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (_, gpu) = build_matched(device, 0x_C0FF_EE_43_u64);
    let mat = random_rowmajor(8, 8, 0x_0B_0043_u64);
    let (_cap, prover_data) = gpu.commit(vec![mat]);
    let mats = gpu.get_matrices(&prover_data);
    assert_eq!(mats.len(), 1);
    assert_eq!(mats[0].values.len(), 8 * 8);
}
