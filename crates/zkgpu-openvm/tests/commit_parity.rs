//! Phase 3.d Stage 2a — OpenVM GPU MMCS commit parity.
//!
//! Pins [`zkgpu_openvm::OpenVmGpuMmcs::commit`] against Plonky3
//! 0.4.1's CPU `MerkleTreeMmcs<Packing, Packing, PaddingFreeSponge<
//! Perm16, 16, 8, 8>, TruncatedPermutation<Perm16, 2, 8, 16>, 8>`
//! — the exact config OpenVM uses in
//! `stark-backend/crates/stark-sdk/src/config/baby_bear_poseidon2.rs`.
//!
//! Stage 2a covers commit + get_matrices only. `open_batch` and
//! `verify_batch` are `todo!()` in the adapter; Stage 2b adds those
//! plus their parity suite. The shared backend's
//! `open_batch_mixed_height` is already parity-pinned in the
//! zkgpu-plonky3 test tree, so 2b is plumbing rather than new
//! correctness work.
//!
//! Scope coverage:
//! * single-matrix commit (degenerate case of mixed-height)
//! * same-height multi-matrix (quotient-chunk-shape commits)
//! * mixed-height (2/3/5-level injection topologies)
//! * multiple matrices at the same non-max height
//! * rejection: non-zero cap_height at construction

use std::sync::Arc;

use p3_baby_bear::BabyBear;
use p3_commit::Mmcs;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_poseidon2::ExternalLayerConstants;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::distr::StandardUniform;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use zkgpu_openvm::{
    babybear_openvm_params, Commitment, OpenVmGpuMmcs, Perm, Val, DIGEST_WIDTH,
};
use zkgpu_wgpu::WgpuDevice;

const ROUNDS_F: usize = 8;

// --- Matched CPU MMCS (exact OpenVM config) ---
type CpuValMmcs = MerkleTreeMmcs<
    <Val as p3_field::Field>::Packing,
    <Val as p3_field::Field>::Packing,
    PaddingFreeSponge<Perm, 16, 8, 8>,
    TruncatedPermutation<Perm, 2, 8, 16>,
    8,
>;

fn try_device() -> Option<Arc<WgpuDevice>> {
    WgpuDevice::new().ok().map(Arc::new)
}

/// Build a matched (CPU MMCS, GPU adapter) pair driven by the same
/// Plonky3 0.4.1 Poseidon2 constants.
fn build_matched(device: Arc<WgpuDevice>, seed: u64) -> (CpuValMmcs, OpenVmGpuMmcs) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let ext: ExternalLayerConstants<BabyBear, 16> =
        ExternalLayerConstants::new_from_rng(ROUNDS_F, &mut rng);
    let int: Vec<BabyBear> =
        (&mut rng).sample_iter(StandardUniform).take(13).collect();
    let perm16: Perm = Perm::new(ext.clone(), int.clone());

    // CPU reference — construction signature matches Plonky3 0.4.1
    // (two-arg `MerkleTreeMmcs::new`, no cap_height parameter).
    let cpu_sponge = PaddingFreeSponge::new(perm16.clone());
    let cpu_compress = TruncatedPermutation::new(perm16.clone());
    let cpu_mmcs = CpuValMmcs::new(cpu_sponge, cpu_compress);

    // GPU adapter — uses the zkgpu-openvm bridge to translate the
    // same constants into zkgpu-side Poseidon2Params<16>.
    let zkgpu_params = babybear_openvm_params(&ext, &int);
    let gpu_mmcs = OpenVmGpuMmcs::new(
        device,
        perm16,
        zkgpu_params.clone(),
        zkgpu_params,
        0,
    )
    .unwrap();

    (cpu_mmcs, gpu_mmcs)
}

fn random_matrix(h: usize, w: usize, seed: u64) -> RowMajorMatrix<BabyBear> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let values: Vec<BabyBear> = (&mut rng).sample_iter(StandardUniform).take(h * w).collect();
    RowMajorMatrix::new(values, w)
}

fn assert_commitments_match(
    cpu: &Commitment,
    gpu: &Commitment,
    ctx: &str,
) {
    let cpu_root: [Val; DIGEST_WIDTH] = OpenVmGpuMmcs::root(cpu);
    let gpu_root: [Val; DIGEST_WIDTH] = OpenVmGpuMmcs::root(gpu);
    for i in 0..DIGEST_WIDTH {
        assert_eq!(
            cpu_root[i].as_canonical_u32(),
            gpu_root[i].as_canonical_u32(),
            "{ctx}: commitment slot {i}"
        );
    }
}

fn run_shapes_parity(
    cpu: &CpuValMmcs,
    gpu: &OpenVmGpuMmcs,
    shapes: &[(usize, usize)],
    seed_base: u64,
) {
    let ctx = format!("shapes={shapes:?}");
    let matrices: Vec<RowMajorMatrix<BabyBear>> = shapes
        .iter()
        .enumerate()
        .map(|(i, &(h, w))| random_matrix(h, w, seed_base ^ (i as u64) * 0xBADC0DE))
        .collect();
    let (cpu_cap, _cpu_pd) = cpu.commit(matrices.clone());
    let (gpu_cap, gpu_pd) = gpu.commit(matrices);
    assert_commitments_match(&cpu_cap, &gpu_cap, &ctx);

    // get_matrices returns refs to the stored inputs in order.
    let gpu_mats = gpu.get_matrices(&gpu_pd);
    assert_eq!(gpu_mats.len(), shapes.len(), "{ctx}: matrix count");
    for (i, (&(h, w), m)) in shapes.iter().zip(gpu_mats.iter()).enumerate() {
        assert_eq!(m.values.len(), h * w, "{ctx}: matrix {i} values len");
    }
}

// ==========================================================================
// Single-matrix
// ==========================================================================

#[test]
fn commit_single_matrix_matches_openvm_reference() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, gpu) = build_matched(device, 0x_0BE_0001_u64);
    for &(h, w) in &[(1usize, 8usize), (2, 8), (4, 8), (16, 8), (64, 3), (1024, 8)] {
        run_shapes_parity(&cpu, &gpu, &[(h, w)], 0x_0BE_1000_u64);
    }
}

// ==========================================================================
// Same-height multi-matrix (quotient-chunk-shape)
// ==========================================================================

#[test]
fn commit_same_height_multi_matrix_matches_openvm_reference() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, gpu) = build_matched(device, 0x_0BE_0002_u64);
    for shapes in &[
        vec![(16usize, 4usize), (16, 4)],
        vec![(8, 3), (8, 5), (8, 1)],
        vec![(64, 8); 4],
    ] {
        run_shapes_parity(&cpu, &gpu, shapes, 0x_0BE_2000_u64);
    }
}

// ==========================================================================
// Mixed-height injection DAG (the OpenVM VERIFY_BATCH shape)
// ==========================================================================

#[test]
fn commit_mixed_height_two_levels_matches_openvm_reference() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, gpu) = build_matched(device, 0x_0BE_0003_u64);
    for shapes in &[
        vec![(8usize, 4usize), (4, 2)],
        vec![(16, 8), (8, 4)],
        vec![(32, 3), (16, 7)],
    ] {
        run_shapes_parity(&cpu, &gpu, shapes, 0x_0BE_3000_u64);
    }
}

#[test]
fn commit_mixed_height_three_levels_matches_openvm_reference() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, gpu) = build_matched(device, 0x_0BE_0004_u64);
    for shapes in &[
        vec![(16usize, 4usize), (8, 3), (2, 1)],
        vec![(32, 6), (16, 2), (4, 5)],
        vec![(64, 8), (32, 4), (4, 2)],
    ] {
        run_shapes_parity(&cpu, &gpu, shapes, 0x_0BE_4000_u64);
    }
}

#[test]
fn commit_mixed_height_every_level_matches_openvm_reference() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, gpu) = build_matched(device, 0x_0BE_0005_u64);
    let shapes: Vec<(usize, usize)> =
        [16usize, 8, 4, 2, 1].iter().map(|&h| (h, 3usize)).collect();
    run_shapes_parity(&cpu, &gpu, &shapes, 0x_0BE_5000_u64);
}

#[test]
fn commit_multi_matrices_at_same_non_max_height_matches_openvm_reference() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, gpu) = build_matched(device, 0x_0BE_0006_u64);
    for shapes in &[
        vec![(32usize, 4usize), (16, 3), (16, 5)],
        vec![(64, 2), (32, 8), (32, 4), (8, 1), (8, 6)],
    ] {
        run_shapes_parity(&cpu, &gpu, shapes, 0x_0BE_6000_u64);
    }
}

// ==========================================================================
// Scaled-down proxy for a realistic OpenVM shape
// ==========================================================================

#[test]
fn commit_bench_shape_proxy_matches_openvm_reference() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, gpu) = build_matched(device, 0x_0BE_0007_u64);
    // Trace + quotient-chunk-style injection at half height.
    let shapes = [(1024usize, 8usize), (512, 4)];
    run_shapes_parity(&cpu, &gpu, &shapes, 0x_0BE_7000_u64);
}

// ==========================================================================
// Guards
// ==========================================================================

#[test]
fn new_rejects_non_zero_cap_height() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut rng = SmallRng::seed_from_u64(0x_0BE_9001_u64);
    let ext: ExternalLayerConstants<BabyBear, 16> =
        ExternalLayerConstants::new_from_rng(ROUNDS_F, &mut rng);
    let int: Vec<BabyBear> =
        (&mut rng).sample_iter(StandardUniform).take(13).collect();
    let perm16: Perm = Perm::new(ext.clone(), int.clone());
    let zkgpu_params = babybear_openvm_params(&ext, &int);

    for bogus in [1usize, 2, 8] {
        let err = OpenVmGpuMmcs::new(
            device.clone(),
            perm16.clone(),
            zkgpu_params.clone(),
            zkgpu_params.clone(),
            bogus,
        );
        assert!(
            err.is_err(),
            "cap_height={bogus}: constructor must reject (Plonky3 0.4.1 has no \
             cap support in MerkleTreeMmcs anyway)"
        );
    }

    // cap_height=0 must succeed (golden path).
    OpenVmGpuMmcs::new(
        device,
        perm16,
        zkgpu_params.clone(),
        zkgpu_params,
        0,
    )
    .expect("cap_height=0 must construct cleanly");
}

#[test]
#[should_panic(expected = "called with 0 matrices")]
fn commit_panics_on_empty_input() {
    let Some(device) = try_device() else {
        panic!("called with 0 matrices (skipping GPU body)");
    };
    let (_cpu, gpu) = build_matched(device, 0x_0BE_9002_u64);
    let empty: Vec<RowMajorMatrix<BabyBear>> = Vec::new();
    let _ = gpu.commit(empty);
}
