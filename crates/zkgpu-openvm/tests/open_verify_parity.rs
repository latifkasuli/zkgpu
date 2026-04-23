//! Phase 3.d Stage 2b — OpenVM GPU MMCS open + verify parity.
//!
//! Pins both sides of the Stage 2b trait impl:
//!
//! * [`zkgpu_openvm::OpenVmGpuMmcs::open_batch`] (and the inherent
//!   `open_batch_inherent` wrapping it) against Plonky3 0.4.1's
//!   `MerkleTreeMmcs::open_batch` — same `(opened_values,
//!   opening_proof)` shape, element-by-element.
//! * `OpenVmGpuMmcs::verify_batch` accepting GPU-produced openings,
//!   and Plonky3's CPU `MerkleTreeMmcs::verify_batch` accepting the
//!   same openings — cross-verifier roundtrip.
//!
//! Both sides use the exact OpenVM config (W16/RATE=8/DIGEST=8,
//! single `Perm16` for leaf + compression).
//!
//! Coverage: single-matrix, same-height multi, mixed-height 2/3/5
//! levels, edge cases (h=1 with empty proof, multiple matrices at
//! the same non-max height). Every parity test also
//! cross-verifies: GPU opening → CPU verify, and GPU opening → GPU
//! verify (the latter tests the self-consistency of the adapter's
//! verify path).

use std::sync::Arc;

use p3_baby_bear::BabyBear;
use p3_commit::{BatchOpening, BatchOpeningRef, Mmcs};
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Dimensions;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_poseidon2::ExternalLayerConstants;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::distr::StandardUniform;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use zkgpu_openvm::{
    babybear_openvm_params, OpenVmGpuMmcs, Perm, Proof, Val, DIGEST_WIDTH,
};
use zkgpu_wgpu::WgpuDevice;

const ROUNDS_F: usize = 8;

type CpuValMmcs = MerkleTreeMmcs<
    <Val as p3_field::Field>::Packing,
    <Val as p3_field::Field>::Packing,
    PaddingFreeSponge<Perm, 16, 8, 8>,
    TruncatedPermutation<Perm, 2, 8, 16>,
    8,
>;

fn try_device() -> Option<Arc<WgpuDevice>> {
    WgpuDevice::new().ok()
    .map(Arc::new)
}

fn build_matched(device: Arc<WgpuDevice>, seed: u64) -> (CpuValMmcs, OpenVmGpuMmcs) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let ext: ExternalLayerConstants<BabyBear, 16> =
        ExternalLayerConstants::new_from_rng(ROUNDS_F, &mut rng);
    let int: Vec<BabyBear> =
        (&mut rng).sample_iter(StandardUniform).take(13).collect();
    let perm16: Perm = Perm::new(ext.clone(), int.clone());

    let cpu_sponge = PaddingFreeSponge::new(perm16.clone());
    let cpu_compress = TruncatedPermutation::new(perm16.clone());
    let cpu_mmcs = CpuValMmcs::new(cpu_sponge, cpu_compress);

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

/// Compare two `(opened_values, opening_proof)` pairs
/// element-by-element.
fn assert_openings_match(
    cpu_opened: &[Vec<Val>],
    cpu_proof: &[[Val; DIGEST_WIDTH]],
    gpu_opened: &[Vec<Val>],
    gpu_proof: &Proof,
    ctx: &str,
) {
    assert_eq!(
        cpu_opened.len(),
        gpu_opened.len(),
        "{ctx}: opened_values matrix count"
    );
    for (i, (c, g)) in cpu_opened.iter().zip(gpu_opened.iter()).enumerate() {
        assert_eq!(c.len(), g.len(), "{ctx}: matrix {i} row length");
        for (j, (cv, gv)) in c.iter().zip(g.iter()).enumerate() {
            assert_eq!(
                cv.as_canonical_u32(),
                gv.as_canonical_u32(),
                "{ctx}: matrix {i} col {j}"
            );
        }
    }
    assert_eq!(cpu_proof.len(), gpu_proof.len(), "{ctx}: proof length");
    for (level, (c, g)) in cpu_proof.iter().zip(gpu_proof.iter()).enumerate() {
        for k in 0..DIGEST_WIDTH {
            assert_eq!(
                c[k].as_canonical_u32(),
                g[k].as_canonical_u32(),
                "{ctx}: proof level {level} slot {k}"
            );
        }
    }
}

/// Drive both CPU and GPU commit + open at each supplied index
/// and assert byte-identical openings. Additionally verify the GPU
/// opening through:
/// 1. the adapter's own `verify_batch` (self-consistency)
/// 2. the CPU `MerkleTreeMmcs::verify_batch` (cross-verifier)
fn run_shapes_open_parity(
    cpu: &CpuValMmcs,
    gpu: &OpenVmGpuMmcs,
    shapes: &[(usize, usize)],
    indices: &[usize],
    seed_base: u64,
) {
    let ctx_base = format!("shapes={shapes:?}");
    let matrices: Vec<RowMajorMatrix<BabyBear>> = shapes
        .iter()
        .enumerate()
        .map(|(i, &(h, w))| random_matrix(h, w, seed_base ^ (i as u64) * 0xBADC0DE))
        .collect();
    let (cpu_cap, cpu_pd) = cpu.commit(matrices.clone());
    let (gpu_cap, gpu_pd) = gpu.commit(matrices);

    let dims: Vec<Dimensions> = shapes
        .iter()
        .map(|&(h, w)| Dimensions { width: w, height: h })
        .collect();

    // Sanity: commit roots match (covered by the commit_parity suite
    // too, but asserting here keeps each test self-describing).
    let cpu_root = OpenVmGpuMmcs::root(&cpu_cap);
    let gpu_root = OpenVmGpuMmcs::root(&gpu_cap);
    for k in 0..DIGEST_WIDTH {
        assert_eq!(
            cpu_root[k].as_canonical_u32(),
            gpu_root[k].as_canonical_u32(),
            "{ctx_base}: commit root slot {k}"
        );
    }

    for &idx in indices {
        let ctx = format!("{ctx_base} idx={idx}");

        // CPU opening via Plonky3 0.4.1's Mmcs trait.
        let cpu_opening: BatchOpening<BabyBear, CpuValMmcs> =
            cpu.open_batch(idx, &cpu_pd);
        let (cpu_opened, cpu_proof) = cpu_opening.unpack();

        // GPU opening via the adapter's Mmcs trait (Stage 2b).
        let gpu_opening_trait: BatchOpening<BabyBear, OpenVmGpuMmcs> =
            gpu.open_batch(idx, &gpu_pd);
        let (gpu_opened_trait, gpu_proof_trait) = gpu_opening_trait.unpack();

        // Also exercise the inherent method to confirm it matches
        // the trait dispatch.
        let (gpu_opened_inherent, gpu_proof_inherent) =
            gpu.open_batch_inherent(idx, &gpu_pd);

        // Trait- and inherent-produced openings must be identical.
        assert_openings_match(
            &gpu_opened_trait,
            &gpu_proof_trait,
            &gpu_opened_inherent,
            &gpu_proof_inherent,
            &format!("{ctx} (trait vs inherent)"),
        );

        // CPU ↔ GPU opening parity.
        assert_openings_match(
            &cpu_opened,
            &cpu_proof,
            &gpu_opened_trait,
            &gpu_proof_trait,
            &ctx,
        );

        // Adapter self-verify: GPU opening must verify against GPU
        // commit through the adapter's own `verify_batch`.
        let gpu_ref_self: BatchOpeningRef<'_, BabyBear, OpenVmGpuMmcs> =
            BatchOpeningRef::new(&gpu_opened_trait, &gpu_proof_trait);
        gpu.verify_batch(&gpu_cap, &dims, idx, gpu_ref_self)
            .unwrap_or_else(|e| {
                panic!("{ctx}: adapter self-verify rejected GPU opening: {e:?}")
            });

        // Cross-verifier: GPU opening must also pass through the CPU
        // `MerkleTreeMmcs::verify_batch`.
        let gpu_ref_cross: BatchOpeningRef<'_, BabyBear, CpuValMmcs> =
            BatchOpeningRef::new(&gpu_opened_trait, &gpu_proof_trait);
        cpu.verify_batch(&cpu_cap, &dims, idx, gpu_ref_cross)
            .unwrap_or_else(|e| {
                panic!(
                    "{ctx}: CPU MerkleTreeMmcs rejected GPU opening (cross-verifier): {e:?}"
                )
            });
    }
}

// ==========================================================================
// Single-matrix
// ==========================================================================

#[test]
fn open_single_matrix_matches_openvm_reference() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, gpu) = build_matched(device, 0x_2B_0001_u64);
    run_shapes_open_parity(
        &cpu,
        &gpu,
        &[(64, 8)],
        &[0, 1, 7, 31, 32, 63],
        0x_2BF_0000_u64,
    );
}

#[test]
fn open_single_row_matrix_matches_openvm_reference() {
    // h=1 edge case: no compression, empty proof.
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, gpu) = build_matched(device, 0x_2B_0002_u64);
    run_shapes_open_parity(&cpu, &gpu, &[(1, 8)], &[0], 0x_2BF_0010_u64);
}

// ==========================================================================
// Same-height multi-matrix (quotient-chunk-shape)
// ==========================================================================

#[test]
fn open_same_height_multi_matrix_matches_openvm_reference() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, gpu) = build_matched(device, 0x_2B_0003_u64);
    run_shapes_open_parity(
        &cpu,
        &gpu,
        &[(32, 4), (32, 3)],
        &[0, 5, 15, 31],
        0x_2BF_0020_u64,
    );
}

// ==========================================================================
// Mixed-height injection DAG (OpenVM VERIFY_BATCH shape)
// ==========================================================================

#[test]
fn open_mixed_height_two_levels_matches_openvm_reference() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, gpu) = build_matched(device, 0x_2B_0004_u64);
    run_shapes_open_parity(
        &cpu,
        &gpu,
        &[(16, 4), (4, 2)],
        &[0, 3, 7, 15],
        0x_2BF_0030_u64,
    );
}

#[test]
fn open_mixed_height_three_levels_matches_openvm_reference() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, gpu) = build_matched(device, 0x_2B_0005_u64);
    run_shapes_open_parity(
        &cpu,
        &gpu,
        &[(32, 6), (16, 2), (4, 5)],
        &[0, 1, 5, 12, 20, 31],
        0x_2BF_0040_u64,
    );
}

#[test]
fn open_mixed_height_every_level_matches_openvm_reference() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, gpu) = build_matched(device, 0x_2B_0006_u64);
    let shapes: Vec<(usize, usize)> =
        [16usize, 8, 4, 2, 1].iter().map(|&h| (h, 3usize)).collect();
    run_shapes_open_parity(&cpu, &gpu, &shapes, &[0, 5, 10, 15], 0x_2BF_0050_u64);
}

#[test]
fn open_multi_at_same_non_max_height_matches_openvm_reference() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, gpu) = build_matched(device, 0x_2B_0007_u64);
    run_shapes_open_parity(
        &cpu,
        &gpu,
        &[(32, 4), (16, 3), (16, 5)],
        &[0, 7, 15, 23, 31],
        0x_2BF_0060_u64,
    );
}

// ==========================================================================
// Scaled-down bench-shape proxy
// ==========================================================================

#[test]
fn open_bench_shape_proxy_matches_openvm_reference() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, gpu) = build_matched(device, 0x_2B_0008_u64);
    run_shapes_open_parity(
        &cpu,
        &gpu,
        &[(1024, 8), (512, 4)],
        &[0, 100, 500, 1023],
        0x_2BF_0070_u64,
    );
}

// ==========================================================================
// Trait-dispatch smoke test
// ==========================================================================

/// Confirm the adapter can plug into generic `Mmcs<BabyBear>`-
/// consuming code. This test takes the adapter by `&impl Mmcs<...>`
/// and exercises the full commit → open → verify pipeline through
/// trait dispatch only, validating that the Stage 2b `Mmcs` impl is
/// actually wired up (not just that the types line up).
#[test]
fn trait_generic_consumer_roundtrip() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (_cpu, gpu) = build_matched(device, 0x_2B_0009_u64);

    fn commit_and_verify<M>(
        mmcs: &M,
        inputs: Vec<RowMajorMatrix<BabyBear>>,
        indices: &[usize],
    ) where
        M: Mmcs<BabyBear>,
    {
        let (cap, pd) = mmcs.commit(inputs);
        // Exercise the full Mmcs surface — max_height + get_matrices
        // alongside commit/open/verify — so a regression in any trait
        // method shows up here.
        let _max_height = mmcs.get_max_height(&pd);
        let mats = mmcs.get_matrices(&pd);
        let dims: Vec<Dimensions> = mats
            .iter()
            .map(|m| Dimensions {
                height: p3_matrix::Matrix::<BabyBear>::height(*m),
                width: p3_matrix::Matrix::<BabyBear>::width(*m),
            })
            .collect();

        for &idx in indices {
            let opening = mmcs.open_batch(idx, &pd);
            let (opened, proof) = opening.unpack();
            let r: BatchOpeningRef<'_, BabyBear, M> = BatchOpeningRef::new(&opened, &proof);
            mmcs.verify_batch(&cap, &dims, idx, r)
                .expect("trait-generic verify");
        }
    }

    let matrices = vec![
        random_matrix(16, 4, 0x_2BF_A000_u64),
        random_matrix(8, 3, 0x_2BF_A001_u64),
        random_matrix(2, 2, 0x_2BF_A002_u64),
    ];
    commit_and_verify(&gpu, matrices, &[0, 5, 15]);
}

// ==========================================================================
// Negative tests
// ==========================================================================

#[test]
fn verify_batch_rejects_tampered_opening() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (_cpu, gpu) = build_matched(device, 0x_2B_0010_u64);

    let matrices = vec![random_matrix(16, 4, 0x_2BF_B000_u64)];
    let (cap, pd) = gpu.commit(matrices);
    let (mut opened, proof) = gpu.open_batch(3, &pd).unpack();
    let dims = vec![Dimensions { width: 4, height: 16 }];

    // Flip a single field element of opened_values.
    opened[0][0] = Val::new(opened[0][0].as_canonical_u32() ^ 1);

    let r: BatchOpeningRef<'_, BabyBear, OpenVmGpuMmcs> =
        BatchOpeningRef::new(&opened, &proof);
    let result = gpu.verify_batch(&cap, &dims, 3, r);
    assert!(
        result.is_err(),
        "verify_batch must reject tampered opened_values"
    );
}

#[test]
fn verify_batch_rejects_wrong_index() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (_cpu, gpu) = build_matched(device, 0x_2B_0011_u64);

    let matrices = vec![random_matrix(16, 4, 0x_2BF_B100_u64)];
    let (cap, pd) = gpu.commit(matrices);
    let (opened, proof) = gpu.open_batch(3, &pd).unpack();
    let dims = vec![Dimensions { width: 4, height: 16 }];

    // Present the opening as if it was for a different index.
    let r: BatchOpeningRef<'_, BabyBear, OpenVmGpuMmcs> =
        BatchOpeningRef::new(&opened, &proof);
    let result = gpu.verify_batch(&cap, &dims, 7, r);
    assert!(
        result.is_err(),
        "verify_batch must reject index mismatch"
    );
}
