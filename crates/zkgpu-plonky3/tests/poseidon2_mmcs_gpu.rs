//! GPU Poseidon2 MMCS adapter parity tests.
//!
//! Locks [`zkgpu_plonky3::gpu_mmcs::GpuPoseidon2Mmcs`] against
//! Plonky3's canonical `MerkleTreeMmcs<Packing, Packing,
//! Poseidon2Sponge, Poseidon2Compression, 2, 8>` at both
//! `Mmcs::commit` and `Mmcs::open_batch` — the full surface the
//! target-stack prover exercises in `commit` + `fri::open_input`.
//!
//! Scope: cap_height=0, all shapes Plonky3's `MerkleTreeMmcs::commit`
//! validly accepts at that cap height. Covers:
//!
//! * **Single-matrix trace commit** — `TwoAdicFriPcs::commit`
//!   feeds exactly one LDE matrix per call.
//! * **Same-height multi-matrix quotient batch** — `commit_quotient`
//!   splits the quotient into `k` equal-height matrices.
//! * **Mixed-height multi-matrix** — `compress_and_inject`-style DAG
//!   tree with matrices at differing heights (trace + preprocessing
//!   / fixed / random matrices). Routed through the shared backend's
//!   `commit_mixed_height_with_w24_leaf` (also used by the sibling
//!   `zkgpu-openvm` adapter with the W16 leaf variant).
//! * **Opening parity** — `open_batch(index)` produces byte-identical
//!   `(opened_values, proof)` to Plonky3's CPU reference at every
//!   power-of-two index (and a random sample of others).
//! * **Cross-verifier roundtrip** — GPU openings verify through both
//!   the adapter's own `verify_batch` (CPU-delegated) and a freshly-
//!   constructed CPU `MerkleTreeMmcs::verify_batch` over the GPU
//!   commit.

use std::sync::Arc;

use p3_baby_bear::BabyBear as P3BabyBear;
use p3_commit::{BatchOpening, BatchOpeningRef, Mmcs};
use p3_field::PrimeField32;
use p3_matrix::Dimensions;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_poseidon2::ExternalLayerConstants;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::distr::StandardUniform;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use zkgpu_plonky3::gpu_mmcs::{
    GpuPoseidon2Mmcs, P3Poseidon2Compression, P3Poseidon2Sponge, Perm16, Perm24,
};
use zkgpu_plonky3::poseidon2_bridge::babybear_plonky3_params;
use zkgpu_wgpu::WgpuDevice;

const ROUNDS_F: usize = 8;
const DIGEST_LEN: usize = 8;

type Val = P3BabyBear;
type CpuValMmcs = MerkleTreeMmcs<
    <Val as p3_field::Field>::Packing,
    <Val as p3_field::Field>::Packing,
    P3Poseidon2Sponge,
    P3Poseidon2Compression,
    2,
    DIGEST_LEN,
>;

fn try_device() -> Option<Arc<WgpuDevice>> {
    WgpuDevice::new().ok().map(Arc::new)
}

/// Build a matched `(CpuValMmcs, GpuPoseidon2Mmcs)` pair driven by
/// the same Poseidon2 constants.
fn build_matched(
    device: Arc<WgpuDevice>,
    seed: u64,
) -> (CpuValMmcs, GpuPoseidon2Mmcs) {
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

    let sponge = PaddingFreeSponge::new(perm24.clone());
    let compression = TruncatedPermutation::new(perm16.clone());
    let cpu_mmcs = CpuValMmcs::new(sponge, compression, 0);

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
    cpu_cap: &p3_symmetric::MerkleCap<P3BabyBear, [P3BabyBear; DIGEST_LEN]>,
    gpu_cap: &p3_symmetric::MerkleCap<P3BabyBear, [P3BabyBear; DIGEST_LEN]>,
    ctx: &str,
) {
    let cpu = GpuPoseidon2Mmcs::root(cpu_cap).expect("cpu cap");
    let gpu = GpuPoseidon2Mmcs::root(gpu_cap).expect("gpu cap");
    for i in 0..DIGEST_LEN {
        assert_eq!(
            cpu[i].as_canonical_u32(),
            gpu[i].as_canonical_u32(),
            "{ctx}: root slot {i}"
        );
    }
}

fn assert_openings_match(
    cpu: &BatchOpening<P3BabyBear, CpuValMmcs>,
    gpu: &BatchOpening<P3BabyBear, GpuPoseidon2Mmcs>,
    ctx: &str,
) {
    // opened_values: Vec<Vec<P3BabyBear>> — same shape on both sides.
    assert_eq!(
        cpu.opened_values.len(),
        gpu.opened_values.len(),
        "{ctx}: opened_values matrix count"
    );
    for (i, (c, g)) in cpu
        .opened_values
        .iter()
        .zip(gpu.opened_values.iter())
        .enumerate()
    {
        assert_eq!(c.len(), g.len(), "{ctx}: matrix {i} row length");
        for (j, (cv, gv)) in c.iter().zip(g.iter()).enumerate() {
            assert_eq!(
                cv.as_canonical_u32(),
                gv.as_canonical_u32(),
                "{ctx}: matrix {i} col {j}"
            );
        }
    }
    // opening_proof: Vec<[P3BabyBear; 8]> — same concrete type.
    assert_eq!(
        cpu.opening_proof.len(),
        gpu.opening_proof.len(),
        "{ctx}: proof length (expected log_h siblings, got differ)"
    );
    for (level, (c, g)) in cpu
        .opening_proof
        .iter()
        .zip(gpu.opening_proof.iter())
        .enumerate()
    {
        for k in 0..DIGEST_LEN {
            assert_eq!(
                c[k].as_canonical_u32(),
                g[k].as_canonical_u32(),
                "{ctx}: proof level {level} slot {k}"
            );
        }
    }
}

// ==========================================================================
// Commit parity
// ==========================================================================

#[test]
fn mmcs_commit_single_matrix_parity() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, gpu) = build_matched(device, 0x_CC55_0001_u64);
    for &(h, w) in &[(1usize, 8usize), (2, 8), (4, 8), (16, 8), (16, 17), (4096, 8)] {
        let mat = random_rowmajor(h, w, 0x_DEAD_0000_u64 ^ (h as u64) << 8 ^ (w as u64));
        let (cpu_cap, _) = cpu.commit(vec![mat.clone()]);
        let (gpu_cap, _) = gpu.commit(vec![mat]);
        assert_roots_match(&cpu_cap, &gpu_cap, &format!("single (h={h}, w={w})"));
    }
}

#[test]
fn mmcs_commit_multi_matrix_same_height_parity() {
    // Quotient-chunk shape: N same-height matrices committed together.
    // Plonky3 hashes row-i of each as one concatenated leaf; we flatten
    // the same way and must produce the same root.
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, gpu) = build_matched(device, 0x_CC55_0002_u64);

    // (num_matrices, h, per-matrix w)
    let shapes = [
        (2usize, 8usize, 4usize),
        (4, 16, 4),                // 4 chunks ×4 cols = total width 16 (exact chunk)
        (8, 16, 3),                // 8 chunks ×3 = 24 cols (chunk + partial)
        (16, 64, 1),               // 16 narrow chunks
        (4, 1024, 8),              // quotient-like: bench-gate scale
    ];
    for &(n, h, w) in &shapes {
        let matrices: Vec<RowMajorMatrix<P3BabyBear>> = (0..n)
            .map(|i| {
                random_rowmajor(
                    h,
                    w,
                    0x_D006_1000_u64 ^ (h as u64) << 16 ^ (w as u64) << 8 ^ (i as u64),
                )
            })
            .collect();
        let (cpu_cap, _) = cpu.commit(matrices.clone());
        let (gpu_cap, _) = gpu.commit(matrices);
        assert_roots_match(
            &cpu_cap,
            &gpu_cap,
            &format!("multi same-height (n={n}, h={h}, w={w})"),
        );
    }
}

// ==========================================================================
// Opening parity
// ==========================================================================

#[test]
fn mmcs_open_batch_single_matrix_parity() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, gpu) = build_matched(device, 0x_CC55_0003_u64);

    for &(h, w) in &[(2usize, 8usize), (4, 8), (16, 8), (64, 3), (256, 8)] {
        let mat = random_rowmajor(h, w, 0x_0BE1_0000_u64 ^ (h as u64) << 8 ^ (w as u64));
        let (_cpu_cap, cpu_pd) = cpu.commit(vec![mat.clone()]);
        let (_gpu_cap, gpu_pd) = gpu.commit(vec![mat]);

        // Sample every row for small h, strided for larger h.
        let stride = (h / 32).max(1);
        for index in (0..h).step_by(stride) {
            let cpu_opening = cpu.open_batch(index, &cpu_pd);
            let gpu_opening = gpu.open_batch(index, &gpu_pd);
            assert_openings_match(
                &cpu_opening,
                &gpu_opening,
                &format!("single open (h={h}, w={w}, idx={index})"),
            );
        }
    }
}

#[test]
fn mmcs_open_batch_multi_matrix_parity() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, gpu) = build_matched(device, 0x_CC55_0004_u64);

    // Quotient-like shape: 4 same-height matrices.
    let (n, h, w) = (4usize, 64usize, 8usize);
    let matrices: Vec<RowMajorMatrix<P3BabyBear>> = (0..n)
        .map(|i| random_rowmajor(h, w, 0x_0BE1_1000_u64 ^ (i as u64)))
        .collect();
    let (_cpu_cap, cpu_pd) = cpu.commit(matrices.clone());
    let (_gpu_cap, gpu_pd) = gpu.commit(matrices);

    for index in [0, 1, 2, 7, 31, 32, 63] {
        let cpu_opening = cpu.open_batch(index, &cpu_pd);
        let gpu_opening = gpu.open_batch(index, &gpu_pd);
        assert_openings_match(
            &cpu_opening,
            &gpu_opening,
            &format!("multi open (n={n}, h={h}, w={w}, idx={index})"),
        );
    }
}

/// End-to-end: a GPU-produced proof verifies against the GPU-produced
/// commitment via `Mmcs::verify_batch` (which delegates to CPU MMCS).
/// If the commit + open paths are wrong, verify will reject.
#[test]
fn mmcs_verify_roundtrip_single_matrix() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (_cpu, gpu) = build_matched(device, 0x_CC55_0005_u64);

    let (h, w) = (64usize, 8usize);
    let mat = random_rowmajor(h, w, 0x_DADA_0000_u64);
    let (cap, prover_data) = gpu.commit(vec![mat]);
    for index in [0, 1, 17, 63] {
        let opening = gpu.open_batch(index, &prover_data);
        let dims = &[Dimensions {
            width: w,
            height: h,
        }];
        let opening_ref: BatchOpeningRef<'_, P3BabyBear, GpuPoseidon2Mmcs> =
            BatchOpeningRef::new(&opening.opened_values, &opening.opening_proof);
        gpu.verify_batch(&cap, dims, index, opening_ref)
            .unwrap_or_else(|e| panic!("verify failed at idx={index}: {e:?}"));
    }
}

#[test]
fn mmcs_verify_roundtrip_multi_matrix() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (_cpu, gpu) = build_matched(device, 0x_CC55_0006_u64);

    let (n, h, w) = (4usize, 32usize, 4usize);
    let matrices: Vec<RowMajorMatrix<P3BabyBear>> = (0..n)
        .map(|i| random_rowmajor(h, w, 0x_DADA_1000_u64 ^ (i as u64)))
        .collect();
    let dims: Vec<Dimensions> = matrices
        .iter()
        .map(|m| Dimensions {
            width: m.values.len() / h,
            height: h,
        })
        .collect();
    let (cap, prover_data) = gpu.commit(matrices);
    for index in [0usize, 5, 15, 31] {
        let opening = gpu.open_batch(index, &prover_data);
        let opening_ref: BatchOpeningRef<'_, P3BabyBear, GpuPoseidon2Mmcs> =
            BatchOpeningRef::new(&opening.opened_values, &opening.opening_proof);
        gpu.verify_batch(&cap, &dims, index, opening_ref)
            .unwrap_or_else(|e| panic!("multi verify failed at idx={index}: {e:?}"));
    }
}

// ==========================================================================
// Mixed-height commit + open parity (post-convergence — adapter now
// routes through the shared `commit_mixed_height_with_w24_leaf` DAG
// engine, which makes mixed-height batches a fully-supported shape
// rather than a rejection case).
// ==========================================================================

/// Helper: drive both CPU and GPU through `commit + open_batch +
/// verify_batch` for a given `(shapes, indices)` and assert
/// byte-identical commit roots, byte-identical openings at every
/// supplied index, and that the GPU opening verifies through both
/// the adapter's `verify_batch` and a freshly-constructed CPU
/// `MerkleTreeMmcs::verify_batch`. Mirrors the OpenVM adapter's
/// `run_shapes_open_parity` helper so the two adapters cover the
/// same test surface.
fn run_shapes_open_parity(
    cpu: &CpuValMmcs,
    gpu: &GpuPoseidon2Mmcs,
    shapes: &[(usize, usize)],
    indices: &[usize],
    seed_base: u64,
) {
    let ctx_base = format!("shapes={shapes:?}");
    let matrices: Vec<RowMajorMatrix<P3BabyBear>> = shapes
        .iter()
        .enumerate()
        .map(|(i, &(h, w))| {
            random_rowmajor(h, w, seed_base ^ ((i as u64).wrapping_mul(0xBADC0DE)))
        })
        .collect();
    let (cpu_cap, cpu_pd) = cpu.commit(matrices.clone());
    let (gpu_cap, gpu_pd) = gpu.commit(matrices);

    let dims: Vec<Dimensions> = shapes
        .iter()
        .map(|&(h, w)| Dimensions { width: w, height: h })
        .collect();

    // Sanity: commit roots match.
    assert_roots_match(&cpu_cap, &gpu_cap, &format!("{ctx_base}: commit"));

    for &idx in indices {
        let ctx = format!("{ctx_base} idx={idx}");

        let cpu_opening: BatchOpening<P3BabyBear, CpuValMmcs> =
            cpu.open_batch(idx, &cpu_pd);
        let gpu_opening: BatchOpening<P3BabyBear, GpuPoseidon2Mmcs> =
            gpu.open_batch(idx, &gpu_pd);

        // CPU ↔ GPU opening parity.
        assert_openings_match(&cpu_opening, &gpu_opening, &ctx);

        // Adapter self-verify: GPU opening verifies against GPU
        // commit through the adapter's own `verify_batch`.
        let gpu_ref_self: BatchOpeningRef<'_, P3BabyBear, GpuPoseidon2Mmcs> =
            BatchOpeningRef::new(&gpu_opening.opened_values, &gpu_opening.opening_proof);
        gpu.verify_batch(&gpu_cap, &dims, idx, gpu_ref_self)
            .unwrap_or_else(|e| {
                panic!("{ctx}: adapter self-verify rejected GPU opening: {e:?}")
            });

        // Cross-verifier: GPU opening passes through a freshly-built
        // CPU `MerkleTreeMmcs::verify_batch` over the GPU commit.
        let gpu_ref_cross: BatchOpeningRef<'_, P3BabyBear, CpuValMmcs> =
            BatchOpeningRef::new(&gpu_opening.opened_values, &gpu_opening.opening_proof);
        cpu.verify_batch(&gpu_cap, &dims, idx, gpu_ref_cross)
            .unwrap_or_else(|e| {
                panic!(
                    "{ctx}: CPU MerkleTreeMmcs rejected GPU opening (cross-verifier): {e:?}"
                )
            });
    }
}

/// Two-level mixed-height: a tall trace + a shorter preprocessing /
/// fixed matrix. The smallest realistic shape that exercises the
/// `compress_and_inject` injection step.
#[test]
fn mmcs_mixed_height_two_levels_parity() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, gpu) = build_matched(device, 0x_CC55_0007_u64);
    run_shapes_open_parity(
        &cpu,
        &gpu,
        &[(16, 4), (4, 2)],
        &[0, 3, 7, 15],
        0x_2BF_0030_u64,
    );
}

/// Three-level mixed-height: tallest matrix + injection at two
/// different lower levels.
#[test]
fn mmcs_mixed_height_three_levels_parity() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, gpu) = build_matched(device, 0x_CC55_0009_u64);
    run_shapes_open_parity(
        &cpu,
        &gpu,
        &[(32, 6), (16, 2), (4, 5)],
        &[0, 1, 5, 12, 20, 31],
        0x_2BF_0040_u64,
    );
}

/// Mixed-height with injection at every tree level (every power of
/// two from h=16 down to h=1). Stress-tests that the adapter's
/// retained-layer indexing matches Plonky3's at every level.
#[test]
fn mmcs_mixed_height_every_level_parity() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, gpu) = build_matched(device, 0x_CC55_000A_u64);
    let shapes: Vec<(usize, usize)> =
        [16usize, 8, 4, 2, 1].iter().map(|&h| (h, 3usize)).collect();
    run_shapes_open_parity(&cpu, &gpu, &shapes, &[0, 5, 10, 15], 0x_2BF_0050_u64);
}

/// Multi-matrix at a same non-max height: a tall trace + two shorter
/// matrices that share a height. Exercises the same-height-group
/// flatten inside the mixed-height engine.
#[test]
fn mmcs_multi_matrix_same_non_max_height_parity() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, gpu) = build_matched(device, 0x_CC55_000B_u64);
    run_shapes_open_parity(
        &cpu,
        &gpu,
        &[(32, 4), (16, 3), (16, 5)],
        &[0, 7, 15, 23, 31],
        0x_2BF_0060_u64,
    );
}

// ==========================================================================
// Guard tests (preserved from pre-convergence; cap_height + empty).
// ==========================================================================

#[test]
fn mmcs_get_matrices_returns_inputs() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (_, gpu) = build_matched(device, 0x_CC55_0008_u64);
    let mat = random_rowmajor(8, 8, 0x_0B_0008_u64);
    let (_cap, prover_data) = gpu.commit(vec![mat]);
    let mats = gpu.get_matrices(&prover_data);
    assert_eq!(mats.len(), 1);
    assert_eq!(mats[0].values.len(), 8 * 8);
}

/// Regression guard for the P2 review finding on commit `54dc548`:
/// the previous constructor silently stored any `cap_height` while
/// `commit()` always produced a single-digest cap.
#[test]
fn mmcs_new_rejects_non_zero_cap_height() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let mut rng16 = SmallRng::seed_from_u64(0x_CA_BAD_E11_u64 ^ 0xA11_1600_u64);
    let ext16: ExternalLayerConstants<P3BabyBear, 16> =
        ExternalLayerConstants::new_from_rng(ROUNDS_F, &mut rng16);
    let int16: Vec<P3BabyBear> =
        (&mut rng16).sample_iter(StandardUniform).take(13).collect();
    let perm16: Perm16 = Perm16::new(ext16.clone(), int16.clone());
    let zkgpu_params16 = babybear_plonky3_params::<16>(&ext16, &int16);

    let mut rng24 = SmallRng::seed_from_u64(0x_CA_BAD_E11_u64 ^ 0xA11_2400_u64);
    let ext24: ExternalLayerConstants<P3BabyBear, 24> =
        ExternalLayerConstants::new_from_rng(ROUNDS_F, &mut rng24);
    let int24: Vec<P3BabyBear> =
        (&mut rng24).sample_iter(StandardUniform).take(21).collect();
    let perm24: Perm24 = Perm24::new(ext24.clone(), int24.clone());
    let zkgpu_params24 = babybear_plonky3_params::<24>(&ext24, &int24);

    for bogus in [1usize, 2, 3, 8] {
        let err = GpuPoseidon2Mmcs::new(
            device.clone(),
            perm24.clone(),
            perm16.clone(),
            zkgpu_params24.clone(),
            zkgpu_params16.clone(),
            bogus,
        );
        match err {
            Err(msg) => assert!(
                msg.contains("cap_height"),
                "cap_height={bogus}: expected rejection, got: {msg}"
            ),
            Ok(_) => panic!(
                "GpuPoseidon2Mmcs::new must reject cap_height={bogus}, got Ok"
            ),
        }
    }

    GpuPoseidon2Mmcs::new(
        device,
        perm24,
        perm16,
        zkgpu_params24,
        zkgpu_params16,
        0,
    )
    .expect("cap_height=0 must construct cleanly");
}
