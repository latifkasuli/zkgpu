//! Phase 7 Step 3.b — GPU Poseidon2 Merkle commit differential.
//!
//! Target: [`zkgpu_wgpu::WgpuPoseidon2MerkleCommit`] must produce a
//! Merkle root bit-identical to Plonky3's
//! `MerkleTreeMmcs<Packing, Packing, PaddingFreeSponge<Perm24,24,16,8>,
//!                 TruncatedPermutation<Perm16,2,8,16>, 2, 8>::commit`
//! when both sides are seeded with identical `(Perm16, Perm24)` round
//! constants and the same input matrix.
//!
//! This is the correctness gate for Step 3 — if this passes, the GPU
//! Merkle commit path is a semantic drop-in for Plonky3's canonical
//! `Poseidon2MerkleMmcs` config and we can wire it through the
//! Plonky3 `Mmcs` adapter for the bench gate.
//!
//! Scope (matches the GPU plan): **single matrix, power-of-two h**.
//! Multi-matrix commits and non-power-of-two heights are future work
//! — not required for the Step 3 `fri_commit` go/no-go bench.
//!
//! Skips the test body when no GPU adapter is available on the host
//! (matches the convention used by the other GPU differential tests).

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
use zkgpu_core::GpuDevice;
use zkgpu_plonky3::poseidon2_bridge::{babybear_plonky3_params, p3_to_zkgpu};
use zkgpu_wgpu::{WgpuDevice, WgpuPoseidon2MerkleCommit};

const ROUNDS_F: usize = 8;
const DIGEST_LEN: usize = 8;

type Val = P3BabyBear;
type Perm16 = Poseidon2BabyBear<16>;
type Perm24 = Poseidon2BabyBear<24>;
type Poseidon2Sponge = PaddingFreeSponge<Perm24, 24, 16, 8>;
type Poseidon2Compression = TruncatedPermutation<Perm16, 2, 8, 16>;
type ValMmcs = MerkleTreeMmcs<
    <Val as p3_field::Field>::Packing,
    <Val as p3_field::Field>::Packing,
    Poseidon2Sponge,
    Poseidon2Compression,
    2,
    8,
>;

fn try_device() -> Option<WgpuDevice> {
    WgpuDevice::new().ok()
}

/// Build a matched pair of CPU `MerkleTreeMmcs` and GPU
/// `WgpuPoseidon2MerkleCommit` with identical (Perm16, Perm24)
/// constants derived from `seed`. Uses two independent RNG draws for
/// W16 and W24 so the two permutations have distinct constants (as
/// they do in Plonky3's canonical config).
fn build_matched(
    device: &WgpuDevice,
    seed: u64,
) -> (ValMmcs, WgpuPoseidon2MerkleCommit) {
    // --- W16 (compression) constants ---
    let mut rng16 = SmallRng::seed_from_u64(seed ^ 0xB07_1600_u64);
    let ext16: ExternalLayerConstants<P3BabyBear, 16> =
        ExternalLayerConstants::new_from_rng(ROUNDS_F, &mut rng16);
    let int16: Vec<P3BabyBear> = (&mut rng16)
        .sample_iter(StandardUniform)
        .take(13) // Plonky3 BabyBear W16 rounds_p
        .collect();
    let p3_perm16: Perm16 = Perm16::new(ext16.clone(), int16.clone());
    let zkgpu_params16 = babybear_plonky3_params::<16>(&ext16, &int16);

    // --- W24 (leaf sponge) constants ---
    let mut rng24 = SmallRng::seed_from_u64(seed ^ 0xB07_2400_u64);
    let ext24: ExternalLayerConstants<P3BabyBear, 24> =
        ExternalLayerConstants::new_from_rng(ROUNDS_F, &mut rng24);
    let int24: Vec<P3BabyBear> = (&mut rng24)
        .sample_iter(StandardUniform)
        .take(21) // Plonky3 BabyBear W24 rounds_p
        .collect();
    let p3_perm24: Perm24 = Perm24::new(ext24.clone(), int24.clone());
    let zkgpu_params24 = babybear_plonky3_params::<24>(&ext24, &int24);

    // --- CPU MMCS (cap_height = 0 → commitment IS the root) ---
    let sponge = Poseidon2Sponge::new(p3_perm24);
    let compression = Poseidon2Compression::new(p3_perm16);
    let cpu_mmcs = ValMmcs::new(sponge, compression, 0);

    // --- GPU commit orchestrator ---
    let gpu_commit =
        WgpuPoseidon2MerkleCommit::new(device, zkgpu_params24, zkgpu_params16).unwrap();

    (cpu_mmcs, gpu_commit)
}

/// Row-major `h × w` random matrix as `(P3 matrix, flat GPU input)`.
fn random_matrix(
    h: usize,
    w: usize,
    seed: u64,
) -> (RowMajorMatrix<P3BabyBear>, Vec<ZkgpuBabyBear>) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let p3_values: Vec<P3BabyBear> =
        (&mut rng).sample_iter(StandardUniform).take(h * w).collect();
    let gpu_flat: Vec<ZkgpuBabyBear> =
        p3_values.iter().map(|e| p3_to_zkgpu(*e)).collect();
    (RowMajorMatrix::new(p3_values, w), gpu_flat)
}

fn cpu_root(mmcs: &ValMmcs, mat: RowMajorMatrix<P3BabyBear>) -> [P3BabyBear; DIGEST_LEN] {
    let (cap, _tree) = mmcs.commit(vec![mat]);
    // cap_height = 0 → cap is a single-element vec holding the root.
    let root: [P3BabyBear; DIGEST_LEN] = cap[0];
    root
}

fn assert_roots_match(
    cpu: [P3BabyBear; DIGEST_LEN],
    gpu: [ZkgpuBabyBear; DIGEST_LEN],
    ctx: &str,
) {
    for i in 0..DIGEST_LEN {
        assert_eq!(
            cpu[i].as_canonical_u32(),
            gpu[i].0,
            "{ctx}: root slot {i} mismatch"
        );
    }
}

/// Shared driver for all "random matrix, compare root" cases.
fn run_shape(
    device: &WgpuDevice,
    mmcs: &ValMmcs,
    commit: &mut WgpuPoseidon2MerkleCommit,
    h: usize,
    w: usize,
    seed: u64,
) {
    let (p3_mat, gpu_flat) = random_matrix(h, w, seed);
    let cpu = cpu_root(mmcs, p3_mat);
    let gpu = commit
        .commit_host_matrix(device, &gpu_flat, h as u32, w as u32)
        .unwrap();
    assert_roots_match(cpu, gpu, &format!("(h={h}, w={w})"));
}

// -- Height sweep (power-of-two only, Step 3.b scope) ----------------------

/// h = 1 — no compression; root is the leaf-sponge digest of the only
/// row. Edge case: tree has zero internal nodes.
#[test]
fn commit_root_matches_plonky3_h1() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (mmcs, mut commit) = build_matched(&device, 0x_C0FF_EE_30_u64);
    run_shape(&device, &mmcs, &mut commit, 1, 8, 0x_D006_0001_u64);
    run_shape(&device, &mmcs, &mut commit, 1, 17, 0x_D006_0011_u64);
}

/// h = 2 — one compression level. The root is `compress(leaf[0], leaf[1])`.
#[test]
fn commit_root_matches_plonky3_h2() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (mmcs, mut commit) = build_matched(&device, 0x_C0FF_EE_31_u64);
    run_shape(&device, &mmcs, &mut commit, 2, 8, 0x_D006_0002_u64);
}

/// h = 4, 8, 16 — exercises the ping-pong buffer logic across 2/3/4
/// compression levels. At these depths any off-by-one in the
/// read-from-ping / read-from-pong toggle surfaces immediately.
#[test]
fn commit_root_matches_plonky3_small_powers_of_two() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (mmcs, mut commit) = build_matched(&device, 0x_C0FF_EE_32_u64);
    for log_h in 2..=4 {
        let h = 1usize << log_h;
        run_shape(&device, &mmcs, &mut commit, h, 8, 0x_D006_0000_u64 + h as u64);
    }
}

/// h = 1024, w = 8 — the go/no-go bench gate uses `log_h=18, w=8`.
/// Full parity at that scale is an in-test cost too far for CI, so
/// this smaller shape exercises the same code path with room to spare.
#[test]
fn commit_root_matches_plonky3_log10_w8() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (mmcs, mut commit) = build_matched(&device, 0x_C0FF_EE_33_u64);
    run_shape(&device, &mmcs, &mut commit, 1024, 8, 0x_D006_0400_u64);
}

// -- Width sweep (row_width bucket coverage) ------------------------------

/// w < RATE (RATE=16): leaf sponge takes the "remainder only" branch.
/// Exercise every w in {1, 7, 15} at a fixed mid-size h.
#[test]
fn commit_root_matches_plonky3_narrow_widths() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (mmcs, mut commit) = build_matched(&device, 0x_C0FF_EE_34_u64);
    for &w in &[1usize, 7, 15] {
        run_shape(&device, &mmcs, &mut commit, 8, w, 0x_D006_0800_u64 + w as u64);
    }
}

/// w exactly RATE, and RATE + r: leaf sponge takes the full-chunk
/// and chunk+partial branches, respectively.
#[test]
fn commit_root_matches_plonky3_rate_boundary_widths() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (mmcs, mut commit) = build_matched(&device, 0x_C0FF_EE_35_u64);
    for &w in &[16usize, 17, 31, 32, 33] {
        run_shape(&device, &mmcs, &mut commit, 8, w, 0x_D006_0900_u64 + w as u64);
    }
}

// -- Guards --------------------------------------------------------------

#[test]
fn commit_rejects_non_power_of_two_height() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (_mmcs, mut commit) = build_matched(&device, 0x_C0FF_EE_36_u64);
    let bogus: Vec<ZkgpuBabyBear> = vec![ZkgpuBabyBear(1); 3 * 8];
    let err = commit.commit_host_matrix(&device, &bogus, 3, 8);
    assert!(
        err.is_err(),
        "commit must reject non-power-of-two h (got h=3), got Ok"
    );
}

#[test]
fn commit_rejects_shape_mismatch() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (_mmcs, mut commit) = build_matched(&device, 0x_C0FF_EE_37_u64);
    // Claim h=4 w=8 (expects 32 elements) but only supply 30.
    let bogus: Vec<ZkgpuBabyBear> = vec![ZkgpuBabyBear(1); 30];
    let err = commit.commit_host_matrix(&device, &bogus, 4, 8);
    assert!(
        err.is_err(),
        "commit must reject slice length != h*w, got Ok"
    );
}

#[test]
fn commit_rejects_zero_height() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (_mmcs, mut commit) = build_matched(&device, 0x_C0FF_EE_38_u64);
    let empty: Vec<ZkgpuBabyBear> = Vec::new();
    let err = commit.commit_host_matrix(&device, &empty, 0, 8);
    assert!(
        err.is_err(),
        "commit must reject h=0 (Plonky3 also panics on h=0), got Ok"
    );
}

// -- GPU-resident `commit()` seam --------------------------------------
//
// The production path (Step 2 GPU-resident coset LDE, Plonky3 `Mmcs`
// adapter) calls `WgpuPoseidon2MerkleCommit::commit` directly with an
// already-uploaded matrix buffer, bypassing `commit_host_matrix`. All
// shape guards must live on the GPU-resident seam too — otherwise a
// GPU-resident caller gets weaker contract enforcement than the host
// convenience wrapper.

/// Parity: the GPU-resident `commit()` entry point (what the Plonky3
/// `Mmcs` adapter will call) must produce the same root as the host
/// convenience wrapper, which is already pinned against Plonky3 by
/// the tests above. This locks the production seam, not just the
/// convenience path.
#[test]
fn commit_gpu_resident_matches_plonky3() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (mmcs, mut commit) = build_matched(&device, 0x_C0FF_EE_39_u64);
    // One shape from each major regime: short row (leaf partial),
    // exact chunk boundary, deeper tree.
    for &(h, w) in &[(8usize, 5usize), (16, 16), (64, 8)] {
        let (p3_mat, gpu_flat) = random_matrix(h, w, 0x_D006_9000_u64 ^ (h as u64) << 16 ^ (w as u64));
        let cpu = cpu_root(&mmcs, p3_mat);

        // Upload the matrix and call the GPU-resident seam directly.
        let matrix_buf = device.upload::<ZkgpuBabyBear>(&gpu_flat).unwrap();
        let gpu = commit
            .commit(&device, &matrix_buf, h as u32, w as u32)
            .unwrap();
        assert_roots_match(cpu, gpu, &format!("GPU-resident commit (h={h}, w={w})"));
    }
}

/// The P2 review finding: `commit()` must reject `w == 0` up front,
/// not just `commit_host_matrix`. A GPU-resident caller who uploads a
/// zero-length buffer and passes `h > 0, w == 0` must see an error,
/// not a silently produced root.
#[test]
fn commit_gpu_resident_rejects_zero_width() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (_mmcs, mut commit) = build_matched(&device, 0x_C0FF_EE_3A_u64);
    // Upload a 1-element placeholder — wgpu rejects zero-size buffer
    // init, so we need *some* buffer to pass into commit(). The shape
    // check inside commit() is what we're testing: (h=4, w=0) ⇒
    // expected_input_len = 0 ≠ buffer.len() = 1, but the w==0 guard
    // fires FIRST, so the error message should be the "w must be ≥ 1"
    // one — not the length-mismatch one.
    let placeholder = device
        .upload::<ZkgpuBabyBear>(&[ZkgpuBabyBear(0)])
        .unwrap();
    let err = commit.commit(&device, &placeholder, 4, 0);
    match err {
        Err(zkgpu_core::ZkGpuError::InvalidNttSize(msg)) => {
            assert!(
                msg.contains("w must be ≥ 1"),
                "expected w==0 guard to fire first, got: {msg}"
            );
        }
        Err(e) => panic!("expected InvalidNttSize(w must be ≥ 1), got {e:?}"),
        Ok(_) => panic!("commit() must reject w=0, got Ok"),
    }
}

/// Same P2 finding, surfaced through the host wrapper: it must still
/// reject `w == 0` (delegation to `commit()` must actually run the
/// guard).
#[test]
fn commit_host_wrapper_rejects_zero_width() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (_mmcs, mut commit) = build_matched(&device, 0x_C0FF_EE_3B_u64);
    let empty: Vec<ZkgpuBabyBear> = Vec::new();
    let err = commit.commit_host_matrix(&device, &empty, 4, 0);
    assert!(
        err.is_err(),
        "commit_host_matrix must reject w=0 via delegated commit() guard, got Ok"
    );
}

/// `commit()` must also enforce power-of-two h on the GPU-resident
/// seam, not just in the convenience wrapper.
#[test]
fn commit_gpu_resident_rejects_non_power_of_two_height() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (_mmcs, mut commit) = build_matched(&device, 0x_C0FF_EE_3C_u64);
    let bogus: Vec<ZkgpuBabyBear> = vec![ZkgpuBabyBear(1); 3 * 8];
    let buf = device.upload::<ZkgpuBabyBear>(&bogus).unwrap();
    let err = commit.commit(&device, &buf, 3, 8);
    assert!(
        err.is_err(),
        "commit() must reject non-power-of-two h (got h=3), got Ok"
    );
}

/// `commit()` must enforce `matrix.len() == h*w` on the GPU-resident
/// seam — a caller who uploads the wrong-sized buffer must see an
/// error, not a corrupted root.
#[test]
fn commit_gpu_resident_rejects_shape_mismatch() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (_mmcs, mut commit) = build_matched(&device, 0x_C0FF_EE_3D_u64);
    // Upload 30 elements but claim h=4, w=8 (expects 32).
    let bogus: Vec<ZkgpuBabyBear> = vec![ZkgpuBabyBear(1); 30];
    let buf = device.upload::<ZkgpuBabyBear>(&bogus).unwrap();
    let err = commit.commit(&device, &buf, 4, 8);
    assert!(
        err.is_err(),
        "commit() must reject matrix.len() != h*w, got Ok"
    );
}
