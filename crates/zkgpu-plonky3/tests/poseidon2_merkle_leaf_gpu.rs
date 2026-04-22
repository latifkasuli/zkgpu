//! Phase 7 Step 3.a — GPU Poseidon2 leaf sponge differential test.
//!
//! Target: `WgpuPoseidon2MerkleLeafPlan` output must match
//! `p3_symmetric::PaddingFreeSponge<Poseidon2BabyBear<24>, 24, 16, 8>::hash_iter`
//! row-by-row for arbitrary `(num_leaves, row_width)` shapes.
//!
//! This is the semantic correctness gate for the leaf-sponge leg of
//! Step 3's Poseidon2 Merkle commit. Row-widths probe the three
//! absorption branches of `PaddingFreeSponge`'s padding-free,
//! overwrite-mode loop:
//!
//!   * `row_width == 0`           — no permutation, output is all zero
//!   * `row_width < RATE`         — single partial-chunk permute
//!   * `row_width == k * RATE`    — exactly `k` full-chunk permutes
//!   * `row_width == k*RATE + r`  — `k` full + 1 partial permute
//!
//! Together with the per-permutation Step 1.5b tests in
//! `poseidon2_bridge_gpu.rs`, this locks both primitives used by
//! Plonky3's `Poseidon2MerkleMmcs` leaf path on the GPU side.
//!
//! Skips the test body when no GPU adapter is available (matches the
//! convention used by the other GPU differential tests).

use p3_baby_bear::{BabyBear as P3BabyBear, Poseidon2BabyBear};
use p3_field::{PrimeCharacteristicRing, PrimeField32};
use p3_poseidon2::ExternalLayerConstants;
use p3_symmetric::{CryptographicHasher, PaddingFreeSponge};
use rand::distr::StandardUniform;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use zkgpu_babybear::BabyBear as ZkgpuBabyBear;
use zkgpu_plonky3::poseidon2_bridge::{babybear_plonky3_params, p3_to_zkgpu};
use zkgpu_wgpu::{WgpuDevice, WgpuPoseidon2MerkleLeafPlan};

const ROUNDS_F: usize = 8;
const WIDTH: usize = 24;
const RATE: usize = 16;
const DIGEST_LEN: usize = 8;

type Perm24 = Poseidon2BabyBear<24>;
type Poseidon2Sponge = PaddingFreeSponge<Perm24, 24, 16, 8>;

fn try_device() -> Option<WgpuDevice> {
    WgpuDevice::new().ok()
}

/// Build a matched (CPU PaddingFreeSponge, GPU leaf plan) pair driven
/// by identical constants from the Step 1.5a bridge.
fn build_matched(
    device: &WgpuDevice,
    seed: u64,
) -> (Poseidon2Sponge, WgpuPoseidon2MerkleLeafPlan) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let ext: ExternalLayerConstants<P3BabyBear, 24> =
        ExternalLayerConstants::new_from_rng(ROUNDS_F, &mut rng);
    let int: Vec<P3BabyBear> =
        (&mut rng).sample_iter(StandardUniform).take(21).collect();

    let p3_perm: Perm24 = Perm24::new(ext.clone(), int.clone());
    let sponge = Poseidon2Sponge::new(p3_perm);

    let zkgpu_params = babybear_plonky3_params::<24>(&ext, &int);
    let plan = WgpuPoseidon2MerkleLeafPlan::new(device, zkgpu_params).unwrap();
    (sponge, plan)
}

/// Random `h × w` matrix as (P3 row-major matrix, GPU flat input).
fn random_matrix(
    num_leaves: usize,
    row_width: usize,
    seed: u64,
) -> (Vec<Vec<P3BabyBear>>, Vec<ZkgpuBabyBear>) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut p3_rows: Vec<Vec<P3BabyBear>> = Vec::with_capacity(num_leaves);
    let mut gpu_flat: Vec<ZkgpuBabyBear> = Vec::with_capacity(num_leaves * row_width);
    for _ in 0..num_leaves {
        let row: Vec<P3BabyBear> =
            (&mut rng).sample_iter(StandardUniform).take(row_width).collect();
        gpu_flat.extend(row.iter().map(|e| p3_to_zkgpu(*e)));
        p3_rows.push(row);
    }
    (p3_rows, gpu_flat)
}

/// Runs CPU-vs-GPU differential for a specific `(num_leaves, row_width)`.
fn run_shape(
    device: &WgpuDevice,
    sponge: &Poseidon2Sponge,
    plan: &mut WgpuPoseidon2MerkleLeafPlan,
    num_leaves: usize,
    row_width: usize,
    seed: u64,
) {
    let (p3_rows, gpu_flat) = random_matrix(num_leaves, row_width, seed);

    // CPU reference.
    let cpu_digests: Vec<[P3BabyBear; DIGEST_LEN]> = p3_rows
        .iter()
        .map(|row| sponge.hash_iter(row.iter().copied()))
        .collect();

    // GPU path (host-fed convenience wrapper).
    let gpu_digests = plan
        .hash_host_matrix(device, &gpu_flat, num_leaves as u32, row_width as u32)
        .unwrap();
    assert_eq!(gpu_digests.len(), num_leaves * DIGEST_LEN);

    for (leaf, cpu_digest) in cpu_digests.iter().enumerate() {
        for i in 0..DIGEST_LEN {
            assert_eq!(
                cpu_digest[i].as_canonical_u32(),
                gpu_digests[leaf * DIGEST_LEN + i].0,
                "shape (h={num_leaves}, w={row_width}) digest mismatch \
                 at leaf {leaf} slot {i}"
            );
        }
    }
}

// -- Absorption-branch coverage --------------------------------------------

#[test]
fn leaf_sponge_row_width_zero_is_zero_digest() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (sponge, mut plan) = build_matched(&device, 0x_C0FF_EE_00_u64);

    // `hash_iter` on an empty iterator returns [0; OUT] (no permute).
    // GPU path must produce the same all-zero digest for every row.
    let num_leaves = 5usize;
    let row_width = 0usize;

    // CPU reference: all zeros.
    let cpu_expected = [P3BabyBear::from_u64(0); DIGEST_LEN];
    for _ in 0..num_leaves {
        let out: [P3BabyBear; DIGEST_LEN] = sponge.hash_iter(core::iter::empty());
        assert_eq!(out, cpu_expected, "CPU empty-input digest not all-zero");
    }

    // GPU path.
    let empty: Vec<ZkgpuBabyBear> = Vec::new();
    let gpu_digests = plan
        .hash_host_matrix(&device, &empty, num_leaves as u32, row_width as u32)
        .unwrap();
    assert_eq!(gpu_digests.len(), num_leaves * DIGEST_LEN);
    for v in gpu_digests {
        assert_eq!(v.0, 0, "GPU empty-input digest slot not zero");
    }
}

#[test]
fn leaf_sponge_partial_single_chunk() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (sponge, mut plan) = build_matched(&device, 0x_C0FF_EE_01_u64);

    // Short rows: every width in `[1, RATE-1]` exercises the "remainder
    // only, no full chunk" branch.
    for w in 1..RATE {
        run_shape(&device, &sponge, &mut plan, 3, w, 0x_A110_0000_u64 + w as u64);
    }
}

#[test]
fn leaf_sponge_exact_one_full_chunk() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (sponge, mut plan) = build_matched(&device, 0x_C0FF_EE_02_u64);
    // w == RATE (one permute, no partial).
    run_shape(&device, &sponge, &mut plan, 7, RATE, 0xD00D_FEED_u64);
}

#[test]
fn leaf_sponge_one_plus_partial() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (sponge, mut plan) = build_matched(&device, 0x_C0FF_EE_03_u64);
    // RATE + r for r in 1..RATE — one full chunk then partial.
    for r in 1..RATE {
        run_shape(
            &device,
            &sponge,
            &mut plan,
            4,
            RATE + r,
            0xBEEF_0000_u64 + r as u64,
        );
    }
}

#[test]
fn leaf_sponge_exact_two_full_chunks() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (sponge, mut plan) = build_matched(&device, 0x_C0FF_EE_04_u64);
    run_shape(&device, &sponge, &mut plan, 6, 2 * RATE, 0xCAFE_BABE_u64);
}

// -- Batch-size coverage ---------------------------------------------------

#[test]
fn leaf_sponge_prime_batch_w8() {
    // 17 leaves (prime) × w=8 — catches 2D-dispatch-fold off-by-ones.
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (sponge, mut plan) = build_matched(&device, 0x_C0FF_EE_05_u64);
    run_shape(&device, &sponge, &mut plan, 17, 8, 0xFEED_C0DE_u64);
}

#[test]
fn leaf_sponge_target_stack_shape_log_h18_w8() {
    // The go/no-go gate shape: 2^18 leaves × w=8.
    // This is semantic-parity only (the bench gate is separate);
    // smaller than the perf shape to keep the test fast.
    //
    // w=8 is < RATE so each row = single partial-chunk permute.
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (sponge, mut plan) = build_matched(&device, 0x_C0FF_EE_06_u64);
    // Use a smaller h here than 2^18 — the semantic mapping is
    // identical and this keeps CI-time bounded on non-discrete GPUs.
    run_shape(&device, &sponge, &mut plan, 1024, 8, 0x_1111_2222_u64);
}

#[test]
fn leaf_sponge_multi_full_plus_partial() {
    // w = 3*RATE + 5 — probes the loop bound for multiple full chunks.
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (sponge, mut plan) = build_matched(&device, 0x_C0FF_EE_07_u64);
    run_shape(&device, &sponge, &mut plan, 11, 3 * RATE + 5, 0xBAD_D00D_u64);
}

// -- Guards ----------------------------------------------------------------

#[test]
fn leaf_sponge_rejects_horizonlabs_variant() {
    use zkgpu_poseidon2::{M4Variant, Poseidon2Params};

    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut rng = SmallRng::seed_from_u64(0x_BADC_0DE_u64);
    let ext: ExternalLayerConstants<P3BabyBear, 24> =
        ExternalLayerConstants::new_from_rng(ROUNDS_F, &mut rng);
    let int: Vec<P3BabyBear> =
        (&mut rng).sample_iter(StandardUniform).take(21).collect();
    // Start from the Plonky3 bridge, then flip the variant — tests the
    // guard, not params construction.
    let mut params: Poseidon2Params<ZkgpuBabyBear, WIDTH> =
        babybear_plonky3_params::<24>(&ext, &int);
    params.m4_variant = M4Variant::HorizonLabs;

    let err = WgpuPoseidon2MerkleLeafPlan::new(&device, params);
    assert!(
        err.is_err(),
        "plan must reject M4Variant::HorizonLabs, got Ok"
    );
}

#[test]
fn leaf_sponge_empty_batch_is_noop() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (_sponge, mut plan) = build_matched(&device, 0x_C0FF_EE_08_u64);
    let empty: Vec<ZkgpuBabyBear> = Vec::new();
    let out = plan.hash_host_matrix(&device, &empty, 0, 8).unwrap();
    assert!(out.is_empty());
}

