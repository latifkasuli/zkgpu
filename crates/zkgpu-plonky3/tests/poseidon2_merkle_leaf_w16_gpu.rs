//! Phase 3.d Stage 1a — GPU W16/RATE=8 leaf sponge differential test.
//!
//! Target: [`zkgpu_wgpu::WgpuPoseidon2MerkleLeafW16R8Plan`] must match
//! `p3_symmetric::PaddingFreeSponge<Poseidon2BabyBear<16>, 16, 8, 8>::hash_iter`
//! row-by-row for arbitrary `(num_leaves, row_width)` shapes.
//!
//! This is the leaf sponge shape used by OpenVM's BabyBear Poseidon2
//! MMCS config (`stark-backend/crates/stark-sdk/src/config/baby_bear_poseidon2.rs`).
//! A companion to the existing W24/RATE=16 parity suite in
//! `poseidon2_merkle_leaf_gpu.rs`; both must pass before the mixed-
//! height backend can build commits with either shape.
//!
//! Coverage mirrors the W24 suite:
//! * `row_width == 0`          — no permutation, all-zero digest
//! * `row_width < RATE=8`      — single partial-chunk permute
//! * `row_width == k*8`        — k full-chunk permutes
//! * `row_width == k*8 + r`    — k full + 1 partial permute
//! * prime-batch 2D-dispatch fold check
//! * log_h=10 target-stack proxy
//! * host-wrapper shape-validation guards

use p3_baby_bear::{BabyBear as P3BabyBear, Poseidon2BabyBear};
use p3_field::{PrimeCharacteristicRing, PrimeField32};
use p3_poseidon2::ExternalLayerConstants;
use p3_symmetric::{CryptographicHasher, PaddingFreeSponge};
use rand::distr::StandardUniform;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use zkgpu_babybear::BabyBear as ZkgpuBabyBear;
use zkgpu_plonky3::poseidon2_bridge::{babybear_plonky3_params, p3_to_zkgpu};
use zkgpu_wgpu::{WgpuDevice, WgpuPoseidon2MerkleLeafW16R8Plan};

const ROUNDS_F: usize = 8;
const WIDTH: usize = 16;
const RATE: usize = 8;
const DIGEST_LEN: usize = 8;

type Perm16 = Poseidon2BabyBear<16>;
type Poseidon2W16Sponge = PaddingFreeSponge<Perm16, 16, 8, 8>;

fn try_device() -> Option<WgpuDevice> {
    WgpuDevice::new().ok()
}

/// Build a matched (CPU `PaddingFreeSponge<Perm16, 16, 8, 8>`, GPU
/// W16/R8 leaf plan) pair driven by identical constants from the
/// Step 1.5a bridge.
fn build_matched(
    device: &WgpuDevice,
    seed: u64,
) -> (Poseidon2W16Sponge, WgpuPoseidon2MerkleLeafW16R8Plan) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let ext: ExternalLayerConstants<P3BabyBear, 16> =
        ExternalLayerConstants::new_from_rng(ROUNDS_F, &mut rng);
    let int: Vec<P3BabyBear> =
        (&mut rng).sample_iter(StandardUniform).take(13).collect();

    let p3_perm: Perm16 = Perm16::new(ext.clone(), int.clone());
    let sponge = Poseidon2W16Sponge::new(p3_perm);

    let zkgpu_params = babybear_plonky3_params::<16>(&ext, &int);
    let plan = WgpuPoseidon2MerkleLeafW16R8Plan::new(device, zkgpu_params).unwrap();
    (sponge, plan)
}

/// Generate a random `h × w` matrix as (CPU rows, GPU flat input).
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

fn run_shape(
    device: &WgpuDevice,
    sponge: &Poseidon2W16Sponge,
    plan: &mut WgpuPoseidon2MerkleLeafW16R8Plan,
    num_leaves: usize,
    row_width: usize,
    seed: u64,
) {
    let (p3_rows, gpu_flat) = random_matrix(num_leaves, row_width, seed);
    let cpu_digests: Vec<[P3BabyBear; DIGEST_LEN]> = p3_rows
        .iter()
        .map(|row| sponge.hash_iter(row.iter().copied()))
        .collect();

    let gpu_digests = plan
        .hash_host_matrix(device, &gpu_flat, num_leaves as u32, row_width as u32)
        .unwrap();
    assert_eq!(gpu_digests.len(), num_leaves * DIGEST_LEN);

    for (leaf, cpu_digest) in cpu_digests.iter().enumerate() {
        for i in 0..DIGEST_LEN {
            assert_eq!(
                cpu_digest[i].as_canonical_u32(),
                gpu_digests[leaf * DIGEST_LEN + i].0,
                "W16R8 shape (h={num_leaves}, w={row_width}) mismatch \
                 at leaf {leaf} slot {i}"
            );
        }
    }
}

// -- Absorption-branch coverage --------------------------------------------

#[test]
fn w16r8_leaf_row_width_zero_is_zero_digest() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (sponge, mut plan) = build_matched(&device, 0x_E16_0000_u64);

    // `hash_iter` on empty input returns [0; OUT].
    let expected_zero = [P3BabyBear::from_u64(0); DIGEST_LEN];
    let cpu: [P3BabyBear; DIGEST_LEN] = sponge.hash_iter(core::iter::empty());
    assert_eq!(cpu, expected_zero);

    let num_leaves = 5usize;
    let empty: Vec<ZkgpuBabyBear> = Vec::new();
    let gpu = plan
        .hash_host_matrix(&device, &empty, num_leaves as u32, 0)
        .unwrap();
    assert_eq!(gpu.len(), num_leaves * DIGEST_LEN);
    for v in gpu {
        assert_eq!(v.0, 0, "W16R8 row_width=0 digest slot must be zero");
    }
}

#[test]
fn w16r8_leaf_partial_single_chunk() {
    // Widths 1..RATE=8 → "remainder only, no full chunk" branch.
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (sponge, mut plan) = build_matched(&device, 0x_E16_0001_u64);
    for w in 1..RATE {
        run_shape(&device, &sponge, &mut plan, 3, w, 0x_0B_0100_u64 + w as u64);
    }
}

#[test]
fn w16r8_leaf_exact_one_full_chunk() {
    // w == RATE=8 → one permute, no partial.
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (sponge, mut plan) = build_matched(&device, 0x_E16_0002_u64);
    run_shape(&device, &sponge, &mut plan, 7, RATE, 0x_D00D_0008_u64);
}

#[test]
fn w16r8_leaf_one_plus_partial() {
    // RATE + r, r in 1..RATE → one full chunk then partial.
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (sponge, mut plan) = build_matched(&device, 0x_E16_0003_u64);
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
fn w16r8_leaf_exact_two_full_chunks() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (sponge, mut plan) = build_matched(&device, 0x_E16_0004_u64);
    run_shape(&device, &sponge, &mut plan, 6, 2 * RATE, 0x_CAFE_BABE_u64);
}

#[test]
fn w16r8_leaf_multi_full_plus_partial() {
    // w = 3*RATE + 5 — multiple full chunks + partial.
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (sponge, mut plan) = build_matched(&device, 0x_E16_0005_u64);
    run_shape(&device, &sponge, &mut plan, 11, 3 * RATE + 5, 0x_BAD_D00D_u64);
}

// -- Batch coverage --------------------------------------------------------

#[test]
fn w16r8_leaf_prime_batch_w8() {
    // 17 leaves (prime) × w=8 — 2D-dispatch-fold off-by-one canary.
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (sponge, mut plan) = build_matched(&device, 0x_E16_0006_u64);
    run_shape(&device, &sponge, &mut plan, 17, 8, 0x_FEED_C0DE_u64);
}

#[test]
fn w16r8_leaf_log10_w8() {
    // 1024 × 8 — target-stack proxy (smaller than the 2^18 bench
    // shape to keep CI-time bounded).
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (sponge, mut plan) = build_matched(&device, 0x_E16_0007_u64);
    run_shape(&device, &sponge, &mut plan, 1024, 8, 0x_1111_2222_u64);
}

// -- Guards ----------------------------------------------------------------

#[test]
fn w16r8_leaf_rejects_horizonlabs_variant() {
    use zkgpu_poseidon2::{M4Variant, Poseidon2Params};

    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut rng = SmallRng::seed_from_u64(0x_BAD_C0DE_u64);
    let ext: ExternalLayerConstants<P3BabyBear, 16> =
        ExternalLayerConstants::new_from_rng(ROUNDS_F, &mut rng);
    let int: Vec<P3BabyBear> =
        (&mut rng).sample_iter(StandardUniform).take(13).collect();
    let mut params: Poseidon2Params<ZkgpuBabyBear, WIDTH> =
        babybear_plonky3_params::<16>(&ext, &int);
    params.m4_variant = M4Variant::HorizonLabs;

    let err = WgpuPoseidon2MerkleLeafW16R8Plan::new(&device, params);
    assert!(err.is_err(), "W16R8 plan must reject HorizonLabs variant");
}

#[test]
fn w16r8_leaf_empty_batch_is_noop() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (_sponge, mut plan) = build_matched(&device, 0x_E16_0008_u64);
    let out = plan
        .hash_host_matrix(&device, &[], 0, 8)
        .unwrap();
    assert!(out.is_empty());
}

#[test]
fn w16r8_host_wrapper_rejects_bogus_num_leaves_zero() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (_sponge, mut plan) = build_matched(&device, 0x_E16_0009_u64);
    let bogus = vec![ZkgpuBabyBear(1); 8];
    let err = plan.hash_host_matrix(&device, &bogus, 0, 8);
    assert!(err.is_err(), "must reject num_leaves=0 with non-empty matrix");
}

#[test]
fn w16r8_host_wrapper_rejects_bogus_row_width_zero() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (_sponge, mut plan) = build_matched(&device, 0x_E16_000A_u64);
    let bogus = vec![ZkgpuBabyBear(1); 8];
    let err = plan.hash_host_matrix(&device, &bogus, 4, 0);
    assert!(err.is_err(), "must reject row_width=0 with non-empty matrix");
}
