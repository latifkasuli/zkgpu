//! Phase 7 Step 1.5a — differential tests: `zkgpu-poseidon2` output
//! equals `p3_baby_bear::Poseidon2BabyBear` output, bit-for-bit, at
//! widths 16 and 24.
//!
//! If these pass, `zkgpu-poseidon2` (configured via the bridge in
//! `zkgpu_plonky3::poseidon2_bridge`) is a drop-in replacement for
//! Plonky3's CPU Poseidon2 permutation. Step 1.5b (GPU width-24
//! permutation plan) and Step 3 (GPU Poseidon2 Merkle commit) both
//! depend on this bit-identical behaviour to stay Plonky3-compatible.

use p3_baby_bear::{BabyBear as P3BabyBear, Poseidon2BabyBear};
use p3_field::PrimeField32;
use p3_poseidon2::ExternalLayerConstants;
use p3_symmetric::Permutation;
use rand::distr::StandardUniform;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use zkgpu_babybear::BabyBear as ZkgpuBabyBear;
use zkgpu_core::GpuField;
use zkgpu_plonky3::poseidon2_bridge::{
    babybear_plonky3_params, p3_to_zkgpu, zkgpu_to_p3,
};
use zkgpu_poseidon2::Poseidon2;

const ROUNDS_F: usize = 8;

// Plonky3's `Poseidon2BabyBear<W>` type only has impls for concrete
// widths (16, 24, 32), so we can't easily share a generic build/assert
// helper. Each width gets its own pair of tiny helpers below; the
// body is structurally identical.

// -- Width 16 ----------------------------------------------------------------

fn build_pair_16(seed: u64) -> (Poseidon2BabyBear<16>, Poseidon2<ZkgpuBabyBear, 16>) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let ext: ExternalLayerConstants<P3BabyBear, 16> =
        ExternalLayerConstants::new_from_rng(ROUNDS_F, &mut rng);
    let int: Vec<P3BabyBear> =
        (&mut rng).sample_iter(StandardUniform).take(13).collect();

    let p3_perm: Poseidon2BabyBear<16> = Poseidon2BabyBear::new(ext.clone(), int.clone());
    let zkgpu_params = babybear_plonky3_params::<16>(&ext, &int);
    let zkgpu_perm = Poseidon2::new(zkgpu_params);
    (p3_perm, zkgpu_perm)
}

fn assert_output_matches_16(
    p3_perm: &Poseidon2BabyBear<16>,
    zkgpu_perm: &Poseidon2<ZkgpuBabyBear, 16>,
    p3_state: [P3BabyBear; 16],
) {
    let mut p3_out = p3_state;
    p3_perm.permute_mut(&mut p3_out);

    let mut zkgpu_state = [ZkgpuBabyBear::from_u64(0); 16];
    for i in 0..16 {
        zkgpu_state[i] = p3_to_zkgpu(p3_state[i]);
    }
    zkgpu_perm.permute(&mut zkgpu_state);

    for i in 0..16 {
        assert_eq!(
            p3_out[i].as_canonical_u32(),
            zkgpu_state[i].0,
            "W=16 mismatch at slot {i}: p3={} zkgpu={}",
            p3_out[i].as_canonical_u32(),
            zkgpu_state[i].0,
        );
    }
}

#[test]
fn babybear_w16_zero_state() {
    let (p3, zk) = build_pair_16(0xCAFE_BABE);
    assert_output_matches_16(&p3, &zk, [P3BabyBear::new(0); 16]);
}

#[test]
fn babybear_w16_unit_state() {
    let (p3, zk) = build_pair_16(0xCAFE_BABE);
    let mut s = [P3BabyBear::new(0); 16];
    s[0] = P3BabyBear::new(1);
    assert_output_matches_16(&p3, &zk, s);
}

#[test]
fn babybear_w16_sequential_state() {
    let (p3, zk) = build_pair_16(0xCAFE_BABE);
    let mut s = [P3BabyBear::new(0); 16];
    for i in 0..16u32 {
        s[i as usize] = P3BabyBear::new(i.wrapping_mul(0x9E37_79B9));
    }
    assert_output_matches_16(&p3, &zk, s);
}

#[test]
fn babybear_w16_random_states() {
    let (p3, zk) = build_pair_16(0xC01D_DA7A);
    let mut rng = SmallRng::seed_from_u64(0xFEED_C0DE);
    for _ in 0..16 {
        let state: [P3BabyBear; 16] = rng.random();
        assert_output_matches_16(&p3, &zk, state);
    }
}

#[test]
fn babybear_w16_multiple_seeds() {
    for seed in [1u64, 42, 0xDEAD_BEEF, 0xA5A5_A5A5_A5A5_A5A5] {
        let (p3, zk) = build_pair_16(seed);
        let mut s = [P3BabyBear::new(0); 16];
        s[0] = P3BabyBear::new(1);
        assert_output_matches_16(&p3, &zk, s);
    }
}

// -- Width 24 ----------------------------------------------------------------

fn build_pair_24(seed: u64) -> (Poseidon2BabyBear<24>, Poseidon2<ZkgpuBabyBear, 24>) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let ext: ExternalLayerConstants<P3BabyBear, 24> =
        ExternalLayerConstants::new_from_rng(ROUNDS_F, &mut rng);
    let int: Vec<P3BabyBear> =
        (&mut rng).sample_iter(StandardUniform).take(21).collect();

    let p3_perm: Poseidon2BabyBear<24> = Poseidon2BabyBear::new(ext.clone(), int.clone());
    let zkgpu_params = babybear_plonky3_params::<24>(&ext, &int);
    let zkgpu_perm = Poseidon2::new(zkgpu_params);
    (p3_perm, zkgpu_perm)
}

fn assert_output_matches_24(
    p3_perm: &Poseidon2BabyBear<24>,
    zkgpu_perm: &Poseidon2<ZkgpuBabyBear, 24>,
    p3_state: [P3BabyBear; 24],
) {
    let mut p3_out = p3_state;
    p3_perm.permute_mut(&mut p3_out);

    let mut zkgpu_state = [ZkgpuBabyBear::from_u64(0); 24];
    for i in 0..24 {
        zkgpu_state[i] = p3_to_zkgpu(p3_state[i]);
    }
    zkgpu_perm.permute(&mut zkgpu_state);

    for i in 0..24 {
        assert_eq!(
            p3_out[i].as_canonical_u32(),
            zkgpu_state[i].0,
            "W=24 mismatch at slot {i}: p3={} zkgpu={}",
            p3_out[i].as_canonical_u32(),
            zkgpu_state[i].0,
        );
    }
}

#[test]
fn babybear_w24_zero_state() {
    let (p3, zk) = build_pair_24(0xCAFE_BABE);
    assert_output_matches_24(&p3, &zk, [P3BabyBear::new(0); 24]);
}

#[test]
fn babybear_w24_unit_state() {
    let (p3, zk) = build_pair_24(0xCAFE_BABE);
    let mut s = [P3BabyBear::new(0); 24];
    s[0] = P3BabyBear::new(1);
    assert_output_matches_24(&p3, &zk, s);
}

#[test]
fn babybear_w24_random_states() {
    let (p3, zk) = build_pair_24(0xC01D_DA7A);
    let mut rng = SmallRng::seed_from_u64(0xFEED_C0DE);
    for _ in 0..16 {
        let state: [P3BabyBear; 24] = rng.random();
        assert_output_matches_24(&p3, &zk, state);
    }
}

// -- Smoke: field conversion is its own inverse -----------------------------

#[test]
fn field_conversion_roundtrip() {
    for raw in [0u32, 1, 42, 0x0000_0FFF, 0x7800_0000] {
        let p3 = P3BabyBear::new(raw);
        let zk = p3_to_zkgpu(p3);
        let back = zkgpu_to_p3(zk);
        assert_eq!(
            p3.as_canonical_u32(),
            back.as_canonical_u32(),
            "conversion roundtrip failed for {raw}"
        );
    }
}
