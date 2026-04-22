//! Phase 7 Step 1.5b — GPU Poseidon2 (widths 16 and 24, Plonky3
//! variant) must produce bit-identical output to
//! `p3_baby_bear::Poseidon2BabyBear<W>` for arbitrary inputs.
//!
//! This is the Step 1.5b gate. If these pass, both GPU Plonky3
//! Poseidon2 plans are drop-in-compatible with Plonky3's CPU
//! permutation. Step 3 (GPU Poseidon2 Merkle commit) builds on both
//! widths: width-16 for the compression path, width-24 for the leaf
//! sponge.
//!
//! Skips the test body when no GPU adapter is available on the host
//! (matches the convention used by the existing Phase F.1 tests in
//! `zkgpu-wgpu/src/poseidon2/plan.rs`).

use p3_baby_bear::{BabyBear as P3BabyBear, Poseidon2BabyBear};
use p3_field::PrimeField32;
use p3_poseidon2::ExternalLayerConstants;
use p3_symmetric::Permutation;
use rand::distr::StandardUniform;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use zkgpu_babybear::BabyBear as ZkgpuBabyBear;
use zkgpu_core::GpuField;
use zkgpu_plonky3::poseidon2_bridge::{babybear_plonky3_params, p3_to_zkgpu};
use zkgpu_wgpu::{
    WgpuBabyBearPoseidon2PlonkyW16Plan, WgpuBabyBearPoseidon2PlonkyW24Plan, WgpuDevice,
};
use zkgpu_core::{GpuBuffer, GpuDevice};

const ROUNDS_F: usize = 8;

fn try_device() -> Option<WgpuDevice> {
    WgpuDevice::new().ok()
}

// -- Width 16 ---------------------------------------------------------------

#[test]
fn gpu_w16_matches_plonky3_single() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    // Build a matched (p3, zkgpu) params pair.
    let mut rng = SmallRng::seed_from_u64(0xC0DE_BABE);
    let ext: ExternalLayerConstants<P3BabyBear, 16> =
        ExternalLayerConstants::new_from_rng(ROUNDS_F, &mut rng);
    let int: Vec<P3BabyBear> =
        (&mut rng).sample_iter(StandardUniform).take(13).collect();

    let p3_perm: Poseidon2BabyBear<16> =
        Poseidon2BabyBear::new(ext.clone(), int.clone());
    let zkgpu_params = babybear_plonky3_params::<16>(&ext, &int);
    let mut gpu_plan =
        WgpuBabyBearPoseidon2PlonkyW16Plan::new(&device, zkgpu_params).unwrap();

    // Fixed non-trivial input.
    let mut p3_state = [P3BabyBear::new(0); 16];
    p3_state[0] = P3BabyBear::new(1);
    let p3_in = p3_state;
    p3_perm.permute_mut(&mut p3_state);

    // GPU path: upload in canonical form, permute, read back.
    let mut gpu_state: Vec<ZkgpuBabyBear> =
        p3_in.iter().map(|e| p3_to_zkgpu(*e)).collect();
    let mut buf = device.upload::<ZkgpuBabyBear>(&gpu_state).unwrap();
    gpu_plan.execute(&device, &mut buf).unwrap();
    gpu_state = buf.read_to_vec().unwrap();

    for i in 0..16 {
        assert_eq!(
            p3_state[i].as_canonical_u32(),
            gpu_state[i].0,
            "W=16 GPU vs Plonky3 mismatch at slot {i}"
        );
    }
}

#[test]
fn gpu_w16_matches_plonky3_batch_17() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let mut rng = SmallRng::seed_from_u64(0xC01D_DA7A);
    let ext: ExternalLayerConstants<P3BabyBear, 16> =
        ExternalLayerConstants::new_from_rng(ROUNDS_F, &mut rng);
    let int: Vec<P3BabyBear> =
        (&mut rng).sample_iter(StandardUniform).take(13).collect();

    let p3_perm: Poseidon2BabyBear<16> =
        Poseidon2BabyBear::new(ext.clone(), int.clone());
    let zkgpu_params = babybear_plonky3_params::<16>(&ext, &int);
    let mut gpu_plan =
        WgpuBabyBearPoseidon2PlonkyW16Plan::new(&device, zkgpu_params).unwrap();

    // 17 permutations — prime, catches any off-by-one in 2D dispatch fold.
    let num = 17usize;
    let mut state_rng = SmallRng::seed_from_u64(0xFEED_C0DE);

    let mut p3_states: Vec<[P3BabyBear; 16]> = Vec::with_capacity(num);
    let mut gpu_flat: Vec<ZkgpuBabyBear> = Vec::with_capacity(num * 16);
    for _ in 0..num {
        let s: [P3BabyBear; 16] = state_rng.random();
        gpu_flat.extend(s.iter().map(|e| p3_to_zkgpu(*e)));
        p3_states.push(s);
    }

    // CPU reference.
    for s in p3_states.iter_mut() {
        p3_perm.permute_mut(s);
    }

    // GPU batch.
    let mut buf = device.upload::<ZkgpuBabyBear>(&gpu_flat).unwrap();
    gpu_plan.execute(&device, &mut buf).unwrap();
    let gpu_out = buf.read_to_vec().unwrap();

    for (p, p3_state) in p3_states.iter().enumerate() {
        for i in 0..16 {
            assert_eq!(
                p3_state[i].as_canonical_u32(),
                gpu_out[p * 16 + i].0,
                "W=16 batch mismatch at perm {p} slot {i}",
            );
        }
    }
}

// -- Width 24 ---------------------------------------------------------------

#[test]
fn gpu_w24_matches_plonky3_single() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let mut rng = SmallRng::seed_from_u64(0xDEAD_BEEF);
    let ext: ExternalLayerConstants<P3BabyBear, 24> =
        ExternalLayerConstants::new_from_rng(ROUNDS_F, &mut rng);
    let int: Vec<P3BabyBear> =
        (&mut rng).sample_iter(StandardUniform).take(21).collect();

    let p3_perm: Poseidon2BabyBear<24> =
        Poseidon2BabyBear::new(ext.clone(), int.clone());
    let zkgpu_params = babybear_plonky3_params::<24>(&ext, &int);
    let mut gpu_plan =
        WgpuBabyBearPoseidon2PlonkyW24Plan::new(&device, zkgpu_params).unwrap();

    let mut p3_state = [P3BabyBear::new(0); 24];
    p3_state[0] = P3BabyBear::new(1);
    let p3_in = p3_state;
    p3_perm.permute_mut(&mut p3_state);

    let gpu_in: Vec<ZkgpuBabyBear> =
        p3_in.iter().map(|e| p3_to_zkgpu(*e)).collect();
    let mut buf = device.upload::<ZkgpuBabyBear>(&gpu_in).unwrap();
    gpu_plan.execute(&device, &mut buf).unwrap();
    let gpu_out = buf.read_to_vec().unwrap();

    for i in 0..24 {
        assert_eq!(
            p3_state[i].as_canonical_u32(),
            gpu_out[i].0,
            "W=24 GPU vs Plonky3 mismatch at slot {i}"
        );
    }
}

#[test]
fn gpu_w24_matches_plonky3_batch_17() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let mut rng = SmallRng::seed_from_u64(0xA5A5_A5A5);
    let ext: ExternalLayerConstants<P3BabyBear, 24> =
        ExternalLayerConstants::new_from_rng(ROUNDS_F, &mut rng);
    let int: Vec<P3BabyBear> =
        (&mut rng).sample_iter(StandardUniform).take(21).collect();

    let p3_perm: Poseidon2BabyBear<24> =
        Poseidon2BabyBear::new(ext.clone(), int.clone());
    let zkgpu_params = babybear_plonky3_params::<24>(&ext, &int);
    let mut gpu_plan =
        WgpuBabyBearPoseidon2PlonkyW24Plan::new(&device, zkgpu_params).unwrap();

    let num = 17usize;
    let mut state_rng = SmallRng::seed_from_u64(0xB00B_CAFE);

    let mut p3_states: Vec<[P3BabyBear; 24]> = Vec::with_capacity(num);
    let mut gpu_flat: Vec<ZkgpuBabyBear> = Vec::with_capacity(num * 24);
    for _ in 0..num {
        let s: [P3BabyBear; 24] = state_rng.random();
        gpu_flat.extend(s.iter().map(|e| p3_to_zkgpu(*e)));
        p3_states.push(s);
    }

    for s in p3_states.iter_mut() {
        p3_perm.permute_mut(s);
    }

    let mut buf = device.upload::<ZkgpuBabyBear>(&gpu_flat).unwrap();
    gpu_plan.execute(&device, &mut buf).unwrap();
    let gpu_out = buf.read_to_vec().unwrap();

    for (p, p3_state) in p3_states.iter().enumerate() {
        for i in 0..24 {
            assert_eq!(
                p3_state[i].as_canonical_u32(),
                gpu_out[p * 24 + i].0,
                "W=24 batch mismatch at perm {p} slot {i}",
            );
        }
    }
}

// -- Empty batch + mis-size guards ------------------------------------------

#[test]
fn gpu_w24_empty_batch_is_noop() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut rng = SmallRng::seed_from_u64(1);
    let ext: ExternalLayerConstants<P3BabyBear, 24> =
        ExternalLayerConstants::new_from_rng(ROUNDS_F, &mut rng);
    let int: Vec<P3BabyBear> =
        (&mut rng).sample_iter(StandardUniform).take(21).collect();
    let params = babybear_plonky3_params::<24>(&ext, &int);
    let empty: Vec<ZkgpuBabyBear> = Vec::new();
    let mut buf = device.upload::<ZkgpuBabyBear>(&empty).unwrap();
    let mut plan =
        WgpuBabyBearPoseidon2PlonkyW24Plan::new(&device, params).unwrap();
    plan.execute(&device, &mut buf).unwrap();
}

#[test]
fn gpu_w16_rejects_missized_buffer() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut rng = SmallRng::seed_from_u64(1);
    let ext: ExternalLayerConstants<P3BabyBear, 16> =
        ExternalLayerConstants::new_from_rng(ROUNDS_F, &mut rng);
    let int: Vec<P3BabyBear> =
        (&mut rng).sample_iter(StandardUniform).take(13).collect();
    let params = babybear_plonky3_params::<16>(&ext, &int);
    // 17 is not a multiple of 16.
    let bogus = vec![ZkgpuBabyBear::from_u64(0); 17];
    let mut buf = device.upload::<ZkgpuBabyBear>(&bogus).unwrap();
    let mut plan =
        WgpuBabyBearPoseidon2PlonkyW16Plan::new(&device, params).unwrap();
    assert!(plan.execute(&device, &mut buf).is_err());
}
