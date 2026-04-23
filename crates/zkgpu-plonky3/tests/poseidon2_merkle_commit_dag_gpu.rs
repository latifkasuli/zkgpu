//! Phase 3.d Stage 1b — mixed-height Merkle DAG engine parity.
//!
//! Locks the new
//! [`zkgpu_wgpu::commit_mixed_height_host_matrices_with_retained_layers`]
//! backend against Plonky3's CPU `MerkleTreeMmcs::commit` for the
//! N=2 binary / `cap_height=0` / power-of-two-height case — the
//! superset of shapes the shared backend needs to support so that
//! both the Plonky3 canonical config (W24 leaf + W16 compress) and
//! OpenVM's config (W16 leaf + W16 compress) drive the same commit
//! DAG.
//!
//! This test uses the Plonky3 canonical config for the oracle side
//! because that's the one the in-tree adapter is already pinned
//! against via Step 1.5a/1.5b/3. The OpenVM-shaped leaf variant is
//! covered by its own parity suite
//! (`poseidon2_merkle_leaf_w16_gpu.rs`) at the leaf level; once the
//! Stage 2 `zkgpu-openvm` adapter lands, it will add an
//! OpenVM-config parity suite on top of this same backend.
//!
//! Coverage:
//! * single-matrix power-of-two-h (regression for the same-height
//!   path through the DAG engine)
//! * same-height multi-matrix at a single power-of-two (matches the
//!   quotient-chunk shape the Step 3.c adapter supports today)
//! * true mixed-height batches: heights
//!     `{4}`, `{8, 4}`, `{8, 4, 2}`, `{16, 8, 4, 2}`, `{64, 32, 8}`,
//!   with varying per-matrix widths per group
//! * guards: empty input, zero-width, non-power-of-two height,
//!   shape-length mismatch all surface as `InvalidNttSize`

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
    commit_mixed_height_host_matrices_with_retained_layers, root_from_retained,
    MixedHeightMatrixInput, WgpuDevice, WgpuPoseidon2MerkleCompressPlan,
    WgpuPoseidon2MerkleLeafPlan,
};

const ROUNDS_F: usize = 8;
const DIGEST_LEN: usize = 8;

type Val = P3BabyBear;
type Perm16 = Poseidon2BabyBear<16>;
type Perm24 = Poseidon2BabyBear<24>;
type Poseidon2Sponge = PaddingFreeSponge<Perm24, 24, 16, DIGEST_LEN>;
type Poseidon2Compression = TruncatedPermutation<Perm16, 2, DIGEST_LEN, 16>;
type CpuValMmcs = MerkleTreeMmcs<
    <Val as p3_field::Field>::Packing,
    <Val as p3_field::Field>::Packing,
    Poseidon2Sponge,
    Poseidon2Compression,
    2,
    DIGEST_LEN,
>;

fn try_device() -> Option<Arc<WgpuDevice>> {
    WgpuDevice::new().ok().map(Arc::new)
}

/// Build matched (CPU MerkleTreeMmcs, GPU leaf plan, GPU compress plan)
/// with Plonky3's canonical W24+W16 Poseidon2 config. W16 and W24
/// get distinct RNG draws so constants differ, matching the Plonky3
/// canonical pattern.
fn build_matched(
    device: Arc<WgpuDevice>,
    seed: u64,
) -> (CpuValMmcs, WgpuPoseidon2MerkleLeafPlan, WgpuPoseidon2MerkleCompressPlan) {
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

    let sponge = PaddingFreeSponge::new(perm24);
    let compression = TruncatedPermutation::new(perm16);
    let cpu_mmcs = CpuValMmcs::new(sponge, compression, 0);

    let leaf = WgpuPoseidon2MerkleLeafPlan::new(device.as_ref(), zkgpu_params24).unwrap();
    let compress =
        WgpuPoseidon2MerkleCompressPlan::new(device.as_ref(), zkgpu_params16).unwrap();

    (cpu_mmcs, leaf, compress)
}

/// Generate `n` random rows of `w` P3 BabyBear elements.
fn random_matrix(h: usize, w: usize, seed: u64) -> (RowMajorMatrix<P3BabyBear>, Vec<ZkgpuBabyBear>) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let p3_values: Vec<P3BabyBear> =
        (&mut rng).sample_iter(StandardUniform).take(h * w).collect();
    let gpu_flat: Vec<ZkgpuBabyBear> =
        p3_values.iter().map(|e| p3_to_zkgpu(*e)).collect();
    (RowMajorMatrix::new(p3_values, w), gpu_flat)
}

/// Compare a CPU cap (from `MerkleTreeMmcs::commit`) against the
/// GPU retained-layers root element-by-element.
fn assert_root_matches(
    cpu_cap: &p3_symmetric::MerkleCap<P3BabyBear, [P3BabyBear; DIGEST_LEN]>,
    gpu_root: [ZkgpuBabyBear; DIGEST_LEN],
    ctx: &str,
) {
    let cpu_root: [P3BabyBear; DIGEST_LEN] =
        cpu_cap.as_ref()[0];
    for i in 0..DIGEST_LEN {
        assert_eq!(
            cpu_root[i].as_canonical_u32(),
            gpu_root[i].0,
            "{ctx}: root slot {i}"
        );
    }
}

/// Drive both CPU and GPU commit paths with `shapes = [(h, w), ...]`
/// and assert roots match. GPU path goes through the mixed-height
/// DAG engine exclusively.
fn run_mixed_shapes(
    device: &WgpuDevice,
    cpu: &CpuValMmcs,
    leaf: &mut WgpuPoseidon2MerkleLeafPlan,
    compress: &mut WgpuPoseidon2MerkleCompressPlan,
    shapes: &[(usize, usize)],
    seed_base: u64,
) {
    let ctx = format!("shapes={shapes:?}");

    // Generate matching matrix pairs.
    let mut cpu_matrices: Vec<RowMajorMatrix<P3BabyBear>> = Vec::with_capacity(shapes.len());
    let mut gpu_inputs_storage: Vec<(Vec<ZkgpuBabyBear>, u32, u32)> =
        Vec::with_capacity(shapes.len());
    for (i, &(h, w)) in shapes.iter().enumerate() {
        let (cpu_m, gpu_flat) = random_matrix(h, w, seed_base ^ (i as u64) * 0xBADC0DE);
        cpu_matrices.push(cpu_m);
        gpu_inputs_storage.push((gpu_flat, h as u32, w as u32));
    }

    // CPU side: hand matrices to MerkleTreeMmcs::commit directly.
    let (cpu_cap, _cpu_pd) = cpu.commit(cpu_matrices);

    // GPU side: build MixedHeightMatrixInput refs over the stored flats.
    let gpu_inputs: Vec<MixedHeightMatrixInput<'_>> = gpu_inputs_storage
        .iter()
        .map(|(flat, h, w)| MixedHeightMatrixInput {
            flat: flat.as_slice(),
            height: *h,
            width: *w,
        })
        .collect();
    let retained = commit_mixed_height_host_matrices_with_retained_layers(
        device, leaf, compress, &gpu_inputs,
    )
    .unwrap();
    let gpu_root = root_from_retained(&retained).unwrap();

    assert_root_matches(&cpu_cap, gpu_root, &ctx);
}

// ==========================================================================
// Parity — same-height cases (DAG engine should match simpler paths)
// ==========================================================================

#[test]
fn dag_single_matrix_matches_plonky3() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, mut leaf, mut compress) = build_matched(device.clone(), 0x_D50_0001_u64);

    for &(h, w) in &[(1usize, 8usize), (2, 8), (4, 8), (16, 8), (64, 3), (1024, 8)] {
        run_mixed_shapes(&device, &cpu, &mut leaf, &mut compress, &[(h, w)], 0x_F00_0000_u64);
    }
}

#[test]
fn dag_same_height_multi_matrix_matches_plonky3() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, mut leaf, mut compress) = build_matched(device.clone(), 0x_D50_0002_u64);

    for shapes in &[
        vec![(16usize, 4usize), (16, 4)],
        vec![(8, 3), (8, 5), (8, 1)],
        vec![(64, 8); 4],
    ] {
        run_mixed_shapes(&device, &cpu, &mut leaf, &mut compress, shapes, 0x_F00_0010_u64);
    }
}

// ==========================================================================
// Parity — true mixed-height batches (the reason Stage 1b exists)
// ==========================================================================

#[test]
fn dag_mixed_height_two_levels_matches_plonky3() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, mut leaf, mut compress) = build_matched(device.clone(), 0x_D50_0003_u64);

    // Tallest + one injection level.
    for shapes in &[
        vec![(8usize, 4usize), (4, 2)],
        vec![(16, 8), (8, 4)],
        vec![(32, 3), (16, 7)],
    ] {
        run_mixed_shapes(&device, &cpu, &mut leaf, &mut compress, shapes, 0x_F00_0020_u64);
    }
}

#[test]
fn dag_mixed_height_three_levels_matches_plonky3() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, mut leaf, mut compress) = build_matched(device.clone(), 0x_D50_0004_u64);

    // Injection at two non-adjacent levels.
    for shapes in &[
        vec![(16usize, 4usize), (8, 3), (2, 1)],
        vec![(32, 6), (16, 2), (4, 5)],
        vec![(64, 8), (32, 4), (4, 2)],
    ] {
        run_mixed_shapes(&device, &cpu, &mut leaf, &mut compress, shapes, 0x_F00_0030_u64);
    }
}

#[test]
fn dag_mixed_height_every_level_matches_plonky3() {
    // Inject at every power-of-two height from h_max/2 down.
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, mut leaf, mut compress) = build_matched(device.clone(), 0x_D50_0005_u64);

    let shapes: Vec<(usize, usize)> =
        [16usize, 8, 4, 2, 1].iter().map(|&h| (h, 3usize)).collect();
    run_mixed_shapes(&device, &cpu, &mut leaf, &mut compress, &shapes, 0x_F00_0040_u64);
}

#[test]
fn dag_mixed_height_with_multiple_per_level_matches_plonky3() {
    // Multiple matrices at the SAME non-max height → still injects as
    // one group at that level (Plonky3 semantics: same-height
    // matrices at the tallest bucket concat horizontally; same rule
    // applies at non-tallest levels via compress_and_inject).
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, mut leaf, mut compress) = build_matched(device.clone(), 0x_D50_0006_u64);

    for shapes in &[
        vec![(32usize, 4usize), (16, 3), (16, 5)],           // two at h=16
        vec![(64, 2), (32, 8), (32, 4), (8, 1), (8, 6)],     // two at h=32 + two at h=8
    ] {
        run_mixed_shapes(&device, &cpu, &mut leaf, &mut compress, shapes, 0x_F00_0050_u64);
    }
}

#[test]
fn dag_bench_shape_proxy_matches_plonky3() {
    // Scaled-down proxy for a realistic target-stack commit:
    // a tall trace at h=1024 plus quotient-chunk-style injection at h=512.
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (cpu, mut leaf, mut compress) = build_matched(device.clone(), 0x_D50_0007_u64);

    let shapes = [(1024usize, 8usize), (512, 4)];
    run_mixed_shapes(&device, &cpu, &mut leaf, &mut compress, &shapes, 0x_F00_0060_u64);
}

// ==========================================================================
// Guards
// ==========================================================================

#[test]
fn dag_rejects_empty_input() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (_cpu, mut leaf, mut compress) = build_matched(device.clone(), 0x_D50_0100_u64);
    let result = commit_mixed_height_host_matrices_with_retained_layers(
        &device,
        &mut leaf,
        &mut compress,
        &[],
    );
    assert!(result.is_err(), "DAG must reject empty inputs");
}

#[test]
fn dag_rejects_non_power_of_two_height() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (_cpu, mut leaf, mut compress) = build_matched(device.clone(), 0x_D50_0101_u64);
    let flat: Vec<ZkgpuBabyBear> = vec![ZkgpuBabyBear(1); 3 * 4];
    let input = MixedHeightMatrixInput {
        flat: &flat,
        height: 3,
        width: 4,
    };
    let result = commit_mixed_height_host_matrices_with_retained_layers(
        &device,
        &mut leaf,
        &mut compress,
        &[input],
    );
    assert!(result.is_err(), "DAG must reject non-power-of-two heights");
}

#[test]
fn dag_rejects_zero_width() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (_cpu, mut leaf, mut compress) = build_matched(device.clone(), 0x_D50_0102_u64);
    let flat: Vec<ZkgpuBabyBear> = Vec::new();
    let input = MixedHeightMatrixInput {
        flat: &flat,
        height: 4,
        width: 0,
    };
    let result = commit_mixed_height_host_matrices_with_retained_layers(
        &device,
        &mut leaf,
        &mut compress,
        &[input],
    );
    assert!(result.is_err(), "DAG must reject width=0");
}

#[test]
fn dag_rejects_shape_mismatch() {
    let Some(device) = try_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (_cpu, mut leaf, mut compress) = build_matched(device.clone(), 0x_D50_0103_u64);
    // claim 4×8 but supply only 30 elements
    let flat: Vec<ZkgpuBabyBear> = vec![ZkgpuBabyBear(0); 30];
    let input = MixedHeightMatrixInput {
        flat: &flat,
        height: 4,
        width: 8,
    };
    let result = commit_mixed_height_host_matrices_with_retained_layers(
        &device,
        &mut leaf,
        &mut compress,
        &[input],
    );
    assert!(result.is_err(), "DAG must reject matrix len != h*w");
}
