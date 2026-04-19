use zkgpu_babybear::{BabyBear, P as BB_P};
use zkgpu_goldilocks::{Goldilocks, P as GL_P};
use zkgpu_poseidon2::WIDTH as POSEIDON2_WIDTH;
use zkgpu_report::HashInputPattern;

use crate::suite::InputPattern;

pub fn make_input(log_n: u32, pattern: &InputPattern) -> Vec<BabyBear> {
    let n = 1usize << log_n;
    match pattern {
        InputPattern::Sequential => (0..n as u32).map(BabyBear::new).collect(),
        InputPattern::AllZeros => vec![BabyBear::new(0); n],
        InputPattern::AllOnes => vec![BabyBear::new(1); n],
        InputPattern::LargeValuesDescending => {
            (0..n as u32).map(|i| BabyBear::new(BB_P - 1 - i)).collect()
        }
        InputPattern::PseudoRandomDeterministic { seed } => (0..n as u64)
            .map(|i| {
                let v = i
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(seed.wrapping_mul(1442695040888963407))
                    >> 33;
                BabyBear::new((v as u32) % BB_P)
            })
            .collect(),
    }
}

/// Goldilocks twin of [`make_input`]. Phase E.1.c — kept as a parallel
/// function (not generic) because `InputPattern::LargeValuesDescending`
/// packs one 32-bit value per index into the field, and the Goldilocks
/// field is 64-bit so the "descending" anchor is different (`P - 1 - i`
/// uses the full 64-bit modulus). Goldilocks `Sequential` / `AllZeros` /
/// `AllOnes` are element-for-element mirrors of the BabyBear path.
pub fn make_goldilocks_input(log_n: u32, pattern: &InputPattern) -> Vec<Goldilocks> {
    let n = 1usize << log_n;
    match pattern {
        InputPattern::Sequential => (0..n as u64).map(Goldilocks::new).collect(),
        InputPattern::AllZeros => vec![Goldilocks::new(0); n],
        InputPattern::AllOnes => vec![Goldilocks::new(1); n],
        InputPattern::LargeValuesDescending => (0..n as u64)
            .map(|i| Goldilocks::new(GL_P.wrapping_sub(1).wrapping_sub(i)))
            .collect(),
        InputPattern::PseudoRandomDeterministic { seed } => (0..n as u64)
            .map(|i| {
                // Same SplitMix64-ish mixer used by the BabyBear path,
                // but the output is reduced mod the Goldilocks modulus
                // (still fits a u64 so the % is a single op).
                let v = i
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(seed.wrapping_mul(1442695040888963407));
                Goldilocks::new(v % GL_P)
            })
            .collect(),
    }
}

/// SplitMix64-style mixer used by both hash-input paths. Factored
/// out so BabyBear and Goldilocks generators produce byte-identical
/// `(permutation_index, slot)` → seed state, which keeps diff tests
/// portable across fields.
fn hash_mix64(idx: u64, seed: u64) -> u64 {
    // Constants lifted from the canonical SplitMix64; two mix rounds
    // are enough to decorrelate neighbouring `idx` inputs.
    let mut z = idx.wrapping_add(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Build a flat Poseidon2 state batch for BabyBear. Output length is
/// `num_permutations * zkgpu_poseidon2::WIDTH`; each `WIDTH`-element
/// block is one independent permutation instance.
///
/// Phase F.3.b — input shape matches what
/// [`zkgpu_wgpu::WgpuBabyBearPoseidon2Plan::execute`] expects.
pub fn make_babybear_hash_input(
    num_permutations: u32,
    pattern: &HashInputPattern,
) -> Vec<BabyBear> {
    let total = (num_permutations as usize) * POSEIDON2_WIDTH;
    match pattern {
        HashInputPattern::AllZeros => vec![BabyBear::new(0); total],
        HashInputPattern::AllOnes => vec![BabyBear::new(1); total],
        HashInputPattern::Sequential => (0..num_permutations)
            .flat_map(|p| {
                (0..POSEIDON2_WIDTH as u32).map(move |i| {
                    // `p * WIDTH + i + 1`, reduced mod BabyBear. The
                    // `+1` keeps slot 0 from being the identity input.
                    let raw = (p as u64) * (POSEIDON2_WIDTH as u64) + (i as u64) + 1;
                    BabyBear::new((raw % (BB_P as u64)) as u32)
                })
            })
            .collect(),
        HashInputPattern::SplitMix64 { seed } => (0..total as u64)
            .map(|idx| BabyBear::new((hash_mix64(idx, *seed) % (BB_P as u64)) as u32))
            .collect(),
    }
}

/// Goldilocks twin of [`make_babybear_hash_input`]. The mixer output
/// is reduced mod the 64-bit Goldilocks modulus so the second u32
/// limb is populated for about half of the samples — ensures GPU
/// u32x2 arithmetic is exercised by the differential tests.
pub fn make_goldilocks_hash_input(
    num_permutations: u32,
    pattern: &HashInputPattern,
) -> Vec<Goldilocks> {
    let total = (num_permutations as usize) * POSEIDON2_WIDTH;
    match pattern {
        HashInputPattern::AllZeros => vec![Goldilocks::new(0); total],
        HashInputPattern::AllOnes => vec![Goldilocks::new(1); total],
        HashInputPattern::Sequential => (0..num_permutations)
            .flat_map(|p| {
                (0..POSEIDON2_WIDTH as u32).map(move |i| {
                    let raw = (p as u64) * (POSEIDON2_WIDTH as u64) + (i as u64) + 1;
                    Goldilocks::new(raw % GL_P)
                })
            })
            .collect(),
        HashInputPattern::SplitMix64 { seed } => (0..total as u64)
            .map(|idx| Goldilocks::new(hash_mix64(idx, *seed) % GL_P))
            .collect(),
    }
}
