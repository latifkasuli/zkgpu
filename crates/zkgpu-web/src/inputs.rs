//! Input generation for browser test cases.
//!
//! Mirrors `zkgpu-testkit/src/inputs.rs` without the testkit dependency.

use zkgpu_babybear::{BabyBear, P as BB_P};
use zkgpu_goldilocks::{Goldilocks, P as GL_P};
use zkgpu_poseidon2::WIDTH as POSEIDON2_WIDTH;
use zkgpu_report::{HashInputPattern, InputPattern};

pub(crate) fn make_input(log_n: u32, pattern: &InputPattern) -> Vec<BabyBear> {
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

/// Goldilocks twin of [`make_input`]. Phase E.2.b — kept as a parallel
/// function rather than generified because `InputPattern::LargeValuesDescending`
/// anchors at `P - 1 - i` and the Goldilocks 64-bit modulus flips the
/// descending range. Mirrors `zkgpu-testkit::inputs::make_goldilocks_input`
/// so browser and native runners produce byte-identical inputs for the
/// same `(log_n, pattern)`.
pub(crate) fn make_goldilocks_input(log_n: u32, pattern: &InputPattern) -> Vec<Goldilocks> {
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
                let v = i
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(seed.wrapping_mul(1442695040888963407));
                Goldilocks::new(v % GL_P)
            })
            .collect(),
    }
}

/// SplitMix64-style mixer for hash-case input generation. Must stay
/// byte-identical to the testkit's `inputs::hash_mix64` so native and
/// browser runners produce the same permutation inputs for the same
/// `(idx, seed)` pair — lets a single differential harness compare
/// across runners without pre-shuffling inputs.
fn hash_mix64(idx: u64, seed: u64) -> u64 {
    let mut z = idx.wrapping_add(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// BabyBear Poseidon2 state batch for the browser runner. Output
/// length is `num_permutations * zkgpu_poseidon2::WIDTH`. Matches
/// `zkgpu_testkit::inputs::make_babybear_hash_input` byte-for-byte.
pub(crate) fn make_babybear_hash_input(
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

/// Goldilocks twin of [`make_babybear_hash_input`].
pub(crate) fn make_goldilocks_hash_input(
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
