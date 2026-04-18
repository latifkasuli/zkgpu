use zkgpu_babybear::{BabyBear, P as BB_P};
use zkgpu_goldilocks::{Goldilocks, P as GL_P};

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
