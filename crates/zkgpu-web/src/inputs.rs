//! Input generation for browser test cases.
//!
//! Mirrors `zkgpu-testkit/src/inputs.rs` without the testkit dependency.

use zkgpu_babybear::{BabyBear, P as BB_P};
use zkgpu_goldilocks::{Goldilocks, P as GL_P};
use zkgpu_report::InputPattern;

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
