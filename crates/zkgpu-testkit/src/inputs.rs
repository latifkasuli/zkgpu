use zkgpu_babybear::{BabyBear, P};

use crate::suite::InputPattern;

pub fn make_input(log_n: u32, pattern: &InputPattern) -> Vec<BabyBear> {
    let n = 1usize << log_n;
    match pattern {
        InputPattern::Sequential => (0..n as u32).map(BabyBear::new).collect(),
        InputPattern::AllZeros => vec![BabyBear::new(0); n],
        InputPattern::AllOnes => vec![BabyBear::new(1); n],
        InputPattern::LargeValuesDescending => {
            (0..n as u32).map(|i| BabyBear::new(P - 1 - i)).collect()
        }
        InputPattern::PseudoRandomDeterministic { seed } => (0..n as u64)
            .map(|i| {
                let v = i
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(seed.wrapping_mul(1442695040888963407))
                    >> 33;
                BabyBear::new((v as u32) % P)
            })
            .collect(),
    }
}
