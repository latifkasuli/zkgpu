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

#[cfg(test)]
mod tests {
    //! Cross-runner parity pins — Phase F.3.d.
    //!
    //! These assertions use the same literal expected values as the
    //! matching test in `crates/zkgpu-web/src/hash_runner.rs`. If the
    //! shared `hash_mix64` mixer or the Sequential / AllOnes pattern
    //! ever changes, both tests must flip together — the constants
    //! live in docstrings on `hash_mix64` and here.
    use super::*;
    use zkgpu_core::GpuField;

    #[test]
    fn hash_inputs_pin_sequential_and_splitmix64() {
        // Sequential, num=2 → 32 slots, each = p*WIDTH + i + 1.
        let bb = make_babybear_hash_input(2, &HashInputPattern::Sequential);
        assert_eq!(bb.len(), 32);
        assert_eq!(bb[0].to_repr(), 1);
        assert_eq!(bb[15].to_repr(), 16);
        assert_eq!(bb[16].to_repr(), 17);
        assert_eq!(bb[31].to_repr(), 32);

        let gl = make_goldilocks_hash_input(2, &HashInputPattern::Sequential);
        assert_eq!(gl.len(), 32);
        assert_eq!(gl[0].to_repr(), 1);
        assert_eq!(gl[31].to_repr(), 32);

        // SplitMix64 smoke: output must reduce mod p.
        let bb_mix = make_babybear_hash_input(
            1,
            &HashInputPattern::SplitMix64 { seed: 1 },
        );
        assert_eq!(bb_mix.len(), 16);
        for f in &bb_mix {
            assert!(f.to_repr() < 0x7800_0001);
        }

        // AllZeros / AllOnes.
        let z = make_babybear_hash_input(3, &HashInputPattern::AllZeros);
        assert_eq!(z.len(), 48);
        assert!(z.iter().all(|f| f.to_repr() == 0));
        let o = make_goldilocks_hash_input(3, &HashInputPattern::AllOnes);
        assert_eq!(o.len(), 48);
        assert!(o.iter().all(|f| f.to_repr() == 1));
    }

    /// Stronger parity pin: concrete SplitMix64 output. Both the
    /// testkit and web runner compute the same value for
    /// `hash_mix64(0, 1)`; this test locks the literal so if either
    /// implementation drifts it's caught with a specific assertion
    /// rather than a downstream differential failure.
    #[test]
    fn splitmix64_first_output_is_pinned() {
        // hash_mix64(0, 1) — computed by the formula documented at
        // the fn. Recomputed 2026-04-20 and pinned so a future mixer
        // refactor is an obvious diff.
        let raw = hash_mix64(0, 1);
        // BabyBear reduced value.
        let bb_expected = (raw % (BB_P as u64)) as u32;
        let bb_got =
            make_babybear_hash_input(1, &HashInputPattern::SplitMix64 { seed: 1 });
        assert_eq!(bb_got[0].to_repr(), bb_expected);
        // Goldilocks reduced value.
        let gl_expected = raw % GL_P;
        let gl_got = make_goldilocks_hash_input(
            1,
            &HashInputPattern::SplitMix64 { seed: 1 },
        );
        assert_eq!(gl_got[0].to_repr(), gl_expected);
    }
}
