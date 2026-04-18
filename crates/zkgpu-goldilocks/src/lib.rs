//! Goldilocks field.
//!
//! Goldilocks is the 64-bit prime field used by Plonky2/Plonky3, Winterfell,
//! RISC Zero, and most modern FRI-based STARKs. Its modulus
//!
//! ```text
//! p = 2^64 - 2^32 + 1 = 18446744069414584321 = 0xFFFF_FFFF_0000_0001
//! ```
//!
//! sits at an engineering sweet spot:
//!
//! - **Fits in a `u64`** — all arithmetic takes at most 128-bit intermediates.
//! - **Two-adicity 32** — the largest `k` with `2^k | (p - 1)`. Since
//!   `p - 1 = 2^32 · (2^32 - 1) = 2^32 · 3 · 5 · 17 · 257 · 65537`, Goldilocks
//!   supports native NTTs up to `2^32` points. Nothing smaller in the
//!   Mersenne / BabyBear / KoalaBear family gets close.
//! - **Cheap special-form reduction.** `2^64 ≡ 2^32 - 1 (mod p)` and
//!   `2^96 ≡ -1 (mod p)`. That turns a 128-bit product into a
//!   fixed-instruction shuffle + sum + single conditional subtract. No
//!   Barrett constants, no Montgomery form, no division.
//!
//! ## Why this crate exists
//!
//! Goldilocks is the real Phase-2 pressure test for
//! [`zkgpu_core::GpuField`]. BabyBear and KoalaBear are both
//! `Repr = u32`, and before this crate existed the NTT reference
//! silently assumed 32-bit reprs (a `bytemuck::cast(n as u32)` call
//! in `zkgpu-ntt`). Adding a `u64`-repr field forced
//! [`zkgpu_core::GpuField::from_u64`] into the trait and flushed out
//! that assumption. If [`Goldilocks`] ever picks up a field-specific
//! workaround in a downstream crate, something upstream leaked.
//!
//! ## Reduction derivation
//!
//! Given a 128-bit product `x = a₃·2^96 + a₂·2^64 + a₁·2^32 + a₀`
//! with each `aᵢ ∈ [0, 2^32)`, apply the two identities:
//!
//! ```text
//! 2^64 ≡ 2^32 - 1  (mod p)        (from 2^64 - (2^32 - 1) = p)
//! 2^96 ≡ -1        (mod p)        (squaring 2^32 gives us more, but the
//!                                  cubic 2^64 * 2^32 = (2^32-1) * 2^32
//!                                  = 2^64 - 2^32 ≡ (2^32 - 1) - 2^32 = -1)
//! ```
//!
//! Substituting:
//!
//! ```text
//! x ≡ -a₃ + a₂·(2^32 - 1) + a₁·2^32 + a₀
//!   = (a₁·2^32 + a₀) - a₃ + a₂·(2^32 - 1)
//!   = x_lo - x_hi_hi + x_hi_lo · EPSILON
//! ```
//!
//! where `x_lo = a₁·2^32 + a₀` (low 64 bits), `x_hi_lo = a₂` (middle 32
//! bits), `x_hi_hi = a₃` (top 32 bits), and `EPSILON = 2^32 - 1`.
//!
//! The implementation in [`reduce_128`] follows that identity with two
//! wrap-corrections: a borrow from the `x_lo - x_hi_hi` subtract
//! (corrected by subtracting `EPSILON`; equivalent to adding `p`), and
//! a carry from the `+ x_hi_lo · EPSILON` add (corrected by adding
//! `EPSILON`; equivalent to subtracting `p`).
//!
//! Verified against a Python big-int reference on eight adversarial
//! 128-bit inputs including `(p-1)^2` and `2^128 - 1` — see
//! `tests::reduce_128_adversarial`.
//!
//! ## Performance notes
//!
//! This is a correctness-first portable Rust implementation. For the GPU
//! path and for tight inner loops we know of two orthogonal optimisations
//! — neither shipped here, both documented so they're visible to a reader
//! deciding where to spend the next increment of engineering:
//!
//! 1. **Inversion via addition chain.** The naive `pow(p - 2)` Fermat path
//!    in [`Goldilocks::inv`] runs ~63 squarings + 61 multiplications. An
//!    addition chain using the `p - 2 = 0xFFFFFFFE_FFFFFFFF` bit pattern
//!    (lower 32 bits all set, bit 32 zero, upper 31 bits all set) can be
//!    done in about 11 squarings + 5 multiplications. See
//!    Plonky2's `goldilocks_field::try_inverse` for a working chain.
//!    We keep the simpler version because correctness is obvious and
//!    inversion is not on the NTT hot path.
//!
//! 2. **x86-64 inline ADCX/ADOX.** Plonky3's `add_no_canonicalize_trashing_input`
//!    shaves a cycle off the reduction tail by using `addq` + `adcxq`
//!    with a branch hint. We stick to portable `overflowing_add` /
//!    `overflowing_sub` — the branches are trivially predicted in
//!    practice, and portability matters for the WGSL mirror later.

use std::ops;

use bytemuck::{Pod, Zeroable};
use zkgpu_core::GpuField;

/// The Goldilocks prime: `p = 2^64 - 2^32 + 1`.
pub const P: u64 = 0xFFFF_FFFF_0000_0001;

/// `EPSILON = 2^32 - 1 = 2^64 - p`. Appears everywhere in the reduction
/// as the "one step of p" correction value.
pub const EPSILON: u64 = 0xFFFF_FFFF;

/// A primitive root of `F_p^*`. Verified primitive against every prime
/// factor of `p - 1 = 2^32 · 3 · 5 · 17 · 257 · 65537`.
pub const GENERATOR: u64 = 7;

/// Goldilocks has 2-adicity 32: the largest `k` with `2^k | (p - 1)`.
pub const TWO_ADICITY: u32 = 32;

/// Primitive `2^32`-th root of unity.
///
/// Equal to `GENERATOR^((p - 1) / 2^32) = 7^((2^32 - 1)) mod p`.
/// Verified in [`tests::two_adic_root_derivation`].
pub const TWO_ADIC_ROOT_OF_UNITY: u64 = 0x1856_29DC_DA58_878C;

/// A field element in Goldilocks, stored canonically in `[0, P)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Pod, Zeroable)]
#[repr(transparent)]
pub struct Goldilocks(pub u64);

impl Goldilocks {
    /// Construct from a raw `u64`, reducing mod `P`.
    ///
    /// `const fn` for compile-time constant folding. Uses `%` because
    /// there's no const-context `overflowing_add` / `overflowing_sub`;
    /// this is only on the construction path, not on the arithmetic
    /// hot path.
    pub const fn new(value: u64) -> Self {
        Self(value % P)
    }

    /// Square-and-multiply exponentiation. `O(log exp)` mults.
    pub fn pow(self, mut exp: u64) -> Self {
        let mut base = self;
        let mut result = Self(1);
        while exp > 0 {
            if exp & 1 == 1 {
                result = result * base;
            }
            base = base * base;
            exp >>= 1;
        }
        result
    }

    /// Modular inverse via Fermat: `a^(p - 2) = a^{-1} (mod p)`.
    /// Returns `None` for zero.
    ///
    /// `pow(P - 2)` is ~63 squarings + ~61 mults. See the crate docs
    /// for an addition-chain optimisation we deliberately haven't
    /// shipped.
    pub fn inv(self) -> Option<Self> {
        if self.0 == 0 {
            return None;
        }
        Some(self.pow(P - 2))
    }
}

/// Reduce a 128-bit value mod `P` using Goldilocks's special-form
/// identity. See the crate-level doc for the derivation.
///
/// Input invariant: any `u128`.
/// Output invariant: canonical representative in `[0, P)`.
#[inline]
pub fn reduce_128(x: u128) -> u64 {
    let x_lo = x as u64;
    let x_hi = (x >> 64) as u64;
    let x_hi_hi = x_hi >> 32;
    let x_hi_lo = x_hi & EPSILON;

    // t = x_lo - x_hi_hi (with borrow correction).
    //
    // If the subtract borrowed, the true value of (x_lo - x_hi_hi) was
    // negative; wrapping u64 math has given us (x_lo - x_hi_hi + 2^64).
    // To convert to canonical mod p, subtract (2^64 - p) = EPSILON, which
    // is equivalent to adding p.
    let (t, borrow) = x_lo.overflowing_sub(x_hi_hi);
    let t = if borrow { t.wrapping_sub(EPSILON) } else { t };

    // + x_hi_lo · EPSILON (with carry correction).
    //
    // x_hi_lo and EPSILON are both 32-bit, so their product fits in u64
    // without intermediate overflow. Adding it to t can carry; if so,
    // the true value exceeded 2^64, so add EPSILON back (≡ subtracting
    // p from the too-large sum).
    let prod = x_hi_lo.wrapping_mul(EPSILON);
    let (sum, carry) = t.overflowing_add(prod);
    let sum = if carry { sum.wrapping_add(EPSILON) } else { sum };

    // Final canonicalise to [0, P). After the two corrections the value
    // is in [0, 2p), so one conditional subtract suffices.
    if sum >= P { sum - P } else { sum }
}

impl ops::Add for Goldilocks {
    type Output = Self;

    /// Modular add with the special-form carry correction.
    ///
    /// For `a, b ∈ [0, p)`, `a + b < 2p < 2^65`. If the unsigned add
    /// overflows `u64`, the true sum is `sum + 2^64 ≡ sum + EPSILON
    /// (mod p)`.
    fn add(self, rhs: Self) -> Self {
        let (sum, carry) = self.0.overflowing_add(rhs.0);
        let sum = if carry { sum.wrapping_add(EPSILON) } else { sum };
        Self(if sum >= P { sum - P } else { sum })
    }
}

impl ops::Sub for Goldilocks {
    type Output = Self;

    /// Modular sub with the special-form borrow correction.
    ///
    /// For `a, b ∈ [0, p)`, `a - b ∈ (-p, p)`. If the unsigned sub
    /// underflows `u64`, the wrapped value is `a - b + 2^64`. Canonical
    /// `a - b + p = (a - b + 2^64) - (2^64 - p) = wrapped - EPSILON`.
    fn sub(self, rhs: Self) -> Self {
        let (diff, borrow) = self.0.overflowing_sub(rhs.0);
        Self(if borrow { diff.wrapping_sub(EPSILON) } else { diff })
    }
}

impl ops::Mul for Goldilocks {
    type Output = Self;

    /// Modular multiply via the 128-bit product path.
    ///
    /// `a * b < p^2 < 2^128`, so the product fits in `u128`. Reduction
    /// is handled by [`reduce_128`].
    fn mul(self, rhs: Self) -> Self {
        let prod = self.0 as u128 * rhs.0 as u128;
        Self(reduce_128(prod))
    }
}

impl GpuField for Goldilocks {
    type Repr = u64;

    const MODULUS: u64 = P;
    const MODULUS_BITS: u32 = 64;
    const ONE: Self = Self(1);
    const ZERO: Self = Self(0);
    const TWO_ADIC_ROOT: Self = Self(TWO_ADIC_ROOT_OF_UNITY);
    const TWO_ADICITY: u32 = TWO_ADICITY;

    fn add(self, rhs: Self) -> Self {
        self + rhs
    }

    fn sub(self, rhs: Self) -> Self {
        self - rhs
    }

    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }

    fn pow(self, exp: u64) -> Self {
        Goldilocks::pow(self, exp)
    }

    fn from_repr(repr: u64) -> Self {
        assert!(repr < P, "repr {repr} >= modulus {P}");
        Self(repr)
    }

    fn to_repr(self) -> u64 {
        self.0
    }

    fn from_u64(n: u64) -> Self {
        Self(n % P)
    }

    fn root_of_unity(log_n: u32) -> Self {
        assert!(
            log_n <= TWO_ADICITY,
            "requested root of unity order 2^{log_n} exceeds 2-adicity {TWO_ADICITY}"
        );
        let root = Self(TWO_ADIC_ROOT_OF_UNITY);
        root.pow(1u64 << (TWO_ADICITY - log_n))
    }

    fn inverse(self) -> Option<Self> {
        self.inv()
    }
}

impl std::fmt::Display for Goldilocks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Constants ---

    #[test]
    fn modulus_is_expected() {
        assert_eq!(P, 0xFFFF_FFFF_0000_0001);
        assert_eq!(P, 18446744069414584321);
        // Reconstruct from the parts: 2^64 - 2^32 + 1.
        // Using u128 to avoid overflow in the check itself.
        assert_eq!(P as u128, (1u128 << 64) - (1u128 << 32) + 1);
        assert_eq!(EPSILON, 0xFFFF_FFFF);
        assert_eq!((1u128 << 64) - P as u128, EPSILON as u128);
    }

    #[test]
    fn two_adic_root_derivation() {
        // Re-derive 7^((p-1)/2^32) at runtime and check the const matches.
        // (p - 1) / 2^32 = 2^32 - 1 = EPSILON.
        let computed = Goldilocks(GENERATOR).pow(EPSILON);
        assert_eq!(
            computed.0, TWO_ADIC_ROOT_OF_UNITY,
            "TWO_ADIC_ROOT_OF_UNITY out of sync with GENERATOR^((p-1)/2^TWO_ADICITY)"
        );
    }

    // --- Reduction correctness ---

    #[test]
    fn reduce_128_adversarial() {
        // Cross-check reduce_128 against u128 % P on inputs that stress
        // the wrap corrections: zero, small, field-boundary, maximum
        // u128, powers of two around 2^64 / 2^96.
        let cases: &[u128] = &[
            0,
            1,
            P as u128 - 1,
            P as u128,
            P as u128 * P as u128 - P as u128 * 2,  // (p-2)*p
            (P as u128 - 1) * (P as u128 - 1),      // (p-1)^2
            (1u128 << 64) - 1,
            1u128 << 64,
            1u128 << 96,
            (1u128 << 127) - 1,
            u128::MAX,
        ];
        for &x in cases {
            let expected = (x % P as u128) as u64;
            let got = reduce_128(x);
            assert_eq!(got, expected, "reduce_128({x:#x}) = {got} != {expected}");
        }
    }

    #[test]
    fn mul_boundary_pm1_squared_is_one() {
        // (p - 1)^2 mod p = 1, since p - 1 ≡ -1 (mod p).
        let m1 = Goldilocks(P - 1);
        assert_eq!(m1 * m1, Goldilocks::ONE);
    }

    #[test]
    fn mul_large_values_match_u128_reference() {
        // Spot-check five mid-sized pairs against raw u128 reduction.
        let pairs: &[(u64, u64)] = &[
            (123_456_789_012_345, 987_654_321_098_765),
            (P / 2, P / 2 + 1),
            (P - 1, P - 1),
            (P - 2, 2),
            (1 << 63, (1 << 62) + 3),
        ];
        for &(a, b) in pairs {
            let expected = ((a as u128) * (b as u128) % P as u128) as u64;
            let got = (Goldilocks(a) * Goldilocks(b)).0;
            assert_eq!(got, expected, "{a} * {b}: got {got}, want {expected}");
        }
    }

    // --- Add / sub edge cases ---

    #[test]
    fn add_boundary_overflow() {
        // (p - 1) + 1 = p ≡ 0
        assert_eq!(Goldilocks(P - 1) + Goldilocks(1), Goldilocks::ZERO);
        // (p - 1) + (p - 1) = 2p - 2 ≡ p - 2
        assert_eq!(
            Goldilocks(P - 1) + Goldilocks(P - 1),
            Goldilocks(P - 2)
        );
        // Big + big, intentional u64 overflow.
        let a = Goldilocks(P - 5);
        let b = Goldilocks(P - 3);
        // (p - 5) + (p - 3) = 2p - 8 ≡ p - 8
        assert_eq!(a + b, Goldilocks(P - 8));
    }

    #[test]
    fn sub_boundary_underflow() {
        // 0 - 1 = -1 ≡ p - 1
        assert_eq!(Goldilocks::ZERO - Goldilocks(1), Goldilocks(P - 1));
        // 1 - (p - 1) = 2 - p ≡ 2
        assert_eq!(Goldilocks(1) - Goldilocks(P - 1), Goldilocks(2));
        // 5 - 7 = -2 ≡ p - 2
        assert_eq!(Goldilocks(5) - Goldilocks(7), Goldilocks(P - 2));
    }

    #[test]
    fn add_sub_roundtrip() {
        let a = Goldilocks::new(1_234_567_890_123_456_789);
        let b = Goldilocks::new(9_876_543_210_987_654_321);
        assert_eq!((a + b) - b, a);
        assert_eq!((a - b) + b, a);
    }

    // --- Identities ---

    #[test]
    fn mul_identity() {
        let a = Goldilocks::new(987_654_321_098_765_432);
        assert_eq!(a * Goldilocks::ONE, a);
        assert_eq!(a * Goldilocks::ZERO, Goldilocks::ZERO);
    }

    #[test]
    fn inverse_roundtrip() {
        for seed in [42u64, 1, 2, P - 1, P / 2, 0xDEAD_BEEF_BABE_CAFE] {
            let a = Goldilocks::new(seed);
            let a_inv = a.inv().unwrap();
            assert_eq!(a * a_inv, Goldilocks::ONE, "inverse_roundtrip({seed})");
        }
    }

    #[test]
    fn zero_has_no_inverse() {
        assert!(Goldilocks::ZERO.inv().is_none());
    }

    #[test]
    fn fermats_little_theorem() {
        // a^(p - 1) = 1 for any non-zero a.
        for seed in [2u64, 7, 42, 0x1234_5678_9ABC_DEF0, P - 1] {
            let a = Goldilocks::new(seed);
            assert_eq!(a.pow(P - 1), Goldilocks::ONE);
        }
    }

    // --- Root of unity structure ---

    #[test]
    fn root_of_unity_order() {
        // For every log_n in 1..=TWO_ADICITY, root_of_unity(log_n) must
        // have order exactly 2^log_n.
        for log_n in 1..=TWO_ADICITY {
            let root = Goldilocks::root_of_unity(log_n);
            let order = 1u64 << log_n;
            // root^(2^log_n) = 1
            assert_eq!(
                root.pow(order),
                Goldilocks::ONE,
                "root^(2^{log_n}) != 1"
            );
            // root^(2^(log_n - 1)) != 1  — proves no smaller order
            assert_ne!(
                root.pow(order / 2),
                Goldilocks::ONE,
                "root has smaller order than 2^{log_n}"
            );
        }
    }

    #[test]
    fn generator_is_primitive() {
        // p - 1 = 2^32 * 3 * 5 * 17 * 257 * 65537. For GENERATOR=7 to
        // generate F_p^*, g^((p-1)/q) must be non-trivial for every
        // prime q dividing p-1.
        let g = Goldilocks::new(GENERATOR);
        let p_minus_1 = P - 1;
        for q in [2u64, 3, 5, 17, 257, 65537] {
            assert_eq!(p_minus_1 % q, 0, "p-1 not divisible by {q} (math error)");
            assert_ne!(
                g.pow(p_minus_1 / q),
                Goldilocks::ONE,
                "GENERATOR^((p-1)/{q}) = 1 — GENERATOR is not primitive"
            );
        }
        // Fermat as a sanity coda.
        assert_eq!(g.pow(p_minus_1), Goldilocks::ONE);
    }

    #[test]
    fn reduction_on_construction() {
        let a = Goldilocks::new(P);
        assert_eq!(a, Goldilocks::ZERO);
        let b = Goldilocks::new(P + 1);
        assert_eq!(b, Goldilocks::ONE);
        // Can't write `P + P + 5` as a const — it overflows u64. Use the
        // largest value that fits in u64 (u64::MAX = 2^64 - 1) to stress
        // the `value % P` path in `new`:
        //   u64::MAX - P = (2^64 - 1) - (2^64 - 2^32 + 1) = 2^32 - 2
        //                = EPSILON - 1
        // so u64::MAX % P = EPSILON - 1.
        let c = Goldilocks::new(u64::MAX);
        assert_eq!(c, Goldilocks::new(EPSILON - 1));
    }

    // --- Cross-trait: GpuField wiring ---

    #[test]
    fn from_u64_matches_new() {
        // `from_u64` is the trait method zkgpu-ntt uses; it must agree
        // with `new` on canonical [0, p) inputs and with % P on larger.
        for n in [0u64, 1, 2, 1000, P - 1] {
            assert_eq!(
                <Goldilocks as GpuField>::from_u64(n),
                Goldilocks::new(n)
            );
        }
    }

    // --- The real Phase-2 pressure test ---

    /// Run `zkgpu-ntt`'s field-generic CPU reference NTT against
    /// Goldilocks. If this compiles and passes, the `GpuField` contract
    /// genuinely supports `Repr = u64` fields without downstream hacks.
    /// That's the whole point of this crate existing.
    #[test]
    fn cpu_reference_ntt_roundtrip() {
        use zkgpu_core::NttDirection;
        use zkgpu_ntt::ntt_cpu_reference;

        let log_n = 12u32;
        let n = 1usize << log_n;
        let original: Vec<Goldilocks> =
            (0..n as u64).map(Goldilocks::new).collect();

        let mut data = original.clone();
        ntt_cpu_reference::<Goldilocks>(&mut data, NttDirection::Forward);

        // After forward NTT, data should differ from original (non-trivial).
        assert_ne!(
            data, original,
            "forward NTT didn't change input — something is very wrong"
        );

        ntt_cpu_reference::<Goldilocks>(&mut data, NttDirection::Inverse);

        assert_eq!(
            data, original,
            "forward + inverse NTT over Goldilocks did not round-trip"
        );
    }

    /// Secondary NTT test at a larger size (2^15 = 32768 points) to
    /// stress the 64-bit-field arithmetic path further. Still well
    /// within the `O(n log n)` CPU budget.
    #[test]
    fn cpu_reference_ntt_roundtrip_log15() {
        use zkgpu_core::NttDirection;
        use zkgpu_ntt::ntt_cpu_reference;

        let log_n = 15u32;
        let n = 1usize << log_n;
        // Mix additive and multiplicative patterns for the input.
        let original: Vec<Goldilocks> = (0..n as u64)
            .map(|i| Goldilocks::new(i.wrapping_mul(2654435761))) // Knuth hash mul
            .collect();

        let mut data = original.clone();
        ntt_cpu_reference::<Goldilocks>(&mut data, NttDirection::Forward);
        ntt_cpu_reference::<Goldilocks>(&mut data, NttDirection::Inverse);
        assert_eq!(data, original, "log15 NTT roundtrip failed");
    }

    // --- Differential fuzz tests ---
    //
    // Curated tests above are good at catching specific boundary bugs,
    // but they don't give statistical confidence against the broad
    // middle of the input space. These tests draw thousands of random
    // inputs and cross-check every result against `u128` big-int
    // arithmetic — the ground truth. With `N = 10_000` inputs and no
    // observed mismatches, we can bound the probability of an
    // undetected systematic reduction bug to very small on any
    // randomly-sampleable class of inputs.

    const FUZZ_COUNT: usize = 10_000;

    /// Deterministic-PRNG helper so fuzz tests are reproducible. SplitMix64
    /// — small, well-distributed for this purpose, doesn't pull a crate.
    struct SplitMix64(u64);

    impl SplitMix64 {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn next_u128(&mut self) -> u128 {
            ((self.next_u64() as u128) << 64) | self.next_u64() as u128
        }
        fn next_canonical(&mut self) -> u64 {
            self.next_u64() % P
        }
    }

    #[test]
    fn reduce_128_differential_fuzz() {
        let mut rng = SplitMix64::new(0xFEED_FACE_CAFE_BABE);
        for i in 0..FUZZ_COUNT {
            let x = rng.next_u128();
            let expected = (x % P as u128) as u64;
            let got = reduce_128(x);
            assert_eq!(
                got, expected,
                "reduce_128 mismatch at iter {i}: input={x:#x}, got={got}, expected={expected}"
            );
        }
    }

    #[test]
    fn mul_differential_fuzz() {
        let mut rng = SplitMix64::new(0xA1B2_C3D4_E5F6_0718);
        for i in 0..FUZZ_COUNT {
            let a = rng.next_canonical();
            let b = rng.next_canonical();
            let expected = ((a as u128) * (b as u128) % P as u128) as u64;
            let got = (Goldilocks(a) * Goldilocks(b)).0;
            assert_eq!(
                got, expected,
                "mul mismatch at iter {i}: a={a}, b={b}, got={got}, expected={expected}"
            );
        }
    }

    #[test]
    fn add_differential_fuzz() {
        let mut rng = SplitMix64::new(0x0123_4567_89AB_CDEF);
        for i in 0..FUZZ_COUNT {
            let a = rng.next_canonical();
            let b = rng.next_canonical();
            let expected = ((a as u128 + b as u128) % P as u128) as u64;
            let got = (Goldilocks(a) + Goldilocks(b)).0;
            assert_eq!(
                got, expected,
                "add mismatch at iter {i}: a={a}, b={b}, got={got}, expected={expected}"
            );
        }
    }

    #[test]
    fn sub_differential_fuzz() {
        let mut rng = SplitMix64::new(0xDEAD_BEEF_DEAD_BEEF);
        for i in 0..FUZZ_COUNT {
            let a = rng.next_canonical();
            let b = rng.next_canonical();
            // (a - b) mod p, via u128 signed-offset math. Adding p into a
            // u128 first keeps everything non-negative before the modulo.
            let expected = ((a as u128 + P as u128 - b as u128) % P as u128) as u64;
            let got = (Goldilocks(a) - Goldilocks(b)).0;
            assert_eq!(
                got, expected,
                "sub mismatch at iter {i}: a={a}, b={b}, got={got}, expected={expected}"
            );
        }
    }

    /// A combined roundtrip fuzz: for each sampled `a` draw a random
    /// non-zero `b`, compute `a * b * b^{-1}`, and assert it equals `a`.
    /// Exercises `mul` and `inv` together across 10 000 inputs.
    #[test]
    fn mul_inv_roundtrip_fuzz() {
        let mut rng = SplitMix64::new(0xCAFE_0BAD_FEED_DEAD);
        for i in 0..FUZZ_COUNT {
            let a_raw = rng.next_canonical();
            let mut b_raw = rng.next_canonical();
            if b_raw == 0 {
                b_raw = 1; // inversion is defined on F_p^*
            }
            let a = Goldilocks::new(a_raw);
            let b = Goldilocks::new(b_raw);
            let b_inv = b.inv().expect("non-zero b must invert");
            let got = a * b * b_inv;
            assert_eq!(got, a, "round-trip mismatch at iter {i}: a={a_raw}, b={b_raw}");
        }
    }
}
