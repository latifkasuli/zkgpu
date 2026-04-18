//! KoalaBear field.
//!
//! KoalaBear is a 31-bit prime field with modulus `p = 2^31 - 2^24 + 1`.
//! Like BabyBear it sits inside a `u32`, has cheap modular reduction, and
//! exposes a large 2-adic subgroup suitable for NTT — just at a different
//! balance point. KoalaBear's `2-adicity = 24` (vs BabyBear's 27) trades a
//! smaller NTT-length ceiling for a tighter `p - 1 = 2^24 · 127` factorization
//! and — in practice — better compatibility with 24-bit-oriented zkVM
//! designs (VALIDA, some RISC-V provers).
//!
//! The math surface mirrors `zkgpu-babybear`: simple `u32` storage, textbook
//! adds / subtracts via overflow checks, and 64-bit intermediate for `mul`.
//! The purpose of adding this crate is to apply real pressure on
//! [`zkgpu_core::GpuField`] so we know the contract isn't silently
//! BabyBear-shaped.
//!
//! ## Constants
//!
//! ```text
//! p              = 2^31 - 2^24 + 1 = 2130706433 = 0x7F000001
//! p - 1          = 2^24 * 127
//! TWO_ADICITY    = 24   (largest k with 2^k | p - 1)
//! GENERATOR      = 3    (primitive root of F_p^*)
//! 2^24-th root   = 3^127 mod p = 1791270792 = 0x6AC49F88
//! ```
//!
//! Derivation of `TWO_ADIC_ROOT_OF_UNITY` is verified in
//! [`tests::two_adic_root_derivation`] — if you change the constants
//! above, that test will fail fast rather than silently producing a
//! non-root-of-unity.

use std::ops;

use bytemuck::{Pod, Zeroable};
use zkgpu_core::GpuField;

/// The KoalaBear prime: `p = 2^31 - 2^24 + 1 = 2130706433`.
pub const P: u32 = 0x7F000001;

/// A generator of the multiplicative group of KoalaBear (`|F_p^*| = 2^24 · 127`).
pub const GENERATOR: u32 = 3;

/// KoalaBear has 2-adicity 24: the largest k with `2^k | (p - 1)`.
pub const TWO_ADICITY: u32 = 24;

/// A primitive `2^24`-th root of unity in KoalaBear.
///
/// Equal to `GENERATOR^((p - 1) / 2^24) = 3^127 mod p = 1791270792`.
/// Verified in the tests module.
pub const TWO_ADIC_ROOT_OF_UNITY: u32 = 0x6AC49F88;

/// A field element in KoalaBear, stored as a `u32` in `[0, P)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Pod, Zeroable)]
#[repr(transparent)]
pub struct KoalaBear(pub u32);

impl KoalaBear {
    /// Construct from a raw `u32`, reducing mod `P`.
    pub const fn new(value: u32) -> Self {
        Self(value % P)
    }

    /// Square-and-multiply exponentiation.
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

    /// Modular inverse via Fermat: `a^(p-2) = a^{-1} mod p`. Returns
    /// `None` for zero.
    pub fn inv(self) -> Option<Self> {
        if self.0 == 0 {
            return None;
        }
        Some(self.pow(P as u64 - 2))
    }
}

impl ops::Add for KoalaBear {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let sum = self.0 as u64 + rhs.0 as u64;
        let reduced = if sum >= P as u64 { sum - P as u64 } else { sum };
        Self(reduced as u32)
    }
}

impl ops::Sub for KoalaBear {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        let (diff, borrow) = self.0.overflowing_sub(rhs.0);
        Self(if borrow { diff.wrapping_add(P) } else { diff })
    }
}

impl ops::Mul for KoalaBear {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        let prod = self.0 as u64 * rhs.0 as u64;
        Self((prod % P as u64) as u32)
    }
}

impl GpuField for KoalaBear {
    type Repr = u32;

    const MODULUS: u32 = P;
    const MODULUS_BITS: u32 = 31;
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
        KoalaBear::pow(self, exp)
    }

    fn from_repr(repr: u32) -> Self {
        assert!(repr < P, "repr {repr} >= modulus {P}");
        Self(repr)
    }

    fn to_repr(self) -> u32 {
        self.0
    }

    fn from_u64(n: u64) -> Self {
        Self((n % P as u64) as u32)
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

impl std::fmt::Display for KoalaBear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modulus_is_expected() {
        assert_eq!(P, (1u32 << 31) - (1u32 << 24) + 1);
        assert_eq!(P, 2130706433);
    }

    #[test]
    fn two_adic_root_derivation() {
        // Verify the hardcoded constant really is 3^127 mod p.
        let computed = KoalaBear(GENERATOR).pow(127);
        assert_eq!(
            computed.0, TWO_ADIC_ROOT_OF_UNITY,
            "TWO_ADIC_ROOT_OF_UNITY out of sync with GENERATOR^((p-1)/2^TWO_ADICITY)"
        );
    }

    #[test]
    fn add_sub_roundtrip() {
        let a = KoalaBear::new(1234567);
        let b = KoalaBear::new(7654321);
        assert_eq!((a + b) - b, a);
    }

    #[test]
    fn mul_identity() {
        let a = KoalaBear::new(999999);
        assert_eq!(a * KoalaBear::ONE, a);
        assert_eq!(a * KoalaBear::ZERO, KoalaBear::ZERO);
    }

    #[test]
    fn inverse_roundtrip() {
        let a = KoalaBear::new(42);
        let a_inv = a.inv().unwrap();
        assert_eq!(a * a_inv, KoalaBear::ONE);
    }

    #[test]
    fn zero_has_no_inverse() {
        assert!(KoalaBear::ZERO.inv().is_none());
    }

    #[test]
    fn root_of_unity_order() {
        for log_n in 1..=TWO_ADICITY {
            let root = KoalaBear::root_of_unity(log_n);
            let order = 1u64 << log_n;
            assert_eq!(root.pow(order), KoalaBear::ONE, "root^(2^{log_n}) != 1");
            assert_ne!(
                root.pow(order / 2),
                KoalaBear::ONE,
                "root has smaller order than 2^{log_n}"
            );
        }
    }

    #[test]
    fn generator_is_primitive() {
        // A generator of F_p^* must NOT be a k-th root of unity for any
        // proper divisor k of p - 1. For KoalaBear p - 1 = 2^24 * 127,
        // so we check 3^((p-1)/2) != 1 AND 3^((p-1)/127) != 1.
        let g = KoalaBear::new(GENERATOR);
        assert_eq!(g.pow(P as u64 - 1), KoalaBear::ONE, "Fermat failed");
        assert_ne!(
            g.pow((P as u64 - 1) / 2),
            KoalaBear::ONE,
            "GENERATOR is a quadratic residue — not primitive"
        );
        assert_ne!(
            g.pow((P as u64 - 1) / 127),
            KoalaBear::ONE,
            "GENERATOR^((p-1)/127) = 1 — not primitive"
        );
    }

    #[test]
    fn reduction_on_construction() {
        let a = KoalaBear::new(P);
        assert_eq!(a, KoalaBear::ZERO);
        let b = KoalaBear::new(P + 1);
        assert_eq!(b, KoalaBear::ONE);
    }

    /// Sanity-check that `zkgpu-ntt`'s field-generic CPU reference accepts
    /// `KoalaBear` unchanged. This is the real "does the abstraction hold"
    /// test for Phase 2 field expansion — if this stops compiling or stops
    /// passing, the `GpuField` contract has leaked field-specific assumptions
    /// somewhere upstream.
    #[test]
    fn cpu_reference_ntt_roundtrip() {
        use zkgpu_core::NttDirection;
        use zkgpu_ntt::ntt_cpu_reference;

        // log_n = 10 stays well inside TWO_ADICITY=24 and runs fast.
        let log_n = 10u32;
        let n = 1usize << log_n;
        let original: Vec<KoalaBear> =
            (0..n as u32).map(KoalaBear::new).collect();

        let mut data = original.clone();
        ntt_cpu_reference::<KoalaBear>(&mut data, NttDirection::Forward);
        ntt_cpu_reference::<KoalaBear>(&mut data, NttDirection::Inverse);

        assert_eq!(
            data, original,
            "forward+inverse NTT over KoalaBear did not round-trip"
        );
    }
}
