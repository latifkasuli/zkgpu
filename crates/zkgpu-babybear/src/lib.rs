use std::ops;

use bytemuck::{Pod, Zeroable};
use zkgpu_core::GpuField;

/// The BabyBear prime: p = 2^31 - 2^27 + 1 = 2013265921
pub const P: u32 = 0x78000001;

/// A generator of the multiplicative group of BabyBear.
pub const GENERATOR: u32 = 31;

/// BabyBear has 2-adicity 27: the largest k with 2^k | (p-1).
pub const TWO_ADICITY: u32 = 27;

/// A primitive 2^27-th root of unity in BabyBear.
/// This is GENERATOR^((p-1) / 2^27) = 31^15 mod p.
pub const TWO_ADIC_ROOT_OF_UNITY: u32 = 0x1A427A41; // 440564289

/// A field element in BabyBear, stored as a u32 in [0, P).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Pod, Zeroable)]
#[repr(transparent)]
pub struct BabyBear(pub u32);

impl BabyBear {
    pub const fn new(value: u32) -> Self {
        Self(value % P)
    }

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

    pub fn inv(self) -> Option<Self> {
        if self.0 == 0 {
            return None;
        }
        Some(self.pow(P as u64 - 2))
    }
}

impl ops::Add for BabyBear {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let sum = self.0 as u64 + rhs.0 as u64;
        let reduced = if sum >= P as u64 { sum - P as u64 } else { sum };
        Self(reduced as u32)
    }
}

impl ops::Sub for BabyBear {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        let (diff, borrow) = self.0.overflowing_sub(rhs.0);
        Self(if borrow { diff.wrapping_add(P) } else { diff })
    }
}

impl ops::Mul for BabyBear {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        let prod = self.0 as u64 * rhs.0 as u64;
        Self((prod % P as u64) as u32)
    }
}

impl GpuField for BabyBear {
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
        BabyBear::pow(self, exp)
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

impl std::fmt::Display for BabyBear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modulus_is_prime() {
        assert_eq!(P, (1u32 << 31) - (1u32 << 27) + 1);
        assert_eq!(P, 2013265921);
    }

    #[test]
    fn add_sub_roundtrip() {
        let a = BabyBear::new(1234567);
        let b = BabyBear::new(7654321);
        assert_eq!((a + b) - b, a);
    }

    #[test]
    fn mul_identity() {
        let a = BabyBear::new(999999);
        assert_eq!(a * BabyBear::ONE, a);
        assert_eq!(a * BabyBear::ZERO, BabyBear::ZERO);
    }

    #[test]
    fn inverse_roundtrip() {
        let a = BabyBear::new(42);
        let a_inv = a.inv().unwrap();
        assert_eq!(a * a_inv, BabyBear::ONE);
    }

    #[test]
    fn zero_has_no_inverse() {
        assert!(BabyBear::ZERO.inv().is_none());
    }

    #[test]
    fn root_of_unity_order() {
        for log_n in 1..=TWO_ADICITY {
            let root = BabyBear::root_of_unity(log_n);
            let order = 1u64 << log_n;
            assert_eq!(root.pow(order), BabyBear::ONE, "root^(2^{log_n}) != 1");
            if log_n > 0 {
                assert_ne!(
                    root.pow(order / 2),
                    BabyBear::ONE,
                    "root has smaller order than 2^{log_n}"
                );
            }
        }
    }

    #[test]
    fn generator_order() {
        let g = BabyBear::new(GENERATOR);
        assert_eq!(g.pow(P as u64 - 1), BabyBear::ONE);
    }

    #[test]
    fn reduction_on_construction() {
        let a = BabyBear::new(P);
        assert_eq!(a, BabyBear::ZERO);
        let b = BabyBear::new(P + 1);
        assert_eq!(b, BabyBear::ONE);
    }
}
