use bytemuck::Pod;

/// A prime field that can be represented on the GPU as a fixed-width unsigned integer.
///
/// Implementors provide the field modulus, primitive root of unity, and GPU-friendly
/// byte layout. The GPU kernels operate on the raw `Repr` and perform modular arithmetic
/// in shader code using the constants exposed here.
pub trait GpuField: Pod + Copy + Clone + Send + Sync + 'static {
    /// The GPU-side representation type (e.g., `u32` for 31-bit fields).
    type Repr: Pod + Copy + Send + Sync;

    /// Field modulus.
    const MODULUS: Self::Repr;

    /// Number of bits in the modulus.
    const MODULUS_BITS: u32;

    /// Multiplicative identity.
    const ONE: Self;

    /// Additive identity.
    const ZERO: Self;

    /// A primitive root of unity of order `2^TWO_ADICITY`.
    const TWO_ADIC_ROOT: Self;

    /// The largest `k` such that `2^k` divides `MODULUS - 1`.
    const TWO_ADICITY: u32;

    /// Convert from the internal representation to this field element.
    fn from_repr(repr: Self::Repr) -> Self;

    /// Convert this field element to its GPU representation.
    fn to_repr(self) -> Self::Repr;

    /// Return the root of unity of order `2^log_n`.
    ///
    /// Panics if `log_n > TWO_ADICITY`.
    fn root_of_unity(log_n: u32) -> Self;

    fn add(self, rhs: Self) -> Self;

    fn sub(self, rhs: Self) -> Self;

    fn mul(self, rhs: Self) -> Self;

    fn pow(self, exp: u64) -> Self;

    /// Modular inverse. Returns `None` for zero.
    fn inverse(self) -> Option<Self>;
}
