//! Poseidon2 permutation (field-parametric reference implementation).
//!
//! This crate is the first non-NTT primitive in `zkgpu`. Its purpose is
//! explicitly to pressure-test [`zkgpu_core::GpuField`]: does the same
//! contract that backs NTT machinery also carry a hash-oriented kernel,
//! or does the trait quietly bake in NTT-specific assumptions?
//!
//! ## Scope
//!
//! **Shipped.** Correct Poseidon2 *permutation algebra* over any
//! [`GpuField`]:
//!
//! - S-box `x → x^α` (α coprime to `p-1`)
//! - External layer `M_E ⊗ M_4` with `M_4 = circ(2,1,1,1)` and
//!   `M_E(i,j) = 2·M_4 if i==j else M_4`
//! - Internal layer `1 + D` with a field-specific diagonal `D`
//! - Canonical round structure: `rounds_f_start × external` → `rounds_p × internal`
//!   → `rounds_f_end × external`
//! - Two instances: BabyBear (α=7, rounds 8/13/8) and KoalaBear (α=3, rounds 8/20/8)
//!   at width=16
//!
//! **Deliberately deferred.** *Plonky3-interoperable round constants.*
//! Plonky3's Poseidon2 constants are specific Grain-LFSR outputs tuned for
//! security; transcribing them correctly for BabyBear and KoalaBear is
//! ~160 field elements per instance and carries no architectural value.
//! This crate uses a deterministic placeholder constant generator
//! (`PlaceholderConstants`) that is clearly *not* cryptographically
//! chosen — it's seeded from a per-(instance, round) counter. Plonky3
//! interop is a future feature, not an abstraction question. See
//! [`PlaceholderConstants`] for the exact generator.
//!
//! **Not in scope for this crate.** Sponge construction, Merkle-tree
//! compression, GPU kernel. Those are separate primitives that build on
//! the permutation.
//!
//! ## Why width = 16
//!
//! The Poseidon2 paper and Plonky3 both pick width 16 for 31-bit fields:
//! it gives 128-bit security at rounds_f=8, rounds_p=13 for α=7 / rounds_p=20
//! for α=3. Narrower widths trade proof-cost for round-count and aren't
//! more useful for the abstractions test, so we start there.
//!
//! ## Using it
//!
//! ```rust,ignore
//! use zkgpu_poseidon2::{Poseidon2, Poseidon2Params};
//! use zkgpu_babybear::BabyBear;
//!
//! let params: Poseidon2Params<BabyBear, 16> = Poseidon2Params::babybear_default();
//! let perm = Poseidon2::new(params);
//!
//! let mut state = [BabyBear::ZERO; 16];
//! state[0] = BabyBear::new(1);
//! perm.permute(&mut state);
//! ```

use zkgpu_core::GpuField;

/// Standard Poseidon2 state width for 31-bit fields.
pub const WIDTH: usize = 16;

/// Block size of the external MDS matrix `M_4`.
pub const M4_WIDTH: usize = 4;

/// Number of `M_4` blocks composing the width-16 external layer.
const NUM_BLOCKS: usize = WIDTH / M4_WIDTH;

/// Full-width round parameters for Poseidon2 at a fixed `WIDTH`.
///
/// Field-agnostic in shape, field-specific in data. See
/// [`Poseidon2Params::babybear_default`] / [`Poseidon2Params::koalabear_default`]
/// for constructors that supply the placeholder constants described in the
/// crate docs.
#[derive(Debug, Clone)]
pub struct Poseidon2Params<F: GpuField, const W: usize> {
    /// S-box exponent. Must be coprime to `p - 1`. For BabyBear use `7`;
    /// for KoalaBear use `3`.
    pub alpha: u64,

    /// Number of external (full S-box) rounds applied *before* the
    /// internal rounds. Plonky3 uses 4 on each side for 31-bit fields,
    /// but the original Poseidon2 paper's canonical parameters are 4
    /// before + 4 after = 8 external; this crate follows that
    /// convention via `rounds_f_half = 4`.
    pub rounds_f_half: usize,

    /// Number of internal (single-position S-box) rounds between the
    /// two external halves. 13 for α=7 (BabyBear); 20 for α=3 (KoalaBear).
    pub rounds_p: usize,

    /// External-round constants, indexed as
    /// `external_constants[round_index][position]`. Length is
    /// `2 * rounds_f_half`. First half applied before internal rounds,
    /// second half applied after.
    pub external_constants: Vec<[F; W]>,

    /// Internal-round constants, applied only at `state[0]`. Length is
    /// `rounds_p`.
    pub internal_constants: Vec<F>,

    /// Diagonal of the internal-layer matrix `M_int = 1 + D`. Plonky3
    /// picks small-integer entries per field; this crate uses
    /// `diagonal[i] = F::from_repr(i + 1)` from the placeholder
    /// generator.
    pub internal_diagonal: [F; W],
}

/// Deterministic placeholder constant generator.
///
/// **This is explicitly not the Plonky3 constants.** Its sole purpose
/// is to ship a complete, reproducible, field-parametric Poseidon2
/// permutation without transcribing security-tuned constants from
/// another repository. Every constant is `F::from_repr(counter)` where
/// `counter` is a dense sequential index starting at 1, reduced mod `p`.
///
/// For a secure Plonky3-compatible Poseidon2, depend on a future
/// `zkgpu-poseidon2-plonky3` bridge crate (to be added when the
/// proof-system adapter needs it) — not this generator.
pub struct PlaceholderConstants;

impl PlaceholderConstants {
    /// Produce an external-round constants array of length
    /// `2 * rounds_f_half`, filling each `[F; W]` from sequential
    /// counter values.
    pub fn external<F: GpuField<Repr = u32>, const W: usize>(
        rounds_f_half: usize,
        seed: u32,
    ) -> Vec<[F; W]> {
        let mut counter: u32 = seed.saturating_add(1);
        let mut out = Vec::with_capacity(2 * rounds_f_half);
        for _ in 0..(2 * rounds_f_half) {
            let mut row = [F::ZERO; W];
            for slot in row.iter_mut() {
                let val = counter % F::MODULUS;
                *slot = F::from_repr(val);
                counter = counter.wrapping_add(1);
            }
            out.push(row);
        }
        out
    }

    /// Internal-round constants (length `rounds_p`), seeded to follow
    /// the external constants in the counter sequence so both halves
    /// of the permutation stay distinct-per-round.
    pub fn internal<F: GpuField<Repr = u32>>(
        rounds_p: usize,
        seed: u32,
    ) -> Vec<F> {
        let mut counter: u32 = seed.saturating_add(1);
        let mut out = Vec::with_capacity(rounds_p);
        for _ in 0..rounds_p {
            let val = counter % F::MODULUS;
            out.push(F::from_repr(val));
            counter = counter.wrapping_add(1);
        }
        out
    }

    /// Internal diagonal: `diagonal[i] = F::from_repr(i + 1)`.
    pub fn diagonal<F: GpuField<Repr = u32>, const W: usize>() -> [F; W] {
        let mut out = [F::ZERO; W];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = F::from_repr((i as u32 + 1) % F::MODULUS);
        }
        out
    }
}

impl<F: GpuField<Repr = u32>, const W: usize> Poseidon2Params<F, W> {
    /// Construct params with all fields supplied explicitly. Useful for
    /// plugging in external (e.g. Plonky3) constants later.
    pub fn new(
        alpha: u64,
        rounds_f_half: usize,
        rounds_p: usize,
        external_constants: Vec<[F; W]>,
        internal_constants: Vec<F>,
        internal_diagonal: [F; W],
    ) -> Self {
        assert_eq!(
            external_constants.len(),
            2 * rounds_f_half,
            "external_constants length must equal 2 * rounds_f_half"
        );
        assert_eq!(
            internal_constants.len(),
            rounds_p,
            "internal_constants length must equal rounds_p"
        );
        Self {
            alpha,
            rounds_f_half,
            rounds_p,
            external_constants,
            internal_constants,
            internal_diagonal,
        }
    }
}

impl<F: GpuField<Repr = u32>> Poseidon2Params<F, WIDTH> {
    /// BabyBear defaults: α = 7, rounds 4 / 13 / 4 (external / internal /
    /// external), placeholder constants.
    pub fn babybear_default() -> Self {
        let rounds_f_half = 4;
        let rounds_p = 13;
        Self::new(
            7,
            rounds_f_half,
            rounds_p,
            PlaceholderConstants::external::<F, WIDTH>(rounds_f_half, 0),
            PlaceholderConstants::internal::<F>(rounds_p, 1_000),
            PlaceholderConstants::diagonal::<F, WIDTH>(),
        )
    }

    /// KoalaBear defaults: α = 3, rounds 4 / 20 / 4, placeholder constants.
    ///
    /// α=3 requires more internal rounds than α=7 to hit the same
    /// security level (Plonky3 uses 20). This crate follows that count
    /// so adding the Plonky3 constants later doesn't force a round-count
    /// change.
    pub fn koalabear_default() -> Self {
        let rounds_f_half = 4;
        let rounds_p = 20;
        Self::new(
            3,
            rounds_f_half,
            rounds_p,
            PlaceholderConstants::external::<F, WIDTH>(rounds_f_half, 0),
            PlaceholderConstants::internal::<F>(rounds_p, 1_000),
            PlaceholderConstants::diagonal::<F, WIDTH>(),
        )
    }
}

/// Poseidon2 permutation instance. Wraps [`Poseidon2Params`] and exposes
/// [`Poseidon2::permute`].
#[derive(Debug, Clone)]
pub struct Poseidon2<F: GpuField, const W: usize> {
    params: Poseidon2Params<F, W>,
}

impl<F: GpuField<Repr = u32>, const W: usize> Poseidon2<F, W> {
    pub fn new(params: Poseidon2Params<F, W>) -> Self {
        Self { params }
    }

    pub fn params(&self) -> &Poseidon2Params<F, W> {
        &self.params
    }

    /// Add the external-round constants for round `round_idx` into
    /// `state`.
    fn add_external_rc(&self, state: &mut [F; W], round_idx: usize) {
        let rc = &self.params.external_constants[round_idx];
        for (s, c) in state.iter_mut().zip(rc.iter()) {
            *s = s.add(*c);
        }
    }

    /// Apply the full S-box `x → x^α` to every state position.
    fn sbox_full(&self, state: &mut [F; W]) {
        for s in state.iter_mut() {
            *s = s.pow(self.params.alpha);
        }
    }

    /// Apply the S-box only to `state[0]`.
    fn sbox_single(&self, state: &mut [F; W]) {
        state[0] = state[0].pow(self.params.alpha);
    }

    /// External matrix `M_E ⊗ M_4` where `M_4 = circ(2,1,1,1)` and
    /// `M_E(i,j) = 2·M_4 if i == j else M_4`.
    ///
    /// Implemented in two passes:
    /// 1. Apply `M_4` to each 4-element block independently.
    /// 2. Mix blocks: for each position `j ∈ 0..4`, each block receives
    ///    the sum of all blocks' `j`-th positions plus itself (the
    ///    diagonal doubling).
    fn mul_external(&self, state: &mut [F; W]) {
        assert_eq!(W, WIDTH, "external matrix is specialised to W=16");

        // Step 1: M_4 per block.
        let mut blocks: [[F; M4_WIDTH]; NUM_BLOCKS] =
            [[F::ZERO; M4_WIDTH]; NUM_BLOCKS];
        for (b, block) in blocks.iter_mut().enumerate() {
            for i in 0..M4_WIDTH {
                block[i] = state[b * M4_WIDTH + i];
            }
            *block = m4_times(*block);
        }

        // Step 2: cross-block sum + diagonal doubling.
        // For each column j, sum_all[j] = Σ_b blocks[b][j].
        let mut sum_all = [F::ZERO; M4_WIDTH];
        for block in blocks.iter() {
            for j in 0..M4_WIDTH {
                sum_all[j] = sum_all[j].add(block[j]);
            }
        }
        // Each block's j-th slot becomes blocks[b][j] + sum_all[j].
        // That implements M_E(i,i) = 2·M_4 and M_E(i,j) = M_4 — the
        // doubled-on-diagonal block matrix.
        for (b, block) in blocks.iter().enumerate() {
            for j in 0..M4_WIDTH {
                state[b * M4_WIDTH + j] = block[j].add(sum_all[j]);
            }
        }
    }

    /// Internal matrix `M_int = 1 + D` where `D = diag(d_0, ..., d_{W-1})`.
    ///
    /// `M_int * state = sum_broadcast + D·state`, where
    /// `sum_broadcast[i] = Σ_j state[j]` for all i.
    fn mul_internal(&self, state: &mut [F; W]) {
        // Compute Σ state[j] once.
        let mut sum = F::ZERO;
        for s in state.iter() {
            sum = sum.add(*s);
        }
        // Rewrite state[i] = sum + (d_i - 1) * state[i]
        //                 = sum + d_i*state[i] - state[i].
        // Keeping it as (sum + d_i*state[i]) - state[i] means no
        // subtract-one constant handling in the field layer.
        for (s, d) in state.iter_mut().zip(self.params.internal_diagonal.iter()) {
            let dd = d.mul(*s);
            *s = sum.add(dd).sub(*s);
        }
    }

    /// Run the full permutation in place.
    pub fn permute(&self, state: &mut [F; W]) {
        // Initial external mix (Poseidon2 spec — sets up linear
        // coverage before the first round).
        self.mul_external(state);

        // First half of external rounds.
        for r in 0..self.params.rounds_f_half {
            self.add_external_rc(state, r);
            self.sbox_full(state);
            self.mul_external(state);
        }

        // Internal rounds.
        for r in 0..self.params.rounds_p {
            let c = self.params.internal_constants[r];
            state[0] = state[0].add(c);
            self.sbox_single(state);
            self.mul_internal(state);
        }

        // Second half of external rounds.
        for r in 0..self.params.rounds_f_half {
            let idx = self.params.rounds_f_half + r;
            self.add_external_rc(state, idx);
            self.sbox_full(state);
            self.mul_external(state);
        }
    }
}

/// `M_4 = circ(2, 1, 1, 1)` applied to a 4-element block.
///
/// Explicit form (expressed without a subtraction so both BabyBear and
/// KoalaBear's `add` suffice):
/// ```text
/// y[0] = 2*x[0] + x[1] + x[2] + x[3]
/// y[1] = x[0] + 2*x[1] + x[2] + x[3]
/// y[2] = x[0] + x[1] + 2*x[2] + x[3]
/// y[3] = x[0] + x[1] + x[2] + 2*x[3]
/// ```
fn m4_times<F: GpuField>(x: [F; M4_WIDTH]) -> [F; M4_WIDTH] {
    let sum = x[0].add(x[1]).add(x[2]).add(x[3]);
    [
        sum.add(x[0]),
        sum.add(x[1]),
        sum.add(x[2]),
        sum.add(x[3]),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generic-over-F smoke test. The whole point of this crate is that
    /// the permutation compiles and runs against any `GpuField<Repr = u32>`.
    /// If either assertion fails on a new field, `GpuField` has picked up
    /// a field-specific assumption somewhere.
    fn permutation_smoke<F: GpuField<Repr = u32>>(params: Poseidon2Params<F, WIDTH>) {
        let perm = Poseidon2::new(params);

        // 1. Deterministic: same input → same output.
        let mut a = [F::ZERO; WIDTH];
        a[0] = F::from_repr(1);
        let mut b = a;
        perm.permute(&mut a);
        perm.permute(&mut b);
        assert_eq!(a.map(|f| f.to_repr()), b.map(|f| f.to_repr()));

        // 2. Non-trivial: permutation must move state away from identity
        //    for a non-zero input.
        let mut state = [F::ZERO; WIDTH];
        state[0] = F::from_repr(1);
        let original = state;
        perm.permute(&mut state);
        assert_ne!(
            state.map(|f| f.to_repr()),
            original.map(|f| f.to_repr()),
            "permutation left non-zero state unchanged"
        );

        // 3. Injectivity sample: two distinct inputs → distinct outputs.
        let mut s1 = [F::ZERO; WIDTH];
        s1[0] = F::from_repr(1);
        let mut s2 = [F::ZERO; WIDTH];
        s2[0] = F::from_repr(2);
        perm.permute(&mut s1);
        perm.permute(&mut s2);
        assert_ne!(
            s1.map(|f| f.to_repr()),
            s2.map(|f| f.to_repr()),
            "distinct inputs collided — permutation appears non-injective"
        );
    }

    #[test]
    fn babybear_permutation_smoke() {
        use zkgpu_babybear::BabyBear;
        permutation_smoke::<BabyBear>(Poseidon2Params::babybear_default());
    }

    #[test]
    fn koalabear_permutation_smoke() {
        use zkgpu_koalabear::KoalaBear;
        permutation_smoke::<KoalaBear>(Poseidon2Params::koalabear_default());
    }

    #[test]
    fn m4_times_linearity() {
        use zkgpu_babybear::BabyBear;
        // M_4 is linear, so M_4(x + y) should equal M_4(x) + M_4(y).
        let x: [BabyBear; 4] = [
            BabyBear::new(11),
            BabyBear::new(22),
            BabyBear::new(33),
            BabyBear::new(44),
        ];
        let y: [BabyBear; 4] = [
            BabyBear::new(5),
            BabyBear::new(7),
            BabyBear::new(11),
            BabyBear::new(13),
        ];
        let mut xy = [BabyBear::ZERO; 4];
        for i in 0..4 {
            xy[i] = x[i].add(y[i]);
        }
        let mx = m4_times(x);
        let my = m4_times(y);
        let mxy = m4_times(xy);
        for i in 0..4 {
            assert_eq!(
                mxy[i].to_repr(),
                mx[i].add(my[i]).to_repr(),
                "M_4 not linear at position {i}"
            );
        }
    }

    /// Regression guard: pin the output of BabyBear Poseidon2 on a fixed
    /// input so later refactors can't silently change the permutation.
    /// Computed once by running this test with all zeros in the
    /// `expected` array and then transcribing the failure message.
    #[test]
    fn babybear_regression_state_0001() {
        use zkgpu_babybear::BabyBear;
        let perm =
            Poseidon2::new(Poseidon2Params::<BabyBear, WIDTH>::babybear_default());
        let mut state = [BabyBear::ZERO; WIDTH];
        state[0] = BabyBear::new(1);
        perm.permute(&mut state);

        // Transcribed from a passing run on 2026-04-18. Any change here
        // means the placeholder constants or the permutation algebra
        // changed — intentional or otherwise. If this crate starts
        // consuming Plonky3 constants, replace these with values derived
        // from a Plonky3 reference run.
        let expected: [u32; WIDTH] = [
            1646996371, 1788689999, 602438123, 1506086531,
            748277907, 1860416619, 1005521241, 522487477,
            1853726457, 740563310, 1495084457, 816004387,
            268492728, 1545584133, 820438449, 558558427,
        ];
        let got: [u32; WIDTH] = state.map(|f| f.to_repr());
        assert_eq!(got, expected, "BabyBear Poseidon2 regression");
    }
}
