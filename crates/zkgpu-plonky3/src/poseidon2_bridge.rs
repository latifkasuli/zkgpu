//! Phase 7 Step 1.5a — Plonky3 Poseidon2 compatibility bridge.
//!
//! This module bridges between Plonky3's canonical Poseidon2 layout
//! and zkgpu-poseidon2's generic Poseidon2 permutation so that the
//! latter can produce bit-identical output to `p3_baby_bear::Poseidon2BabyBear`
//! at widths 16 and 24.
//!
//! # What Plonky3 does that we need to match
//!
//! 1. **4×4 MDS matrix**: `circ(2, 3, 1, 1)` (Plonky3's
//!    `apply_mat4`), not the simpler `circ(2, 1, 1, 1)` that
//!    zkgpu-poseidon2 originally shipped. Selected via
//!    [`zkgpu_poseidon2::M4Variant::Plonky3`].
//!
//! 2. **Internal-layer matrix**: `1 + Diag(V)` where `1` is the
//!    all-ones matrix and `V` is a specific vector of small integers
//!    and inverse powers of 2 chosen for performance on Monty31
//!    fields.
//!
//!    Our `mul_internal` uses the equivalent `y[i] = sum + (d[i]-1) * x[i]`
//!    form, so the bridge stores `d[i] = V[i] + 1` in
//!    `internal_diagonal`.
//!
//! 3. **Round constants**: Plonky3 generates them deterministically
//!    via `ExternalLayerConstants::new_from_rng` and
//!    `rng.sample_iter`. We **don't** re-derive these — the bridge
//!    accepts them as inputs and converts field representations.
//!
//! 4. **Field conversion**: `p3_baby_bear::BabyBear` is `MontyField31`
//!    (Montgomery form); `zkgpu_babybear::BabyBear` is canonical
//!    `[0, P)`. Conversion via `PrimeField32::as_canonical_u32` and
//!    `MontyField31::new`, same pattern used throughout the adapter.
//!
//! # Typical use
//!
//! See the `poseidon2_bridge_*` tests in
//! `tests/poseidon2_bridge_differential.rs` for the canonical
//! "build both permutations with identical constants and prove
//! output parity" pattern.

use p3_baby_bear::BabyBear as P3BabyBear;
use p3_field::PrimeField32;
use p3_poseidon2::ExternalLayerConstants;

use zkgpu_babybear::BabyBear as ZkgpuBabyBear;
use zkgpu_core::GpuField;
use zkgpu_poseidon2::{M4Variant, Poseidon2Params};

// -- BabyBear internal diagonals (Plonky3 canonical, V values) ---------------
//
// Source: `plonky3/baby-bear/src/poseidon2.rs` header comment. These
// are Plonky3's `V` values in the `1 + Diag(V)` convention. We store
// the zkgpu-side diagonal as `V + 1` per `mul_internal`'s convention
// (see the module-level doc above).
//
// All values are BabyBear field elements. Inverse powers of 2 are
// computed at construction time via the field's inv() helper.

/// Plonky3's canonical `V` diagonal for BabyBear width-16 Poseidon2.
///
/// From `plonky3/baby-bear/src/poseidon2.rs`:
///   `[-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/4, 1/8, 1/2^27, -1/2^8, -1/16, -1/2^27]`
///
/// Encoded as signed integers and inverse power-of-2 descriptors;
/// resolved to concrete field elements by [`babybear_v16`].
const BABYBEAR_V16_SPEC: [VSpec; 16] = [
    VSpec::NegInt(2),
    VSpec::PosInt(1),
    VSpec::PosInt(2),
    VSpec::PosInvPow2(1), //  1/2
    VSpec::PosInt(3),
    VSpec::PosInt(4),
    VSpec::NegInvPow2(1), // -1/2
    VSpec::NegInt(3),
    VSpec::NegInt(4),
    VSpec::PosInvPow2(8),  //  1/2^8
    VSpec::PosInvPow2(2),  //  1/4
    VSpec::PosInvPow2(3),  //  1/8
    VSpec::PosInvPow2(27), //  1/2^27
    VSpec::NegInvPow2(8),  // -1/2^8
    VSpec::NegInvPow2(4),  // -1/16
    VSpec::NegInvPow2(27), // -1/2^27
];

/// Plonky3's canonical `V` diagonal for BabyBear width-24 Poseidon2.
///
/// From `plonky3/baby-bear/src/poseidon2.rs`:
///   `[-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/4, 1/8, 1/16, 1/2^7, 1/2^9, 1/2^27,
///     -1/2^8, -1/4, -1/8, -1/16, -1/32, -1/64, -1/2^7, -1/2^27]`
const BABYBEAR_V24_SPEC: [VSpec; 24] = [
    VSpec::NegInt(2),
    VSpec::PosInt(1),
    VSpec::PosInt(2),
    VSpec::PosInvPow2(1),
    VSpec::PosInt(3),
    VSpec::PosInt(4),
    VSpec::NegInvPow2(1),
    VSpec::NegInt(3),
    VSpec::NegInt(4),
    VSpec::PosInvPow2(8),
    VSpec::PosInvPow2(2),
    VSpec::PosInvPow2(3),
    VSpec::PosInvPow2(4),
    VSpec::PosInvPow2(7),
    VSpec::PosInvPow2(9),
    VSpec::PosInvPow2(27),
    VSpec::NegInvPow2(8),
    VSpec::NegInvPow2(2),
    VSpec::NegInvPow2(3),
    VSpec::NegInvPow2(4),
    VSpec::NegInvPow2(5),
    VSpec::NegInvPow2(6),
    VSpec::NegInvPow2(7),
    VSpec::NegInvPow2(27),
];

/// Descriptor for a single `V` entry. Resolved to a field element via
/// `resolve` at runtime.
#[derive(Debug, Clone, Copy)]
enum VSpec {
    PosInt(u32),
    NegInt(u32),
    PosInvPow2(u32),
    NegInvPow2(u32),
}

impl VSpec {
    fn resolve(self) -> ZkgpuBabyBear {
        let zero = ZkgpuBabyBear::ZERO;
        let one = ZkgpuBabyBear::from_u64(1);
        let two = ZkgpuBabyBear::from_u64(2);
        match self {
            VSpec::PosInt(k) => ZkgpuBabyBear::from_u64(k as u64),
            VSpec::NegInt(k) => zero.sub(ZkgpuBabyBear::from_u64(k as u64)),
            VSpec::PosInvPow2(k) => {
                // 1 / 2^k — compute 2^k, invert.
                let mut p = one;
                for _ in 0..k {
                    p = p.mul(two);
                }
                p.inverse().expect("2^k is nonzero in BabyBear")
            }
            VSpec::NegInvPow2(k) => {
                let mut p = one;
                for _ in 0..k {
                    p = p.mul(two);
                }
                zero.sub(p.inverse().expect("2^k is nonzero in BabyBear"))
            }
        }
    }
}

/// Resolve Plonky3's width-16 `V` diagonal into zkgpu field elements.
///
/// Returns `V` (Plonky3's convention); call the bridge's
/// `to_zkgpu_diagonal` helper if you need `d = V + 1` for zkgpu
/// `Poseidon2Params::internal_diagonal`.
pub fn babybear_v16() -> [ZkgpuBabyBear; 16] {
    let mut out = [ZkgpuBabyBear::ZERO; 16];
    for (i, spec) in BABYBEAR_V16_SPEC.iter().enumerate() {
        out[i] = spec.resolve();
    }
    out
}

/// Resolve Plonky3's width-24 `V` diagonal into zkgpu field elements.
pub fn babybear_v24() -> [ZkgpuBabyBear; 24] {
    let mut out = [ZkgpuBabyBear::ZERO; 24];
    for (i, spec) in BABYBEAR_V24_SPEC.iter().enumerate() {
        out[i] = spec.resolve();
    }
    out
}

// -- Field conversion helpers ------------------------------------------------

/// Convert a `p3 BabyBear` (Monty form) to a `zkgpu BabyBear` (canonical).
pub fn p3_to_zkgpu(x: P3BabyBear) -> ZkgpuBabyBear {
    ZkgpuBabyBear(x.as_canonical_u32())
}

/// Convert a `zkgpu BabyBear` (canonical) to a `p3 BabyBear` (Monty form).
pub fn zkgpu_to_p3(x: ZkgpuBabyBear) -> P3BabyBear {
    P3BabyBear::new(x.0)
}

/// Convert a `[P3BabyBear; W]` slice to a `[ZkgpuBabyBear; W]`.
pub fn p3_array_to_zkgpu<const W: usize>(src: &[P3BabyBear; W]) -> [ZkgpuBabyBear; W] {
    let mut out = [ZkgpuBabyBear::ZERO; W];
    for i in 0..W {
        out[i] = p3_to_zkgpu(src[i]);
    }
    out
}

/// Convert a `[ZkgpuBabyBear; W]` slice to a `[P3BabyBear; W]`.
pub fn zkgpu_array_to_p3<const W: usize>(src: &[ZkgpuBabyBear; W]) -> [P3BabyBear; W] {
    let mut out = [P3BabyBear::new(0); W];
    for i in 0..W {
        out[i] = zkgpu_to_p3(src[i]);
    }
    out
}

// -- Params construction -----------------------------------------------------

/// Build a zkgpu [`Poseidon2Params`] matching Plonky3's BabyBear
/// Poseidon2 configuration at width `W` ∈ {16, 24}, with the given
/// external + internal round constants (expressed in p3 field elements).
///
/// Caller provides the constants — they come from Plonky3's
/// `ExternalLayerConstants::new` and a `Vec<P3BabyBear>` for the
/// internal constants. Typical pattern is to generate once, pass to
/// both `p3_poseidon2::Poseidon2::new` and this function, then the
/// two permutations produce identical output for any input.
///
/// Width-specific parameters baked in per `plonky3/baby-bear/src/poseidon2.rs`:
///
/// - W=16: α = 7, `rounds_f_half = 4`, `rounds_p = 13`, V from `BABYBEAR_V16_SPEC`.
/// - W=24: α = 7, `rounds_f_half = 4`, `rounds_p = 21`, V from `BABYBEAR_V24_SPEC`.
pub fn babybear_plonky3_params<const W: usize>(
    external_constants: &ExternalLayerConstants<P3BabyBear, W>,
    internal_constants: &[P3BabyBear],
) -> Poseidon2Params<ZkgpuBabyBear, W> {
    let rounds_f_half = 4;
    let rounds_p = plonky3_rounds_p_babybear::<W>();

    // External constants: Plonky3 splits into initial + terminal; our
    // external_constants is a flat Vec indexed "initial first, terminal
    // second".
    let initial = external_constants.get_initial_constants();
    let terminal = external_constants.get_terminal_constants();
    assert_eq!(initial.len(), rounds_f_half, "initial constants count");
    assert_eq!(terminal.len(), rounds_f_half, "terminal constants count");
    let mut zkgpu_external: Vec<[ZkgpuBabyBear; W]> =
        Vec::with_capacity(2 * rounds_f_half);
    for row in initial {
        zkgpu_external.push(p3_array_to_zkgpu::<W>(row));
    }
    for row in terminal {
        zkgpu_external.push(p3_array_to_zkgpu::<W>(row));
    }

    // Internal constants: one per partial round.
    assert_eq!(internal_constants.len(), rounds_p, "internal constants count");
    let zkgpu_internal: Vec<ZkgpuBabyBear> =
        internal_constants.iter().copied().map(p3_to_zkgpu).collect();

    // Internal diagonal: store d = V + 1 for our mul_internal's
    // `y[i] = sum + (d[i]-1)*x[i]` form.
    let v: [ZkgpuBabyBear; W] = plonky3_v_babybear::<W>();
    let one = ZkgpuBabyBear::from_u64(1);
    let mut diagonal = [ZkgpuBabyBear::ZERO; W];
    for i in 0..W {
        diagonal[i] = v[i].add(one);
    }

    Poseidon2Params::new(
        7,
        rounds_f_half,
        rounds_p,
        zkgpu_external,
        zkgpu_internal,
        diagonal,
    )
    .with_m4_variant(M4Variant::Plonky3)
}

// -- Internal helpers --------------------------------------------------------

fn plonky3_rounds_p_babybear<const W: usize>() -> usize {
    match W {
        16 => 13,
        24 => 21,
        _ => panic!("Plonky3 BabyBear Poseidon2 bridge supports W ∈ {{16, 24}}, got {W}"),
    }
}

fn plonky3_v_babybear<const W: usize>() -> [ZkgpuBabyBear; W] {
    // Width-specific diagonal. Can't use generics nicely here because
    // the arrays are different sizes; dispatch by W.
    let mut out = [ZkgpuBabyBear::ZERO; W];
    match W {
        16 => {
            let v = babybear_v16();
            out[..16].copy_from_slice(&v);
        }
        24 => {
            let v = babybear_v24();
            out[..24].copy_from_slice(&v);
        }
        _ => panic!("Plonky3 BabyBear Poseidon2 bridge supports W ∈ {{16, 24}}, got {W}"),
    }
    out
}
