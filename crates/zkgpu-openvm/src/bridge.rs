//! Plonky3-0.4.1 → zkgpu Poseidon2 constants bridge.
//!
//! Parallel to `zkgpu-plonky3::poseidon2_bridge` but pinned to the
//! Plonky3 version OpenVM consumes (`=0.4.1`). The math is identical
//! — Plonky3's BabyBear Poseidon2 round constants and the `1 +
//! Diag(V)` internal-layer matrix haven't changed between 0.4.1 and
//! 0.5.x. Only the Plonky3 dependency version differs between the
//! two bridge modules, which is why we duplicate the ~80 lines here
//! rather than share code across the adapter crates.
//!
//! Provides:
//! * [`babybear_openvm_params`] — build a
//!   `Poseidon2Params<ZkgpuBabyBear, 16>` from Plonky3 0.4.1's
//!   `ExternalLayerConstants<P3BabyBear, 16>` + internal constants.
//! * [`p3_to_zkgpu`], [`zkgpu_to_p3`] — scalar field conversion.
//! * [`p3_array_to_zkgpu`], [`zkgpu_array_to_p3`] — array variants.

use p3_baby_bear::BabyBear as P3BabyBear;
use p3_field::PrimeField32;
use p3_poseidon2::ExternalLayerConstants;

use zkgpu_babybear::BabyBear as ZkgpuBabyBear;
use zkgpu_core::GpuField;
use zkgpu_poseidon2::{M4Variant, Poseidon2Params};

// --- V16 diagonal (Plonky3 BabyBear, width 16) ---
//
// Source: `plonky3/baby-bear/src/poseidon2.rs`:
//   `[-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/4, 1/8,
//      1/2^27, -1/2^8, -1/16, -1/2^27]`
//
// We store the zkgpu-side diagonal as `d = V + 1` because
// `zkgpu-poseidon2::mul_internal` uses the `y[i] = sum + (d[i]-1)*x[i]`
// form, which is algebraically equivalent to Plonky3's `1 + Diag(V)`
// when we store `d = V + 1`.

/// Descriptor form for each V entry. Resolved to a concrete
/// BabyBear field element by [`VSpec::resolve`] at build time.
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

const BABYBEAR_V16_SPEC: [VSpec; 16] = [
    VSpec::NegInt(2),
    VSpec::PosInt(1),
    VSpec::PosInt(2),
    VSpec::PosInvPow2(1),  //  1/2
    VSpec::PosInt(3),
    VSpec::PosInt(4),
    VSpec::NegInvPow2(1),  // -1/2
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

fn babybear_v16() -> [ZkgpuBabyBear; 16] {
    let mut out = [ZkgpuBabyBear::ZERO; 16];
    for (i, spec) in BABYBEAR_V16_SPEC.iter().enumerate() {
        out[i] = spec.resolve();
    }
    out
}

// --- Field conversion helpers ---

/// Convert a Plonky3 `BabyBear` (Monty form) to a zkgpu `BabyBear`
/// (canonical `[0, P)` form).
pub fn p3_to_zkgpu(x: P3BabyBear) -> ZkgpuBabyBear {
    ZkgpuBabyBear(x.as_canonical_u32())
}

/// Convert a zkgpu `BabyBear` (canonical) to a Plonky3 `BabyBear`
/// (Monty form).
pub fn zkgpu_to_p3(x: ZkgpuBabyBear) -> P3BabyBear {
    P3BabyBear::new(x.0)
}

/// Convert a `[P3BabyBear; W]` to `[ZkgpuBabyBear; W]`.
pub fn p3_array_to_zkgpu<const W: usize>(src: &[P3BabyBear; W]) -> [ZkgpuBabyBear; W] {
    let mut out = [ZkgpuBabyBear::ZERO; W];
    for i in 0..W {
        out[i] = p3_to_zkgpu(src[i]);
    }
    out
}

/// Convert a `[ZkgpuBabyBear; W]` to `[P3BabyBear; W]`.
pub fn zkgpu_array_to_p3<const W: usize>(src: &[ZkgpuBabyBear; W]) -> [P3BabyBear; W] {
    let mut out = [P3BabyBear::new(0); W];
    for i in 0..W {
        out[i] = zkgpu_to_p3(src[i]);
    }
    out
}

// --- Params construction ---

/// Build a zkgpu [`Poseidon2Params`] matching OpenVM's BabyBear
/// Poseidon2 MMCS config at width 16 (`RATE = 8`, `DIGEST = 8`).
///
/// `ExternalLayerConstants` and the internal constants slice are
/// taken at Plonky3 version 0.4.1 — this crate's pin. Caller
/// typically generates them once and hands the same values to both
/// `Poseidon2BabyBear::new(ext, int)` (CPU side) and this function
/// (GPU side); the two permutations then produce identical output
/// for any input.
///
/// Fixed parameters per `plonky3/baby-bear/src/poseidon2.rs`:
/// * α = 7
/// * `rounds_f_half = 4`
/// * `rounds_p = 13` (W=16)
/// * V diagonal from [`BABYBEAR_V16_SPEC`]
/// * `M4Variant::Plonky3` (circ(2, 3, 1, 1))
pub fn babybear_openvm_params(
    external_constants: &ExternalLayerConstants<P3BabyBear, 16>,
    internal_constants: &[P3BabyBear],
) -> Poseidon2Params<ZkgpuBabyBear, 16> {
    let rounds_f_half = 4;
    let rounds_p = 13;

    let initial = external_constants.get_initial_constants();
    let terminal = external_constants.get_terminal_constants();
    assert_eq!(initial.len(), rounds_f_half, "initial constants count");
    assert_eq!(terminal.len(), rounds_f_half, "terminal constants count");
    let mut zkgpu_external: Vec<[ZkgpuBabyBear; 16]> =
        Vec::with_capacity(2 * rounds_f_half);
    for row in initial {
        zkgpu_external.push(p3_array_to_zkgpu::<16>(row));
    }
    for row in terminal {
        zkgpu_external.push(p3_array_to_zkgpu::<16>(row));
    }

    assert_eq!(internal_constants.len(), rounds_p, "internal constants count");
    let zkgpu_internal: Vec<ZkgpuBabyBear> =
        internal_constants.iter().copied().map(p3_to_zkgpu).collect();

    // Store `d = V + 1` per zkgpu-poseidon2's mul_internal convention.
    let v = babybear_v16();
    let one = ZkgpuBabyBear::from_u64(1);
    let mut diagonal = [ZkgpuBabyBear::ZERO; 16];
    for i in 0..16 {
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
