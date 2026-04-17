use zkgpu_babybear::{BabyBear, P};
use zkgpu_core::{GpuField, NttDirection};

use super::planner::{FourStepPlanConfig, BLOCK_SIZE, LOG_BLOCK};

/// Shoup precomputed quotient for constant-multiplier modular reduction.
///
/// For a twiddle factor `w`, computes `floor(w * 2^32 / P)`.
/// Used with `mod_mul_shoup` in WGSL kernels to replace the iterative
/// reduction loop with a single multiply + subtract.
///
/// Reference: Harvey (2014), "Faster arithmetic for number-theoretic transforms".
pub(crate) fn shoup_quotient(w: u32) -> u32 {
    ((w as u64 * (1u64 << 32)) / P as u64) as u32
}

/// Twiddles for the global DIF Stockham stages (stages 0..num_global_stages).
#[allow(dead_code)]
pub(crate) fn precompute_stockham_twiddles(
    log_n: u32,
    direction: NttDirection,
    num_stages: u32,
) -> Vec<u32> {
    let n = 1u32 << log_n;
    let omega = BabyBear::root_of_unity(log_n);
    let omega = match direction {
        NttDirection::Forward => omega,
        NttDirection::Inverse => omega.inv().expect("root of unity must be invertible"),
    };

    let mut twiddles = Vec::new();

    for h in 0..num_stages {
        let s = 1u32 << h;
        let m = n >> (h + 1);
        let step = omega.pow(s as u64);

        let mut w = BabyBear::ONE;
        for _p in 0..m {
            twiddles.push(w.to_repr());
            w = w * step;
        }
    }

    twiddles
}

/// Twiddles for radix-4 global DIF Stockham stages.
///
/// Each R4 stage combines two R2 stages (h, h+1). For butterfly position p,
/// stores (w1, w2, w3) = (omega_N^(s*p), w1^2, w1^3) as consecutive triples
/// so the shader can use Shoup's method for all output multiplies.
/// Also computes omega_4 = omega_N^(N/4) and its Shoup quotient.
///
/// Returns `(twiddles, twiddles_prime, omega4, omega4_prime)`.
/// Twiddles are stored as (w1, w2, w3) triples per butterfly position,
/// where w2 = w1^2 and w3 = w1^3. This allows the shader to use
/// `mod_mul_shoup` for all output multiplies without deriving w2/w3.
pub(crate) fn precompute_stockham_r4_twiddles(
    log_n: u32,
    direction: NttDirection,
    r4_stages: &[(u32, u32)], // (h, m4) pairs for each R4 dispatch
) -> (Vec<u32>, Vec<u32>, u32, u32) {
    let omega = BabyBear::root_of_unity(log_n);
    let omega = match direction {
        NttDirection::Forward => omega,
        NttDirection::Inverse => omega.inv().expect("root of unity must be invertible"),
    };

    let n = 1u32 << log_n;
    let omega4 = omega.pow((n / 4) as u64);
    let omega4_repr = omega4.to_repr();
    let omega4_prime = shoup_quotient(omega4_repr);

    let mut twiddles = Vec::new();
    let mut twiddles_prime = Vec::new();

    for &(h, m4) in r4_stages {
        let s = 1u32 << h;
        let step = omega.pow(s as u64);

        let mut w1 = BabyBear::ONE;
        for _p in 0..m4 {
            let w2 = w1 * w1;
            let w3 = w2 * w1;
            let r1 = w1.to_repr();
            let r2 = w2.to_repr();
            let r3 = w3.to_repr();
            twiddles.push(r1);
            twiddles.push(r2);
            twiddles.push(r3);
            twiddles_prime.push(shoup_quotient(r1));
            twiddles_prime.push(shoup_quotient(r2));
            twiddles_prime.push(shoup_quotient(r3));
            w1 = w1 * step;
        }
    }

    (twiddles, twiddles_prime, omega4_repr, omega4_prime)
}

/// Twiddles for radix-8 global DIF Stockham stages (NVIDIA scale-up T3.A, 2026-04-17).
///
/// Each R8 stage combines three logical Stockham stages (h, h+1, h+2).
/// For butterfly position p, stores 7 outer twiddle factors
/// (w^1, w^2, w^3, w^4, w^5, w^6, w^7) where w = omega_N^(s*p), as 7
/// consecutive values so the shader can apply `mod_mul_shoup` to each
/// of outputs 1..7. Output 0 is untwiddled (w^0 = 1).
///
/// Also computes and returns the inner R8 constants: omega_8 = omega_N^(N/8)
/// (primitive 8th root), omega_4 = omega_N^(N/4), and omega_8_cubed
/// (= omega_8 * omega_4), each with its Shoup quotient. These are passed
/// as uniform constants to the shader.
///
/// Returns `(twiddles, twiddles_prime, omega8, omega8_prime, omega4,
///           omega4_prime, omega8_cubed, omega8_cubed_prime)`.
#[allow(clippy::type_complexity)]
pub(crate) fn precompute_stockham_r8_twiddles(
    log_n: u32,
    direction: NttDirection,
    r8_stages: &[(u32, u32)], // (h, m8) pairs for each R8 dispatch
) -> (Vec<u32>, Vec<u32>, u32, u32, u32, u32, u32, u32) {
    let omega = BabyBear::root_of_unity(log_n);
    let omega = match direction {
        NttDirection::Forward => omega,
        NttDirection::Inverse => omega.inv().expect("root of unity must be invertible"),
    };

    let n = 1u32 << log_n;
    let omega8 = omega.pow((n / 8) as u64);
    let omega4 = omega.pow((n / 4) as u64);
    let omega8_cubed = omega8 * omega4; // omega8^3 = omega8 * omega8^2 = omega8 * omega4

    let omega8_repr = omega8.to_repr();
    let omega8_prime = shoup_quotient(omega8_repr);
    let omega4_repr = omega4.to_repr();
    let omega4_prime = shoup_quotient(omega4_repr);
    let omega8_cubed_repr = omega8_cubed.to_repr();
    let omega8_cubed_prime = shoup_quotient(omega8_cubed_repr);

    let mut twiddles = Vec::new();
    let mut twiddles_prime = Vec::new();

    for &(h, m8) in r8_stages {
        let s = 1u32 << h;
        let step = omega.pow(s as u64);

        // For each butterfly position p, emit w^1..w^7.
        let mut w1 = BabyBear::ONE;
        for _p in 0..m8 {
            // w_k = w^k = (omega_N^(s*p))^k for k=1..7
            let w2 = w1 * w1;
            let w3 = w2 * w1;
            let w4 = w3 * w1;
            let w5 = w4 * w1;
            let w6 = w5 * w1;
            let w7 = w6 * w1;
            for w in [w1, w2, w3, w4, w5, w6, w7] {
                let r = w.to_repr();
                twiddles.push(r);
                twiddles_prime.push(shoup_quotient(r));
            }
            w1 = w1 * step; // advance to next p: ω^(s*(p+1)) = ω^(s*p) * ω^s
        }
    }

    (
        twiddles,
        twiddles_prime,
        omega8_repr,
        omega8_prime,
        omega4_repr,
        omega4_prime,
        omega8_cubed_repr,
        omega8_cubed_prime,
    )
}

/// Twiddles for the workgroup-local kernel: a BLOCK_SIZE-point DIF Stockham.
///
/// Independent of N — the local sub-problems always use the BLOCK_SIZE-th
/// root of unity regardless of the overall transform size.
#[allow(dead_code)]
pub(crate) fn precompute_local_twiddles(direction: NttDirection) -> Vec<u32> {
    let omega = BabyBear::root_of_unity(LOG_BLOCK);
    let omega = match direction {
        NttDirection::Forward => omega,
        NttDirection::Inverse => omega.inv().expect("root of unity must be invertible"),
    };

    let mut twiddles = Vec::with_capacity((BLOCK_SIZE - 1) as usize);

    for h in 0..LOG_BLOCK {
        let s = 1u32 << h;
        let m = BLOCK_SIZE >> (h + 1);
        let step = omega.pow(s as u64);

        let mut w = BabyBear::ONE;
        for _p in 0..m {
            twiddles.push(w.to_repr());
            w = w * step;
        }
    }

    twiddles
}

/// Twiddles for the R4 local kernel: 5 R4 stage pairs (10 logical stages).
///
/// Returns `(twiddles, twiddles_prime, omega_4, omega4_prime)`.
/// Twiddles are stored as (w1, w2, w3) triples per butterfly position.
///
/// Layout: [R4 pair 0 (256*3)] [pair 1 (64*3)] [pair 2 (16*3)]
///         [pair 3 (4*3)] [pair 4 (1*3)] = 1023 twiddles total.
pub(crate) fn precompute_local_r4_twiddles(direction: NttDirection) -> (Vec<u32>, Vec<u32>, u32, u32) {
    let omega = BabyBear::root_of_unity(LOG_BLOCK);
    let omega = match direction {
        NttDirection::Forward => omega,
        NttDirection::Inverse => omega.inv().expect("root of unity must be invertible"),
    };

    let omega4 = omega.pow((BLOCK_SIZE / 4) as u64);
    let omega4_repr = omega4.to_repr();
    let omega4_prime = shoup_quotient(omega4_repr);

    let mut twiddles = Vec::new();
    let mut twiddles_prime = Vec::new();

    let num_r4_pairs = LOG_BLOCK / 2;
    for pair_idx in 0..num_r4_pairs {
        let h = pair_idx * 2;
        let s = 1u32 << h;
        let m4 = BLOCK_SIZE / (4 * s);
        let step = omega.pow(s as u64);

        let mut w1 = BabyBear::ONE;
        for _p in 0..m4 {
            let w2 = w1 * w1;
            let w3 = w2 * w1;
            let r1 = w1.to_repr();
            let r2 = w2.to_repr();
            let r3 = w3.to_repr();
            twiddles.push(r1);
            twiddles.push(r2);
            twiddles.push(r3);
            twiddles_prime.push(shoup_quotient(r1));
            twiddles_prime.push(shoup_quotient(r2));
            twiddles_prime.push(shoup_quotient(r3));
            w1 = w1 * step;
        }
    }

    (twiddles, twiddles_prime, omega4_repr, omega4_prime)
}

/// Twiddles for a single radix-2 DIF stage at index `stage_h`.
///
/// Generates m = N/(2^(h+1)) twiddles: omega_N^(2^h * p) for p in 0..m.
/// Used for the R2 remainder stage in a mixed R4/R2 plan.
///
/// Returns `(twiddles, twiddles_prime)`.
pub(crate) fn precompute_single_r2_twiddles(
    log_n: u32,
    direction: NttDirection,
    stage_h: u32,
) -> (Vec<u32>, Vec<u32>) {
    let n = 1u32 << log_n;
    let omega = BabyBear::root_of_unity(log_n);
    let omega = match direction {
        NttDirection::Forward => omega,
        NttDirection::Inverse => omega.inv().expect("root of unity must be invertible"),
    };
    let s = 1u32 << stage_h;
    let m = n >> (stage_h + 1);
    let step = omega.pow(s as u64);
    let mut twiddles = Vec::with_capacity(m as usize);
    let mut twiddles_prime = Vec::with_capacity(m as usize);
    let mut w = BabyBear::ONE;
    for _ in 0..m {
        let repr = w.to_repr();
        twiddles.push(repr);
        twiddles_prime.push(shoup_quotient(repr));
        w = w * step;
    }
    (twiddles, twiddles_prime)
}

/// Twiddle factors for the four-step diagonal: omega_N^(k_r * c).
///
/// Stored in C×R layout (the data layout after the initial transpose):
/// table[c * rows + k_r] = omega_N^(k_r * c).
///
/// NVIDIA scale-up Tier 2B Option A (2026-04-16): on the inverse
/// direction the table is pre-multiplied by `1/N` so that Phase 3's
/// diagonal-multiply pass ALSO applies the inverse-NTT normalization
/// `*= 1/N`. This makes Phase 7 (the separate scale-by-1/N dispatch)
/// unnecessary. Mathematically sound because:
/// 1. Phase 7 is `data[i] *= 1/N` for all i.
/// 2. Phase 3 is `data[i] *= twiddle[i]` for all i.
/// 3. Storing `twiddle[i] * (1/N)` makes Phase 3 produce
///    `data[i] * twiddle[i] * (1/N)` — the exact composed result.
/// 4. Phases 4–6 (transpose, leaf NTT, transpose) are linear, so the
///    `1/N` factor propagates unchanged through.
/// At log 22 on RTX 4090 this eliminates ~193 µs of per-inverse-call
/// dispatch. Works on all backends — no kernel, shader, or binding
/// change required. See:
/// `research/benchmarks/nvidia-scale-up-2026-04-16/tier-2b-inverse-pathology-hypothesis.md`.
///
/// Returns `(twiddles, twiddles_prime)`.
pub(crate) fn precompute_fourstep_twiddles(
    config: &FourStepPlanConfig,
    direction: NttDirection,
) -> (Vec<u32>, Vec<u32>) {
    let omega = BabyBear::root_of_unity(config.log_n);
    let omega = match direction {
        NttDirection::Forward => omega,
        NttDirection::Inverse => omega.inv().expect("root of unity must be invertible"),
    };

    // For inverse NTT, fold the `1/N` normalization into the diagonal
    // twiddle so Phase 7 (separate scale pass) can be skipped. For
    // forward NTT, use the identity — no normalization required.
    let normalize = match direction {
        NttDirection::Forward => BabyBear::ONE,
        NttDirection::Inverse => BabyBear::new(config.n)
            .inv()
            .expect("N must be invertible in BabyBear"),
    };

    let rows = config.rows;
    let cols = config.cols;
    let cap = (rows * cols) as usize;
    let mut table = Vec::with_capacity(cap);
    let mut table_prime = Vec::with_capacity(cap);

    for c in 0..cols {
        let omega_c = omega.pow(c as u64);
        let mut w = normalize;
        for _k_r in 0..rows {
            let repr = w.to_repr();
            table.push(repr);
            table_prime.push(shoup_quotient(repr));
            w = w * omega_c;
        }
    }

    (table, table_prime)
}
