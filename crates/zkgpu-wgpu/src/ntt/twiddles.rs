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

    let rows = config.rows;
    let cols = config.cols;
    let cap = (rows * cols) as usize;
    let mut table = Vec::with_capacity(cap);
    let mut table_prime = Vec::with_capacity(cap);

    for c in 0..cols {
        let omega_c = omega.pow(c as u64);
        let mut w = BabyBear::ONE;
        for _k_r in 0..rows {
            let repr = w.to_repr();
            table.push(repr);
            table_prime.push(shoup_quotient(repr));
            w = w * omega_c;
        }
    }

    (table, table_prime)
}
