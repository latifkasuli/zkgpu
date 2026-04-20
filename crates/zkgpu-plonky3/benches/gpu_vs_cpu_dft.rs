//! Phase 7.4 — perf microbench comparing `GpuDft::strict_gpu` against
//! `Radix2DitParallel` at FRI-realistic sizes.
//!
//! Answers the question the 7.0 spec (§8) promised: "zkgpu wins at
//! (log_h, width) above X" — with specific coordinates on the host
//! this runs on. Per Codex review (§8), `p3-zk-proofs` is not a
//! valid perf harness because its AIRs pad to 2 rows; the dedicated
//! microbench lives here.
//!
//! # Running
//!
//! ```bash
//! cargo bench -p zkgpu-plonky3 --bench gpu_vs_cpu_dft
//! ```
//!
//! # What it measures
//!
//! For each `(log_h, width)` pair:
//! * **GPU**: `GpuDft::strict_gpu().dft_batch(mat)` — silent CPU
//!   fallback would panic, so times reflect the GPU path only.
//! * **CPU**: `Radix2DitParallel::default().dft_batch(mat)` — Plonky3's
//!   production CPU DFT, the one `p3-zk-proofs` and SP1 use today.
//!
//! Plan-build cost is amortized via [`GpuDft::preload_plans`] before
//! the timed section. Field-conversion cost (Monty ↔ canonical) is
//! part of the GPU timing since it's overhead a real consumer pays.
//!
//! # Expected shape of results
//!
//! At width=1, GPU should win above `log_h ≈ 18–20` on a discrete
//! NVIDIA GPU (the historical zkgpu crossover for single-poly NTT).
//! At width=8 under Path A (column loop), launch overhead
//! multiplies 8×, pushing the crossover higher — possibly not
//! winning until `log_h ≥ 22` or not at all. That's the data point
//! that motivates Path B (2D batched plan) in Phase 7.5.

use criterion::{black_box, BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use zkgpu_plonky3::GpuDft;

/// Generate a deterministic random trace matrix.
fn random_matrix(log_h: usize, w: usize, seed: u64) -> RowMajorMatrix<BabyBear> {
    let h = 1usize << log_h;
    let mut rng = StdRng::seed_from_u64(seed);
    let values: Vec<BabyBear> = (0..h * w)
        .map(|_| BabyBear::from_u64(rng.random::<u64>()))
        .collect();
    RowMajorMatrix::new(values, w)
}

/// Parameters swept by the benchmark.
///
/// log_h coverage: picks sample points across the range where we
/// expect behavior to change. `12` is firmly in the CPU's favor
/// (smaller than zkgpu's historical crossover). `14`/`16` bracket
/// the typical FRI blowup-domain sizes for small circuits. `18`
/// through `22` are realistic production sizes. `20` is the size
/// zkgpu's own NTT benchmarks report as the sweet spot for
/// discrete GPUs.
///
/// width=1 exercises the single-poly path (FRI query folds).
/// width=8 exercises a modest batch (typical mid-sized AIR column
/// count); Path A should struggle here, which is exactly the
/// signal 7.5 needs.
const LOG_HS: &[usize] = &[12, 14, 16, 18, 20];
const WIDTHS: &[usize] = &[1, 8];

fn bench_dft_batch(c: &mut Criterion) {
    // Warm up the GPU plan cache for every size we're about to
    // benchmark, so plan-build cost doesn't contaminate measurements.
    // Preload uses the same process-global context the bench runs will.
    let warmup = GpuDft::<BabyBear>::strict_gpu();
    let log_ns: Vec<u32> = LOG_HS.iter().map(|&l| l as u32).collect();
    warmup.preload_plans(&log_ns);

    let mut group = c.benchmark_group("dft_batch");

    // Criterion defaults target a 5s measurement window per size.
    // At log_h=20 × w=8, a single `dft_batch` call is ~50ms on CPU
    // and we'd want ~100 samples — tune here to keep runtime
    // tractable without sacrificing statistical meaning.
    group.sample_size(20);
    group.measurement_time(std::time::Duration::from_secs(10));

    for &log_h in LOG_HS {
        for &w in WIDTHS {
            let h = 1usize << log_h;
            // Skip combinations that blow past comfortable memory.
            // At log_h=20, w=8 → 8M elements × 4B = 32 MiB per
            // matrix × 2 (in + out) = 64 MiB. Fine.
            let bytes = (h * w * 4) as u64;
            if bytes > 512 * 1024 * 1024 {
                continue;
            }

            let mat = random_matrix(log_h, w, 0xBADF00D);
            let param = format!("log_h={log_h}/w={w}");

            group.bench_with_input(
                BenchmarkId::new("cpu", &param),
                &mat,
                |bencher, input| {
                    let dft = Radix2DitParallel::<BabyBear>::default();
                    bencher.iter(|| {
                        let _ = black_box(dft.dft_batch(black_box(input.clone())));
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new("gpu", &param),
                &mat,
                |bencher, input| {
                    let dft = GpuDft::<BabyBear>::strict_gpu();
                    bencher.iter(|| {
                        let _ = black_box(dft.dft_batch(black_box(input.clone())));
                    });
                },
            );
        }
    }

    group.finish();
}

/// Same comparison but on `coset_lde_batch` — the actual FRI hot
/// path. Default trait decomposition routes through our `dft_batch`
/// twice (idft + coset_dft after resize), so this measures the whole
/// FRI-commit-critical DFT pipeline. If GPU loses here, it loses for
/// real-world Plonky3 consumers even if the standalone `dft_batch` bench
/// looked favorable.
fn bench_coset_lde_batch(c: &mut Criterion) {
    use p3_field::Field;

    let warmup = GpuDft::<BabyBear>::strict_gpu();
    // Preload both log_h and log_h + added_bits since coset_lde
    // invokes DFT at the extended size too.
    let mut log_ns: Vec<u32> = Vec::new();
    for &l in LOG_HS {
        log_ns.push(l as u32);
        log_ns.push((l + 1) as u32); // added_bits=1
    }
    log_ns.sort_unstable();
    log_ns.dedup();
    warmup.preload_plans(&log_ns);

    let mut group = c.benchmark_group("coset_lde_batch");
    group.sample_size(15);
    group.measurement_time(std::time::Duration::from_secs(10));

    let shift = BabyBear::GENERATOR;

    for &log_h in LOG_HS {
        for &w in WIDTHS {
            let h = 1usize << log_h;
            // Working set doubles with added_bits=1. Skip cases
            // exceeding ~1 GiB.
            let bytes = (h * w * 4) as u64 * 2;
            if bytes > 1024 * 1024 * 1024 {
                continue;
            }

            let mat = random_matrix(log_h, w, 0xC050EAFE);
            let param = format!("log_h={log_h}/w={w}/added_bits=1");

            group.bench_with_input(
                BenchmarkId::new("cpu", &param),
                &mat,
                |bencher, input| {
                    let dft = Radix2DitParallel::<BabyBear>::default();
                    bencher.iter(|| {
                        let _ = black_box(
                            dft.coset_lde_batch(black_box(input.clone()), 1, shift),
                        );
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new("gpu", &param),
                &mat,
                |bencher, input| {
                    let dft = GpuDft::<BabyBear>::strict_gpu();
                    bencher.iter(|| {
                        let _ = black_box(
                            dft.coset_lde_batch(black_box(input.clone()), 1, shift),
                        );
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench_dft_batch, bench_coset_lde_batch);
criterion_main!(benches);
