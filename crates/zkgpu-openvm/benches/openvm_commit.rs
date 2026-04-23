//! Phase 3.d Stage 2c — OpenVM MMCS commit bench.
//!
//! Two Criterion groups, both CPU-vs-GPU:
//!
//! * `target_stack/commit` — commit-only. Primary gate: the
//!   portability claim's headline metric lives here.
//! * `target_stack/commit_open_40q` — commit + 40 consecutive
//!   queries. Mirrors the FRI-side workload shape the adapter is
//!   ultimately consumed by (OpenVM's FRI config uses the same
//!   per-query opening pattern zkgpu-plonky3 already benches).
//!
//! Deliberately **not** in any timed group: `verify_batch`. The
//! adapter accelerates prover-side commit/open production; verifier
//! delegation is CPU by design (the commitment + proof types are
//! concretely identical to Plonky3's CPU reference, so cross-
//! verification works without any GPU involvement). Timing a CPU
//! delegate would blur the story.
//!
//! Shapes:
//!
//! * **Single-matrix** at `log_h ∈ {14, 16, 18, 20}, w = 8` —
//!   sanity rows. Also provides a direct W24-leaf (zkgpu-plonky3)
//!   vs W16-leaf (zkgpu-openvm) comparison at the same nominal
//!   height, because the two adapters bench the same
//!   non-mixed-height shape.
//! * **Mixed-height trace + quotient** at `log_h_max ∈ {14, 16, 18}` —
//!   the headline shape. Trace at `h_max` with `w = 8`, plus 4
//!   quotient-chunk-shaped matrices at `h_max / 2` each with
//!   `w = 2`. Matches OpenVM's commit_quotient injection pattern.
//!
//! Methodology matches zkgpu-plonky3's `target_stack/fri_commit`
//! block: 15 samples per bench, 10s target measurement time.

use std::sync::Arc;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use p3_baby_bear::BabyBear;
use p3_commit::Mmcs;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_poseidon2::ExternalLayerConstants;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::distr::StandardUniform;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use zkgpu_openvm::{babybear_openvm_params, OpenVmGpuMmcs, Perm, Val};
use zkgpu_wgpu::WgpuDevice;

const ROUNDS_F: usize = 8;

// Plonky3 0.4.1 CPU reference MMCS — exact OpenVM config.
type CpuValMmcs = MerkleTreeMmcs<
    <Val as p3_field::Field>::Packing,
    <Val as p3_field::Field>::Packing,
    PaddingFreeSponge<Perm, 16, 8, 8>,
    TruncatedPermutation<Perm, 2, 8, 16>,
    8,
>;

// Number of FRI-style queries per commit. Matches the `num_queries`
// setting used by zkgpu-plonky3's target stack.
const NUM_QUERIES: usize = 40;

fn try_device() -> Option<Arc<WgpuDevice>> {
    WgpuDevice::new().ok().map(Arc::new)
}

/// Build matched CPU + GPU MMCS instances with identical Plonky3
/// 0.4.1 Poseidon2 constants. Seed fixed so bench runs are
/// reproducible.
fn build_matched(device: Arc<WgpuDevice>) -> (CpuValMmcs, OpenVmGpuMmcs) {
    let mut rng = SmallRng::seed_from_u64(0x_0BE_BE1C_u64);
    let ext: ExternalLayerConstants<BabyBear, 16> =
        ExternalLayerConstants::new_from_rng(ROUNDS_F, &mut rng);
    let int: Vec<BabyBear> =
        (&mut rng).sample_iter(StandardUniform).take(13).collect();
    let perm16: Perm = Perm::new(ext.clone(), int.clone());

    let cpu_sponge = PaddingFreeSponge::new(perm16.clone());
    let cpu_compress = TruncatedPermutation::new(perm16.clone());
    let cpu_mmcs = CpuValMmcs::new(cpu_sponge, cpu_compress);

    let zkgpu_params = babybear_openvm_params(&ext, &int);
    let gpu_mmcs = OpenVmGpuMmcs::new(
        device,
        perm16,
        zkgpu_params.clone(),
        zkgpu_params,
        0,
    )
    .expect("build_matched: OpenVmGpuMmcs construction");

    (cpu_mmcs, gpu_mmcs)
}

/// Generate a random `h × w` matrix, seeded for reproducibility.
fn random_matrix(h: usize, w: usize, seed: u64) -> RowMajorMatrix<BabyBear> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let values: Vec<BabyBear> =
        (&mut rng).sample_iter(StandardUniform).take(h * w).collect();
    RowMajorMatrix::new(values, w)
}

// --- Shape builders ---

/// Single-matrix shape — sanity row.
fn single_matrix(log_h: usize, w: usize) -> Vec<RowMajorMatrix<BabyBear>> {
    let h = 1usize << log_h;
    vec![random_matrix(h, w, 0x_BE10_0000_u64 ^ ((log_h as u64) << 16) ^ (w as u64))]
}

/// Mixed-height shape: trace + 4 quotient chunks at half height.
/// Mirrors OpenVM's commit_quotient pattern — a trace at `h_max`
/// plus multiple quotient-polynomial chunks at `h_max / 2`.
fn trace_plus_quotient(log_h_max: usize) -> Vec<RowMajorMatrix<BabyBear>> {
    let h_max = 1usize << log_h_max;
    let h_q = h_max / 2;
    let trace = random_matrix(h_max, 8, 0x_B2_0000_u64 ^ (log_h_max as u64));
    let quotient_chunks: Vec<RowMajorMatrix<BabyBear>> = (0..4)
        .map(|i| {
            random_matrix(
                h_q,
                2,
                0x_B2_0C00_u64 ^ (log_h_max as u64) << 8 ^ (i as u64),
            )
        })
        .collect();
    let mut out = Vec::with_capacity(5);
    out.push(trace);
    out.extend(quotient_chunks);
    out
}

// --- commit-only bench (primary gate) ---

fn bench_commit(c: &mut Criterion) {
    let Some(device) = try_device() else {
        eprintln!("no GPU adapter available; skipping GPU rows");
        return;
    };
    let (cpu, gpu) = build_matched(device);

    let mut group = c.benchmark_group("target_stack/commit");
    group.sample_size(15);
    group.measurement_time(Duration::from_secs(10));

    // Single-matrix rows (sanity + adapter-to-adapter comparison vs
    // zkgpu-plonky3 bench). Cap at log_h=20 to match the fri_commit
    // bench's range.
    for &log_h in &[14usize, 16, 18, 20] {
        let matrices = single_matrix(log_h, 8);
        let param = format!("single/log_h={log_h}/w=8");

        group.bench_with_input(
            BenchmarkId::new("cpu_mmcs", &param),
            &matrices,
            |bencher, input| {
                bencher.iter(|| {
                    let (cap, _pd) = cpu.commit(black_box(input.clone()));
                    black_box(cap);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("gpu_mmcs", &param),
            &matrices,
            |bencher, input| {
                bencher.iter(|| {
                    let (cap, _pd) = gpu.commit(black_box(input.clone()));
                    black_box(cap);
                });
            },
        );
    }

    // Mixed-height rows — headline shape. log_h=20 would take long
    // with the quotient-chunk flatten cost; cap at 18 here.
    for &log_h_max in &[14usize, 16, 18] {
        let matrices = trace_plus_quotient(log_h_max);
        let param = format!("mixed/log_h_max={log_h_max}/trace_plus_4q");

        group.bench_with_input(
            BenchmarkId::new("cpu_mmcs", &param),
            &matrices,
            |bencher, input| {
                bencher.iter(|| {
                    let (cap, _pd) = cpu.commit(black_box(input.clone()));
                    black_box(cap);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("gpu_mmcs", &param),
            &matrices,
            |bencher, input| {
                bencher.iter(|| {
                    let (cap, _pd) = gpu.commit(black_box(input.clone()));
                    black_box(cap);
                });
            },
        );
    }

    group.finish();
}

// --- commit + 40 queries bench (full-FRI-workload shape) ---

fn bench_commit_open_40q(c: &mut Criterion) {
    let Some(device) = try_device() else {
        eprintln!("no GPU adapter available; skipping GPU rows");
        return;
    };
    let (cpu, gpu) = build_matched(device);

    let mut group = c.benchmark_group("target_stack/commit_open_40q");
    group.sample_size(15);
    group.measurement_time(Duration::from_secs(10));

    // Generate 40 fixed query indices per shape (seeded so bench
    // runs are reproducible across hosts).
    fn gen_query_indices(h_max: usize, seed: u64) -> Vec<usize> {
        let mut rng = SmallRng::seed_from_u64(seed);
        (0..NUM_QUERIES)
            .map(|_| rng.random_range(0..h_max))
            .collect()
    }

    // Single-matrix rows.
    for &log_h in &[14usize, 16, 18] {
        let h_max = 1usize << log_h;
        let matrices = single_matrix(log_h, 8);
        let indices = gen_query_indices(h_max, 0x_10_1D_0000_u64 ^ (log_h as u64));
        let param = format!("single/log_h={log_h}/w=8");

        group.bench_with_input(
            BenchmarkId::new("cpu_mmcs", &param),
            &(matrices.clone(), indices.clone()),
            |bencher, (input, idxs)| {
                bencher.iter(|| {
                    let (cap, pd) = cpu.commit(black_box(input.clone()));
                    black_box(&cap);
                    for &i in idxs {
                        let opening = cpu.open_batch(i, &pd);
                        black_box(opening);
                    }
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("gpu_mmcs", &param),
            &(matrices, indices),
            |bencher, (input, idxs)| {
                bencher.iter(|| {
                    let (cap, pd) = gpu.commit(black_box(input.clone()));
                    black_box(&cap);
                    for &i in idxs {
                        let opening = gpu.open_batch(i, &pd);
                        black_box(opening);
                    }
                });
            },
        );
    }

    // Mixed-height rows.
    for &log_h_max in &[14usize, 16, 18] {
        let h_max = 1usize << log_h_max;
        let matrices = trace_plus_quotient(log_h_max);
        let indices = gen_query_indices(h_max, 0x_10_10_0000_u64 ^ (log_h_max as u64));
        let param = format!("mixed/log_h_max={log_h_max}/trace_plus_4q");

        group.bench_with_input(
            BenchmarkId::new("cpu_mmcs", &param),
            &(matrices.clone(), indices.clone()),
            |bencher, (input, idxs)| {
                bencher.iter(|| {
                    let (cap, pd) = cpu.commit(black_box(input.clone()));
                    black_box(&cap);
                    for &i in idxs {
                        let opening = cpu.open_batch(i, &pd);
                        black_box(opening);
                    }
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("gpu_mmcs", &param),
            &(matrices, indices),
            |bencher, (input, idxs)| {
                bencher.iter(|| {
                    let (cap, pd) = gpu.commit(black_box(input.clone()));
                    black_box(&cap);
                    for &i in idxs {
                        let opening = gpu.open_batch(i, &pd);
                        black_box(opening);
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_commit, bench_commit_open_40q);
criterion_main!(benches);
