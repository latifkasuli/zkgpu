//! A/B bench for Gate 2 item #6 — Poseidon2 constants in storage vs uniform.
//!
//! Measures the standalone `WgpuBabyBearPoseidon2Plan` permutation
//! throughput under both `Poseidon2ConstantsSource::Storage` (today's
//! default) and `Poseidon2ConstantsSource::Uniform` (the pilot path).
//! See the design rationale in `crates/zkgpu-wgpu/src/poseidon2/plan.rs`
//! and item #6 of `docs/research/zkgpu-wgpu-speed-opportunities.md`.
//!
//! Bench shape: a `b.iter` loop calls `plan.execute` on a pre-uploaded
//! state buffer of `BATCH * WIDTH` BabyBear elements. The buffer
//! contents are scrambled by the previous permutation, but Poseidon2
//! is bijective on the field so the GPU work is the same every
//! iteration. Upload + plan-build + readback are all pulled out of
//! the timing loop; only the permutation itself is measured.
//!
//! Three batch sizes:
//! - 64    — one workgroup, dispatch-overhead-dominated regime.
//! - 4096  — moderate work, matches small FRI-fold leaf counts.
//! - 65536 — kernel-time-dominated, matches `log_h ≈ 16` leaf counts.
//!
//! Measurement caveat: GPU clocking variability on discrete cards has
//! a documented ~±30% floor in this project (see
//! `docs/two-consumers.md` v0.2 note). Treat single-digit deltas at
//! the 65 K size on NVIDIA as inside the noise floor; the regime
//! where uniforms are most likely to win is the small-batch /
//! dispatch-overhead one anyway.
//!
//! Output groups:
//! - `poseidon2_storage` — Storage-bound kernel (today)
//! - `poseidon2_uniform` — Uniform-bound kernel (pilot)
//!
//! Comparison: Criterion writes both groups to `target/criterion/`;
//! eyeballing the median deltas at each batch size gives the gate
//! signal. ≥ 5% win on Apple Silicon at any batch size = propagate
//! to the production kernels (merkle_leaf, merkle_compress, plonky3
//! W16/W24). Null/regression on every host = keep pilot path as
//! documented and move to item #3.

use std::sync::OnceLock;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use zkgpu_babybear::BabyBear;
use zkgpu_core::{GpuDevice, GpuField};
use zkgpu_poseidon2::{Poseidon2Params, WIDTH};
use zkgpu_wgpu::{Poseidon2ConstantsSource, WgpuBabyBearPoseidon2Plan, WgpuDevice};

static DEVICE: OnceLock<WgpuDevice> = OnceLock::new();

fn device() -> &'static WgpuDevice {
    DEVICE.get_or_init(|| {
        let d = WgpuDevice::new().expect("GPU device required for benchmarks");
        eprintln!("Benchmark device: {}", d.caps());
        d
    })
}

const BATCH_SIZES: &[usize] = &[64, 4096, 65536];

fn make_state(num_permutations: usize) -> Vec<BabyBear> {
    let n = num_permutations * WIDTH;
    // Non-trivial pseudo-random pattern. Use `from_u64` so values
    // get reduced into the BabyBear range (modulus 2^31 - 2^27 + 1
    // is below 2^31, so a 31-bit mask isn't sufficient).
    (0..n as u64)
        .map(|i| BabyBear::from_u64(i.wrapping_mul(2654435761)))
        .collect()
}

fn bench_one_variant(
    c: &mut Criterion,
    group_name: &str,
    source: Poseidon2ConstantsSource,
) {
    let dev = device();
    let mut group = c.benchmark_group(group_name);

    for &batch in BATCH_SIZES {
        // Throughput in permutations/sec — divides bench wall time
        // by the batch size.
        group.throughput(Throughput::Elements(batch as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch),
            &batch,
            |b, &batch| {
                let params = Poseidon2Params::<BabyBear, WIDTH>::babybear_default();
                let mut plan =
                    WgpuBabyBearPoseidon2Plan::new_with_constants_source(
                        dev, params, source,
                    )
                    .expect("plan creation failed");
                let initial = make_state(batch);
                let mut buf = dev.upload(&initial).expect("upload failed");
                // Warm the pipeline + cache: one pre-iteration so the
                // first `b.iter` call doesn't pay shader compilation.
                plan.execute(dev, &mut buf).expect("warmup execute failed");
                b.iter(|| {
                    plan.execute(dev, &mut buf).expect("execute failed");
                });
            },
        );
    }
    group.finish();
}

fn bench_storage(c: &mut Criterion) {
    bench_one_variant(c, "poseidon2_storage", Poseidon2ConstantsSource::Storage);
}

fn bench_uniform(c: &mut Criterion) {
    bench_one_variant(c, "poseidon2_uniform", Poseidon2ConstantsSource::Uniform);
}

criterion_group!(benches, bench_storage, bench_uniform);
criterion_main!(benches);
