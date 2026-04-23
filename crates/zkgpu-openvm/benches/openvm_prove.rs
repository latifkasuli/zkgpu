//! Stage 3 prove/verify bench — end-to-end OpenVM Fibonacci.
//!
//! One Criterion group, `target_stack/prove/fib_air`, two rows per
//! `LOG_TRACE_DEGREE`: CPU `BabyBearPoseidon2Engine` vs GPU
//! `ZkgpuOpenVmEngine`. Both use OpenVM's dummy Fibonacci AIR +
//! `FriParameters::standard_with_100_bits_security(LOG_BLOWUP)`,
//! so this is a direct same-shape prove comparison.
//!
//! Methodology matches `openvm_commit.rs`: 15 samples, 10s
//! measurement time.
//!
//! The engine under test lives in `tests/support/engine.rs`. The
//! `#[path]` import below keeps one copy of that code between the
//! integration test and the bench.

use std::sync::Arc;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use openvm_stark_backend::{
    engine::StarkEngine,
    prover::types::{AirProvingContext, ProvingContext},
    AirRef,
};
use openvm_stark_sdk::{
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, FriParameters},
    dummy_airs::fib_air::{air::FibonacciAir, trace::generate_trace_rows},
    engine::StarkFriEngine,
};

use p3_baby_bear::BabyBear;
use p3_field::PrimeCharacteristicRing;

use zkgpu_wgpu::WgpuDevice;

// Share the engine with the integration test.
#[path = "../tests/support/engine.rs"]
mod engine;
use engine::ZkgpuOpenVmEngine;

const LOG_BLOWUP: usize = 1;

// Public inputs and initial values match OpenVM's fibonacci.rs
// example. `LOG_TRACE_DEGREE` is swept per-bench below.
const A: u32 = 0;
const B: u32 = 1;

type Val = BabyBear;

/// Fibonacci's nth value **in the BabyBear field**. `u32` overflow
/// breaks the public-input check at larger `n` because the AIR
/// computes in BabyBear arithmetic (mod `p = 2^31 - 2^27 + 1`) and
/// u32 wraps mod `2^32` — the two wrap shapes diverge by `n ≈ 47`.
/// OpenVM's `get_fib_number(n: usize) -> u32` is fine at their
/// example's `n = 8` but not at the scales we sweep here.
fn fib_field(n: usize) -> Val {
    let mut a = Val::from_u32(A);
    let mut b = Val::from_u32(B);
    for _ in 0..n - 1 {
        let c = a + b;
        a = b;
        b = c;
    }
    b
}

fn try_device() -> Option<Arc<WgpuDevice>> {
    WgpuDevice::new().ok().map(Arc::new)
}

fn bench_prove_fib(c: &mut Criterion) {
    let Some(device) = try_device() else {
        eprintln!("no GPU adapter available; skipping GPU rows");
        return;
    };

    let fri_params = FriParameters::standard_with_100_bits_security(LOG_BLOWUP);
    let cpu_engine = BabyBearPoseidon2Engine::new(fri_params);
    let gpu_engine = ZkgpuOpenVmEngine::new(device, fri_params);

    let mut group = c.benchmark_group("target_stack/prove/fib_air");
    group.sample_size(15);
    group.measurement_time(Duration::from_secs(10));

    // 14, 16, 18 per the Stage 3 plan; 20 gated on runtime.
    for &log_n in &[14usize, 16, 18] {
        let n = 1usize << log_n;
        let trace = generate_trace_rows::<Val>(A, B, n);
        let public_values = vec![
            BabyBear::from_u32(A),
            BabyBear::from_u32(B),
            fib_field(n),
        ];
        let trace_arc = Arc::new(trace);
        let param = format!("log_h={log_n}");

        // --- Keygen hoisted out of the measured region ---
        //
        // Keygen is MMCS-independent (runs over the AIR's
        // preprocessed trace + circuit metadata, not the main
        // trace), so timing it would mix in a fixed cost that both
        // engines pay identically and dilute the prove-path signal
        // we actually care about. Each engine gets its own `pk`
        // because their `StarkConfig` associated types differ at
        // the Rust level (`Pcs = TwoAdicFriPcs<..., CpuValMmcs, ...>`
        // vs `Pcs = TwoAdicFriPcs<..., OpenVmGpuMmcs, ...>`), so
        // `MultiStarkProvingKey<SC>` is a different concrete type
        // per engine and can't be shared across rows.
        let cpu_airs: Vec<AirRef<_>> = vec![Arc::new(FibonacciAir)];
        let mut cpu_kb = cpu_engine.keygen_builder();
        let cpu_air_ids = cpu_engine.set_up_keygen_builder(&mut cpu_kb, &cpu_airs);
        let cpu_pk = cpu_kb.generate_pk();

        let gpu_airs: Vec<AirRef<_>> = vec![Arc::new(FibonacciAir)];
        let mut gpu_kb = gpu_engine.keygen_builder();
        let gpu_air_ids = gpu_engine.set_up_keygen_builder(&mut gpu_kb, &gpu_airs);
        let gpu_pk = gpu_kb.generate_pk();

        group.bench_with_input(
            BenchmarkId::new("cpu_engine", &param),
            &(trace_arc.clone(), public_values.clone()),
            |bencher, (t, pis)| {
                bencher.iter(|| {
                    // Rebuild AirProvingContext + ProvingContext
                    // per iteration because `ProvingContext` is
                    // consumed by `prove_then_verify`. Arc clone is
                    // cheap; public_values clone is a tiny Vec of
                    // 3 field elements. Neither is what we're
                    // trying to time.
                    let air_ctx = AirProvingContext::simple(t.clone(), pis.clone());
                    let ctx = ProvingContext::new(
                        cpu_air_ids
                            .iter()
                            .cloned()
                            .zip(std::iter::once(air_ctx))
                            .collect(),
                    );
                    cpu_engine
                        .prove_then_verify(black_box(&cpu_pk), black_box(ctx))
                        .expect("cpu prove_then_verify");
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("gpu_engine", &param),
            &(trace_arc, public_values),
            |bencher, (t, pis)| {
                bencher.iter(|| {
                    let air_ctx = AirProvingContext::simple(t.clone(), pis.clone());
                    let ctx = ProvingContext::new(
                        gpu_air_ids
                            .iter()
                            .cloned()
                            .zip(std::iter::once(air_ctx))
                            .collect(),
                    );
                    gpu_engine
                        .prove_then_verify(black_box(&gpu_pk), black_box(ctx))
                        .expect("gpu prove_then_verify");
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_prove_fib);
criterion_main!(benches);
