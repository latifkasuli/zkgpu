//! Stage 3 end-to-end prove/verify test.
//!
//! Closes the Stage 2c limitation ("not yet integrated into a full
//! OpenVM prove/verify end-to-end test") by running OpenVM's own
//! dummy Fibonacci AIR through two engines:
//!
//! 1. **CPU control:** `BabyBearPoseidon2Engine` — OpenVM's own
//!    default engine, stock `MerkleTreeMmcs` throughout. Confirms the
//!    test setup itself is sound.
//! 2. **GPU under test:** `ZkgpuOpenVmEngine` — same StarkConfig
//!    shape, same Poseidon2 constants, same FRI parameters, but
//!    with `OpenVmGpuMmcs` as the base-field `ValMmcs`. Prove on
//!    GPU, verify on GPU (trait-level verify delegation to CPU is
//!    the Stage 2b story; Stage 3 just confirms it all hangs
//!    together through the full prover loop).
//!
//! Both engines prove + verify the same trace, so both code paths
//! are exercised in-process in one test.

#![allow(clippy::useless_vec)]

use std::sync::Arc;

use openvm_stark_backend::{
    engine::StarkEngine, p3_matrix::dense::RowMajorMatrix,
    prover::types::AirProvingContext, AirRef,
};
use openvm_stark_sdk::{
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, FriParameters},
    dummy_airs::fib_air::{air::FibonacciAir, trace::generate_trace_rows},
    engine::StarkFriEngine,
};
use p3_baby_bear::BabyBear;
use p3_field::PrimeCharacteristicRing;

use zkgpu_wgpu::WgpuDevice;

// Single-file shared helper — no `support/mod.rs` needed. The
// bench imports the same file via `#[path]` from its own directory.
#[path = "support/engine.rs"]
mod engine;
use engine::ZkgpuOpenVmEngine;

const LOG_BLOWUP: usize = 1;
const LOG_TRACE_DEGREE: usize = 3;

// Public inputs match OpenVM's own fibonacci.rs example: start at
// (a, b) = (0, 1) and expose the nth Fibonacci value.
const A: u32 = 0;
const B: u32 = 1;
const N: usize = 1usize << LOG_TRACE_DEGREE;

type Val = BabyBear;

/// Fibonacci's nth value in the BabyBear field. See the bench file
/// for the full rationale — u32 overflow breaks the public-input
/// check at `n ≳ 47` because the AIR works in BabyBear arithmetic.
/// At `N = 8` `u32` would still be fine, but we match the bench's
/// shape so any future resize of this test doesn't silently break.
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

/// Build the shared trace + public inputs for both tests.
///
/// AIRs aren't returned here because `AirRef<SC>` binds to a
/// specific config type — and the two tests use different SCs
/// (control: `BabyBearPoseidon2Config`, GPU: `ZkgpuOpenVmConfig`).
/// `FibonacciAir` is `Clone` + SC-agnostic at the trait level, so
/// each test just re-wraps it.
fn build_trace_and_pis() -> (RowMajorMatrix<Val>, Vec<Val>) {
    let public_values = vec![
        BabyBear::from_u32(A),
        BabyBear::from_u32(B),
        fib_field(N),
    ];
    let trace = generate_trace_rows::<Val>(A, B, N);
    (trace, public_values)
}

/// Control case: OpenVM's own CPU engine proves + verifies the
/// dummy Fibonacci AIR. If this breaks, the upgrade path (GPU
/// engine) is also broken — so this gates whether the test failure
/// is in our code or in the harness setup.
#[test]
fn cpu_control_prove_verify_fibonacci() {
    let (trace, public_values) = build_trace_and_pis();
    let engine = BabyBearPoseidon2Engine::new(
        FriParameters::standard_with_100_bits_security(LOG_BLOWUP),
    );

    let airs: Vec<AirRef<_>> = vec![Arc::new(FibonacciAir)];
    let mut keygen_builder = engine.keygen_builder();
    let air_ids = engine.set_up_keygen_builder(&mut keygen_builder, &airs);
    let pk = keygen_builder.generate_pk();

    let trace_arc = Arc::new(trace);
    let air_ctx = AirProvingContext::simple(trace_arc, public_values);
    let ctx = openvm_stark_backend::prover::types::ProvingContext::new(
        air_ids.into_iter().zip(std::iter::once(air_ctx)).collect(),
    );

    engine
        .prove_then_verify(&pk, ctx)
        .expect("cpu control: prove+verify failed");
}

/// The primary Stage 3 claim: `OpenVmGpuMmcs` plugs into a full
/// OpenVM-shape prover loop and produces a verifiable proof of
/// OpenVM's dummy Fibonacci AIR.
#[test]
fn gpu_prove_verify_fibonacci() {
    let Some(gpu_device) = WgpuDevice::new().ok().map(Arc::new) else {
        eprintln!("no GPU adapter available; skipping gpu_prove_verify_fibonacci");
        return;
    };

    let (trace, public_values) = build_trace_and_pis();
    let engine = ZkgpuOpenVmEngine::new(
        gpu_device,
        FriParameters::standard_with_100_bits_security(LOG_BLOWUP),
    );

    let airs: Vec<AirRef<_>> = vec![Arc::new(FibonacciAir)];
    let mut keygen_builder = engine.keygen_builder();
    let air_ids = engine.set_up_keygen_builder(&mut keygen_builder, &airs);
    let pk = keygen_builder.generate_pk();

    let trace_arc = Arc::new(trace);
    let air_ctx = AirProvingContext::simple(trace_arc, public_values);
    let ctx = openvm_stark_backend::prover::types::ProvingContext::new(
        air_ids.into_iter().zip(std::iter::once(air_ctx)).collect(),
    );

    engine
        .prove_then_verify(&pk, ctx)
        .expect("gpu engine: prove+verify failed");
}
