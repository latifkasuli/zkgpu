//! Phase 7 Step 1 — prover hot path bench against the Poseidon2-MMCS
//! target stack.
//!
//! Unlike `gpu_vs_cpu_dft.rs` which isolates `dft_batch` /
//! `coset_lde_batch`, this harness times the *full* FRI commit and
//! end-to-end `prove` on the Plonky3 Poseidon2-MMCS config — i.e.
//! the stack that the decision doc pinned as the first target
//! consumer (`research/phase-7-plonky3-adapter/c4-target-stack/
//! decision.md`).
//!
//! Baseline numbers collected here (CPU-only and CPU-MMCS-with-
//! GPU-DFT) become the reference that every later phase is graded
//! against. GPU Poseidon2 Merkle-commit work (Step 3) gets new
//! lines in the same groups for direct comparison.
//!
//! # Running
//!
//! ```bash
//! cargo bench -p zkgpu-plonky3 --bench prover_hot_path
//! ```

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::{ExtensionMmcs, Pcs};
use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::{StarkConfig, prove, verify};
use rand::distr::StandardUniform;
use rand::rngs::{SmallRng, StdRng};
use rand::{RngExt, SeedableRng};

use std::sync::Arc;

use zkgpu_plonky3::GpuDft;
use zkgpu_plonky3::gpu_mmcs::{GpuPoseidon2Mmcs, Perm16 as GpuPerm16, Perm24 as GpuPerm24};
use zkgpu_plonky3::poseidon2_bridge::babybear_plonky3_params;
use zkgpu_wgpu::WgpuDevice;

type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;
type Perm16 = Poseidon2BabyBear<16>;
type Perm24 = Poseidon2BabyBear<24>;
type Poseidon2Sponge = PaddingFreeSponge<Perm24, 24, 16, 8>;
type Poseidon2Compression = TruncatedPermutation<Perm16, 2, 8, 16>;
type ValMmcs = MerkleTreeMmcs<
    <Val as Field>::Packing,
    <Val as Field>::Packing,
    Poseidon2Sponge,
    Poseidon2Compression,
    2,
    8,
>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Challenger = DuplexChallenger<Val, Perm24, 24, 16>;
type DummyChallenger = Challenger;

const MMCS_SEED: u64 = 0x_5EED_CAFE_u64;

fn build_mmcs() -> (ValMmcs, ChallengeMmcs, Perm24) {
    let mut rng = SmallRng::seed_from_u64(MMCS_SEED);
    let perm16 = Perm16::new_from_rng_128(&mut rng);
    let perm24 = Perm24::new_from_rng_128(&mut rng);
    let hash = Poseidon2Sponge::new(perm24.clone());
    let compress = Poseidon2Compression::new(perm16);
    let val_mmcs = ValMmcs::new(hash, compress, 0);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    (val_mmcs, challenge_mmcs, perm24)
}

fn build_fri_params(
    challenge_mmcs: ChallengeMmcs,
    log_blowup: usize,
) -> FriParameters<ChallengeMmcs> {
    FriParameters {
        log_blowup,
        log_final_poly_len: 0,
        max_log_arity: 2,
        num_queries: 40,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: 8,
        mmcs: challenge_mmcs,
    }
}

fn random_trace(log_h: usize, w: usize, seed: u64) -> RowMajorMatrix<Val> {
    let h = 1usize << log_h;
    let mut rng = StdRng::seed_from_u64(seed);
    let values: Vec<Val> = (0..h * w)
        .map(|_| Val::from_u64(rng.random::<u64>()))
        .collect();
    RowMajorMatrix::new(values, w)
}

// --- DFT-only sweep (same shapes as gpu_vs_cpu_dft for continuity) -------

// Extended downward (10, 12) for Gate 2 item #4 (multi-dispatch per
// compute pass) — at small log_h the per-pass driver overhead is a
// larger fraction of total NTT time, where the fold benefit lands.
// Existing 14/16/18/20 sizes preserve continuity with v0.1/v0.2
// publish numbers.
const LOG_HS: &[usize] = &[10, 12, 14, 16, 18, 20];
const WIDTHS: &[usize] = &[1, 8];

fn bench_coset_lde_batch(c: &mut Criterion) {
    let shift = Val::GENERATOR;

    let mut group = c.benchmark_group("target_stack/coset_lde_batch");
    group.sample_size(15);
    group.measurement_time(std::time::Duration::from_secs(10));

    for &log_h in LOG_HS {
        for &w in WIDTHS {
            let h = 1usize << log_h;
            let bytes = (h * w * 4) as u64 * 2;
            if bytes > 1024 * 1024 * 1024 {
                continue;
            }
            let seed = 0x_C05E_7C0D_E0A0_1D00_u64
                ^ (log_h as u64)
                ^ ((w as u64) << 16);
            let mat = random_trace(log_h, w, seed);
            let param = format!("log_h={log_h}/w={w}");

            group.bench_with_input(
                BenchmarkId::new("cpu_dft", &param),
                &mat,
                |bencher, input| {
                    let dft = Radix2DitParallel::<Val>::default();
                    bencher.iter(|| {
                        let out = dft.coset_lde_batch(black_box(input.clone()), 1, shift);
                        black_box(out.to_row_major_matrix());
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new("gpu_dft", &param),
                &mat,
                |bencher, input| {
                    let dft = GpuDft::<Val>::strict_gpu();
                    bencher.iter(|| {
                        let out = dft.coset_lde_batch(black_box(input.clone()), 1, shift);
                        black_box(out.to_row_major_matrix());
                    });
                },
            );
        }
    }
    group.finish();
}

// --- FRI commit phase (LDE + Poseidon2 leaves + Merkle tree) -------------

fn commit_one<Dft>(dft: Dft, trace: RowMajorMatrix<Val>) -> <ValMmcs as p3_commit::Mmcs<Val>>::Commitment
where
    Dft: TwoAdicSubgroupDft<Val> + Clone,
{
    let (val_mmcs, challenge_mmcs, _perm24) = build_mmcs();
    let fri_params = build_fri_params(challenge_mmcs, 1);
    let pcs: TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs> =
        TwoAdicFriPcs::new(dft, val_mmcs, fri_params);
    let h = trace.height();
    let domain = <TwoAdicFriPcs<_, _, _, _> as Pcs<Challenge, DummyChallenger>>::natural_domain_for_degree(
        &pcs, h,
    );
    let (commitment, _prover_data) =
        <TwoAdicFriPcs<_, _, _, _> as Pcs<Challenge, DummyChallenger>>::commit(
            &pcs,
            vec![(domain, trace)],
        );
    commitment
}

// --- Step 3 variant: swap InputMmcs for `GpuPoseidon2Mmcs` -------------
//
// FriMmcs stays CPU (ExtensionMmcs<…, ValMmcs>) — FRI fold commits are
// tiny compared to the initial trace commit and GpuPoseidon2Mmcs is a
// bench-gate, commit-only adapter that can't serve the fold path.
//
// The bench reuses the same (device, GPU mmcs) across every iter of
// Criterion's inner loop to keep setup out of the measured region.
// Pipeline construction + buffer uploads for Poseidon2 constants are
// one-shot per-process.

/// Build the GPU MMCS (shared across all bench iterations) plus the
/// matching CPU `perm24` instance for the challenger.
fn build_gpu_mmcs(
    device: Arc<WgpuDevice>,
) -> (GpuPoseidon2Mmcs, Perm24) {
    // Mirrors `build_mmcs`'s constant-derivation pattern but with
    // explicit (ext, int) draws so we can hand the same constants to
    // both the CPU Perm24 (for the challenger) and the GPU Poseidon2
    // params (for commit).
    let mut rng = SmallRng::seed_from_u64(MMCS_SEED);
    // W16 draws first (matches the order in build_mmcs's
    // new_from_rng_128 implementation).
    const ROUNDS_F: usize = 8;
    let ext16: p3_poseidon2::ExternalLayerConstants<Val, 16> =
        p3_poseidon2::ExternalLayerConstants::new_from_rng(ROUNDS_F, &mut rng);
    let int16: Vec<Val> = (&mut rng).sample_iter(StandardUniform).take(13).collect();
    let ext24: p3_poseidon2::ExternalLayerConstants<Val, 24> =
        p3_poseidon2::ExternalLayerConstants::new_from_rng(ROUNDS_F, &mut rng);
    let int24: Vec<Val> = (&mut rng).sample_iter(StandardUniform).take(21).collect();

    let perm16: GpuPerm16 = GpuPerm16::new(ext16.clone(), int16.clone());
    let perm24: GpuPerm24 = GpuPerm24::new(ext24.clone(), int24.clone());
    let zkgpu_params16 = babybear_plonky3_params::<16>(&ext16, &int16);
    let zkgpu_params24 = babybear_plonky3_params::<24>(&ext24, &int24);

    let gpu_mmcs = GpuPoseidon2Mmcs::new(
        device,
        perm24.clone(),
        perm16,
        zkgpu_params24,
        zkgpu_params16,
        0,
    )
    .expect("build_gpu_mmcs: failed to construct GPU MMCS");

    (gpu_mmcs, perm24)
}

/// Step 3 gate: commit a single matrix via `TwoAdicFriPcs<Val, Dft,
/// GpuPoseidon2Mmcs, ChallengeMmcs>`. InputMmcs is the GPU adapter;
/// FriMmcs stays CPU so the FRI-level commits (fold) don't route
/// through `GpuPoseidon2Mmcs::open_batch` (which is unimplemented —
/// this adapter is commit-only).
///
/// `TwoAdicFriPcs::commit` is documented to only call
/// `mmcs.commit(ldes)` on the InputMmcs during the commit phase, so
/// this path exercises only the GPU commit kernel; no unimplemented
/// opening method is hit.
fn commit_one_gpu_mmcs<Dft>(
    dft: Dft,
    gpu_mmcs: GpuPoseidon2Mmcs,
    trace: RowMajorMatrix<Val>,
) -> <GpuPoseidon2Mmcs as p3_commit::Mmcs<Val>>::Commitment
where
    Dft: TwoAdicSubgroupDft<Val> + Clone,
{
    let (cpu_val_mmcs, challenge_mmcs, _) = build_mmcs();
    let _ = cpu_val_mmcs; // cpu_val_mmcs unused here — challenge_mmcs wraps its own clone
    let fri_params = build_fri_params(challenge_mmcs, 1);
    let pcs: TwoAdicFriPcs<Val, Dft, GpuPoseidon2Mmcs, ChallengeMmcs> =
        TwoAdicFriPcs::new(dft, gpu_mmcs, fri_params);
    let h = trace.height();
    let domain = <TwoAdicFriPcs<_, _, _, _> as Pcs<Challenge, DummyChallenger>>::natural_domain_for_degree(
        &pcs, h,
    );
    let (commitment, _prover_data) =
        <TwoAdicFriPcs<_, _, _, _> as Pcs<Challenge, DummyChallenger>>::commit(
            &pcs,
            vec![(domain, trace)],
        );
    commitment
}

fn bench_commit_phase(c: &mut Criterion) {
    let mut group = c.benchmark_group("target_stack/fri_commit");
    group.sample_size(15);
    group.measurement_time(std::time::Duration::from_secs(10));

    // Pre-build the GPU MMCS once — device init, pipeline compilation,
    // and Poseidon2 constants upload should not fall inside the
    // measured region. `GpuPoseidon2Mmcs: Clone` (Arc-backed) so we
    // can hand a fresh Clone per `commit_one_gpu_mmcs` call without
    // rebuilding.
    let gpu_mmcs_shared: Option<GpuPoseidon2Mmcs> = WgpuDevice::new()
        .ok()
        .map(Arc::new)
        .map(|dev| build_gpu_mmcs(dev).0);
    if gpu_mmcs_shared.is_none() {
        eprintln!("fri_commit: GPU unavailable; GPU-MMCS rows will be skipped");
    }

    for &log_h in LOG_HS {
        for &w in WIDTHS {
            let h = 1usize << log_h;
            let bytes = (h * w * 4) as u64 * 2;
            if bytes > 512 * 1024 * 1024 {
                continue;
            }
            let seed = 0x_C011_0AD0_C0DE_CAFE_u64
                ^ (log_h as u64)
                ^ ((w as u64) << 16);
            let mat = random_trace(log_h, w, seed);
            let param = format!("log_h={log_h}/w={w}");

            // --- Baseline: CPU DFT + CPU MMCS (Plonky3 stock). ---
            group.bench_with_input(
                BenchmarkId::new("cpu_dft_cpu_mmcs", &param),
                &mat,
                |bencher, input| {
                    bencher.iter(|| {
                        black_box(commit_one(
                            Radix2DitParallel::<Val>::default(),
                            black_box(input.clone()),
                        ));
                    });
                },
            );

            // --- DFT-only GPU (Step 1-2 scope). ---
            group.bench_with_input(
                BenchmarkId::new("gpu_dft_cpu_mmcs", &param),
                &mat,
                |bencher, input| {
                    bencher.iter(|| {
                        black_box(commit_one(
                            GpuDft::<Val>::strict_gpu(),
                            black_box(input.clone()),
                        ));
                    });
                },
            );

            // --- Step 3 go/no-go isolators ---------------------------
            //
            // MMCS-only GPU (CPU DFT): isolates the hash/tree swap.
            // Since the Step 1 baseline showed commit time is 93–96%
            // hash/tree-bound, this row is the most direct test of
            // the Step 3 thesis. If this row doesn't beat baseline
            // by ≥1.5× at log_h=18, w=8, the gate is not cleared.
            //
            // Both-GPU: composed win. Only meaningful once MMCS-only
            // clears the gate; otherwise a net win here could be a
            // DFT win masking an MMCS loss.
            if let Some(gpu_mmcs) = &gpu_mmcs_shared {
                let mmcs_clone = gpu_mmcs.clone();
                group.bench_with_input(
                    BenchmarkId::new("cpu_dft_gpu_mmcs", &param),
                    &mat,
                    |bencher, input| {
                        bencher.iter(|| {
                            black_box(commit_one_gpu_mmcs(
                                Radix2DitParallel::<Val>::default(),
                                mmcs_clone.clone(),
                                black_box(input.clone()),
                            ));
                        });
                    },
                );

                let mmcs_clone = gpu_mmcs.clone();
                group.bench_with_input(
                    BenchmarkId::new("gpu_dft_gpu_mmcs", &param),
                    &mat,
                    |bencher, input| {
                        bencher.iter(|| {
                            black_box(commit_one_gpu_mmcs(
                                GpuDft::<Val>::strict_gpu(),
                                mmcs_clone.clone(),
                                black_box(input.clone()),
                            ));
                        });
                    },
                );
            }
        }
    }
    group.finish();
}

// --- Full prove / verify round-trip on a trivial AIR --------------------

pub struct FibAir;

impl<F> BaseAir<F> for FibAir {
    fn width(&self) -> usize {
        2
    }
}

impl<AB: AirBuilder> Air<AB> for FibAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.current_slice();
        let next = main.next_slice();
        builder
            .when_transition()
            .assert_eq(next[0].into(), local[0].into() + local[1].into());
    }
}

fn fib_trace(h: usize) -> RowMajorMatrix<Val> {
    let mut values = Vec::with_capacity(h * 2);
    let (mut a, mut b) = (Val::ONE, Val::ONE);
    for _ in 0..h {
        values.push(a);
        values.push(b);
        let next_a = a + b;
        b = a;
        a = next_a;
    }
    RowMajorMatrix::new(values, 2)
}

fn build_stark_config<Dft>(
    dft: Dft,
) -> StarkConfig<TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>, Challenge, Challenger>
where
    Dft: Clone + TwoAdicSubgroupDft<Val>,
{
    let (val_mmcs, challenge_mmcs, perm24) = build_mmcs();
    let fri_params = build_fri_params(challenge_mmcs, 1);
    let pcs = TwoAdicFriPcs::new(dft, val_mmcs, fri_params);
    let challenger = Challenger::new(perm24);
    StarkConfig::new(pcs, challenger)
}

/// Step 3.c: StarkConfig built with `GpuPoseidon2Mmcs` as InputMmcs.
/// FriMmcs stays CPU (ExtensionMmcs wrapping ValMmcs) — GPU MMCS is
/// single-field-only; FRI fold commits are at much smaller sizes and
/// can stay on CPU without losing the win.
fn build_stark_config_gpu_mmcs<Dft>(
    dft: Dft,
    gpu_mmcs: GpuPoseidon2Mmcs,
    perm24: Perm24,
) -> StarkConfig<
    TwoAdicFriPcs<Val, Dft, GpuPoseidon2Mmcs, ChallengeMmcs>,
    Challenge,
    Challenger,
>
where
    Dft: Clone + TwoAdicSubgroupDft<Val>,
{
    let (_val_mmcs, challenge_mmcs, _) = build_mmcs();
    let fri_params = build_fri_params(challenge_mmcs, 1);
    let pcs = TwoAdicFriPcs::new(dft, gpu_mmcs, fri_params);
    let challenger = Challenger::new(perm24);
    StarkConfig::new(pcs, challenger)
}

fn bench_prove(c: &mut Criterion) {
    let mut group = c.benchmark_group("target_stack/prove");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(15));

    // Build the GPU MMCS once per process. The prove path drives it
    // through both the trace commit (single matrix) and the quotient
    // commit (multi-matrix, same-height chunks) — the two code paths
    // Step 3.c's adapter supports.
    let gpu_mmcs_shared: Option<(GpuPoseidon2Mmcs, Perm24)> = WgpuDevice::new()
        .ok()
        .map(Arc::new)
        .map(build_gpu_mmcs);
    if gpu_mmcs_shared.is_none() {
        eprintln!("prove: GPU unavailable; GPU-MMCS rows will be skipped");
    }

    // FibAir is width-2 so the real driver of AIR cost is log_h.
    // Step 3.c targets larger log_h to expose commit-phase cost.
    // log_h=18 is the Step 3 bench-gate shape; the others frame the
    // scale-up curve so the write-up can show the trend, not a single
    // spot measurement.
    for &log_h in &[10usize, 14, 16, 18] {
        let trace = fib_trace(1 << log_h);
        let param = format!("log_h={log_h}");

        group.bench_with_input(
            BenchmarkId::new("cpu_dft_cpu_mmcs", &param),
            &trace,
            |bencher, t| {
                let config = build_stark_config(Radix2DitParallel::<Val>::default());
                let air = FibAir;
                bencher.iter(|| {
                    let proof = prove(&config, &air, black_box(t.clone()), &[]);
                    verify(&config, &air, &proof, &[]).expect("verify");
                    black_box(proof);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("gpu_dft_cpu_mmcs", &param),
            &trace,
            |bencher, t| {
                let config = build_stark_config(GpuDft::<Val>::strict_gpu());
                let air = FibAir;
                bencher.iter(|| {
                    let proof = prove(&config, &air, black_box(t.clone()), &[]);
                    verify(&config, &air, &proof, &[]).expect("verify");
                    black_box(proof);
                });
            },
        );
        if let Some((gpu_mmcs, perm24)) = &gpu_mmcs_shared {
            let mmcs_clone = gpu_mmcs.clone();
            let perm24_clone = perm24.clone();
            group.bench_with_input(
                BenchmarkId::new("cpu_dft_gpu_mmcs", &param),
                &trace,
                |bencher, t| {
                    let config = build_stark_config_gpu_mmcs(
                        Radix2DitParallel::<Val>::default(),
                        mmcs_clone.clone(),
                        perm24_clone.clone(),
                    );
                    let air = FibAir;
                    bencher.iter(|| {
                        let proof =
                            prove(&config, &air, black_box(t.clone()), &[]);
                        verify(&config, &air, &proof, &[]).expect("verify");
                        black_box(proof);
                    });
                },
            );

            let mmcs_clone = gpu_mmcs.clone();
            let perm24_clone = perm24.clone();
            group.bench_with_input(
                BenchmarkId::new("gpu_dft_gpu_mmcs", &param),
                &trace,
                |bencher, t| {
                    let config = build_stark_config_gpu_mmcs(
                        GpuDft::<Val>::strict_gpu(),
                        mmcs_clone.clone(),
                        perm24_clone.clone(),
                    );
                    let air = FibAir;
                    bencher.iter(|| {
                        let proof =
                            prove(&config, &air, black_box(t.clone()), &[]);
                        verify(&config, &air, &proof, &[]).expect("verify");
                        black_box(proof);
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_coset_lde_batch,
    bench_commit_phase,
    bench_prove,
);
criterion_main!(benches);
