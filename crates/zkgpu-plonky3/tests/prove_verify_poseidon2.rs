//! Phase 7 Step 1 — end-to-end prove/verify with **Poseidon2 MMCS
//! target stack**.
//!
//! Mirrors `prove_verify_keccak.rs` but uses the Plonky3 Poseidon2
//! MMCS + `DuplexChallenger` configuration that is the Phase 7 target
//! (see `research/phase-7-plonky3-adapter/c4-target-stack/decision.md`).
//!
//! Today the DFT runs on GPU via `GpuDft::strict_gpu`, but the MMCS
//! hashes on CPU. This test is the correctness witness for the
//! target stack; Phase 7 Step 3 (GPU Poseidon2 Merkle commit) will
//! move the MMCS to GPU and this same test confirms end-to-end
//! parity survives that migration.

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_field::PrimeCharacteristicRing;
use p3_field::extension::BinomialExtensionField;
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::{StarkConfig, prove, verify};
use rand::rngs::SmallRng;
use rand::SeedableRng;

use zkgpu_plonky3::GpuDft;

type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;
type Perm16 = Poseidon2BabyBear<16>;
type Perm24 = Poseidon2BabyBear<24>;
type Poseidon2Sponge = PaddingFreeSponge<Perm24, 24, 16, 8>;
type Poseidon2Compression = TruncatedPermutation<Perm16, 2, 8, 16>;
type ValMmcs = MerkleTreeMmcs<
    <Val as p3_field::Field>::Packing,
    <Val as p3_field::Field>::Packing,
    Poseidon2Sponge,
    Poseidon2Compression,
    2,
    8,
>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Challenger = DuplexChallenger<Val, Perm24, 24, 16>;

const NUM_QUERIES: usize = 40;
const QUERY_POW_BITS: usize = 8;
const LOG_BLOWUP: usize = 1;
const MMCS_SEED: u64 = 0x_5EED_CAFE_u64;

// Reuse the FibAir from prove_verify_keccak.rs — trivial constraint,
// width 2. Defined inline here so the test file is self-contained.
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

fn build_config<Dft>(
    dft: Dft,
) -> StarkConfig<TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>, Challenge, Challenger>
where
    Dft: Clone + p3_dft::TwoAdicSubgroupDft<Val>,
{
    let mut rng = SmallRng::seed_from_u64(MMCS_SEED);
    let perm16 = Perm16::new_from_rng_128(&mut rng);
    let perm24 = Perm24::new_from_rng_128(&mut rng);

    let hash = Poseidon2Sponge::new(perm24.clone());
    let compress = Poseidon2Compression::new(perm16);
    let val_mmcs = ValMmcs::new(hash, compress, 0);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    let fri_params = FriParameters {
        log_blowup: LOG_BLOWUP,
        log_final_poly_len: 0,
        max_log_arity: 2,
        num_queries: NUM_QUERIES,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: QUERY_POW_BITS,
        mmcs: challenge_mmcs,
    };

    let pcs = TwoAdicFriPcs::new(dft, val_mmcs, fri_params);
    let challenger = Challenger::new(perm24);
    StarkConfig::new(pcs, challenger)
}

#[test]
fn prove_verify_poseidon2_mmcs_gpu_dft_strict() {
    let log_h = 6usize;
    let trace = fib_trace(1 << log_h);
    let config = build_config(GpuDft::<Val>::strict_gpu());
    let air = FibAir;
    let proof = prove(&config, &air, trace, &[]);
    verify(&config, &air, &proof, &[])
        .expect("verifier rejected GPU-backed Poseidon2-MMCS proof");
}

#[test]
fn prove_verify_poseidon2_mmcs_cpu_control() {
    let log_h = 6usize;
    let trace = fib_trace(1 << log_h);
    let config = build_config(p3_dft::Radix2DitParallel::<Val>::default());
    let air = FibAir;
    let proof = prove(&config, &air, trace, &[]);
    verify(&config, &air, &proof, &[])
        .expect("verifier rejected CPU-backed Poseidon2-MMCS proof");
}
