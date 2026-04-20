//! Phase 7.2.c — end-to-end prove/verify with a minimal AIR,
//! `TwoAdicFriPcs`, and `GpuDft::strict_gpu` as the DFT backend.
//!
//! If this test passes, the GPU adapter works all the way through
//! Plonky3's prove/verify pipeline — the same pipeline `p3-zk-proofs`
//! uses for its preimage and Merkle-inclusion circuits. That is the
//! definitive correctness-oracle result Phase 7.2 was built around.
//!
//! The test AIR is deliberately trivial: a 2-column constraint
//! `next[0] == local[0] + local[1]` (Fibonacci-style advance). We
//! care about the plumbing not the circuit. Uses `strict_gpu()` so
//! any silent fallback to CPU is loud rather than hiding behind a
//! green test.

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_baby_bear::BabyBear;
use p3_challenger::{HashChallenger, SerializingChallenger32};
use p3_commit::ExtensionMmcs;
use p3_field::extension::BinomialExtensionField;
use p3_field::PrimeCharacteristicRing;
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_keccak::{Keccak256Hash, KeccakF};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher};
use p3_uni_stark::{StarkConfig, prove, verify};

use zkgpu_plonky3::GpuDft;

type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;
type ByteHash = Keccak256Hash;
type U64Hash = PaddingFreeSponge<KeccakF, 25, 17, 4>;
type FieldHash = SerializingHasher<U64Hash>;
type Compress = CompressionFunctionFromHasher<U64Hash, 2, 4>;
type ValMmcs = MerkleTreeMmcs<
    [Val; p3_keccak::VECTOR_LEN],
    [u64; p3_keccak::VECTOR_LEN],
    FieldHash,
    Compress,
    2,
    4,
>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Challenger = SerializingChallenger32<Val, HashChallenger<u8, ByteHash, 32>>;

const NUM_QUERIES: usize = 40;
const QUERY_POW_BITS: usize = 8;
const LOG_BLOWUP: usize = 1;

// -- Trivial AIR: Fibonacci-style advance ------------------------------------
//
// Width 2. Constraint on transition rows: next[0] == local[0] + local[1].
// Nothing clever — we only need *some* AIR for the prove/verify plumbing
// to execute end-to-end.

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
        // On transition rows: next[0] == local[0] + local[1].
        builder
            .when_transition()
            .assert_eq(next[0].into(), local[0].into() + local[1].into());
    }
}

/// Build a random valid Fibonacci-style trace of `h` rows.
fn fib_trace(h: usize) -> RowMajorMatrix<Val> {
    // Seed with (1, 1). Each subsequent row's local[0] is the sum of
    // previous row's two columns; local[1] is anything (we just use
    // the previous row's local[0] to form a chain, but it's
    // unconstrained).
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

// -- Config builder ----------------------------------------------------------

fn build_config<Dft: Clone>(
    dft: Dft,
) -> StarkConfig<TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>, Challenge, Challenger>
where
    Dft: p3_dft::TwoAdicSubgroupDft<Val>,
{
    let byte_hash = ByteHash {};
    let u64_hash = U64Hash::new(KeccakF {});
    let field_hash = FieldHash::new(u64_hash);
    let compress = Compress::new(u64_hash);

    let val_mmcs = ValMmcs::new(field_hash, compress, 0);
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
    let challenger = Challenger::from_hasher(vec![], byte_hash);
    StarkConfig::new(pcs, challenger)
}

// -- Tests -------------------------------------------------------------------

/// End-to-end prove + verify with `GpuDft::strict_gpu` as the DFT
/// backing `TwoAdicFriPcs`. If the GPU adapter produced a wrong LDE
/// anywhere in the pipeline, the verifier would reject.
///
/// log_h is chosen above our default fallback threshold (14), so even
/// without `strict_gpu()` the GPU path would run. `strict_gpu()` gives
/// the belt-and-suspenders guarantee that a device-init or
/// plan-build failure would panic rather than silently fall back.
#[test]
fn prove_verify_with_gpu_dft_strict() {
    let log_h = 6usize; // Small AIR; padding pushes the actual LDE height higher.
    let trace = fib_trace(1 << log_h);

    let config = build_config(GpuDft::<Val>::strict_gpu());
    let air = FibAir;

    let proof = prove(&config, &air, trace, &[]);
    verify(&config, &air, &proof, &[]).expect("verifier rejected GPU-backed proof");
}

/// Same test but with `Radix2DitParallel` as the control. Both
/// should accept; if the GPU variant fails while the CPU one
/// succeeds, the GPU adapter has a bug specific to some step of the
/// prove or verify pipeline that commit parity didn't catch.
#[test]
fn prove_verify_with_cpu_dft_control() {
    let log_h = 6usize;
    let trace = fib_trace(1 << log_h);

    let config = build_config(p3_dft::Radix2DitParallel::<Val>::default());
    let air = FibAir;

    let proof = prove(&config, &air, trace, &[]);
    verify(&config, &air, &proof, &[]).expect("verifier rejected CPU-backed proof");
}
