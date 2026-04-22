//! Phase 7.2.b — FRI commit root-hash parity test.
//!
//! Builds a `TwoAdicFriPcs` twice with identical MMCS/FRI params,
//! once backed by `Radix2DitParallel` and once by `GpuDft::strict_gpu`.
//! Commits the same random trace through both. Asserts the
//! commitment roots are bit-identical — which means the full FRI
//! commit pipeline (coset_lde_batch → Merkle tree → root hash) produced
//! the same output with either DFT backend.
//!
//! This is the "GPU DFT validated end-to-end through FRI commit"
//! claim the Phase 7 spec promised. Stronger than the
//! `coset_lde_batch` differential in `lib.rs` because it verifies the
//! output survives the Merkle MMCS's bit-reversal + row-hashing
//! machinery.
//!
//! Config mirrors `p3-zk-proofs`'s `StandardBackend` so the result
//! transfers directly: if the commitment roots match here, the
//! `prove → verify` path also matches (the prover hashes the trace
//! into the same root the verifier opens against).

use p3_baby_bear::BabyBear;
use p3_commit::{ExtensionMmcs, Pcs};
use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::extension::BinomialExtensionField;
use p3_field::PrimeCharacteristicRing;
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_keccak::{Keccak256Hash, KeccakF};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use zkgpu_plonky3::GpuDft;

// Types mirror p3-zk-proofs/src/backend.rs exactly — we want the same
// config so a commit parity here extends to their prove/verify path.
type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;
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

// `p3-zk-proofs` uses these exact values (backend.rs:51-55).
const NUM_QUERIES: usize = 40;
const QUERY_POW_BITS: usize = 8;
const LOG_BLOWUP: usize = 1;

fn build_mmcs() -> (ValMmcs, ChallengeMmcs) {
    let u64_hash = U64Hash::new(KeccakF {});
    let field_hash = FieldHash::new(u64_hash);
    let compress = Compress::new(u64_hash);
    let val_mmcs = ValMmcs::new(field_hash, compress, 0);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    (val_mmcs, challenge_mmcs)
}

fn build_fri_params(challenge_mmcs: ChallengeMmcs) -> FriParameters<ChallengeMmcs> {
    FriParameters {
        log_blowup: LOG_BLOWUP,
        log_final_poly_len: 0,
        max_log_arity: 2,
        num_queries: NUM_QUERIES,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: QUERY_POW_BITS,
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

/// Commit a trace through `TwoAdicFriPcs` and return the commitment
/// root. Uses whatever DFT impl the caller hands in.
fn commit_with_dft<Dft>(dft: Dft, trace: RowMajorMatrix<Val>) -> <ValMmcs as p3_commit::Mmcs<Val>>::Commitment
where
    Dft: TwoAdicSubgroupDft<Val> + Clone,
{
    let (val_mmcs, challenge_mmcs) = build_mmcs();
    let fri_params = build_fri_params(challenge_mmcs);
    let pcs: TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs> =
        TwoAdicFriPcs::new(dft, val_mmcs, fri_params);

    let h = trace.height();
    let domain = <TwoAdicFriPcs<_, _, _, _> as Pcs<Challenge, DummyChallenger>>::natural_domain_for_degree(
        &pcs, h,
    );
    let (commitment, _prover_data) = <TwoAdicFriPcs<_, _, _, _> as Pcs<Challenge, DummyChallenger>>::commit(
        &pcs,
        vec![(domain, trace)],
    );
    commitment
}

// `Pcs::commit` is generic over a challenger type, but never invokes
// it — so any placeholder with the required trait bounds works. We
// can't easily inline-construct one, so we use a thin type alias to
// the same challenger p3-zk-proofs uses; we never actually run it.
type DummyChallenger = p3_challenger::SerializingChallenger32<
    Val,
    p3_challenger::HashChallenger<u8, Keccak256Hash, 32>,
>;

// -- Tests -------------------------------------------------------------------

/// Commit a random trace with both DFTs at matching config. Roots
/// must be bit-identical: the GPU DFT's LDE flows through the same
/// Merkle tree and must produce the same root.
#[test]
fn fri_commit_roots_match() {
    for &log_h in &[12usize, 14] {
        for &w in &[1usize, 4] {
            let trace = random_trace(log_h, w, 0x_7E57_C0DE_u64 + log_h as u64 + w as u64);

            let cpu_root = commit_with_dft(
                Radix2DitParallel::<Val>::default(),
                trace.clone(),
            );

            let gpu_root = commit_with_dft(GpuDft::<Val>::strict_gpu(), trace);

            assert_eq!(
                cpu_root, gpu_root,
                "commit root mismatch at log_h={log_h} w={w}: \
                 GPU DFT produced different LDE than Radix2DitParallel"
            );
        }
    }
}

/// Same as above but with a hiding-style `log_blowup = 2` to exercise
/// `coset_lde_batch(_, 2, _)` (the branch `HidingFriPcs::commit` uses).
#[test]
fn fri_commit_roots_match_hiding_blowup() {
    fn build_fri_params_hiding(challenge_mmcs: ChallengeMmcs) -> FriParameters<ChallengeMmcs> {
        FriParameters {
            log_blowup: LOG_BLOWUP + 1, // hiding uses +1 per backend.rs:54
            log_final_poly_len: 0,
            max_log_arity: 2,
            num_queries: NUM_QUERIES,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: QUERY_POW_BITS,
            mmcs: challenge_mmcs,
        }
    }

    fn commit_hiding<Dft>(dft: Dft, trace: RowMajorMatrix<Val>) -> <ValMmcs as p3_commit::Mmcs<Val>>::Commitment
    where
        Dft: TwoAdicSubgroupDft<Val> + Clone,
    {
        let (val_mmcs, challenge_mmcs) = build_mmcs();
        let fri_params = build_fri_params_hiding(challenge_mmcs);
        let pcs: TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs> =
            TwoAdicFriPcs::new(dft, val_mmcs, fri_params);

        let h = trace.height();
        let domain = <TwoAdicFriPcs<_, _, _, _> as Pcs<Challenge, DummyChallenger>>::natural_domain_for_degree(
            &pcs, h,
        );
        let (commitment, _prover_data) = <TwoAdicFriPcs<_, _, _, _> as Pcs<Challenge, DummyChallenger>>::commit(
            &pcs,
            vec![(domain, trace)],
        );
        commitment
    }

    let trace = random_trace(12, 2, 0x_B0AB0A_B0AB0A_u64);

    let cpu_root = commit_hiding(Radix2DitParallel::<Val>::default(), trace.clone());
    let gpu_root = commit_hiding(GpuDft::<Val>::strict_gpu(), trace);

    assert_eq!(
        cpu_root, gpu_root,
        "hiding-blowup commit root mismatch: \
         GPU DFT produced different LDE than Radix2DitParallel"
    );
}
