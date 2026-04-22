//! Phase 7 Step 1 — FRI commit root-hash parity test,
//! **Poseidon2 MMCS target stack**.
//!
//! Mirrors `fri_commit_parity_keccak.rs` but uses the Plonky3
//! Poseidon2-MMCS pattern from upstream `examples/src/types.rs`:
//! width-16 permutation for compression (`TruncatedPermutation`) and
//! width-24 permutation for the leaf sponge (`PaddingFreeSponge`).
//!
//! This is the fixture that Phase 7 Step 3 (GPU Poseidon2 Merkle
//! commit) is graded against. Today both sides of the commit pipeline
//! hash on CPU; only `dft_batch` / `coset_lde_batch` run on GPU via
//! `zkgpu_plonky3::GpuDft`. This test confirms bit-identical
//! commitment roots at that state. When Step 3 later adds GPU
//! Poseidon2 Merkle-commit, the same test is the witness that the
//! GPU hash path stays bit-compatible.

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_commit::{ExtensionMmcs, Pcs};
use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::PrimeCharacteristicRing;
use p3_field::extension::BinomialExtensionField;
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::rngs::{SmallRng, StdRng};
use rand::{RngExt, SeedableRng};

use zkgpu_plonky3::GpuDft;

// Mirrors Plonky3's `examples/src/types.rs` `Poseidon2MerkleMmcs`
// exactly — the canonical Plonky3 Poseidon2 MMCS config.
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

// Required by `Pcs` trait bound even though `commit` never draws
// from the challenger.
type DummyChallenger = p3_challenger::DuplexChallenger<Val, Perm24, 24, 16>;

const MMCS_SEED: u64 = 0x_5EED_CAFE_u64;
const TRACE_SEED_STD: u64 = 0x_0000_C105_EDDA_7Au64;
const TRACE_SEED_HIDING: u64 = 0x_BAD_0005_A11D_u64;

fn build_mmcs() -> (ValMmcs, ChallengeMmcs) {
    let mut rng = SmallRng::seed_from_u64(MMCS_SEED);
    let perm16 = Perm16::new_from_rng_128(&mut rng);
    let perm24 = Perm24::new_from_rng_128(&mut rng);
    let hash = Poseidon2Sponge::new(perm24);
    let compress = Poseidon2Compression::new(perm16);
    let val_mmcs = ValMmcs::new(hash, compress, 0);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    (val_mmcs, challenge_mmcs)
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

fn commit_with_dft<Dft>(
    dft: Dft,
    trace: RowMajorMatrix<Val>,
    log_blowup: usize,
) -> <ValMmcs as p3_commit::Mmcs<Val>>::Commitment
where
    Dft: TwoAdicSubgroupDft<Val> + Clone,
{
    let (val_mmcs, challenge_mmcs) = build_mmcs();
    let fri_params = build_fri_params(challenge_mmcs, log_blowup);
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

#[test]
fn fri_commit_roots_match_poseidon2_mmcs() {
    for &log_h in &[12usize, 14] {
        for &w in &[1usize, 4] {
            let trace = random_trace(log_h, w, TRACE_SEED_STD ^ (log_h as u64) ^ ((w as u64) << 16));
            let cpu_root = commit_with_dft(
                Radix2DitParallel::<Val>::default(),
                trace.clone(),
                1,
            );
            let gpu_root = commit_with_dft(GpuDft::<Val>::strict_gpu(), trace, 1);
            assert_eq!(
                cpu_root, gpu_root,
                "Poseidon2 MMCS commit root mismatch at log_h={log_h} w={w}"
            );
        }
    }
}

#[test]
fn fri_commit_roots_match_poseidon2_mmcs_hiding_blowup() {
    let trace = random_trace(12, 2, TRACE_SEED_HIDING);
    let cpu_root = commit_with_dft(Radix2DitParallel::<Val>::default(), trace.clone(), 2);
    let gpu_root = commit_with_dft(GpuDft::<Val>::strict_gpu(), trace, 2);
    assert_eq!(
        cpu_root, gpu_root,
        "Poseidon2 MMCS hiding-blowup commit root mismatch"
    );
}
