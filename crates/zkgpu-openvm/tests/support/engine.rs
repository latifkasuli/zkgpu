//! Dev-only `ZkgpuOpenVmEngine` — Stage 3 harness.
//!
//! This file is **not** part of the crate's public API. It lives in
//! `tests/support/` so both the integration test
//! (`prove_verify_openvm.rs`) and the prove bench
//! (`benches/openvm_prove.rs`, via `#[path]`) can share one copy of
//! the engine without publishing a `zkgpu_openvm::engine` module.
//!
//! # What this is
//!
//! A `StarkEngine` whose `StarkConfig` is the exact shape OpenVM's
//! own `BabyBearPoseidon2Engine` uses, with one substitution: the
//! base-field `ValMmcs` is replaced with [`OpenVmGpuMmcs`] (our
//! GPU-accelerated MMCS). Everything else — Poseidon2 constants,
//! challenger, extension field, RAP phase, FRI config wiring —
//! mirrors `openvm_stark_sdk::config::baby_bear_poseidon2` byte for
//! byte.
//!
//! The two engines' `StarkConfig` types are **not** the same Rust
//! type (their `Pcs` parameter differs: `CpuValMmcs` for the control
//! engine, `OpenVmGpuMmcs` for this one), so a proof from one is
//! not directly passable to the other's `verify` at the type level
//! — but the wire-level shapes match: commitment
//! (`Hash<Val, Val, 8>`), opening proof (`Vec<[Val; 8]>`),
//! challenger trajectory, and FRI parameters are constructed to be
//! bit-identical. Cross-engine verification is shape-compatible in
//! principle; in Stage 3 each engine is only tested round-tripping
//! against itself, so a GPU-prove / CPU-verify handoff test is a
//! future item.
//!
//! # Why not implement `StarkFriEngine`
//!
//! `StarkFriEngine::new(fri_params: FriParameters)` can't accept a
//! `WgpuDevice`. We'd need either a `thread_local` device slot or a
//! different trait seam. Both are ugly. Instead we impl only the
//! base `StarkEngine` trait — that's enough for
//! `prove_then_verify`, which is what the Stage 3 test and bench
//! need. We lose `run_test_fast` helpers but don't need them.
//!
//! # Why the Poseidon2 constants match OpenVM's defaults exactly
//!
//! We call `openvm_stark_sdk::config::baby_bear_poseidon2::{default_perm,
//! horizen_round_consts_16}` directly. Both are `pub`. This gives us
//! the Horizen round constants + Plonky3 Mat4 — the exact
//! combination OpenVM uses — without having to depend on `zkhash`
//! ourselves.

use std::sync::Arc;

use openvm_stark_backend::{
    config::StarkConfig,
    engine::StarkEngine,
    interaction::fri_log_up::FriLogUpPhase,
    keygen::MultiStarkKeygenBuilder,
    prover::{
        cpu::{CpuBackend, CpuDevice},
        MultiTraceStarkProver,
    },
};
use openvm_stark_sdk::config::{
    baby_bear_poseidon2::{default_perm, horizen_round_consts_16, Challenger},
    fri_params::{
        SecurityParameters, MAX_BATCH_SIZE_LOG_BLOWUP_1, MAX_BATCH_SIZE_LOG_BLOWUP_2,
        MAX_NUM_CONSTRAINTS,
    },
    FriParameters,
};

use p3_baby_bear::BabyBear;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::{extension::BinomialExtensionField, Field};
use p3_fri::{FriParameters as P3FriParameters, TwoAdicFriPcs};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};

use zkgpu_openvm::{
    babybear_openvm_params,
    config::{Perm, DIGEST_WIDTH, RATE, WIDTH},
    OpenVmGpuMmcs,
};
use zkgpu_wgpu::WgpuDevice;

// ---------------------------------------------------------------------------
// Type aliases — mirror openvm_stark_sdk::config::baby_bear_poseidon2
// ---------------------------------------------------------------------------

type Val = BabyBear;
type PackedVal = <Val as Field>::Packing;
type Challenge = BinomialExtensionField<Val, 4>;

/// CPU-side `ValMmcs` — identical shape to `openvm_stark_sdk`'s.
/// Used as the **inner** MMCS for the challenge (extension-field)
/// MMCS. Per Stage 3 scope, GPU acceleration covers the **base-field**
/// trace commit only; FRI-fold commits over the extension field stay
/// on the CPU, which means the inner MMCS of the `ExtensionMmcs`
/// here is a plain `MerkleTreeMmcs`, not our GPU adapter.
type CpuValMmcs = MerkleTreeMmcs<
    PackedVal,
    PackedVal,
    PaddingFreeSponge<Perm, WIDTH, RATE, DIGEST_WIDTH>,
    TruncatedPermutation<Perm, 2, DIGEST_WIDTH, WIDTH>,
    DIGEST_WIDTH,
>;

type ChallengeMmcsType = ExtensionMmcs<Val, Challenge, CpuValMmcs>;
type Dft = Radix2DitParallel<Val>;

/// Base-field PCS — this is where the GPU swap happens. Instead of
/// `TwoAdicFriPcs<..., CpuValMmcs, ChallengeMmcsType>` (OpenVM's
/// default), we use `OpenVmGpuMmcs` as the base-field MMCS. Same
/// trait surface, GPU-accelerated commit path.
type GpuPcs = TwoAdicFriPcs<Val, Dft, OpenVmGpuMmcs, ChallengeMmcsType>;

type GpuRapPhase = FriLogUpPhase<Val, Challenge, Challenger<Perm>>;

/// The Stage 3 `StarkConfig`. Layout-identical to
/// `BabyBearPoseidon2Config` except the `Pcs` param is `GpuPcs`.
pub type ZkgpuOpenVmConfig =
    StarkConfig<GpuPcs, GpuRapPhase, Challenge, Challenger<Perm>>;

// ---------------------------------------------------------------------------
// The engine
// ---------------------------------------------------------------------------

/// Dev-only `StarkEngine` that plugs `OpenVmGpuMmcs` into OpenVM's
/// StarkConfig. See module docs for the full story.
pub struct ZkgpuOpenVmEngine {
    pub fri_params: FriParameters,
    pub device: CpuDevice<ZkgpuOpenVmConfig>,
    pub perm: Perm,
    pub max_constraint_degree: usize,
}

impl ZkgpuOpenVmEngine {
    /// Build the engine. `gpu_device` is the shared wgpu device; the
    /// engine holds an `Arc` clone inside `OpenVmGpuMmcs`.
    pub fn new(
        gpu_device: Arc<WgpuDevice>,
        fri_params: FriParameters,
    ) -> Self {
        // Use OpenVM's exact default permutation + its round
        // constants so the Poseidon2 schedule matches bit-for-bit on
        // both CPU (control) and GPU (engine under test) paths.
        let perm = default_perm();
        let (ext, int) = horizen_round_consts_16();

        // --- GPU base-field MMCS ------------------------------------
        let zkgpu_params = babybear_openvm_params(&ext, &int);
        let val_mmcs = OpenVmGpuMmcs::new(
            gpu_device,
            perm.clone(),
            zkgpu_params.clone(),
            zkgpu_params,
            0,
        )
        .expect("ZkgpuOpenVmEngine: OpenVmGpuMmcs::new failed");

        // --- CPU challenge (extension-field) MMCS -------------------
        //
        // Same Poseidon2 constants; separate instance. This is
        // intentional — Stage 3 scope keeps the extension-field
        // commits on the CPU. Widening the GPU adapter to the
        // extension field is a future item.
        let cpu_hash =
            PaddingFreeSponge::<Perm, WIDTH, RATE, DIGEST_WIDTH>::new(perm.clone());
        let cpu_compress =
            TruncatedPermutation::<Perm, 2, DIGEST_WIDTH, WIDTH>::new(perm.clone());
        let cpu_val_mmcs = CpuValMmcs::new(cpu_hash, cpu_compress);
        let challenge_mmcs = ChallengeMmcsType::new(cpu_val_mmcs);

        // --- StarkConfig wiring (mirrors `config_from_perm`) --------
        let dft = Dft::default();
        let security_params = SecurityParameters::new_baby_bear_100_bits(fri_params);
        let SecurityParameters {
            fri_params: fri_p,
            log_up_params,
            deep_ali_params,
        } = security_params;
        let fri_config = P3FriParameters {
            log_blowup: fri_p.log_blowup,
            log_final_poly_len: fri_p.log_final_poly_len,
            num_queries: fri_p.num_queries,
            commit_proof_of_work_bits: fri_p.commit_proof_of_work_bits,
            query_proof_of_work_bits: fri_p.query_proof_of_work_bits,
            mmcs: challenge_mmcs,
        };
        let pcs = GpuPcs::new(dft, val_mmcs, fri_config);
        let challenger = Challenger::<Perm>::new(perm.clone());
        let rap_phase = FriLogUpPhase::new(log_up_params, fri_p.log_blowup);
        let config = ZkgpuOpenVmConfig::new(pcs, challenger, rap_phase, deep_ali_params);

        let max_constraint_degree = fri_params.max_constraint_degree();
        ZkgpuOpenVmEngine {
            device: CpuDevice::new(Arc::new(config), fri_params.log_blowup),
            perm,
            fri_params,
            max_constraint_degree,
        }
    }
}

impl StarkEngine for ZkgpuOpenVmEngine {
    type SC = ZkgpuOpenVmConfig;
    type PB = CpuBackend<Self::SC>;
    type PD = CpuDevice<Self::SC>;

    fn config(&self) -> &Self::SC {
        &self.device.config
    }

    fn device(&self) -> &CpuDevice<Self::SC> {
        &self.device
    }

    fn keygen_builder(&self) -> MultiStarkKeygenBuilder<'_, Self::SC> {
        let mut builder = MultiStarkKeygenBuilder::new(self.config());
        builder.set_max_constraint_degree(self.max_constraint_degree);
        let max_batch_size = if self.fri_params.log_blowup == 1 {
            MAX_BATCH_SIZE_LOG_BLOWUP_1
        } else {
            MAX_BATCH_SIZE_LOG_BLOWUP_2
        };
        builder.max_batch_size = Some(max_batch_size);
        builder.max_num_constraints = Some(MAX_NUM_CONSTRAINTS);
        builder
    }

    fn prover(&self) -> MultiTraceStarkProver<Self::SC> {
        MultiTraceStarkProver::new(
            CpuBackend::default(),
            self.device.clone(),
            self.new_challenger(),
        )
    }

    fn max_constraint_degree(&self) -> Option<usize> {
        Some(self.max_constraint_degree)
    }

    fn new_challenger(&self) -> Challenger<Perm> {
        Challenger::<Perm>::new(self.perm.clone())
    }
}
