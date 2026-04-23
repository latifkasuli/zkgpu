//! OpenVM BabyBear Poseidon2 MMCS configuration.
//!
//! Mirrors the public OpenVM config at
//! `openvm-org/stark-backend/crates/stark-sdk/src/config/baby_bear_poseidon2.rs`:
//!
//! ```ignore
//! const WIDTH: usize = 16;
//! const RATE: usize = 8;
//! const DIGEST_WIDTH: usize = 8;
//! type Perm = Poseidon2BabyBear<WIDTH>;
//! type Hash<P> = PaddingFreeSponge<P, WIDTH, RATE, DIGEST_WIDTH>;
//! type Compress<P> = TruncatedPermutation<P, 2, DIGEST_WIDTH, WIDTH>;
//! type ValMmcs<P> = MerkleTreeMmcs<..., Hash<P>, Compress<P>, DIGEST_WIDTH>;
//! ```
//!
//! We rebuild the aliases locally from Plonky3 0.4.1 primitives
//! rather than pulling `openvm-stark-backend` as a dev-dependency —
//! the config is ~30 lines and rebuilding keeps this adapter's
//! dependency surface minimal. See the crate-level docs for the
//! rationale.

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{Hash, PaddingFreeSponge, TruncatedPermutation};

// --- OpenVM constants (exact values from stark-sdk/config) ---

/// Poseidon2 state width. OpenVM uses a single width for both leaf
/// and compression (unlike Plonky3's canonical W24-leaf + W16-compress
/// example config).
pub const WIDTH: usize = 16;

/// Leaf sponge rate (`PaddingFreeSponge` inner `RATE` parameter).
pub const RATE: usize = 8;

/// Merkle digest length in field elements.
pub const DIGEST_WIDTH: usize = 8;

/// Base field — BabyBear (`p = 2^31 - 2^27 + 1`).
pub type Val = BabyBear;

/// Permutation type — width-16 Plonky3-variant Poseidon2, single
/// instance shared by both leaf sponge and compression.
pub type Perm = Poseidon2BabyBear<WIDTH>;

/// Leaf sponge: `PaddingFreeSponge<Perm16, 16, 8, 8>`.
///
/// Named `LeafHash` (not `Hash`) to avoid colliding with
/// `p3_symmetric::Hash` — the commitment-wrapper type Plonky3 0.4.1
/// exports under the same name. OpenVM's config file at
/// `stark-sdk/src/config/baby_bear_poseidon2.rs` avoids this clash
/// by never importing both into the same module.
pub type LeafHash = PaddingFreeSponge<Perm, WIDTH, RATE, DIGEST_WIDTH>;

/// Tree compression: `TruncatedPermutation<Perm16, 2, 8, 16>`.
pub type Compress = TruncatedPermutation<Perm, 2, DIGEST_WIDTH, WIDTH>;

/// OpenVM's CPU MMCS type at Plonky3 0.4.1 — note this is a
/// 5-generic alias, not 6. Plonky3 0.4.1's `MerkleTreeMmcs` hardcodes
/// binary arity (N=2); the explicit `N` arity parameter + `cap_height`
/// constructor argument were both added in 0.5.x. OpenVM was on
/// 0.4.1 as of its v1.5.0 release.
pub type ValMmcs = MerkleTreeMmcs<
    <Val as p3_field::Field>::Packing,
    <Val as p3_field::Field>::Packing,
    LeafHash,
    Compress,
    DIGEST_WIDTH,
>;

/// The commitment type — `Hash<Val, Val, DIGEST_WIDTH>` (a thin
/// phantom-typed wrapper over `[Val; DIGEST_WIDTH]`). Matches
/// Plonky3 0.4.1's `MerkleTreeMmcs::Commitment` exactly. 0.5.x
/// changed this to `MerkleCap<Val, [Val; DIGEST_WIDTH]>` to carry
/// `cap_height > 0` commitments; under 0.4.1 the commitment is
/// always a single root digest, so the wrapper is unit-cardinality
/// by construction.
pub type Commitment = Hash<Val, Val, DIGEST_WIDTH>;

/// Proof type — `Vec<[Val; DIGEST_WIDTH]>` (one sibling per tree
/// level for binary N=2 trees).
pub type Proof = Vec<[Val; DIGEST_WIDTH]>;

/// Error type shared with the CPU MerkleTreeMmcs.
pub type Error = p3_merkle_tree::MerkleTreeError;

/// Plonky3 0.4.1's `MerkleTreeMmcs` has no `cap_height` parameter
/// (commitment is always a single root). We still keep this
/// constant for API symmetry with `zkgpu-plonky3::gpu_mmcs`: the
/// adapter's `new()` accepts a `cap_height` parameter and rejects
/// anything but `0`, so a caller migrating between the two
/// adapters sees the same shape.
pub const SUPPORTED_CAP_HEIGHT: usize = 0;
