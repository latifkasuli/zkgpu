# zkgpu-plonky3 — future work

Two pieces of work that are deliberately out of scope for the current
adapter but are known next steps, listed in the order we'd most
likely tackle them. Neither blocks the narrow claim recorded in
[`docs/research/plonky3-poseidon2-mmcs.md`](../../docs/research/plonky3-poseidon2-mmcs.md).

---

## 1. Mixed-height multi-matrix MMCS adapter

**Priority: second.** Do this if the goal is adoption and a stronger
Plonky3 integration story — it broadens the valid use envelope.

### What's in scope today

`GpuPoseidon2Mmcs::commit` handles:

- single-matrix commits (trace commit in `uni-stark::prove`),
- same-height multi-matrix commits (the quotient-chunk batch produced
  by `uni-stark::commit_quotient`).

It rejects anything else loudly (`panic!("same-height ..."`) rather
than silently falling back to CPU.

### What this doesn't cover

Anything Plonky3's `MerkleTreeMmcs::commit` can validly accept where
the heights differ. Concretely:

- preprocessing / fixed / random matrices committed alongside the
  trace at different heights,
- mixed-height batches in any future Plonky3 consumer that doesn't
  happen to line up with same-height chunks.

### Why it's non-trivial

Plonky3 handles mixed heights via a DAG-shaped Merkle tree rather
than a straight binary compression (see
`p3_merkle_tree::merkle_tree::MerkleTree::new` and
`compress_and_inject`, lines ~316-419). Matrices are sorted by
height, the tallest ones are hashed first into `digest_layers[0]`,
and then at each compression level any matrices whose height rounds
up to the next layer size are *injected* into the compression inputs
alongside the previous layer's compressed digests. The result is a
binary compression at most tree levels but an N-ary step at levels
where injection happens.

Our current kernel stack is binary-only (width-16 Poseidon2 as
`TruncatedPermutation<Perm16, 2, 8, 16>`). Supporting injection means
one of:

1. A new GPU compression kernel that accepts `N > 2` inputs at levels
   with injection. Keeps the fast single-dispatch per level model but
   needs a second compile-time variant of the compress plan.
2. A host-side DAG walker that composes the existing width-16 binary
   compression with an injected-hash step at injection levels,
   paying an extra dispatch per injection level. Simpler, smaller
   blast radius, probably enough at target-stack depths.

Opening semantics also change: the proof walks the DAG, so the
arity per layer becomes variable and the retained-layer indexing in
`GpuProverData::open_batch` needs an `arity_schedule` like the one
Plonky3 tracks.

### Suggested incremental path

1. Extend `WgpuPoseidon2MerkleCommit` with an injection-aware variant
   (option 2 above) that takes a pre-sorted `Vec<(height, flattened-
   matrix)>` and walks levels one at a time, composing binary
   compression with injection-hash steps as Plonky3 does.
2. Update `GpuPoseidon2Mmcs::commit` to accept `inputs` of differing
   heights and drive the new backend path.
3. Update `GpuProverData<M>` to store the `arity_schedule` and teach
   `open_batch` to walk it.
4. Extend the parity test suite in
   `tests/poseidon2_mmcs_gpu.rs` with mixed-height shapes, pinned
   byte-for-byte against Plonky3's `MerkleTreeMmcs::{commit,
   open_batch, verify_batch}`.

### Success criterion

The rejection test `mmcs_commit_rejects_mixed_height` flips to a
parity test, and the adapter becomes a drop-in replacement for
Plonky3's `Poseidon2MerkleMmcs` at full-API level — not just the
target-stack subset.

---

## 2. GPU-resident `coset_lde_batch` (Step 2)

**Priority: last.** Do this only if you specifically want pipeline
cleanliness or to set up a future fully-GPU-resident end-to-end
commit path. Worthwhile, but not strategically urgent given the
current bench data.

### Why it's no longer strategically urgent

The Step 3 and Step 3.c bench data (RTX 4090 + Ryzen 9 7950X)
already established that the DFT/LDE is a small fraction of prove
time at the shapes consumers care about:

- `fri_commit @ log_h=18, w=8`: `gpu_dft_cpu_mmcs` = 0.99× of
  `cpu_dft_cpu_mmcs`, i.e. no measurable win from GPU DFT on its own.
- Full `prove @ log_h=18`: `gpu_dft_gpu_mmcs` is within 1% of
  `cpu_dft_gpu_mmcs` across the whole log_h sweep.

So replacing the CPU LDE with a GPU-resident LDE doesn't produce a
headline speedup number on its own. The MMCS swap is where the
prove-time win lives.

### Why it's still worth doing eventually

1. **Pipeline cleanup.** Today each trace column goes through:

   ```
   host → GPU DFT → host → CPU coset scaling → host → GPU MMCS
   ```

   There's a host round-trip between the LDE output and the leaf
   sponge input. A GPU-resident path would feed the GPU LDE output
   straight into `WgpuPoseidon2MerkleCommit::commit` without the
   intermediate download. The Merkle leaf plan already has a
   GPU-resident entry point (`hash_rows(device, input_buf, digests,
   num_leaves, row_width)`); Step 2 plugs into that directly.

2. **Groundwork for a GPU-resident full commit.** The retained-layers
   commit already stays GPU-resident through leaves → compression →
   root readback. If the LDE is also GPU-resident, the whole
   `commit()` pipeline runs end-to-end on device with no host bounce
   between stages. That pays off more under a future consumer that
   calls `commit()` many times back-to-back (e.g. recursive proofs),
   where the per-call host allocation cost compounds.

3. **Memory footprint.** The current host round-trip materialises
   the full LDE matrix on the host heap. A GPU-resident path skips
   that allocation.

### What's already in place

- `zkgpu_wgpu::WgpuPoseidon2MerkleLeafPlan::hash_rows` — accepts a
  GPU-resident `WgpuBuffer<BabyBear>` matrix and writes digests to a
  GPU-resident output buffer. This is the Step 2 handoff point.
- `GpuDft<BabyBear>` — the existing `TwoAdicSubgroupDft` adapter.
  Today it downloads the LDE to host after each call. Step 2 would
  add a sibling entry point that returns the GPU buffer directly,
  or keeps it internally until a sibling `commit_gpu` call consumes
  it.

### Suggested incremental path

1. Add a GPU-resident `coset_lde_batch` entry point that returns
   `WgpuBuffer<BabyBear>` (or an opaque handle) instead of a host
   `RowMajorMatrix`.
2. Add a parallel `GpuPoseidon2Mmcs::commit_gpu_resident` that takes
   such a handle and skips the host flatten step in `commit`.
3. Microbench the GPU-resident commit path at `log_h ∈ {16, 18, 20}`
   and document the delta vs the current host-mediated path. Expected
   to be single-digit percent at target-stack sizes.

### Success criterion

End-to-end `commit()` walks from Plonky3-facing input to root
readback without any intermediate host copy of the LDE matrix, and
the bench harness records the delta so future consumers (recursive
STARKs, batch proofs) can reason about when the round-trip cost
matters.

---

## Out-of-scope notes

- **MSM and BN254 / pairing-based SNARKs.** Tracked at the workspace
  roadmap level, not here. `zkgpu-plonky3` is intentionally STARK-
  family only.
- **Sum-check / zkVM consumer adapters** (Jolt, Ceno, SP1 Hypercube,
  etc.). Would live in separate crates — `zkgpu-jolt`,
  `zkgpu-ceno`, etc. — and reuse the primitive layer. Not inside
  this crate.
- **ICICLE-style retained-layer cutoff** ("keep upper layers,
  recompute lower on demand"). Current retained-layer cost is
  ~16 MiB at h=2¹⁸, ~256 MiB at h=2²². Cutoff becomes interesting
  around h=2²²; not needed for the narrow claim.
