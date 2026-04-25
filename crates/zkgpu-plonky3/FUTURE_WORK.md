# zkgpu-plonky3 — future work

The mixed-height adapter and the same-height fast path that
recovers post-convergence NVIDIA performance both shipped (see
`docs/research/plonky3-poseidon2-mmcs.md` for the latest numbers).
One remaining item, kept here for visibility.

---

## 1. GPU-resident `coset_lde_batch` (Step 2)

**Priority: low.** Do this only if you specifically want pipeline
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
