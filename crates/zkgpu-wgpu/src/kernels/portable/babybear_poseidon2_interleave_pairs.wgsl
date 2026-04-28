// Pair-interleave of two digest arrays for mixed-height MMCS injection.
//
// Takes two N-digest input arrays (`left`, `right`) of DIGEST_LEN u32s each
// and produces a 2N-digest output where pair i is `(left[i], right[i])`.
// Output layout:
//   out[2*i * DIGEST_LEN + k]                = left[i * DIGEST_LEN + k]
//   out[(2*i + 1) * DIGEST_LEN + k]          = right[i * DIGEST_LEN + k]
// for i in [0, N), k in [0, DIGEST_LEN).
//
// Used by the GPU-resident mixed-height commit DAG engine to merge an
// injected matrix's leaf hashes (`right`) with the previous level's
// pairwise compression output (`left`), before the second compression
// at injection levels. Replaces the previous host-side
// `interleave_pairs_host` memcpy + upload-back-to-GPU pattern.
//
// Workgroup size is 64 — small enough that very small N (e.g. N = 1,
// 2, 4 at the deepest injection levels) doesn't waste invocations,
// large enough that big N (e.g. N = 2^20+ on the tallest matrices)
// dispatches efficiently on every backend we target.
//
// Dispatch shape uses the standard 2D-fold pattern from the rest of
// the backend (see `plan_linear_dispatch` in `dispatch.rs`): when N
// requires more workgroups than fit in a single x-dimension dispatch
// (typical limit 65,535), they wrap into the y dimension. The kernel
// reconstructs the flat index as
//   i = gid.x + gid.y * groups_per_row * WORKGROUP_SIZE
// matching the convention used by the Stockham R2/R4, four-step leaf,
// and tile-transpose kernels. Without 2D folding, the largest mixed-
// height shape this engine can handle is bounded by
// (max_compute_workgroups_per_dimension * WORKGROUP_SIZE) digests
// per injection level; with folding, the only remaining bound is the
// `max_compute_workgroups_per_dimension^2` total-grid limit.

const DIGEST_LEN: u32 = 8u;
const WORKGROUP_SIZE: u32 = 64u;

@group(0) @binding(0) var<storage, read>       left: array<u32>;
@group(0) @binding(1) var<storage, read>       right: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;

struct InterleaveParams {
    /// Number of digests per input buffer. The output buffer holds 2*n
    /// digests = 2*n*DIGEST_LEN u32s.
    n: u32,
    /// Number of workgroups laid out across the x dimension before
    /// wrapping into y. Mirrors `LinearDispatch::groups_per_row`.
    groups_per_row: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(3) var<uniform> params: InterleaveParams;

@compute @workgroup_size(64)
fn interleave_pairs(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    // 2D-folded thread index reconstruction. For dispatches that fit
    // in one x-row (`gid.y == 0`), this collapses to `i = gid.x`.
    let i = gid.x + gid.y * params.groups_per_row * WORKGROUP_SIZE;
    if i >= params.n {
        return;
    }

    let in_off  = i * DIGEST_LEN;
    let out_off = (i * 2u) * DIGEST_LEN;

    // Unrolled DIGEST_LEN = 8 element copy. The WGSL compiler will
    // also vectorise this into u32x2 / u32x4 accesses on backends
    // where alignment permits.
    out[out_off + 0u]              = left[in_off + 0u];
    out[out_off + 1u]              = left[in_off + 1u];
    out[out_off + 2u]              = left[in_off + 2u];
    out[out_off + 3u]              = left[in_off + 3u];
    out[out_off + 4u]              = left[in_off + 4u];
    out[out_off + 5u]              = left[in_off + 5u];
    out[out_off + 6u]              = left[in_off + 6u];
    out[out_off + 7u]              = left[in_off + 7u];

    out[out_off + DIGEST_LEN + 0u] = right[in_off + 0u];
    out[out_off + DIGEST_LEN + 1u] = right[in_off + 1u];
    out[out_off + DIGEST_LEN + 2u] = right[in_off + 2u];
    out[out_off + DIGEST_LEN + 3u] = right[in_off + 3u];
    out[out_off + DIGEST_LEN + 4u] = right[in_off + 4u];
    out[out_off + DIGEST_LEN + 5u] = right[in_off + 5u];
    out[out_off + DIGEST_LEN + 6u] = right[in_off + 6u];
    out[out_off + DIGEST_LEN + 7u] = right[in_off + 7u];
}
