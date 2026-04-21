// BabyBear Stockham autosort NTT kernel — column-batched variant
// (Phase 7.5 Path B for zkgpu-plonky3).
//
// Same algorithm as babybear_stockham_r2.wgsl, but operates on a
// row-major matrix of shape `h × w` where each of the `w` columns is
// an independent polynomial of length `h`. A single dispatch performs
// one stage for all `w` columns simultaneously; threads are indexed
// as (butterfly_index, column_index).
//
// Memory layout: `buf[row * w + col]`. Consecutive threads access
// consecutive memory (column is the low bit of tid) → coalesced reads
// and writes.
//
// Twiddles are column-independent — each butterfly's twiddle depends
// only on its row pair, so every column of a given butterfly shares
// the same twiddle value. One `twiddles[params.twiddle_offset + p]`
// read serves all `w` column threads via the usual GPU cache path.
//
// P = 2^31 - 2^27 + 1 = 2013265921

const P: u32 = 2013265921u;

@group(0) @binding(0) var<storage, read>       src: array<u32>;
@group(0) @binding(1) var<storage, read_write>  dst: array<u32>;
@group(0) @binding(2) var<storage, read>       twiddles: array<u32>;

struct BatchedStockhamParams {
    // Height of each column polynomial (the NTT size).
    n: u32,
    // Butterfly-window parameters (same as the single-poly kernel).
    s: u32,
    m: u32,
    // Offset into the flat twiddle table for this stage.
    twiddle_offset: u32,
    // Number of columns in the batch (each an independent polynomial
    // of length `n`). Total butterfly-element threads = (n/2) * width.
    width: u32,
    // 2D-folded dispatch: reconstruct tid as
    //   tid = gid.x + gid.y * groups_per_row * WORKGROUP_SIZE
    // Same pattern as the single-poly kernel.
    groups_per_row: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(3) var<uniform> params: BatchedStockhamParams;
@group(0) @binding(4) var<storage, read> twiddles_prime: array<u32>;

fn mulhi(a: u32, b: u32) -> u32 {
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;

    let ll = a_lo * b_lo;
    let lh = a_lo * b_hi;
    let hl = a_hi * b_lo;
    let hh = a_hi * b_hi;

    let mid = lh + hl;
    let mid_carry = select(0u, 1u, mid < lh);

    let mid_lo_shifted = mid << 16u;
    let lo_sum = ll + mid_lo_shifted;
    let lo_carry = select(0u, 1u, lo_sum < ll);

    return hh + (mid >> 16u) + (mid_carry << 16u) + lo_carry;
}

fn mod_mul_shoup(a: u32, w: u32, w_prime: u32) -> u32 {
    let q = mulhi(a, w_prime);
    var r = a * w - q * P;
    if r >= P { r -= P; }
    return r;
}

fn mod_add(a: u32, b: u32) -> u32 {
    let sum = a + b;
    return select(sum, sum - P, sum >= P);
}

fn mod_sub(a: u32, b: u32) -> u32 {
    return select(a - b, a + P - b, a < b);
}

@compute @workgroup_size(256)
fn batched_stockham_r2_butterfly(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x + gid.y * params.groups_per_row * 256u;

    let s = params.s;
    let m = params.m;
    let w = params.width;

    // Total butterflies per stage = n/2 = m * s. Column is the
    // low-index dimension for coalesced memory access.
    let butterfly_idx = tid / w;
    let c = tid % w;

    // Decode (q, p) from butterfly_idx exactly like the single-poly
    // kernel. q spans [0, s), p spans [0, m).
    let q = butterfly_idx % s;
    let p = butterfly_idx / s;

    if p >= m {
        return;
    }

    // Row indices along the height dimension.
    let src_row_a = q + s * p;
    let src_row_b = q + s * (p + m);
    let dst_row_0 = q + s * (2u * p);
    let dst_row_1 = q + s * (2u * p + 1u);

    // Flat memory indices: row_major[row, c] = row * w + c.
    let a = src[src_row_a * w + c];
    let b = src[src_row_b * w + c];

    let tw = twiddles[params.twiddle_offset + p];
    let tw_prime = twiddles_prime[params.twiddle_offset + p];

    dst[dst_row_0 * w + c] = mod_add(a, b);
    dst[dst_row_1 * w + c] = mod_mul_shoup(mod_sub(a, b), tw, tw_prime);
}

// The inverse-scale pass is handled by reusing `babybear_scale.wgsl`
// with `n = h * w`. Every element in the batched buffer gets the same
// per-column 1/h scalar, so no batched-specific scale kernel is
// needed.
