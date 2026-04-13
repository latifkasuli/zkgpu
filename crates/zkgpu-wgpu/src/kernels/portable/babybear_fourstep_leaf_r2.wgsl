// BabyBear batched Stockham DIF stage for four-step leaf NTTs (portable)
// P = 2^31 - 2^27 + 1 = 2013265921
//
// Same butterfly as babybear_stockham_r2.wgsl, but processes `batch_count`
// independent sub-NTTs of size `leaf_n` packed contiguously:
//   batch 0: [0 .. leaf_n), batch 1: [leaf_n .. 2*leaf_n), ...
//
// Thread global_id maps to (batch_idx, butterfly_idx) within the batch.

const P: u32 = 2013265921u;

@group(0) @binding(0) var<storage, read>       src: array<u32>;
@group(0) @binding(1) var<storage, read_write>  dst: array<u32>;
@group(0) @binding(2) var<storage, read>       twiddles: array<u32>;

struct BatchedStockhamParams {
    leaf_n: u32,
    s: u32,
    m: u32,
    twiddle_offset: u32,
    batch_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
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

fn mod_mul(a: u32, b: u32) -> u32 {
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
    var lo = ll + mid_lo_shifted;
    let lo_carry = select(0u, 1u, lo < ll);

    var hi = hh + (mid >> 16u) + (mid_carry << 16u) + lo_carry;

    for (var i = 0u; i < 10u; i = i + 1u) {
        if hi == 0u { break; }

        let h_new = hi >> 4u;
        let pos   = (hi & 0xFu) << 28u;
        let neg   = hi << 1u;

        var sum = lo + pos;
        let carry = select(0u, 1u, sum < lo);

        if sum >= neg {
            lo = sum - neg;
            hi = h_new + carry;
        } else {
            lo = sum - neg;
            hi = h_new + carry - 1u;
        }
    }

    if lo >= P { lo -= P; }
    if lo >= P { lo -= P; }
    return lo;
}

@compute @workgroup_size(256)
fn batched_stockham_butterfly(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let s = params.s;
    let m = params.m;
    let leaf_n = params.leaf_n;
    let butterflies_per_leaf = leaf_n / 2u;

    let batch_idx = tid / butterflies_per_leaf;
    if batch_idx >= params.batch_count {
        return;
    }

    let local_tid = tid % butterflies_per_leaf;
    let base = batch_idx * leaf_n;

    let q = local_tid % s;
    let p = local_tid / s;

    if p >= m {
        return;
    }

    let src_a = base + q + s * p;
    let src_b = base + q + s * (p + m);
    let dst_0 = base + q + s * (2u * p);
    let dst_1 = base + q + s * (2u * p + 1u);

    let a = src[src_a];
    let b = src[src_b];
    let tw = twiddles[params.twiddle_offset + p];
    let tw_prime = twiddles_prime[params.twiddle_offset + p];

    dst[dst_0] = mod_add(a, b);
    dst[dst_1] = mod_mul_shoup(mod_sub(a, b), tw, tw_prime);
}
