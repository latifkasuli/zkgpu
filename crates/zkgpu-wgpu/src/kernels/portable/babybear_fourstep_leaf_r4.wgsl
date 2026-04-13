// BabyBear batched Stockham DIF stage (radix-4) for four-step leaf NTTs (portable)
// P = 2^31 - 2^27 + 1 = 2013265921
//
// Same R4 butterfly as babybear_stockham_r4.wgsl, but processes `batch_count`
// independent sub-NTTs of size `leaf_n` packed contiguously.
// Thread global_id maps to (batch_idx, butterfly_idx) within the batch.

const P: u32 = 2013265921u;

@group(0) @binding(0) var<storage, read>       src: array<u32>;
@group(0) @binding(1) var<storage, read_write>  dst: array<u32>;
@group(0) @binding(2) var<storage, read>       twiddles: array<u32>;

struct BatchedR4Params {
    leaf_n: u32,
    s: u32,
    m4: u32,
    twiddle_offset: u32,
    batch_count: u32,
    omega4: u32,
    omega4_prime: u32,
    _pad0: u32,
}

@group(0) @binding(3) var<uniform> params: BatchedR4Params;
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
fn batched_stockham_r4_butterfly(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let s = params.s;
    let m4 = params.m4;
    let leaf_n = params.leaf_n;
    let r4_butterflies_per_leaf = leaf_n / 4u;

    let batch_idx = tid / r4_butterflies_per_leaf;
    if batch_idx >= params.batch_count {
        return;
    }

    let local_tid = tid % r4_butterflies_per_leaf;
    let base = batch_idx * leaf_n;

    let q = local_tid % s;
    let p = local_tid / s;

    if p >= m4 {
        return;
    }

    let a0 = src[base + q + s * p];
    let a1 = src[base + q + s * (p + m4)];
    let a2 = src[base + q + s * (p + 2u * m4)];
    let a3 = src[base + q + s * (p + 3u * m4)];

    let tw_base = params.twiddle_offset + 3u * p;
    let w1       = twiddles[tw_base];
    let w2       = twiddles[tw_base + 1u];
    let w3       = twiddles[tw_base + 2u];
    let w1_prime = twiddles_prime[tw_base];
    let w2_prime = twiddles_prime[tw_base + 1u];
    let w3_prime = twiddles_prime[tw_base + 2u];

    let t0 = mod_add(a0, a2);
    let t1 = mod_sub(a0, a2);
    let t2 = mod_add(a1, a3);
    let t3 = mod_mul_shoup(mod_sub(a1, a3), params.omega4, params.omega4_prime);

    dst[base + q + s * (4u * p)]       = mod_add(t0, t2);
    dst[base + q + s * (4u * p + 1u)]  = mod_mul_shoup(mod_add(t1, t3), w1, w1_prime);
    dst[base + q + s * (4u * p + 2u)]  = mod_mul_shoup(mod_sub(t0, t2), w2, w2_prime);
    dst[base + q + s * (4u * p + 3u)]  = mod_mul_shoup(mod_sub(t1, t3), w3, w3_prime);
}
