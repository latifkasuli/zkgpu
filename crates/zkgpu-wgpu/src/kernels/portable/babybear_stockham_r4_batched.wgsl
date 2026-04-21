// BabyBear Stockham autosort NTT kernel — column-batched radix-4
// (Phase 7.5 Path C.1).
//
// Same R4 butterfly as babybear_stockham_r4.wgsl, extended to a
// row-major `h × pitch` buffer where each of `width` columns is an
// independent polynomial. Column is the fast dimension for
// coalesced memory access; `pitch >= width` is rounded up to a
// multiple of 8 by the Rust planner so row starts land on 32-byte
// boundaries.
//
// For combined stages (h, h+1) with s = 2^h:
//   Reads:  a_k = src[(q + s*(p + k*m4))*pitch + c]   for k = 0..3
//   Writes: y_k = dst[(q + s*(4*p + k))*pitch + c]    for k = 0..3
//
// Radix-4 DIF butterfly (identical to single-column):
//   t0 = a0 + a2,  t1 = a0 - a2
//   t2 = a1 + a3,  t3 = (a1 - a3) * omega4
//   y0 = t0 + t2
//   y1 = (t1 + t3) * w1
//   y2 = (t0 - t2) * w2
//   y3 = (t1 - t3) * w3

const P: u32 = 2013265921u;

@group(0) @binding(0) var<storage, read>       src: array<u32>;
@group(0) @binding(1) var<storage, read_write>  dst: array<u32>;
@group(0) @binding(2) var<storage, read>       twiddles: array<u32>;

struct R4BatchedParams {
    n: u32,
    s: u32,
    m4: u32,
    twiddle_offset: u32,
    omega4: u32,
    omega4_prime: u32,
    width: u32,
    pitch: u32,
    groups_per_row: u32,
    _pad0: u32,
}

@group(0) @binding(3) var<uniform> params: R4BatchedParams;
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
    let tid = gid.x + gid.y * params.groups_per_row * 256u;

    let s = params.s;
    let m4 = params.m4;
    let w_col = params.width;
    let pitch = params.pitch;

    // Column is the fast dimension (low bit of tid) for coalesced
    // memory access: consecutive threads access consecutive u32s.
    let c = tid % w_col;
    let butterfly_idx = tid / w_col;

    let q = butterfly_idx % s;
    let p = butterfly_idx / s;

    if p >= m4 {
        return;
    }

    // Row indices along the height dimension (same as single-column R4).
    let row_a = q + s * p;
    let row_b = q + s * (p + m4);
    let row_c = q + s * (p + 2u * m4);
    let row_d = q + s * (p + 3u * m4);

    let a0 = src[row_a * pitch + c];
    let a1 = src[row_b * pitch + c];
    let a2 = src[row_c * pitch + c];
    let a3 = src[row_d * pitch + c];

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

    let dst_row_0 = q + s * (4u * p);
    let dst_row_1 = q + s * (4u * p + 1u);
    let dst_row_2 = q + s * (4u * p + 2u);
    let dst_row_3 = q + s * (4u * p + 3u);

    dst[dst_row_0 * pitch + c] = mod_add(t0, t2);
    dst[dst_row_1 * pitch + c] = mod_mul_shoup(mod_add(t1, t3), w1, w1_prime);
    dst[dst_row_2 * pitch + c] = mod_mul_shoup(mod_sub(t0, t2), w2, w2_prime);
    dst[dst_row_3 * pitch + c] = mod_mul_shoup(mod_sub(t1, t3), w3, w3_prime);
}
