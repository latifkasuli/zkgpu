// BabyBear batched Stockham DIF stage (radix-8) for four-step leaf NTTs (portable)
// P = 2^31 - 2^27 + 1 = 2013265921
//
// Processes `batch_count` independent sub-NTTs of size `leaf_n` packed
// contiguously. Each R8 butterfly consumes 3 logical Stockham DIF stages
// (h, h+1, h+2) in one dispatch — halving memory round-trips vs 3 R4
// stages (which would be 1.5 dispatches).
//
// NVIDIA scale-up Tier 3 Option A (T3.A, 2026-04-17).
//
// Structure: Cooley-Tukey DIT decomposition of size-8 DFT,
// analogous to the R4 kernel's split-by-parity structure.
//   Split inputs by (i mod 2): evens {a_0, a_2, a_4, a_6}, odds {a_1, a_3, a_5, a_7}.
//   Compute DFT_4 on evens → E_0..E_3.
//   Compute DFT_4 on odds  → O_0..O_3.
//   Apply size-8 twiddles ω_8^k to O_k for k=0..3.
//   Combine: X_k   = E_k + ω_8^k·O_k  for k=0..3,
//            X_{k+4} = E_k - ω_8^k·O_k.
//   Apply outer Stockham twiddles w^1..w^7 (w = ω_N^(s·p)) to outputs 1..7.
//
// Input indexing: a_i at src[base + q + s·(p + i·m8)] for i=0..7.
// Output indexing: dst[base + q + s·(8p + i)] for i=0..7 (natural order).
//
// Twiddle layout per butterfly position p:
//   twiddles[twiddle_offset + 7p + k-1] = w^k for k=1..7
//   twiddles_prime[...] = corresponding Shoup quotients.

const P: u32 = 2013265921u;

@group(0) @binding(0) var<storage, read>       src: array<u32>;
@group(0) @binding(1) var<storage, read_write>  dst: array<u32>;
@group(0) @binding(2) var<storage, read>       twiddles: array<u32>;

struct BatchedR8Params {
    leaf_n: u32,
    s: u32,
    m8: u32,
    twiddle_offset: u32,
    batch_count: u32,
    omega8: u32,         // primitive 8th root of unity in the leaf: ω_{leaf_n}^(leaf_n/8)
    omega8_prime: u32,
    omega4: u32,         // = omega8^2 (primitive 4th root)
    omega4_prime: u32,
    omega8_cubed: u32,       // ω_8^3 = ω_8 · ω_4 (precomputed, saves an in-shader mul)
    omega8_cubed_prime: u32,
    groups_per_row: u32, // 2D-folded dispatch (Tier 1 Fix 2b pattern)
}

@group(0) @binding(3) var<uniform> params: BatchedR8Params;
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
fn batched_stockham_r8_butterfly(@builtin(global_invocation_id) gid: vec3<u32>) {
    // 2D-folded dispatch (Tier 1 Fix 2b pattern).
    let tid = gid.x + gid.y * params.groups_per_row * 256u;
    let s = params.s;
    let m8 = params.m8;
    let leaf_n = params.leaf_n;
    let r8_butterflies_per_leaf = leaf_n / 8u;

    let batch_idx = tid / r8_butterflies_per_leaf;
    if batch_idx >= params.batch_count {
        return;
    }

    let local_tid = tid % r8_butterflies_per_leaf;
    let base = batch_idx * leaf_n;

    let q = local_tid % s;
    let p = local_tid / s;

    if p >= m8 {
        return;
    }

    // --- Gather 8 inputs at stride s·m8 ---
    let a0 = src[base + q + s * p];
    let a1 = src[base + q + s * (p + m8)];
    let a2 = src[base + q + s * (p + 2u * m8)];
    let a3 = src[base + q + s * (p + 3u * m8)];
    let a4 = src[base + q + s * (p + 4u * m8)];
    let a5 = src[base + q + s * (p + 5u * m8)];
    let a6 = src[base + q + s * (p + 6u * m8)];
    let a7 = src[base + q + s * (p + 7u * m8)];

    // --- DFT_4 on evens (a0, a2, a4, a6) — R4 structure from leaf_r4.wgsl ---
    let e_t0 = mod_add(a0, a4);
    let e_t1 = mod_sub(a0, a4);
    let e_t2 = mod_add(a2, a6);
    let e_t3 = mod_mul_shoup(mod_sub(a2, a6), params.omega4, params.omega4_prime);
    let e0 = mod_add(e_t0, e_t2);
    let e1 = mod_add(e_t1, e_t3);
    let e2 = mod_sub(e_t0, e_t2);
    let e3 = mod_sub(e_t1, e_t3);

    // --- DFT_4 on odds (a1, a3, a5, a7) ---
    let o_t0 = mod_add(a1, a5);
    let o_t1 = mod_sub(a1, a5);
    let o_t2 = mod_add(a3, a7);
    let o_t3 = mod_mul_shoup(mod_sub(a3, a7), params.omega4, params.omega4_prime);
    let o0 = mod_add(o_t0, o_t2);
    let o1 = mod_add(o_t1, o_t3);
    let o2 = mod_sub(o_t0, o_t2);
    let o3 = mod_sub(o_t1, o_t3);

    // --- Apply size-8 twiddles ω_8^k to O_k (k=0..3) ---
    // ω_8^0 = 1 (pass-through), ω_8^1 = omega8, ω_8^2 = omega4, ω_8^3 = omega8_cubed.
    // o0_tw := o0 (unchanged)
    let o1_tw = mod_mul_shoup(o1, params.omega8, params.omega8_prime);
    let o2_tw = mod_mul_shoup(o2, params.omega4, params.omega4_prime);
    let o3_tw = mod_mul_shoup(o3, params.omega8_cubed, params.omega8_cubed_prime);

    // --- Combine to X_0..X_7 ---
    let x0 = mod_add(e0, o0);
    let x1 = mod_add(e1, o1_tw);
    let x2 = mod_add(e2, o2_tw);
    let x3 = mod_add(e3, o3_tw);
    let x4 = mod_sub(e0, o0);
    let x5 = mod_sub(e1, o1_tw);
    let x6 = mod_sub(e2, o2_tw);
    let x7 = mod_sub(e3, o3_tw);

    // --- Apply outer Stockham twiddles w^1..w^7 ---
    // Layout per butterfly position p: 7 consecutive values [w^1, w^2, ..., w^7].
    let tw_base = params.twiddle_offset + 7u * p;
    let w1 = twiddles[tw_base];
    let w2 = twiddles[tw_base + 1u];
    let w3 = twiddles[tw_base + 2u];
    let w4 = twiddles[tw_base + 3u];
    let w5 = twiddles[tw_base + 4u];
    let w6 = twiddles[tw_base + 5u];
    let w7 = twiddles[tw_base + 6u];
    let w1_prime = twiddles_prime[tw_base];
    let w2_prime = twiddles_prime[tw_base + 1u];
    let w3_prime = twiddles_prime[tw_base + 2u];
    let w4_prime = twiddles_prime[tw_base + 3u];
    let w5_prime = twiddles_prime[tw_base + 4u];
    let w6_prime = twiddles_prime[tw_base + 5u];
    let w7_prime = twiddles_prime[tw_base + 6u];

    // --- Store 8 outputs at stride s (output 0 is untwiddled: w^0 = 1) ---
    dst[base + q + s * (8u * p)]       = x0;
    dst[base + q + s * (8u * p + 1u)]  = mod_mul_shoup(x1, w1, w1_prime);
    dst[base + q + s * (8u * p + 2u)]  = mod_mul_shoup(x2, w2, w2_prime);
    dst[base + q + s * (8u * p + 3u)]  = mod_mul_shoup(x3, w3, w3_prime);
    dst[base + q + s * (8u * p + 4u)]  = mod_mul_shoup(x4, w4, w4_prime);
    dst[base + q + s * (8u * p + 5u)]  = mod_mul_shoup(x5, w5, w5_prime);
    dst[base + q + s * (8u * p + 6u)]  = mod_mul_shoup(x6, w6, w6_prime);
    dst[base + q + s * (8u * p + 7u)]  = mod_mul_shoup(x7, w7, w7_prime);
}
