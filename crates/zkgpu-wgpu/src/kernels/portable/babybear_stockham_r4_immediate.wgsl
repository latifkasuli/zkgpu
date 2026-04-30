// BabyBear Stockham autosort NTT kernel (radix-4 DIF, portable) —
// IMMEDIATE-bound params variant.
//
// Item #3 of `docs/research/zkgpu-wgpu-speed-opportunities.md` (Gate 2,
// pilot kernel). Pilot scope: this kernel only. The other Stockham
// kernels (R2 global, local fused) and the Poseidon2 plans still bind
// their per-stage params at a uniform-buffer entry; they migrate in
// follow-up commits if this pilot shows a measurable win.
//
// Algorithmic body is byte-identical to `babybear_stockham_r4.wgsl`.
// The only difference: `params: R4Params` is declared as `var<immediate>`
// instead of `@group(0) @binding(3) var<uniform>`. Concrete effects:
//
//   1. The bind group drops to 4 entries (0=src, 1=dst, 2=twiddles,
//      4=twiddles_prime — binding 3 is gone).
//   2. The pipeline layout sets `immediate_size = 32` (8 × u32).
//   3. The encoder calls `pass.set_immediates(0, &params_bytes)` once
//      per dispatch, writing directly to register-resident bytes
//      instead of binding a pre-built `wgpu::Buffer`.
//   4. Plan build no longer creates one `wgpu::Buffer` per R4 stage.
//
// Why this might be faster: the per-stage param block is small (32
// bytes) and read once per thread. Uniform-buffer access goes through
// a constant-cache load; immediates are direct register reads, the
// fastest possible path. Encoder-side, skipping one bind-group entry
// + one buffer allocation per stage shaves CPU cost. The wins compound
// for small `log_n` on integrated/mobile/browser, where encoder cost
// dominates.

const P: u32 = 2013265921u;

@group(0) @binding(0) var<storage, read>       src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<storage, read>       twiddles: array<u32>;

struct R4Params {
    n: u32,
    s: u32,
    m4: u32,
    twiddle_offset: u32,
    omega4: u32,
    omega4_prime: u32,
    // 2D-folded dispatch (see babybear_stockham_r2.wgsl for rationale).
    groups_per_row: u32,
    _pad1: u32,
}

// IMMEDIATE-mapped params block. Keep `binding(2)` for twiddles_prime
// at slot 4 (NOT 3) so the BGL layout stays stable across additions:
// the migration is "drop binding 3", not "renumber every binding".
var<immediate> params: R4Params;
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
fn stockham_r4_butterfly(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x + gid.y * params.groups_per_row * 256u;
    let s = params.s;
    let m4 = params.m4;

    let q = tid % s;
    let p = tid / s;

    if p >= m4 {
        return;
    }

    let a0 = src[q + s * p];
    let a1 = src[q + s * (p + m4)];
    let a2 = src[q + s * (p + 2u * m4)];
    let a3 = src[q + s * (p + 3u * m4)];

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

    dst[q + s * (4u * p)]       = mod_add(t0, t2);
    dst[q + s * (4u * p + 1u)]  = mod_mul_shoup(mod_add(t1, t3), w1, w1_prime);
    dst[q + s * (4u * p + 2u)]  = mod_mul_shoup(mod_sub(t0, t2), w2, w2_prime);
    dst[q + s * (4u * p + 3u)]  = mod_mul_shoup(mod_sub(t1, t3), w3, w3_prime);
}
