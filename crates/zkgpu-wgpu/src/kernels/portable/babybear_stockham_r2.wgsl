// BabyBear Stockham autosort NTT kernel (radix-2 DIF, portable)
// P = 2^31 - 2^27 + 1 = 2013265921
//
// Out-of-place per stage: reads from src, writes to dst with permuted
// indices that absorb the bit-reversal into the stage write pattern.
// The host ping-pongs src/dst bindings each stage.

const P: u32 = 2013265921u;

@group(0) @binding(0) var<storage, read>       src: array<u32>;
@group(0) @binding(1) var<storage, read_write>  dst: array<u32>;
@group(0) @binding(2) var<storage, read>       twiddles: array<u32>;

struct StockhamParams {
    n: u32,
    s: u32,
    m: u32,
    twiddle_offset: u32,
}

@group(0) @binding(3) var<uniform> params: StockhamParams;
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
    // Compute (a * b) mod P for a, b < P < 2^31.
    //
    // WGSL has no native u64, so we emulate the 64-bit product using
    // 16-bit limb schoolbook multiplication, then reduce using the
    // algebraic structure of P = 2^31 - 2^27 + 1.

    // --- Step 1: 64-bit product (hi, lo) via 16-bit limb split ---
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

    // --- Step 2: Reduce (hi * 2^32 + lo) mod P iteratively ---
    //
    // Key identity: 2^32 ≡ 2^28 - 2 (mod P)
    //   because P = 2^31 - 2^27 + 1, so 2*P = 2^32 - 2^28 + 2,
    //   hence 2^32 = 2*P + 2^28 - 2.
    //
    // Each round: hi' = hi >> 4, lo' = lo + (hi & 0xF)*2^28 - 2*hi
    // Shrinks hi by factor ~16. Converges in at most 10 rounds.

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
fn stockham_butterfly(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let s = params.s;
    let m = params.m;

    // Decode (q, p) from flat thread index.
    // Total threads per stage = m * s = n / 2.
    let q = tid % s;
    let p = tid / s;

    if p >= m {
        return;
    }

    let src_a = q + s * p;
    let src_b = q + s * (p + m);
    let dst_0 = q + s * (2u * p);
    let dst_1 = q + s * (2u * p + 1u);

    let a = src[src_a];
    let b = src[src_b];
    let tw = twiddles[params.twiddle_offset + p];
    let tw_prime = twiddles_prime[params.twiddle_offset + p];

    dst[dst_0] = mod_add(a, b);
    dst[dst_1] = mod_mul_shoup(mod_sub(a, b), tw, tw_prime);
}
