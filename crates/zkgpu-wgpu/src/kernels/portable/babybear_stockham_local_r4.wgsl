// BabyBear workgroup-local Stockham NTT kernel (radix-4 DIF, portable)
// P = 2^31 - 2^27 + 1 = 2013265921
//
// Processes a BLOCK_SIZE-point NTT entirely within workgroup shared memory.
// Uses 5 radix-4 stage pairs for all 10 logical stages (LOG_BLOCK = 10).
// Twiddles stored as (w1, w2, w3) triples; all output multiplies use Shoup.
// All 256 threads are active during every R4 stage pair (BLOCK_SIZE/4 = 256
// R4 butterflies, matching WORKGROUP_SIZE exactly).
//
// Shared memory uses bank-conflict-free padding (1 extra slot per 32).

const P: u32 = 2013265921u;
const BLOCK_SIZE: u32 = 1024u;
const LOG_BLOCK: u32 = 10u;
const PADDED_SIZE: u32 = 1056u;

var<workgroup> shmem_a: array<u32, 1056>;
var<workgroup> shmem_b: array<u32, 1056>;

fn padded_idx(i: u32) -> u32 {
    return i + (i >> 5u);
}

@group(0) @binding(0) var<storage, read>       src: array<u32>;
@group(0) @binding(1) var<storage, read_write>  dst: array<u32>;
@group(0) @binding(2) var<storage, read>       twiddles: array<u32>;

struct LocalR4Params {
    stride: u32,
    omega4: u32,
    omega4_prime: u32,
    // NVIDIA scale-up Tier 1 Fix 2 (2026-04-16): 2D-folded dispatch.
    // Local kernel uses `block_id = wg_id.x + wg_id.y * groups_per_row`
    // so workloads with >65535 workgroups (log_n ≥ 25) dispatch
    // without hitting the wgpu per-dimension limit. See
    // babybear_stockham_r2.wgsl for the full rationale.
    groups_per_row: u32,
}

@group(0) @binding(3) var<uniform> params: LocalR4Params;
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


// Radix-4 butterfly: reads 4 from `rd`, writes 4 to `wr`.
// s4 = stride of first of the two combined stages.
// m4 = BLOCK_SIZE / (4 * s4) = number of R4 butterflies per column.
// tw_off = offset into twiddles buffer for w1 values.
// omega4 = primitive 4th root of unity for the BLOCK_SIZE-point NTT.
//
// For thread t: q = t % s4, p = t / s4.  Processes one R4 butterfly.
// With BLOCK_SIZE = 4 * WORKGROUP_SIZE = 1024, s4*m4 = 256 = WORKGROUP_SIZE,
// so all 256 threads are active in every R4 stage pair.

@compute @workgroup_size(256)
fn stockham_local_r4(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    // 2D-folded workgroup index (see struct comment).
    let block_id = wg_id.x + wg_id.y * params.groups_per_row;
    let t = lid.x;
    let stride = params.stride;
    let omega4 = params.omega4;

    // --- Gather: load BLOCK_SIZE elements from global memory (strided) ---
    // Each thread loads 4 elements (1024 elements / 256 threads).
    shmem_a[padded_idx(t * 4u)]       = src[block_id + (t * 4u) * stride];
    shmem_a[padded_idx(t * 4u + 1u)]  = src[block_id + (t * 4u + 1u) * stride];
    shmem_a[padded_idx(t * 4u + 2u)]  = src[block_id + (t * 4u + 2u) * stride];
    shmem_a[padded_idx(t * 4u + 3u)]  = src[block_id + (t * 4u + 3u) * stride];
    workgroupBarrier();

    // ====================================================================
    // R4 stage pair 0: combines stages 0+1 (s4=1, m4=256)
    // shmem_a → shmem_b
    // ====================================================================
    {
        let s4 = 1u;
        let m4 = 256u;
        let tw_off = 0u;

        let q = t % s4;
        let p = t / s4;

        let a0 = shmem_a[padded_idx(q + s4 * p)];
        let a1 = shmem_a[padded_idx(q + s4 * (p + m4))];
        let a2 = shmem_a[padded_idx(q + s4 * (p + 2u * m4))];
        let a3 = shmem_a[padded_idx(q + s4 * (p + 3u * m4))];

        let tw_base = tw_off + 3u * p;
        let w1       = twiddles[tw_base];
        let w2       = twiddles[tw_base + 1u];
        let w3       = twiddles[tw_base + 2u];
        let w1_prime = twiddles_prime[tw_base];
        let w2_prime = twiddles_prime[tw_base + 1u];
        let w3_prime = twiddles_prime[tw_base + 2u];

        let t0 = mod_add(a0, a2);
        let t1 = mod_sub(a0, a2);
        let t2 = mod_add(a1, a3);
        let t3 = mod_mul_shoup(mod_sub(a1, a3), omega4, params.omega4_prime);

        shmem_b[padded_idx(q + s4 * (4u * p))]       = mod_add(t0, t2);
        shmem_b[padded_idx(q + s4 * (4u * p + 1u))]  = mod_mul_shoup(mod_add(t1, t3), w1, w1_prime);
        shmem_b[padded_idx(q + s4 * (4u * p + 2u))]  = mod_mul_shoup(mod_sub(t0, t2), w2, w2_prime);
        shmem_b[padded_idx(q + s4 * (4u * p + 3u))]  = mod_mul_shoup(mod_sub(t1, t3), w3, w3_prime);
    }
    workgroupBarrier();

    // ====================================================================
    // R4 stage pair 1: combines stages 2+3 (s4=4, m4=64)
    // shmem_b → shmem_a
    // ====================================================================
    {
        let s4 = 4u;
        let m4 = 64u;
        let tw_off = 768u;

        let q = t % s4;
        let p = t / s4;

        let a0 = shmem_b[padded_idx(q + s4 * p)];
        let a1 = shmem_b[padded_idx(q + s4 * (p + m4))];
        let a2 = shmem_b[padded_idx(q + s4 * (p + 2u * m4))];
        let a3 = shmem_b[padded_idx(q + s4 * (p + 3u * m4))];

        let tw_base = tw_off + 3u * p;
        let w1       = twiddles[tw_base];
        let w2       = twiddles[tw_base + 1u];
        let w3       = twiddles[tw_base + 2u];
        let w1_prime = twiddles_prime[tw_base];
        let w2_prime = twiddles_prime[tw_base + 1u];
        let w3_prime = twiddles_prime[tw_base + 2u];

        let t0 = mod_add(a0, a2);
        let t1 = mod_sub(a0, a2);
        let t2 = mod_add(a1, a3);
        let t3 = mod_mul_shoup(mod_sub(a1, a3), omega4, params.omega4_prime);

        shmem_a[padded_idx(q + s4 * (4u * p))]       = mod_add(t0, t2);
        shmem_a[padded_idx(q + s4 * (4u * p + 1u))]  = mod_mul_shoup(mod_add(t1, t3), w1, w1_prime);
        shmem_a[padded_idx(q + s4 * (4u * p + 2u))]  = mod_mul_shoup(mod_sub(t0, t2), w2, w2_prime);
        shmem_a[padded_idx(q + s4 * (4u * p + 3u))]  = mod_mul_shoup(mod_sub(t1, t3), w3, w3_prime);
    }
    workgroupBarrier();

    // ====================================================================
    // R4 stage pair 2: combines stages 4+5 (s4=16, m4=16)
    // shmem_a → shmem_b
    // ====================================================================
    {
        let s4 = 16u;
        let m4 = 16u;
        let tw_off = 960u;

        let q = t % s4;
        let p = t / s4;

        let a0 = shmem_a[padded_idx(q + s4 * p)];
        let a1 = shmem_a[padded_idx(q + s4 * (p + m4))];
        let a2 = shmem_a[padded_idx(q + s4 * (p + 2u * m4))];
        let a3 = shmem_a[padded_idx(q + s4 * (p + 3u * m4))];

        let tw_base = tw_off + 3u * p;
        let w1       = twiddles[tw_base];
        let w2       = twiddles[tw_base + 1u];
        let w3       = twiddles[tw_base + 2u];
        let w1_prime = twiddles_prime[tw_base];
        let w2_prime = twiddles_prime[tw_base + 1u];
        let w3_prime = twiddles_prime[tw_base + 2u];

        let t0 = mod_add(a0, a2);
        let t1 = mod_sub(a0, a2);
        let t2 = mod_add(a1, a3);
        let t3 = mod_mul_shoup(mod_sub(a1, a3), omega4, params.omega4_prime);

        shmem_b[padded_idx(q + s4 * (4u * p))]       = mod_add(t0, t2);
        shmem_b[padded_idx(q + s4 * (4u * p + 1u))]  = mod_mul_shoup(mod_add(t1, t3), w1, w1_prime);
        shmem_b[padded_idx(q + s4 * (4u * p + 2u))]  = mod_mul_shoup(mod_sub(t0, t2), w2, w2_prime);
        shmem_b[padded_idx(q + s4 * (4u * p + 3u))]  = mod_mul_shoup(mod_sub(t1, t3), w3, w3_prime);
    }
    workgroupBarrier();

    // ====================================================================
    // R4 stage pair 3: combines stages 6+7 (s4=64, m4=4)
    // shmem_b → shmem_a
    // ====================================================================
    {
        let s4 = 64u;
        let m4 = 4u;
        let tw_off = 1008u;

        let q = t % s4;
        let p = t / s4;

        let a0 = shmem_b[padded_idx(q + s4 * p)];
        let a1 = shmem_b[padded_idx(q + s4 * (p + m4))];
        let a2 = shmem_b[padded_idx(q + s4 * (p + 2u * m4))];
        let a3 = shmem_b[padded_idx(q + s4 * (p + 3u * m4))];

        let tw_base = tw_off + 3u * p;
        let w1       = twiddles[tw_base];
        let w2       = twiddles[tw_base + 1u];
        let w3       = twiddles[tw_base + 2u];
        let w1_prime = twiddles_prime[tw_base];
        let w2_prime = twiddles_prime[tw_base + 1u];
        let w3_prime = twiddles_prime[tw_base + 2u];

        let t0 = mod_add(a0, a2);
        let t1 = mod_sub(a0, a2);
        let t2 = mod_add(a1, a3);
        let t3 = mod_mul_shoup(mod_sub(a1, a3), omega4, params.omega4_prime);

        shmem_a[padded_idx(q + s4 * (4u * p))]       = mod_add(t0, t2);
        shmem_a[padded_idx(q + s4 * (4u * p + 1u))]  = mod_mul_shoup(mod_add(t1, t3), w1, w1_prime);
        shmem_a[padded_idx(q + s4 * (4u * p + 2u))]  = mod_mul_shoup(mod_sub(t0, t2), w2, w2_prime);
        shmem_a[padded_idx(q + s4 * (4u * p + 3u))]  = mod_mul_shoup(mod_sub(t1, t3), w3, w3_prime);
    }
    workgroupBarrier();

    // ====================================================================
    // R4 stage pair 4: combines stages 8+9 (s4=256, m4=1)
    // shmem_a → shmem_b
    // ====================================================================
    {
        let s4 = 256u;
        let m4 = 1u;
        let tw_off = 1020u;

        let q = t % s4;
        let p = t / s4;

        let a0 = shmem_a[padded_idx(q + s4 * p)];
        let a1 = shmem_a[padded_idx(q + s4 * (p + m4))];
        let a2 = shmem_a[padded_idx(q + s4 * (p + 2u * m4))];
        let a3 = shmem_a[padded_idx(q + s4 * (p + 3u * m4))];

        let tw_base = tw_off + 3u * p;
        let w1       = twiddles[tw_base];
        let w2       = twiddles[tw_base + 1u];
        let w3       = twiddles[tw_base + 2u];
        let w1_prime = twiddles_prime[tw_base];
        let w2_prime = twiddles_prime[tw_base + 1u];
        let w3_prime = twiddles_prime[tw_base + 2u];

        let t0 = mod_add(a0, a2);
        let t1 = mod_sub(a0, a2);
        let t2 = mod_add(a1, a3);
        let t3 = mod_mul_shoup(mod_sub(a1, a3), omega4, params.omega4_prime);

        shmem_b[padded_idx(q + s4 * (4u * p))]       = mod_add(t0, t2);
        shmem_b[padded_idx(q + s4 * (4u * p + 1u))]  = mod_mul_shoup(mod_add(t1, t3), w1, w1_prime);
        shmem_b[padded_idx(q + s4 * (4u * p + 2u))]  = mod_mul_shoup(mod_sub(t0, t2), w2, w2_prime);
        shmem_b[padded_idx(q + s4 * (4u * p + 3u))]  = mod_mul_shoup(mod_sub(t1, t3), w3, w3_prime);
    }
    workgroupBarrier();

    // --- Scatter: write BLOCK_SIZE elements back to global memory (strided) ---
    // After 5 stage pairs (odd), result is in shmem_b.
    // Each thread scatters 4 elements.
    dst[block_id + (t * 4u) * stride]       = shmem_b[padded_idx(t * 4u)];
    dst[block_id + (t * 4u + 1u) * stride]  = shmem_b[padded_idx(t * 4u + 1u)];
    dst[block_id + (t * 4u + 2u) * stride]  = shmem_b[padded_idx(t * 4u + 2u)];
    dst[block_id + (t * 4u + 3u) * stride]  = shmem_b[padded_idx(t * 4u + 3u)];
}
