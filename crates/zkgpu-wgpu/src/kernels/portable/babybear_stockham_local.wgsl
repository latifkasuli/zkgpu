// BabyBear workgroup-local Stockham NTT kernel (radix-2 DIF, portable)
// P = 2^31 - 2^27 + 1 = 2013265921
//
// Processes a BLOCK_SIZE-point NTT entirely within workgroup shared memory.
// Each workgroup handles one independent sub-problem:
//   1. Gather: strided read from global src into shared memory
//   2. Local stages: LOG_BLOCK DIF Stockham stages via shared memory ping-pong
//   3. Scatter: strided write from shared memory to global dst
//
// After the global DIF stages reduce the full N-point NTT to N/BLOCK_SIZE
// independent BLOCK_SIZE-point sub-problems, this kernel finishes the job
// in a single dispatch.
//
// Shared memory uses bank-conflict-free padding: one extra slot per 32
// elements eliminates the 100% conflict rate that power-of-2 butterfly
// strides cause on 32-bank GPUs (Adreno, AMD, NVIDIA, Intel).

const P: u32 = 2013265921u;
const BLOCK_SIZE: u32 = 512u;
const LOG_BLOCK: u32 = 9u;
const PADDED_SIZE: u32 = 528u; // 512 + 512/32

var<workgroup> shmem_a: array<u32, 528>;
var<workgroup> shmem_b: array<u32, 528>;

fn padded_idx(i: u32) -> u32 {
    return i + (i >> 5u);
}

@group(0) @binding(0) var<storage, read>       src: array<u32>;
@group(0) @binding(1) var<storage, read_write>  dst: array<u32>;
@group(0) @binding(2) var<storage, read>       twiddles: array<u32>;

struct LocalParams {
    stride: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(3) var<uniform> params: LocalParams;

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
fn stockham_local(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let block_id = wg_id.x;
    let t = lid.x;
    let stride = params.stride;

    // --- Gather: load BLOCK_SIZE elements from global memory (strided) ---
    // Sub-problem `block_id` has its elements at global indices:
    //   block_id, block_id + stride, block_id + 2*stride, ...
    shmem_a[padded_idx(t * 2u)] = src[block_id + (t * 2u) * stride];
    shmem_a[padded_idx(t * 2u + 1u)] = src[block_id + (t * 2u + 1u) * stride];
    workgroupBarrier();

    // --- Local DIF Stockham stages (LOG_BLOCK iterations) ---
    // Stage h=0: read shmem_a, write shmem_b
    // Stage h=1: read shmem_b, write shmem_a
    // ...alternating each stage.
    //
    // Twiddle offset for stage h: BLOCK_SIZE - (BLOCK_SIZE >> h)
    //   h=0 → 0, h=1 → 256, h=2 → 384, ...

    // Stage 0: s=1, m=256
    {
        let tw_off = 0u;

        let a = shmem_a[padded_idx(t)];
        let b = shmem_a[padded_idx(t + 256u)];
        let tw = twiddles[tw_off + t];

        shmem_b[padded_idx(t * 2u)] = mod_add(a, b);
        shmem_b[padded_idx(t * 2u + 1u)] = mod_mul(mod_sub(a, b), tw);
    }
    workgroupBarrier();

    // Stage 1: s=2, m=128
    {
        let s_local = 2u;
        let m_local = 128u;
        let tw_off = 256u;

        let q = t % s_local;
        let p = t / s_local;

        if p < m_local {
            let sa = q + s_local * p;
            let sb = q + s_local * (p + m_local);
            let d0 = q + s_local * (2u * p);
            let d1 = q + s_local * (2u * p + 1u);

            let a = shmem_b[padded_idx(sa)];
            let b = shmem_b[padded_idx(sb)];
            let tw = twiddles[tw_off + p];

            shmem_a[padded_idx(d0)] = mod_add(a, b);
            shmem_a[padded_idx(d1)] = mod_mul(mod_sub(a, b), tw);
        }
    }
    workgroupBarrier();

    // Stage 2: s=4, m=64
    {
        let s_local = 4u;
        let m_local = 64u;
        let tw_off = 384u;

        let q = t % s_local;
        let p = t / s_local;

        if p < m_local {
            let sa = q + s_local * p;
            let sb = q + s_local * (p + m_local);
            let d0 = q + s_local * (2u * p);
            let d1 = q + s_local * (2u * p + 1u);

            let a = shmem_a[padded_idx(sa)];
            let b = shmem_a[padded_idx(sb)];
            let tw = twiddles[tw_off + p];

            shmem_b[padded_idx(d0)] = mod_add(a, b);
            shmem_b[padded_idx(d1)] = mod_mul(mod_sub(a, b), tw);
        }
    }
    workgroupBarrier();

    // Stage 3: s=8, m=32
    {
        let s_local = 8u;
        let m_local = 32u;
        let tw_off = 448u;

        let q = t % s_local;
        let p = t / s_local;

        if p < m_local {
            let sa = q + s_local * p;
            let sb = q + s_local * (p + m_local);
            let d0 = q + s_local * (2u * p);
            let d1 = q + s_local * (2u * p + 1u);

            let a = shmem_b[padded_idx(sa)];
            let b = shmem_b[padded_idx(sb)];
            let tw = twiddles[tw_off + p];

            shmem_a[padded_idx(d0)] = mod_add(a, b);
            shmem_a[padded_idx(d1)] = mod_mul(mod_sub(a, b), tw);
        }
    }
    workgroupBarrier();

    // Stage 4: s=16, m=16
    {
        let s_local = 16u;
        let m_local = 16u;
        let tw_off = 480u;

        let q = t % s_local;
        let p = t / s_local;

        if p < m_local {
            let sa = q + s_local * p;
            let sb = q + s_local * (p + m_local);
            let d0 = q + s_local * (2u * p);
            let d1 = q + s_local * (2u * p + 1u);

            let a = shmem_a[padded_idx(sa)];
            let b = shmem_a[padded_idx(sb)];
            let tw = twiddles[tw_off + p];

            shmem_b[padded_idx(d0)] = mod_add(a, b);
            shmem_b[padded_idx(d1)] = mod_mul(mod_sub(a, b), tw);
        }
    }
    workgroupBarrier();

    // Stage 5: s=32, m=8
    {
        let s_local = 32u;
        let m_local = 8u;
        let tw_off = 496u;

        let q = t % s_local;
        let p = t / s_local;

        if p < m_local {
            let sa = q + s_local * p;
            let sb = q + s_local * (p + m_local);
            let d0 = q + s_local * (2u * p);
            let d1 = q + s_local * (2u * p + 1u);

            let a = shmem_b[padded_idx(sa)];
            let b = shmem_b[padded_idx(sb)];
            let tw = twiddles[tw_off + p];

            shmem_a[padded_idx(d0)] = mod_add(a, b);
            shmem_a[padded_idx(d1)] = mod_mul(mod_sub(a, b), tw);
        }
    }
    workgroupBarrier();

    // Stage 6: s=64, m=4
    {
        let s_local = 64u;
        let m_local = 4u;
        let tw_off = 504u;

        let q = t % s_local;
        let p = t / s_local;

        if p < m_local {
            let sa = q + s_local * p;
            let sb = q + s_local * (p + m_local);
            let d0 = q + s_local * (2u * p);
            let d1 = q + s_local * (2u * p + 1u);

            let a = shmem_a[padded_idx(sa)];
            let b = shmem_a[padded_idx(sb)];
            let tw = twiddles[tw_off + p];

            shmem_b[padded_idx(d0)] = mod_add(a, b);
            shmem_b[padded_idx(d1)] = mod_mul(mod_sub(a, b), tw);
        }
    }
    workgroupBarrier();

    // Stage 7: s=128, m=2
    {
        let s_local = 128u;
        let m_local = 2u;
        let tw_off = 508u;

        let q = t % s_local;
        let p = t / s_local;

        if p < m_local {
            let sa = q + s_local * p;
            let sb = q + s_local * (p + m_local);
            let d0 = q + s_local * (2u * p);
            let d1 = q + s_local * (2u * p + 1u);

            let a = shmem_b[padded_idx(sa)];
            let b = shmem_b[padded_idx(sb)];
            let tw = twiddles[tw_off + p];

            shmem_a[padded_idx(d0)] = mod_add(a, b);
            shmem_a[padded_idx(d1)] = mod_mul(mod_sub(a, b), tw);
        }
    }
    workgroupBarrier();

    // Stage 8: s=256, m=1
    {
        let s_local = 256u;
        let m_local = 1u;
        let tw_off = 510u;

        let q = t;
        let p = 0u;

        if q < s_local {
            let a = shmem_a[padded_idx(q)];
            let b = shmem_a[padded_idx(q + s_local)];
            let tw = twiddles[tw_off];

            shmem_b[padded_idx(q)] = mod_add(a, b);
            shmem_b[padded_idx(q + s_local)] = mod_mul(mod_sub(a, b), tw);
        }
    }
    workgroupBarrier();

    // --- Scatter: write BLOCK_SIZE elements back to global memory (strided) ---
    // After 9 stages (odd), result is in shmem_b.
    dst[block_id + (t * 2u) * stride] = shmem_b[padded_idx(t * 2u)];
    dst[block_id + (t * 2u + 1u) * stride] = shmem_b[padded_idx(t * 2u + 1u)];
}
