// BabyBear workgroup-local Stockham NTT kernel (subgroup-accelerated DIT, native)
// P = 2^31 - 2^27 + 1 = 2013265921
//
// Uses DIT (decimation-in-time) Cooley-Tukey algorithm with three phases:
//   Phase 1a: Thread-local R4 butterfly for stages 0+1 (no memory, no barriers)
//   Phase 1b: Subgroup shuffle R2 butterflies for stages 2..2+subgroup_log-1
//             (register-only, no barriers — uses subgroupShuffleXor)
//   Phase 2:  Shared memory R2 butterflies for remaining stages (with barriers)
//   Phase 3:  Scatter to global memory
//
// Compared to the portable R4 DIF kernel (6 barriers), this kernel eliminates
// barriers for the first (2 + subgroup_log) stages by performing them entirely
// in registers and subgroup shuffles:
//   subgroup_size=32 (Apple, NVIDIA):  4 barriers (saves 2)
//   subgroup_size=64 (Adreno, AMD):    3 barriers (saves 3)
//   subgroup_size=128 (rare):          2 barriers (saves 4)
//
// Requires wgpu::Features::SUBGROUP and min_subgroup_size >= 32.
// Input is loaded with bit-reversal permutation (DIT requires bit-reversed
// input to produce natural-order output, matching the DIF kernel's I/O).

enable subgroups;

const P: u32 = 2013265921u;
const BLOCK_SIZE: u32 = 1024u;
const LOG_BLOCK: u32 = 10u;
const PADDED_SIZE: u32 = 1056u;

var<workgroup> shmem_a: array<u32, 1056>;
var<workgroup> shmem_b: array<u32, 1056>;

@group(0) @binding(0) var<storage, read>       src: array<u32>;
@group(0) @binding(1) var<storage, read_write>  dst: array<u32>;
@group(0) @binding(2) var<storage, read>       twiddles: array<u32>;

struct SubgroupLocalParams {
    stride: u32,
    omega4: u32,
    omega4_prime: u32,
    subgroup_log: u32,
}

@group(0) @binding(3) var<uniform> params: SubgroupLocalParams;
@group(0) @binding(4) var<storage, read> twiddles_prime: array<u32>;

fn padded_idx(i: u32) -> u32 {
    return i + (i >> 5u);
}

// ---------- Modular arithmetic (identical to portable kernel) ----------

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

// ---------- DIT twiddle table offset ----------
//
// Twiddle table stores DIT twiddles for stages 2..9 contiguously:
//   [stage2: 4 vals] [stage3: 8] [stage4: 16] [stage5: 32]
//   [stage6: 64] [stage7: 128] [stage8: 256] [stage9: 512]  = 1020 total.
// Offset for stage h (h >= 2) = 2^h - 4.

fn dit_tw_off(h: u32) -> u32 {
    return (1u << h) - 4u;
}

// ---------- Subgroup DIT R2 butterfly ----------
//
// Performs one DIT radix-2 stage via subgroupShuffleXor.
// Each thread holds 4 elements at positions [4t, 4t+1, 4t+2, 4t+3].
// For stage h (h >= 2), all 4 elements have the same is_top value
// (they differ only in bits 0-1, and h >= 2), so one shuffle + branch
// handles all 4 elements uniformly.
//
// DIT butterfly: top' = top + tw * bottom, bottom' = top - tw * bottom.
// After shuffle, top thread has (my_val=top, pe=bottom),
// bottom thread has (my_val=bottom, pe=top).

fn subgroup_dit_r2(
    h: u32, t: u32,
    e0: ptr<function, u32>, e1: ptr<function, u32>,
    e2: ptr<function, u32>, e3: ptr<function, u32>,
) {
    let mask_sh = 1u << (h - 2u);
    let j_base = (t << 2u) & ((1u << h) - 1u);
    let tw_base = dit_tw_off(h) + j_base;
    let is_top = (((t << 2u) >> h) & 1u) == 0u;

    let pe0 = subgroupShuffleXor(*e0, mask_sh);
    let pe1 = subgroupShuffleXor(*e1, mask_sh);
    let pe2 = subgroupShuffleXor(*e2, mask_sh);
    let pe3 = subgroupShuffleXor(*e3, mask_sh);

    let tw0 = twiddles[tw_base];       let tp0 = twiddles_prime[tw_base];
    let tw1 = twiddles[tw_base + 1u];  let tp1 = twiddles_prime[tw_base + 1u];
    let tw2 = twiddles[tw_base + 2u];  let tp2 = twiddles_prime[tw_base + 2u];
    let tw3 = twiddles[tw_base + 3u];  let tp3 = twiddles_prime[tw_base + 3u];

    if is_top {
        *e0 = mod_add(*e0, mod_mul_shoup(pe0, tw0, tp0));
        *e1 = mod_add(*e1, mod_mul_shoup(pe1, tw1, tp1));
        *e2 = mod_add(*e2, mod_mul_shoup(pe2, tw2, tp2));
        *e3 = mod_add(*e3, mod_mul_shoup(pe3, tw3, tp3));
    } else {
        *e0 = mod_sub(pe0, mod_mul_shoup(*e0, tw0, tp0));
        *e1 = mod_sub(pe1, mod_mul_shoup(*e1, tw1, tp1));
        *e2 = mod_sub(pe2, mod_mul_shoup(*e2, tw2, tp2));
        *e3 = mod_sub(pe3, mod_mul_shoup(*e3, tw3, tp3));
    }
}

// ---------- Shared memory DIT R2 butterflies ----------
//
// Each thread handles 2 of the 512 butterflies per stage.
// For stage h with distance d = 2^h:
//   butterfly bi -> group = bi >> h, k = bi & (d-1)
//   p_top = group * 2d + k, p_bot = p_top + d
//   twiddle index = k

fn dit_r2_a_to_b(h: u32, t: u32) {
    let d = 1u << h;
    let h_mask = d - 1u;
    let tw_off = dit_tw_off(h);

    let bi0 = t;
    let group0 = bi0 >> h;
    let k0 = bi0 & h_mask;
    let pt0 = (group0 << (h + 1u)) + k0;
    let pb0 = pt0 + d;

    let a0 = shmem_a[padded_idx(pt0)];
    let b0 = shmem_a[padded_idx(pb0)];
    let tb0 = mod_mul_shoup(b0, twiddles[tw_off + k0], twiddles_prime[tw_off + k0]);
    shmem_b[padded_idx(pt0)] = mod_add(a0, tb0);
    shmem_b[padded_idx(pb0)] = mod_sub(a0, tb0);

    let bi1 = t + 256u;
    let group1 = bi1 >> h;
    let k1 = bi1 & h_mask;
    let pt1 = (group1 << (h + 1u)) + k1;
    let pb1 = pt1 + d;

    let a1 = shmem_a[padded_idx(pt1)];
    let b1 = shmem_a[padded_idx(pb1)];
    let tb1 = mod_mul_shoup(b1, twiddles[tw_off + k1], twiddles_prime[tw_off + k1]);
    shmem_b[padded_idx(pt1)] = mod_add(a1, tb1);
    shmem_b[padded_idx(pb1)] = mod_sub(a1, tb1);
}

fn dit_r2_b_to_a(h: u32, t: u32) {
    let d = 1u << h;
    let h_mask = d - 1u;
    let tw_off = dit_tw_off(h);

    let bi0 = t;
    let group0 = bi0 >> h;
    let k0 = bi0 & h_mask;
    let pt0 = (group0 << (h + 1u)) + k0;
    let pb0 = pt0 + d;

    let a0 = shmem_b[padded_idx(pt0)];
    let b0 = shmem_b[padded_idx(pb0)];
    let tb0 = mod_mul_shoup(b0, twiddles[tw_off + k0], twiddles_prime[tw_off + k0]);
    shmem_a[padded_idx(pt0)] = mod_add(a0, tb0);
    shmem_a[padded_idx(pb0)] = mod_sub(a0, tb0);

    let bi1 = t + 256u;
    let group1 = bi1 >> h;
    let k1 = bi1 & h_mask;
    let pt1 = (group1 << (h + 1u)) + k1;
    let pb1 = pt1 + d;

    let a1 = shmem_b[padded_idx(pt1)];
    let b1 = shmem_b[padded_idx(pb1)];
    let tb1 = mod_mul_shoup(b1, twiddles[tw_off + k1], twiddles_prime[tw_off + k1]);
    shmem_a[padded_idx(pt1)] = mod_add(a1, tb1);
    shmem_a[padded_idx(pb1)] = mod_sub(a1, tb1);
}

// ---------- Entry point ----------

@compute @workgroup_size(256)
fn stockham_local_subgroup(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_id: u32,
    @builtin(subgroup_size) sg_size: u32,
) {
    let block_id = wg_id.x;
    let t = lid.x;
    let stride = params.stride;
    let p0 = t * 4u;

    // ================================================================
    // Bit-reversed gather into registers
    // ================================================================
    // DIT requires bit-reversed input to produce natural-order output.
    // reverseBits gives 32-bit reversal; >> 22 extracts the 10-bit reversal.
    var elem0 = src[block_id + (reverseBits(p0)      >> 22u) * stride];
    var elem1 = src[block_id + (reverseBits(p0 + 1u) >> 22u) * stride];
    var elem2 = src[block_id + (reverseBits(p0 + 2u) >> 22u) * stride];
    var elem3 = src[block_id + (reverseBits(p0 + 3u) >> 22u) * stride];

    // ================================================================
    // Phase 1a: Thread-local R4 DIT butterfly (stages 0+1)
    // ================================================================
    // Stage 0 (d=1, tw=1): pairs (elem0,elem1) and (elem2,elem3)
    // Stage 1 (d=2): pairs with tw=1 and tw=omega4
    //   Combined: standard radix-4 DIT butterfly.
    {
        let s0 = mod_add(elem0, elem1);
        let d0 = mod_sub(elem0, elem1);
        let s1 = mod_add(elem2, elem3);
        let d1 = mod_sub(elem2, elem3);
        let tw_d1 = mod_mul_shoup(d1, params.omega4, params.omega4_prime);
        elem0 = mod_add(s0, s1);
        elem1 = mod_add(d0, tw_d1);
        elem2 = mod_sub(s0, s1);
        elem3 = mod_sub(d0, tw_d1);
    }

    // ================================================================
    // Phase 1b: Subgroup shuffle R2 DIT stages
    // ================================================================
    // Stages 2-6 always execute (guaranteed by min_subgroup_size >= 32).
    // Stages 7-8 execute as subgroup stages for larger subgroup sizes.
    subgroup_dit_r2(2u, t, &elem0, &elem1, &elem2, &elem3);
    subgroup_dit_r2(3u, t, &elem0, &elem1, &elem2, &elem3);
    subgroup_dit_r2(4u, t, &elem0, &elem1, &elem2, &elem3);
    subgroup_dit_r2(5u, t, &elem0, &elem1, &elem2, &elem3);
    subgroup_dit_r2(6u, t, &elem0, &elem1, &elem2, &elem3);

    if params.subgroup_log >= 6u {
        subgroup_dit_r2(7u, t, &elem0, &elem1, &elem2, &elem3);
    }
    if params.subgroup_log >= 7u {
        subgroup_dit_r2(8u, t, &elem0, &elem1, &elem2, &elem3);
    }

    // ================================================================
    // Phase 2: Write to shared memory + remaining DIT stages
    // ================================================================
    shmem_a[padded_idx(p0)]       = elem0;
    shmem_a[padded_idx(p0 + 1u)]  = elem1;
    shmem_a[padded_idx(p0 + 2u)]  = elem2;
    shmem_a[padded_idx(p0 + 3u)]  = elem3;
    workgroupBarrier();

    // Shared memory stages: from (2 + subgroup_log) through 9.
    // Ping-pong: a -> b -> a -> b ...
    // Number of shmem stages = 8 - subgroup_log.
    // Scatter from shmem_b if odd count, shmem_a if even.
    //
    //   subgroup_log=5: stages 7,8,9 (3 shmem, odd)  -> scatter from shmem_b
    //   subgroup_log=6: stages 8,9   (2 shmem, even)  -> scatter from shmem_a
    //   subgroup_log=7: stage 9      (1 shmem, odd)   -> scatter from shmem_b

    if params.subgroup_log <= 5u {
        // 3 shared memory stages: 7, 8, 9
        dit_r2_a_to_b(7u, t);
        workgroupBarrier();
        dit_r2_b_to_a(8u, t);
        workgroupBarrier();
        dit_r2_a_to_b(9u, t);
        workgroupBarrier();

        // Scatter from shmem_b
        dst[block_id + p0 * stride]         = shmem_b[padded_idx(p0)];
        dst[block_id + (p0 + 1u) * stride]  = shmem_b[padded_idx(p0 + 1u)];
        dst[block_id + (p0 + 2u) * stride]  = shmem_b[padded_idx(p0 + 2u)];
        dst[block_id + (p0 + 3u) * stride]  = shmem_b[padded_idx(p0 + 3u)];
    } else if params.subgroup_log == 6u {
        // 2 shared memory stages: 8, 9
        dit_r2_a_to_b(8u, t);
        workgroupBarrier();
        dit_r2_b_to_a(9u, t);
        workgroupBarrier();

        // Scatter from shmem_a
        dst[block_id + p0 * stride]         = shmem_a[padded_idx(p0)];
        dst[block_id + (p0 + 1u) * stride]  = shmem_a[padded_idx(p0 + 1u)];
        dst[block_id + (p0 + 2u) * stride]  = shmem_a[padded_idx(p0 + 2u)];
        dst[block_id + (p0 + 3u) * stride]  = shmem_a[padded_idx(p0 + 3u)];
    } else {
        // 1 shared memory stage: 9
        dit_r2_a_to_b(9u, t);
        workgroupBarrier();

        // Scatter from shmem_b
        dst[block_id + p0 * stride]         = shmem_b[padded_idx(p0)];
        dst[block_id + (p0 + 1u) * stride]  = shmem_b[padded_idx(p0 + 1u)];
        dst[block_id + (p0 + 2u) * stride]  = shmem_b[padded_idx(p0 + 2u)];
        dst[block_id + (p0 + 3u) * stride]  = shmem_b[padded_idx(p0 + 3u)];
    }
}
