// BabyBear Poseidon2 permutation kernel — width 16, Plonky3 variant.
// P = 2^31 - 2^27 + 1 = 2013265921
//
// Phase 7 Step 1.5b. Sibling to babybear_poseidon2.wgsl (the
// HorizonLabs variant shipped in Phase F.1). Same structure, same
// width 16, same S-box. Single algorithmic difference: the 4x4 MDS
// matrix is Plonky3's circ(2, 3, 1, 1) instead of circ(2, 1, 1, 1).
//
// Required for Step 3 (GPU Poseidon2 Merkle commit) because Plonky3's
// Poseidon2MerkleMmcs uses width-16 Plonky3-variant Poseidon2 for
// node compression (`TruncatedPermutation<Poseidon2BabyBear<16>, ..>`).
//
// Differential tests in
// `zkgpu-plonky3/tests/poseidon2_bridge_gpu_w16.rs` assert
// element-wise parity with `p3_baby_bear::Poseidon2BabyBear<16>`.

const P: u32 = 2013265921u;
const WIDTH: u32 = 16u;
const M4_WIDTH: u32 = 4u;
const NUM_BLOCKS: u32 = 4u;  // 16 / 4
const WORKGROUP_SIZE: u32 = 64u;

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

fn sbox7(x: u32) -> u32 {
    let x2 = mod_mul(x, x);
    let x4 = mod_mul(x2, x2);
    let x6 = mod_mul(x4, x2);
    return mod_mul(x6, x);
}

// Plonky3's `M_4 = circ(2, 3, 1, 1)`. See the width-24 sibling kernel
// for the full derivation comment; arithmetic tree matches Plonky3's
// `apply_mat4` in `poseidon2/src/external.rs`.
fn m4_block_plonky3(x0: u32, x1: u32, x2: u32, x3: u32) -> vec4<u32> {
    let t01 = mod_add(x0, x1);
    let t23 = mod_add(x2, x3);
    let t0123 = mod_add(t01, t23);
    let t01123 = mod_add(t0123, x1);
    let t01233 = mod_add(t0123, x3);
    let dbl_x0 = mod_add(x0, x0);
    let dbl_x2 = mod_add(x2, x2);
    let y3 = mod_add(t01233, dbl_x0);
    let y1 = mod_add(t01123, dbl_x2);
    let y0 = mod_add(t01123, t01);
    let y2 = mod_add(t01233, t23);
    return vec4<u32>(y0, y1, y2, y3);
}

fn mul_external(state: ptr<function, array<u32, 16>>) {
    let b0 = m4_block_plonky3((*state)[0],  (*state)[1],  (*state)[2],  (*state)[3]);
    let b1 = m4_block_plonky3((*state)[4],  (*state)[5],  (*state)[6],  (*state)[7]);
    let b2 = m4_block_plonky3((*state)[8],  (*state)[9],  (*state)[10], (*state)[11]);
    let b3 = m4_block_plonky3((*state)[12], (*state)[13], (*state)[14], (*state)[15]);

    let c0 = mod_add(mod_add(b0.x, b1.x), mod_add(b2.x, b3.x));
    let c1 = mod_add(mod_add(b0.y, b1.y), mod_add(b2.y, b3.y));
    let c2 = mod_add(mod_add(b0.z, b1.z), mod_add(b2.z, b3.z));
    let c3 = mod_add(mod_add(b0.w, b1.w), mod_add(b2.w, b3.w));

    (*state)[0]  = mod_add(b0.x, c0);
    (*state)[1]  = mod_add(b0.y, c1);
    (*state)[2]  = mod_add(b0.z, c2);
    (*state)[3]  = mod_add(b0.w, c3);
    (*state)[4]  = mod_add(b1.x, c0);
    (*state)[5]  = mod_add(b1.y, c1);
    (*state)[6]  = mod_add(b1.z, c2);
    (*state)[7]  = mod_add(b1.w, c3);
    (*state)[8]  = mod_add(b2.x, c0);
    (*state)[9]  = mod_add(b2.y, c1);
    (*state)[10] = mod_add(b2.z, c2);
    (*state)[11] = mod_add(b2.w, c3);
    (*state)[12] = mod_add(b3.x, c0);
    (*state)[13] = mod_add(b3.y, c1);
    (*state)[14] = mod_add(b3.z, c2);
    (*state)[15] = mod_add(b3.w, c3);
}

struct Poseidon2Params {
    num_permutations: u32,
    rounds_f_half: u32,
    rounds_p: u32,
    row_stride: u32,
}

@group(0) @binding(0) var<storage, read_write> states: array<u32>;
@group(0) @binding(1) var<storage, read>       external_constants: array<u32>;
@group(0) @binding(2) var<storage, read>       internal_constants: array<u32>;
@group(0) @binding(3) var<storage, read>       internal_diagonal: array<u32>;
@group(0) @binding(4) var<uniform>             params: Poseidon2Params;

@compute @workgroup_size(64)
fn poseidon2_permute_w16(@builtin(global_invocation_id) gid: vec3<u32>) {
    let perm_idx = gid.x + gid.y * params.row_stride;
    if (perm_idx >= params.num_permutations) { return; }

    var state: array<u32, 16>;
    let base = perm_idx * WIDTH;
    for (var i = 0u; i < WIDTH; i = i + 1u) {
        state[i] = states[base + i];
    }

    mul_external(&state);

    for (var r = 0u; r < params.rounds_f_half; r = r + 1u) {
        let rc_base = r * WIDTH;
        for (var i = 0u; i < WIDTH; i = i + 1u) {
            state[i] = mod_add(state[i], external_constants[rc_base + i]);
        }
        for (var i = 0u; i < WIDTH; i = i + 1u) {
            state[i] = sbox7(state[i]);
        }
        mul_external(&state);
    }

    for (var r = 0u; r < params.rounds_p; r = r + 1u) {
        state[0] = mod_add(state[0], internal_constants[r]);
        state[0] = sbox7(state[0]);

        var sum: u32 = 0u;
        for (var i = 0u; i < WIDTH; i = i + 1u) {
            sum = mod_add(sum, state[i]);
        }
        for (var i = 0u; i < WIDTH; i = i + 1u) {
            let dd = mod_mul(internal_diagonal[i], state[i]);
            state[i] = mod_sub(mod_add(sum, dd), state[i]);
        }
    }

    for (var r = 0u; r < params.rounds_f_half; r = r + 1u) {
        let idx = params.rounds_f_half + r;
        let rc_base = idx * WIDTH;
        for (var i = 0u; i < WIDTH; i = i + 1u) {
            state[i] = mod_add(state[i], external_constants[rc_base + i]);
        }
        for (var i = 0u; i < WIDTH; i = i + 1u) {
            state[i] = sbox7(state[i]);
        }
        mul_external(&state);
    }

    for (var i = 0u; i < WIDTH; i = i + 1u) {
        states[base + i] = state[i];
    }
}
