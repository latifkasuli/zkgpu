// BabyBear Poseidon2 Merkle tree-compression kernel — width 16,
// Plonky3 variant.
// P = 2^31 - 2^27 + 1 = 2013265921
//
// Phase 7 Step 3.b. One thread per output digest. Each thread reads
// a pair of 8-element child digests from `input_digests`, runs
// `TruncatedPermutation<Perm16, 2, 8, 16>` — pack `left ∥ right` into a
// 16-element state, apply Plonky3 Poseidon2 W16 permutation, output
// the first 8 slots — and writes the result into `output_digests`.
//
// Matches Plonky3's semantics exactly (see p3_symmetric/src/compression.rs
// `TruncatedPermutation::compress`):
//   pre[0..CHUNK]       = left
//   pre[CHUNK..2*CHUNK] = right
//   post                = Perm16(pre)
//   out                 = post[0..CHUNK]
//
// Binding layout mirrors `babybear_poseidon2_merkle_leaf.wgsl`:
// packed `poseidon_constants` buffer with offsets in the uniform so
// the whole BGL fits inside the WebGPU baseline 4-storage cap.

const P: u32 = 2013265921u;
const WIDTH: u32 = 16u;
const DIGEST_LEN: u32 = 8u;
const M4_WIDTH: u32 = 4u;
const WORKGROUP_SIZE: u32 = 64u;

// --- BabyBear modular arithmetic (inlined per project convention) ---

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

// --- Plonky3 Poseidon2 W16 permutation (M_4 = circ(2, 3, 1, 1)) ---

fn sbox7(x: u32) -> u32 {
    let x2 = mod_mul(x, x);
    let x4 = mod_mul(x2, x2);
    let x6 = mod_mul(x4, x2);
    return mod_mul(x6, x);
}

fn m4_block_plonky3(x0: u32, x1: u32, x2: u32, x3: u32) -> vec4<u32> {
    let t01 = mod_add(x0, x1);
    let t23 = mod_add(x2, x3);
    let t0123 = mod_add(t01, t23);
    let t01123 = mod_add(t0123, x1);
    let t01233 = mod_add(t0123, x3);
    let dbl_x0 = mod_add(x0, x0);
    let dbl_x2 = mod_add(x2, x2);
    return vec4<u32>(
        mod_add(t01123, t01),    // y0
        mod_add(t01123, dbl_x2), // y1
        mod_add(t01233, t23),    // y2
        mod_add(t01233, dbl_x0), // y3
    );
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

// --- Bindings (3 storage + 1 uniform, fits WebGPU baseline cap=4) ---

struct MerkleCompressParams {
    // Number of output digests at this level (== input_digests / 2).
    num_outputs: u32,
    rounds_f_half: u32,
    rounds_p: u32,
    // 2D-fold dispatch: row_stride = groups_per_row * WORKGROUP_SIZE.
    row_stride: u32,
    // Offsets into `poseidon_constants`:
    //   external: 0 .. internal_rc_offset (2 * rounds_f_half * WIDTH)
    //   internal_rc: internal_rc_offset .. internal_diag_offset (rounds_p)
    //   internal_diag: internal_diag_offset .. (WIDTH)
    internal_rc_offset: u32,
    internal_diag_offset: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read>        input_digests: array<u32>;
@group(0) @binding(1) var<storage, read_write>  output_digests: array<u32>;
@group(0) @binding(2) var<storage, read>        poseidon_constants: array<u32>;
@group(0) @binding(3) var<uniform>              params: MerkleCompressParams;

@compute @workgroup_size(64)
fn merkle_compress(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_idx = gid.x + gid.y * params.row_stride;
    if (out_idx >= params.num_outputs) { return; }

    // Pack sibling pair into the 16-element state:
    //   state[0..8]  = input_digests[2*out_idx]
    //   state[8..16] = input_digests[2*out_idx + 1]
    var state: array<u32, 16>;
    let left_base  = out_idx * 2u * DIGEST_LEN;
    let right_base = left_base + DIGEST_LEN;
    for (var i = 0u; i < DIGEST_LEN; i = i + 1u) {
        state[i] = input_digests[left_base + i];
        state[i + DIGEST_LEN] = input_digests[right_base + i];
    }

    // --- Plonky3 Poseidon2 W16 permutation (inline, single-shot) ---
    let rc_ofs = params.internal_rc_offset;
    let diag_ofs = params.internal_diag_offset;

    mul_external(&state);

    for (var r = 0u; r < params.rounds_f_half; r = r + 1u) {
        let rc_base = r * WIDTH;
        for (var i = 0u; i < WIDTH; i = i + 1u) {
            state[i] = mod_add(state[i], poseidon_constants[rc_base + i]);
        }
        for (var i = 0u; i < WIDTH; i = i + 1u) {
            state[i] = sbox7(state[i]);
        }
        mul_external(&state);
    }

    for (var r = 0u; r < params.rounds_p; r = r + 1u) {
        state[0] = mod_add(state[0], poseidon_constants[rc_ofs + r]);
        state[0] = sbox7(state[0]);

        var sum: u32 = 0u;
        for (var i = 0u; i < WIDTH; i = i + 1u) {
            sum = mod_add(sum, state[i]);
        }
        for (var i = 0u; i < WIDTH; i = i + 1u) {
            let dd = mod_mul(poseidon_constants[diag_ofs + i], state[i]);
            state[i] = mod_sub(mod_add(sum, dd), state[i]);
        }
    }

    for (var r = 0u; r < params.rounds_f_half; r = r + 1u) {
        let idx = params.rounds_f_half + r;
        let rc_base = idx * WIDTH;
        for (var i = 0u; i < WIDTH; i = i + 1u) {
            state[i] = mod_add(state[i], poseidon_constants[rc_base + i]);
        }
        for (var i = 0u; i < WIDTH; i = i + 1u) {
            state[i] = sbox7(state[i]);
        }
        mul_external(&state);
    }

    // Truncate to CHUNK=8: write state[0..8] into the output digest slot.
    let out_base = out_idx * DIGEST_LEN;
    for (var i = 0u; i < DIGEST_LEN; i = i + 1u) {
        output_digests[out_base + i] = state[i];
    }
}
