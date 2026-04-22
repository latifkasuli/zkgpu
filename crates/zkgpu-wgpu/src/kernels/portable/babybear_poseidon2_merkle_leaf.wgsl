// BabyBear Poseidon2 Merkle leaf sponge — width 24, Plonky3 variant.
// P = 2^31 - 2^27 + 1 = 2013265921
//
// Phase 7 Step 3.a. One thread per leaf (row of the committed matrix).
// Each thread runs a `PaddingFreeSponge<Perm24, 24, 16, 8>` absorption
// over its row and writes the 8-element digest into the output buffer.
//
// Matches Plonky3's sponge semantics exactly (overwrite-mode, not XOR/add):
//   state = [0; 24]
//   for each full 16-element chunk of input:
//     state[0..16] = chunk
//     state = Perm24(state)
//   if there's a partial chunk of length r > 0:
//     state[0..r] = partial
//     state = Perm24(state)
//   digest = state[0..8]
//
// Binding layout note: WebGPU baseline caps
// `max_storage_buffers_per_shader_stage = 4`. This kernel needs to
// expose input_matrix + digests as two separate buffers (because the
// Merkle commit plan wants GPU-resident inputs/outputs), so we pack
// the three Poseidon2 constant tables (`external_constants`,
// `internal_constants`, `internal_diagonal`) into a single storage
// buffer:
//
//   `poseidon_constants` = [ external | internal_rc | internal_diag ]
//
// Offsets are computed kernel-side from `params.rounds_f_half` /
// `params.rounds_p` / `WIDTH`, matching the flat layout the host
// uploads in `WgpuPoseidon2MerkleLeafPlan::new`.

const P: u32 = 2013265921u;
const WIDTH: u32 = 24u;
const DIGEST_LEN: u32 = 8u;
const RATE: u32 = 16u;
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

// --- Poseidon2 layer helpers (Plonky3 M_4 = circ(2,3,1,1)) ---

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

fn mul_external(state: ptr<function, array<u32, 24>>) {
    let b0 = m4_block_plonky3((*state)[0],  (*state)[1],  (*state)[2],  (*state)[3]);
    let b1 = m4_block_plonky3((*state)[4],  (*state)[5],  (*state)[6],  (*state)[7]);
    let b2 = m4_block_plonky3((*state)[8],  (*state)[9],  (*state)[10], (*state)[11]);
    let b3 = m4_block_plonky3((*state)[12], (*state)[13], (*state)[14], (*state)[15]);
    let b4 = m4_block_plonky3((*state)[16], (*state)[17], (*state)[18], (*state)[19]);
    let b5 = m4_block_plonky3((*state)[20], (*state)[21], (*state)[22], (*state)[23]);

    let c0 = mod_add(
        mod_add(mod_add(b0.x, b1.x), mod_add(b2.x, b3.x)),
        mod_add(b4.x, b5.x),
    );
    let c1 = mod_add(
        mod_add(mod_add(b0.y, b1.y), mod_add(b2.y, b3.y)),
        mod_add(b4.y, b5.y),
    );
    let c2 = mod_add(
        mod_add(mod_add(b0.z, b1.z), mod_add(b2.z, b3.z)),
        mod_add(b4.z, b5.z),
    );
    let c3 = mod_add(
        mod_add(mod_add(b0.w, b1.w), mod_add(b2.w, b3.w)),
        mod_add(b4.w, b5.w),
    );

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
    (*state)[16] = mod_add(b4.x, c0);
    (*state)[17] = mod_add(b4.y, c1);
    (*state)[18] = mod_add(b4.z, c2);
    (*state)[19] = mod_add(b4.w, c3);
    (*state)[20] = mod_add(b5.x, c0);
    (*state)[21] = mod_add(b5.y, c1);
    (*state)[22] = mod_add(b5.z, c2);
    (*state)[23] = mod_add(b5.w, c3);
}

// --- Bindings (4 storage + 1 uniform, fits WebGPU baseline) ---

struct MerkleLeafParams {
    num_leaves: u32,
    row_width: u32,
    rounds_f_half: u32,
    rounds_p: u32,
    row_stride: u32,
    // Offsets into `poseidon_constants`:
    // external_offset is always 0 (convention).
    internal_rc_offset: u32,   // == 2 * rounds_f_half * WIDTH
    internal_diag_offset: u32, // == internal_rc_offset + rounds_p
    _pad: u32,
}

@group(0) @binding(0) var<storage, read>        input_matrix: array<u32>;
@group(0) @binding(1) var<storage, read_write>  digests: array<u32>;
@group(0) @binding(2) var<storage, read>        poseidon_constants: array<u32>;
@group(0) @binding(3) var<uniform>              params: MerkleLeafParams;

@compute @workgroup_size(64)
fn merkle_leaf_hash(@builtin(global_invocation_id) gid: vec3<u32>) {
    let leaf_idx = gid.x + gid.y * params.row_stride;
    if (leaf_idx >= params.num_leaves) { return; }

    // --- Sponge absorption (PaddingFreeSponge semantics) ---
    var state: array<u32, 24>;
    for (var i = 0u; i < WIDTH; i = i + 1u) {
        state[i] = 0u;
    }

    let row_base = leaf_idx * params.row_width;
    let w = params.row_width;
    let full_chunks = w / RATE;
    let remainder = w - full_chunks * RATE;  // == w % RATE
    let rc_ofs = params.internal_rc_offset;
    let diag_ofs = params.internal_diag_offset;

    // Full chunks: overwrite state[0..RATE], permute.
    for (var c = 0u; c < full_chunks; c = c + 1u) {
        let chunk_base = row_base + c * RATE;
        for (var i = 0u; i < RATE; i = i + 1u) {
            state[i] = input_matrix[chunk_base + i];
        }

        // --- Inline perm24 (Plonky3 Poseidon2 W24) ---
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
    }

    // Partial final chunk: overwrite state[0..remainder], permute.
    // state[remainder..RATE] retain values from the previous permute.
    if (remainder > 0u) {
        let chunk_base = row_base + full_chunks * RATE;
        for (var i = 0u; i < remainder; i = i + 1u) {
            state[i] = input_matrix[chunk_base + i];
        }

        // --- Inline perm24 (duplicated — WGSL has no generic fn body) ---
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
    }

    // --- Write 8-element digest ---
    let digest_base = leaf_idx * DIGEST_LEN;
    for (var i = 0u; i < DIGEST_LEN; i = i + 1u) {
        digests[digest_base + i] = state[i];
    }
}
