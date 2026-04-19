// BabyBear Poseidon2 permutation kernel, portable WGSL.
// P = 2^31 - 2^27 + 1 = 2013265921
//
// Phase F.1 — first GPU hash kernel in zkgpu. One thread per
// permutation instance. The state (16 × u32 field elements) lives in
// thread-local `var` space, so no workgroup barriers are required.
//
// Layout:
//   - `states[perm_idx * WIDTH + slot]` — input/output (read-write).
//     The thread reads its 16 slots, runs the full permutation, then
//     writes back.
//   - `external_constants[round * WIDTH + slot]` — one row per external
//     round, `2 * rounds_f_half` rows total (first half applied before
//     internal rounds, second half applied after).
//   - `internal_constants[round]` — scalar constant added to `state[0]`
//     once per internal round.
//   - `internal_diagonal[slot]` — field element `d_i` used in the
//     internal matrix `M_int = 1 + D`.
//
// Algorithm mirrors `zkgpu_poseidon2::Poseidon2::permute` exactly. The
// external layer is `M_E ⊗ M_4` with `M_4 = circ(2,1,1,1)` and
// `M_E(i,j) = 2·M_4 if i == j else M_4`; the internal layer is
// `M_int = 1 + D` with `D = diag(d_0..d_{W-1})`. S-box is `x^7`.
//
// 2D-folded dispatch mirrors the NTT kernels: the flat permutation
// index is reconstructed as `perm_idx = gid.x + gid.y * row_stride`
// where `row_stride = groups_per_row * WORKGROUP_SIZE`. Lets batch
// sizes that exceed `max_compute_workgroups_per_dimension * 64`
// (≈ 4.2M permutations on WebGPU baseline) dispatch without hitting
// the per-dimension limit.

const P: u32 = 2013265921u;
const WIDTH: u32 = 16u;
const M4_WIDTH: u32 = 4u;
const NUM_BLOCKS: u32 = 4u;  // WIDTH / M4_WIDTH
const WORKGROUP_SIZE: u32 = 64u;

// --- BabyBear modular arithmetic ----------------------------------------
//
// Inlined from `babybear_stockham_r2.wgsl` — the project convention is
// per-kernel inlining rather than a shared helper include. Any future
// refactor that extracts these into `babybear_arith_helpers.wgsl`
// should update all BabyBear kernels together.

fn mod_add(a: u32, b: u32) -> u32 {
    let sum = a + b;
    return select(sum, sum - P, sum >= P);
}

fn mod_sub(a: u32, b: u32) -> u32 {
    return select(a - b, a + P - b, a < b);
}

fn mod_mul(a: u32, b: u32) -> u32 {
    // (a * b) mod P for a, b < P < 2^31. WGSL has no native u64, so
    // emulate the 64-bit product via 16-bit-limb schoolbook, then
    // reduce using P = 2^31 - 2^27 + 1, i.e. 2^32 ≡ 2^28 - 2 (mod P).

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

    // Iterative reduction: hi shrinks ~16× per round.
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

// --- Poseidon2 layer helpers -------------------------------------------

// S-box `x → x^7`. Four multiplies: x^2, x^4, x^6, x^7.
fn sbox7(x: u32) -> u32 {
    let x2 = mod_mul(x, x);
    let x4 = mod_mul(x2, x2);
    let x6 = mod_mul(x4, x2);
    return mod_mul(x6, x);
}

// `M_4 = circ(2, 1, 1, 1)` on a 4-element block: y[i] = sum + x[i].
// Written as three helper calls to keep `mul_external` readable.
fn m4_block(x0: u32, x1: u32, x2: u32, x3: u32) -> vec4<u32> {
    let s01 = mod_add(x0, x1);
    let s23 = mod_add(x2, x3);
    let sum = mod_add(s01, s23);
    return vec4<u32>(
        mod_add(sum, x0),
        mod_add(sum, x1),
        mod_add(sum, x2),
        mod_add(sum, x3),
    );
}

// External matrix `M_E ⊗ M_4`:
//   1. Apply `M_4` to each 4-element block independently.
//   2. For each column j, each block's j-slot becomes
//      `block[b][j] + sum_all_blocks[j]`. Implements the doubled-
//      diagonal `M_E(i,i) = 2·M_4`, `M_E(i,j) = M_4`.
fn mul_external(state: ptr<function, array<u32, 16>>) {
    // Step 1: M_4 per block, stored contiguously back into state.
    let b0 = m4_block((*state)[0],  (*state)[1],  (*state)[2],  (*state)[3]);
    let b1 = m4_block((*state)[4],  (*state)[5],  (*state)[6],  (*state)[7]);
    let b2 = m4_block((*state)[8],  (*state)[9],  (*state)[10], (*state)[11]);
    let b3 = m4_block((*state)[12], (*state)[13], (*state)[14], (*state)[15]);

    // Step 2: cross-block column sums.
    let c0 = mod_add(mod_add(b0.x, b1.x), mod_add(b2.x, b3.x));
    let c1 = mod_add(mod_add(b0.y, b1.y), mod_add(b2.y, b3.y));
    let c2 = mod_add(mod_add(b0.z, b1.z), mod_add(b2.z, b3.z));
    let c3 = mod_add(mod_add(b0.w, b1.w), mod_add(b2.w, b3.w));

    // Step 3: write back blocks[b][j] + sum_all[j].
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

// --- Bindings + kernel --------------------------------------------------

struct Poseidon2Params {
    num_permutations: u32,
    rounds_f_half: u32,
    rounds_p: u32,
    // 2D-fold dispatch: row_stride = groups_per_row * WORKGROUP_SIZE.
    row_stride: u32,
}

@group(0) @binding(0) var<storage, read_write> states: array<u32>;
@group(0) @binding(1) var<storage, read>       external_constants: array<u32>;
@group(0) @binding(2) var<storage, read>       internal_constants: array<u32>;
@group(0) @binding(3) var<storage, read>       internal_diagonal: array<u32>;
@group(0) @binding(4) var<uniform>             params: Poseidon2Params;

@compute @workgroup_size(64)
fn poseidon2_permute(@builtin(global_invocation_id) gid: vec3<u32>) {
    let perm_idx = gid.x + gid.y * params.row_stride;
    if (perm_idx >= params.num_permutations) { return; }

    // --- Load state into thread-local array ---
    var state: array<u32, 16>;
    let base = perm_idx * WIDTH;
    for (var i = 0u; i < WIDTH; i = i + 1u) {
        state[i] = states[base + i];
    }

    // --- Initial external mix (Poseidon2 spec) ---
    mul_external(&state);

    // --- First half of external rounds ---
    for (var r = 0u; r < params.rounds_f_half; r = r + 1u) {
        // Add external round constants.
        let rc_base = r * WIDTH;
        for (var i = 0u; i < WIDTH; i = i + 1u) {
            state[i] = mod_add(state[i], external_constants[rc_base + i]);
        }
        // Full S-box: every position through x^7.
        for (var i = 0u; i < WIDTH; i = i + 1u) {
            state[i] = sbox7(state[i]);
        }
        mul_external(&state);
    }

    // --- Internal rounds ---
    for (var r = 0u; r < params.rounds_p; r = r + 1u) {
        // Single-position constant add + S-box.
        state[0] = mod_add(state[0], internal_constants[r]);
        state[0] = sbox7(state[0]);

        // M_int = 1 + D applied as: state[i] = sum + d_i*state[i] - state[i].
        var sum: u32 = 0u;
        for (var i = 0u; i < WIDTH; i = i + 1u) {
            sum = mod_add(sum, state[i]);
        }
        for (var i = 0u; i < WIDTH; i = i + 1u) {
            let dd = mod_mul(internal_diagonal[i], state[i]);
            state[i] = mod_sub(mod_add(sum, dd), state[i]);
        }
    }

    // --- Second half of external rounds ---
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

    // --- Write back ---
    for (var i = 0u; i < WIDTH; i = i + 1u) {
        states[base + i] = state[i];
    }
}
