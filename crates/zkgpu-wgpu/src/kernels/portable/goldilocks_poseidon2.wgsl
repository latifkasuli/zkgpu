// Goldilocks Poseidon2 permutation kernel, portable u32x2 WGSL.
//
// Phase F.2 — Goldilocks twin of `babybear_poseidon2.wgsl`. Prepended
// at build time with `goldilocks_arith_helpers.wgsl`, which provides
// `gl_add`, `gl_sub`, `gl_mul` over the u32x2 limb representation
// (one field element = one `vec2<u32>` = little-endian lo/hi).
//
// Structure mirrors the BabyBear kernel exactly — only the element
// type, arithmetic dispatch, and per-slot buffer addressing change.
// See `babybear_poseidon2.wgsl` for the algorithm walkthrough; this
// file sticks to the deltas.
//
// Storage layout: each Goldilocks field element is two u32 limbs, so
// the state buffer is indexed as `states[perm_idx * WIDTH * 2 + slot * 2]`
// (lo) and `+1` (hi). The kernel wraps reads/writes in
// `state_load(idx) -> vec2<u32>` / `state_store(idx, vec2<u32>)`
// helpers to keep the permutation body readable.
//
// 2D-folded dispatch, one thread per permutation instance — same
// model as the BabyBear kernel. The only operational difference is
// that each `mod_mul` in the S-box / `mul_internal` costs more work
// (gl_mul emulates 64×64→128 + Goldilocks reduction); throughput
// expectations follow accordingly.

const WIDTH: u32 = 16u;
const M4_WIDTH: u32 = 4u;
const NUM_BLOCKS: u32 = 4u;  // WIDTH / M4_WIDTH
const WORKGROUP_SIZE: u32 = 64u;

// --- Poseidon2 layer helpers (Goldilocks) -------------------------------

// S-box `x → x^7`. Four multiplies: x^2, x^4, x^6, x^7.
fn gl_sbox7(x: vec2<u32>) -> vec2<u32> {
    let x2 = gl_mul(x, x);
    let x4 = gl_mul(x2, x2);
    let x6 = gl_mul(x4, x2);
    return gl_mul(x6, x);
}

// `M_4 = circ(2, 1, 1, 1)` on a 4-element block: y[i] = sum + x[i].
// Returns the four results as an `array<vec2<u32>, 4>` via positional
// out-params packed into a struct, because WGSL arrays of vec2<u32>
// as return types are awkward.
struct M4Block {
    y0: vec2<u32>,
    y1: vec2<u32>,
    y2: vec2<u32>,
    y3: vec2<u32>,
}

fn gl_m4_block(x0: vec2<u32>, x1: vec2<u32>, x2: vec2<u32>, x3: vec2<u32>) -> M4Block {
    let s01 = gl_add(x0, x1);
    let s23 = gl_add(x2, x3);
    let sum = gl_add(s01, s23);
    return M4Block(
        gl_add(sum, x0),
        gl_add(sum, x1),
        gl_add(sum, x2),
        gl_add(sum, x3),
    );
}

// External matrix `M_E ⊗ M_4` — same two-pass structure as BabyBear's
// `mul_external`. Mutates the 16-slot state via a pointer arg.
fn gl_mul_external(state: ptr<function, array<vec2<u32>, 16>>) {
    // Step 1: M_4 per block.
    let b0 = gl_m4_block((*state)[0],  (*state)[1],  (*state)[2],  (*state)[3]);
    let b1 = gl_m4_block((*state)[4],  (*state)[5],  (*state)[6],  (*state)[7]);
    let b2 = gl_m4_block((*state)[8],  (*state)[9],  (*state)[10], (*state)[11]);
    let b3 = gl_m4_block((*state)[12], (*state)[13], (*state)[14], (*state)[15]);

    // Step 2: cross-block column sums.
    let c0 = gl_add(gl_add(b0.y0, b1.y0), gl_add(b2.y0, b3.y0));
    let c1 = gl_add(gl_add(b0.y1, b1.y1), gl_add(b2.y1, b3.y1));
    let c2 = gl_add(gl_add(b0.y2, b1.y2), gl_add(b2.y2, b3.y2));
    let c3 = gl_add(gl_add(b0.y3, b1.y3), gl_add(b2.y3, b3.y3));

    // Step 3: blocks[b][j] + sum_all[j] written back.
    (*state)[0]  = gl_add(b0.y0, c0);
    (*state)[1]  = gl_add(b0.y1, c1);
    (*state)[2]  = gl_add(b0.y2, c2);
    (*state)[3]  = gl_add(b0.y3, c3);
    (*state)[4]  = gl_add(b1.y0, c0);
    (*state)[5]  = gl_add(b1.y1, c1);
    (*state)[6]  = gl_add(b1.y2, c2);
    (*state)[7]  = gl_add(b1.y3, c3);
    (*state)[8]  = gl_add(b2.y0, c0);
    (*state)[9]  = gl_add(b2.y1, c1);
    (*state)[10] = gl_add(b2.y2, c2);
    (*state)[11] = gl_add(b2.y3, c3);
    (*state)[12] = gl_add(b3.y0, c0);
    (*state)[13] = gl_add(b3.y1, c1);
    (*state)[14] = gl_add(b3.y2, c2);
    (*state)[15] = gl_add(b3.y3, c3);
}

// --- Bindings + kernel --------------------------------------------------

struct Poseidon2Params {
    num_permutations: u32,
    rounds_f_half: u32,
    rounds_p: u32,
    row_stride: u32,
}

// Storage is u32 flat; each Goldilocks element occupies two consecutive
// u32s. vec2<u32> storage alignment is 8 bytes (spec §13.7.3), matching
// bytemuck's layout of `Goldilocks::Repr = u64` uploaded via
// `cast_slice`. Binding the buffer as `array<vec2<u32>>` lets the
// kernel index by field element instead of by u32 limb.
@group(0) @binding(0) var<storage, read_write> states: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read>       external_constants: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read>       internal_constants: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read>       internal_diagonal: array<vec2<u32>>;
@group(0) @binding(4) var<uniform>             params: Poseidon2Params;

@compute @workgroup_size(64)
fn gl_poseidon2_permute(@builtin(global_invocation_id) gid: vec3<u32>) {
    let perm_idx = gid.x + gid.y * params.row_stride;
    if (perm_idx >= params.num_permutations) { return; }

    // --- Load state into thread-local array of vec2<u32> ---
    var state: array<vec2<u32>, 16>;
    let base = perm_idx * WIDTH;
    for (var i = 0u; i < WIDTH; i = i + 1u) {
        state[i] = states[base + i];
    }

    // --- Initial external mix ---
    gl_mul_external(&state);

    // --- First half of external rounds ---
    for (var r = 0u; r < params.rounds_f_half; r = r + 1u) {
        let rc_base = r * WIDTH;
        for (var i = 0u; i < WIDTH; i = i + 1u) {
            state[i] = gl_add(state[i], external_constants[rc_base + i]);
        }
        for (var i = 0u; i < WIDTH; i = i + 1u) {
            state[i] = gl_sbox7(state[i]);
        }
        gl_mul_external(&state);
    }

    // --- Internal rounds ---
    for (var r = 0u; r < params.rounds_p; r = r + 1u) {
        state[0] = gl_add(state[0], internal_constants[r]);
        state[0] = gl_sbox7(state[0]);

        // M_int = 1 + D applied as: state[i] = sum + d_i*state[i] - state[i].
        var sum: vec2<u32> = vec2<u32>(0u, 0u);
        for (var i = 0u; i < WIDTH; i = i + 1u) {
            sum = gl_add(sum, state[i]);
        }
        for (var i = 0u; i < WIDTH; i = i + 1u) {
            let dd = gl_mul(internal_diagonal[i], state[i]);
            state[i] = gl_sub(gl_add(sum, dd), state[i]);
        }
    }

    // --- Second half of external rounds ---
    for (var r = 0u; r < params.rounds_f_half; r = r + 1u) {
        let idx = params.rounds_f_half + r;
        let rc_base = idx * WIDTH;
        for (var i = 0u; i < WIDTH; i = i + 1u) {
            state[i] = gl_add(state[i], external_constants[rc_base + i]);
        }
        for (var i = 0u; i < WIDTH; i = i + 1u) {
            state[i] = gl_sbox7(state[i]);
        }
        gl_mul_external(&state);
    }

    // --- Write back ---
    for (var i = 0u; i < WIDTH; i = i + 1u) {
        states[base + i] = state[i];
    }
}
