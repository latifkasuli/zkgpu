// BabyBear Poseidon2 permutation — UNIFORM-bound constants variant.
//
// Item #6 of `docs/research/zkgpu-wgpu-speed-opportunities.md` (Gate 2,
// safer portable path). Pilot scope: standalone `WgpuBabyBearPoseidon2Plan`
// only. The Plonky3 W16/W24 + merkle leaf/compress kernels still use
// the storage-bound path; they migrate in a follow-up commit if this
// pilot shows a measurable win on Apple Silicon (the platform with the
// strongest constant-cache-vs-storage delta).
//
// Algorithmic body is byte-identical to `babybear_poseidon2.wgsl`. The
// only differences are:
//
//   1. Three `var<storage, read>` arrays collapse into one `var<uniform>`
//      struct holding `array<vec4<u32>, N>` — the canonical WGSL way to
//      pack `u32` constants without 4× padding waste in std140-like
//      uniform layout.
//   2. Index lookups change from `external_constants[i]` to
//      `constants.external[i >> 2u][i & 3u]` (vec4 chunk + lane select).
//
// Why this might be faster: every thread in a warp reads the same
// constant index at the same time (all threads at round `r`, position
// `i`). That's the textbook broadcast access pattern uniforms are
// optimized for: Metal `constant` address space hits a dedicated
// constant cache; Vulkan/SPIR-V `Uniform` storage class maps to
// constant-buffer hardware (cmem on NVIDIA, scalar broadcast on
// AMD/Intel). Storage buffers go through the general L1/L2 path —
// correct, but pays full memory-system latency on cache miss.
//
// Why it might NOT win: small constant footprint (~700 bytes total
// for W16) already fits in L1 on every modern GPU after the first
// permutation, so the cache-tier difference may be unmeasurable in
// practice. We measure before propagating.

const P: u32 = 2013265921u;
const WIDTH: u32 = 16u;
const M4_WIDTH: u32 = 4u;
const NUM_BLOCKS: u32 = 4u;
const WORKGROUP_SIZE: u32 = 64u;

// --- BabyBear modular arithmetic (identical to babybear_poseidon2.wgsl) -

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

// --- Poseidon2 layer helpers (identical) -------------------------------

fn sbox7(x: u32) -> u32 {
    let x2 = mod_mul(x, x);
    let x4 = mod_mul(x2, x2);
    let x6 = mod_mul(x4, x2);
    return mod_mul(x6, x);
}

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

fn mul_external(state: ptr<function, array<u32, 16>>) {
    let b0 = m4_block((*state)[0],  (*state)[1],  (*state)[2],  (*state)[3]);
    let b1 = m4_block((*state)[4],  (*state)[5],  (*state)[6],  (*state)[7]);
    let b2 = m4_block((*state)[8],  (*state)[9],  (*state)[10], (*state)[11]);
    let b3 = m4_block((*state)[12], (*state)[13], (*state)[14], (*state)[15]);

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

// --- Bindings + kernel --------------------------------------------------

struct Poseidon2Params {
    num_permutations: u32,
    rounds_f_half: u32,
    rounds_p: u32,
    row_stride: u32,
}

// Packed constants for the W16 BabyBear configuration.
//
// Sizes are fixed at the upper bounds we expect from any plonky3-shape
// BabyBear W16 Poseidon2 (8 external rounds, ≤ 32 partial rounds).
// Smaller configs leave the trailing slots zero — never read.
//
//   external:  8 rounds × 16 width = 128 u32 → 32 × vec4<u32>
//   internal:  ≤ 32 partial rounds          →  8 × vec4<u32>
//   diagonal:  16 slots                     →  4 × vec4<u32>
//
// Total: 44 × 16 B = 704 bytes. Well under the 16 KiB conservative
// uniform-buffer ceiling on mobile, let alone the 64 KiB desktop one.
// Field names avoid WGSL reserved keywords (`external`, `internal` are
// reserved in WGSL — using them here triggers a silent shader-compile
// failure that surfaces only as zero-valued GPU output unless the
// caller pushes a validation error scope).
struct Poseidon2W16ConstantsUniform {
    rc_ext: array<vec4<u32>, 32>,
    rc_int: array<vec4<u32>, 8>,
    diag: array<vec4<u32>, 4>,
}

@group(0) @binding(0) var<storage, read_write> states: array<u32>;
@group(0) @binding(1) var<uniform>             constants: Poseidon2W16ConstantsUniform;
@group(0) @binding(2) var<uniform>             params: Poseidon2Params;

// Lane-select helpers. WGSL allows `vec[i]` with a runtime `u32` index;
// Naga lowers this to a backend-appropriate four-way select. All
// threads in a warp share `i` (every thread is at the same round and
// position), so there's no divergence cost on top of the constant-
// cache load.
fn ext_const(r: u32, slot: u32) -> u32 {
    let flat = r * WIDTH + slot;
    return constants.rc_ext[flat >> 2u][flat & 3u];
}

fn int_const(r: u32) -> u32 {
    return constants.rc_int[r >> 2u][r & 3u];
}

fn int_diag(slot: u32) -> u32 {
    return constants.diag[slot >> 2u][slot & 3u];
}

@compute @workgroup_size(64)
fn poseidon2_permute(@builtin(global_invocation_id) gid: vec3<u32>) {
    let perm_idx = gid.x + gid.y * params.row_stride;
    if (perm_idx >= params.num_permutations) { return; }

    var state: array<u32, 16>;
    let base = perm_idx * WIDTH;
    for (var i = 0u; i < WIDTH; i = i + 1u) {
        state[i] = states[base + i];
    }

    mul_external(&state);

    for (var r = 0u; r < params.rounds_f_half; r = r + 1u) {
        for (var i = 0u; i < WIDTH; i = i + 1u) {
            state[i] = mod_add(state[i], ext_const(r, i));
        }
        for (var i = 0u; i < WIDTH; i = i + 1u) {
            state[i] = sbox7(state[i]);
        }
        mul_external(&state);
    }

    for (var r = 0u; r < params.rounds_p; r = r + 1u) {
        state[0] = mod_add(state[0], int_const(r));
        state[0] = sbox7(state[0]);

        var sum: u32 = 0u;
        for (var i = 0u; i < WIDTH; i = i + 1u) {
            sum = mod_add(sum, state[i]);
        }
        for (var i = 0u; i < WIDTH; i = i + 1u) {
            let dd = mod_mul(int_diag(i), state[i]);
            state[i] = mod_sub(mod_add(sum, dd), state[i]);
        }
    }

    for (var r = 0u; r < params.rounds_f_half; r = r + 1u) {
        let idx = params.rounds_f_half + r;
        for (var i = 0u; i < WIDTH; i = i + 1u) {
            state[i] = mod_add(state[i], ext_const(idx, i));
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
