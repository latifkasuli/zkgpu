// Portable Goldilocks field arithmetic kernel.
//
// Phase B.1 deliverable. Implements the u32x2 limb representation of the
// Goldilocks field (p = 2^64 - 2^32 + 1) in pure WGSL — no `u64` shader
// type required. One element = one `vec2<u32>` = little-endian (lo, hi)
// where value = lo + (hi << 32).
//
// The same arithmetic helpers (gl_add / gl_sub / gl_mul + 128-bit
// reduction) will be inlined into the Stockham and scale kernels in
// Phase B.2. Exposed here via a test kernel that does elementwise
// a+b, a-b, a*b so the Rust side can differentially validate them
// against `zkgpu-goldilocks`'s CPU implementation.
//
// Why inline instead of `#include`? WGSL has no preprocessor. Any
// shared helpers must be duplicated into each shader source. We keep
// one canonical copy here and Phase B.2 will `include_str!` the same
// helper block into the NTT kernels (via a host-side string concat).

// --- Goldilocks field constants -------------------------------------

const GL_P_LO: u32 = 0x00000001u;
const GL_P_HI: u32 = 0xFFFFFFFFu;
// EPSILON = 2^32 - 1 = 2^64 - p. Appears in every wrap correction.
const GL_EPSILON: u32 = 0xFFFFFFFFu;

// --- 64-bit integer helpers on vec2<u32> = (lo, hi) -----------------

// Unsigned 64-bit add with carry-out. Returns (lo, hi, carry) where
// carry ∈ {0, 1}. `vec3<u32>` chosen over a struct for WGSL brevity.
fn u64_add_carry(a: vec2<u32>, b: vec2<u32>) -> vec3<u32> {
    let lo = a.x + b.x;
    let c0 = u32(lo < a.x);       // wrap on lo add
    let hi_p = a.y + b.y;          // partial hi (before carry-in)
    let c1 = u32(hi_p < a.y);      // wrap on hi partial add
    let hi = hi_p + c0;
    let c2 = u32(hi < hi_p);       // wrap on carry-in add
    // c1 + c2 is at most 1 for any valid (a, b) pair, because the true
    // sum a + b is bounded by 2*(2^64 - 1) < 2^65, so only one carry
    // bit ever escapes. The addition is written as c1 + c2 because
    // the two sources are architecturally distinct — tracking them
    // separately keeps the invariant debuggable.
    return vec3<u32>(lo, hi, c1 + c2);
}

// Unsigned 64-bit sub with borrow-out. Returns (lo, hi, borrow) where
// borrow ∈ {0, 1}.
fn u64_sub_borrow(a: vec2<u32>, b: vec2<u32>) -> vec3<u32> {
    let lo = a.x - b.x;
    let b0 = u32(a.x < b.x);       // borrow on lo sub
    let hi_p = a.y - b.y;          // partial hi (before borrow-in)
    let b1 = u32(a.y < b.y);       // borrow on hi partial sub
    let hi = hi_p - b0;
    let b2 = u32(hi_p < b0);       // borrow on borrow-in sub
    return vec3<u32>(lo, hi, b1 + b2);
}

// Canonical `a >= b` comparison for two 64-bit values.
fn u64_ge(a: vec2<u32>, b: vec2<u32>) -> bool {
    return a.y > b.y || (a.y == b.y && a.x >= b.x);
}

// 32 × 32 → 64-bit unsigned multiply. Essential primitive — WGSL's
// native `u32 * u32` drops the high 32 bits.
//
// Split each input into 16-bit halves, take four partial products (each
// fits in a u32), assemble. The output is always a valid u64 since
// (2^32 - 1)^2 < 2^64.
fn mul_u32_full(a: u32, b: u32) -> vec2<u32> {
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;

    let ll = a_lo * b_lo;           // bits [0, 32)
    let lh = a_lo * b_hi;           // bits [16, 48)
    let hl = a_hi * b_lo;           // bits [16, 48)
    let hh = a_hi * b_hi;           // bits [32, 64)

    // Middle cross-product: lh + hl. Can overflow u32 (max 2^33 - 2^18 + 2).
    let mid = lh + hl;
    let mid_carry = u32(mid < lh);  // 0 or 1

    // Low 32 bits: ll (bits 0-31) plus the low 16 bits of mid, shifted.
    let lo = ll + (mid << 16u);
    let lo_carry = u32(lo < ll);    // 0 or 1 — wrap-detection on u32 add

    // High 32 bits: hh (bits 32-63 of product) + high 16 of mid (bits 32-47
    // via mid >> 16 landing in bits 0-15 of hi) + the mid overflow bit
    // (which represents 2^48 in the full product, i.e. bit 16 of hi) +
    // the lo carry.
    let hi = hh + (mid >> 16u) + (mid_carry << 16u) + lo_carry;

    return vec2<u32>(lo, hi);
}

// --- Goldilocks field operations ------------------------------------

// Reduce a 128-bit value `x` (given as four 32-bit limbs, lsb first)
// modulo p using the Goldilocks special-form identity:
//
//   x ≡ x_lo - x_hi_hi + x_hi_lo · EPSILON   (mod p)
//
// where x_lo = bits [0, 64), x_hi_lo = bits [64, 96), x_hi_hi = bits
// [96, 128). See `zkgpu-goldilocks` crate docs for the derivation.
fn gl_reduce_128(prod: vec4<u32>) -> vec2<u32> {
    let x_lo = vec2<u32>(prod.x, prod.y);
    let x_hi_lo = prod.z;
    let x_hi_hi = prod.w;
    let p = vec2<u32>(GL_P_LO, GL_P_HI);

    // Step 1: t = x_lo - (x_hi_hi, 0). If this borrows, the true value
    // of the subtraction was negative; correcting by subtracting
    // EPSILON from the wrapped result is equivalent to adding p.
    let t_sub = u64_sub_borrow(x_lo, vec2<u32>(x_hi_hi, 0u));
    let t_raw = vec2<u32>(t_sub.x, t_sub.y);
    let t_corr = u64_sub_borrow(t_raw, vec2<u32>(GL_EPSILON, 0u));
    let t = select(t_raw, vec2<u32>(t_corr.x, t_corr.y), t_sub.z == 1u);

    // Step 2: s = t + x_hi_lo · EPSILON. On overflow, add EPSILON
    // (which is -p + 2^64, i.e. another wrap correction in the same
    // direction as the previous borrow one).
    let prod_v = mul_u32_full(x_hi_lo, GL_EPSILON);
    let s_add = u64_add_carry(t, prod_v);
    let s_raw = vec2<u32>(s_add.x, s_add.y);
    let s_corr = u64_add_carry(s_raw, vec2<u32>(GL_EPSILON, 0u));
    let s = select(s_raw, vec2<u32>(s_corr.x, s_corr.y), s_add.z == 1u);

    // Step 3: final canonical reduction — s lies in [0, 2p) so one
    // conditional subtract suffices.
    let s_final = u64_sub_borrow(s, p);
    return select(s, vec2<u32>(s_final.x, s_final.y), u64_ge(s, p));
}

// Modular add. For a, b ∈ [0, p), a + b < 2p < 2^65, so one carry bit
// is enough.
fn gl_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let p = vec2<u32>(GL_P_LO, GL_P_HI);
    let add_res = u64_add_carry(a, b);
    let sum = vec2<u32>(add_res.x, add_res.y);
    let carry = add_res.z;
    // If carry, the true sum was ≥ 2^64; add EPSILON (≡ subtracting p).
    let corr = u64_add_carry(sum, vec2<u32>(GL_EPSILON, 0u));
    let s = select(sum, vec2<u32>(corr.x, corr.y), carry == 1u);
    // Final canonicalise.
    let sub = u64_sub_borrow(s, p);
    return select(s, vec2<u32>(sub.x, sub.y), u64_ge(s, p));
}

// Modular sub. For a, b ∈ [0, p), a - b ∈ (-p, p).
fn gl_sub(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let sub_res = u64_sub_borrow(a, b);
    let diff = vec2<u32>(sub_res.x, sub_res.y);
    let borrow = sub_res.z;
    // If borrow, true value is diff - EPSILON in wrapped u64 arithmetic
    // (≡ adding p).
    let corr = u64_sub_borrow(diff, vec2<u32>(GL_EPSILON, 0u));
    return select(diff, vec2<u32>(corr.x, corr.y), borrow == 1u);
}

// Modular multiply. a * b < p^2 < 2^128.
fn gl_mul(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    // 64 × 64 → 128 via four 32 × 32 → 64 partial products.
    //   a * b = ll + (lh + hl) · 2^32 + hh · 2^64
    // Each partial is a vec2<u32> in (lo, hi) form.
    let ll = mul_u32_full(a.x, b.x);   // bits [0, 64)
    let lh = mul_u32_full(a.x, b.y);   // bits [32, 96)
    let hl = mul_u32_full(a.y, b.x);   // bits [32, 96)
    let hh = mul_u32_full(a.y, b.y);   // bits [64, 128)

    // p0 = ll.x (bits 0-31)
    let p0 = ll.x;

    // p1 = ll.y + lh.x + hl.x (all contribute to bits 32-63).
    // Track a carry for p2.
    let p1_s1 = ll.y + lh.x;
    let p1_c1 = u32(p1_s1 < ll.y);
    let p1 = p1_s1 + hl.x;
    let p1_c2 = u32(p1 < p1_s1);
    let p1_carry = p1_c1 + p1_c2;  // 0, 1, or 2 — all three are u32s

    // p2 = lh.y + hl.y + hh.x + p1_carry (bits 64-95).
    // Track a carry for p3.
    let p2_s1 = lh.y + hl.y;
    let p2_c1 = u32(p2_s1 < lh.y);
    let p2_s2 = p2_s1 + hh.x;
    let p2_c2 = u32(p2_s2 < p2_s1);
    let p2 = p2_s2 + p1_carry;
    let p2_c3 = u32(p2 < p2_s2);
    let p2_carry = p2_c1 + p2_c2 + p2_c3;  // 0, 1, 2, or 3

    // p3 = hh.y + p2_carry (bits 96-127). Cannot overflow because the
    // full product is < 2^128 and we've accounted for every bit below.
    let p3 = hh.y + p2_carry;

    return gl_reduce_128(vec4<u32>(p0, p1, p2, p3));
}

// --- Test kernel ----------------------------------------------------
//
// Reads `a` and `b` from their respective storage buffers, writes
// elementwise `a + b`, `a - b`, `a * b` to three output buffers.
// Differential-tested against the Rust `Goldilocks` impl in Phase
// B.1's validator. If any `gl_*` primitive is wrong, this test fires.

// Bindings layout (3 storage buffers, comfortably inside the WebGPU
// baseline limit of 4 per compute stage):
//   0: a_buf   — read-only inputs, length N
//   1: b_buf   — read-only inputs, length N
//   2: out_buf — read-write outputs, length 3N. Region layout is
//      [ add (N) | sub (N) | mul (N) ] so `out_buf[N_UNIFORM*op + i]`
//      addresses the i-th result of operation `op`.
// The uniform carries N so the shader can index the regions without
// a second `arrayLength` call (and without ambiguity when the compiler
// can't inline the result).

@group(0) @binding(0) var<storage, read>        a_buf:   array<vec2<u32>>;
@group(0) @binding(1) var<storage, read>        b_buf:   array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write>  out_buf: array<vec2<u32>>;

struct ArithParams {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(3) var<uniform> params: ArithParams;

const OUT_ADD_OFFSET: u32 = 0u;
const OUT_SUB_OFFSET: u32 = 1u;
const OUT_MUL_OFFSET: u32 = 2u;

@compute @workgroup_size(64)
fn gl_test_arith(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = params.n;
    if (i >= n) { return; }
    let a = a_buf[i];
    let b = b_buf[i];
    out_buf[OUT_ADD_OFFSET * n + i] = gl_add(a, b);
    out_buf[OUT_SUB_OFFSET * n + i] = gl_sub(a, b);
    out_buf[OUT_MUL_OFFSET * n + i] = gl_mul(a, b);
}

// Debug entry point — writes a sentinel pattern unconditionally.
// If this writes and the main entry point doesn't, the problem is in
// the arithmetic, not the dispatch plumbing.
@compute @workgroup_size(64)
fn gl_test_sentinel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = params.n;
    if (i >= n) { return; }
    out_buf[OUT_ADD_OFFSET * n + i] = vec2<u32>(0xDEADBEEFu, 0xCAFEBABEu);
    out_buf[OUT_SUB_OFFSET * n + i] = vec2<u32>(0x11111111u, 0x22222222u);
    out_buf[OUT_MUL_OFFSET * n + i] = vec2<u32>(i, i);
}
