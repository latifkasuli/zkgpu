// Portable Goldilocks arithmetic — test kernel body.
//
// This file is NEVER used on its own: it expects the arithmetic
// helpers from `goldilocks_arith_helpers.wgsl` to be prepended to it
// at build time (see `ntt/goldilocks/arith_test.rs` for the
// `concat!(include_str!(helpers), include_str!(this))` pattern).
//
// Exposes two entry points:
//   - `gl_test_arith`    — elementwise a+b, a-b, a*b, written to a
//                           packed [add | sub | mul] output region.
//   - `gl_test_sentinel` — diagnostic write of a fixed pattern for
//                           dispatch-plumbing verification.

// Bindings layout (3 storage buffers, comfortably inside the WebGPU
// baseline limit of 4 per compute stage):
//   0: a_buf   — read-only inputs, length N
//   1: b_buf   — read-only inputs, length N
//   2: out_buf — read-write outputs, length 3N. Region layout is
//      [ add (N) | sub (N) | mul (N) ] so `out_buf[params.n*op + i]`
//      addresses the i-th result of operation `op`.
// The uniform carries N so the shader can index the regions without
// a second `arrayLength` call.

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

@compute @workgroup_size(64)
fn gl_test_sentinel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = params.n;
    if (i >= n) { return; }
    out_buf[OUT_ADD_OFFSET * n + i] = vec2<u32>(0xDEADBEEFu, 0xCAFEBABEu);
    out_buf[OUT_SUB_OFFSET * n + i] = vec2<u32>(0x11111111u, 0x22222222u);
    out_buf[OUT_MUL_OFFSET * n + i] = vec2<u32>(i, i);
}
