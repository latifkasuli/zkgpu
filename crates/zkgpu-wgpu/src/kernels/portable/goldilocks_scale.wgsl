// Goldilocks in-place element-wise scaling kernel.
//
// Prepended with `goldilocks_arith_helpers.wgsl` at build time. Used
// for inverse-NTT n⁻¹ normalisation — the scalar is set by the host
// (Goldilocks::new(n).inv()) and passed via a uniform in u32x2 limb
// form (lo, hi).
//
// One thread per element; single dispatch.

struct ScaleParams {
    n: u32,
    scalar_lo: u32,
    scalar_hi: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> data: array<vec2<u32>>;
@group(0) @binding(1) var<uniform>             params: ScaleParams;

@compute @workgroup_size(64)
fn gl_scale(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    let scalar = vec2<u32>(params.scalar_lo, params.scalar_hi);
    data[i] = gl_mul(data[i], scalar);
}
