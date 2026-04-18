// Goldilocks Stockham radix-2 DIF global-only stage kernel.
//
// Prepended at build time with `goldilocks_arith_helpers.wgsl` (see
// `ntt/goldilocks/plan.rs` for the concat! wiring). The helpers provide
// `gl_add`, `gl_sub`, `gl_mul`, and the u32x2 limb primitives this
// kernel depends on.
//
// Algorithm: auto-sort Stockham DIF, one butterfly per thread. Input
// buffer is consumed, output buffer is written. The host swaps their
// roles across stages (ping-pong). Twiddles are a host-precomputed
// flat array of `N - 1` entries, indexed by `params.twiddle_offset + j`.
//
// For butterfly index `bfly_idx = g * half + j`:
//   i_top = g * 2 * half + j
//   i_bot = i_top + half
//   tw    = twiddles[twiddle_offset + j]         (= ω^(j · n_groups))
//   out_top = g * half + j                       (in the "sum" region)
//   out_bot = out_top + half_total               (in the "tw · diff" region)
// with half_total = N / 2 independent of stage (total butterflies).
//
// At stage s: n_groups = 2^s, half = N / 2^(s+1), so half_total = n_groups · half.

struct StockhamR2Params {
    n: u32,
    half: u32,
    half_total: u32,
    twiddle_offset: u32,
    // Number of threads per row of the 2D-folded dispatch grid.
    // Equals `groups_per_row * WORKGROUP_SIZE`. The kernel reconstructs
    // the flat butterfly index as `gid.x + gid.y * row_stride`. Folding
    // is required because WebGPU guarantees only
    // `max_compute_workgroups_per_dimension >= 65535`, which a 1D
    // dispatch exceeds at `log_n >= 23` (N/2 > 65535 * 64).
    row_stride: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read>        input_buf:    array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write>  output_buf:   array<vec2<u32>>;
@group(0) @binding(2) var<storage, read>        twiddles_buf: array<vec2<u32>>;
@group(0) @binding(3) var<uniform>              params:       StockhamR2Params;

@compute @workgroup_size(64)
fn gl_stockham_r2(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bfly_idx = gid.x + gid.y * params.row_stride;
    if (bfly_idx >= params.half_total) { return; }

    let half = params.half;
    let g = bfly_idx / half;
    let j = bfly_idx % half;

    let i_base = g * 2u * half;
    let i_top = i_base + j;
    let i_bot = i_top + half;

    let o_top = g * half + j;
    let o_bot = o_top + params.half_total;

    let u = input_buf[i_top];
    let v = input_buf[i_bot];
    let tw = twiddles_buf[params.twiddle_offset + j];

    let sum = gl_add(u, v);
    let diff = gl_sub(u, v);
    let diff_tw = gl_mul(diff, tw);

    output_buf[o_top] = sum;
    output_buf[o_bot] = diff_tw;
}
