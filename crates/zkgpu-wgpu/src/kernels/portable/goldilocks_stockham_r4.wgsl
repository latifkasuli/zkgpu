// Goldilocks Stockham radix-4 DIF global-only stage kernel.
//
// Prepended at build time with `goldilocks_arith_helpers.wgsl`.
//
// One R4 stage combines two R2 stages. Total R4 stages for a size-N
// NTT with even log_n = log_n / 2. For odd log_n, the Rust plan falls
// back to the radix-2 kernel (see `goldilocks_stockham_r2.wgsl`) —
// this kernel is only dispatched when log_n is even.
//
// Algorithm (auto-sort Stockham DIF R4, one butterfly per thread):
//
//   For R4 stage `s` (0-indexed, 0..log_n/2):
//     n_groups = 4^s                (1 at the coarsest)
//     quarter  = N / 4^(s+1)         (butterflies per group)
//     total    = N / 4               (butterflies this stage)
//
//   For butterfly bfly_idx = g * quarter + j:
//     i_base = g * 4 * quarter
//     x0..x3 = input[i_base + k*quarter + j] for k in 0..4
//
//     Inner R4 (NTT analogue of FFT radix-4, with i_N = ω^(N/4)
//     playing the role of √-1):
//       t0 = x0 + x2
//       t1 = x0 - x2
//       t2 = x1 + x3
//       t3 = i_N · (x1 - x3)
//       y0 = t0 + t2
//       y1 = t1 + t3
//       y2 = t0 - t2
//       y3 = t1 - t3
//
//     Twiddles (precomputed by host as 3 entries per butterfly):
//       tw1 = ω^(j · n_groups · 1)
//       tw2 = ω^(j · n_groups · 2)
//       tw3 = ω^(j · n_groups · 3)
//
//     Output (4-way auto-sort split, with twiddles applied):
//       out[g * quarter + j + 0 * (N/4)] = y0
//       out[g * quarter + j + 1 * (N/4)] = y1 · tw1
//       out[g * quarter + j + 2 * (N/4)] = y2 · tw2
//       out[g * quarter + j + 3 * (N/4)] = y3 · tw3

struct StockhamR4Params {
    n: u32,
    quarter: u32,
    total_bfly: u32,      // = N / 4
    twiddle_offset: u32,  // start index into twiddles_buf for this stage
    i_n_lo: u32,          // ω^(N/4) stored as u32x2 limbs
    i_n_hi: u32,
    // Number of threads per row of the 2D-folded dispatch grid.
    // See the R2 kernel's `row_stride` doc for the WebGPU-limit rationale.
    row_stride: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read>        input_buf:    array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write>  output_buf:   array<vec2<u32>>;
@group(0) @binding(2) var<storage, read>        twiddles_buf: array<vec2<u32>>;
@group(0) @binding(3) var<uniform>              params:       StockhamR4Params;

@compute @workgroup_size(64)
fn gl_stockham_r4(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bfly_idx = gid.x + gid.y * params.row_stride;
    if (bfly_idx >= params.total_bfly) { return; }

    let quarter = params.quarter;
    let g = bfly_idx / quarter;
    let j = bfly_idx % quarter;

    // Input addresses: 4 consecutive quarter-strides within one group.
    let i_base = g * 4u * quarter;
    let x0 = input_buf[i_base + 0u * quarter + j];
    let x1 = input_buf[i_base + 1u * quarter + j];
    let x2 = input_buf[i_base + 2u * quarter + j];
    let x3 = input_buf[i_base + 3u * quarter + j];

    // Inner R4 butterfly, using i_N (the primitive 4th root of unity
    // in F_p) to play the role of √-1.
    let i_n = vec2<u32>(params.i_n_lo, params.i_n_hi);
    let t0 = gl_add(x0, x2);
    let t1 = gl_sub(x0, x2);
    let t2 = gl_add(x1, x3);
    let t3_pre = gl_sub(x1, x3);
    let t3 = gl_mul(i_n, t3_pre);

    let y0 = gl_add(t0, t2);
    let y1 = gl_add(t1, t3);
    let y2 = gl_sub(t0, t2);
    let y3 = gl_sub(t1, t3);

    // Twiddles (3 per butterfly, laid out tw1, tw2, tw3 per j).
    let tw_base = params.twiddle_offset + 3u * j;
    let tw1 = twiddles_buf[tw_base + 0u];
    let tw2 = twiddles_buf[tw_base + 1u];
    let tw3 = twiddles_buf[tw_base + 2u];

    // Output addresses: 4-way split into output buffer.
    let o_base = g * quarter + j;
    let nover4 = params.n / 4u;
    output_buf[o_base + 0u * nover4] = y0;
    output_buf[o_base + 1u * nover4] = gl_mul(y1, tw1);
    output_buf[o_base + 2u * nover4] = gl_mul(y2, tw2);
    output_buf[o_base + 3u * nover4] = gl_mul(y3, tw3);
}
