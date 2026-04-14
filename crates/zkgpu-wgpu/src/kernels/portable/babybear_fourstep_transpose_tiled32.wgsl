// Tiled matrix transpose for four-step NTT (portable, 32x32 tile variant)
//
// Transposes an R x C row-major matrix into a C x R row-major matrix.
// Differs from the baseline `babybear_fourstep_transpose.wgsl` in two ways:
//
//   1. Tile is 32x32 (vs 16x16), so shared memory = 32 * 33 * 4B = 4224 bytes
//      — still comfortably under the 16352-byte ceiling reported by every
//      Android Vulkan device we currently benchmark against.
//   2. Workgroup layout is (32, 8) = 256 threads (vs 16x16 = 256). Each
//      thread processes TILE / BLOCK_ROWS = 4 elements (striped rows), which
//      gives 4 outstanding memory requests per thread for latency hiding
//      and divides the workgroup count by 4 at any given problem size.
//
// Shared memory still uses `+1` padding per row to eliminate bank conflicts
// on the transposed read.
//
// Reference: NVIDIA "An Efficient Matrix Transpose in CUDA C/C++" (Harris,
// 2013). The same 32x32 / (32, 8) / 4-rows-per-thread shape is the canonical
// tuning for bandwidth-bound transposes across NVIDIA, AMD, and mobile GPUs.

const TILE: u32 = 32u;
const BLOCK_ROWS: u32 = 8u;               // workgroup y-dim
const ROWS_PER_THREAD: u32 = 4u;          // TILE / BLOCK_ROWS
const PADDED_STRIDE: u32 = 33u;           // TILE + 1 for bank-conflict-free access

var<workgroup> tile: array<u32, 1056>;    // TILE * PADDED_STRIDE

@group(0) @binding(0) var<storage, read>       src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;

struct TransposeParams {
    rows: u32,
    cols: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(2) var<uniform> params: TransposeParams;

@compute @workgroup_size(32, 8)
fn transpose_tiles(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let rows = params.rows;
    let cols = params.cols;

    let tile_col = wg_id.x;
    let tile_row = wg_id.y;

    let src_col = tile_col * TILE + lid.x;
    let src_row_base = tile_row * TILE + lid.y;

    // Load: 4 coalesced rows per thread, striped by BLOCK_ROWS.
    for (var i: u32 = 0u; i < ROWS_PER_THREAD; i = i + 1u) {
        let src_row = src_row_base + i * BLOCK_ROWS;
        if src_row < rows && src_col < cols {
            tile[(lid.y + i * BLOCK_ROWS) * PADDED_STRIDE + lid.x] =
                src[src_row * cols + src_col];
        }
    }

    workgroupBarrier();

    // Store: transposed write, same 4-row striping swapped across the tile.
    // dst is C x R row-major, so dst_row walks along the original column axis
    // and dst_col walks along the original row axis.
    let dst_col = tile_row * TILE + lid.x;
    let dst_row_base = tile_col * TILE + lid.y;

    for (var i: u32 = 0u; i < ROWS_PER_THREAD; i = i + 1u) {
        let dst_row = dst_row_base + i * BLOCK_ROWS;
        if dst_row < cols && dst_col < rows {
            dst[dst_row * rows + dst_col] =
                tile[lid.x * PADDED_STRIDE + (lid.y + i * BLOCK_ROWS)];
        }
    }
}
