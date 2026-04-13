// Tiled matrix transpose for four-step NTT (portable)
//
// Transposes an R x C row-major matrix into a C x R row-major matrix.
// Uses workgroup shared memory for coalesced reads and writes.
// Tile dimension matches workgroup layout: TILE x TILE.
//
// Shared memory has +1 padding per row to eliminate bank conflicts on
// the transposed read (without padding: 8-way conflicts when TILE=16
// on 32-bank GPUs).

const TILE: u32 = 16u;
const PADDED_STRIDE: u32 = 17u; // TILE + 1 for bank-conflict-free access

var<workgroup> tile: array<u32, 272>; // TILE * PADDED_STRIDE

@group(0) @binding(0) var<storage, read>       src: array<u32>;
@group(0) @binding(1) var<storage, read_write>  dst: array<u32>;

struct TransposeParams {
    rows: u32,
    cols: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(2) var<uniform> params: TransposeParams;

@compute @workgroup_size(16, 16)
fn transpose_tiles(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let rows = params.rows;
    let cols = params.cols;

    let tile_col = wg_id.x;
    let tile_row = wg_id.y;

    let src_row = tile_row * TILE + lid.y;
    let src_col = tile_col * TILE + lid.x;

    // Load: coalesced read from src (row-major R x C)
    if src_row < rows && src_col < cols {
        tile[lid.y * PADDED_STRIDE + lid.x] = src[src_row * cols + src_col];
    }

    workgroupBarrier();

    // Store: transposed write to dst (row-major C x R)
    let dst_row = tile_col * TILE + lid.y;
    let dst_col = tile_row * TILE + lid.x;

    if dst_row < cols && dst_col < rows {
        dst[dst_row * rows + dst_col] = tile[lid.x * PADDED_STRIDE + lid.y];
    }
}
