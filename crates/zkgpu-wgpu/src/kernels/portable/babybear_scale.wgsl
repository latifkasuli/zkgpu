// BabyBear in-place element-wise scaling kernel (portable, vectorised)
// P = 2^31 - 2^27 + 1 = 2013265921
//
// Multiplies every element by a uniform scalar.
// Used for inverse NTT 1/n scaling without a host round-trip.
// Each thread processes 4 consecutive elements for better ILP.

const P: u32 = 2013265921u;
const WORKGROUP_SIZE_X: u32 = 256u;
const ELEMS_PER_THREAD: u32 = 4u;

@group(0) @binding(0) var<storage, read_write> data: array<u32>;

struct ScaleParams {
    n: u32,
    scalar: u32,
    groups_per_row: u32,
    _pad1: u32,
}

@group(0) @binding(1) var<uniform> params: ScaleParams;

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

@compute @workgroup_size(256)
fn scale_elements(@builtin(global_invocation_id) gid: vec3<u32>) {
    let thread_idx = gid.x + gid.y * params.groups_per_row * WORKGROUP_SIZE_X;
    let base = thread_idx * ELEMS_PER_THREAD;
    let n = params.n;
    let s = params.scalar;

    if base >= n {
        return;
    }

    data[base] = mod_mul(data[base], s);

    if base + 1u < n {
        data[base + 1u] = mod_mul(data[base + 1u], s);
    }
    if base + 2u < n {
        data[base + 2u] = mod_mul(data[base + 2u], s);
    }
    if base + 3u < n {
        data[base + 3u] = mod_mul(data[base + 3u], s);
    }
}
