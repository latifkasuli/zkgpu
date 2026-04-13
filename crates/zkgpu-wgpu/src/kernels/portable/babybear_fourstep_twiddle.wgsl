// BabyBear four-step twiddle multiply kernel (portable, vectorised)
// P = 2^31 - 2^27 + 1 = 2013265921
//
// For data viewed as a rows x cols matrix in row-major order,
// multiplies element (r, c) by twiddle_table[r * cols + c].
// The twiddle table is precomputed on the host.
//
// In-place: reads and writes the same storage buffer.
// Each thread processes 4 consecutive elements for better ILP.

const P: u32 = 2013265921u;
const WORKGROUP_SIZE_X: u32 = 256u;
const ELEMS_PER_THREAD: u32 = 4u;

@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<storage, read>       twiddle_table: array<u32>;

struct TwiddleParams {
    total_elements: u32,
    groups_per_row: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(2) var<uniform> params: TwiddleParams;
@group(0) @binding(3) var<storage, read> twiddle_table_prime: array<u32>;

fn mulhi(a: u32, b: u32) -> u32 {
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
    let lo_sum = ll + mid_lo_shifted;
    let lo_carry = select(0u, 1u, lo_sum < ll);

    return hh + (mid >> 16u) + (mid_carry << 16u) + lo_carry;
}

fn mod_mul_shoup(a: u32, w: u32, w_prime: u32) -> u32 {
    let q = mulhi(a, w_prime);
    var r = a * w - q * P;
    if r >= P { r -= P; }
    return r;
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

@compute @workgroup_size(256)
fn fourstep_twiddle(@builtin(global_invocation_id) gid: vec3<u32>) {
    let thread_idx = gid.x + gid.y * params.groups_per_row * WORKGROUP_SIZE_X;
    let base = thread_idx * ELEMS_PER_THREAD;
    let total = params.total_elements;

    if base >= total {
        return;
    }

    let d0 = data[base];
    let t0 = twiddle_table[base];
    let tp0 = twiddle_table_prime[base];
    data[base] = mod_mul_shoup(d0, t0, tp0);

    if base + 1u < total {
        let d1 = data[base + 1u];
        let t1 = twiddle_table[base + 1u];
        let tp1 = twiddle_table_prime[base + 1u];
        data[base + 1u] = mod_mul_shoup(d1, t1, tp1);
    }
    if base + 2u < total {
        let d2 = data[base + 2u];
        let t2 = twiddle_table[base + 2u];
        let tp2 = twiddle_table_prime[base + 2u];
        data[base + 2u] = mod_mul_shoup(d2, t2, tp2);
    }
    if base + 3u < total {
        let d3 = data[base + 3u];
        let t3 = twiddle_table[base + 3u];
        let tp3 = twiddle_table_prime[base + 3u];
        data[base + 3u] = mod_mul_shoup(d3, t3, tp3);
    }
}
