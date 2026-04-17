// BabyBear four-step twiddle multiply kernel (portable, vectorised)
// P = 2^31 - 2^27 + 1 = 2013265921
//
// For data viewed as a rows x cols matrix in row-major order,
// multiplies element (r, c) by twiddle_table[r * cols + c].
// The twiddle table is precomputed on the host.
//
// In-place: reads and writes the same storage buffer.
// Each thread processes 4 consecutive elements for better ILP.
//
// NVIDIA scale-up Tier 2A Option A (2026-04-16): the companion
// `twiddle_table_prime` buffer was dropped from this pass. Shoup's
// Barrett reduction (`mod_mul_shoup`) saves a few cycles per call but
// required a parallel `w_prime` buffer the same size as `twiddle_table`.
// At log 22 the combined 32 MiB twiddle working set was the second-
// biggest contributor to the RTX 4090 L2 partial-fit cache-thrashing
// cliff (see `research/benchmarks/nvidia-scale-up-2026-04-16/tier-2a-
// log22-cliff-investigation.md`). Using the 10-iteration `mod_mul`
// reducer here costs ~60 µs of compute at log 22 but saves ~450 µs
// of twiddle-buffer memory traffic — net ~390 µs win, plus knock-on
// effects from shrinking the working set back under L2 capacity.
// Leaf kernels keep Shoup's optimization because their twiddle
// buffers are O(√N), cache-friendly, and don't participate in the
// cliff.

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
    data[base] = mod_mul(d0, t0);

    if base + 1u < total {
        let d1 = data[base + 1u];
        let t1 = twiddle_table[base + 1u];
        data[base + 1u] = mod_mul(d1, t1);
    }
    if base + 2u < total {
        let d2 = data[base + 2u];
        let t2 = twiddle_table[base + 2u];
        data[base + 2u] = mod_mul(d2, t2);
    }
    if base + 3u < total {
        let d3 = data[base + 3u];
        let t3 = twiddle_table[base + 3u];
        data[base + 3u] = mod_mul(d3, t3);
    }
}
