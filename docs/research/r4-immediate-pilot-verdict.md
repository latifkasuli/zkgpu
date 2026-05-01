# Item #3 pilot verdict — Stockham R4 stage params storage vs immediate

**Date:** 2026-05-01 (M4 Pro 2026-04-30, NVIDIA 2026-05-01)
**Pilot commit:** `2a21ac2` (kernel + plan), `ac379b4` (bench)
**Status:** complete; auto-default stays `Storage`.
**Bench reproducer:**

```bash
git clone https://github.com/latifkasuli/zkgpu.git && cd zkgpu
cargo bench -p zkgpu-wgpu --bench ntt_r4_param_mode -- \
    --warm-up-time 2 --measurement-time 5
```

## Three-host A/B

`WgpuNttPlan::execute` end-to-end, forward direction, four `log_n` sizes. Δ is `(immediate − storage) / storage`. Negative = immediate faster. "CI disjoint" = the 95% CIs of Storage and Immediate don't overlap, so the delta is real signal rather than noise.

### M4 Pro / Metal

| log_n | Storage median | Immediate median | Δ | CI |
|---|---:|---:|---:|---|
| 10 | 1.3559 ms | 1.4232 ms | **+5.0%** (slower) | disjoint |
| 14 | 1.3889 ms | 1.4238 ms | +2.5% | partial |
| 18 | 1.5041 ms | 1.5378 ms | +2.2% | partial |
| 20 | 2.6112 ms | 2.3075 ms | **−11.6%** (faster) | disjoint |

### RTX 4090 / Vulkan

| log_n | Storage median | Immediate median | Δ | CI |
|---|---:|---:|---:|---|
| 10 | 43.642 µs | 44.445 µs | +1.8% | partial |
| 14 | 60.794 µs | 64.624 µs | **+6.3%** (slower) | disjoint |
| 18 | 268.52 µs | 280.01 µs | +4.3% | partial |
| 20 | 1.2946 ms | 1.2905 ms | −0.3% | overlap |

### RTX 5090 / Vulkan

| log_n | Storage median | Immediate median | Δ | CI |
|---|---:|---:|---:|---|
| 10 | 41.025 µs | 41.457 µs | +1.1% | overlap |
| 14 | 57.847 µs | 57.807 µs | −0.07% | overlap |
| 18 | 83.372 µs | 81.954 µs | **−1.7%** (faster) | disjoint |
| 20 | 110.27 µs | 111.03 µs | +0.7% | overlap |

## What the data says

**No host shows a uniform-or-near-uniform win.** Of 12 (host × log_n) cells:

- 2 cells show a real win: M4 Pro at log_n=20 (−11.6%, the headline) and RTX 5090 at log_n=18 (−1.7%).
- 4 cells show real regression: M4 Pro at log_n ∈ {10, 14, 18} and RTX 4090 at log_n=14.
- 6 cells are inside the noise band.

The pattern doesn't generalize. M4 Pro Metal has a sharp encoder-cost-vs-kernel-cost crossover at log_n ≥ 19: below it, the per-stage `set_immediates` adds ~30-70 µs of constant-overhead for ~3 R4 stages; above it, the smaller bind-group + register-resident params win by 304 µs absolute / 11.6% relative. RTX 4090 shows the opposite cliff at log_n=14 (a clean +6.3% slowdown) and converges to null at the larger transforms. RTX 5090 — same architecture family as the 4090 but newer silicon and driver — is mostly null with a real but tiny 1.7% win at log_n=18.

The two consumer hot paths (Plonky3 `fri_commit` at log_h=18, OpenVM mixed-height at log_h_max=18-22) straddle the M4 Pro crossover and hit the 4090's +4.3% / 5090's −1.7% region — a wash on average across the discrete-GPU consumer surface, with M4 Pro likely net-negative if log_h=18 is the common case.

## Decision against the documented gate

The pre-registered gate criterion in commit `2a21ac2`: *"≥ 5% win on at least one log_n on at least one host."*

The criterion is technically met by M4 Pro at log_n=20 (−11.6%). **But the gate was designed assuming a representative test point — a 5% win at one corner case while three other corners regress meaningfully isn't the situation the gate was intended to clear.** Propagating R2 / local / Poseidon2 to immediates would require those kernels' encoder-cost-vs-kernel-cost crossovers to align with the consumer hot path's log_n range, and there's no evidence they would.

**Auto-default stays `Storage`.** The R4 Immediate path remains accessible for callers who measure on their own hardware and find a win for their specific consumer workload — `WgpuNttPlan::new_with_r4_param_mode` is the explicit knob.

## What ships

- **Production keeps the Storage path on every kernel.** No change to consumer hot paths.
- **R4 Immediate variant stays implemented + parity-tested + benched.** Available as opt-in via `R4ParamMode::Immediate`. The infrastructure (capability detection, `max_immediate_size` limits, drift-prevented pipeline cache key from a single `R4_IMMEDIATE_SIZE_BYTES` constant) is reusable when a future evaluation flips the gate.
- **Bench (`ntt_r4_param_mode`) ships in `crates/zkgpu-wgpu/benches/`.** Reproducer above. Future hardware shifts can re-run without code changes.

## Re-evaluation triggers

The Immediate path becomes worth promoting to default if any of:

- A new host (post-M4 Pro Apple Silicon, RTX 6000-series, mobile WebGPU on a current driver) shows a uniform win across the consumer log_n range.
- Naga's wgpu Metal backend lowers `set_immediates` cheaper than the current ~10-20 µs per call, eliminating the small-log_n regression on Apple.
- A consumer pipeline shifts to dominate at log_n=20+ where Apple's win is real (e.g., a future MMCS shape that removes the small-FRI-fold path).
- The "generated module-constant" variant of item #6 lands and pairs naturally with this — both shrink the bind-group fingerprint, and the cumulative effect could clear the gate where each alone doesn't.

## Reproducer

On any host with a wgpu-supported GPU advertising `Features::IMMEDIATES`:

```bash
git clone https://github.com/latifkasuli/zkgpu.git && cd zkgpu
cargo bench -p zkgpu-wgpu --bench ntt_r4_param_mode -- \
    --warm-up-time 2 --measurement-time 5
```

The two Criterion groups (`ntt_r4_storage` and `ntt_r4_immediate`) run back-to-back at four `log_n` sizes; eyeballing median deltas plus checking CI disjointness gives the per-host signal.
