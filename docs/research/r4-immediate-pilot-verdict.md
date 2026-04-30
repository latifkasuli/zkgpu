# Item #3 pilot interim verdict — Stockham R4 stage params storage vs immediate

**Date:** 2026-04-30
**Pilot commit:** `2a21ac2` (kernel + plan), `ac379b4` (bench), this file
**Status:** **interim — NVIDIA Vulkan A/B pending vast.ai availability.**
**Bench reproducer:**

```bash
git clone https://github.com/latifkasuli/zkgpu.git && cd zkgpu
cargo bench -p zkgpu-wgpu --bench ntt_r4_param_mode -- \
    --warm-up-time 2 --measurement-time 5
```

## What the pilot ships

- `R4ParamMode { Storage, Immediate }` on the public NTT API.
- Auto-detect default is **`Storage`** even when the device advertises `Features::IMMEDIATES`. Mirrors item #6's pattern: the pilot path is validated for parity but doesn't switch the default until per-host bench data confirms a clean win.
- New WGSL kernel `babybear_stockham_r4_immediate.wgsl` — algorithmically identical to the existing R4 kernel, with the binding-3 uniform replaced by `var<immediate>` and a 4-entry BGL.
- `WgpuNttPlan::new_with_r4_param_mode` exposes the explicit override for benching and for callers who measure on their own hardware and want to flip per-plan.

## Headline numbers (M4 Pro / Metal)

`WgpuNttPlan::execute` end-to-end, forward direction, four log_n sizes. Δ is `(immediate − storage) / storage`. Negative = immediate faster.

| log_n | Storage median | Immediate median | Δ | CI overlap? |
|---|---:|---:|---:|---|
| 10 | 1.3559 ms | 1.4232 ms | **+5.0%** (immediate slower) | none |
| 14 | 1.3889 ms | 1.4238 ms | +2.5% | partial |
| 18 | 1.5041 ms | 1.5378 ms | +2.2% | partial |
| 20 | 2.6112 ms | 2.3075 ms | **−11.6%** (immediate faster) | none |

The +5.0% at log_n=10 has Storage's 95% CI upper bound (1.3628 ms) below Immediate's lower bound (1.4109 ms) — the regression is real, not noise. Same for the −11.6% at log_n=20: Immediate's upper (2.4224 ms) is below Storage's lower (2.5062 ms).

## Read

The constant ~30-70 µs slowdown at small log_n suggests an encoder-side cost: the per-stage `pass.set_immediates(0, &bytes)` call apparently adds ~10-20 µs per R4 stage on the wgpu Metal backend — small in absolute terms, but visible on a 1.4 ms baseline with 3 R4 stages. At log_n=20, kernel time crosses over and the smaller bind group + register-resident params win by 304 µs absolute / 11.6% relative.

This is roughly the encoder-cost-vs-kernel-cost crossover the speed-opportunities doc predicted, but on Metal it lands at log_n ≥ 19 rather than universally. The two consumer hot paths (Plonky3 `fri_commit` at log_h=18, OpenVM mixed-height at log_h_max=18-22) straddle the boundary.

## Why default stays `Storage` for now

1. **Reproducible regression on Apple Silicon at log_n ∈ {10, 14, 18}.** Apple is the portability gate for the v0.2/v0.3 narrative; flipping the default with a known 2-5% small-log_n regression there isn't worth it without compensating data.
2. **No NVIDIA data yet.** Both vast.ai instances were unreachable when the bench ran. NVIDIA Vulkan may show a uniform win or may show the same crossover; without that data the auto-detect decision is speculative.
3. **Nothing else regresses.** 243 zkgpu-wgpu + 118 plonky3 + 22 openvm tests all pass with Immediate as the default — bit-parity is solid. The decision is purely about wall time.

## Re-evaluation triggers

Flip the auto-detect default to Immediate (or to a thresholded Immediate-at-large-log_n mix) when any of:

- NVIDIA Vulkan A/B (RTX 4090 + 5090) shows a clean win at the consumer-hot-path log_n range.
- Naga's Metal codegen / wgpu's Metal `set_immediates` lowering improves enough that the small-log_n regression collapses.
- The pilot propagates to R2/local/Poseidon2 (item #3's stretch goal) and the cumulative encoder-side savings from dropping multiple bind-group entries cross the threshold.

## What's next

- **NVIDIA bench when boxes are back.** Reproducer command above; output goes into `research/benchmarks/r4-immediates-pilot-2026-04-30/{rtx4090,rtx5090}_vulkan.txt`.
- **iOS WebGPU via BrowserStack.** Apple A-series GPUs share Apple Silicon's constant-cache architecture; same regression on M4 Pro's small-log_n is a fair prediction. Worth a confirmatory run, but not a blocker.
- **Item #5 (re-scoped).** Per the speed-opportunities Gate 2 ordering, item #5 follows item #3. The encoder-cluster category continues there; this verdict doesn't gate it.
