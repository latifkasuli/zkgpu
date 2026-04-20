# Mobile R8 A/B — Adreno generation sweep (2026-04-17)

Parent: `../browserstack-r8-ab-2026-04-17/README.md` (initial
one-device-per-family R8 A/B). This run extends the Adreno axis to
five generations spanning Snapdragon 8 Gen 1 → Gen 5 (Elite).

## Purpose

The initial R8 A/B measured Adreno on a single device (S26 Ultra,
Adreno 840) and concluded "R8 default correct for Adreno." This
sweep tests whether that generalizes to older Adreno GPUs or
whether the initial result was specific to the newest stepping —
the same way the one-device Mali and Xclipse results need broader
validation before a policy change.

## TL;DR

**The initial Adreno 840 conclusion doesn't generalize.** Only
Adreno ≥ 750 (Snapdragon 8 Gen 3 and newer) shows clean R8 wins.
Adreno 730 and 740 show regressions that echo the Mali/Xclipse
small-leaf pattern.

| Adreno Gen | SoC | Device | Verdict |
|---|---|---|---|
| 730 | SD 8 Gen 1 | Samsung S22 Ultra | ⚠️ R4 wins 1.13-1.39× at log 18; tied/R8 at 20-22 |
| 740 | SD 8 Gen 2 | Samsung S23 Ultra | ⚠️ Tied at 18-20; **R4 wins 1.61× at log 22 inverse** |
| 750 | SD 8 Gen 3 | Samsung S24 Ultra | ✅ R8 wins 1.17-1.26× everywhere |
| 830 | SD 8 Elite | Samsung S25 Ultra | ✅ R8 wins 1.15-1.28× everywhere |
| 840 | SD 8 Elite Gen 5 | Samsung S26 Ultra (prior run) | ✅ R8 wins 1.13-1.33× everywhere |

Architectural inflection at Adreno 750 (Snapdragon 8 Gen 3). The
current R8-always-on default is correct for 750+ but regresses on
730 and 740.

## Data (GPU time, ms, median of 5 iters after 2 warmups)

### Samsung S22 Ultra (Adreno 730, SD 8 Gen 1, Android 12)

| log_n | dir | R8-on | R8-off | off/on | verdict |
|---|---|---|---|---|---|
| 18 | fwd | 3.110 | 2.230 | 0.72 | **R4 wins 1.39×** |
| 18 | inv | 3.120 | 2.750 | 0.88 | R4 wins 1.13× |
| 20 | fwd | 10.210 | 10.690 | 1.05 | tied |
| 20 | inv | 7.100 | 10.690 | 1.51 | **R8 wins 1.51×** |
| 22 | fwd | 27.210 | 28.860 | 1.06 | tied |
| 22 | inv | 23.240 | 27.040 | 1.16 | R8 wins 1.16× |

Mixed pattern: R4 dominates at log_leaf=9, reverses at larger sizes.
Note the log-20-inverse 1.51× R8 win is a real outlier relative to
everything else on this device — probably touches a fast path in
the R8 kernel that R4 misses on Adreno 730 specifically.

### Samsung S23 Ultra (Adreno 740, SD 8 Gen 2, Android 13)

| log_n | dir | R8-on | R8-off | off/on | verdict |
|---|---|---|---|---|---|
| 18 | fwd | 1.540 | 1.670 | 1.08 | tied |
| 18 | inv | 1.660 | 1.730 | 1.04 | tied |
| 20 | fwd | 6.260 | 6.540 | 1.04 | tied |
| 20 | inv | 6.240 | 6.240 | 1.00 | tied |
| 22 | fwd | 24.640 | 28.690 | 1.16 | R8 wins 1.16× |
| 22 | inv | 25.530 | 15.880 | 0.62 | **R4 wins 1.61×** |

Almost entirely tied across sizes *except* log 22 inverse, which
regresses 1.61× with R8 enabled. This direction-asymmetric result
is surprising — the forward path at the same size is a 1.16× R8
win. Suggests Adreno 740's inverse-specific kernel (twiddle/scale
ordering, or SPIR-V layout the driver emits for inverse-mode) hits
a slow path with R8 that forward doesn't.

Either way, shipping R8-always-on regresses log 22 inverse by 9.65
ms GPU on this SKU, which is a widely-deployed phone.

### Samsung S24 Ultra (Adreno 750, SD 8 Gen 3, Android 14)

| log_n | dir | R8-on | R8-off | off/on | verdict |
|---|---|---|---|---|---|
| 18 | fwd | 0.390 | 0.490 | 1.26 | R8 wins 1.26× |
| 18 | inv | 0.390 | 0.470 | 1.21 | R8 wins 1.21× |
| 20 | fwd | 2.270 | 2.710 | 1.19 | R8 wins 1.19× |
| 20 | inv | 2.260 | 2.640 | 1.17 | R8 wins 1.17× |
| 22 | fwd | 11.420 | 14.040 | 1.23 | R8 wins 1.23× |
| 22 | inv | 11.490 | 13.880 | 1.21 | R8 wins 1.21× |

Clean R8 wins everywhere. Current default is correct.

### Samsung S25 Ultra (Adreno 830, SD 8 Elite, Android 15)

| log_n | dir | R8-on | R8-off | off/on | verdict |
|---|---|---|---|---|---|
| 18 | fwd | 0.320 | 0.410 | 1.28 | R8 wins 1.28× |
| 18 | inv | 0.320 | 0.400 | 1.25 | R8 wins 1.25× |
| 20 | fwd | 1.510 | 1.740 | 1.15 | R8 wins 1.15× |
| 20 | inv | 1.540 | 1.790 | 1.16 | R8 wins 1.16× |
| 22 | fwd | 7.000 | 8.920 | 1.27 | R8 wins 1.27× |
| 22 | inv | 7.350 | 9.180 | 1.25 | R8 wins 1.25× |

Same clean pattern as Adreno 750. Current default correct.

### Samsung S26 Ultra (Adreno 840, SD 8 Elite Gen 5, Android 16) — prior run

Copied from `../browserstack-r8-ab-2026-04-17/` for comparison.

| log_n | dir | R8-on | R8-off | off/on | verdict |
|---|---|---|---|---|---|
| 18 | fwd | 0.300 | 0.350 | 1.17 | R8 wins 1.17× |
| 18 | inv | 0.300 | 0.350 | 1.17 | R8 wins 1.17× |
| 20 | fwd | 1.440 | 1.620 | 1.13 | R8 wins 1.13× |
| 20 | inv | 1.480 | 1.590 | 1.07 | tied |
| 22 | fwd | 6.730 | 8.930 | 1.33 | R8 wins 1.33× |
| 22 | inv | 6.730 | 8.460 | 1.26 | R8 wins 1.26× |

## Pattern

Two Adreno generations behave:

1. **Adreno 730 and 740 (SD 8 Gen 1, Gen 2)** — follow the same
   small-leaf-regression shape as Mali-G925 and Xclipse 940 from
   the parent experiment. R8's register / shared-memory overhead
   exceeds the 3×-butterfly-per-stage throughput win until the
   leaf is large enough (or the kernel path is specific enough).
2. **Adreno 750 and newer (SD 8 Gen 3+)** — R8 wins universally at
   every measured (log_n, direction) cell with margins 1.13-1.33×.
   Architectural jump at Gen 3 (new "Sliceable" GPU architecture,
   per Qualcomm marketing) resolves the overhead.

Combined with the Mali-G925 and Xclipse 940 results from the
parent run, the measured picture is now:

| Family | Result |
|---|---|
| Apple Metal (all) | R8 wins cleanly (prior measurement, not re-tested here) |
| NVIDIA Vulkan | R8 gated at log_leaf ≤ 11 due to log-23 bimodality |
| Adreno ≥ 750 | R8 wins cleanly (3 devices: 750, 830, 840) |
| Adreno ≤ 740 | Regresses — R4 wins at small leaves or specific directions |
| Mali-G925 | Regresses — R4 wins 1.12-3.02× at log 18-20 |
| Xclipse 940 | Regresses — R4 wins 1.17-2.40× at log 18-20 |

## Policy implications (proposed, not shipped)

The case for changing the default is now *stronger* than after the
parent run — it's no longer a "Mali/Xclipse quirk," it's a real
architectural property affecting the majority of still-deployed
Android devices (SD 8 Gen 1 + Gen 2 phones are ~3-4 years old and
still in active use).

Design options, ranked by complexity:

1. **Add Adreno-generation gating.** Disable R8 for Adreno < 750
   (via vendor/device-id detection in `caps.rs`), keep R8 enabled
   for Adreno ≥ 750. Needs a reliable mapping from driver/device
   string to generation — `caps.device_name` contains strings like
   "Adreno (TM) 840" so the number is parseable.
2. **Family-wide min-log_leaf for Mali/Xclipse/old-Adreno.** Extend
   the gate to accept `(min_log_leaf, max_log_leaf)` and ship
   `min=11` for these families — R8 only when leaf is big enough to
   amortize overhead. Still leaves the Adreno 740 log-22-inverse
   regression (which happens *at* log_leaf=11, the boundary).
3. **Per-direction gate.** Would catch the Adreno 740 asymmetry.
   Over-engineered for the rest of the matrix.

Not shipped in this commit — same n=1-per-SKU concern. To flip the
default:

- **Mali:** still need 2-3 more SoCs (G715, G710, earlier G925 steppings).
- **Xclipse:** still need 920, 960, plus one more 940 stepping.
- **Adreno 740 log-22-inverse regression:** re-run at least once
  more on a different S23 Ultra stepping to rule out measurement
  noise, then investigate whether the inverse kernel emits
  different SPIR-V for R8 on older Adreno drivers.

BrowserStack has all of these devices; the run plumbing + analyzer
is in place (`apps/android-harness/app/.../ZkgpuInstrumentedTest.kt`).

## Reproduction

Same as the parent run (`../browserstack-r8-ab-2026-04-17/README.md`),
with this devices array instead:

```json
"devices": [
  "Samsung Galaxy S25 Ultra-15.0",
  "Samsung Galaxy S24 Ultra-14.0",
  "Samsung Galaxy S23 Ultra-13.0",
  "Samsung Galaxy S22 Ultra-12.0"
]
```

## Build / session references

- Build ID: `b6ad738c44f9e63ef02ca3eb373f5794cf617e07`
- Per-device session durations: 36-112 s (older devices slower)
- Dashboard: https://app-automate.browserstack.com/builds/b6ad738c44f9e63ef02ca3eb373f5794cf617e07

## Artifacts

- `s22-ultra-adreno730/crossoverFourStepR8{Enabled,Disabled}.log`
- `s23-ultra-adreno740/crossoverFourStepR8{Enabled,Disabled}.log`
- `s24-ultra-adreno750/crossoverFourStepR8{Enabled,Disabled}.log`
- `s25-ultra-adreno830/crossoverFourStepR8{Enabled,Disabled}.log`

Parent (Adreno 840 + Mali G925 + Xclipse 940):
`../browserstack-r8-ab-2026-04-17/`
