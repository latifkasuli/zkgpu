# Galaxy Tab S11 — Mali-G925-Immortalis regression + capability check (2026-04-16)

**Phase.** Post-G.2.3 regression check + Mali cohort coverage
extension. Not a milestone on its own; slots in alongside G.2.3
post-landing validation.
**Status.** ✅ **Complete** — newest-gen Mali silicon fires
`HeuristicMaliCollapse` exactly as older generations, G925-Immortalis
is **faster than every prior Mali generation measured** at every
tested size.
**Budget spent.** ~3 min wall-clock (single BrowserStack App
Automate session, Free-plan parallel pool).

## TL;DR outcome

**Rule-correctness (regression):** All three family rules remain
correctly scoped on the newest Mali silicon. Mali rule fires 20×;
Xclipse and Adreno rules correctly abstain.

**Capability:** Mali-G925-Immortalis is the **fastest Mali GPU
we've measured** — log-22 forward GlobalR4 = **8.64 ms**, beating
every Mali-G715 unit (14.70–15.37 ms) by a factor of ~1.7×. Puts
it in the same performance tier as Adreno 750 / Xclipse 960 on
zkgpu NTT workloads. Absolute log-22 GlobalR4 timings now approach
parity with the Adreno-830-class silicon (pa3q 6.02 ms, Adreno 840
S26 Ultra 5.83 ms) despite being a different architecture family.

**Shape:** UNCONDITIONAL @ log21, weakest cell +46.3% (log 21 inv),
strongest +95.1% (log 18 fwd). Every cell in the 10-cell matrix
was a Global win, and 10/10 cells were `global-big` (≥20% gate).
This is the crispest UNCONDITIONAL signal in the whole Mali cohort.

## Why this run exists

Three independent motivations:

1. **Mali rule regression check on newest silicon.** G.1.4 shipped
   the rule based on 5 devices × 4 gens (G78/G710/G715×3). Mali-G925-
   Immortalis (Valhall-5 / Immortalis-flagship generation, 2025/2026
   Dimensity 9400+ silicon) is the first available post-G715 Mali
   GPU. Confirming the rule still fires cleanly forward-extrapolates
   the rule's validity statement from "Valhall/Bifrost G78..G715"
   to "Valhall..Immortalis G78..G925" — covering the full Mali
   product family currently in the market.
2. **G.2.3 cross-family regression.** Post-landing validation
   already showed the Xclipse rule doesn't misfire on Adreno 840 /
   830 or Mali-G715. Adding Mali-G925 gives the sharpest possible
   "is the Xclipse arm's `matches!(caps.gpu_family, Xclipse)` guard
   tight across all current Mali silicon" check.
3. **Capability gate for the cohort story.** G.1.3 documented "all
   Mali generations viable for zk-proving, weakest cell within 1.2×
   of flagship G715." G925-Immortalis is a new datapoint that shifts
   that story: the flagship ceiling is no longer G715, and the
   1.2×-of-flagship gate now implies a tighter absolute threshold.

## Matrix

- Device: `Samsung Galaxy Tab S11-16.0` (BrowserStack App Automate
  Free-plan axis, Android 16, real hardware)
- App APK: G.2.3 build (`bs://e484c9802f49f5be7f85aea71eb7cd6c5aee291f`,
  app-debug.apk 19.82 MB, built 2026-04-16 16:25, Xclipse rule baked in)
- Test APK: G.1.4/G.2.3 test APK
  (`bs://8c74fce5129ce38757d682367be6bed6ede90d11`, unchanged from
  cohort runs)
- Full `ZkgpuInstrumentedTest` class (10 test cases)

### BrowserStack build info

| Field | Value |
|---|---|
| Build ID | `40629f504db3ba8ac6ad28c0df761a97561d2feb` |
| Session ID | `be20d3e72b8ba8b03871f451c2f275a829c2892e` |
| Outcome | Passed 10/10 |
| Duration | 151 seconds |
| Project | `zkgpu-g23-tabs11-newest-mali` |

## Silicon identified from logcat

| Field | Value |
|---|---|
| GPU name | **Samsung / ARM Mali-G925-Immortalis** |
| Family | Mali |
| Backend | Vulkan |
| Tier | UnifiedMemoryNative |
| Driver | `v1.r49p1-03bet0.f8a98b506b89b21c80a508d70abfcb3f` |
| Max buffer | 268435456 (256 MB) |
| Workgroup limits | max_wg_x=256, max_invocations=256, max_wg_storage_bytes=16352 |

**Note on naming.** The logcat reports `Mali-G925-Immortalis` —
ARM's Immortalis tier is the top-end G9xx-series flagship. "Mali"
prefix + "Immortalis" suffix is the current naming convention (Mali
is the umbrella; Immortalis is the ray-tracing-capable tier). For
our purposes this is a single family tag: `GpuFamily::Mali`.

**Note on driver.** `r49p1-03bet0` is a third Mali major driver
revision added to the cohort — previous coverage was `r38p1` (Tensor
G1/G2 on Pixel 6/7) and `r51p0` (Tensor G3/G4 on Pixel 8/9). `r49p1`
sits between those two versions chronologically. That the rule
shape is identical across r38 → r49 → r51 reinforces the G.1.3
finding that the signal is a GPU-microarchitecture property rather
than a driver-version artifact.

## Rule-firing summary

| Counter | Count | Expectation |
|---|---:|---|
| `HeuristicMaliCollapse` | **20** | ✅ matches G.1.4 baseline (every Mali device fires 20× under full suite) |
| `HeuristicAdrenoCollapse` | 0 | ✅ correctly abstains |
| `HeuristicXclipseCollapse` | 0 | ✅ G.2.3 rule correctly scoped |
| `HeuristicDefaultLocal` | 0 | ✅ no fall-through |
| ForcedLocal | 10 | ✅ crossover test fired all 10 forced-A cells |
| ForcedGlobal | 10 | ✅ crossover test fired all 10 forced-B cells |
| Stale (`Heuristic*LargeN`) | 0 | ✅ no cached phase-e APK, fresh rule set |

## Forced-A/B timing table

Raw GPU timings from `tail_ab_report.json`:

| log_n | dir | local ms | global ms | global win |
|---|---|---:|---:|---:|
| 18 | fwd |  2.63 |  0.13 | **+95.1%** |
| 18 | inv |  3.01 |  0.18 | **+94.0%** |
| 19 | fwd |  6.16 |  0.42 | **+93.2%** |
| 19 | inv |  6.35 |  0.61 | **+90.4%** |
| 20 | fwd |  9.43 |  1.33 | **+85.9%** |
| 20 | inv |  9.85 |  2.41 | **+75.5%** |
| 21 | fwd | 11.65 |  6.05 | **+48.1%** |
| 21 | inv | 12.22 |  6.56 | **+46.3%** |
| 22 | fwd | 28.98 |  8.64 | **+70.2%** |
| 22 | inv | 31.50 |  8.41 | **+73.3%** |

**`zkgpu-tail-analyze` verdict:** UNCONDITIONAL @ log21, 10/10 cells
`global-big`. **Weakest cell (log 21 inv) at +46.3% — higher than
any Mali-G715 device's weakest cell.**

## Capability comparison — updated cohort table

Log-22 forward GlobalR4 timings, newest measurement bold:

| Silicon | Device | SoC (year) | log22 fwd ms | ratio vs G925 |
|---|---|---|---:|---:|
| **Mali-G925-Immortalis** | **Galaxy Tab S11** | **Dimensity 9400+ (2025)** | **8.64** | **1.00× (new baseline)** |
| Mali-G715 (r51p0) | Pixel 9 Pro XL (komodo) | Tensor G4 (2024) | 14.85 | 1.72× |
| Mali-G715 (r51p0) | Pixel 9 Pro Fold (comet) | Tensor G4 (2024) | 15.37 | 1.78× |
| Mali-G715 (r51p0) | Pixel 8 Pro (husky) | Tensor G3 (2023) | 14.70 | 1.70× |
| Mali-G710 (r38p1) | Pixel 7 (panther) | Tensor G2 (2022) | 15.83 | 1.83× |
| Mali-G78 MP20 (r38p1) | Pixel 6 (oriole) | Tensor G1 (2021) | 18.01 | 2.08× |

Across all Mali flavors measured, G925-Immortalis is **~1.7–2.1×
faster at log 22 GlobalR4 than any predecessor** — the biggest
intra-Mali generational jump in the cohort (G78→G710→G715 changes
were much smaller).

### Cross-family benchmarking snapshot

Comparing G925 to the fastest per-family ceiling we've measured:

| Silicon | log22 fwd ms | Family |
|---|---:|---|
| Adreno 840 (S26 Ultra, Snapdragon 8 Elite Gen 5) | 5.83 | Adreno (fastest) |
| Adreno 830 (pa3q, Snapdragon 8 Elite) | 6.02 | Adreno |
| **Mali-G925-Immortalis (Tab S11, Dimensity 9400+)** | **8.64** | **Mali (new fastest)** |
| Xclipse 960 (S26, Exynos 2600) | 9.16 | Xclipse (fastest) |
| Adreno 750 (e1q/e2q, Snapdragon 8 Gen 3) | ~6 | Adreno |
| Xclipse 940 (S24, Exynos 2400) | 10.50 | Xclipse |

**Mobile GPU performance landscape for zkgpu NTT (2026):** current-
gen flagships cluster between 5.8–10.5 ms log22-fwd GlobalR4 across
all three mobile family types. The intra-family spread (e.g.,
Adreno 750→840) is now smaller than the cross-family spread between
Mali's flagship and Adreno's flagship — a convergence that simplifies
capability gating substantially.

### Why G925 is so much faster

Likely architectural factors (not verified; reasoned from the data):

1. **Immortalis tier** uses 10-core+ Valhall-5 configurations with
   higher FP32 throughput per core.
2. **Dimensity 9400+** ships LPDDR5X at higher effective bandwidth
   than Tensor G4's LPDDR5 — and NTT at log 22 is memory-bound.
3. **Driver r49p1** may have workgroup-storage tuning that improves
   our strided-gather path *less* than predecessors (making LocalR4
   even more unattractive and widening the GlobalR4 win).

None of these factors affects the *rule shape* — the unconditional
win is still present, just with larger margins. The rule is
correct; the silicon has gotten faster.

## Implications

### Mali cohort summary update

The full Mali coverage span now reads:

| Gen | SoC | GPU | Driver | Rule fires | log22 fwd GlobalR4 |
|---|---|---|---|:-:|---:|
| 1 | Tensor G1 (2021) | G78 MP20 | r38p1 | ✅ | 18.01 ms |
| 2 | Tensor G2 (2022) | G710 MC7 | r38p1 | ✅ | 15.83 ms |
| 3 | Tensor G3 (2023) | G715 MC7 | r51p0 | ✅ | 14.70 ms |
| 4 | Tensor G4 (2024) | G715 MC7 | r51p0 | ✅ | 14.85 ms (komodo) / 15.37 (comet) |
| **5** | **Dimensity 9400+ (2025)** | **G925-Immortalis** | **r49p1** | **✅** | **8.64 ms** |

**5 Mali silicon generations across 3 Mali driver major revisions
(r38p1 / r49p1 / r51p0) — rule shape identical on every cell.**

### G.2.3 cross-family regression on Mali-G925

| Counter | Expected | Observed | Verdict |
|---|---:|---:|:-:|
| HeuristicMaliCollapse | 20 | 20 | ✅ |
| HeuristicXclipseCollapse | 0 | 0 | ✅ |
| HeuristicAdrenoCollapse | 0 | 0 | ✅ |
| Stale | 0 | 0 | ✅ |

The Xclipse rule's `matches!(caps.gpu_family, Xclipse)` guard is
tight on the newest Mali silicon, confirming the insertion order
(Adreno → Mali → Xclipse → Default) does not leak across families.

### Capability-gate update

The "≤30 ms at log 22 GlobalR4 is viable for Plonky3-scale provers"
gate from G.1.3 is satisfied with a very generous margin by every
measured Mali device, and the ceiling has shifted down. For a
Plonky3 prover doing 100 NTTs at log 22 on Galaxy Tab S11
(Mali-G925), the NTT portion is **~0.9 seconds** — competitive with
Snapdragon flagships.

## Links / references

- G.2.3 rule-landing code: `crates/zkgpu-wgpu/src/ntt/planner/tail_policy.rs`
- G.2.3 post-landing: `../xclipse-rule-postlanding-2026-04-16/README.md`
- G.1.4 Mali rule landing: `../mali-rule-validation-2026-04-16/README.md`
- G.1.3 Mali older-gen: `../mali-older-gen-2026-04-16/README.md`
- Parent plan: `../../../../../plan/foundation-audit-and-family-rules-roadmap-2026-04-15.md`
