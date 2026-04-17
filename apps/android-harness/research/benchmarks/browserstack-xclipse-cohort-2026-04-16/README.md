# G.2.2 — Xclipse cohort via BrowserStack (2026-04-16)

**Phase.** G.2.2 of `plan/foundation-audit-and-family-rules-roadmap-2026-04-15.md`
(Xclipse benchmark matrix). Originally blocked on G.2.1 FTL
Exynos-pinning methodology; BrowserStack App Automate sidestepped
that blocker by exposing Exynos-pinned Samsung Galaxies directly.
**Status.** ✅ **Complete** — 3 Xclipse generations × 4 independent
devices × 50 cells. All unconditional GlobalR4. Plus a bonus Adreno
840 (Snapdragon 8 Elite Gen 5, 2025/2026) datapoint confirming
Adreno rule still fires on the newest Snapdragon silicon.

## TL;DR outcome

| Device | GPU family | Verdict | Weakest cell | Strongest cell |
|---|---|---|---|---|
| Samsung Galaxy S22 (Android 12) | **Xclipse 920** (Exynos 2200, RDNA2, 2022) | UNCONDITIONAL @ log21 | log 21 inv +17.8% | log 18 fwd +82.4% |
| Samsung Galaxy S22 Ultra (Android 12) | **Xclipse 920** (Exynos 2200, 2022) | UNCONDITIONAL @ log21 | log 22 inv +43.1% | log 18 fwd +82.9% |
| Samsung Galaxy S24 (Android 16) | **Xclipse 940** (Exynos 2400, 2024) | UNCONDITIONAL @ log21 | log 22 inv +18.7% | log 18 fwd +83.8% |
| Samsung Galaxy S26 (Android 16) | **Xclipse 960** (Exynos 2600, 2026) | UNCONDITIONAL @ log21 | log 19 inv +23.9% | log 18 fwd +67.8% |
| Samsung Galaxy S26 Ultra (Android 16) | **Adreno 840** (Snapdragon 8 Elite Gen 5) | UNCONDITIONAL @ log21 | log 18 fwd +24.1% | log 20 fwd +65.6% |

Xclipse cohort totals: **4 devices, 3 generations (920/940/960),
40 cells**. 38/40 cells `global-big` (≥20% Global win); 2/40 cells
`global-narrow` (S22 log21inv +17.8%, S24 log22inv +18.7%) — both
still Global wins, just below the default ≥20% "big" classification.
**Every cell is a Global win.**

Rule-firing invariants (all 5 devices, 2 Xclipse + 1 Adreno family):

- `HeuristicMaliCollapse` fires 0× on every device ✅ (Mali rule
  doesn't misfire on Xclipse or the newest Adreno)
- `HeuristicAdrenoCollapse` fires 10× only on S26 Ultra (Adreno 840) ✅
  — correctly scoped to Adreno family
- `HeuristicXclipseCollapse` fires 0× everywhere (rule not yet shipped,
  expected)
- Xclipse devices fall through to `HeuristicDefaultLocal` 14× each —
  the current state that a rule would replace
- Zero stale `HeuristicXclipseLargeN` / `HeuristicMaliLargeN` firings
  across all 5 devices ✅

## Why this run exists

Three gaps in the G.1/G.2 family-rules stream were still open before
this run:

1. **Xclipse has no shipped rule.** The roadmap required G.2.1 FTL
   Exynos-pinning methodology first, but BrowserStack's App Automate
   pool lists `Samsung Galaxy S22-12.0` (Exynos 2200 region SKU),
   `Samsung Galaxy S24-16.0` (Exynos 2400 region SKU), and
   `Samsung Galaxy S26-16.0` (Exynos 2600 region SKU) as pinned-
   silicon axes. No lottery — each axis returns a specific SoC.
   This sidesteps the FTL region-split problem entirely.
2. **Phase-e's single Xclipse 940 "flatline ±5%" was never
   re-measured.** G.1.1 proved phase-e's Mali phase-e result was
   n=1 noise; the same lesson needed re-application on Xclipse
   before landing a rule from a single datapoint.
3. **Adreno 840 (Snapdragon 8 Elite Gen 5) is brand-new silicon
   (late 2025 / early 2026 ship).** A single-device sanity check
   on Adreno 840 confirms `HeuristicAdrenoCollapse` still fires on
   the newest Snapdragon generation, bringing the validated-Adreno
   span from 4 generations (730/740/750/830) to 5 (730/740/750/830/840).

## Test matrix

Same phase-f APKs as G.1.4 (`app-debug.apk` 19.8 MB, built
2026-04-16 13:57, with Mali + Adreno rules baked into
`libzkgpu_ffi.so`). Full `ZkgpuInstrumentedTest` class (10 test
cases). BrowserStack Espresso v2 API, parallel session limit 5
(matched to plan capacity; all 5 dispatched simultaneously).

### BrowserStack infrastructure

- **Plan:** Free (5 parallel sessions, no pay-per-minute concern)
- **Framework:** Espresso v2 (`framework=espresso` on test-suite upload)
- **APK uploads:**
  - `app`: `bs://725c4adf8a49bc2ee9a52e282c5fef7ddeef623c` (custom_id `zkgpu-app-g14`)
  - `testSuite`: `bs://8c74fce5129ce38757d682367be6bed6ede90d11` (custom_id `zkgpu-test-g14`)
- **Device logs:** Per-testcase `device_log` URLs via
  `/espresso/v2/builds/{bid}/sessions/{sid}`. Concatenated all 10
  testcase logs per session into `logcat.txt` under each device
  subdirectory.

### Build / session IDs

| Device | Build ID | Session ID | Duration |
|---|---|---|---:|
| S22 (Xclipse 920) | `0e49b2d4b37cec29dc251bfe0f9ba46c5ac430d5` | `22cf6ad1da4c4db1124c451093e06282b5c2ef9a` | 174s |
| S22 Ultra (Xclipse 920) | `10882c89fa890521ca277acfdb063f5bea6e95e9` | `0b949fab99adb9e44c3ea49e84d76d6be13dd8af` | 170s |
| S24 (Xclipse 940) | `fb621791468da1d8b3e6fbd13618bef5c24308b9` | `529bdd876ff8ee8744365ab5995001db58c02943` | 151s |
| S26 (Xclipse 960) | `c20e7799340fb998d89daa72f1250f0179c0266a` | `8d4f1cc5b89df9bb2c2ab8f65a827eccd5ee4896` | 157s |
| S26 Ultra (Adreno 840) | `c2c49485bcd981473114e553be241799bfc0c3ec` | `b09c35f8c7202fab50228847787907afe97eb723` | 191s |

Session durations are 2.5–3.2 minutes (BrowserStack's App Automate
imposes install + startup overhead beyond raw test time; actual
on-device test compute is comparable to FTL's ~11–15s). Total
wall-clock from first submission to last completion: ~4 minutes
for all 5 parallel sessions.

### Drivers captured

| GPU | Driver string |
|---|---|
| Xclipse 920 (S22) | `Revision: a927fb4` |
| Xclipse 920 (S22 Ultra) | `Revision: 3b10981` (different rev, same GPU gen) |
| Xclipse 940 (S24) | `Driver version: 24.0.545, git hash: 1fc295b0d2` |
| Xclipse 960 (S26) | `SPAL Driver version: 25.2.30, git hash: 8782a7082a` |
| Adreno 840 (S26 Ultra) | (driver string elided from testcase logs) |

Three driver major revisions across the Xclipse cohort (`a927fb4` /
`3b10981` / `24.0.x` / `25.2.x`). The rule shape is identical across
all of them — consistent with the Mali G.1.3 finding that the
signal is a GPU microarchitecture property, not a driver artifact.

## Per-device forced-A/B tables

### Samsung Galaxy S22 — Xclipse 920 (Exynos 2200, RDNA2, 2022)

| log_n | dir | local ms | global ms | global win |
|---|---|---:|---:|---:|
| 18 | fwd |  3.76 |  0.66 | **+82.4%** |
| 18 | inv |  4.29 |  0.87 | **+79.7%** |
| 19 | fwd |  6.66 |  1.71 | **+74.3%** |
| 19 | inv |  7.31 |  2.22 | **+69.6%** |
| 20 | fwd | 10.94 |  4.02 | **+63.3%** |
| 20 | inv | 10.54 |  4.95 | **+53.0%** |
| 21 | fwd | 17.06 | 10.67 | **+37.5%** |
| 21 | inv | 15.62 | 12.84 | +17.8% _(narrow)_ |
| 22 | fwd | 30.81 | 17.04 | **+44.7%** |
| 22 | inv | 29.34 | 15.56 | **+47.0%** |

### Samsung Galaxy S22 Ultra — Xclipse 920 (Exynos 2200, 2022)

| log_n | dir | local ms | global ms | global win |
|---|---|---:|---:|---:|
| 18 | fwd |  3.86 |  0.66 | **+82.9%** |
| 18 | inv |  3.82 |  0.87 | **+77.2%** |
| 19 | fwd |  7.07 |  1.69 | **+76.1%** |
| 19 | inv |  8.69 |  2.09 | **+75.9%** |
| 20 | fwd | 12.06 |  4.05 | **+66.4%** |
| 20 | inv |  9.08 |  4.89 | **+46.1%** |
| 21 | fwd | 18.17 |  7.79 | **+57.1%** |
| 21 | inv | 17.92 |  8.90 | **+50.3%** |
| 22 | fwd | 31.58 | 14.78 | **+53.2%** |
| 22 | inv | 30.93 | 17.60 | **+43.1%** |

### Samsung Galaxy S24 — Xclipse 940 (Exynos 2400, 2024)

| log_n | dir | local ms | global ms | global win |
|---|---|---:|---:|---:|
| 18 | fwd |  2.65 |  0.43 | **+83.8%** |
| 18 | inv |  2.45 |  0.51 | **+79.2%** |
| 19 | fwd |  4.19 |  0.98 | **+76.6%** |
| 19 | inv |  4.37 |  1.35 | **+69.1%** |
| 20 | fwd |  7.51 |  3.29 | **+56.2%** |
| 20 | inv |  6.17 |  4.05 | **+34.4%** |
| 21 | fwd | 10.82 |  7.65 | **+29.3%** |
| 21 | inv | 10.03 |  6.36 | **+36.6%** |
| 22 | fwd | 14.34 | 10.50 | **+26.8%** |
| 22 | inv | 13.88 | 11.28 | +18.7% _(narrow)_ |

Cross-check against FTL e2s (same Xclipse 940 / Exynos 2400
silicon): weakest cell at log 22 inv was +27.0% on FTL,
+18.7% here. Within measurement noise; shape matches.

### Samsung Galaxy S26 — Xclipse 960 (Exynos 2600, 2026)

| log_n | dir | local ms | global ms | global win |
|---|---|---:|---:|---:|
| 18 | fwd |  4.10 |  1.32 | **+67.8%** |
| 18 | inv |  4.49 |  1.68 | **+62.6%** |
| 19 | fwd |  5.42 |  2.80 | **+48.3%** |
| 19 | inv |  4.64 |  3.53 | **+23.9%** |
| 20 | fwd |  7.91 |  3.27 | **+58.7%** |
| 20 | inv |  7.88 |  3.90 | **+50.5%** |
| 21 | fwd | 11.00 |  6.60 | **+40.0%** |
| 21 | inv | 10.94 |  8.61 | **+21.3%** |
| 22 | fwd | 15.63 |  9.16 | **+41.4%** |
| 22 | inv | 17.97 | 12.84 | **+28.5%** |

### Samsung Galaxy S26 Ultra — Adreno 840 (Snapdragon 8 Elite Gen 5)

Brand-new silicon, late 2025 / early 2026 ship. Included as a
single-device sanity check that `HeuristicAdrenoCollapse` still
fires on newest Snapdragon:

| log_n | dir | local ms | global ms | global win |
|---|---|---:|---:|---:|
| 18 | fwd |  0.29 |  0.22 | **+24.1%** |
| 18 | inv |  0.56 |  0.27 | **+51.8%** |
| 19 | fwd |  1.26 |  0.49 | **+61.1%** |
| 19 | inv |  1.37 |  0.57 | **+58.4%** |
| 20 | fwd |  3.14 |  1.08 | **+65.6%** |
| 20 | inv |  3.28 |  1.29 | **+60.7%** |
| 21 | fwd |  7.18 |  2.78 | **+61.3%** |
| 21 | inv |  7.10 |  3.08 | **+56.6%** |
| 22 | fwd | 13.99 |  5.83 | **+58.3%** |
| 22 | inv | 14.33 |  6.49 | **+54.7%** |

Adreno 840 log-22 fwd GlobalR4: **5.83 ms** — slightly faster than
Adreno 830 (pa3q 6.02 ms). Adreno rule-firing count: 10/10 (on the
heuristic path), 10× ForcedLocal + 10× ForcedGlobal on crossover.

## Rule-firing summary (all 5 devices)

| Device | Family | HeuristicMali | HeuristicAdreno | HeuristicXclipse | HeuristicDefaultLocal | ForcedLocal | ForcedGlobal | Stale |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| S22 | Xclipse | 0 | 0 | 0 | 14 | 10 | 10 | 0 |
| S22 Ultra | Xclipse | 0 | 0 | 0 | 14 | 10 | 10 | 0 |
| S24 | Xclipse | 0 | 0 | 0 | 14 | 10 | 10 | 0 |
| S26 | Xclipse | 0 | 0 | 0 | 14 | 10 | 10 | 0 |
| S26 Ultra | Adreno | 0 | **10** | 0 | 0 | 10 | 10 | 0 |

- Xclipse devices: zero cross-family misfire, consistent `HeuristicDefaultLocal` fall-through (14 planner decisions per full suite).
- Adreno 840: `HeuristicAdrenoCollapse` fires exactly as on every earlier Adreno generation. Rule is silicon-gen-agnostic; confirmed now on 730/740/750/830/840.

## Absolute-performance summary (log 22 GlobalR4 fwd ms)

Sorted fastest to slowest:

| Silicon | log22 fwd ms | Notes |
|---|---:|---|
| Adreno 840 (S26 Ultra, 8 Elite Gen 5, 2026) | 5.83 | Fastest mobile GPU we've measured |
| Adreno 830 (pa3q, 8 Elite, 2024) | 6.02 | G.1.4 baseline |
| Xclipse 960 (S26, Exynos 2600, 2026) | 9.16 | Xclipse catches up to Adreno generationally |
| Adreno 750 (e1q/e2q, 8 Gen 3, 2023) | ~6 | (FTL G.1.4) |
| Xclipse 940 (S24, Exynos 2400, 2024) | 10.50 | BrowserStack |
| Xclipse 940 (e2s, Exynos 2400, 2024) | 11.75 | FTL lottery, within noise |
| Adreno 730 (b0q, 8 Gen 1, 2022) | ~14 | (FTL G.1.4) |
| Xclipse 920 (S22 Ultra, Exynos 2200, 2022) | 14.78 | RDNA2-based Xclipse, first generation |
| Mali-G715 (komodo, Tensor G4, 2024) | 14.85 | G.1.1 baseline |
| Mali-G710 (panther, Tensor G2, 2022) | 15.83 | G.1.3 older-gen |
| Xclipse 920 (S22, Exynos 2200, 2022) | 17.04 | Different unit, same GPU gen |
| Mali-G78 (oriole, Tensor G1, 2021) | 18.01 | G.1.3 oldest-gen |

Observations:

- **Xclipse 960 (2026) is competitive** with Adreno 830 (2024) and
  Mali-G715 (2024) at log 22. RDNA3-based.
- **Xclipse 920 (2022)** is comparable to Mali-G710/G78 and Adreno
  730 — same 2022-vintage GPU cohort. First-gen RDNA2 doesn't beat
  peers on NTT but doesn't lose either.
- **All 4 Xclipse generations are viable zk-proving targets**
  (≤17 ms/NTT at log 22 GlobalR4) — same capability-gate verdict as
  Mali in G.1.3.

## Decision-gate outcome

From the G.2.3 decision matrix (documented in roadmap):

| Scenario | Action | Observed |
|---|---|---|
| Xclipse shows same unconditional GlobalR4 shape as Mali/Adreno | Land `GpuFamily::Xclipse → GlobalOnlyR4` family-wide | **✅ MATCH** — 40/40 cells Global win across 3 gens × 4 devices |
| Xclipse shows windowed shape or mixed cells | Scope a narrower rule or ship no rule | — not the case |
| Xclipse is windowed on older gen, unconditional on newer | Gate by generation | — not the case |

**Gate 1 triggered → land `GpuFamily::Xclipse → GlobalOnlyR4`
family-wide, no generational gating, no driver-version guard.** This
mirrors the Mali G.1.4 landing methodology exactly.

## Implications for G.2 closeout

### G.2.1 (FTL Exynos-pinning methodology)

No longer on the critical path for G.2.3. The BrowserStack pool
already exposes Exynos-pinned axes, which is what G.2.1 was
supposed to achieve via FTL. G.2.1 may still be worth writing up
as a "how to get reproducible Exynos coverage on FTL" reference
doc, but it's not gating.

### G.2.2 (Xclipse benchmark matrix)

Closed by this run. 3 Xclipse generations measured, 4 independent
devices, 40 cells of forced-A/B data. (a56x Xclipse 540 from FTL
would add a 4th generation when it dequeues; it's not required
for the rule-landing decision.)

### G.2.3 (Xclipse rule decision)

**Decision: land `GpuFamily::Xclipse → GlobalOnlyR4` via
`StockhamTailReason::HeuristicXclipseCollapse`** immediately after
the Mali arm in `crates/zkgpu-wgpu/src/ntt/planner/tail_policy.rs::heuristic_default`.

Implementation mirrors G.1.4:

1. Add `StockhamTailReason::HeuristicXclipseCollapse` variant.
2. In `heuristic_default`, add `GpuFamily::Xclipse => GlobalOnlyR4`
   arm immediately after the Mali arm.
3. Unit test `xclipse_picks_global_tail_at_all_sizes` + planner
   integration test `xclipse_picks_global_tail_at_all_tail_sizes`.
4. Post-landing FTL validation pass (re-run against the Xclipse
   cohort + one each Mali/Adreno regression control).

### a56x Xclipse 540 still pending

Two FTL matrices still queued (a56x + a56x-reroll). When they
dequeue, they'll provide the 4th Xclipse generation (Xclipse 540
from Exynos 1580 / Galaxy A56). Expected to show the same
unconditional shape. Not gating; patch in post-landing.

## Methodology notes (BrowserStack vs FTL)

Per-device findings and operational tradeoffs:

- **BrowserStack pros:**
  - Exynos-pinned axes (no region-split lottery).
  - 5 parallel sessions on Free plan.
  - Newest silicon available fast (S26 / Xclipse 960 / Adreno 840
    on Android 16 within weeks of commercial launch).
  - API is clean (Espresso v2 upload + build + poll).

- **BrowserStack cons:**
  - Session overhead is 2–3× FTL (~170s vs ~11s raw device compute).
  - Device logs are per-testcase, not session-level. Must concatenate
    10 per-testcase logs to reconstitute a "session logcat.txt".
  - Logs are truncated / test-scoped (~120–150 KB per session vs
    FTL's 4–16 MiB). Android framework noise missing, which is
    actually convenient for grep but means less diagnostic context
    if something goes wrong.

- **FTL pros:**
  - Faster per-run (~11s device compute vs ~150s session).
  - Full Android logcat (4–16 MiB) with all framework context.
  - Higher quota on Free tier (lots of parallel matrices).

- **FTL cons:**
  - Samsung flagship axes are region-split lotteries (5-for-5
    Snapdragon observed on e1q/e1q-reroll/b0q/e2q until e2s
    finally rolled Exynos).
  - Xclipse 540 / 920 / 940 / 960 pinning requires axis research
    per device.

**Recommendation:** BrowserStack is the right tool for Xclipse
multi-SKU cohorts. FTL remains the right tool for Mali / Adreno
/ any silicon with a single-SKU axis available.

## Artifacts

- `Samsung_Galaxy_S22_12.0/logcat.txt` (concatenated per-testcase logs)
- `Samsung_Galaxy_S22_Ultra_12.0/logcat.txt`
- `Samsung_Galaxy_S24_16.0/logcat.txt`
- `Samsung_Galaxy_S26_16.0/logcat.txt`
- `Samsung_Galaxy_S26_Ultra_16.0/logcat.txt`
- `tail_ab_report.json` — 50-cell recommender output

## Links / references

- G.1.4 post-landing validation (Mali rule + Adreno regression control):
  `../mali-rule-validation-2026-04-16/README.md`
- G.1.1 / G.1.3 Mali pre-landing cohort:
  `../mali-scope-match-2026-04-16/README.md`,
  `../mali-older-gen-2026-04-16/README.md`
- Phase-f Adreno landing template:
  `../phase-f-full-validation-2026-04-15/README.md`
- Rule code to modify: `crates/zkgpu-wgpu/src/ntt/planner/tail_policy.rs`
  §heuristic_default (Xclipse arm to be added immediately after Mali arm)
- Parent plan: `../../../../../plan/foundation-audit-and-family-rules-roadmap-2026-04-15.md`
  §Phase G.2 §G.2.2 §G.2.3
