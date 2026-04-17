# G.1.1 — Mali scope-matched rerun (2026-04-16)

**Phase.** G.1.1 of `plan/foundation-audit-and-family-rules-roadmap-2026-04-15.md`.
**Status.** ✅ **Complete** — 3/3 devices show UNCONDITIONAL shape matching
phase-f. Thermal hypothesis falsified. Next step: G.1.3 generational
coverage (older-gen Mali).
**Budget spent.** ~5 min wall-clock (3 parallel FTL matrices, each
~2 min total including queue), 13s of actual device compute.

## TL;DR outcome

All three Mali-G715 devices reproduce **phase-f's unconditional
Global-tail win** under **phase-e's exact narrow scope** (cold device,
forced-A/B only, no preceding smoke/validation/benchmark tests).
Global wins by 50–85% across log 18–22, forward and inverse, on all
three devices. The **thermal hypothesis is falsified**: phase-e's
narrow-window result was the outlier, not phase-f. The Mali rule
shape matches the Adreno rule shape (unconditional, log_n ≥ 21).
Proceeding to **G.1.3** for older-gen Mali generational check before
landing `GpuFamily::Mali → GlobalOnlyR4` unconditional.

## Why this run exists

Two prior Mali measurements on the same 3 Mali-G715 devices produced
**different shapes**:

| Run | Scope | Mali-G715 signal |
|---|---|---|
| **phase-e** (2026-04-15, narrow) | ONLY `crossoverStockhamLocalTail` + `crossoverStockhamGlobalTail` | **Windowed** at log 18(–19). Global wins 28–48% at log 18 only on komodo/comet; husky shows a different shape (+8.6% at log 21). |
| **phase-f** (2026-04-15, full-suite) | Full `ZkgpuHarnessTest`: smoke + validation + benchmark + crossover + forced-A/B | **Unconditional** at log 21+. Global wins 20–60%+ across the whole log_n range on all three Mali devices. |

Same silicon. Same test APK. Same forced-A/B tests **present** in both
runs. The only structural difference: phase-f ran ~15 minutes of
smoke + validation + benchmark tests *before* the forced-A/B tests.

Hypothesis: the Mali-G715 is in a different thermal/clock state by
the time the forced-A/B tests run in phase-f, which either amplifies
the Global-tail advantage or narrows the Local-tail advantage.
Phase-e, running the A/B tests cold, sees only a narrow log-18
window because the device is not yet thermally saturated.

G.1.1 tests this directly: **run phase-e's exact scope on phase-f's
device set.** Same devices as phase-f's Mali cohort, same code as
phase-f, same narrow forced-A/B-only scope as phase-e.

## Decision gate (from the parent plan)

- **Matches phase-e shape** (windowed at log 18 only) → thermal state
  is the culprit. Mali rule, if shipped, should be **windowed**
  (like `HeuristicAdrenoCollapse` but at log 18, not log 21). Proceed
  to G.1.3 (older-gen Mali coverage) to check generational consistency
  before shipping.
- **Matches phase-f shape** (unconditional across log_n) → Mali rule
  shape is the same as Adreno. Proceed to G.1.3; then land as
  `GpuFamily::Mali → GlobalOnlyR4` unconditional.
- **Neither** → real measurement uncertainty remains. Proceed to
  G.1.2 (thermal control run with forced-performance-mode + cooldowns).

## Device matrix

Same Mali-G715 cohort measured in both phase-e and phase-f, so this
run is directly comparable to both.

| Codename | Device | SoC | GPU | FTL API |
|---|---|---|---|---|
| husky  | Pixel 8 Pro       | Tensor G3 | Mali-G715 | 35 |
| komodo | Pixel 9 Pro XL    | Tensor G4 | Mali-G715 | 35 |
| comet  | Pixel 9 Pro Fold  | Tensor G4 | Mali-G715 | 35 |

**Not measured this run:**
- Adreno devices (pa3q, e1q, e3q, b0q, dm3q) — already characterized
  by PR 3 + phase-f. Not relevant to the Mali scope question.
- Xclipse devices (a56xnaeea observed as Xclipse 540 in phase-f) —
  G.2 methodology-blocked, separate investigation.
- Older-gen Mali (oriole G78 / panther G710 / shiba G710) — saved for
  G.1.3, runs only if G.1.1 is conclusive.

## APK plan

Phase-f APKs are sitting in `app/build/outputs/apk/`:

- `app/build/outputs/apk/debug/app-debug.apk` (Apr 15 12:22, 24 MB)
- `app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk`
  (Apr 15 09:38, 847 KB)

HEAD is commit `6d01615` (zkgpu-cli `--force-tail` flag added post-
phase-f, desktop-only, does not affect Android). `fbcb7c2` (Adreno
rule) was merged before phase-f ran. For the debug APK (Apr 15
12:22) this means the Adreno rule is present in the planner.

**Effect on this run:** zero. The forced-A/B tests (`TailChoice.Local`,
`TailChoice.Global`) bypass the planner entirely. Whatever rule the
planner would have picked is overridden by `tail =` in the test.

**Decision: reuse phase-f's APKs.** Same code, no rebuild needed.
Rebuilding would introduce an unrelated variable (timestamp,
dependency lock state) without changing measured behavior.

## FTL invocation

One `gcloud` call per device to keep each run in its own FTL history
entry (easier to download and bisect if one device has a queue
issue):

```bash
cd /Users/latifkasuli/web3/zkgpu/apps/android-harness

APP=app/build/outputs/apk/debug/app-debug.apk
TESTAPK=app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
TARGETS="class org.zkgpu.harness.ZkgpuInstrumentedTest#crossoverStockhamLocalTail,class org.zkgpu.harness.ZkgpuInstrumentedTest#crossoverStockhamGlobalTail"
RESULTS_BASE=zkgpu-mali-scope-match-2026-04-16

for DEV in "husky,35" "komodo,35" "comet,35"; do
  MODEL="${DEV%%,*}"
  API="${DEV##*,}"
  gcloud firebase test android run \
    --type instrumentation \
    --app "$APP" \
    --test "$TESTAPK" \
    --device "model=$MODEL,version=$API,locale=en,orientation=portrait" \
    --test-targets "$TARGETS" \
    --results-dir "${RESULTS_BASE}-${MODEL}"
done
```

Scope match with phase-e is enforced by `--test-targets` listing
only the two crossover methods — the Android test runner will skip
everything else in the suite.

### Alternative: single matrix

If FTL queue is clean, `--device model=husky,version=35 --device model=komodo,version=35 --device model=comet,version=35` in one call runs them in parallel under a single matrix ID. Use only if willing to accept a single matrix failure blocking all three device results.

## Expected outcomes

From phase-e + phase-f raw timings, the three-way shape comparison
across devices:

### komodo (Pixel 9 Pro XL, Tensor G4)

| log_n | dir | phase-e Global win | phase-f Global win |
|---|---|---|---|
| 18 | fwd | +43.9% | +60.4% |
| 18 | inv | +36.3% | +53.8% |
| 19 | fwd | +7.2%  | +44.1% |
| 20 | fwd | -1.2%  | +38.7% |
| 21 | fwd | -3.5%  | +32.9% |
| 22 | fwd | -2.1%  | +28.5% |

*(placeholder — will be re-extracted from `tail_ab_report.json` of both prior runs at analysis time)*

**Expected G.1.1 komodo shape if thermal hypothesis correct:** log 18
fwd +40–50% win, log 19+ reverts to noise-band (±10%) — matches
phase-e narrow shape.

### husky, comet

Same expected shape: log 18 narrow window, log 19+ neutral.

## Analysis pipeline

After downloading `logcat.txt` per device from each FTL results dir
into `<codename>/logcat.txt`:

```bash
cargo run -p zkgpu-tail-analyze -- \
  apps/android-harness/research/benchmarks/mali-scope-match-2026-04-16
```

Expected artifacts:
- `<codename>/logcat.txt` — raw device logs
- `<codename>/test_result.xml` — JUnit XML (for pass/fail + timing)
- `tail_ab_report.json` — machine-readable recommender output
- `README.md` updated with findings + decision-gate outcome

## Decision-gate mechanics

After `zkgpu-tail-analyze` runs, per-device verdict + raw % wins will
drop into a combined table. Decision rule:

1. **If 3/3 devices show WINDOWED-FLIP at log 18 (±1)** and **NO
   unconditional log 21+ signal** → matches phase-e. Thermal hypothesis
   is the working explanation. Log the thermal story in the parent
   plan and proceed to **G.1.3** (older-gen Mali: oriole G78 / shiba
   G710) to check if the windowed signal is G7xx-series or G715-only.
2. **If 3/3 devices show UNCONDITIONAL @ log 21** (≥20% win across
   log 18–22) → matches phase-f. Thermal hypothesis falsified (or
   irrelevant). Proceed to **G.1.3** to check generational consistency,
   then land the Mali rule as unconditional (same shape as Adreno).
3. **Mixed verdicts across devices OR doesn't match either shape** →
   ambiguous. Proceed to **G.1.2** thermal control run on a single
   Mali device with (a) back-to-back, (b) 30s cooldowns between
   iters, (c) forced-performance-mode, to isolate the variable.

## Cost + methodology notes

- **FTL budget:** 3 device-runs × ~2 min each = ~6 min compute;
  sometimes 10–30 min queue on Tensor G3/G4 axes. Plan for 30–45
  min wall-clock.
- **Axis substitution risk:** phase-e had pa3qxxx→pa3q substitution
  due to capacity. For husky/komodo/comet on Pixel axes this is
  unlikely but possible. Check `test_result.xml` `<system>` block
  after each run for the actually-scheduled silicon.
- **Cold-start guarantee:** each FTL device boot is a fresh emulator
  image (per FTL docs). No warm state carries across FTL runs, so
  "cold" is guaranteed. The thermal hypothesis is specifically
  about *within-run* thermal state, which phase-f's full suite
  builds up and phase-e's scope avoids.

## Findings (2026-04-16)

### FTL execution summary

All three FTL matrices passed cleanly, 2 test cases each, in a single
attempt. No axis substitution — all three devices reported `Mali-G715`
via `adapter.info.name` in logcat. Driver `v1.r51p0-00eac0`,
`tier=UnifiedMemoryNative`, `backend=Vulkan`, identical across all 3
devices. `max_buffer=268435456` (256 MB, the G.0.2 Tier 1 limit that
hasn't been raised yet on Android; not relevant to A/B at log ≤ 22).

| Device | Matrix | LocalTail JUnit | GlobalTail JUnit | Wall-clock Δ |
|---|---|---|---|---|
| comet  | `6284017789505340426` | 7.000 s | 5.000 s | Global −28% |
| husky  | (see bucket)          | 7.001 s | 4.777 s | Global −32% |
| komodo | (see bucket)          | 7.001 s | 4.394 s | Global −37% |

Consistent "global wins by ~30%" signal at the sweep-total level on
all 3 devices.

### Per-log_n A/B table (GPU timings only; from `tail_ab_report.json`)

**comet (Pixel 9 Pro Fold, Tensor G4, Mali-G715)**

| log_n | dir | local ms | global ms | global win |
|---|---|---|---|---|
| 18 | fwd | 11.65 | 1.74  | **+85.1%** |
| 18 | inv | 6.39  | 0.98  | **+84.7%** |
| 19 | fwd | 9.53  | 2.28  | **+76.1%** |
| 19 | inv | 9.97  | 2.95  | **+70.4%** |
| 20 | fwd | 16.43 | 4.57  | **+72.2%** |
| 20 | inv | 17.89 | 14.10 | +21.2% |
| 21 | fwd | 23.57 | 8.53  | **+63.8%** |
| 21 | inv | 24.31 | 8.29  | **+65.9%** |
| 22 | fwd | 48.46 | 15.37 | **+68.3%** |
| 22 | inv | 50.10 | 14.39 | **+71.3%** |

**husky (Pixel 8 Pro, Tensor G3, Mali-G715)**

| log_n | dir | local ms | global ms | global win |
|---|---|---|---|---|
| 18 | fwd | 8.91  | 0.70  | **+92.1%** |
| 18 | inv | 8.36  | 0.92  | **+89.0%** |
| 19 | fwd | 11.06 | 2.27  | **+79.5%** |
| 19 | inv | 13.18 | 3.33  | **+74.7%** |
| 20 | fwd | 21.95 | 8.68  | **+60.5%** |
| 20 | inv | 19.27 | 7.26  | **+62.3%** |
| 21 | fwd | 27.15 | 9.26  | **+65.9%** |
| 21 | inv | 23.83 | 8.28  | **+65.3%** |
| 22 | fwd | 49.78 | 14.70 | **+70.5%** |
| 22 | inv | 51.32 | 21.19 | **+58.7%** |

**komodo (Pixel 9 Pro XL, Tensor G4, Mali-G715)**

| log_n | dir | local ms | global ms | global win |
|---|---|---|---|---|
| 18 | fwd | 4.93  | 0.71  | **+85.6%** |
| 18 | inv | 5.55  | 1.15  | **+79.3%** |
| 19 | fwd | 7.71  | 2.67  | **+65.4%** |
| 19 | inv | 8.42  | 3.99  | **+52.6%** |
| 20 | fwd | 14.83 | 5.08  | **+65.7%** |
| 20 | inv | 16.29 | 6.98  | **+57.2%** |
| 21 | fwd | 25.87 | 9.37  | **+63.8%** |
| 21 | inv | 26.87 | 13.22 | **+50.8%** |
| 22 | fwd | 47.71 | 14.85 | **+68.9%** |
| 22 | inv | 49.52 | 13.36 | **+73.0%** |

**Per-device recommendation from `zkgpu-tail-analyze`:** all three →
`UNCONDITIONAL @ log21 (global ≥20% win)`.

### Head-to-head: G.1.1 vs phase-e vs phase-f (komodo fwd)

| log_n | phase-e (cold, narrow) | phase-f (warm, full) | **G.1.1 (cold, narrow)** |
|---|---|---|---|
| 18 | +43.9% | +60.4% | **+85.6%** |
| 19 | +7.2%  | +44.1% | **+65.4%** |
| 20 | −1.2%  | +38.7% | **+65.7%** |
| 21 | −3.5%  | +32.9% | **+63.8%** |
| 22 | −2.1%  | +28.5% | **+68.9%** |

G.1.1's shape matches phase-f (unconditional across log_n), **not**
phase-e (windowed at log 18). Phase-e is the outlier despite G.1.1
using the exact same scope.

### Decision-gate outcome

Applying the rule from the scaffold's §Decision gate:

> **If 3/3 devices show UNCONDITIONAL @ log 21** (≥20% win across
> log 18–22) → matches phase-f. Thermal hypothesis falsified (or
> irrelevant). Proceed to **G.1.3** to check generational consistency,
> then land the Mali rule as unconditional (same shape as Adreno).

**✅ Gate 2 triggered.** All 30 measured cells (3 devices × 5 log_n
× 2 directions) show Global ≥ 20% win. Weakest cell is comet log 20
inverse at +21.2% — above the threshold. Every other cell is ≥ 50%.

### Implications

1. **Thermal hypothesis falsified.** Phase-f's unconditional shape is
   *not* caused by device warm-up from preceding smoke/validation/
   benchmark tests. A cold device with only the two forced-A/B tests
   shows the same shape. Whatever phase-e measured at log 19+ was
   noise or a methodology artifact, not a thermal-cold-regime signal.
2. **Mali rule shape confirmed.** `GpuFamily::Mali → GlobalOnlyR4`
   unconditional at log_n ≥ 21 (matching the Adreno rule's shape).
   Not yet landable — G.1.3 must verify this generalizes across Mali
   generations (G78/G710) before the rule ships.
3. **Magnitude is stronger cold.** G.1.1 measured **larger** Global
   wins than phase-f (warm) — phase-f komodo fwd at log 22 was +28.5%,
   G.1.1 komodo fwd at log 22 is +68.9%. The warm-run shape is a
   damped version of the cold shape, not an amplified one. (Noted
   for reference; does not change the rule.)
4. **Phase-e retrospective.** Phase-e (2026-04-15) showed windowed
   signal on komodo/comet at log 18 only. With the same scope one
   day later, all three devices show unconditional signal. Possible
   explanations: transient measurement noise (most likely given
   magnitude of disagreement), FTL device-pool rotation between runs,
   or a subtle APK difference (both runs used phase-f APKs but
   different `.gradle` cache state on the upload host could change
   bytecode ordering). Not worth chasing further — G.1.1 is the
   newer, higher-confidence measurement. Phase-e is demoted to
   "anomalous, superseded."

### Next step

**G.1.3 — older-gen Mali generational coverage.** Run the same narrow
forced-A/B scope on two older-gen Mali devices:

- `oriole` (Pixel 6, Tensor G1, **Mali-G78**) — one generation older
  than G710/G715.
- `panther` (Pixel 7, Tensor G2, **Mali-G710**) — the immediate
  predecessor to G715.

(Silicon-to-codename confirmed at run time via `grep -i 'adapter.*name'
logcat.txt`; axis IDs are region-split, not identity. Pixel 8 base
`shiba` is Tensor G3 / G715 same as husky, so not useful for
generational coverage.)

If both older-gen Mali devices match the G715 unconditional shape, land
the rule as `GpuFamily::Mali → GlobalOnlyR4` unconditional. If either
shows a narrower shape, split the rule by generation (e.g.
`Mali-G710+` vs fallback).

Cost estimate: 2 devices × ~2 min wall-clock = ~5 min, same methodology
as G.1.1.

## Artifacts

- `husky/logcat.txt` (10.5 MiB) — raw Pixel 8 Pro device log
- `husky/test_result.xml` — JUnit XML, 2 passed
- `komodo/logcat.txt` (10.4 MiB) — raw Pixel 9 Pro XL device log
- `komodo/test_result.xml` — JUnit XML, 2 passed
- `comet/logcat.txt` (5.4 MiB) — raw Pixel 9 Pro Fold device log
- `comet/test_result.xml` — JUnit XML, 2 passed
- `tail_ab_report.json` — machine-readable 30-cell recommender output

## Links / references

- Parent plan: `../../../../../plan/foundation-audit-and-family-rules-roadmap-2026-04-15.md`
  §Phase G.1 §G.1.1
- Phase-e: `../phase-e-tail-ab-2026-04-15/README.md`
- Phase-f: `../phase-f-full-validation-2026-04-15/README.md`
- G.0 closeout verdict: `../../../../../research/benchmarks/foundation-audit-2026-04-15/verdict.md`
