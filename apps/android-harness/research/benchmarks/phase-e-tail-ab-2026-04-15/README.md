# Phase E — Stockham tail A/B (2026-04-15)

> **2026-04-15 follow-up:** phase-e's Adreno conclusion ("flatline ±5%")
> was pa3q-specific, not family-wide. The S24 Ultra local run
> (`../s24-ultra-local-2026-04-15/`) and the follow-up FTL generation
> confirmation (`../adreno-gen-confirm-2026-04-15/`) showed Adreno
> 730/740/750 all collapse at +40-72%. A new `HeuristicAdrenoCollapse`
> rule was added in PR 3; Adreno 830 (pa3q) remains the outlier.

First forced-tail A/B benchmark using the `crossoverStockhamLocalTail` /
`crossoverStockhamGlobalTail` instrumented tests landed in PR 1 of the
Stockham tail-strategy refactor (see `commit 6a09be7`).

Goal: drive the `log_n` thresholds in
`crates/zkgpu-wgpu/src/ntt/planner/tail_policy.rs::heuristic_default` from
measured field data, replacing the Xclipse-540-only scaling table inherited
from `research/stockham-local-fused-rewrite.md`.

## FTL matrix

- App APK: `app-debug.apk`
- Test APK: `app-debug-androidTest.apk`
- Results dir: `gs://test-lab-zktc5pzjt539i-i05bjnq39twzw/zkgpu-tail-ab-2026-04-15-093826/`
- Matrix ID (5-device pass): `matrix-22zykxig0kbwr` ([console](https://console.firebase.google.com/project/connect-4-ai-466116/testlab/histories/bh.428d98a7f5710a25/matrices/8870929038526537488))
- Matrix ID (pa3q swap pass): `matrix-4qnbmjxk4qaqa` ([console](https://console.firebase.google.com/project/connect-4-ai-466116/testlab/histories/bh.428d98a7f5710a25/matrices/7879806556957212949))

| Codename   | Device           | SoC                | GPU         | API | Outcome |
|------------|------------------|--------------------|-------------|-----|---------|
| e1q        | Galaxy S24       | Exynos 2400        | Xclipse 940 | 36  | Passed  |
| pa3q       | Galaxy S25 Ultra | Snapdragon 8 Elite | Adreno 830  | 35  | Passed  |
| komodo     | Pixel 9 Pro XL   | Tensor G4          | Mali-G715   | 35  | Passed  |
| husky      | Pixel 8 Pro      | Tensor G3          | Mali-G715   | 35  | Passed  |
| a56xnaeea  | Galaxy A56       | Exynos 1580        | Mali-G720   | 36  | Passed  |
| comet      | Pixel 9 Pro Fold | Tensor G4          | Mali-G715   | 35  | Passed  |

The originally-requested `pa3qxxx-36` axis was swapped to `pa3q-35` after
30 min of FTL queue contention (`pa3qxxx-36` is `DEVICE_CAPACITY_LOW` while
`pa3q-35` is `DEVICE_CAPACITY_HIGH`). Same Snapdragon 8 Elite / Adreno 830
silicon; API level is irrelevant for our wgpu/Vulkan path.

## Tests run

For each device, two forced-A/B instrumented tests:

- `crossoverStockhamLocalTail`  → `StockhamTailOverride::Local`,
  emits `CROSSOVER_STOCKHAM_LOCAL_TAIL` log lines.
- `crossoverStockhamGlobalTail` → `StockhamTailOverride::Global`,
  emits `CROSSOVER_STOCKHAM_GLOBAL_TAIL` log lines.

Each test sweeps `log_n ∈ {18, 19, 20, 21, 22} × {Forward, Inverse}` =
10 cases, 5 iterations + 2 warmups, GPU timestamps on, **forced Stockham
family** so the four-step planner doesn't influence the comparison.

## Recommender verdicts (6/6 devices)

`cargo run -p zkgpu-tail-analyze -- apps/android-harness/research/benchmarks/phase-e-tail-ab-2026-04-15`

### Original run (monotone-only recommender, PR 2 scaffolding)

The first pass through the analyzer — before the `WindowedFlip` variant
landed in PR 2 close-out — returned `NO-CHANGE` on 5/6 devices because
its candidate window was fixed to `[21, 22]` and the actual Mali signal
sits at `log_n ∈ [18, 19]`:

| Device     | Verdict (monotone-only, candidates [21, 22]) |
|------------|----------------------------------------------|
| a56xnaeea  | NO-CHANGE                                    |
| comet      | NO-CHANGE                                    |
| e1q        | NO-CHANGE                                    |
| husky      | PER-DEVICE @ log21 (8.6% avg)                |
| komodo     | NO-CHANGE                                    |
| pa3q       | NO-CHANGE                                    |

That motivated extending the recommender with a `WindowedFlip` verdict
(contiguous ≥20% run that doesn't extend to max `log_n`).

### Post-`WindowedFlip` run (current recommender)

| Device     | Verdict                                              |
|------------|------------------------------------------------------|
| a56xnaeea  | WINDOWED-FLIP @ log18..=log19 (47.8% avg win)        |
| comet      | WINDOWED-FLIP @ log18 (28.3% avg win)                |
| e1q        | NO-CHANGE (Xclipse 940: no small-N or large-N flip)  |
| husky      | PER-DEVICE @ log21 (8.6% avg win)                    |
| komodo     | WINDOWED-FLIP @ log18 (43.9% avg win)                |
| pa3q       | NO-CHANGE (Adreno 830: flatline ±5%)                 |

Full per-(device, log_n, direction) table is in `tail_ab_report.json`.

Three of four Mali devices (a56xnaeea, komodo, comet) surface the log18
(and on a56xnaeea log18-19) windowed Global win; husky shows a different
monotone shape with a narrow signal at log21. Adreno and Xclipse 940 are
noise-band flat. This confirms the small-N Mali phenomenon is real and
family-gated but warrants the "defer until 4/4 agree" conservatism of
Path A (see below).

## What the data actually says

Reading the raw timings (not just the recommender's verdict at
`log_n ∈ [21, 22]`) reveals an **inversion** of the research doc's
hypothesis. The pattern across the four Mali devices is:

| log_n | typical Mali global-vs-local win (fwd, inv averaged across a56xnaeea/comet/husky/komodo) |
|-------|------------------------------------------------------------------------------------------|
| 18    | **+30 to +50%** (GlobalOnlyR4 wins big)                                                  |
| 19    | +5 to +25% (GlobalOnlyR4 narrow win, mixed)                                              |
| 20    | -10 to +0% (LocalFusedR4 narrow win)                                                     |
| 21    | -10 to +5% (mixed; LocalFusedR4 leans winner)                                            |
| 22    | -3 to +1% (neutral; both paths bandwidth-bound)                                          |

Implications for the current `heuristic_default`:

1. **Mali @ log22 → GlobalOnlyR4** is currently in the heuristic but the
   data says it's neutral. No measurable benefit; not a regression either.
2. **Mali @ log18-19 → LocalFusedR4** is currently in the heuristic but
   the data says GlobalOnlyR4 wins by 30–50% on this fleet. **This is the
   real opportunity surfaced by phase-e** — the opposite direction from
   the Xclipse-540 scaling-table the research doc was built around.
3. **Xclipse 940 (e1q) does not show the Xclipse-540 collapse**. The
   current `HeuristicXclipseLargeN` rule (flip at log20) is at best
   neutral, at worst a small regression on this newer silicon. The
   collapse measurement in `research/stockham-local-fused-rewrite.md` was
   on Exynos 2200 / Xclipse 540; Exynos 2400 / Xclipse 940 looks to have
   fixed it.
4. **Adreno 830 (pa3q) is a flatline** — every (log_n, direction) pair is
   within ±5% noise. The Snapdragon 8 Elite has neither pathology. This is
   the decisive datapoint: **the small-N global win is Mali-specific**, not
   a backend-wide phenomenon. The rule should be `GpuFamily::Mali`-gated,
   not `Backend::Vulkan`-gated.

The likely mechanism for the small-N global win on Mali: at log18-19, the
LocalFusedR4 path's single-workgroup tail dispatch has launch +
shared-memory-setup overhead that dominates the actual work. The
GlobalOnlyR4 path keeps the GPU saturated with R4 workgroups all the way
through. At log20+, the strided gather in subsequent R4 stages starts
hitting L2 misses, so the savings cancel. Adreno's larger LDS / scheduling
hides the launch cost; Xclipse 940's newer arch fixes the gather collapse.

## Recommended planner change

Two clean wins (drops) and one new rule (add):

| Change                                                     | Justification                       |
|------------------------------------------------------------|-------------------------------------|
| **Drop** `Mali @ log_n ≥ 22 → GlobalOnlyR4`                | Phase-e: neutral, no measured benefit. |
| **Drop** `Xclipse @ log_n ≥ 20 → GlobalOnlyR4`             | Phase-e: e1q (Xclipse 940) does not reproduce Xclipse 540 collapse. |
| **Add**  `Mali @ log_n ∈ [18, 19] → GlobalOnlyR4`          | Phase-e: 30–50% wins on a56xnaeea / komodo / husky; comet shows mixed signal but inverse direction agrees. |

Conservative variant: ship the two drops, defer the add until comet's
forward-log19 -11% anomaly is understood (foldable thermal? G4 driver
quirk?). The drops are unambiguous; the add is a 3-of-4 majority on Mali.

## PR 2 close-out status

1. ✅ **`zkgpu-tail-analyze::recommend` extended with `WindowedFlip`** —
   scans all `log_n` buckets for contiguous runs ≥ `UNCONDITIONAL_WIN`
   (20%) that don't extend to the largest measured size. Surfaces the
   Mali small-N signal the old monotone recommender missed.
2. ✅ **`tail_policy::heuristic_default` updated** — the two large-N
   rules falsified by phase-e (`Xclipse @ log_n ≥ 20`, `Mali @ log_n ≥
   22`) are dropped along with their `StockhamTailReason` variants.
   Conservative variant per §"Recommended planner change" above: the
   Mali small-N add is **deferred** until comet's log19 mixed signal is
   explained (foldable thermal profile? G4 driver quirk?) — 3-of-4
   majority isn't a strong enough basis to add a new rule.
3. ✅ Doc comments in `tail_policy.rs` cross-link this benchmark dir.

## Deferred follow-up

* Repeat the small-N A/B on comet under forced-performance mode to
  separate thermal effects from driver effects.
* If 4/4 Mali devices eventually agree, add `Mali @ log_n ∈ [18, 19] →
  GlobalOnlyR4` with `StockhamTailReason::HeuristicMaliSmallN`.
* Re-measure husky's `log21` per-device signal on a second device run
  to check it's not a G3-specific artifact.

## Artifacts

- `<device>/logcat.txt` — full device logcat (~4 MB each)
- `<device>/test_result.xml` — JUnit XML (2/2 tests passing on each device)
- `tail_ab_report.json` — machine-readable recommender output
