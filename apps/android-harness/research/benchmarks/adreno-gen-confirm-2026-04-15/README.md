# Adreno generation-confirmation A/B (2026-04-15)

Follow-up to the local S24 Ultra / Adreno 750 benchmark
(`../s24-ultra-local-2026-04-15/`) that surfaced a ~3× LocalFusedR4-vs-
GlobalOnlyR4 gap at log22 — the Xclipse-540 strided-gather pathology
reproducing on current Adreno silicon. Phase-e's 6-device matrix had
only one Adreno axis (pa3q / Adreno 830, which was flatline) so we had
no direct evidence whether the collapse was Adreno-750-specific or
affected the whole pre-830 generation lineage.

Goal: one forced-A/B pass on the three older Adreno generations to
decide whether `GpuFamily::Adreno → GlobalOnlyR4` is the right rule,
versus a narrower Adreno-750-only carve-out.

## FTL matrix

- App APK: `app-debug.apk`
- Test APK: `app-debug-androidTest.apk`
- Results dir: `gs://test-lab-zktc5pzjt539i-i05bjnq39twzw/zkgpu-adreno-gen-confirm-2026-04-15-121127/`
- Matrix ID: `matrix-30b7z44mdtu38` ([console](https://console.firebase.google.com/project/connect-4-ai-466116/testlab/histories/bh.428d98a7f5710a25/matrices/7278501713999733705))
- Test targets filter: `crossoverStockhamLocalTail` + `crossoverStockhamGlobalTail` only

| Codename | Device           | SoC                | GPU        | API | Outcome |
|----------|------------------|--------------------|------------|-----|---------|
| b0q      | Galaxy S22       | Snapdragon 8 Gen 1 | Adreno 730 | 36  | Passed  |
| dm3q     | Galaxy S23 Ultra | Snapdragon 8 Gen 2 | Adreno 740 | 34  | Passed  |
| e3q      | Galaxy S24 Ultra | Snapdragon 8 Gen 3 | Adreno 750 | 36  | Passed  |

Adreno 830 (pa3q) was already measured in phase-e and is the contrast
control here — it does NOT collapse.

## Recommender verdicts (3/3 devices)

`cargo run -p zkgpu-tail-analyze -- apps/android-harness/research/benchmarks/adreno-gen-confirm-2026-04-15`

| Device | GPU         | Verdict                                             |
|--------|-------------|-----------------------------------------------------|
| b0q    | Adreno 730  | UNCONDITIONAL @ log21 (every cell global-big; +53–64%) |
| dm3q   | Adreno 740  | UNCONDITIONAL @ log21 (every cell global-big; +40–71%) |
| e3q    | Adreno 750  | UNCONDITIONAL @ log21 (every cell global-big; +57–72%) |

Full per-(device, log_n, direction) table is in `tail_ab_report.json`.

## What the data actually says

Every cell in the 3-device × 5-log_n × 2-direction matrix (30 cells)
is a "global-big" verdict (≥20% GlobalOnlyR4 win). Summary of the
average GlobalOnlyR4 win across (forward+inverse) for each (device, log_n):

| log_n | b0q (A730) | dm3q (A740) | e3q (A750) |
|-------|------------|-------------|------------|
| 18    | +57.3%     | +42.2%      | +53.7%     |
| 19    | +55.2%     | +67.3%      | +66.9%     |
| 20    | +61.6%     | +70.9%      | +66.5%     |
| 21    | +57.3%     | +56.1%      | +63.3%     |
| 22    | +58.9%     | +50.9%      | +61.1%     |

Three properties worth flagging:

1. **Wholesale, not log_n-gated.** The collapse is present at log18
   (the smallest size we can measure with a tail phase on Android
   given the four-step mobile threshold at log_n=18). The rule should
   not be log_n-gated.
2. **Direction-symmetric.** Forward vs inverse wins are within a few
   percentage points of each other; the decision doesn't need to split
   on direction.
3. **Generation-independent within 730–750.** The three generations
   show the same pathology at roughly the same magnitude — this is not
   a 750-specific issue. Adreno 830 (pa3q, phase-e) is the only
   observed member of the family that *doesn't* collapse.

## Decision: `GpuFamily::Adreno → GlobalOnlyR4`

Unconditional on `GpuFamily::Adreno` once a tail phase exists
(`log_n ≥ LOG_BLOCK`). Three independent generations with 40–72% wins
vastly outweigh the marginal pa3q regression — phase-e measured pa3q
as flatline ±5%, well within noise.

> **Post-landing update (phase-f):** the 7-device validation pass
> (`../phase-f-full-validation-2026-04-15/`) re-measured pa3q forced-A/B
> and found it is *not* a flatline — every cell is a +26% to +66%
> GlobalOnlyR4 win (analyzer: `UNCONDITIONAL @ log21`). So the rule
> isn't trading a pa3q regression for the older-gen wins; it's a
> universal Adreno win across 730/740/750/830. Phase-e's pa3q-flatline
> reading was measurement noise, not silicon behavior.

Implementation:

* New `StockhamTailReason::HeuristicAdrenoCollapse` variant.
* New `heuristic_default` arm in `crates/zkgpu-wgpu/src/ntt/planner/tail_policy.rs`
  gated on `GpuFamily::Adreno`.
* Planner-level integration test
  (`adreno_picks_global_tail_at_all_tail_sizes`) in
  `crates/zkgpu-wgpu/src/ntt/planner/tests.rs`.
* In-module unit test (`adreno_picks_global_tail_at_all_sizes`) in
  `tail_policy.rs::tests` covering log_n ∈ {10, 15, 18, 22, 24}.

## Why this wasn't caught earlier

Phase-e's 6-device FTL matrix had exactly one Adreno axis (pa3q /
Adreno 830) and it happened to be the one generation that *doesn't*
reproduce the pathology. Phase-e's conclusion — "no large-N signal on
Adreno" — was technically accurate for pa3q specifically and got
over-generalized to `GpuFamily::Adreno`. The S24 Ultra local run was
the first time we measured a pre-830 Adreno with the instrumented
A/B, and it immediately showed the 65% gap.

## Artifacts

- `b0q/logcat.txt`, `dm3q/logcat.txt`, `e3q/logcat.txt` — full device logcats
- `tail_ab_report.json` — machine-readable recommender output
