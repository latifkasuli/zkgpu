# Phase F — 7-device validation of HeuristicAdrenoCollapse (2026-04-15)

Full-matrix regression pass for PR 3 (the new `GpuFamily::Adreno →
GlobalOnlyR4` rule added in `crates/zkgpu-wgpu/src/ntt/planner/tail_policy.rs`).
Runs the whole `ZkgpuHarnessTest` suite — smoke + validation + benchmark
+ crossover + forced-A/B — across the same 7-device family span phase-e
used, so we have apples-to-apples coverage of Adreno, Mali, and Xclipse
after the rule change.

## FTL matrix

- App APK: `app-debug.apk`
- Test APK: `app-debug-androidTest.apk`
- Results dir: `gs://test-lab-zktc5pzjt539i-i05bjnq39twzw/zkgpu-phase-f-full-validation-2026-04-15/`
- Matrix ID: [console](https://console.firebase.google.com/project/connect-4-ai-466116/testlab/histories/bh.428d98a7f5710a25)

Devices actually observed (FTL occasionally substitutes under capacity
pressure — noted below where the axis shipped different silicon than
requested):

| Axis      | Observed GPU            | Family  | Backend | Outcome |
|-----------|-------------------------|---------|---------|---------|
| a56xnaeea | Samsung Xclipse 540     | Xclipse | Vulkan  | Passed  |
| comet     | Mali-G715               | Mali    | Vulkan  | Passed  |
| e1q       | Adreno (TM) 750 *(sub)* | Adreno  | Vulkan  | Passed  |
| e3q       | Adreno (TM) 750         | Adreno  | Vulkan  | Passed  |
| husky     | Mali-G715               | Mali    | Vulkan  | Passed  |
| komodo    | Mali-G715               | Mali    | Vulkan  | Passed  |
| pa3q      | Adreno (TM) 830         | Adreno  | Vulkan  | Passed  |

All 7/7 devices: 10/10 test methods each = **70/70 green**.

## Rule-firing verification

Grepping each device logcat for `reason=HeuristicAdrenoCollapse`:

| Device    | Family  | `HeuristicAdrenoCollapse` firings | `HeuristicDefaultLocal` firings | Stale phase-e rules |
|-----------|---------|-----------------------------------|---------------------------------|---------------------|
| a56xnaeea | Xclipse | 0                                 | 14                              | 0                   |
| comet     | Mali    | 0                                 | 20                              | 0                   |
| e1q       | Adreno  | 10                                | 0                               | 0                   |
| e3q       | Adreno  | 10                                | 0                               | 0                   |
| husky     | Mali    | 0                                 | 20                              | 0                   |
| komodo    | Mali    | 0                                 | 20                              | 0                   |
| pa3q      | Adreno  | 10                                | 0                               | 0                   |

Crisp: the new rule fires on 10/10 crossover cases on every Adreno axis
(3 × 10 = 30 firings) and **never fires** on Mali/Xclipse. No stale
`HeuristicXclipseLargeN` or `HeuristicMaliLargeN` firings — phase-e's
rule-drops are in effect as expected.

## Surprise: pa3q (Adreno 830) also wins under the new rule

Phase-e concluded pa3q was a "flatline ±5%" with neither pathology. The
phase-f forced-A/B pass tells a different story — every pa3q cell is a
≥25% GlobalOnlyR4 win:

| log_n | forward | inverse |
|-------|--------:|--------:|
| 18    | +25.8%  | +56.9%  |
| 19    | +65.8%  | +58.2%  |
| 20    | +63.8%  | +63.8%  |
| 21    | +61.1%  | +56.2%  |
| 22    | +56.5%  | +55.0%  |

Analyzer verdict: **UNCONDITIONAL @ log21 (global ≥20% win)** — same as
the three older Adreno generations. The "pa3q is flatline" claim in
phase-e looks to have been measurement noise from phase-e's run, not a
generational property; the S24 Ultra local follow-up
(`../s24-ultra-local-2026-04-15/`) was the first run where we saw a
clean forced-A/B signal on an Adreno, and phase-f now shows pa3q had
the same signal all along.

Implication: the `GpuFamily::Adreno` unconditional gate is not a
tradeoff with a marginal pa3q cost — it's a win on **every** Adreno
generation we can measure (730/740/750/830). The decision is cleaner
in retrospect than the decision we made from adreno-gen-confirm's
3-generation data alone.

## Mali forced-A/B side-signal

The forced-A/B analyzer also flags Mali-G715 (husky, komodo, comet) and
Xclipse 540 (a56xnaeea) as UNCONDITIONAL @ log21 global-big on this run.
That is a stronger signal than phase-e observed (phase-e saw a log18-19
window on Mali and neutral at log21; phase-f sees wins across the whole
range). Deferring action on this:

* The phase-e close-out explicitly left the Mali small-N add deferred
  pending a 4/4 agreement and an explanation for comet's log19 anomaly.
  Phase-f's new cross-range Mali signal doesn't slot into that decision
  path — it's a separate question.
* Mali-G715 has now been measured on three different Tensor SKUs
  (husky/G3, komodo/G4, comet/G4-fold) all showing the same pattern.
  That's 3/3, which is the "4/4 agreement" caveat satisfied in spirit
  if not in device count.
* But: the forced-A/B tests stress the tail specifically; the
  non-forced `testCrossoverBenchmarkSucceeds` auto-planner numbers on
  Mali don't show the same magnitude of regret. Before adding a Mali
  rule we want to understand why — bench harness timing variance? auto
  planner picking something different from forced-Local?

Leaving this to a separate phase-g investigation. PR 3 is narrowly
scoped to the Adreno rule.

## Artifacts

- `<device>/logcat.txt` — full device logcat
- `tail_ab_report.json` — machine-readable recommender output
