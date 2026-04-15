# S24 Ultra local benchmark (2026-04-15)

First ad-hoc benchmark on a locally-connected Samsung Galaxy S24 Ultra
(codename `e3q`, Snapdragon 8 Gen 3, Adreno 750). Motivated by the
post-phase-e question "did we leave any big wins on the table?" — the
phase-e 6-device FTL matrix had no Adreno pre-830 silicon, so this was
the first chance to measure an Adreno 730/740/750 with the instrumented
forced-A/B tests.

## Device

* Galaxy S24 Ultra (SM-S928U1)
* Snapdragon 8 Gen 3
* Adreno 750
* Android 15 (API 36)
* Connected over USB; ADB instrumentation run (`adb shell am instrument ...`)

## Tests run

The same forced-A/B pair used in phase-e:

- `crossoverStockhamLocalTail`  → `StockhamTailOverride::Local`
- `crossoverStockhamGlobalTail` → `StockhamTailOverride::Global`

Each test sweeps `log_n ∈ {18, 19, 20, 21, 22} × {Forward, Inverse}` =
10 cases, 5 iterations + 2 warmups, GPU timestamps on, forced Stockham
family.

## Headline finding

Every (log_n, direction) cell is a "global-big" verdict (≥20%
GlobalOnlyR4 win over LocalFusedR4); the recommender returns
UNCONDITIONAL @ log21.

| log_n | forward local_ms | forward global_ms | forward win | inverse win |
|-------|-----------------:|------------------:|------------:|------------:|
| 18    |            0.46  |             0.27  |      +41.3% |      +64.8% |
| 19    |            2.28  |             0.64  |      +71.9% |      +64.5% |
| 20    |            5.28  |             1.76  |      +66.7% |      +51.4% |
| 21    |           11.57  |             4.29  |      +62.9% |      +57.3% |
| 22    |           24.66  |             8.48  |      +65.6% |      +57.6% |

At log22 forward, the local fused tail kernel is ~2.9× slower than
extending the global R4 chain. Inspection of the per-stage timing log
lines shows the single "local fused" dispatch takes **19.711ms** — ~28×
the cost of one global R4 stage at this size (~0.7ms). This is the
Xclipse-540 strided-gather pathology reappearing on Adreno silicon.

The production default picks `tail=LocalFusedR4 reason=HeuristicDefaultLocal`
on this device today — i.e. the wrong path. Fixing this was the
motivation for the Adreno generation-confirmation A/B
(`../adreno-gen-confirm-2026-04-15/`) and the ensuing
`HeuristicAdrenoCollapse` rule.

## Why this contradicts the phase-e conclusion

Phase-e's only Adreno axis was pa3q / Adreno 830, which measured as a
flatline ±5%. We over-generalized that flatline to `GpuFamily::Adreno`
as a whole. The S24 Ultra run here, plus the follow-up FTL confirmation
on b0q (Adreno 730) and dm3q (Adreno 740), shows the flatline was
Adreno-830-specific.

## Artifacts

- `e3q/logcat.txt` — full ADB logcat captured during the instrumented run
- `tail_ab_report.json` — machine-readable recommender output
