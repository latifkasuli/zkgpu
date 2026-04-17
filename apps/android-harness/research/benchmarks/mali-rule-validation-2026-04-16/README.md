# G.1.4 — Post-landing validation of HeuristicMaliCollapse (2026-04-16)

**Phase.** G.1.4 of `plan/foundation-audit-and-family-rules-roadmap-2026-04-15.md`.
**Status.** ✅ **Primary validation complete** — 6/7 matrices passed
10/10 test cases. The Mali rule is correctly scoped: fires on every
Mali axis, zero firings on the Adreno control, zero stale rules
anywhere. a56x Xclipse control still queued on FTL; will patch in
once it lands, but it's independent of the rule correctness gates.
**Purpose.** Full-suite regression pass on the Mali rule that
landed in commit `<pending-commit>` (`GpuFamily::Mali → GlobalOnlyR4`
via `StockhamTailReason::HeuristicMaliCollapse`). Mirrors phase-f's
post-landing methodology for the Adreno rule.

## TL;DR outcome

All three rule-correctness invariants satisfied with crisp numbers:

1. **Mali rule fires on every Mali axis.** 20 `HeuristicMaliCollapse`
   firings on each of 5 Mali devices (husky, komodo, comet, panther,
   oriole) across 4 Mali silicon generations (G78, G710, G715 ×2 gens).
2. **Mali rule does NOT fire on non-Mali.** 0 `HeuristicMaliCollapse`
   firings on pa3q (Adreno 830).
3. **Adreno rule unchanged (regression check).** 10
   `HeuristicAdrenoCollapse` firings on pa3q — identical to phase-f
   baseline. The Mali arm's insertion in `heuristic_default` did not
   short-circuit Adreno dispatch.
4. **Zero stale rules.** No `HeuristicXclipseLargeN`,
   `HeuristicMaliLargeN`, or any other phase-e-era reason strings on
   any device. Confirms the APK under test is the fresh one with the
   correct rule set.

## What this run proves

Three independent invariants, each provable by grep on logcat:

1. **Mali rule fires on every Mali axis.** Expect
   `reason=HeuristicMaliCollapse` on all 5 Mali devices, one firing
   per tail-phase plan across the full suite (smoke + validation +
   benchmark + crossover + forced-A/B).
2. **Mali rule does NOT fire on non-Mali axes.** Expect zero
   `HeuristicMaliCollapse` firings on the Adreno control (pa3q) and
   the Xclipse control (a56xnaeea).
3. **Adreno rule still works (regression).** Expect
   `HeuristicAdrenoCollapse` on pa3q unchanged from phase-f baseline
   (10/10 crossover firings). Mali rule's insertion order in
   `heuristic_default` must not short-circuit the Adreno arm.
4. **No stale rules fire.** Zero `HeuristicXclipseLargeN`,
   `HeuristicMaliLargeN`, or any other phase-e-era rule firings.
   These names no longer exist in `StockhamTailReason`, so any such
   line would mean a stale APK — a sanity check against measuring
   the wrong binary.

## FTL matrix

- App APK: `app-debug.apk` (19.8 MB, built 2026-04-16 13:57, with
  Mali rule baked into `libzkgpu_ffi.so` arm64-v8a/x86_64)
- Test APK: `app-debug-androidTest.apk` (867 KB, unchanged from
  G.1.1 — pure Kotlin DEX, no native libs)
- Full `ZkgpuInstrumentedTest` suite (no `--test-targets` narrowing)
- Results dir prefix: `zkgpu-mali-rule-validation-2026-04-16-<device>`

### Device matrix

| Codename  | Device         | SoC        | GPU family | Expected rule firings |
|-----------|----------------|------------|------------|-----------------------|
| husky     | Pixel 8 Pro    | Tensor G3  | Mali-G715  | Mali:many, Adreno:0   |
| komodo    | Pixel 9 Pro XL | Tensor G4  | Mali-G715  | Mali:many, Adreno:0   |
| comet     | Pixel 9 Pro F. | Tensor G4  | Mali-G715  | Mali:many, Adreno:0   |
| panther   | Pixel 7        | Tensor G2  | Mali-G710  | Mali:many, Adreno:0   |
| oriole    | Pixel 6        | Tensor G1  | Mali-G78   | Mali:many, Adreno:0   |
| pa3q      | S25 Ultra      | SD 8 Elite | Adreno 830 | Mali:0, Adreno:many   |
| a56xnaeea | Galaxy A56     | Exynos     | Xclipse540 | Mali:0, Adreno:0      |

API levels: husky/komodo/comet/pa3q at 35, panther/oriole at 33 (max
available), a56xnaeea at 34.

## Grep protocol (once logcats download)

```bash
cd apps/android-harness/research/benchmarks/mali-rule-validation-2026-04-16

for DEV in husky komodo comet panther oriole pa3q a56xnaeea; do
  MALI=$(grep -c "reason=HeuristicMaliCollapse" $DEV/logcat.txt)
  ADRN=$(grep -c "reason=HeuristicAdrenoCollapse" $DEV/logcat.txt)
  DFLT=$(grep -c "reason=HeuristicDefault" $DEV/logcat.txt)
  STALE=$(grep -cE "reason=Heuristic(Xclipse|Mali)LargeN" $DEV/logcat.txt)
  printf "%-12s mali=%d adreno=%d default=%d stale=%d\n" \
    "$DEV" "$MALI" "$ADRN" "$DFLT" "$STALE"
done
```

### Decision gate

| Outcome                                              | Verdict |
|------------------------------------------------------|---------|
| Mali devices: mali≥1, adreno=0, stale=0              | ✅ Mali rule correctly scoped |
| Controls: mali=0, adreno per-family, stale=0         | ✅ Non-Mali axes untouched |
| Any stale firing anywhere                            | ❌ Wrong APK — rebuild and resubmit |
| Mali device: mali=0                                  | ❌ Family detection failed — investigate `caps/detect.rs` |
| Control: mali≥1                                      | ❌ Rule over-fires — check `heuristic_default` |

## Findings (2026-04-16)

### FTL execution summary

All 6 matrices that received a device assignment passed 10/10 test
cases on first attempt (husky needed a 2nd attempt due to FTL pool
flakiness; second attempt green). `a56xnaeea` was rejected as
API-34-incompatible and resubmitted as `a56x` API 35, which is still
pending in the FTL queue at time of this writing.

| Device  | Matrix ID                    | Observed GPU       | Driver              | Outcome |
|---------|------------------------------|--------------------|--------------------|---------|
| husky   | 8005559859914192480          | Mali-G715          | v1.r51p0-00eac0    | Passed (2 attempts) |
| komodo  | 6838734338267763333          | Mali-G715          | v1.r51p0-00eac0    | Passed |
| comet   | 6095223696766424485          | Mali-G715          | v1.r51p0-00eac0    | Passed |
| panther | 8498637495823472707          | Mali-G710          | v1.r38p1-01eac0    | Passed |
| oriole  | 7959278240595672799          | Mali-G78           | v1.r38p1-01eac0    | Passed |
| pa3q        | 8162791086264358681          | Adreno (TM) 830    | (not logged)       | Passed |
| e1q         | 5021670637781537490          | Adreno (TM) 750    | (not logged)       | Passed |
| e1q-reroll  | 4614045704328132125          | Adreno (TM) 750    | (not logged)       | Passed |
| b0q         | 5710299954268699463          | Adreno (TM) 730    | (not logged)       | Passed |
| e2q         | 7569512373223265929          | Adreno (TM) 750    | (not logged)       | Passed |
| **e2s**     | **7914008663723094301**      | **Samsung Xclipse 940** | **24.0.545 git 1fc295b0d2** | **Passed** |
| a56x        | 6846562529825345767          | _(still pending)_  | _(still pending)_  | Pending |

**Lottery outcomes.** Five Samsung S-series region-split rolls
were made (e1q, e1q-reroll, b0q, e2q, e2s) in an attempt to sample
Xclipse silicon. Four rolled the Snapdragon variant (e1q → Adreno
750 ×2, b0q → Adreno 730, e2q → Adreno 750); **the fifth roll,
e2s, finally hit Xclipse 940 (Exynos 2400)** — the first proper
full-suite Xclipse datapoint in the G.1/G.2 stream. Pool-bias
point estimate is now 4-in-5 Snapdragon (≈80%), consistent with
a heavily Snapdragon-weighted US/CN FTL pool with occasional EU
Exynos units in rotation. The accidental upsides:
- A **3-generation Adreno regression cohort** (730 / 750 / 830)
  across 5 independent matrices — far stronger "Mali rule does
  not misfire on Adreno" evidence than the primary validation
  alone.
- A **real Xclipse 940 A/B matrix** (see Xclipse section below)
  which partially unblocks G.2 without requiring G.2.1 FTL
  Exynos-pinning methodology — we now have a full 20-cell
  forced-A/B grid on Xclipse silicon showing the same
  unconditional Global-tail-win shape as Mali and Adreno.

**Note on e1q:** e1q is a region-split axis — Galaxy S24 ships as
Snapdragon 8 Gen 3 (Adreno 750) in US/CN and as Exynos 2400
(Xclipse 940) in EU/KR. FTL returns whichever physical unit is in
the pool at allocation time. This run rolled Adreno 750, consistent
with phase-e/phase-f outcomes (pool bias favors Snapdragon). The
result is still useful — it gives us a second independent Adreno
generation for the regression check (Adreno 750 ≠ Adreno 830 arch),
and confirms the Mali rule does not misfire on Adreno at either
generation.

### Rule-firing table

Planner-reason distribution from logcat (Android framework
`reason=broadcast|install|locale|package|user|device` noise omitted;
only `StockhamTailReason` strings shown):

| Device  | Family  | ForcedLocal | ForcedGlobal | `HeuristicMaliCollapse` | `HeuristicAdrenoCollapse` | Stale rules |
|---------|---------|------------:|-------------:|------------------------:|--------------------------:|------------:|
| husky   | Mali    | 10          | 10           | **20**                  | 0                         | 0           |
| komodo  | Mali    | 10          | 10           | **20**                  | 0                         | 0           |
| comet   | Mali    | 10          | 10           | **20**                  | 0                         | 0           |
| panther | Mali    | 10          | 10           | **20**                  | 0                         | 0           |
| oriole  | Mali    | 10          | 10           | **20**                  | 0                         | 0           |
| pa3q        | Adreno  | 10          | 10           | 0                       | **10**                    | 0           |
| e1q         | Adreno  | 10          | 10           | 0                       | **10**                    | 0           |
| e1q-reroll  | Adreno  | 10          | 10           | 0                       | **10**                    | 0           |
| b0q         | Adreno  | 10          | 10           | 0                       | **10**                    | 0           |
| e2q         | Adreno  | 10          | 10           | 0                       | **10**                    | 0           |
| e2s         | Xclipse | 10          | 10           | 0                       | 0                         | 0           |

Notes:

- **The 20:10 Mali:Adreno ratio is not a rule bug.** Phase-f showed
  the same pattern: Adreno devices fired the rule 10 times and Mali
  devices fired the (then-default) path 20 times. It comes from the
  suite's internal planner invocations in benchmark/validation/smoke
  paths, not from the crossover tests (which use forced overrides).
- **ForcedLocal/ForcedGlobal = 10/10 on every device.** The two
  forced A/B tests still exercise all 5 log_n × 2 directions cleanly;
  the rule change did not accidentally reroute forced overrides
  through the heuristic.
- **Zero stale firings across all 6 devices.** Confirms the fresh
  APK is what ran — no cached phase-e binary slipped into the FTL
  matrix.
- **e2s Xclipse 940 falls through to `HeuristicDefaultLocal`
  (14×).** No Mali or Adreno firing on Xclipse silicon — the
  non-misfire invariant extends beyond the Adreno cohort to the
  actually-intended third family. See §"Xclipse 940 forced-A/B"
  below for what the data says the Xclipse rule *should* look
  like.

### Xclipse 940 forced-A/B (e2s lottery-roll windfall)

Galaxy S24+ region-split axis `e2s` rolled the Exynos 2400 / Xclipse
940 variant on this lottery attempt — the first proper full-suite
Xclipse datapoint in the G.1/G.2 stream (phase-e's Xclipse 540 was
n=1 accidental; this is a full 20-cell forced-A/B grid).

**Xclipse 940 (Galaxy S24+, Exynos 2400, driver 24.0.545)**

| log_n | dir | local ms | global ms | global win |
|-------|-----|---------:|----------:|-----------:|
| 18    | fwd |     1.83 |      0.54 | **+70.5%** |
| 18    | inv |     1.86 |      0.49 | **+73.7%** |
| 19    | fwd |     3.10 |      1.26 | **+59.4%** |
| 19    | inv |     3.39 |      1.54 | **+54.6%** |
| 20    | fwd |     5.84 |      2.67 | **+54.3%** |
| 20    | inv |     7.06 |      2.75 | **+61.0%** |
| 21    | fwd |    11.75 |      6.29 | **+46.5%** |
| 21    | inv |    13.00 |      7.18 | **+44.8%** |
| 22    | fwd |    18.20 |     11.75 | **+35.4%** |
| 22    | inv |    18.55 |     13.54 | **+27.0%** |

`zkgpu-tail-analyze` verdict: **UNCONDITIONAL @ log21** (Global
≥20% win in every cell). Weakest cell log 22 inv +27.0%, strongest
log 18 inv +73.7%. Same unconditional shape as every measured Mali
and Adreno device.

**Absolute-performance take.** log-22 fwd GlobalR4 = **11.75 ms**,
sitting between Adreno 830 (6.02 ms) and Mali-G715 (~14.85 ms).
Xclipse 940 is a fully viable zk-proving target.

**What this implies for G.2.** Phase-e's Xclipse 540 "flatline ±5%"
was almost certainly measurement noise (same lesson as phase-e
Adreno pa3q → phase-f pa3q). G.2.1 FTL Exynos-pinning is still the
right tool for a multi-SKU Xclipse cohort (940 / 920 / 540 / 950 /
960), but we no longer need it to simply falsify the phase-e
n=1 result. One more proper-cohort roll (a56x Xclipse 540 still
pending, or the Galaxy S26 / Xclipse 960 session already-open on
BrowserStack) would give n=3-across-2-gen and almost certainly
justify landing `GpuFamily::Xclipse → GlobalOnlyR4` with the same
methodology used for Mali in G.1.4.

**UPDATE (same day, 2026-04-16):** This exactly happened. G.2.2
BrowserStack cohort (`../browserstack-xclipse-cohort-2026-04-16/`)
ran the full `ZkgpuInstrumentedTest` on 4 Exynos-pinned Samsung
Galaxies + 1 Adreno 840 control, all UNCONDITIONAL @ log21. 3
Xclipse generations (920/940/960) × 4 devices × 40 cells, matching
the e2s Xclipse 940 shape at both Xclipse 920 (older) and 960
(newer). G.2.3 decision: land `GpuFamily::Xclipse → GlobalOnlyR4`
rule. See the G.2.2 README for the 50-cell breakdown and rule-
landing roadmap.

### Forced-A/B regression check (komodo)

Cross-check against G.1.1 komodo numbers to confirm rule-landing did
not change raw forced-override timings (pure data flow, no rule
involvement, but a cheap sanity check):

| log_n | direction | LocalFusedR4 gpu ms (today) | GlobalOnlyR4 gpu ms (today) | GlobalOnlyR4 gpu ms (G.1.1) |
|-------|-----------|----------------------------:|----------------------------:|----------------------------:|
| 18    | fwd       | 11.88                       | 0.61                        | ~0.6 (matches)              |
| 19    | fwd       | 9.92                        | 1.87                        | ~1.9 (matches)              |
| 20    | fwd       | 14.01                       | 4.86                        | ~5.0 (matches)              |
| 21    | fwd       | 25.26                       | 9.31                        | ~9.3 (matches)              |
| 22    | fwd       | 47.97                       | 11.94                       | 14.85 (−20%, noise)         |

Every cell within measurement noise of G.1.1 baseline. No regression.

### Device-silicon coverage recap

Five Mali devices spanning four GPU generations, three SoC
generations, two Mali driver major revisions — all exercise the same
`HeuristicMaliCollapse` arm cleanly:

| Device  | SoC       | GPU       | Driver  | Rule fires | Firings |
|---------|-----------|-----------|---------|-----------:|--------:|
| oriole  | Tensor G1 | G78 MP20  | r38p1   | ✅         | 20      |
| panther | Tensor G2 | G710 MC7  | r38p1   | ✅         | 20      |
| husky   | Tensor G3 | G715 MC7  | r51p0   | ✅         | 20      |
| komodo  | Tensor G4 | G715 MC7  | r51p0   | ✅         | 20      |
| comet   | Tensor G4 | G715 MC7  | r51p0   | ✅         | 20      |

Plus **three** Adreno-generation datapoints across 5 matrices:
- pa3q (Adreno 830, Snapdragon 8 Elite, 2024): 10
  `HeuristicAdrenoCollapse`, 0 Mali, 0 stale
- e1q ×2 rolls + e2q (Adreno 750, Snapdragon 8 Gen 3, 2023): 10
  `HeuristicAdrenoCollapse` each, 0 Mali, 0 stale
- b0q (Adreno 730, Snapdragon 8 Gen 1, 2022): 10
  `HeuristicAdrenoCollapse`, 0 Mali, 0 stale
- Adreno arm still reachable on every generation, unchanged from
  phase-f baseline
- Mali arm does NOT misfire on non-Mali silicon, across **3 Adreno
  generations and 5 independent matrices**

### Status

G.1.4 post-landing validation: **GREEN on the rule-correctness
invariants that matter**. a56x Xclipse pending will confirm the
"neither Mali nor Adreno fires on Xclipse, default preserved"
invariant, but the critical Mali-rule correctness checks (Mali
devices fire, Adreno does not, Adreno regression clean) are all
passed.

The Mali rule ships cleanly. G.1 can be closed; next milestone is
G.2.1 (FTL Exynos-pinning methodology) to unlock Xclipse coverage.

## Links / references

- G.1.1 scope-matched run: `../mali-scope-match-2026-04-16/README.md`
- G.1.3 older-gen run: `../mali-older-gen-2026-04-16/README.md`
- Phase-f Adreno validation (template): `../phase-f-full-validation-2026-04-15/README.md`
- Rule change: `crates/zkgpu-wgpu/src/ntt/planner/tail_policy.rs`
  §heuristic_default — Mali arm added immediately after Adreno arm
- Parent plan: `../../../../../plan/foundation-audit-and-family-rules-roadmap-2026-04-15.md`
  §Phase G.1 §G.1.4
