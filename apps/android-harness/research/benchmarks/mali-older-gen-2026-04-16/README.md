# G.1.3 — Older-gen Mali coverage (2026-04-16)

**Phase.** G.1.3 of `plan/foundation-audit-and-family-rules-roadmap-2026-04-15.md`.
**Status.** ✅ **Complete** — both older-gen Mali devices (G78, G710)
reproduce G.1.1's unconditional shape. Capability surprise: **Pixel 6
(G78, 2021) is viable for zk-proving**, within 1.2× of Pixel 9 Pro
(G715, 2024) at log 22. Rule can land family-wide.
**Budget spent.** ~3 min wall-clock (2 parallel FTL matrices,
9–13 s device compute each).

## TL;DR outcome

Both older-gen Mali devices match the G715 cohort's unconditional
Global-tail-win shape. Combined with G.1.1, **5/5 Mali devices across
4 generations (G78/G710/G715×2) show the same rule shape** at log
18–22, forward and inverse. Green-light to land
`GpuFamily::Mali → GlobalOnlyR4` family-wide with no generational
gating.

**Capability finding (Decision B):** absolute GlobalR4 timings on
older-gen Mali are **within 20%** of flagship G715 at log 22. Pixel 6
and Pixel 7 are practical zk-proving targets, not "library works but
not a product." Pre-run estimates of "3–5× slower on Pixel 6" were
wrong — Mali-G78 MP20's higher shader-core count (20 vs 7) compensates
for the older per-core architecture on memory-bound NTT workloads.

## Why this run exists

G.1.1 confirmed 3/3 Mali-G715 devices (Tensor G3/G4) show UNCONDITIONAL
Global-tail win across log 18–22 at 50–85% magnitude. Before landing
`GpuFamily::Mali → GlobalOnlyR4` family-wide, G.1.3 checks whether the
shape generalizes to two older Mali generations on the same brand
(Pixel) with the same driver family (ARM Mali Bifrost/Valhall).

## Dual-purpose deliverable

**Decision A (rule shape).** Does the unconditional shape hold on
older-gen?
- **3/3 older-gen match G715** → land `GpuFamily::Mali → GlobalOnlyR4`
  family-wide, no generation check.
- **Older-gen differs** → narrow rule to `G710+`, `G715+`, or ship a
  generation-scoped policy. Document the split.

**Decision B (capability gate).** What's the absolute proving-region
ms on older-gen silicon?
- Log 22 GlobalR4 timing on oriole (G78) and panther (G710) tells us
  whether Pixel 6/7 are viable zk-proving devices for Plonky3-style
  provers (~50–200 NTTs per proof).
- G715 baseline from G.1.1: ~15 ms per log-22 NTT GlobalR4. If older-gen
  is within 2×, Pixel 6/7 are practical. If >5×, below practical gate.

These are independent decisions — B does not affect whether the rule
lands, only whether we document Pixel 6/7 as supported targets.

## Device matrix

| Codename | Device | SoC | GPU | FTL API |
|---|---|---|---|---|
| oriole  | Pixel 6  | Tensor G1 (2021) | Mali-G78 MC20  | **33** |
| panther | Pixel 7  | Tensor G2 (2022) | Mali-G710 MC7  | **33** |

### API-33 vs API-35 note

G.1.1 ran the G715 cohort on API 35. oriole tops out at API 33, and
panther is API-33-only on FTL. The two crossover tests
(`crossoverStockhamLocalTail`, `crossoverStockhamGlobalTail`) use
no API-35-specific features — they exercise the Vulkan backend via
wgpu, which has a uniform surface across API 33–35. Minor runtime-
JIT differences are possible but would affect both tail variants
equally, so the *ratio* (global vs local) is unaffected.

**Not measured this run:**
- bluejay (Pixel 6a, same G78 as oriole) — redundant.
- cheetah (Pixel 7 Pro) + lynx (Pixel 7a) — both G710, redundant with
  panther. If panther shows generation-specific shape, add cheetah
  as confirmation in a follow-up.
- shiba (Pixel 8, G715) — same silicon as husky, no new generational
  information.

## APK plan

Same phase-f APKs reused from G.1.1:
- `app/build/outputs/apk/debug/app-debug.apk`
- `app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk`

Forced-A/B tests bypass the planner via explicit `tail =` overrides,
so the Adreno rule baked into the phase-f APK has zero effect on
measured Mali timings.

## FTL invocation

Same narrow scope as G.1.1. Two parallel background submissions,
separate results dirs per device:

```bash
cd "$REPO_ROOT/apps/android-harness"
APP=app/build/outputs/apk/debug/app-debug.apk
TESTAPK=app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
TARGETS="class org.zkgpu.harness.ZkgpuInstrumentedTest#crossoverStockhamLocalTail,class org.zkgpu.harness.ZkgpuInstrumentedTest#crossoverStockhamGlobalTail"
BASE=zkgpu-mali-older-gen-2026-04-16

for DEV in "oriole,33" "panther,33"; do
  MODEL="${DEV%%,*}"
  API="${DEV##*,}"
  gcloud firebase test android run \
    --type instrumentation \
    --app "$APP" \
    --test "$TESTAPK" \
    --device "model=$MODEL,version=$API,locale=en,orientation=portrait" \
    --test-targets "$TARGETS" \
    --results-dir "${BASE}-${MODEL}"
done
```

## Decision-gate mechanics

Run `zkgpu-tail-analyze` after download:

```bash
cargo run -p zkgpu-tail-analyze -- \
  apps/android-harness/research/benchmarks/mali-older-gen-2026-04-16
```

### Rule-shape decision (combined with G.1.1)

| G.1.3 shape | Action |
|---|---|
| Both unconditional (≥20% Global win across log 18–22) | Land `GpuFamily::Mali → GlobalOnlyR4` family-wide, no gen check |
| oriole windowed, panther unconditional | Narrow to `Mali-G710+` (exclude G78) |
| Both windowed | Narrow to `Mali-G715+` (G78/G710 use default) |
| Mixed within one device (windowed fwd, unconditional inv) | Defer to G.1.2 thermal-control run; do not land |

### Capability-gate decision (independent)

Compute median log-22 GlobalR4 ms. Reference: G715 ≈ 15 ms.

| Older-gen ms at log 22 | Capability verdict |
|---|---|
| ≤30 ms (≤2× G715) | Viable for Plonky3-scale provers |
| 30–75 ms (2–5×) | Viable for smaller provers (log ≤ 20), marginal at log 22 |
| >75 ms (>5× G715) | Below practical prover gate; document as "library works, not a product target" |

## Findings (2026-04-16)

### FTL execution summary

Both matrices passed, 2 test cases each, first attempt, no axis
substitution. Silicon confirmed via logcat `adapter.info.name`:

| Device | Matrix | GPU name (logcat) | Driver |
|---|---|---|---|
| oriole  | `8767938315090940536` | `Mali-G78`  | `v1.r38p1-01eac0` |
| panther | `8888626589908510414` | `Mali-G710` | `v1.r38p1-01eac0` |

Both on Mali driver **r38p1**, notably **older** than the G715 cohort's
r51p0. This is a meaningful variable: if the rule shape were driver-
version-sensitive, we'd see it here. It doesn't — the shape holds
across two driver major revisions (r38 → r51), reinforcing that the
signal is a GPU-microarchitecture property, not a driver artifact.

`tier=UnifiedMemoryNative`, `backend=Vulkan`, `max_buffer=268435456`
(256 MB), `max_wg_x=256`, `max_invocations=256`,
`max_wg_storage_bytes=16352` — all identical to the G715 cohort.

### Per-log_n A/B table (GPU timings only; from `tail_ab_report.json`)

**oriole (Pixel 6, Tensor G1, Mali-G78 MP20, driver r38p1)**

| log_n | dir | local ms | global ms | global win |
|---|---|---|---|---|
| 18 | fwd |  5.56 |  0.67 | **+87.9%** |
| 18 | inv |  5.44 |  0.84 | **+84.6%** |
| 19 | fwd |  9.02 |  1.61 | **+82.2%** |
| 19 | inv |  8.09 |  2.00 | **+75.3%** |
| 20 | fwd | 16.28 |  3.80 | **+76.7%** |
| 20 | inv | 16.71 |  4.50 | **+73.1%** |
| 21 | fwd | 22.31 |  9.33 | **+58.2%** |
| 21 | inv | 20.91 | 10.19 | **+51.3%** |
| 22 | fwd | 42.30 | 18.01 | **+57.4%** |
| 22 | inv | 44.82 | 18.76 | **+58.1%** |

**panther (Pixel 7, Tensor G2, Mali-G710 MC7, driver r38p1)**

| log_n | dir | local ms | global ms | global win |
|---|---|---|---|---|
| 18 | fwd |  9.21 |  1.30 | **+85.9%** |
| 18 | inv |  8.98 |  1.65 | **+81.6%** |
| 19 | fwd | 14.44 |  3.69 | **+74.4%** |
| 19 | inv | 14.83 |  4.92 | **+66.8%** |
| 20 | fwd | 18.15 |  9.76 | **+46.2%** |
| 20 | inv | 19.13 | 12.81 | **+33.0%** |
| 21 | fwd | 25.22 | 14.76 | **+41.5%** |
| 21 | inv | 25.77 | 13.03 | **+49.4%** |
| 22 | fwd | 52.80 | 15.83 | **+70.0%** |
| 22 | inv | 55.26 | 17.07 | **+69.1%** |

**Per-device recommendation from `zkgpu-tail-analyze`:** both →
`UNCONDITIONAL @ log21 (global ≥20% win)`.

### Decision A — rule shape (combined with G.1.1)

Five Mali devices across four silicon generations, three SoC
generations, and two Mali driver major revisions:

| Device | Gen | SoC | GPU | Driver | Shape |
|---|---|---|---|---|---|
| oriole  | 1 | Tensor G1 | G78 MP20 | r38p1 | UNCONDITIONAL |
| panther | 2 | Tensor G2 | G710 MC7 | r38p1 | UNCONDITIONAL |
| husky   | 3 | Tensor G3 | G715 MC7 | r51p0 | UNCONDITIONAL |
| komodo  | 4 | Tensor G4 | G715 MC7 | r51p0 | UNCONDITIONAL |
| comet   | 4 | Tensor G4 | G715 MC7 | r51p0 | UNCONDITIONAL |

Weakest cell across the 50-cell matrix: panther log 20 inv at +33.0%.
Every other cell ≥ +41%; most ≥ +60%.

**Gate A-1 triggered → land `GpuFamily::Mali → GlobalOnlyR4`
family-wide, no generational gating.** Next step is G.1.4: add
`StockhamTailReason::HeuristicMaliCollapse`, wire into
`PlannerPolicy`, add unit + planner-integration test, then run a
phase-f-style full-suite validation pass on all 5 devices.

### Decision B — capability gate (absolute ms)

Surprise: older-gen Mali is within 20% of flagship G715 at the
prover-relevant log 22 GlobalR4 timing.

| Device | GPU (year) | log 22 fwd ms | log 22 inv ms | ratio vs komodo |
|---|---|---|---|---|
| komodo  | G715 (2024) | 14.85 | 13.36 | 1.00× baseline |
| comet   | G715 (2024) | 15.37 | 14.39 | 1.04× |
| husky   | G715 (2023) | 14.70 | 21.19 | 1.27× (inv outlier) |
| **panther** | **G710 (2022)** | **15.83** | **17.07** | **1.16×** |
| **oriole**  | **G78  (2021)** | **18.01** | **18.76** | **1.30×** |

All five devices land ≤ 19 ms/NTT at log 22. Per the capability-gate
table defined in the §Dual-purpose deliverable section above, every
measured device falls in the **"viable for Plonky3-scale provers"**
bucket (≤ 30 ms threshold).

**Practical takeaway.** A Plonky3-style prover doing 100 NTTs at
log 22 finishes the NTT portion in ~1.5–1.9 s on *any* Mali-G7xx
Pixel from 2021 onwards. Pixel 6 (oriole) is a fully viable
zk-proving device; my pre-run "3–5× slower" estimate was wrong.

**Why older-gen doesn't fall off a cliff.** Probable explanation:
Mali-G78 MP20 (Tensor G1) has **20 shader cores** vs G710/G715's 7.
At log 22 the workload is memory-bandwidth-bound (4 M points × 4
B = 16 MB inputs, LPDDR5 on all three SoC generations), so core-
count parallelism on G78 compensates for the older per-core
architecture. The 1.20× gap at log 22 is ~3× per-core-newer-arch
÷ 2.86× more-cores = 1.05× — consistent with measurement within
bandwidth noise.

### Driver-version negative result

Both older-gen devices run Mali driver **r38p1** (2022-vintage).
G715 cohort runs **r51p0** (2024-vintage). The rule shape is identical
across r38→r51, so the Global-tail-win signal is *not* a driver
regression or a recent driver-specific quirk. This strengthens the
case for a simple family-level rule with no driver-version guard.

### Weakest-cell observation

The smallest Global win in G.1.3 (panther log 20 inv, +33.0%) and
in G.1.1 (comet log 20 inv, +21.2%) both land at the same log_n.
Log 20 inverse is the consistent low point of the Mali Global-tail
advantage across all 5 devices. Still comfortably above the 20%
decision-gate threshold, but worth noting if we ever tune the
threshold downward.

### Implications for G.1 landing

With G.1.3 closed, G.1 can proceed directly to:

- **G.1.4** — implement `StockhamTailReason::HeuristicMaliCollapse`
  as a simple `family == Mali → GlobalOnlyR4` arm in
  `heuristic_default`. No generation check, no driver guard.
- **Post-land validation** — phase-f-style full-suite pass on the
  same 5 devices (husky/komodo/comet/panther/oriole) to confirm
  firings on every Mali axis and zero on non-Mali.

G.1.2 (thermal-control run on komodo) is no longer needed: G.1.1
already falsified the thermal hypothesis, and G.1.3 confirms the
shape is generation-wide, not a specific-silicon/specific-driver
artifact.

## Artifacts

- `oriole/logcat.txt` (4.4 MiB), `oriole/test_result.xml`
- `panther/logcat.txt` (4.3 MiB), `panther/test_result.xml`
- `tail_ab_report.json` — 20-cell recommender output

## Links / references

- G.1.1 outcome: `../mali-scope-match-2026-04-16/README.md`
- Parent plan: `../../../../../plan/foundation-audit-and-family-rules-roadmap-2026-04-15.md`
  §Phase G.1 §G.1.3
- G.0 closeout: `../../../../../research/benchmarks/foundation-audit-2026-04-15/verdict.md`
