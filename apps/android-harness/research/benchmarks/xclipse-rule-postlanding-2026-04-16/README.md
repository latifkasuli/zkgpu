# G.2.3 — Post-landing validation of HeuristicXclipseCollapse (2026-04-16)

**Phase.** G.2.3 of `plan/foundation-audit-and-family-rules-roadmap-2026-04-15.md`.
**Status.** ✅ **Validation fully complete** — 5/5 BrowserStack
matrices (4 Xclipse + 1 Adreno 840 control) and 2/2 FTL regression
controls (komodo Mali-G715 + pa3q Adreno 830) all passed 10/10
test cases. Every counter on every family matches expectations:
- Xclipse devices fire `HeuristicXclipseCollapse` 14× each (0 pre-landing → 14 post).
- Mali komodo still fires `HeuristicMaliCollapse` 20× (G.1.4 baseline preserved).
- Adreno pa3q still fires `HeuristicAdrenoCollapse` 10× (G.1.4 baseline preserved).
- Adreno 840 (new silicon) still fires `HeuristicAdrenoCollapse` 10× (matches G.2.2 pre-landing).
- Zero Xclipse misfires on Mali or Adreno across 3 independent regression runs.
- Zero stale rules anywhere. Zero counter drift on forced-A/B.
**Purpose.** Phase-f-style post-landing pass on the Xclipse rule
that landed in commit `<pending-commit>` (`GpuFamily::Xclipse →
GlobalOnlyR4` via `StockhamTailReason::HeuristicXclipseCollapse`).
Mirrors G.1.4's post-landing methodology for the Mali rule exactly.

## TL;DR outcome

All three rule-correctness invariants satisfied with crisp numbers:

1. **Xclipse rule fires on every Xclipse axis.** 14×
   `HeuristicXclipseCollapse` firings on every Xclipse BrowserStack
   matrix (S22, S22 Ultra, S24, S26). The same `14` count that
   previously appeared as `HeuristicDefaultLocal` — a 1:1 pre-to-post
   transfer.
2. **Xclipse rule does NOT fire on non-Xclipse.** 0
   `HeuristicXclipseCollapse` firings on the S26 Ultra Adreno 840
   regression control.
3. **Adreno rule unchanged (regression check).** 10
   `HeuristicAdrenoCollapse` firings on S26 Ultra — identical to
   pre-landing count on the same device. The Xclipse arm's
   insertion after the Mali arm in `heuristic_default` did not
   short-circuit Adreno dispatch.
4. **Zero stale rules.** No `HeuristicXclipseLargeN` /
   `HeuristicMaliLargeN` / any other phase-e-era reason strings on
   any device. Confirms the APK under test is the fresh G.2.3 one.
5. **Forced-A/B unchanged.** 10 ForcedLocal + 10 ForcedGlobal on
   every device, matching pre-landing. The rule change did not
   re-route forced overrides through the heuristic.

## What changed since pre-landing (G.2.2)

Code change scope (`crates/zkgpu-wgpu/src/ntt/planner/tail_policy.rs`
+ `tests.rs`):

1. Added `StockhamTailReason::HeuristicXclipseCollapse` enum variant
   with documentation pointing at the G.2.2 cohort.
2. Added the variant to the `as_str()` serialization match (so
   logcat emits `reason=HeuristicXclipseCollapse`).
3. Added a new arm in `heuristic_default`:
   ```rust
   if matches!(caps.gpu_family, GpuFamily::Xclipse) {
       return (StockhamTailStrategy::GlobalOnlyR4,
               StockhamTailReason::HeuristicXclipseCollapse);
   }
   ```
   Placed immediately after the Mali arm, before the
   `LocalFusedR4 / HeuristicDefaultLocal` fall-through.
4. Updated the module docstring with a new §"Xclipse collapse (G.2.3,
   2026-04-16)" section + refreshed §"Current policy" bullets.
5. Replaced the defensive `xclipse_keeps_local_at_all_sizes` unit
   test and its `xclipse_keeps_local_tail_across_all_sizes` planner-
   integration sibling with:
   - `xclipse_picks_global_tail_at_all_sizes` (unit)
   - `xclipse_picks_global_tail_at_all_tail_sizes` (integration)
6. No changes anywhere else. `StockhamTailReason` is `pub(crate)`,
   so the new variant is contained to `zkgpu-wgpu`.

APK impact:

- `app-debug.apk` grew **by 88 bytes** (19,819,085 → 19,819,173):
  the new enum variant, its match arms, and the
  `HeuristicXclipseCollapse` string constant.
- `app-debug-androidTest.apk` unchanged (pure Kotlin DEX, no native
  lib reference to the planner internals).

Workspace tests: **257/257 pass** across all crates including the
two new Xclipse tests (`cargo test --workspace --lib`).

## Matrix

- App APK: `app-debug.apk` (19.82 MB, built 2026-04-16 16:25, with
  Xclipse rule baked into `libzkgpu_ffi.so` arm64-v8a/x86_64)
- Test APK: unchanged from G.1.1/G.1.4/G.2.2 (same custom_id
  `zkgpu-test-g14` / bs://8c74fce5129ce38757d682367be6bed6ede90d11)
- BrowserStack App Automate, Espresso v2, 5 parallel sessions
- `class=org.zkgpu.harness.ZkgpuInstrumentedTest` (full 10-test suite)

### Device matrix (BrowserStack)

| Device | GPU | Gen | Expected rule firings | Outcome |
|---|---|---|---|:---:|
| Samsung Galaxy S22 (Android 12) | Xclipse 920 | Exynos 2200, RDNA2 2022 | Xclipse:14, Mali:0, Adreno:0 | ✅ |
| Samsung Galaxy S22 Ultra (Android 12) | Xclipse 920 | Exynos 2200, 2022 | Xclipse:14, Mali:0, Adreno:0 | ✅ |
| Samsung Galaxy S24 (Android 16) | Xclipse 940 | Exynos 2400, 2024 | Xclipse:14, Mali:0, Adreno:0 | ✅ |
| Samsung Galaxy S26 (Android 16) | Xclipse 960 | Exynos 2600, 2026 | Xclipse:14, Mali:0, Adreno:0 | ✅ |
| Samsung Galaxy S26 Ultra (Android 16) | Adreno 840 | Snapdragon 8 Elite Gen 5, 2025 | Xclipse:0, Mali:0, Adreno:10 | ✅ |

### Regression-control matrix (FTL)

| Device | Axis | API | GPU | Expected firings | Status | Matrix ID |
|---|---|---|---|---|---|---|
| **komodo (Pixel 9 Pro XL)** | komodo | 35 | **Mali-G715** (driver r51p0) | Mali:20, Xclipse:0, Adreno:0 | **✅ Passed 10/10** | `6297822785230473658` |
| **pa3q (Galaxy S25 Ultra)** | pa3q | 35 | **Adreno 830** | Adreno:10, Xclipse:0, Mali:0 | **✅ Passed 10/10** | `5337666835385033436` |

#### pa3q post-landing firings (confirmed)

| Counter | G.1.4 baseline | G.2.3 post | Delta |
|---|---:|---:|:---:|
| HeuristicAdrenoCollapse | 10 | **10** | unchanged ✅ |
| HeuristicMaliCollapse | 0 | 0 | unchanged ✅ |
| HeuristicXclipseCollapse | n/a (variant didn't exist) | **0** | rule scoped correctly ✅ |
| HeuristicDefaultLocal | 0 | 0 | unchanged ✅ |
| ForcedLocal / ForcedGlobal | 10 / 10 | 10 / 10 | unchanged ✅ |
| Stale (XclipseLargeN, MaliLargeN) | 0 | 0 | unchanged ✅ |

The Xclipse arm's insertion after the Mali arm in `heuristic_default`
did not short-circuit Adreno dispatch. Adreno regression clean.

#### komodo post-landing firings (confirmed)

| Counter | G.1.4 baseline | G.2.3 post | Delta |
|---|---:|---:|:---:|
| HeuristicMaliCollapse | 20 | **20** | unchanged ✅ |
| HeuristicAdrenoCollapse | 0 | 0 | unchanged ✅ |
| HeuristicXclipseCollapse | n/a (variant didn't exist) | **0** | rule scoped correctly ✅ |
| HeuristicDefaultLocal | 0 | 0 | unchanged ✅ |
| ForcedLocal / ForcedGlobal | 10 / 10 | 10 / 10 | unchanged ✅ |
| Stale (XclipseLargeN, MaliLargeN) | 0 | 0 | unchanged ✅ |

Mali rule firings preserved at the exact 20× count from the
G.1.4 baseline. The Xclipse arm correctly does not fire on Mali
silicon, confirming the `matches!(caps.gpu_family, GpuFamily::Xclipse)`
guard is tight.

## Findings (BrowserStack, 2026-04-16)

### Build / session IDs

| Device | Build ID | Session ID | Duration |
|---|---|---|---:|
| S22 (Xclipse 920) | `d2b738621352f0d1aead4888e8d8b35e83be3db9` | — | 174s |
| S22 Ultra (Xclipse 920) | `02bd94c0fb303d0c8be8260dfcc0a0e6910cf4b2` | — | 171s |
| S24 (Xclipse 940) | `6cb06bf8c503a3d1c4e2917ee2377e24f82505fe` | — | 151s |
| S26 (Xclipse 960) | `44ac2f7aa6c34a8ee4abea0058348bad2d42b5fc` | — | 110s |
| S26 Ultra (Adreno 840) | `8d64500eb1b2d1da0f2f86776720487d844f9a6b` | — | 213s |

### Rule-firing table (planner-reason distribution from device logs)

| Device | Silicon | `HeuristicXclipseCollapse` | `HeuristicMaliCollapse` | `HeuristicAdrenoCollapse` | `HeuristicDefaultLocal` | ForcedLocal | ForcedGlobal | Stale |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| S22 | Xclipse 920 | **14** | 0 | 0 | 0 | 10 | 10 | 0 |
| S22 Ultra | Xclipse 920 | **14** | 0 | 0 | 0 | 10 | 10 | 0 |
| S24 | Xclipse 940 | **14** | 0 | 0 | 0 | 10 | 10 | 0 |
| S26 | Xclipse 960 | **14** | 0 | 0 | 0 | 10 | 10 | 0 |
| S26 Ultra | Adreno 840 | 0 | 0 | **10** | 0 | 10 | 10 | 0 |

### Pre-landing → post-landing delta

The cleanest possible rule-landing delta: `HeuristicDefaultLocal`
went from 14 on every Xclipse device (G.2.2) to **0**, and
`HeuristicXclipseCollapse` went from 0 to **14**. No other counters
changed. The mapping is exactly what the new arm specifies.

| Device | Counter | G.2.2 (pre) | G.2.3 (post) |
|---|---|---:|---:|
| S22 (Xclipse 920) | HeuristicDefaultLocal | 14 | 0 |
| | HeuristicXclipseCollapse | 0 | **14** |
| S22 Ultra (Xclipse 920) | HeuristicDefaultLocal | 14 | 0 |
| | HeuristicXclipseCollapse | 0 | **14** |
| S24 (Xclipse 940) | HeuristicDefaultLocal | 14 | 0 |
| | HeuristicXclipseCollapse | 0 | **14** |
| S26 (Xclipse 960) | HeuristicDefaultLocal | 14 | 0 |
| | HeuristicXclipseCollapse | 0 | **14** |
| S26 Ultra (Adreno 840) | HeuristicAdrenoCollapse | 10 | 10 _(unchanged)_ |

### Session duration observation

S26 session completed in 110s (vs 157s pre-landing). The new
`GlobalOnlyR4` code path on Xclipse 960 is actually slightly faster
end-to-end than the `LocalFusedR4` default it replaces — consistent
with the A/B data (11–28% win on Xclipse 960 at log 21–22 Global vs
Local). The win is measurable in wall-clock session time.

Other devices' session times changed within noise (~10-20s) — the
benchmark suite dominates and it was exercising both strategies
before.

## Status

G.2.3 post-landing validation: **GREEN on every rule-correctness
invariant across 7/7 matrices**. Both FTL regression controls
(komodo Mali + pa3q Adreno) returned with firing counts matching
their G.1.4 baselines exactly. No hidden state; no regression.

**The Xclipse rule ships cleanly. G.2 is closed.**

| Invariant | Evidence | Status |
|---|---|:-:|
| Xclipse rule fires on every Xclipse axis | S22/S22U/S24/S26 all report `HeuristicXclipseCollapse = 14` | ✅ |
| Xclipse rule does NOT fire on non-Xclipse | Adreno 840 / Adreno 830 / Mali-G715: all `HeuristicXclipseCollapse = 0` | ✅ |
| Adreno rule unchanged (regression) | pa3q `HeuristicAdrenoCollapse = 10` (matches G.1.4), S26 Ultra = 10 (matches G.2.2) | ✅ |
| Mali rule unchanged (regression) | komodo `HeuristicMaliCollapse = 20` (matches G.1.4) | ✅ |
| Zero stale rules | 0 × 7 devices | ✅ |
| Forced-A/B unchanged | 10 + 10 on every device | ✅ |

### Remaining optional work

- a56x + a56x-reroll (Xclipse 540) still queued on FTL from the
  G.1.4 windfall attempt. When they dequeue, patch a final row
  (Xclipse 540 / Exynos 1580) into the G.2.2 cohort README and
  confirm the rule fires there too. Not gating.
- G.2.1 FTL Exynos-pinning methodology is no longer blocking;
  BrowserStack's App Automate delivers Exynos-pinned axes out of
  the box. G.2.1 remains valuable as a reference doc for future
  FTL-only access scenarios.

## Links / references

- Code change: `crates/zkgpu-wgpu/src/ntt/planner/tail_policy.rs`
  §heuristic_default — Xclipse arm added immediately after Mali arm.
- G.2.2 BrowserStack Xclipse cohort (pre-landing forced-A/B):
  `../browserstack-xclipse-cohort-2026-04-16/README.md`
- G.1.4 Mali post-landing validation (methodology template):
  `../mali-rule-validation-2026-04-16/README.md`
- Parent plan: `../../../../../plan/foundation-audit-and-family-rules-roadmap-2026-04-15.md`
  §Phase G.2 §G.2.3
