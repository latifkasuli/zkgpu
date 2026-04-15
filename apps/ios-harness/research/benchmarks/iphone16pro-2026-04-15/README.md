# iPhone 16 Pro FTL regression pass (2026-04-15)

Cross-platform regression check on iPhone 16 Pro via Firebase Test Lab
iOS, run in parallel with the 7-device Android validation of PR 3's
`HeuristicAdrenoCollapse` rule. Purpose: confirm the Stockham tail
refactor didn't regress the Metal + Apple path while we were touching
the Vulkan family-arm code in `heuristic_default`.

## Why this matters despite PR 3 not changing the Apple arm

The PR 3 change gates on `GpuFamily::Adreno` inside `heuristic_default`,
which only fires on Vulkan. Apple silicon takes a completely different
arm (metal → `LocalFusedR4` everywhere, unchanged). So this run isn't
expected to surface any *behavioral* change — it's a sanity check that
the refactor compiles, links, and runs clean through the iOS
`cbindgen` + Swift bridge + Metal backend.

## FTL matrix

- Xctest bundle: `/tmp/zkgpu-ios-ftl/xctest.zip` (built with Xcode 26.4,
  iOS 16.0 deployment target, generic iOS device destination)
- Results dir: `gs://test-lab-zktc5pzjt539i-i05bjnq39twzw/zkgpu-ios-iphone16pro-2026-04-15-123103/`
- Matrix ID: `matrix-uaqhu81hwtf9a` ([console](https://console.firebase.google.com/project/connect-4-ai-466116/testlab/histories/bh.cfbabbfb249a607f/matrices/9019638139014323010))

| Model       | OS   | GPU              | Outcome | Passed |
|-------------|------|------------------|---------|--------|
| iphone16pro | 18.3 | Apple A18 Pro    | Passed  | 10/10  |

## Tests run

All ten `ZkgpuHarnessTests` methods from the existing iOS harness:

- `testVersionHandshakeParses`
- `testSmokeSuiteSucceeds` / `testSmokeSuiteIncludesDeviceMetadata`
- `testSmokeSuiteStockhamFamily` / `testSmokeSuiteFourStepFamily`
- `testValidationSuiteSucceeds`
- `testBenchmarkSuiteSucceeds`
- `testCrossoverBenchmarkSucceeds` (auto family)
- `testCrossoverStockhamFamily` / `testCrossoverFourStepFamily`

Total wall-clock: 4.703 s for all 10 cases.

## Crossover benchmark timings

`testCrossoverBenchmarkSucceeds` — auto family selection, `log_n ∈ [18, 22]`,
forward + inverse:

| log_n | forward wall | forward gpu | inverse wall | inverse gpu |
|-------|-------------:|------------:|-------------:|------------:|
| 18    |     1.44 ms  |    0.27 ms  |     2.35 ms  |    1.14 ms  |
| 19    |     3.73 ms  |    1.53 ms  |     3.29 ms  |    1.52 ms  |
| 20    |     5.56 ms  |    2.50 ms  |     7.85 ms  |    5.44 ms  |
| 21    |     6.48 ms  |    3.75 ms  |     6.28 ms  |    3.05 ms  |
| 22    |    12.15 ms  |    8.94 ms  |    13.62 ms  |    6.54 ms  |

`testCrossoverStockhamFamily` — forced Stockham, same `log_n` sweep:

| log_n | forward gpu | inverse gpu |
|-------|------------:|------------:|
| 18    |    0.43 ms  |    0.19 ms  |
| 19    |    0.23 ms  |    0.49 ms  |
| 20    |    0.73 ms  |    1.98 ms  |
| 21    |    6.13 ms  |    3.06 ms  |
| 22    |    8.43 ms  |    8.92 ms  |

`testCrossoverFourStepFamily` — forced four-step:

| log_n | forward gpu | inverse gpu |
|-------|------------:|------------:|
| 18    |    0.38 ms  |    0.86 ms  |
| 19    |    1.82 ms  |    5.31 ms  |
| 20    |    3.75 ms  |    3.77 ms  |
| 21    |    6.44 ms  |    5.92 ms  |
| 22    |   12.44 ms  |   12.18 ms  |

Forced-Stockham at log22 (8.43ms fwd) beats forced-four-step (12.44ms fwd)
by ~32% — consistent with the planner's "Apple → always Stockham" policy
and with the unchanged `LocalFusedR4` tail. Nothing here suggests an
Apple-side tail rule is needed.

## Limitations / deferred follow-up

This run does **not** exercise the forced-A/B `crossoverStockhamLocalTail`
/ `crossoverStockhamGlobalTail` tests — those don't yet exist on the
iOS side. The Swift `SuiteSpec` struct doesn't plumb
`stockham_tail_override` to the Rust FFI boundary (the FFI accepts it,
Swift just doesn't send it). Adding Swift forced-A/B tests and running
them against iphone16pro + iphone14pro would be the natural next step
if we ever want field data on whether the Apple path has a tail-strategy
signal — so far, the theory is "no, Apple's shared memory is large
enough to never hit the gather collapse."

## Artifacts

- `test_result_0.xml` — JUnit XML (10/10 tests passing)
- `xcodebuild_output.log` — full xcodebuild runner output including per-case timings
- `syslog.txt` — device syslog captured during the run (~5 MB, mostly kernel noise)
