# Mobile R8 A/B via BrowserStack (2026-04-17)

Parent context: `crates/zkgpu-wgpu/src/ntt/planner/policy.rs:87` — the
`r8_max_log_leaf()` gate that controls whether Four-Step leaves use
Radix-8 (R8) or fall back to Radix-4+Radix-2 (R4+R2). The shipped
default for all non-NVIDIA families is `u32::MAX`, i.e. R8 always on,
based on reasoning rather than a measured A/B.

This experiment measures the R8 A/B directly on one real device per
mobile family: Adreno, Mali, Xclipse.

## TL;DR

**The shipped default is wrong on Mali-G925 and Xclipse 940.** R4 wins
by up to 3× at log_n ≤ 20 on both. R8 catches up only at log_n=22.
Adreno 840 matches the current default (R8 wins 1.13-1.33× at every
measured size).

| Family | Device | R8 default today | Measured verdict |
|---|---|---|---|
| Adreno | Samsung S26 Ultra (Adreno 840) | R8 always on | ✅ correct — R8 wins 1.13-1.33× |
| Mali | Samsung Tab S11 (Mali-G925-Immortalis) | R8 always on | ❌ R4 wins 1.1-3.0× at log 18-20, tied at 22 |
| Xclipse | Samsung S24 (Xclipse 940) | R8 always on | ❌ R4 wins 1.2-2.4× at log 18-20, tied at 22 |

**Magnitude matters.** At log_n=18 inverse on Mali-G925, R8 takes
3.89 ms GPU while R4 takes 1.52 ms — an absolute 2.37 ms regression
from the "R8 on by default" policy at a size that's in the hot path
for many ZK proof systems. This is not a theoretical concern.

## Method

Driver: `apps/android-harness` Espresso instrumented tests on
BrowserStack App Automate. Two new test methods added for this run:

- `ZkgpuInstrumentedTest#crossoverFourStepR8Enabled` — forces
  `family_override: FourStep` + `r8_max_log_leaf_override: u32::MAX`
  (R8 enabled regardless of per-family policy).
- `ZkgpuInstrumentedTest#crossoverFourStepR8Disabled` — same, but
  `r8_max_log_leaf_override: 0` (R8 disabled, pure R4+R2 leaves).

Per-case: 5 iterations + 2 warmup, Sequential input, Forward + Inverse,
`profile_gpu_timestamps: true`.

log_n ∈ {18, 20, 22} — covers log_leaf ∈ {9, 10, 11} at balanced
Four-Step row × col split. log_n=24 (log_leaf=12) omitted for the
initial sweep to keep per-device memory reasonable on BrowserStack
devices (~4 GB shared RAM).

### Plumbing added

End-to-end `r8_max_log_leaf_override: Option<u32>` through:
- `zkgpu-report::SuiteSpec` + `HarnessRequest`
- `zkgpu-ffi::json::run_request` (top-level override overrides spec-level)
- `zkgpu-testkit::runner::make_plan` (new param, forwarded to `PlannerPolicy`)
- `zkgpu-web::runner` (same, for future web-harness use)
- `PlannerPolicy::with_r8_max_log_leaf_override` builder method
- `PlannerPolicy::r8_max_log_leaf()` — struct field wins over env var
- `HarnessJson.customBenchmarkRequestJson` (Kotlin harness builder)

Precedence (highest first): struct field override → env
`ZKGPU_R8_MAX_LOG_LEAF` → per-(backend, family) default.

## Data (GPU time, ms, median of 5 iters after 2 warmups)

### Samsung S26 Ultra (Adreno 840, SD 8 Elite Gen 5, Android 16)

| log_n | dir | R8-on | R8-off | off/on | verdict |
|---|---|---|---|---|---|
| 18 | fwd | 0.300 | 0.350 | 1.17 | R8 wins 1.17× |
| 18 | inv | 0.300 | 0.350 | 1.17 | R8 wins 1.17× |
| 20 | fwd | 1.440 | 1.620 | 1.13 | R8 wins 1.13× |
| 20 | inv | 1.480 | 1.590 | 1.07 | tied |
| 22 | fwd | 6.730 | 8.930 | 1.33 | **R8 wins 1.33×** |
| 22 | inv | 6.730 | 8.460 | 1.26 | **R8 wins 1.26×** |

Current default (R8 always on) is correct for Adreno 840. No change.

### Samsung Galaxy Tab S11 (Mali-G925-Immortalis, Dimensity 9400+, Android 16)

| log_n | dir | R8-on | R8-off | off/on | verdict |
|---|---|---|---|---|---|
| 18 | fwd | 3.720 | 1.230 | 0.33 | **R4 wins 3.02×** |
| 18 | inv | 3.890 | 1.520 | 0.39 | **R4 wins 2.56×** |
| 20 | fwd | 9.070 | 8.120 | 0.90 | R4 wins 1.12× |
| 20 | inv | 9.100 | 7.970 | 0.88 | R4 wins 1.14× |
| 22 | fwd | 21.520 | 22.610 | 1.05 | tied |
| 22 | inv | 21.450 | 23.260 | 1.08 | tied |

Shipped default is **wrong** at log 18 and 20.

### Samsung Galaxy S24 (Xclipse 940, Exynos 2400, Android 16)

| log_n | dir | R8-on | R8-off | off/on | verdict |
|---|---|---|---|---|---|
| 18 | fwd | 3.120 | 1.720 | 0.55 | **R4 wins 1.81×** |
| 18 | inv | 4.120 | 1.720 | 0.42 | **R4 wins 2.40×** |
| 20 | fwd | 9.220 | 6.520 | 0.71 | **R4 wins 1.41×** |
| 20 | inv | 7.960 | 6.830 | 0.86 | R4 wins 1.17× |
| 22 | fwd | 19.720 | 21.260 | 1.08 | tied |
| 22 | inv | 17.930 | 21.040 | 1.17 | R8 wins 1.17× |

Shipped default is **wrong** at log 18 and 20.

## Pattern & hypothesis

Mali and Xclipse both show the same shape: R4 dominates at log_leaf
∈ {9, 10} and R8 catches up only at log_leaf=11. Adreno doesn't show
this shape — R8 wins from the bottom.

Likely cause: R8 leaves carry higher register pressure (8-point
butterfly vs 4-point, ~2× more intermediate locals) and larger
shared-memory footprint. On Mali/Xclipse's narrower shader cores,
the extra pressure costs more than the 3× butterfly-per-stage
throughput gain until the leaf is large enough (log_leaf ≥ 11) to
amortize it. Adreno 840's wider execution units appear to swallow
the register/shared-memory cost without slowdown.

This is consistent with the Apple Metal result (R8 wins cleanly
everywhere) — Apple's unified memory + Metal command encoding hides
most of the R8 overhead.

## Policy implications (proposed, not shipped)

The current R8 gate has only a **max** threshold (`use_r8 when
num_global_stages <= r8_max_log_leaf`). The measurement suggests
Mali/Xclipse want the *opposite* polarity — R8 only when
`log_leaf >= 11`. Options:

1. **Disable R8 on Mali and Xclipse entirely** (`r8_max_log_leaf=0`).
   Loses the tied/1.17× log-22 case on Xclipse but recovers 1.4-3.0×
   at log 18-20. Net clearly positive on the measured device.
2. **Add a min threshold** (`r8_min_log_leaf`) and ship
   `min=11, max=u32::MAX` for Mali/Xclipse. Preserves log-22 R8 win
   while avoiding the log ≤ 20 regression. Small code change to
   `stockham_config.rs:234`.
3. **Leave default alone**, wait for more devices.

Not shipped in this commit — `n=1 per family` is insufficient on its
own. Minimum for a policy change: re-run on 2-3 more Mali SoCs
(G715, G710, earlier G925 steppings) and 2-3 more Xclipse SoCs (920,
960) to confirm the pattern generalizes before changing shipped
defaults. BrowserStack already has `Samsung Galaxy S22`
(Xclipse 920), `Samsung Galaxy S26` (Xclipse 960), and Mali-G715
cohort (from earlier sessions) we can replay the A/B on.

## Reproduction

```bash
# Build Rust FFI + Android APKs
cargo ndk -t arm64-v8a -t x86_64 -p 29 \
  -o apps/android-harness/app/src/main/jniLibs \
  build -p zkgpu-ffi --release

cd apps/android-harness
./gradlew assembleDebug assembleDebugAndroidTest

# Upload + run (needs ~/.browserstack/credentials)
source ~/.browserstack/credentials

APP_URL=$(curl -s -u "$BROWSERSTACK_USERNAME:$BROWSERSTACK_ACCESS_KEY" \
  -X POST "https://api-cloud.browserstack.com/app-automate/espresso/v2/app" \
  -F "file=@app/build/outputs/apk/debug/app-debug.apk" | jq -r .app_url)
TEST_URL=$(curl -s -u "$BROWSERSTACK_USERNAME:$BROWSERSTACK_ACCESS_KEY" \
  -X POST "https://api-cloud.browserstack.com/app-automate/espresso/v2/test-suite" \
  -F "file=@app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk" \
  | jq -r .test_suite_url)

curl -s -u "$BROWSERSTACK_USERNAME:$BROWSERSTACK_ACCESS_KEY" \
  -X POST "https://api-cloud.browserstack.com/app-automate/espresso/v2/build" \
  -H "Content-Type: application/json" -d '{
    "devices": [
      "Samsung Galaxy S26 Ultra-16.0",
      "Samsung Galaxy Tab S11-16.0",
      "Samsung Galaxy S24-16.0"
    ],
    "app": "'$APP_URL'",
    "testSuite": "'$TEST_URL'",
    "class": [
      "org.zkgpu.harness.ZkgpuInstrumentedTest#crossoverFourStepR8Enabled",
      "org.zkgpu.harness.ZkgpuInstrumentedTest#crossoverFourStepR8Disabled"
    ],
    "deviceLogs": true, "video": false
  }'
```

## Build / session references

- Build ID: `592396f811439cf321582e53a70964334a75d601`
- Session IDs:
  - S26 Ultra: `e773612bca880971c12a278212e3317cd2b1e735` (passed, 35s)
  - Tab S11: `ccf00716080cac20737dfacba5f9818cf75d67fd` (passed, 55s)
  - S24: `5b4586dd0bbeb028c4e87d7e9fa6f9092b8d22a9` (passed, 52s)

Per-device logcat per test in `{subdir}/{testname}.log`.

## Artifacts in this directory

- `s26-ultra-adreno840/crossoverFourStepR8{Enabled,Disabled}.log`
- `tab-s11-mali-g925/crossoverFourStepR8{Enabled,Disabled}.log`
- `s24-xclipse940/crossoverFourStepR8{Enabled,Disabled}.log`

Each log has the `R8_AB_ENABLED` / `R8_AB_DISABLED` tag prefix so a
single grep attributes lines to an arm.
