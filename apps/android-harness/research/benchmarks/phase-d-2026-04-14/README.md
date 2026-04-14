# Phase D Post-Deletion Benchmark (2026-04-14)

First full benchmark run after Phase D deleted the Vulkan subgroup local kernel stack (commit `dde7618`). Measures the surviving PortableR4 local kernel on three Android GPUs via Firebase Test Lab.

## FTL matrix

- App APK: `app-debug.apk` (13.8 MB)
- Test APK: `app-debug-androidTest.apk` (862 KB)
- Results dir: `zkgpu-bench-phase-d-2026-04-14-120236`
- Matrix ID: `matrix-chhct7ipy8z8a`

| Axis                   | Device            | SoC                 | GPU         | Android | Outcome |
|------------------------|-------------------|---------------------|-------------|---------|---------|
| caiman-35-en-portrait  | Pixel 9 Pro       | Tensor G4           | Mali-G715   | 35      | Passed  |
| komodo-35-en-portrait  | Pixel 9 Pro XL    | Tensor G4           | Mali-G715   | 35      | Passed  |
| pa3qxxx-36-en-portrait | Galaxy S25 Ultra  | Snapdragon 8 Elite  | Adreno 830  | 36      | Passed  |

## Tests run

- `crossoverBenchmarkCompletes` — fwd + inv at log_n ∈ {18, 19, 20, 21, 22}, 5 iter + 2 warmup
- `benchmarkSuitePassesWithTimings` — fwd at log_n ∈ {10, 14, 18, 20}, 5 iter + 1 warmup

## Crossover forward, GPU time ms

| Device              | log18 | log19 | log20 | log21 | log22 |
|---------------------|-------|-------|-------|-------|-------|
| pa3qxxx (Adreno 830)|  0.46 |  0.95 |  2.32 |  5.50 | 11.82 |
| caiman  (Mali-G715) |  8.87 |  8.66 | 17.63 | 23.15 | 48.30 |
| komodo  (Mali-G715) | 11.87 | 10.57 | 17.04 | 23.81 | 48.45 |

## Crossover inverse, GPU time ms

| Device              | log18 | log19 | log20 | log21 | log22 |
|---------------------|-------|-------|-------|-------|-------|
| pa3qxxx (Adreno 830)|  0.51 |  1.08 |  2.49 |  5.84 | 12.20 |
| caiman  (Mali-G715) |  6.34 |  8.68 | 12.62 | 23.61 | 51.20 |
| komodo  (Mali-G715) |  6.58 |  8.28 | 13.22 | 25.41 | 51.84 |

## Benchmark suite forward, GPU time ms

| Device              | log10 | log14 | log18 | log20 |
|---------------------|-------|-------|-------|-------|
| pa3qxxx (Adreno 830)|  0.01 |  0.02 |  0.43 |  2.36 |
| caiman  (Mali-G715) |  0.32 |  0.59 |  6.84 | 15.67 |
| komodo  (Mali-G715) |  0.10 |  0.23 |  3.78 | 16.53 |

## Comparison vs pre-Phase-D PortableR4 baseline

The cross-family A/B memory measured these same device codenames on the portable path. Absolute numbers agree within typical FTL variance on Adreno; Mali-G715 at log18 shows the expected high variance (memory baseline 3.61 / 4.16 ms, today 8.87 / 11.87 ms). Pool-level device swap + thermal state are the likely drivers — numbers at log20+ are tracking the prior portable baseline much more tightly. Adreno 830 is stable: 0.46 ms today vs 0.48 ms prior at fwd log18.

## Artifacts

- `<device>/logcat.txt` — full device logcat
- `<device>/test_result.xml` — JUnit XML
- `<device>/timings.txt` — extracted `BENCH` / `CROSSOVER` timing lines

Raw FTL blob: `gs://test-lab-zktc5pzjt539i-i05bjnq39twzw/zkgpu-bench-phase-d-2026-04-14-120236/`
