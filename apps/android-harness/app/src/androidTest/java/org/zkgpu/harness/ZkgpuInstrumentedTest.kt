package org.zkgpu.harness

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class ZkgpuInstrumentedTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    @Test
    fun smokeSuiteReturnsOk() {
        val response = runPreset(PresetSuite.Smoke)
        assertTrue(response.optString("error"), response.optBoolean("ok", false))
    }

    @Test
    fun validationSuitePassesOnSupportedHardware() {
        val response = runPreset(PresetSuite.Validation)
        assertTrue(response.optString("error"), response.optBoolean("ok", false))

        val report = response.getJSONObject("report")
        val summary = report.getJSONObject("summary")
        assertEquals(0, summary.getInt("failed_cases"))
    }

    @Test
    fun returnedJsonParses() {
        JSONObject(ZkgpuBridge.getVersionJson())
        JSONObject(ZkgpuBridge.runRequestJson(HarnessJson.presetRequestJson(PresetSuite.Smoke)))
    }

    @Test
    fun deviceMetadataIsNonEmpty() {
        val response = runPreset(PresetSuite.Smoke)
        assertTrue(response.optString("error"), response.optBoolean("ok", false))

        val device = response.getJSONObject("report").getJSONObject("device")
        assertNotNull(device.optString("name"))
        assertTrue(device.optString("name").isNotBlank())
        assertTrue(device.optString("backend").isNotBlank())
        assertTrue(device.optString("tier").isNotBlank())
    }

    /**
     * Built-in Benchmark suite: forward NTTs at log_n 10, 14, 18, 20
     * with 5 iterations + 1 warmup and GPU profiling enabled.
     */
    @Test
    fun benchmarkSuitePassesWithTimings() {
        val response = runPreset(PresetSuite.Benchmark)
        assertTrue(response.optString("error"), response.optBoolean("ok", false))

        val report = response.getJSONObject("report")
        val summary = report.getJSONObject("summary")
        assertEquals(0, summary.getInt("failed_cases"))
        assertTrue("expected 4 benchmark cases", summary.getInt("total_cases") == 4)

        // Log per-case timings for Firebase results
        val cases = report.getJSONArray("cases")
        for (i in 0 until cases.length()) {
            val c = cases.getJSONObject(i)
            val name = c.optString("name")
            val timings = c.optJSONObject("timings")
            val wallNs = timings?.optLong("wall_time_ns", -1) ?: -1L
            val gpuNs = timings?.optLong("gpu_total_ns", -1) ?: -1L
            val wallMs = if (wallNs > 0) wallNs / 1_000_000.0 else -1.0
            val gpuMs = if (gpuNs > 0) gpuNs / 1_000_000.0 else -1.0
            Log.i(TAG, "BENCH $name: wall=${"%.2f".format(wallMs)}ms gpu=${"%.2f".format(gpuMs)}ms")
        }
    }

    /**
     * Crossover benchmark: forward + inverse NTTs at log_n 18–22,
     * 5 iterations + 2 warmups with GPU profiling. Auto family:
     * the planner picks Stockham or FourStep based on log_n + capability.
     */
    @Test
    fun crossoverBenchmarkCompletes() {
        runCrossoverBenchmark(FamilyChoice.Auto, tagPrefix = "CROSSOVER")
    }

    /**
     * Crossover benchmark forced onto the Stockham family (all log_n use
     * Stockham autosort; the planner's four-step threshold is overridden).
     */
    @Test
    fun crossoverBenchmarkStockhamFamily() {
        runCrossoverBenchmark(FamilyChoice.Stockham, tagPrefix = "CROSSOVER_STOCKHAM")
    }

    /**
     * Crossover benchmark forced onto the four-step family at all log_n.
     * Exercises the large-N decomposition even at small sizes.
     */
    @Test
    fun crossoverBenchmarkFourStepFamily() {
        runCrossoverBenchmark(FamilyChoice.FourStep, tagPrefix = "CROSSOVER_FOURSTEP")
    }

    /**
     * Crossover under forced Stockham family + forced `LocalFusedR4` tail.
     * Pinpoints the legacy local-fused tail behavior so the all-global
     * variant below has a baseline to A/B against on Xclipse / Mali.
     */
    @Test
    fun crossoverStockhamLocalTail() {
        runCrossoverBenchmark(
            FamilyChoice.Stockham,
            tagPrefix = "CROSSOVER_STOCKHAM_LOCAL_TAIL",
            tail = TailChoice.Local,
        )
    }

    /**
     * Crossover under forced Stockham family + forced `GlobalOnlyR4` tail.
     * This is the production direction PR 1 made possible — extending the
     * global R4 chain through the tail. Compare per-stage timings against
     * `crossoverStockhamLocalTail` to validate the heuristic threshold.
     */
    @Test
    fun crossoverStockhamGlobalTail() {
        runCrossoverBenchmark(
            FamilyChoice.Stockham,
            tagPrefix = "CROSSOVER_STOCKHAM_GLOBAL_TAIL",
            tail = TailChoice.Global,
        )
    }

    /**
     * R8 A/B, R8-on arm: forced Four-Step family with R8 leaves *enabled*
     * regardless of per-family policy (r8_max_log_leaf_override = u32::MAX).
     *
     * Pair with [crossoverFourStepR8Disabled] on the same device to measure
     * the R8-vs-R4 delta on Adreno / Mali / Xclipse — the families where
     * the shipped default is "R8 always on" (u32::MAX) based on reasoning
     * rather than a measured A/B. If the R8-on arm doesn't win by >10% on
     * any (log_n, direction) cell, the default is conservative-safe but
     * not a measured win.
     *
     * log_n ∈ {18, 20, 22} covers log_leaf ∈ {9, 10, 11} (balanced
     * row × col split inside Four-Step). log_n=24 would push log_leaf=12
     * but risks OOM on lower-RAM devices; start with the safer window.
     */
    @Test
    fun crossoverFourStepR8Enabled() {
        runR8ABBenchmark(
            r8MaxLogLeafOverride = R8_FORCE_ENABLED,
            tagPrefix = "R8_AB_ENABLED",
        )
    }

    /**
     * R8 A/B, R8-off arm: forced Four-Step family with R8 leaves *disabled*
     * (r8_max_log_leaf_override = 0). Every leaf falls back to pure R4+R2.
     */
    @Test
    fun crossoverFourStepR8Disabled() {
        runR8ABBenchmark(
            r8MaxLogLeafOverride = R8_FORCE_DISABLED,
            tagPrefix = "R8_AB_DISABLED",
        )
    }

    private fun runR8ABBenchmark(
        r8MaxLogLeafOverride: Long,
        tagPrefix: String,
    ) {
        val requestJson = HarnessJson.customBenchmarkRequestJson(
            logNRange = R8_AB_LOG_N,
            family = FamilyChoice.FourStep,
            tail = TailChoice.Auto,
            r8MaxLogLeafOverride = r8MaxLogLeafOverride,
        )
        val responseJson = ZkgpuBridge.runRequestJson(requestJson)
        val file = HarnessStorage.writeLatestReport(context, responseJson)
        Log.i(
            TAG,
            "$tagPrefix path=${file.absolutePath} " +
                "family_override=FourStep r8_max_log_leaf_override=$r8MaxLogLeafOverride",
        )

        val response = JSONObject(responseJson)
        assertTrue(response.optString("error"), response.optBoolean("ok", false))

        val report = response.getJSONObject("report")
        val summary = report.getJSONObject("summary")
        assertEquals(0, summary.getInt("failed_cases"))
        val expectedCases = R8_AB_LOG_N.size * 2
        assertTrue(
            "expected $expectedCases R8 A/B cases",
            summary.getInt("total_cases") == expectedCases,
        )

        val kernel = report.optJSONObject("kernel")
        Log.i(
            TAG,
            "$tagPrefix kernel.ntt_variant=${kernel?.optString("ntt_variant").orEmpty()}",
        )

        val device = report.optJSONObject("device")
        if (device != null) {
            Log.i(
                TAG,
                "$tagPrefix device name=\"${device.optString("name")}\" " +
                    "backend=${device.optString("backend")} " +
                    "tier=${device.optString("tier")} " +
                    "family=${device.optString("gpu_family")} " +
                    "driver=\"${device.optString("driver_info")}\"",
            )
        }

        val cases = report.getJSONArray("cases")
        for (i in 0 until cases.length()) {
            val c = cases.getJSONObject(i)
            val name = c.optString("name")
            val kernelFamily = c.optString("kernel_family", "unknown")
            val timings = c.optJSONObject("timings")
            val wallNs = timings?.optLong("wall_time_ns", -1) ?: -1L
            val gpuNs = timings?.optLong("gpu_total_ns", -1) ?: -1L
            val wallMs = if (wallNs > 0) wallNs / 1_000_000.0 else -1.0
            val gpuMs = if (gpuNs > 0) gpuNs / 1_000_000.0 else -1.0
            Log.i(
                TAG,
                "$tagPrefix $name: family=$kernelFamily " +
                    "wall=${"%.2f".format(wallMs)}ms gpu=${"%.2f".format(gpuMs)}ms",
            )

            // Per-stage timings help attribute the R4↔R8 delta to specific
            // leaf stages vs the twiddle / transpose shared by both arms.
            val stages = timings?.optJSONArray("gpu_stage_ns")
            if (stages != null) {
                for (s in 0 until stages.length()) {
                    val stage = stages.getJSONObject(s)
                    val label = stage.optString("label")
                    val durNs = stage.optLong("duration_ns", 0)
                    val durMs = durNs / 1_000_000.0
                    Log.i(
                        TAG,
                        "$tagPrefix $name stage[$s] \"$label\" gpu=${"%.3f".format(durMs)}ms",
                    )
                }
            }
        }
    }

    private fun runCrossoverBenchmark(
        family: FamilyChoice,
        tagPrefix: String,
        tail: TailChoice = TailChoice.Auto,
    ) {
        val requestJson = HarnessJson.crossoverBenchmarkRequestJson(family, tail)
        val responseJson = ZkgpuBridge.runRequestJson(requestJson)
        val file = HarnessStorage.writeLatestReport(context, responseJson)
        Log.i(
            TAG,
            "$tagPrefix path=${file.absolutePath} " +
                "family_override=${family.name} tail_override=${tail.name}",
        )

        val response = JSONObject(responseJson)
        assertTrue(response.optString("error"), response.optBoolean("ok", false))

        val report = response.getJSONObject("report")
        val summary = report.getJSONObject("summary")
        assertEquals(0, summary.getInt("failed_cases"))
        assertTrue("expected 10 crossover cases", summary.getInt("total_cases") == 10)

        // Suite-level kernel metadata (field + ntt_variant + tail summary).
        val kernel = report.optJSONObject("kernel")
        val ntt_variant = kernel?.optString("ntt_variant").orEmpty()
        // Suite-level tail summary: one of "LocalFusedR4", "GlobalOnlyR4",
        // "mixed", or omitted. Emitting it here means a single grep over
        // logcat tells you which tail strategy a forced run actually
        // exercised, without parsing per-case JSON.
        val tailVariant = kernel?.optStringOrNull("stockham_tail_strategy")
        Log.i(
            TAG,
            "$tagPrefix kernel.ntt_variant=$ntt_variant " +
                "kernel.stockham_tail_strategy=${tailVariant ?: "none"}",
        )

        // Device metadata — emitted once per run so artifacts capture the
        // adapter limits that drive kernel-tuning decisions (shared-memory
        // budget, workgroup size, etc.).
        val device = report.optJSONObject("device")
        if (device != null) {
            val wgStorage = device.optInt("max_compute_workgroup_storage_size_bytes", -1)
            Log.i(
                TAG,
                "$tagPrefix device name=\"${device.optString("name")}\" " +
                    "backend=${device.optString("backend")} " +
                    "tier=${device.optString("tier")} " +
                    "family=${device.optString("gpu_family")} " +
                    "driver=\"${device.optString("driver_info")}\" " +
                    "max_buffer=${device.optLong("max_buffer_size_bytes")} " +
                    "max_wg_x=${device.optInt("max_workgroup_size_x")} " +
                    "max_invocations=${device.optInt("max_compute_invocations")} " +
                    "max_wg_storage_bytes=$wgStorage",
            )
        }

        // Per-case: actual kernel_family selected + tail metadata + per-stage GPU timings.
        val cases = report.getJSONArray("cases")
        for (i in 0 until cases.length()) {
            val c = cases.getJSONObject(i)
            val name = c.optString("name")
            val kernelFamily = c.optString("kernel_family", "unknown")
            // PR 1 tail observability: `stockham_tail_strategy`,
            // `stockham_tail_reason`, and `tail_stride_bytes` are the
            // three fields that tell you *which* Stockham tail ran and
            // *why*. Logged on every case so the Xclipse / Mali / Browser
            // A/B work doesn't have to re-parse JSON to attribute timing.
            val tailStrategy = c.optStringOrNull("stockham_tail_strategy")
            val tailReason = c.optStringOrNull("stockham_tail_reason")
            val tailStrideBytes = if (c.has("tail_stride_bytes") && !c.isNull("tail_stride_bytes")) {
                c.optLong("tail_stride_bytes", -1L)
            } else {
                -1L
            }
            val timings = c.optJSONObject("timings")
            val wallNs = timings?.optLong("wall_time_ns", -1) ?: -1L
            val gpuNs = timings?.optLong("gpu_total_ns", -1) ?: -1L
            val wallMs = if (wallNs > 0) wallNs / 1_000_000.0 else -1.0
            val gpuMs = if (gpuNs > 0) gpuNs / 1_000_000.0 else -1.0
            Log.i(
                TAG,
                "$tagPrefix $name: family=$kernelFamily " +
                    "tail=${tailStrategy ?: "none"} " +
                    "reason=${tailReason ?: "none"} " +
                    "stride_bytes=${if (tailStrideBytes >= 0) tailStrideBytes.toString() else "none"} " +
                    "wall=${"%.2f".format(wallMs)}ms gpu=${"%.2f".format(gpuMs)}ms",
            )

            // Emit per-stage GPU timings (label + duration_ns) so the
            // Xclipse log22 investigation can attribute time to a specific
            // stage (r4/r2/local fused/inverse scale for Stockham;
            // transpose/leaf/twiddle for four-step).
            val stages = timings?.optJSONArray("gpu_stage_ns")
            if (stages != null) {
                for (s in 0 until stages.length()) {
                    val stage = stages.getJSONObject(s)
                    val label = stage.optString("label")
                    val durNs = stage.optLong("duration_ns", 0)
                    val durMs = durNs / 1_000_000.0
                    Log.i(
                        TAG,
                        "$tagPrefix $name stage[$s] \"$label\" gpu=${"%.3f".format(durMs)}ms",
                    )
                }
            }
        }
    }

    private fun runPreset(suite: PresetSuite): JSONObject {
        val responseJson = ZkgpuBridge.runRequestJson(HarnessJson.presetRequestJson(suite))
        val file = HarnessStorage.writeLatestReport(context, responseJson)
        Log.i(TAG, "suite=${suite.wireName} path=${file.absolutePath}")
        return JSONObject(responseJson)
    }

    // === Phase F.3.e: Poseidon2 hash suite ===

    /**
     * BabyBear Poseidon2 smoke — 5 cases covering every input
     * pattern plus the prime-17 batch that exercises 2D-fold
     * dispatch on mobile. Every case must pass against the CPU
     * reference in `zkgpu-poseidon2`. The response shape uses
     * `hash_report` instead of `report` (F.3.e FFI dispatch).
     */
    @Test
    fun poseidon2BabyBearSmokeSuitePasses() {
        val response = runPoseidon2Smoke(HashField.BabyBear)
        assertTrue(response.optString("error"), response.optBoolean("ok", false))

        val hashReport = response.optJSONObject("hash_report")
        assertNotNull("response must carry hash_report on the hash path", hashReport)
        val summary = hashReport!!.getJSONObject("summary")
        assertEquals(0, summary.getInt("failed_cases"))
        assertEquals(5, summary.getInt("total_cases"))

        val kernel = hashReport.getJSONObject("kernel")
        assertEquals("BabyBear", kernel.getString("field"))
        assertEquals("babybear-poseidon2", kernel.getString("ntt_variant"))
    }

    /**
     * Goldilocks Poseidon2 smoke — same case set, routed through the
     * portable u32x2 GPU plan. Catches any drift between the 32-bit
     * and 64-bit WGSL kernels on the target device.
     */
    @Test
    fun poseidon2GoldilocksSmokeSuitePasses() {
        val response = runPoseidon2Smoke(HashField.Goldilocks)
        assertTrue(response.optString("error"), response.optBoolean("ok", false))

        val hashReport = response.optJSONObject("hash_report")
        assertNotNull("response must carry hash_report on the hash path", hashReport)
        val summary = hashReport!!.getJSONObject("summary")
        assertEquals(0, summary.getInt("failed_cases"))

        val kernel = hashReport.getJSONObject("kernel")
        assertEquals("Goldilocks", kernel.getString("field"))
        assertEquals("goldilocks-poseidon2-portable", kernel.getString("ntt_variant"))
    }

    /**
     * Poseidon2 benchmark ladder — 1k / 16k / 65k permutations,
     * both fields. Wall-time-only on this phase (profiled-execute
     * not yet wired on Poseidon2 plans); log throughput per case so
     * Firebase/BrowserStack captures per-device M perms/s without
     * an external CSV step.
     */
    @Test
    fun poseidon2BenchmarkLadderBothFields() {
        for (field in arrayOf(HashField.BabyBear, HashField.Goldilocks)) {
            val requestJson = HarnessJson.poseidon2BenchmarkRequestJson(
                permutationCounts = POSEIDON2_BENCH_LADDER,
                field = field,
            )
            val responseJson = ZkgpuBridge.runRequestJson(requestJson)
            val file = HarnessStorage.writeLatestReport(context, responseJson)
            Log.i(
                TAG,
                "poseidon2_bench field=${field.wireValue} path=${file.absolutePath}",
            )
            val response = JSONObject(responseJson)
            assertTrue(response.optString("error"), response.optBoolean("ok", false))

            val hashReport = response.getJSONObject("hash_report")
            val summary = hashReport.getJSONObject("summary")
            assertEquals(
                "benchmark ladder failed on field=${field.wireValue}",
                0,
                summary.getInt("failed_cases"),
            )

            val cases = hashReport.getJSONArray("cases")
            for (i in 0 until cases.length()) {
                val c = cases.getJSONObject(i)
                val name = c.optString("name")
                val n = c.optInt("num_permutations", 0)
                val timings = c.optJSONObject("timings")
                val wallNs = timings?.optLong("wall_time_ns", -1) ?: -1L
                val wallUs = if (wallNs > 0) wallNs / 1_000.0 else -1.0
                val permsPerSec = if (wallUs > 0 && n > 0) {
                    n.toDouble() * 1_000_000.0 / wallUs
                } else {
                    0.0
                };
                Log.i(
                    TAG,
                    "POSEIDON2_BENCH field=${field.wireValue} $name n=$n " +
                        "wall=${"%.0f".format(wallUs)}us " +
                        "${"%.3f".format(permsPerSec / 1e6)}M perms/s",
                )
            }
        }
    }

    private fun runPoseidon2Smoke(field: HashField): JSONObject {
        val responseJson = ZkgpuBridge.runRequestJson(
            HarnessJson.poseidon2SmokeRequestJson(field),
        )
        val file = HarnessStorage.writeLatestReport(context, responseJson)
        Log.i(TAG, "poseidon2_smoke field=${field.wireValue} path=${file.absolutePath}")
        return JSONObject(responseJson)
    }

    companion object {
        private const val TAG = "ZkgpuHarnessTest"

        // R8 A/B knob values. `u32::MAX` on the Rust side; passed as Long
        // from Kotlin because Int.MAX_VALUE = 2^31-1 < 2^32-1.
        private const val R8_FORCE_ENABLED = 4_294_967_295L
        private const val R8_FORCE_DISABLED = 0L

        // log_n windows for the R8 A/B. Balanced row×col Four-Step
        // splits these to log_leaf {9, 10, 11}. log_n=24 → log_leaf=12
        // would cover the outer edge but risks OOM on BrowserStack
        // devices with ~4 GB RAM; omit for the initial sweep.
        private val R8_AB_LOG_N = intArrayOf(18, 20, 22)

        /**
         * Phase F.3.e: Poseidon2 benchmark batch sizes. Smaller than
         * the desktop ladder (CLI default is 1024/16384/65536/262144)
         * so the test completes in a few seconds on a mobile GPU. The
         * top end (65k permutations × 16-slot state × 4-byte limb =
         * ~4 MB for BabyBear, ~8 MB for Goldilocks) stays well within
         * every Adreno/Mali/Xclipse buffer budget in the cohort.
         */
        private val POSEIDON2_BENCH_LADDER = intArrayOf(1_024, 16_384, 65_536)
    }
}

/**
 * Read a string field that may be missing or JSON null, returning `null`
 * in either case. `JSONObject.optString(key)` returns the literal string
 * `"null"` for JSON-null values, which would silently misreport a missing
 * `stockham_tail_strategy` as the string `"null"` in logcat.
 */
private fun JSONObject.optStringOrNull(key: String): String? {
    if (!has(key) || isNull(key)) return null
    val value = optString(key)
    return value.takeIf { it.isNotBlank() }
}
