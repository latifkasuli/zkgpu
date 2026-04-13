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
     * 5 iterations + 2 warmups with GPU profiling.
     * This exercises the four-step NTT path on capable hardware.
     */
    @Test
    fun crossoverBenchmarkCompletes() {
        val requestJson = HarnessJson.crossoverBenchmarkRequestJson(FamilyChoice.Auto)
        val responseJson = ZkgpuBridge.runRequestJson(requestJson)
        val file = HarnessStorage.writeLatestReport(context, responseJson)
        Log.i(TAG, "crossover benchmark path=${file.absolutePath}")

        val response = JSONObject(responseJson)
        assertTrue(response.optString("error"), response.optBoolean("ok", false))

        val report = response.getJSONObject("report")
        val summary = report.getJSONObject("summary")
        assertEquals(0, summary.getInt("failed_cases"))
        assertTrue("expected 10 crossover cases", summary.getInt("total_cases") == 10)

        // Log per-case timings
        val cases = report.getJSONArray("cases")
        for (i in 0 until cases.length()) {
            val c = cases.getJSONObject(i)
            val name = c.optString("name")
            val timings = c.optJSONObject("timings")
            val wallNs = timings?.optLong("wall_time_ns", -1) ?: -1L
            val gpuNs = timings?.optLong("gpu_total_ns", -1) ?: -1L
            val wallMs = if (wallNs > 0) wallNs / 1_000_000.0 else -1.0
            val gpuMs = if (gpuNs > 0) gpuNs / 1_000_000.0 else -1.0
            Log.i(TAG, "CROSSOVER $name: wall=${"%.2f".format(wallMs)}ms gpu=${"%.2f".format(gpuMs)}ms")
        }
    }

    private fun runPreset(suite: PresetSuite): JSONObject {
        val responseJson = ZkgpuBridge.runRequestJson(HarnessJson.presetRequestJson(suite))
        val file = HarnessStorage.writeLatestReport(context, responseJson)
        Log.i(TAG, "suite=${suite.wireName} path=${file.absolutePath}")
        return JSONObject(responseJson)
    }

    companion object {
        private const val TAG = "ZkgpuHarnessTest"
    }
}
