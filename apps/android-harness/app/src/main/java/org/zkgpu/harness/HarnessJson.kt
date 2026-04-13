package org.zkgpu.harness

import org.json.JSONArray
import org.json.JSONObject

enum class PresetSuite(val wireName: String) {
    Smoke("Smoke"),
    Validation("Validation"),
    Benchmark("Benchmark"),
}

enum class FamilyChoice(val wireValue: String?) {
    Auto(null),
    Stockham("Stockham"),
    FourStep("FourStep");

    companion object {
        fun fromSpinnerIndex(index: Int): FamilyChoice = when (index) {
            1 -> Stockham
            2 -> FourStep
            else -> Auto
        }
    }
}

data class HarnessSummary(
    val ok: Boolean,
    val error: String?,
    val suite: String?,
    val totalCases: Int?,
    val passedCases: Int?,
    val failedCases: Int?,
    val deviceName: String?,
    val backend: String?,
    val tier: String?,
)

object HarnessJson {
    fun presetRequestJson(suite: PresetSuite, family: FamilyChoice = FamilyChoice.Auto): String {
        val obj = JSONObject()
            .put("suite", suite.wireName)
            .put("spec", JSONObject.NULL)
        if (family.wireValue != null) {
            obj.put("family_override", family.wireValue)
        }
        return obj.toString()
    }

    fun crossoverBenchmarkRequestJson(family: FamilyChoice): String {
        val cases = JSONArray()
        for (logN in intArrayOf(18, 19, 20, 21, 22)) {
            for (dir in arrayOf("Forward", "Inverse")) {
                val name = "${dir.lowercase()}_log${logN}"
                cases.put(
                    JSONObject()
                        .put("name", name)
                        .put("log_n", logN)
                        .put("direction", dir)
                        .put("input", "Sequential")
                        .put("profile_gpu_timestamps", true)
                        .put("iterations", 5)
                        .put("warmup_iterations", 2)
                )
            }
        }

        val spec = JSONObject()
            .put("kind", "Benchmark")
            .put("cases", cases)
            .put("fail_fast", false)
            .put("family_override", family.wireValue ?: "Auto")

        val obj = JSONObject()
            .put("spec", spec)
        return obj.toString()
    }

    fun syntheticErrorJson(message: String): String =
        JSONObject()
            .put("ok", false)
            .put("report", JSONObject.NULL)
            .put("error", message)
            .toString()

    fun parseSummary(responseJson: String): HarnessSummary {
        val root = JSONObject(responseJson)
        val report = root.optJSONObject("report")
        val summary = report?.optJSONObject("summary")
        val device = report?.optJSONObject("device")
        return HarnessSummary(
            ok = root.optBoolean("ok", false),
            error = root.optNullableString("error"),
            suite = report?.optNullableString("suite"),
            totalCases = summary?.optIntOrNull("total_cases"),
            passedCases = summary?.optIntOrNull("passed_cases"),
            failedCases = summary?.optIntOrNull("failed_cases"),
            deviceName = device?.optNullableString("name"),
            backend = device?.optNullableString("backend"),
            tier = device?.optNullableString("tier"),
        )
    }

    fun summaryLine(summary: HarnessSummary): String {
        if (!summary.ok) {
            return buildString {
                append("FAILED")
                summary.error?.let {
                    append('\n')
                    append(it)
                }
            }
        }

        return buildString {
            append(summary.suite ?: "Unknown")
            append(" suite passed")
            append('\n')
            append("cases: ")
            append(summary.passedCases ?: 0)
            append('/')
            append(summary.totalCases ?: 0)
            summary.failedCases?.let {
                append("  failed: ")
                append(it)
            }
            append('\n')
            append("device: ")
            append(summary.deviceName ?: "unknown")
            append('\n')
            append("backend: ")
            append(summary.backend ?: "unknown")
            append("  tier: ")
            append(summary.tier ?: "unknown")
        }
    }

    fun versionLine(versionJson: String): String {
        val root = JSONObject(versionJson)
        val error = root.optNullableString("error")
        return if (error != null) {
            "Rust bridge unavailable: $error"
        } else {
            "${root.optString("crate_name", "zkgpu-ffi")} ${root.optString("version", "unknown")} (ffi ${root.optInt("ffi_api_version", 0)})"
        }
    }

    private fun JSONObject.optNullableString(key: String): String? {
        if (!has(key) || isNull(key)) {
            return null
        }
        return optString(key).takeIf { it.isNotBlank() }
    }

    private fun JSONObject.optIntOrNull(key: String): Int? {
        if (!has(key) || isNull(key)) {
            return null
        }
        return optInt(key)
    }
}
