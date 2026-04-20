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

/**
 * Caller-side override for the Stockham tail-phase selection. Mirrors
 * `zkgpu_report::StockhamTailOverride`. `Auto` (the default) lets the
 * Rust planner heuristic pick `LocalFusedR4` or `GlobalOnlyR4` from
 * device caps + `log_n`. `Local` and `Global` force the corresponding
 * strategy regardless of heuristic — used for forced A/B runs on
 * Xclipse / Mali / Browser, where the new `GlobalOnlyR4` is the
 * production default we want to measure.
 *
 * `wireValue == null` means "omit the field"; `#[serde(default)]` on
 * the Rust side then applies `Auto`.
 */
enum class TailChoice(val wireValue: String?) {
    Auto(null),
    Local("Local"),
    Global("Global");

    companion object {
        fun fromIntent(value: String?): TailChoice = when (value?.lowercase()) {
            "local" -> Local
            "global" -> Global
            else -> Auto
        }
    }
}

/**
 * Which prime field a Poseidon2 hash suite targets. Phase F.3.e.
 * Wire value matches the serde encoding of `zkgpu_report::Field`.
 */
enum class HashField(val wireValue: String) {
    BabyBear("BabyBear"),
    Goldilocks("Goldilocks"),
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
    fun presetRequestJson(
        suite: PresetSuite,
        family: FamilyChoice = FamilyChoice.Auto,
        tail: TailChoice = TailChoice.Auto,
    ): String {
        val obj = JSONObject()
            .put("suite", suite.wireName)
            .put("spec", JSONObject.NULL)
        if (family.wireValue != null) {
            obj.put("family_override", family.wireValue)
        }
        // Top-level override; the FFI layer copies it onto the spec it
        // builds from `suite`, so a forced `Local`/`Global` reaches the
        // planner regardless of which preset spec is materialized.
        if (tail.wireValue != null) {
            obj.put("stockham_tail_override", tail.wireValue)
        }
        return obj.toString()
    }

    fun crossoverBenchmarkRequestJson(
        family: FamilyChoice,
        tail: TailChoice = TailChoice.Auto,
    ): String = customBenchmarkRequestJson(
        logNRange = intArrayOf(18, 19, 20, 21, 22),
        family = family,
        tail = tail,
    )

    /**
     * Build a forced-A/B benchmark request with caller-picked log_n range
     * and caller-picked per-planner overrides.
     *
     * Used by the R8 A/B harness (Adreno/Mali/Xclipse): pass
     * `r8MaxLogLeafOverride = 0L` to force R8 leaves off, or
     * `r8MaxLogLeafOverride = Long.MAX_VALUE` to force them on, and run
     * both runs to measure the R8-vs-R4 delta on mobile.
     */
    fun customBenchmarkRequestJson(
        logNRange: IntArray,
        family: FamilyChoice,
        tail: TailChoice = TailChoice.Auto,
        r8MaxLogLeafOverride: Long? = null,
    ): String {
        val cases = JSONArray()
        for (logN in logNRange) {
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
        // Set on the spec directly so harnesses that POST a fully-built
        // spec (no `suite` shortcut) still carry the override through.
        if (tail.wireValue != null) {
            spec.put("stockham_tail_override", tail.wireValue)
        }
        if (r8MaxLogLeafOverride != null) {
            // Rust side reads u32. `u32::MAX` = 4_294_967_295, which exceeds
            // Int.MAX_VALUE — serialize as JSON number (Long backing) so the
            // "force-on" cap round-trips correctly.
            spec.put("r8_max_log_leaf_override", r8MaxLogLeafOverride)
        }

        val obj = JSONObject()
            .put("spec", spec)
        // Also set at the request top level so a future caller can flip
        // the knob without rebuilding the spec — matches the FFI contract.
        if (tail.wireValue != null) {
            obj.put("stockham_tail_override", tail.wireValue)
        }
        if (r8MaxLogLeafOverride != null) {
            obj.put("r8_max_log_leaf_override", r8MaxLogLeafOverride)
        }
        return obj.toString()
    }

    /**
     * Build a Poseidon2 hash-suite request. Phase F.3.e — the FFI
     * router dispatches on the presence of `hash_spec` in the request
     * (instead of `spec` / `suite`) and calls `run_hash_suite` under
     * the hood. `field` selects BabyBear or Goldilocks; both fields
     * route to their respective GPU Poseidon2 plan.
     *
     * `profile_gpu_timestamps = false` for every case — the Poseidon2
     * plans have no `execute_profiled` yet (the testkit rejects the
     * flag with a structured error). When profiled-execute lands in a
     * future F.3.* sub-phase, flip this to `true` in the same commit.
     */
    fun poseidon2BenchmarkRequestJson(
        permutationCounts: IntArray,
        field: HashField,
    ): String {
        val cases = JSONArray()
        for (num in permutationCounts) {
            cases.put(
                JSONObject()
                    .put("name", "android_poseidon2_n${num}")
                    .put("num_permutations", num)
                    .put("input", JSONObject().put("SplitMix64", JSONObject().put("seed", 1)))
                    .put("profile_gpu_timestamps", false)
                    .put("iterations", 5)
                    .put("warmup_iterations", 1)
            )
        }

        val hashSpec = JSONObject()
            .put("kind", "Benchmark")
            .put("cases", cases)
            .put("fail_fast", false)
            .put("algorithm", "Poseidon2")
            .put("field", field.wireValue)

        return JSONObject()
            .put("hash_spec", hashSpec)
            .toString()
    }

    /**
     * Shipped `poseidon2_smoke_suite` preset — 5 small cases covering
     * AllZeros / AllOnes / Sequential / SplitMix64 inputs and a
     * prime-17 batch that exercises the 2D-fold dispatch path. Good
     * first validation target on an unknown Android GPU.
     */
    fun poseidon2SmokeRequestJson(field: HashField): String {
        // Matches `zkgpu_report::poseidon2_smoke_suite()`. Inline here
        // because the Kotlin side has no Rust-bridge call for "give me
        // the shipped smoke spec as JSON"; easier to transcribe.
        // Unit-variant inputs (AllZeros / AllOnes / Sequential)
        // serialise as bare strings — serde's untagged-unit
        // convention. Only struct-like variants (e.g. SplitMix64 {
        // seed }) use the `{"Tag": {...}}` envelope. Phase F.3.e.2
        // initial draft wrapped AllZeros/AllOnes as `{"AllZeros": {}}`
        // which BrowserStack caught with `invalid type: map, expected
        // unit at line 1 column 401` — fixed here.
        val cases = JSONArray()
            .put(smokeCase("poseidon2_smoke_single", 1, "Sequential"))
            .put(smokeCase("poseidon2_smoke_batch17", 17, "Sequential"))
            .put(smokeCase("poseidon2_smoke_zeros", 8, "AllZeros"))
            .put(smokeCase("poseidon2_smoke_ones", 8, "AllOnes"))
            .put(
                smokeCase(
                    "poseidon2_smoke_rng",
                    32,
                    JSONObject().put(
                        "SplitMix64",
                        // Canonical shipped-preset seed
                        // `0xCAFE_BABE_DEAD_BEEF` matching
                        // `zkgpu_report::poseidon2_smoke_suite()` on
                        // the Rust side. The value is > 2^63 so a
                        // plain Kotlin `Long` literal would be
                        // negative and serde's `u64` deserializer
                        // would refuse it. Route through BigInteger
                        // instead: org.json writes BigInteger as a
                        // raw JSON number, so serde sees the correct
                        // unsigned value `14621396277992245487`.
                        JSONObject().put(
                            "seed",
                            java.math.BigInteger("CAFEBABEDEADBEEF", 16)
                        ),
                    ),
                )
            )

        val hashSpec = JSONObject()
            .put("kind", "Smoke")
            .put("cases", cases)
            .put("fail_fast", true)
            .put("algorithm", "Poseidon2")
            .put("field", field.wireValue)

        return JSONObject().put("hash_spec", hashSpec).toString()
    }

    private fun smokeCase(name: String, num: Int, input: Any): JSONObject =
        JSONObject()
            .put("name", name)
            .put("num_permutations", num)
            .put("input", input)
            .put("profile_gpu_timestamps", false)
            .put("iterations", 1)
            .put("warmup_iterations", 0)

    fun syntheticErrorJson(message: String): String =
        JSONObject()
            .put("ok", false)
            .put("report", JSONObject.NULL)
            .put("error", message)
            .toString()

    fun parseSummary(responseJson: String): HarnessSummary {
        val root = JSONObject(responseJson)
        // Phase F.3.e post-review (P3.2): pick whichever report body
        // is populated. The FFI router returns `report` for NTT
        // suites and `hash_report` for hash suites (see
        // HarnessRequest.hash_spec dispatch in zkgpu-ffi/src/json.rs).
        // parseSummary previously only read `report`, so callers of
        // poseidon2SmokeRequestJson / poseidon2BenchmarkRequestJson
        // got null suite/case/device fields even on successful runs.
        val report = root.optJSONObject("report") ?: root.optJSONObject("hash_report")
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
