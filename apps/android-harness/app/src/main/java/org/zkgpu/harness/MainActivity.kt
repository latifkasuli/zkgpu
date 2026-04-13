package org.zkgpu.harness

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.Spinner
import android.widget.TextView
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : Activity() {
    private val worker: ExecutorService = Executors.newSingleThreadExecutor()

    private lateinit var versionText: TextView
    private lateinit var statusText: TextView
    private lateinit var outputPathText: TextView
    private lateinit var rawResponseText: TextView
    private lateinit var familySpinner: Spinner
    private lateinit var runSmokeButton: Button
    private lateinit var runValidationButton: Button
    private lateinit var runBenchmarkButton: Button
    private lateinit var runCrossoverButton: Button
    private lateinit var refreshVersionButton: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        versionText = findViewById(R.id.versionText)
        statusText = findViewById(R.id.statusText)
        outputPathText = findViewById(R.id.outputPathText)
        rawResponseText = findViewById(R.id.rawResponseText)
        familySpinner = findViewById(R.id.familySpinner)
        runSmokeButton = findViewById(R.id.runSmokeButton)
        runValidationButton = findViewById(R.id.runValidationButton)
        runBenchmarkButton = findViewById(R.id.runBenchmarkButton)
        runCrossoverButton = findViewById(R.id.runCrossoverButton)
        refreshVersionButton = findViewById(R.id.refreshVersionButton)

        runSmokeButton.setOnClickListener { runSuite(PresetSuite.Smoke) }
        runValidationButton.setOnClickListener { runSuite(PresetSuite.Validation) }
        runBenchmarkButton.setOnClickListener { runSuite(PresetSuite.Benchmark) }
        runCrossoverButton.setOnClickListener { runCrossoverBenchmark() }
        refreshVersionButton.setOnClickListener { refreshVersion() }

        refreshVersion()
        if (savedInstanceState == null) {
            handleLaunchIntent(intent)
        }
    }

    override fun onNewIntent(intent: Intent?) {
        super.onNewIntent(intent)
        intent?.let { handleLaunchIntent(it) }
    }

    private fun handleLaunchIntent(intent: Intent) {
        val action = intent.getStringExtra("action") ?: return
        intent.removeExtra("action")

        val family = when (intent.getStringExtra("family")?.lowercase()) {
            "stockham" -> FamilyChoice.Stockham
            "four-step", "fourstep", "four_step" -> FamilyChoice.FourStep
            else -> FamilyChoice.Auto
        }

        when (action.lowercase()) {
            "smoke" -> runSuiteWithFamily(PresetSuite.Smoke, family)
            "validation" -> runSuiteWithFamily(PresetSuite.Validation, family)
            "benchmark" -> runSuiteWithFamily(PresetSuite.Benchmark, family)
            "crossover" -> runCrossoverWithFamily(family)
        }
    }

    override fun onDestroy() {
        worker.shutdownNow()
        super.onDestroy()
    }

    private fun selectedFamily(): FamilyChoice =
        FamilyChoice.fromSpinnerIndex(familySpinner.selectedItemPosition)

    private fun refreshVersion() {
        versionText.text = getString(R.string.version_loading)
        worker.execute {
            val versionLine = try {
                HarnessJson.versionLine(ZkgpuBridge.getVersionJson())
            } catch (t: Throwable) {
                "Rust bridge unavailable: ${t.message ?: t::class.java.simpleName}"
            }
            runOnUiThread {
                versionText.text = versionLine
            }
        }
    }

    private fun runSuite(suite: PresetSuite) {
        val family = selectedFamily()
        val familyLabel = if (family == FamilyChoice.Auto) "" else " [${family.wireValue}]"
        setButtonsEnabled(false)
        statusText.text = "Running ${suite.wireName}$familyLabel suite..."
        worker.execute {
            val responseJson = try {
                ZkgpuBridge.runRequestJson(HarnessJson.presetRequestJson(suite, family))
            } catch (t: Throwable) {
                HarnessJson.syntheticErrorJson(
                    "JNI bridge failure: ${t.message ?: t::class.java.simpleName}"
                )
            }
            val summary = HarnessJson.parseSummary(responseJson)
            val reportFile = HarnessStorage.writeLatestReport(this, responseJson)
            logResult("${suite.wireName}$familyLabel", summary, reportFile)
            runOnUiThread {
                statusText.text = HarnessJson.summaryLine(summary)
                outputPathText.text = getString(R.string.output_path_value, reportFile.absolutePath)
                rawResponseText.text = responseJson
                setButtonsEnabled(true)
            }
        }
    }

    private fun runCrossoverBenchmark() {
        val family = selectedFamily()
        val familyLabel = if (family == FamilyChoice.Auto) "auto" else family.wireValue ?: "auto"
        setButtonsEnabled(false)
        statusText.text = "Running crossover benchmark [$familyLabel] (2^18 - 2^22, fwd+inv)..."
        worker.execute {
            val responseJson = try {
                ZkgpuBridge.runRequestJson(HarnessJson.crossoverBenchmarkRequestJson(family))
            } catch (t: Throwable) {
                HarnessJson.syntheticErrorJson(
                    "JNI bridge failure: ${t.message ?: t::class.java.simpleName}"
                )
            }
            val summary = HarnessJson.parseSummary(responseJson)
            val reportFile = HarnessStorage.writeLatestReport(this, responseJson)
            logResult("crossover[$familyLabel]", summary, reportFile)
            runOnUiThread {
                statusText.text = HarnessJson.summaryLine(summary)
                outputPathText.text = getString(R.string.output_path_value, reportFile.absolutePath)
                rawResponseText.text = responseJson
                setButtonsEnabled(true)
            }
        }
    }

    private fun runSuiteWithFamily(suite: PresetSuite, family: FamilyChoice) {
        val familyLabel = if (family == FamilyChoice.Auto) "" else " [${family.wireValue}]"
        setButtonsEnabled(false)
        statusText.text = "Running ${suite.wireName}$familyLabel suite (intent)..."
        worker.execute {
            val responseJson = try {
                ZkgpuBridge.runRequestJson(HarnessJson.presetRequestJson(suite, family))
            } catch (t: Throwable) {
                HarnessJson.syntheticErrorJson(
                    "JNI bridge failure: ${t.message ?: t::class.java.simpleName}"
                )
            }
            val summary = HarnessJson.parseSummary(responseJson)
            val reportFile = HarnessStorage.writeLatestReport(this, responseJson)
            logResult("${suite.wireName}$familyLabel", summary, reportFile)
            runOnUiThread {
                statusText.text = HarnessJson.summaryLine(summary)
                outputPathText.text = getString(R.string.output_path_value, reportFile.absolutePath)
                rawResponseText.text = responseJson
                setButtonsEnabled(true)
            }
        }
    }

    private fun runCrossoverWithFamily(family: FamilyChoice) {
        val familyLabel = if (family == FamilyChoice.Auto) "auto" else family.wireValue ?: "auto"
        setButtonsEnabled(false)
        statusText.text = "Running crossover benchmark [$familyLabel] (intent)..."
        worker.execute {
            val responseJson = try {
                ZkgpuBridge.runRequestJson(HarnessJson.crossoverBenchmarkRequestJson(family))
            } catch (t: Throwable) {
                HarnessJson.syntheticErrorJson(
                    "JNI bridge failure: ${t.message ?: t::class.java.simpleName}"
                )
            }
            val summary = HarnessJson.parseSummary(responseJson)
            val reportFile = HarnessStorage.writeLatestReport(this, responseJson)
            logResult("crossover[$familyLabel]", summary, reportFile)
            runOnUiThread {
                statusText.text = HarnessJson.summaryLine(summary)
                outputPathText.text = getString(R.string.output_path_value, reportFile.absolutePath)
                rawResponseText.text = responseJson
                setButtonsEnabled(true)
            }
        }
    }

    private fun setButtonsEnabled(enabled: Boolean) {
        runSmokeButton.isEnabled = enabled
        runValidationButton.isEnabled = enabled
        runBenchmarkButton.isEnabled = enabled
        runCrossoverButton.isEnabled = enabled
        refreshVersionButton.isEnabled = enabled
    }

    private fun logResult(label: String, summary: HarnessSummary, reportFile: File) {
        Log.i(
            TAG,
            "suite=$label ok=${summary.ok} backend=${summary.backend ?: "unknown"} " +
                "tier=${summary.tier ?: "unknown"} path=${reportFile.absolutePath}"
        )
    }

    companion object {
        private const val TAG = "ZkgpuHarness"
    }
}
