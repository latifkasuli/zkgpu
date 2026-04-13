package org.zkgpu.harness

import android.content.Context
import java.io.File

object HarnessStorage {
    private const val REPORT_DIR = "zkgpu"
    private const val REPORT_NAME = "latest-report.json"

    fun writeLatestReport(context: Context, reportJson: String): File {
        val dir = File(context.filesDir, REPORT_DIR).apply { mkdirs() }
        val file = File(dir, REPORT_NAME)
        file.writeText(reportJson, Charsets.UTF_8)
        return file
    }
}
