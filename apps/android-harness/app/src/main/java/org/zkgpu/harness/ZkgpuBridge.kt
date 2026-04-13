package org.zkgpu.harness

object ZkgpuBridge {
    init {
        System.loadLibrary("zkgpu_android_bridge")
    }

    @JvmStatic
    external fun runRequestJson(requestJson: String): String

    @JvmStatic
    external fun getVersionJson(): String
}
