package io.github.ayushpatnaikgit.tellwell

import android.content.Context
import android.os.Build
import java.io.File

/**
 * Speech recognition on a Qualcomm NPU. Such phones download a quantised encoder (590 MB) instead of the CPU one,
 * and compile it once for their own NPU (a few minutes); the compiled copy then loads in about a second. If the NPU
 * fails on a phone, that is remembered for this app version and the phone uses the CPU encoder instead.
 */
object Npu {
    fun dir(context: Context) = File(Engine.modelsDir(context), "asr-npu")
    /** The quantised model as downloaded (graph and weights; the graph names the weights file). */
    fun model(context: Context) = File(dir(context), "enc-w8a16-1000.onnx")
    fun weights(context: Context) = File(dir(context), "enc-w8a16-1000.onnx.data")
    /** The compiled model for this phone's NPU: a small graph that names the binary next to it. */
    fun ctx(context: Context) = File(dir(context), "enc-ctx.onnx")
    private fun ctxBin(context: Context) = File(dir(context), "enc-ctx_qnn.bin")
    /** Compilation writes here, and the result moves next to the model only when complete. */
    fun compiling(context: Context) = File(dir(context), "compiling")

    fun compiled(context: Context) = ctx(context).isFile && ctxBin(context).isFile
    fun downloaded(context: Context) = model(context).isFile && weights(context).isFile

    /** A Qualcomm phone (Android 12+ reports the chip maker) whose NPU hasn't failed in this app version. */
    fun candidate(context: Context): Boolean =
        Build.VERSION.SDK_INT >= 31 && Build.SOC_MANUFACTURER.let { it.equals("QTI", true) || it.equals("Qualcomm", true) } &&
            !prefs(context).getBoolean(failedKey, false)

    /** After compiling: keep the compiled copy, drop the download it was made from. */
    fun finishCompile(context: Context): Boolean {
        val src = compiling(context)
        val ok = listOf("enc-ctx.onnx", "enc-ctx_qnn.bin").all { File(src, it).renameTo(File(dir(context), it)) }
        src.deleteRecursively()
        if (ok) { model(context).delete(); weights(context).delete(); prefs(context).edit().remove(attemptsKey).apply() }
        return ok
    }

    /**
     * Counts compile attempts. Android can stop a compile when memory runs short (it needs about 3 GB); after two
     * attempts that never finished, the phone uses the CPU instead of trying forever.
     */
    fun beginCompile(context: Context): Boolean {
        val p = prefs(context)
        val n = p.getInt(attemptsKey, 0)
        if (n >= 2) { markFailed(context); return false }
        p.edit().putInt(attemptsKey, n + 1).commit()
        return true
    }

    fun markFailed(context: Context) {
        prefs(context).edit().putBoolean(failedKey, true).apply()
        dir(context).deleteRecursively()
    }

    private val failedKey = "failed-${BuildConfig.VERSION_CODE}"
    private val attemptsKey = "compile-attempts-${BuildConfig.VERSION_CODE}"
    private fun prefs(context: Context) = context.applicationContext.getSharedPreferences("npu", Context.MODE_PRIVATE)
}
