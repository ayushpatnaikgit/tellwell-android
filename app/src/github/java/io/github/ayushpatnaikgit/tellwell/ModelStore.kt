package io.github.ayushpatnaikgit.tellwell

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import java.io.File

/**
 * Where the model files are and how this build gets them. GitHub build: TellWell downloads them itself
 * (ModelDownloader, the only network code) into its private files folder. The Play build has its own ModelStore
 * that gets them from Google Play as asset packs, without the internet permission.
 */
object ModelStore {
    private fun dir(c: Context) = Engine.modelsDir(c)
    fun decoder(c: Context) = File(dir(c), "asr-v2/decoder.int8.onnx")
    fun joiner(c: Context) = File(dir(c), "asr-v2/joiner.int8.onnx")
    fun tokens(c: Context) = File(dir(c), "asr-v2/tokens.txt")
    fun llm(c: Context) = File(dir(c), "s1-mini-freeflow-cleanup-q4_0.gguf")
    fun cpuEncoder(c: Context) = File(dir(c), "asr-v2/encoder.int8.onnx")
    /** The NPU encoder's graph; its weights file (`.data`) sits next to it. */
    fun npuModel(c: Context) = File(dir(c), "asr-npu/enc-w8a16-1000.onnx")

    fun removeCpuEncoder(c: Context) { cpuEncoder(c).delete() }
    fun removeNpuModel(c: Context) { npuModel(c).delete(); File(npuModel(c).path + ".data").delete() }

    /** Text for the setup screen's privacy line. */
    val SOURCE_NOTE = if (BuildConfig.MULTILINGUAL)
        "What you say never leaves this phone. The internet is only used to download the models from GitHub."
    else "What you say never leaves this phone. The internet is only used once, to download the models from GitHub."

    val busy get() = ModelDownloader.running
    fun missingBytes(c: Context): Long = ModelDownloader.missing(c).sumOf { it.size }

    /** Progress for the setup screen: (bytes done, total, error, finished). */
    var listener: ((Long, Long, String?, Boolean) -> Unit)?
        get() = ModelDownloader.listener
        set(value) { ModelDownloader.listener = value }

    /** Starts the download, or pauses it if it is running. */
    fun start(activity: Activity) {
        if (ModelDownloader.running) {
            activity.startService(Intent(activity, DownloadService::class.java).setAction(DownloadService.ACTION_PAUSE))
            return
        }
        // The progress notification needs this on Android 13+; the download works without it.
        if (activity.checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
            activity.requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), 2)
        }
        activity.startForegroundService(Intent(activity, DownloadService::class.java))
    }

    const val CAN_PAUSE = true

    fun progressText(done: Long, total: Long) = "Downloading… ${done / 1_000_000} of ${total / 1_000_000} MB"
    /** The user starts the download (it's about 1 GB). */
    const val AUTO_FETCH = false
    const val STEP_TITLE = "Download the models"
    fun missingNote(mb: Long) = "About ${(mb + 50) / 100 * 100} MB, once. Use Wi-Fi."
    fun upgradeNote(mb: Long) = "Faster speech for this phone's NPU: about $mb MB more, once."
}
