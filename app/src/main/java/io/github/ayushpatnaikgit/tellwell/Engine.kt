package io.github.ayushpatnaikgit.tellwell

import android.content.Context
import android.system.Os
import com.k2fsa.sherpa.onnx.OfflineModelConfig
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig
import com.k2fsa.sherpa.onnx.OfflineTransducerModelConfig
import java.io.File
import java.util.concurrent.Executors

/** llama.cpp bridge (see cpp/tellwell_llm.cpp). */
object LlmNative {
    external fun load(libDir: String, modelPath: String, prefix: String, devices: String, threads: Int): String
    external fun complete(suffix: String, maxTokens: Int): String
    external fun device(): String
}

/**
 * Speech recognition (Parakeet via sherpa-onnx) and cleanup (S1-mini via llama.cpp), loaded once and kept in memory.
 * All work runs on one background thread. Nothing is written to disk or logged.
 */
object Engine {
    data class Result(val text: String, val asrMs: Long, val cleanupMs: Long, val raw: String)

    private val worker = Executors.newSingleThreadExecutor { r -> Thread(r, "tellwell-engine") }
    @Volatile var ready = false
        private set
    @Volatile var status = "Not loaded"
        private set
    private var asr: OfflineRecognizer? = null
    @Volatile private var cleanupOn = false

    fun modelsDir(context: Context) = File(context.filesDir, "models")
    private fun asrDir(context: Context) = File(modelsDir(context), "asr")
    private fun llmFile(context: Context) = File(modelsDir(context), "s1-mini-freeflow-cleanup-q4_0.gguf")

    fun modelsPresent(context: Context): Boolean =
        listOf("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt").all { File(asrDir(context), it).isFile } &&
            llmFile(context).isFile

    /** Loads both models on the worker thread if needed; [done] runs on the worker thread. */
    fun ensureLoaded(context: Context, done: (String) -> Unit = {}) {
        val app = context.applicationContext
        worker.execute {
            if (!ready) load(app)
            done(status)
        }
    }

    private fun load(context: Context) {
        if (!modelsPresent(context)) {
            status = "Models missing"
            return
        }
        status = "Loading"
        // Speech recognition first. It runs on the CPU (with ONNX Runtime picking instructions at runtime),
        // so it works on any arm64 phone; without it there is nothing to type.
        try {
            val dir = asrDir(context).absolutePath
            asr = OfflineRecognizer(
                config = OfflineRecognizerConfig(
                    modelConfig = OfflineModelConfig(
                        transducer = OfflineTransducerModelConfig(
                            encoder = "$dir/encoder.int8.onnx",
                            decoder = "$dir/decoder.int8.onnx",
                            joiner = "$dir/joiner.int8.onnx",
                        ),
                        tokens = "$dir/tokens.txt",
                        modelType = "nemo_transducer",
                        numThreads = 2,
                    ),
                ),
            )
        } catch (t: Throwable) {
            status = "Speech model failed to load (${t.javaClass.simpleName})"
            return
        }
        // Cleanup is optional: if no backend works on this phone, TellWell types the raw transcript instead.
        var device = ""
        try {
            // The NPU backend loads its DSP code from the app's native library directory.
            Os.setenv("ADSP_LIBRARY_PATH", context.applicationInfo.nativeLibraryDir, true)
            System.loadLibrary("tellwell_llm")
            // Fastest first: the NPU, then the GPU, then the CPU. Each must pass a quick correctness check,
            // since a backend can run but compute wrong results on a chip it doesn't fully support.
            for (d in listOf("HTP0", "GPUOpenCL", "CPU")) {
                val got = LlmNative.load(context.applicationInfo.nativeLibraryDir, llmFile(context).absolutePath, Prompt.PREFIX, d, 4)
                if (got.isNotEmpty() && canaryPasses()) { device = got; break }
            }
        } catch (t: Throwable) {
            android.util.Log.w("TellWellLLM", "cleanup unavailable: ${t.javaClass.simpleName}")
        }
        cleanupOn = device.isNotEmpty()
        ready = true
        status = if (cleanupOn) "Ready (cleanup on ${when (device) { "HTP0" -> "the NPU"; "GPUOpenCL" -> "the GPU"; else -> "the CPU" }})"
            else "Ready (speech only: cleanup isn't available on this phone)"
    }

    /** Transcribes and cleans up 16 kHz mono audio. [done] runs on the worker thread. */
    fun process(samples: FloatArray, done: (Result) -> Unit) {
        worker.execute {
            val recognizer = asr
            // Runs after any load queued before it, so a clip recorded during loading is still processed.
            if (!ready || recognizer == null || samples.size < 1600) {
                done(Result("", 0, 0, ""))
                return@execute
            }
            val t0 = System.nanoTime()
            val stream = recognizer.createStream()
            stream.acceptWaveform(samples, 16000)
            recognizer.decode(stream)
            val raw = recognizer.getResult(stream).text.trim()
            stream.release()
            val t1 = System.nanoTime()
            val text = if (raw.isEmpty()) "" else if (cleanupOn) cleanup(raw) else raw
            val t2 = System.nanoTime()
            done(Result(text, (t1 - t0) / 1_000_000, (t2 - t1) / 1_000_000, raw))
        }
    }

    /** A fixed, invented dictation whose cleanup is known. Garbage output means this backend can't be trusted. */
    private fun canaryPasses(): Boolean {
        val out = LlmNative.complete(Prompt.suffix("um so lets meet at five no wait six on thursday"), 40)
        val ok = out.contains("Thursday") && (out.contains("6") || out.contains("six", ignoreCase = true)) &&
            !out.contains("five", ignoreCase = true) && out.all { it.code < 0x250 }
        android.util.Log.i("TellWellLLM", "self-check on ${LlmNative.device()}: ${if (ok) "passed" else "failed"}")
        return ok
    }

    private fun cleanup(raw: String): String {
        val out = LlmNative.complete(Prompt.suffix(raw), 160).trim()
        if (out == "EMPTY") return ""
        // Guard: a cleanup should never be empty for real speech or much longer than what was said.
        val rawWords = raw.split(Regex("\\s+")).size
        val outWords = out.split(Regex("\\s+")).size
        if (out.isEmpty() || outWords > rawWords * 2 + 8) return raw
        return out
    }
}
