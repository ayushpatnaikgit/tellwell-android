package io.github.ayushpatnaikgit.tellwell

import android.content.Context
import android.system.Os
import com.k2fsa.sherpa.onnx.OfflineModelConfig
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig
import com.k2fsa.sherpa.onnx.OfflineTransducerModelConfig
import java.io.File
import java.util.concurrent.Executors

/** Speech recognition on the Qualcomm NPU (see cpp/tellwell_asr.cpp). */
object AsrNpu {
    /** "" when ready, else why not (no user content). */
    external fun load(libDir: String, ctxPath: String, modelPath: String, decoder: String, joiner: String, tokens: String): String
    /** The transcript, or null if the NPU failed. */
    external fun transcribe(samples: FloatArray): String?
}

/** llama.cpp bridge (see cpp/tellwell_llm.cpp). */
object LlmNative {
    external fun load(libDir: String, modelPath: String, prefix: String, devices: String, threads: Int): String
    external fun complete(suffix: String, maxTokens: Int, draftK: Int): String
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
    /** Speech recognition runs on the NPU (else on the CPU with sherpa-onnx). */
    @Volatile var asrOnNpu = false
        private set
    @Volatile private var cleanupOn = false

    fun modelsDir(context: Context) = File(context.filesDir, "models")

    /** Copy-ahead draft length for cleanup (0 = plain greedy). Same text either way, only faster. */
    @Volatile var draftK = 4
    @Volatile var draftOverridden = false

    /** Backends to try for cleanup, fastest first. Debug builds can override it to test one path. */
    @Volatile var devices = listOf("HTP0", "GPUOpenCL", "CPU")

    /** Speech model folders from earlier versions (0.1.0 used multilingual v3 in models/asr). */
    fun removeOldModels(context: Context) {
        File(modelsDir(context), "asr").takeIf { it.isDirectory }?.deleteRecursively()
    }
    private fun llmFile(context: Context) = ModelStore.llm(context)
    private fun cpuEncoder(context: Context) = ModelStore.cpuEncoder(context)

    /** Enough to dictate: decoder, joiner, tokens and cleanup, plus a speech encoder for the NPU or the CPU. */
    fun modelsPresent(context: Context): Boolean =
        listOf(ModelStore.decoder(context), ModelStore.joiner(context), ModelStore.tokens(context)).all { it.isFile } &&
            llmFile(context).isFile &&
            (cpuEncoder(context).isFile || (Npu.candidate(context) && (Npu.compiled(context) || Npu.downloaded(context))))

    /**
     * After the NPU encoder is downloaded on a phone already dictating on the CPU: compile it and switch to it.
     * [done] runs on the worker thread.
     */
    fun upgradeToNpu(context: Context, done: () -> Unit = {}) {
        val app = context.applicationContext
        worker.execute {
            if (ready && !asrOnNpu && tryNpu(app)) {
                asr?.release()
                asr = null
                status = status.replace("speech on the CPU", "speech on the NPU")
            }
            done()
        }
    }

    /**
     * Loads the NPU encoder, compiling it for this phone first if needed (a few minutes, once), and checks it on a
     * second of silence. On failure, remembers it and falls back to the CPU. Runs on the worker thread.
     */
    private fun tryNpu(context: Context): Boolean {
        if (!Npu.candidate(context) || !(Npu.compiled(context) || Npu.downloaded(context))) return false
        val compile = !Npu.compiled(context)
        if (compile) {
            if (!Npu.beginCompile(context)) return false
            status = "Getting ready for this phone's NPU: keep TellWell open for about a minute (once)"
            Npu.compiling(context).apply { deleteRecursively(); mkdirs() }
        }
        val ctx = if (compile) java.io.File(Npu.compiling(context), "enc-ctx.onnx") else Npu.ctx(context)
        var err: String
        try {
            Os.setenv("ADSP_LIBRARY_PATH", context.applicationInfo.nativeLibraryDir, true)
            System.loadLibrary("tellwell_asr")
            err = AsrNpu.load(context.applicationInfo.nativeLibraryDir, ctx.absolutePath, Npu.model(context).absolutePath,
                ModelStore.decoder(context).absolutePath, ModelStore.joiner(context).absolutePath, ModelStore.tokens(context).absolutePath)
            if (err.isEmpty() && compile && !Npu.finishCompile(context)) err = "could not save the compiled model"
            if (err.isEmpty() && AsrNpu.transcribe(FloatArray(16000)) == null) err = "self-check failed"
        } catch (t: Throwable) {
            err = t.javaClass.simpleName
        }
        if (err.isNotEmpty()) {
            android.util.Log.w("TellWellASR", "NPU speech unavailable: $err")
            Npu.markFailed(context)
            return false
        }
        asrOnNpu = true
        // The CPU encoder (650 MB) isn't needed any more; it is fetched again only if the NPU ever fails.
        if (cpuEncoder(context).isFile) ModelStore.removeCpuEncoder(context)
        return true
    }

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
        removeOldModels(context)
        // Speech recognition first: on the NPU where there is one (about 8x faster), else on the CPU (with ONNX Runtime
        // picking instructions at runtime), which works on any arm64 phone. Without it there is nothing to type.
        if (!tryNpu(context)) {
            if (!cpuEncoder(context).isFile) {
                // The NPU didn't work here and this phone never downloaded the CPU encoder: setup offers it now.
                status = "Models missing"
                return
            }
            status = "Loading"
        }
        if (!asrOnNpu) try {
            asr = OfflineRecognizer(
                config = OfflineRecognizerConfig(
                    modelConfig = OfflineModelConfig(
                        transducer = OfflineTransducerModelConfig(
                            encoder = cpuEncoder(context).absolutePath,
                            decoder = ModelStore.decoder(context).absolutePath,
                            joiner = ModelStore.joiner(context).absolutePath,
                        ),
                        tokens = ModelStore.tokens(context).absolutePath,
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
            for (d in devices) {
                val got = LlmNative.load(context.applicationInfo.nativeLibraryDir, llmFile(context).absolutePath, Prompt.PREFIX, d, 4)
                if (got.isNotEmpty() && canaryPasses()) { device = got; break }
            }
        } catch (t: Throwable) {
            android.util.Log.w("TellWellLLM", "cleanup unavailable: ${t.javaClass.simpleName}")
        }
        cleanupOn = device.isNotEmpty()
        // Copy-ahead decoding checks several tokens per pass. Measured on a Snapdragon 7 Gen 4: 20% faster on the CPU,
        // but slower on the NPU and GPU, whose small-batch passes cost more than they save. Debug runs can override.
        if (!draftOverridden) draftK = if (device == "CPU") 4 else 0
        ready = true
        val speechOn = if (asrOnNpu) "speech on the NPU" else "speech on the CPU"
        status = if (cleanupOn) "Ready ($speechOn, cleanup on ${when (device) { "HTP0" -> "the NPU"; "GPUOpenCL" -> "the GPU"; else -> "the CPU" }})"
            else "Ready ($speechOn; cleanup isn't available on this phone)"
    }

    /** Transcribes and cleans up 16 kHz mono audio. [done] runs on the worker thread. */
    fun process(samples: FloatArray, done: (Result) -> Unit) {
        worker.execute {
            val recognizer = asr
            // Runs after any load queued before it, so a clip recorded during loading is still processed.
            if (!ready || (recognizer == null && !asrOnNpu) || samples.size < 1600) {
                done(Result("", 0, 0, ""))
                return@execute
            }
            val t0 = System.nanoTime()
            // The CPU recogniser isn't loaded when the NPU is used; a failed NPU run then types nothing.
            val raw = ((if (asrOnNpu) AsrNpu.transcribe(samples) else null) ?: recognizer?.let { r ->
                val stream = r.createStream()
                stream.acceptWaveform(samples, 16000)
                r.decode(stream)
                r.getResult(stream).text.also { stream.release() }
            } ?: "").trim()
            val t1 = System.nanoTime()
            val text = if (raw.isEmpty()) "" else if (cleanupOn) cleanup(raw) else raw
            val t2 = System.nanoTime()
            done(Result(text, (t1 - t0) / 1_000_000, (t2 - t1) / 1_000_000, raw))
        }
    }

    /** A fixed, invented dictation whose cleanup is known. Garbage output means this backend can't be trusted. */
    private fun canaryPasses(): Boolean {
        val out = LlmNative.complete(Prompt.suffix("um so lets meet at five no wait six on thursday"), 40, 0)
        val ok = out.contains("Thursday") && (out.contains("6") || out.contains("six", ignoreCase = true)) &&
            !out.contains("five", ignoreCase = true) && out.all { it.code < 0x250 }
        android.util.Log.i("TellWellLLM", "self-check on ${LlmNative.device()}: ${if (ok) "passed" else "failed"}")
        return ok
    }

    private fun cleanup(raw: String): String {
        val out = LlmNative.complete(Prompt.suffix(raw), 160, draftK).trim()
        if (out == "EMPTY") return ""
        // Guard: a cleanup should never be empty for real speech or much longer than what was said.
        val rawWords = raw.split(Regex("\\s+")).size
        val outWords = out.split(Regex("\\s+")).size
        if (out.isEmpty() || outWords > rawWords * 2 + 8) return raw
        return out
    }
}
