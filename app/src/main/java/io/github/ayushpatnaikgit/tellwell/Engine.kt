package io.github.ayushpatnaikgit.tellwell

import android.content.Context
import android.system.Os
import com.k2fsa.sherpa.onnx.OfflineModelConfig
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig
import com.k2fsa.sherpa.onnx.OfflineTransducerModelConfig
import java.io.File
import java.util.concurrent.Executors

/** Speech recognition on the Qualcomm NPU, or the same decoding with the CPU encoder (see cpp/tellwell_asr.cpp). */
object AsrNpu {
    /** "" when ready, else why not (no user content). */
    external fun load(libDir: String, ctxPath: String, modelPath: String, decoder: String, joiner: String, tokens: String): String
    /** The CPU encoder, decoded the same way (phones without a usable NPU). "" when ready. */
    external fun loadCpu(libDir: String, encoder: String, decoder: String, joiner: String, tokens: String): String
    /** The English word list that keeps name correction from replacing real words. "" when loaded. */
    external fun loadDictionary(path: String): String
    /** Contact name words to consider (empty when contacts are off). Held in memory only. */
    external fun setContacts(names: Array<String>)
    /** The transcript, or null if recognition failed. [context]: text around the cursor, whose names are favoured. */
    external fun transcribe(samples: FloatArray, context: String): String?
}

/** llama.cpp bridge (see cpp/tellwell_llm.cpp). Slot 0: English cleanup (S1-mini); slot 1: Hindi (Gemma 4 E2B). */
object LlmNative {
    external fun load(slot: Int, libDir: String, modelPath: String, prefix: String, devices: String, threads: Int): String
    external fun complete(slot: Int, suffix: String, maxTokens: Int, draftK: Int): String
    external fun device(slot: Int): String
    external fun unload(slot: Int)
}

/** Hindi speech recognition: IndicConformer Hindi on the CPU (see cpp/tellwell_hindi_asr.cpp). */
object AsrHindi {
    /** "" when ready, else why not (no user content). */
    external fun load(libDir: String, model: String, tokens: String, threads: Int): String
    /** The Devanagari transcript, or null if recognition failed. */
    external fun transcribe(samples: FloatArray): String?
    external fun unload()
}

/**
 * Speech recognition and cleanup, loaded once and kept in memory. English: Parakeet (NPU, or the CPU) and S1-mini.
 * Hindi, if the user added it: IndicConformer and the Hindi Gemma 4 E2B, both on the CPU. All work runs on one
 * background thread. Nothing is written to disk or logged.
 */
object Engine {
    data class Result(val text: String, val asrMs: Long, val cleanupMs: Long, val raw: String)

    private val worker = Executors.newSingleThreadExecutor { r -> Thread(r, "tellwell-engine") }
    @Volatile var ready = false
        private set
    @Volatile var status = "Not loaded"
        private set
    private var asr: OfflineRecognizer? = null
    /**
     * Compiling the speech model for this phone's NPU (about a minute, once). Dictation waits until it's done, so the
     * keyboard says so instead of listening.
     */
    @Volatile var compiling = false
        private set

    /** Compiling now, or about to (an NPU phone whose downloaded model hasn't been compiled yet). */
    fun preparing(context: Context): Boolean =
        compiling || (!ready && Npu.candidate(context) && !Npu.compiled(context) && Npu.downloaded(context))
    /** Speech recognition runs on the NPU. */
    @Volatile var asrOnNpu = false
        private set
    /** Speech recognition runs in TellWell's own decoder (NPU or CPU encoder), with names; else sherpa-onnx. */
    @Volatile private var asrNative = false
    @Volatile private var cleanupOn = false
    /** Hindi speech and cleanup are loaded (Hindi is offered only when both work). */
    @Volatile private var hindiOn = false
    /** Hindi is being loaded (after English, so English dictation is ready first). */
    @Volatile var hindiLoading = false
        private set

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
                asrNative = true
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
            compiling = true
        }
        val ctx = if (compile) java.io.File(Npu.compiling(context), "enc-ctx.onnx") else Npu.ctx(context)
        var err: String
        try {
            Os.setenv("ADSP_LIBRARY_PATH", context.applicationInfo.nativeLibraryDir, true)
            System.loadLibrary("tellwell_asr")
            err = AsrNpu.load(context.applicationInfo.nativeLibraryDir, ctx.absolutePath, Npu.model(context).absolutePath,
                ModelStore.decoder(context).absolutePath, ModelStore.joiner(context).absolutePath, ModelStore.tokens(context).absolutePath)
            if (err.isEmpty() && compile && !Npu.finishCompile(context)) err = "could not save the compiled model"
            if (err.isEmpty() && AsrNpu.transcribe(FloatArray(16000), "") == null) err = "self-check failed"
        } catch (t: Throwable) {
            err = t.javaClass.simpleName
        } finally {
            compiling = false
        }
        if (err.isNotEmpty()) {
            android.util.Log.w("TellWellASR", "NPU speech unavailable: $err")
            Npu.markFailed(context)
            return false
        }
        asrOnNpu = true
        asrNative = true
        // The CPU encoder (650 MB) isn't needed any more; it is fetched again only if the NPU ever fails.
        if (cpuEncoder(context).isFile) ModelStore.removeCpuEncoder(context)
        return true
    }

    /** The CPU encoder in TellWell's own decoder, so names work without an NPU too. Runs on the worker thread. */
    private fun tryNativeCpu(context: Context): Boolean {
        if (!cpuEncoder(context).isFile) return false
        return try {
            System.loadLibrary("tellwell_asr")
            val err = AsrNpu.loadCpu(context.applicationInfo.nativeLibraryDir, cpuEncoder(context).absolutePath,
                ModelStore.decoder(context).absolutePath, ModelStore.joiner(context).absolutePath, ModelStore.tokens(context).absolutePath)
            val ok = err.isEmpty() && AsrNpu.transcribe(FloatArray(16000), "") != null
            if (!ok) android.util.Log.w("TellWellASR", "CPU speech in TellWell's decoder unavailable: $err")
            ok
        } catch (t: Throwable) {
            android.util.Log.w("TellWellASR", "CPU speech in TellWell's decoder unavailable: ${t.javaClass.simpleName}")
            false
        }
    }

    /** The English word list ships in the app; the native side reads it from a file. Runs on the worker thread. */
    private fun loadDictionary(context: Context) {
        try {
            // Copied once per list version (assets are compressed in the APK, so the native side can't read them).
            // Version 2: FreeBSD's copy of the list. Earlier versions' copies are removed.
            File(context.filesDir, "words-1.txt").delete()
            val file = File(context.filesDir, "words-2.txt")
            if (!file.isFile) {
                val tmp = File(context.filesDir, "words-2.txt.part")
                context.assets.open("words.txt").use { i -> tmp.outputStream().use { i.copyTo(it) } }
                tmp.renameTo(file)
            }
            val err = AsrNpu.loadDictionary(file.absolutePath)
            if (err.isNotEmpty()) android.util.Log.w("TellWellASR", "name correction off: $err")
        } catch (t: Throwable) {
            android.util.Log.w("TellWellASR", "name correction off: ${t.javaClass.simpleName}")
        }
    }

    /** Hands contact names to the recogniser if they're on and not read recently (or clears them if turned off). */
    fun refreshNames(context: Context) {
        val app = context.applicationContext
        worker.execute {
            if (asrNative && Names.stale(app)) AsrNpu.setContacts(Names.contactWords(app))
        }
    }

    /** [lang]'s models are loaded and checked now (only the chosen language's cleanup stays in memory). */
    fun loaded(lang: Lang): Boolean = if (lang == Lang.EN) ready else hindiOn

    /**
     * Loads an optional language after its download (and English first, if that isn't loaded yet).
     * [done] gets whether it worked; it runs on the worker thread.
     */
    fun loadLanguage(context: Context, lang: Lang, done: (Boolean) -> Unit = {}) {
        val app = context.applicationContext
        worker.execute {
            if (!ready) load(app)
            // Check the new language loads and passes its self-check, then keep the chosen one in memory.
            if (ready) useLanguage(app, lang)
            val ok = loaded(lang)
            if (ready) useLanguage(app, Languages.active(app))
            done(ok)
        }
    }

    /** Frees a removed language's models (and brings English cleanup back). */
    fun unload(context: Context, lang: Lang) {
        if (lang != Lang.HI) return
        val app = context.applicationContext
        worker.execute { if (ready) useLanguage(app, Lang.EN) }
    }

    /**
     * Hindi: IndicConformer (4 CPU threads, ~0.9 s for 10 s of speech on a Snapdragon 7 Gen 4) and the Hindi Gemma
     * on the CPU (~0.8 s for a typical sentence with copy-ahead). Both must pass a self-check. Runs on the worker thread.
     */
    private fun loadHindi(context: Context) {
        if (hindiOn || !Languages.downloaded(context, Lang.HI) || !Languages.supported(context, Lang.HI)) return
        hindiLoading = true
        try {
            System.loadLibrary("tellwell_hindi_asr")
            val err = AsrHindi.load(context.applicationInfo.nativeLibraryDir, Languages.hindiModel(context).absolutePath,
                Languages.hindiTokens(context).absolutePath, 4)
            if (err.isNotEmpty() || AsrHindi.transcribe(FloatArray(16000)) == null) {
                android.util.Log.w("TellWellASR", "Hindi speech unavailable")
                Languages.setFailed(context, Lang.HI, true)
                return
            }
            Os.setenv("ADSP_LIBRARY_PATH", context.applicationInfo.nativeLibraryDir, true)
            System.loadLibrary("tellwell_llm")
            val got = LlmNative.load(HINDI_SLOT, context.applicationInfo.nativeLibraryDir, Languages.hindiCleanup(context).absolutePath,
                PromptHindi.PREFIX, "CPU", 4)
            // "um so tomorrow we meet at five, no, six": expected "कल छह बजे मिलते हैं।"
            val out = if (got.isEmpty()) "" else LlmNative.complete(HINDI_SLOT, PromptHindi.suffix("उम्म तो कल पाँच बजे मिलते हैं नहीं छह बजे"), 40, 0)
            val ok = out.contains("छह") && !out.contains("पाँच") && !out.contains("उम्म") &&
                out.all { it == ' ' || it in '\u0900'..'\u097F' || it in ".,?!" }
            android.util.Log.i("TellWellLLM", "Hindi self-check: ${if (ok) "passed" else "failed"}")
            Languages.setFailed(context, Lang.HI, !ok)
            if (!ok) {
                AsrHindi.unload()
                LlmNative.unload(HINDI_SLOT)
                return
            }
            hindiOn = true
        } catch (t: Throwable) {
            android.util.Log.w("TellWellASR", "Hindi unavailable: ${t.javaClass.simpleName}")
        } finally {
            hindiLoading = false
        }
    }

    /** Loads both models on the worker thread if needed; [done] runs on the worker thread. */
    fun ensureLoaded(context: Context, done: (String) -> Unit = {}) {
        val app = context.applicationContext
        worker.execute {
            if (!ready) load(app)
            done(status)
        }
    }

    /** The application context, kept from the first load for swapping languages' models. */
    @Volatile private var appContext: Context? = null

    private fun load(context: Context) {
        appContext = context.applicationContext
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
        if (!asrOnNpu && tryNativeCpu(context)) asrNative = true
        if (asrNative) {
            loadDictionary(context)
            if (Names.stale(context)) AsrNpu.setContacts(Names.contactWords(context))
        }
        if (!asrNative) try {
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
        loadEnglishCleanup(context)
        ready = true
        val speechOn = if (asrOnNpu) "speech on the NPU" else "speech on the CPU"
        status = if (cleanupOn) "Ready ($speechOn, cleanup on ${when (cleanupDevice) { "HTP0" -> "the NPU"; "GPUOpenCL" -> "the GPU"; else -> "the CPU" }})"
            else "Ready ($speechOn; cleanup isn't available on this phone)"
        // Only the chosen language's cleanup stays in memory: Hindi needs most of an 8 GB phone's free memory.
        if (Languages.active(context) == Lang.HI) useLanguage(context, Lang.HI)
    }

    /** The English cleanup backend in use ("HTP0", "GPUOpenCL", "CPU"), or "" if none works. */
    private var cleanupDevice = ""

    /** English cleanup (S1-mini). Optional: if no backend works on this phone, TellWell types the raw transcript. */
    private fun loadEnglishCleanup(context: Context) {
        var device = ""
        try {
            // The NPU backend loads its DSP code from the app's native library directory.
            Os.setenv("ADSP_LIBRARY_PATH", context.applicationInfo.nativeLibraryDir, true)
            System.loadLibrary("tellwell_llm")
            // Fastest first: the NPU, then the GPU, then the CPU. Each must pass a quick correctness check,
            // since a backend can run but compute wrong results on a chip it doesn't fully support.
            for (d in devices) {
                val got = LlmNative.load(ENGLISH_SLOT, context.applicationInfo.nativeLibraryDir, llmFile(context).absolutePath, Prompt.PREFIX, d, 4)
                if (got.isNotEmpty() && canaryPasses()) { device = got; break }
            }
        } catch (t: Throwable) {
            android.util.Log.w("TellWellLLM", "cleanup unavailable: ${t.javaClass.simpleName}")
        }
        cleanupOn = device.isNotEmpty()
        cleanupDevice = device
        englishCleanupLoaded = true
        // Copy-ahead decoding checks several tokens per pass. Measured on a Snapdragon 7 Gen 4: 20% faster on the CPU,
        // but slower on the NPU and GPU, whose small-batch passes cost more than they save. Debug runs can override.
        if (!draftOverridden) draftK = if (device == "CPU") 4 else 0
    }

    /** English cleanup is in memory (or was tried and isn't available on this phone). */
    private var englishCleanupLoaded = false

    /**
     * Keeps only [lang]'s models in memory: Hindi frees English cleanup (English speech stays, small on the NPU);
     * English frees Hindi. A switch costs a load (a few seconds) on the next dictation instead of being killed by
     * Android for using too much memory. Runs on the worker thread.
     */
    private fun useLanguage(context: Context, lang: Lang) {
        if (lang == Lang.HI) {
            if (hindiOn) return
            if (englishCleanupLoaded) {
                LlmNative.unload(ENGLISH_SLOT)
                englishCleanupLoaded = false
            }
            loadHindi(context)
        } else {
            if (hindiOn) {
                hindiOn = false
                AsrHindi.unload()
                LlmNative.unload(HINDI_SLOT)
            }
            if (!englishCleanupLoaded) loadEnglishCleanup(context)
        }
    }

    /** Loads [lang]'s models ahead of the next dictation (the keyboard's switch calls this). */
    fun prepare(context: Context, lang: Lang) {
        val app = context.applicationContext
        worker.execute { if (ready) useLanguage(app, lang) }
    }

    /**
     * Transcribes and cleans up 16 kHz mono audio in [lang]. [context] is the text around the cursor (its names are
     * favoured in English; kept in memory only). [done] runs on the worker thread.
     */
    fun process(samples: FloatArray, context: String = "", lang: Lang = Lang.EN, done: (Result) -> Unit) {
        worker.execute {
            // A dictation in the other language than the one in memory: swap first (a few seconds, once).
            appContext?.let { if (ready) useLanguage(it, lang) }
            if (lang == Lang.HI) {
                if (!hindiOn || samples.size < 1600) { done(Result("", 0, 0, "")); return@execute }
                val t0 = System.nanoTime()
                val raw = (AsrHindi.transcribe(samples) ?: "").trim()
                val t1 = System.nanoTime()
                val text = if (raw.isEmpty()) "" else cleanupHindi(raw)
                done(Result(text, (t1 - t0) / 1_000_000, (System.nanoTime() - t1) / 1_000_000, raw))
                return@execute
            }
            val recognizer = asr
            // Runs after any load queued before it, so a clip recorded during loading is still processed.
            if (!ready || (recognizer == null && !asrNative) || samples.size < 1600) {
                done(Result("", 0, 0, ""))
                return@execute
            }
            val t0 = System.nanoTime()
            // sherpa-onnx isn't loaded when TellWell's decoder is used; a failed run then types nothing.
            val raw = ((if (asrNative) AsrNpu.transcribe(samples, context) else null) ?: recognizer?.let { r ->
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
        val out = LlmNative.complete(ENGLISH_SLOT, Prompt.suffix("um so lets meet at five no wait six on thursday"), 40, 0)
        val ok = out.contains("Thursday") && (out.contains("6") || out.contains("six", ignoreCase = true)) &&
            !out.contains("five", ignoreCase = true) && out.all { it.code < 0x250 }
        android.util.Log.i("TellWellLLM", "self-check on ${LlmNative.device(ENGLISH_SLOT)}: ${if (ok) "passed" else "failed"}")
        return ok
    }

    /**
     * Hindi cleanup on the CPU with copy-ahead, then the guard that undoes a substituted word which neither sounds
     * like what was heard nor was said elsewhere (see HindiGuard), and the same length guard as English.
     */
    private fun cleanupHindi(raw: String): String {
        val out = HindiGuard.apply(raw, LlmNative.complete(HINDI_SLOT, PromptHindi.suffix(raw), 400, 4).trim())
        if (out == "EMPTY") return ""
        val rawWords = raw.split(Regex("\\s+")).size
        val outWords = out.split(Regex("\\s+")).size
        if (out.isEmpty() || outWords > rawWords * 2 + 8) return raw
        return out
    }

    private const val ENGLISH_SLOT = 0
    private const val HINDI_SLOT = 1

    private fun cleanup(raw: String): String {
        val out = LlmNative.complete(ENGLISH_SLOT, Prompt.suffix(raw), 160, draftK).trim()
        if (out == "EMPTY") return ""
        // Guard: a cleanup should never be empty for real speech or much longer than what was said.
        val rawWords = raw.split(Regex("\\s+")).size
        val outWords = out.split(Regex("\\s+")).size
        if (out.isEmpty() || outWords > rawWords * 2 + 8) return raw
        return out
    }
}
