package io.github.ayushpatnaikgit.tellwell

import android.content.Context
import java.io.File

/** The GitHub build's English model files, described like every download (see Fetch). */
typealias Model = Fetch.Model

/**
 * The GitHub build's English models, downloaded once over HTTPS (through Fetch) from TellWell's own model release
 * (github.com/ayushpatnaikgit/tellwell-models, so the links can't break) or, if that fails, from the pinned Hugging
 * Face commits they were copied from. A file is kept only if its SHA-256 matches. Nothing is ever uploaded; the only
 * other network use is an optional language the user adds (Languages).
 */
object ModelDownloader {
    // NVIDIA Parakeet TDT 0.6B v2 (English), int8, exported for sherpa-onnx. Beat the multilingual v3 on English in
    // our tests (hard terms 29 vs 22 of 120; 2.05% vs 2.91% word errors on ordinary sentences).
    private const val ASR = "https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/resolve/84b478d541cdbf0c296a7ab5f1e20a6f8f998114"
    private const val LLM = "https://huggingface.co/ayushpatnaik/S1-mini-FreeFlow-cleanup-GGUF/resolve/6665acc599d42d7adc2d55c577119e56ee22810d"
    private const val MIRROR = "https://github.com/ayushpatnaikgit/tellwell-models/releases/download/v1"

    /** What every phone needs: the speech decoder, joiner and tokens, and the cleanup model. */
    private val COMMON = listOf(
        Model(listOf("$MIRROR/parakeet-tdt-0.6b-v2-int8-tokens.txt", "$ASR/tokens.txt"), "asr-v2/tokens.txt", 9_384, "ec182b70dd42113aff6c5372c75cac58c952443eb22322f57bbd7f53977d497d"),
        Model(listOf("$MIRROR/parakeet-tdt-0.6b-v2-int8-decoder.int8.onnx", "$ASR/decoder.int8.onnx"), "asr-v2/decoder.int8.onnx", 7_257_753, "4d097bae3bac5c96018d5293ba39f1b657296102b142affda40378f38b020f41"),
        Model(listOf("$MIRROR/parakeet-tdt-0.6b-v2-int8-joiner.int8.onnx", "$ASR/joiner.int8.onnx"), "asr-v2/joiner.int8.onnx", 1_739_080, "77a29f3e5ac7d37074e0af465391b80f314015d5b97f0f90c6c7767f1d56f428"),
        Model(listOf("$MIRROR/s1-mini-freeflow-cleanup-q4_0.gguf", "$LLM/s1-mini-freeflow-cleanup-q4_0.gguf"), "s1-mini-freeflow-cleanup-q4_0.gguf", 381_565_696, "ff7dfbc19d7220ac7e1f7cd88bc95da3db7e7e7893a35ea552895727605abddc"),
    )
    /** The speech encoder for the CPU (any phone). */
    val CPU_ENCODER = Model(listOf("$MIRROR/parakeet-tdt-0.6b-v2-int8-encoder.int8.onnx", "$ASR/encoder.int8.onnx"), "asr-v2/encoder.int8.onnx", 652_282_309, "92a10bd9c23936ea0566c29f963bc57c2a5df4261b6a8921c39abea66557ca8b")
    /** The speech encoder for Qualcomm NPUs (see Npu). Made for TellWell, so only on its own release. */
    private val NPU_ENCODER = listOf(
        Model(listOf("$MIRROR/parakeet-tdt-0.6b-v2-npu-w8a16-10s.onnx"), "asr-npu/enc-w8a16-1000.onnx", 1_696_675, "e50e3a70a6b8ae12596a3c709fe4633a154e3ce9ac889a233a0a8eaf74e96848"),
        Model(listOf("$MIRROR/parakeet-tdt-0.6b-v2-npu-w8a16-10s.onnx.data"), "asr-npu/enc-w8a16-1000.onnx.data", 590_338_697, "92538fb20215e4b237d46daf4b082cc43dd478e90ebc6c8c06f8fbdb881a326e"),
    )

    /** The files this phone needs: the NPU encoder on a Qualcomm phone (until compiled), else the CPU one. */
    fun models(context: Context): List<Model> = COMMON + when {
        Npu.candidate(context) -> if (Npu.compiled(context)) emptyList() else NPU_ENCODER
        else -> listOf(CPU_ENCODER)
    }

    /** Needed files not yet on the phone. */
    fun missing(context: Context): List<Model> {
        return Fetch.missing(Engine.modelsDir(context.applicationContext), models(context))
    }

    /** The setup screen's progress view: (done, total, error, finished). Called from the download thread. */
    @Volatile var listener: ((Long, Long, String?, Boolean) -> Unit)? = null

    @Volatile var running = false
        private set
    @Volatile private var cancelled = false

    fun cancel() { cancelled = true }

    /**
     * Downloads whatever is missing. Resumes partial files. [progress] gets (bytes done, total);
     * [done] gets null on success or a short, content-free error message. Both run on the download thread.
     */
    fun start(context: Context, progress: (Long, Long) -> Unit, done: (String?) -> Unit) {
        if (running) return
        running = true
        cancelled = false
        val dir = Engine.modelsDir(context.applicationContext)
        val list = models(context)
        val total = list.sumOf { it.size }
        Thread({
            val error = try {
                var base = 0L
                for (m in list) {
                    val out = File(dir, m.path)
                    if (!out.isFile || out.length() != m.size) Fetch.fetch(m, out, { cancelled }) { n -> progress(base + n, total) }
                    base += m.size
                    progress(base, total)
                }
                null
            } catch (e: Fetch.Cancelled) {
                "Paused"
            } catch (e: Exception) {
                "Download failed (${e.javaClass.simpleName}). Check your connection and try again."
            } finally {
                running = false
            }
            done(error)
        }, "tellwell-download").start()
    }
}
