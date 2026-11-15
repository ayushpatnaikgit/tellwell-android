package io.github.ayushpatnaikgit.tellwell

import android.content.Context
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import javax.net.ssl.HttpsURLConnection

/**
 * The only network code in TellWell. It downloads the two models once, from pinned Hugging Face commits over HTTPS,
 * and keeps a file only if its SHA-256 matches. Nothing is ever uploaded, and nothing else uses the network.
 */
object ModelDownloader {
    data class Model(val url: String, val path: String, val size: Long, val sha256: String)

    private const val ASR = "https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/resolve/ed5d745623472f97cfc2e6c2916ee248798fc5a8"
    private const val LLM = "https://huggingface.co/ayushpatnaik/S1-mini-FreeFlow-cleanup-GGUF/resolve/6665acc599d42d7adc2d55c577119e56ee22810d"

    val MODELS = listOf(
        Model("$ASR/tokens.txt", "asr/tokens.txt", 93_939, "d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d"),
        Model("$ASR/decoder.int8.onnx", "asr/decoder.int8.onnx", 11_845_274, "0222a86147299d4dee90edf95b6d13827edb42ad8b7c9cf0b2582f030937c0b5"),
        Model("$ASR/joiner.int8.onnx", "asr/joiner.int8.onnx", 6_355_277, "cbd02f3696e9944970111fb439d9c3fd365a80374565f13f2b9c7007d253fc98"),
        Model("$LLM/s1-mini-freeflow-cleanup-q4_0.gguf", "s1-mini-freeflow-cleanup-q4_0.gguf", 381_565_696, "ff7dfbc19d7220ac7e1f7cd88bc95da3db7e7e7893a35ea552895727605abddc"),
        Model("$ASR/encoder.int8.onnx", "asr/encoder.int8.onnx", 652_282_294, "1afbe69c1ffdabee94755f42e4daf1d93549131a3da11356bda84fcef0f2c479"),
    )
    val TOTAL = MODELS.sumOf { it.size }

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
        Thread({
            val error = try {
                var base = 0L
                for (m in MODELS) {
                    val out = File(dir, m.path)
                    if (!out.isFile || out.length() != m.size) fetch(m, out) { n -> progress(base + n, TOTAL) }
                    base += m.size
                    progress(base, TOTAL)
                }
                null
            } catch (e: Cancelled) {
                "Paused"
            } catch (e: Exception) {
                "Download failed (${e.javaClass.simpleName}). Check your connection and try again."
            } finally {
                running = false
            }
            done(error)
        }, "tellwell-download").start()
    }

    private class Cancelled : Exception()

    private fun fetch(m: Model, out: File, progress: (Long) -> Unit) {
        out.parentFile?.mkdirs()
        val part = File(out.path + ".part")
        var have = if (part.isFile) part.length() else 0L
        if (have > m.size) { part.delete(); have = 0 }
        if (have < m.size) {
            val conn = URL(m.url).openConnection() as HttpsURLConnection
            conn.instanceFollowRedirects = true
            conn.connectTimeout = 20_000
            conn.readTimeout = 30_000
            conn.setRequestProperty("User-Agent", "TellWell-Android")
            if (have > 0) conn.setRequestProperty("Range", "bytes=$have-")
            val code = conn.responseCode
            if (code == HttpURLConnection.HTTP_OK) have = 0 // server ignored the range: start over
            else if (code != HttpURLConnection.HTTP_PARTIAL) throw java.io.IOException("HTTP $code")
            conn.inputStream.use { input ->
                FileOutputStream(part, have > 0).use { output ->
                    val buf = ByteArray(256 * 1024)
                    var n = have
                    var last = 0L
                    while (true) {
                        if (cancelled) throw Cancelled()
                        val r = input.read(buf)
                        if (r < 0) break
                        output.write(buf, 0, r)
                        n += r
                        if (n - last > 1_000_000) { progress(n); last = n }
                    }
                }
            }
            conn.disconnect()
        }
        if (part.length() != m.size || sha256(part) != m.sha256) {
            part.delete()
            throw java.io.IOException("verification failed")
        }
        if (!part.renameTo(out)) throw java.io.IOException("rename failed")
    }

    private fun sha256(f: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        f.inputStream().use { input ->
            val buf = ByteArray(1024 * 1024)
            while (true) {
                val r = input.read(buf)
                if (r < 0) break
                md.update(buf, 0, r)
            }
        }
        return md.digest().joinToString("") { "%02x".format(it) }
    }
}
