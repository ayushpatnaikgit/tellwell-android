package io.github.ayushpatnaikgit.tellwell

import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import javax.net.ssl.HttpsURLConnection

/**
 * Model downloads over HTTPS: resumes partial files, tries each source in turn, and keeps a file only if its size and
 * SHA-256 match. It only downloads; nothing is ever uploaded. Used for the GitHub build's models and, in both builds,
 * for optional languages.
 */
object Fetch {
    /** [urls] are tried in order; the files are identical, so a partial download carries over between them. */
    data class Model(val urls: List<String>, val path: String, val size: Long, val sha256: String)

    class Cancelled : Exception()

    /** Files in [list] not yet complete under [dir]. */
    fun missing(dir: File, list: List<Model>): List<Model> =
        list.filter { File(dir, it.path).let { f -> !f.isFile || f.length() != it.size } }

    /** Each source in turn; the last one's error is reported if all fail. */
    fun fetch(m: Model, out: File, cancelled: () -> Boolean, progress: (Long) -> Unit) {
        var failure: Exception? = null
        for (url in m.urls) {
            try {
                fetchFrom(url, m, out, cancelled, progress)
                return
            } catch (e: Cancelled) {
                throw e
            } catch (e: Exception) {
                failure = e
            }
        }
        throw failure ?: java.io.IOException("no source")
    }

    private fun fetchFrom(url: String, m: Model, out: File, cancelled: () -> Boolean, progress: (Long) -> Unit) {
        out.parentFile?.mkdirs()
        val part = File(out.path + ".part")
        var have = if (part.isFile) part.length() else 0L
        if (have > m.size) { part.delete(); have = 0 }
        if (have < m.size) {
            val conn = URL(url).openConnection() as HttpsURLConnection
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
                        if (cancelled()) throw Cancelled()
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
