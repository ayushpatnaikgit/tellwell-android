package io.github.ayushpatnaikgit.tellwell

import android.app.ActivityManager
import android.content.Context
import java.io.File

/** A dictation language. [label] is what the keyboard's switch shows. */
enum class Lang(val code: String, val label: String, val englishName: String) {
    EN("en", "English", "English"),
    HI("hi", "हिंदी", "Hindi"),
}

/**
 * Dictation languages, in TellWell Multilingual (TellWell itself is English only). English is the default; its models
 * come with the app (Play asset packs, or the GitHub build's download). Other languages are optional: the user adds
 * one in setup, and its own speech and cleanup models
 * download from TellWell's model release over HTTPS (checksum-verified, see Fetch). Removing a language deletes its
 * models. When more than one language is ready, the keyboard shows a switch; the choice is remembered.
 */
object Languages {
    private const val RELEASE = "https://github.com/ayushpatnaikgit/tellwell-models/releases/download/hindi-v1"

    // IndicConformer Hindi large (AI4Bharat, MIT), exported with its feature computation as one audio-in CTC model;
    // and Gemma 4 E2B (Apache 2.0), fine-tuned for Hindi and English dictation cleanup, Q4_0, vocabulary trimmed.
    // How they were made and tested: tellwell-lab/hindi (RESULTS.md).
    private val HINDI = listOf(
        Fetch.Model(listOf("$RELEASE/indicconformer-hi-tokens.txt"), "hindi/indicconformer-hi-tokens.txt", 1_782, "8b806c485a120e67744443bc1bfcfff86c6c87a675752ba00a9f0de05faf07d3"),
        Fetch.Model(listOf("$RELEASE/indicconformer-hi-ctc.onnx"), "hindi/indicconformer-hi-ctc.onnx", 483_138_372, "d2ea555b5469a7a628e5dd2404acc99dd8da6ea0bc311e9f66a7f3e1782a546e"),
        Fetch.Model(listOf("$RELEASE/gemma-4-E2B-it-qat-hindi1-trim-Q4_0.gguf"), "hindi/gemma-4-E2B-it-qat-hindi1-trim-Q4_0.gguf", 1_909_617_856, "5215f4b0b59a95a38cf2787de9065ef9403b43ab31a00e80b9b071ca88fc149d"),
    )

    /**
     * Hindi keeps about 4 GB in memory while dictating, so it is offered on phones with 8 GB (Android reports about
     * 7.5 GB of that). Measured on a Motorola Edge 70 (8 GB).
     */
    private const val HINDI_MIN_MEMORY = 7_000_000_000L

    private fun prefs(c: Context) = c.getSharedPreferences("languages", Context.MODE_PRIVATE)
    private fun dir(c: Context) = Engine.modelsDir(c.applicationContext)

    fun files(lang: Lang): List<Fetch.Model> = if (lang == Lang.HI) HINDI else emptyList()

    fun hindiModel(c: Context) = File(dir(c), HINDI[1].path)
    fun hindiTokens(c: Context) = File(dir(c), HINDI[0].path)
    fun hindiCleanup(c: Context) = File(dir(c), HINDI[2].path)

    /** The phone has enough memory for [lang]. */
    fun supported(c: Context, lang: Lang): Boolean {
        if (lang == Lang.EN) return true
        if (!BuildConfig.MULTILINGUAL) return false
        val info = ActivityManager.MemoryInfo()
        c.getSystemService(ActivityManager::class.java).getMemoryInfo(info)
        return info.totalMem >= HINDI_MIN_MEMORY
    }

    /** All of [lang]'s files are downloaded and verified (English: see Engine.modelsPresent). */
    fun downloaded(c: Context, lang: Lang): Boolean =
        lang == Lang.EN || Fetch.missing(dir(c), files(lang)).isEmpty()

    fun missingBytes(c: Context, lang: Lang): Long = Fetch.missing(dir(c), files(lang)).sumOf { it.size }
    fun totalBytes(lang: Lang): Long = files(lang).sumOf { it.size }

    /** Deletes [lang]'s models, including partial downloads. English can't be removed. */
    fun remove(c: Context, lang: Lang) {
        if (lang == Lang.EN) return
        Engine.unload(c, lang)
        setFailed(c, lang, false)
        files(lang).forEach { m -> File(dir(c), m.path).delete(); File(dir(c), m.path + ".part").delete() }
        if (active(c) == lang) setActive(c, Lang.EN)
    }

    /** The language the keyboard dictates in: the last one chosen, if it is still downloaded. */
    fun active(c: Context): Lang {
        val l = Lang.entries.firstOrNull { it.code == prefs(c).getString("active", "en") } ?: Lang.EN
        return if (usable(c, l)) l else Lang.EN
    }

    fun setActive(c: Context, lang: Lang) { prefs(c).edit().putString("active", lang.code).apply() }

    /** [lang] didn't load or failed its self-check on this phone (remembered until it is removed or loads). */
    fun failed(c: Context, lang: Lang): Boolean = prefs(c).getBoolean("failed-${lang.code}", false)
    fun setFailed(c: Context, lang: Lang, failed: Boolean) { prefs(c).edit().putBoolean("failed-${lang.code}", failed).apply() }

    /** [lang] can be dictated in: downloaded, supported on this phone and not failed. Its models load when chosen. */
    fun usable(c: Context, lang: Lang): Boolean = downloaded(c, lang) && supported(c, lang) && !failed(c, lang)

    /** Languages the keyboard can switch between. */
    fun ready(c: Context): List<Lang> = Lang.entries.filter { usable(c, it) }

    // Download of an optional language (one at a time), run by LanguageDownloadService.
    @Volatile var downloading: Lang? = null
        private set
    @Volatile private var cancelled = false
    /** The setup screen's progress view: (language, bytes done, total, error, finished). Called from the download thread. */
    @Volatile var listener: ((Lang, Long, Long, String?, Boolean) -> Unit)? = null

    fun cancel() { cancelled = true }

    /** Downloads what [lang] is missing. [done] gets null on success or a short, content-free error. */
    fun download(c: Context, lang: Lang, progress: (Long, Long) -> Unit, done: (String?) -> Unit) {
        if (downloading != null) return
        downloading = lang
        cancelled = false
        val d = dir(c)
        val list = files(lang)
        val total = list.sumOf { it.size }
        Thread({
            val error = try {
                var base = 0L
                for (m in list) {
                    val out = File(d, m.path)
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
                downloading = null
            }
            done(error)
        }, "tellwell-language-download").start()
    }
}
