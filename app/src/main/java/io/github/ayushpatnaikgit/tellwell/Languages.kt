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

    // IndicConformer Hindi large (AI4Bharat, MIT), exported with its feature computation as one audio-in CTC model.
    // How the Hindi models were made and tested: tellwell-lab/hindi (RESULTS.md).
    private val SPEECH = listOf(
        Fetch.Model(listOf("$RELEASE/indicconformer-hi-tokens.txt"), "hindi/indicconformer-hi-tokens.txt", 1_782, "8b806c485a120e67744443bc1bfcfff86c6c87a675752ba00a9f0de05faf07d3"),
        Fetch.Model(listOf("$RELEASE/indicconformer-hi-ctc.onnx"), "hindi/indicconformer-hi-ctc.onnx", 483_138_372, "d2ea555b5469a7a628e5dd2404acc99dd8da6ea0bc311e9f66a7f3e1782a546e"),
    )

    /**
     * Hindi is offered on phones with 8 GB (Android reports about 7.5 GB of that): with Hindi speech, the cleanups keep
     * about 1, 2.5 and 4 GB in memory. Measured on a Motorola Edge 70 (8 GB).
     */
    private const val HINDI_MIN_MEMORY = 7_000_000_000L

    private fun prefs(c: Context) = c.getSharedPreferences("languages", Context.MODE_PRIVATE)
    private fun dir(c: Context) = Engine.modelsDir(c.applicationContext)

    fun hindiModel(c: Context) = File(dir(c), SPEECH[1].path)
    fun hindiTokens(c: Context) = File(dir(c), SPEECH[0].path)
    private fun present(c: Context, m: Fetch.Model) = Fetch.missing(dir(c), listOf(m)).isEmpty()
    private fun present(c: Context, k: HindiCleanup) = present(c, k.model)

    /** The Hindi cleanup the user chose (Standard unless changed). */
    fun chosenCleanup(c: Context): HindiCleanup =
        HindiCleanup.entries.firstOrNull { it.name == prefs(c).getString("hi-cleanup", null) } ?: HindiCleanup.STANDARD

    /** What [lang] downloads: its speech model and the chosen cleanup. */
    fun files(c: Context, lang: Lang): List<Fetch.Model> =
        if (lang != Lang.HI) emptyList() else SPEECH + chosenCleanup(c).model

    /**
     * The Hindi cleanup to load: the chosen one if it is on the phone and loads here; otherwise one that is (while
     * the chosen one downloads, or on a phone that has only the first release's model).
     */
    fun hindiCleanup(c: Context): HindiCleanup? {
        val chosen = chosenCleanup(c)
        if (present(c, chosen) && !cleanupFailed(c, chosen)) return chosen
        return listOf(HindiCleanup.STANDARD, HindiCleanup.TINY, HindiCleanup.LARGE)
            .firstOrNull { present(c, it) && !cleanupFailed(c, it) }
    }

    fun hindiCleanupFile(c: Context, k: HindiCleanup) = File(dir(c), k.model.path)

    fun cleanupFailed(c: Context, k: HindiCleanup) = prefs(c).getBoolean("failed-hi-${k.name}", false)
    fun setCleanupFailed(c: Context, k: HindiCleanup, failed: Boolean) {
        prefs(c).edit().putBoolean("failed-hi-${k.name}", failed).apply()
    }

    /** Chooses a Hindi cleanup; the caller then downloads it. The others are deleted once it is on the phone. */
    fun chooseCleanup(c: Context, k: HindiCleanup) {
        prefs(c).edit().putString("hi-cleanup", k.name).apply()
        setCleanupFailed(c, k, false)
        dropOtherCleanups(c)
    }

    /** Only one Hindi cleanup stays on the phone: once the chosen one is downloaded, the others are deleted. */
    fun dropOtherCleanups(c: Context) {
        val chosen = chosenCleanup(c)
        if (!present(c, chosen)) return
        for (k in HindiCleanup.entries) if (k != chosen) {
            File(dir(c), k.model.path).delete(); File(dir(c), k.model.path + ".part").delete()
            setCleanupFailed(c, k, false)
        }
    }

    /** The phone has enough memory for [lang]. */
    fun supported(c: Context, lang: Lang): Boolean {
        if (lang == Lang.EN) return true
        if (!BuildConfig.MULTILINGUAL) return false
        val info = ActivityManager.MemoryInfo()
        c.getSystemService(ActivityManager::class.java).getMemoryInfo(info)
        return info.totalMem >= HINDI_MIN_MEMORY
    }

    /**
     * [lang] can be dictated in with what is on the phone: Hindi speech and any cleanup (English: see
     * Engine.modelsPresent). A newly chosen cleanup still downloading doesn't stop Hindi.
     */
    fun downloaded(c: Context, lang: Lang): Boolean = when (lang) {
        Lang.EN -> true
        Lang.HI -> Fetch.missing(dir(c), SPEECH).isEmpty() && HindiCleanup.entries.any { present(c, it) }
    }

    /** Everything [lang] should have (speech and the chosen cleanup) is on the phone. */
    fun complete(c: Context, lang: Lang): Boolean = Fetch.missing(dir(c), files(c, lang)).isEmpty()
    fun missingBytes(c: Context, lang: Lang): Long = Fetch.missing(dir(c), files(c, lang)).sumOf { it.size }
    fun totalBytes(c: Context, lang: Lang): Long = files(c, lang).sumOf { it.size }
    /** What removing [lang] would free. */
    fun presentBytes(c: Context, lang: Lang): Long =
        if (lang != Lang.HI) 0 else (SPEECH + HindiCleanup.entries.map { it.model }).filter { present(c, it) }.sumOf { it.size }

    /** Deletes [lang]'s models, including partial downloads. English can't be removed. */
    fun remove(c: Context, lang: Lang) {
        if (lang == Lang.EN) return
        Engine.unload(c, lang)
        setFailed(c, lang, false)
        HindiCleanup.entries.forEach { setCleanupFailed(c, it, false) }
        (SPEECH + HindiCleanup.entries.map { it.model }).forEach { m ->
            File(dir(c), m.path).delete(); File(dir(c), m.path + ".part").delete()
        }
        prefs(c).edit().remove("hi-cleanup").apply()
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
        // Only what is missing, so progress counts what this download fetches.
        val list = Fetch.missing(d, files(c, lang))
        val total = list.sumOf { it.size }.coerceAtLeast(1)
        Thread({
            val error = try {
                var base = 0L
                for (m in list) {
                    val out = File(d, m.path)
                    if (!out.isFile || out.length() != m.size) Fetch.fetch(m, out, { cancelled }) { n -> progress(base + n, total) }
                    base += m.size
                    progress(base, total)
                }
                if (lang == Lang.HI) dropOtherCleanups(c)
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

/**
 * The Hindi cleanup models, smallest first. Each is fine-tuned for Hindi dictation cleanup, vocabulary trimmed, and
 * tested on held-out real Hindi speech (FLEURS, Lahaja) for words it puts in that weren't said. [label] and [blurb]
 * are what setup shows.
 */
enum class HindiCleanup(val label: String, val blurb: String, val model: Fetch.Model, val prompt: PromptHindi) {
    /**
     * Gemma 3 270M (Gemma Terms of Use), LoRA fine-tune, Q8_0 (Q4_0 broke long dictations): ~0.1 s a sentence. Puts
     * in the fewest words that weren't said, but long dictations often come back cut off or looping, so the length
     * checks in Engine.cleanupHindi type them as heard.
     */
    TINY("Tiny", "Fastest and smallest, for short messages. Cleans up the least, and leaves long dictations as spoken.",
        Fetch.Model(listOf("https://github.com/ayushpatnaikgit/tellwell-models/releases/download/hindi-v2/gemma-3-270m-it-hindi2-trim-Q8_0.gguf"),
            "hindi/gemma-3-270m-it-hindi2-trim-Q8_0.gguf", 174_420_288, "23a2719016f2ba96a44a5d5824c53ea7f6c2451905dc0f5296328036d154e688"),
        PromptHindi.GEMMA3),

    /** Gemma 3 1B (Gemma Terms of Use), full fine-tune, Q8_0 (Q4_0 lost too much of the fine-tune): ~0.35 s. */
    STANDARD("Standard", "Removes most fillers and repeated words. Recommended.",
        Fetch.Model(listOf("https://github.com/ayushpatnaikgit/tellwell-models/releases/download/hindi-v2/gemma-3-1b-it-hindi2full-trim-Q8_0.gguf"),
            "hindi/gemma-3-1b-it-hindi2full-trim-Q8_0.gguf", 861_795_616, "76ae8c8400a19be7c0ab7ca9f1c3e1ade6fc4906a553c986d50dd1d4e7277419"),
        PromptHindi.GEMMA3),

    /**
     * Gemma 4 E2B (Apache 2.0), LoRA fine-tune, Q4_0: ~0.8 s and about 4 GB in memory. Same file (path, size,
     * checksum) as the first Hindi release, so phones that have it keep it.
     */
    LARGE("Large", "Catches the most fillers and self-corrections, but slower and needs more memory.",
        Fetch.Model(listOf("https://github.com/ayushpatnaikgit/tellwell-models/releases/download/hindi-v1/gemma-4-E2B-it-qat-hindi1-trim-Q4_0.gguf"),
            "hindi/gemma-4-E2B-it-qat-hindi1-trim-Q4_0.gguf", 1_909_617_856, "5215f4b0b59a95a38cf2787de9065ef9403b43ab31a00e80b9b071ca88fc149d"),
        PromptHindi.GEMMA4),
}
