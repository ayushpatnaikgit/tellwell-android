package io.github.ayushpatnaikgit.tellwell

import java.text.Normalizer

/**
 * The Hindi guard on the cleanup model: a word the model put in place of what was heard is kept only if it sounds
 * like it, or was spoken elsewhere in the dictation (a self-correction moves words). Otherwise the heard words go back.
 * This stops the model from swapping a spoken name for a different one. Port of tellwell-lab/hindi/devsound.py and
 * guard2.py (threshold 0.65: it changed none of the 42 Hindi test cases except the one name swap it exists for, and left
 * word errors on held-out real speech unchanged). Works on text in memory only.
 */
object HindiGuard {
    private const val THRESHOLD = 0.65
    private const val NUKTA = '़'
    private const val PUNCT = "।॥.,!?;:\"'“”‘’()[]-–—/…"

    private val MAP = mapOf(
        "ख" to "क", "घ" to "ग", "छ" to "च", "झ" to "ज", "ठ" to "ट", "ढ" to "ड", "थ" to "त", "ध" to "द", "फ" to "प", "भ" to "ब",
        "ङ" to "N", "ञ" to "N", "ण" to "N", "ं" to "N", "ँ" to "N",
        "ी" to "ि", "ू" to "ु", "ई" to "इ", "ऊ" to "उ", "ै" to "े", "ौ" to "ो", "ऐ" to "ए", "औ" to "ओ", "ॅ" to "े", "ॉ" to "ो", "ऑ" to "ओ",
        "श" to "स", "ष" to "स", "ऋ" to "रि", "ृ" to "रि", "ः" to "", "्" to "", "़" to "", "ऽ" to "",
    )

    /** A coarse phonetic key: aspiration, nasals, vowel length, nukta and sibilants merged. */
    fun key(word: String): String {
        val w = Normalizer.normalize(word, Normalizer.Form.NFD)
        val sb = StringBuilder()
        for (i in w.indices) {
            val ch = w[i].toString()
            var m = MAP[ch] ?: ch
            if ((ch == "न" || ch == "म") && i + 1 < w.length && w[i + 1] == '्') m = "N"
            sb.append(m)
        }
        var k = sb.toString()
        while ("NN" in k) k = k.replace("NN", "N")
        return k
    }

    private fun lev(a: String, b: String): Int {
        var prev = IntArray(b.length + 1) { it }
        for (i in 1..a.length) {
            val cur = IntArray(b.length + 1)
            cur[0] = i
            for (j in 1..b.length) cur[j] = minOf(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + if (a[i - 1] == b[j - 1]) 0 else 1)
            prev = cur
        }
        return prev[b.length]
    }

    fun similarity(a: String, b: String): Double {
        val ka = key(a)
        val kb = key(b)
        if (ka.isEmpty() && kb.isEmpty()) return 1.0
        return 1.0 - lev(ka, kb).toDouble() / maxOf(ka.length, kb.length)
    }

    private fun norm(w: String) = w.trim { it in PUNCT }.replace(NUKTA.toString(), "")

    /** Word alignment (longest common subsequence): pairs of matching indices, in order. */
    private fun matches(a: List<String>, b: List<String>): List<Pair<Int, Int>> {
        val n = a.size
        val m = b.size
        val dp = Array(n + 1) { IntArray(m + 1) }
        for (i in n - 1 downTo 0) for (j in m - 1 downTo 0)
            dp[i][j] = if (a[i] == b[j]) dp[i + 1][j + 1] + 1 else maxOf(dp[i + 1][j], dp[i][j + 1])
        val out = ArrayList<Pair<Int, Int>>()
        var i = 0
        var j = 0
        while (i < n && j < m) {
            when {
                a[i] == b[j] -> { out.add(i to j); i++; j++ }
                dp[i + 1][j] >= dp[i][j + 1] -> i++
                else -> j++
            }
        }
        return out
    }

    /** The cleaned text with unsupported substitutions undone. [heard]: the transcript the model was given. */
    fun apply(heard: String, cleaned: String): String {
        if (cleaned.trim().equals("EMPTY", ignoreCase = true)) return cleaned
        val h = heard.split(Regex("\\s+")).filter { it.isNotEmpty() }
        val o = cleaned.split(Regex("\\s+")).filter { it.isNotEmpty() }
        val hn = h.map(::norm)
        val on = o.map(::norm)
        val spoken = hn.toSet()
        val res = ArrayList<String>()
        var pi = 0
        var pj = 0
        for ((mi, mj) in matches(hn, on) + listOf(h.size to o.size)) {
            // The gap before this match: heard[pi, mi) became out[pj, mj).
            if (mi > pi && mj > pj) {
                val new = on.subList(pj, mj).filter { it.isNotEmpty() && it !in spoken }
                if (new.isNotEmpty() && similarity(hn.subList(pi, mi).joinToString(""), on.subList(pj, mj).joinToString("")) < THRESHOLD) {
                    val words = h.subList(pi, mi).toMutableList()
                    val end = Regex("[।?!,.]+$").find(o[mj - 1])?.value
                    if (end != null && !Regex("[।?!,.]$").containsMatchIn(words.last())) words[words.size - 1] = words.last() + end
                    res.addAll(words)
                } else {
                    res.addAll(o.subList(pj, mj))
                }
            } else if (mj > pj) {
                res.addAll(o.subList(pj, mj))
            }
            if (mi < h.size && mj < o.size) res.add(o[mj])
            pi = mi + 1
            pj = mj + 1
        }
        return res.joinToString(" ")
    }
}
