package io.github.ayushpatnaikgit.tellwell

import java.text.Normalizer

/**
 * Hindi multi-pass cleanup around the cleanup model (the Tiny 270M uses no more memory for it: every pass is the same
 * model). Port of tellwell-lab/hindi2pass (hi_gate.py, pipeline.py, stage 2); tests/HindiTwoPassTest.kt checks it
 * against that code's outputs.
 *
 * - [pieces]: a long dictation is cleaned in pieces of at most [SPLIT] words (the small model runs away or stops early
 *   on long inputs; on 30 long passages this took word errors from 47% to 12%).
 * - [gate]: the model may delete words (fillers, restarts, corrections are its job), except a removed number word not
 *   written as digits there, or an unexplained run of 3+ real words; it may swap or add a word only for a number, near
 *   a correction cue, for a word said after a cue, or by joining heard words ("ए वी टी" -> "एवीटी"). Spelling-only
 *   changes (nukta, chandrabindu) always pass. If most of its edits would be undone, the heard text is used.
 * - Rules: EMPTY when only fillers were said; "A रुको B" corrections anchored on B's last word; tag questions
 *   ("है ना") end in "?".
 * - Choices the model scores (a [Scorer], log-probability of a text): collapsing a repeated word ("मैं मैं" but not
 *   "साफ़ साफ़"), and "?" or "।" after a question word.
 * Works on text in memory only.
 */
object HindiTwoPass {
    const val SPLIT = 30
    /** Declared first: the word lists below are normalised with it. */
    private val TOKEN_SPLIT = Regex("[^\\p{L}\\p{N}_\\u0900-\\u0963\\u0966-\\u097f]+")

    /** Log-probability of [text] under the cleanup model as a plain language model. */
    fun interface Scorer { fun score(text: String): Double }

    private val FILLERS = setOf("उम्म", "उम", "अं", "आं", "हम्म", "हम", "ह्म्म", "अ", "उह", "उहं", "एम", "आ")
    private val SOFT = setOf("मतलब", "वो", "तो", "यानी", "अच्छा", "ना", "हाँ", "हां", "बस", "जैसे", "कि", "और", "फिर",
        "ओके", "यार", "भी")
    private val EMPTY_WORDS = FILLERS + setOf("मतलब", "यानी", "वो")
    /** Only unambiguous cues: not सारी ("all"), बल्कि ("rather") or a bare नहीं (negation). */
    private val CUES = Regex("(^| )(रुको|रुकिए|माफ करना|माफ कीजिए|सॉरी|नहीं नहीं|मेरा मतलब|मेरा मतलब है|एक्चुअली|actually|" +
        "गलत बोला|ओह नहीं)( |$)")
    private val CUE_PHRASES = listOf("रुको", "रुकिए", "माफ करना", "माफ कीजिए", "सॉरी", "नहीं नहीं", "मेरा मतलब है",
        "मेरा मतलब", "एक्चुअली", "actually", "ओह नहीं").map { it.split(" ") }
    private val NUMBER_WORDS = ("शून्य एक दो तीन चार पांच छह छः सात आठ नौ दस ग्यारह बारह तेरह चौदह पंद्रह सोलह सत्रह अठारह " +
        "उन्नीस बीस तीस चालीस पचास साठ सत्तर अस्सी नब्बे सौ हजार लाख करोड़ करोड डेढ़ ढाई सवा साढ़े पहला दूसरा तीसरा चौथा")
        .split(" ").map { norm(it).joinToString(" ") }.toSet()
    /** Words Hindi doubles on purpose (साफ़ साफ़, अलग अलग, कभी कभी): never collapsed. */
    private val REDUP = ("अलग कभी बड़े बड़ी बड़ा छोटे छोटी साफ धीरे जल्दी बहुत थोड़ा थोड़ी कहीं कोई कुछ जब जहाँ जहां " +
        "जैसे साथ बार रोज घर तरह सब कौन क्या किस अपने अपनी अपना एक दो हर पास दूर ऊपर नीचे आगे पीछे गरम गर्म ठंडा हरे हरी " +
        "लाल पीले पीली अच्छे अच्छी नए नई पुराने कोने चलते बैठे देखते").split(" ").map { norm(it).joinToString(" ") }.toSet()
    private val QWORDS = ("क्या कब कहां कहाँ क्यों कैसे कौन किसका किसकी किसके किसे किसको कितना कितनी कितने कौनसा कौनसी " +
        "कौनसे").split(" ").flatMap { norm(it) }.toSet()
    private val TAG_Q = listOf("है ना", "हैं ना", "था ना", "थी ना", "थे ना", "हो ना", "है न", "था न")
        .map { q -> q.split(" ").flatMap { norm(it) }.joinToString(" ") }
    private const val MAX_FREE_DELETE = 3
    private const val CUE_REACH = 6
    private const val UNRELIABLE_MIN = 5
    private const val UNRELIABLE_SHARE = 0.25
    private const val PUNCT = "।॥.,!?;:"

    /** Tokens of one word for comparing spellings: no nukta, chandrabindu = anusvara, Devanagari digits as digits. */
    fun norm(w: String): List<String> {
        var s = Normalizer.normalize(w, Normalizer.Form.NFC).lowercase().replace('ँ', 'ं')
        s = Normalizer.normalize(s, Normalizer.Form.NFD).replace("़", "")
        s = Normalizer.normalize(s, Normalizer.Form.NFC)
        val sb = StringBuilder(s.length)
        for (ch in s) sb.append(if (ch in '०'..'९') '0' + (ch - '०') else ch)
        return TOKEN_SPLIT.split(sb).filter { it.isNotEmpty() }
    }

    private fun isNum(t: String) = t.any { it.isDigit() } || t in NUMBER_WORDS
    private fun words(s: String) = s.trim().split(Regex("\\s+")).filter { it.isNotEmpty() }

    private enum class Op { OK, SUB, DEL, INS }

    /** Word alignment of [a] to [b] (Levenshtein, the lab code's tie-breaking). */
    private fun align(a: List<String>, b: List<String>): List<Op> {
        val n = a.size; val m = b.size
        val d = Array(n + 1) { IntArray(m + 1) }
        for (i in 0..n) d[i][0] = i
        for (j in 0..m) d[0][j] = j
        for (i in 1..n) for (j in 1..m)
            d[i][j] = minOf(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + if (a[i - 1] != b[j - 1]) 1 else 0)
        val ops = ArrayList<Op>()
        var i = n; var j = m
        while (i > 0 || j > 0) {
            if (i > 0 && j > 0 && d[i][j] == d[i - 1][j - 1] + if (a[i - 1] != b[j - 1]) 1 else 0) {
                ops.add(if (a[i - 1] == b[j - 1]) Op.OK else Op.SUB); i--; j--
            } else if (i > 0 && d[i][j] == d[i - 1][j] + 1) { ops.add(Op.DEL); i-- } else { ops.add(Op.INS); j-- }
        }
        ops.reverse()
        return ops
    }

    /** Equal pieces of at most [n] words. */
    fun pieces(text: String, n: Int = SPLIT): List<String> {
        val w = words(text)
        if (w.size <= n) return listOf(text.trim())
        val k = (w.size + n - 1) / n
        val size = (w.size + k - 1) / k
        return w.chunked(size).map { it.joinToString(" ") }
    }

    fun onlyFillers(heard: String): Boolean {
        val toks = words(heard).flatMap { norm(it) }
        return toks.isNotEmpty() && toks.all { it in EMPTY_WORDS }
    }

    /** The cleanup output with the changes the gate doesn't allow undone (see the class comment). */
    fun gate(heard: String, out: String): String {
        val H = words(heard)
        val O = ArrayList<String>(); val SEP = ArrayList<String>()
        for (ch in out.trim()) {
            if (ch.isWhitespace()) {
                if (SEP.size < O.size) SEP.add("")
                SEP[SEP.size - 1] = SEP[SEP.size - 1] + ch
            } else if (SEP.size < O.size) O[O.size - 1] = O[O.size - 1] + ch else O.add(ch.toString())
        }
        while (SEP.size < O.size) SEP.add("")
        val ht = H.withIndex().flatMap { (i, w) -> norm(w).map { it to i } }
        val ot = O.withIndex().flatMap { (j, w) -> norm(w).map { it to j } }
        val hn = ht.map { it.first }
        val ops = align(hn, ot.map { it.first })
        val joined = hn.joinToString(" ")
        val m = CUES.find(joined)
        val late: Set<String> = if (m != null) {
            val before = joined.substring(0, m.range.last + 1).split(" ").filter { it.isNotEmpty() }.size
            hn.drop(before).toSet()
        } else emptySet()

        fun cueNear(i: Int): Boolean {
            val lo = maxOf(0, i - CUE_REACH); val hi = minOf(H.size, i + CUE_REACH + 1)
            return CUES.containsMatchIn(H.subList(lo, hi).flatMap { norm(it) }.joinToString(" "))
        }
        fun joins(t: String, k: Int): Boolean {
            for (s0 in (k - 5)..(k + 1)) {
                val acc = StringBuilder()
                for (mm in maxOf(0, s0) until minOf(hn.size, s0 + 6)) {
                    acc.append(hn[mm])
                    if (acc.toString() == t && mm > s0) return true
                }
            }
            return false
        }
        fun free(t: String, k: Int) = isNum(t) || joins(t, k)

        val badSub = HashMap<Int, MutableList<String>>()
        val badIns = HashSet<Int>()
        val delRuns = ArrayList<Pair<List<Int>, Int>>()
        var run = ArrayList<Int>()
        var hi = 0; var oi = 0
        for (op in ops) {
            if (op != Op.DEL && run.isNotEmpty()) { delRuns.add(run to oi); run = ArrayList() }
            when (op) {
                Op.OK -> { hi++; oi++ }
                Op.SUB -> {
                    val i = ht[hi].second; val j = ot[oi].second
                    val b = ot[oi].first
                    if (!(isNum(hn[hi]) || free(b, hi) || cueNear(i) || b in late)) badSub.getOrPut(j) { ArrayList() }.add(H[i])
                    hi++; oi++
                }
                Op.DEL -> { run.add(hi); hi++ }
                Op.INS -> { if (!free(ot[oi].first, hi)) badIns.add(ot[oi].second); oi++ }
            }
        }
        if (run.isNotEmpty()) delRuns.add(run to oi)
        val blocked = badIns.size + badSub.size
        if (blocked >= UNRELIABLE_MIN && blocked > UNRELIABLE_SHARE * O.size) return heard.trim()

        val restoreBefore = HashMap<Int, MutableList<String>>()
        for ((idx, at) in delRuns) {
            val raw = idx.map { ht[it].second }.toSortedSet().toList()
            val gone = idx.map { hn[it] }
            val j = if (at < ot.size) ot[at].second else O.size
            if (gone.any { isNum(it) }) {
                val near = ot.subList(maxOf(0, at - 3), minOf(ot.size, at + 3)).map { it.first }
                if (near.none { t -> t.any { it.isDigit() } }) restoreBefore.getOrPut(j) { ArrayList() }.addAll(raw.map { H[it] })
                continue
            }
            // Half of a deliberate doubling removed ("साफ़ साफ़" -> "साफ़"): put it back.
            if (idx.any { k -> hn[k] in REDUP && ((k > 0 && hn[k - 1] == hn[k]) || (k + 1 < hn.size && hn[k + 1] == hn[k])) }) {
                restoreBefore.getOrPut(j) { ArrayList() }.addAll(raw.map { H[it] })
                continue
            }
            val content = gone.filter { it !in FILLERS && it !in SOFT && it.length > 1 }
            val around = (hn.subList(maxOf(0, idx[0] - gone.size - 4), idx[0]) +
                hn.subList(minOf(hn.size, idx.last() + 1), minOf(hn.size, idx.last() + 1 + gone.size + 4))).toSet()
            val restart = content.count { it in around } * 2 >= content.size
            if (content.size < MAX_FREE_DELETE || restart || CUES.containsMatchIn(gone.joinToString(" ")) ||
                cueNear(raw.first()) || cueNear(raw.last())) continue
            restoreBefore.getOrPut(j) { ArrayList() }.addAll(raw.map { H[it] })
        }
        val outW = ArrayList<String>(); val outS = ArrayList<String>()
        fun put(w: String, sep: String = " ") {
            if (outS.isNotEmpty() && outS.last().isEmpty()) outS[outS.size - 1] = " "
            outW.add(w); outS.add(sep)
        }
        fun tail(w: String) = w.takeLastWhile { it in PUNCT }
        for ((j, w) in O.withIndex()) {
            restoreBefore[j]?.forEach { put(it) }
            val back = badSub[j]
            if (back != null) {
                for (h in back) if (outW.isEmpty() || outW.last() != h) put(h)
                val t = tail(w)
                if (t.isNotEmpty() && outW.isNotEmpty() && tail(outW.last()).isEmpty()) outW[outW.size - 1] = outW.last() + t
                if (outS.isNotEmpty()) outS[outS.size - 1] = SEP[j]
            } else if (j !in badIns) put(w, SEP[j])
            else if ('\n' in SEP[j] && outS.isNotEmpty()) outS[outS.size - 1] = SEP[j]
        }
        restoreBefore[O.size]?.forEach { put(it) }
        return outW.indices.joinToString("") { outW[it] + outS[it] }.trim()
    }

    /** (start, end) word positions of the last correction cue in normalised words, or null. */
    private fun findCue(nt: List<String>): Pair<Int, Int>? {
        var found: Pair<Int, Int>? = null
        for (i in nt.indices) for (p in CUE_PHRASES) {
            if (i + p.size <= nt.size && nt.subList(i, i + p.size) == p) { found = i to i + p.size; break }
        }
        return found
    }

    /** "A cue B" -> A with B put where it belongs (anchored on B's last word in A), or null. */
    fun applyCorrection(heard: String): String? {
        val toks = words(heard)
        val nt = toks.map { norm(it).joinToString(" ") }
        val cue = findCue(nt) ?: return null
        val a = toks.subList(0, cue.first); val b = toks.subList(cue.second, toks.size)
        val nA = nt.subList(0, cue.first)
        if (a.isEmpty() || b.isEmpty() || b.size > 4) return null
        val last = norm(b.last()).joinToString(" ")
        val k = nA.lastIndexOf(last)
        if (k < 0) return null
        val lo = maxOf(0, k - b.size + 1)
        return (a.subList(0, lo) + b + a.subList(k + 1, a.size)).joinToString(" ")
    }

    private fun collapseRepeats(lm: Scorer, text: String): String {
        val w = words(text).toMutableList()
        var i = 1
        while (i < w.size) {
            val t = norm(w[i]).joinToString(" ")
            if (t.isNotEmpty() && t == norm(w[i - 1]).joinToString(" ") && !isNum(t) && t !in REDUP && w[i].trim { it in PUNCT }.isNotEmpty()) {
                val one = w.toMutableList().also { it.removeAt(i) }
                if (lm.score(one.joinToString(" ")) > lm.score(w.joinToString(" "))) { w.clear(); w.addAll(one); continue }
            }
            i++
        }
        return w.joinToString(" ")
    }

    /**
     * The full chain after the model cleaned each piece of [heard] ([cleaned]: the pieces' outputs joined): gate,
     * correction, repeats, question mark. "" means nothing was said but filler.
     */
    fun finish(heard: String, cleaned: String, lm: Scorer?): String {
        if (onlyFillers(heard)) return ""
        var out = gate(heard, cleaned)
        val toks = words(heard)
        val cue = findCue(toks.map { norm(it).joinToString(" ") })
        if (cue != null) {
            val fixed = applyCorrection(heard)
            if (fixed != null) {
                out = fixed.trimEnd { c -> c == ' ' || c in PUNCT } + "।"
            } else if (lm != null && cue.first >= 1 && CUES.containsMatchIn(words(out).flatMap { norm(it) }.joinToString(" "))) {
                // The model left the cue in: try dropping the word before it, if the model reads that better.
                val alt = (toks.subList(0, cue.first - 1) + toks.subList(cue.second, toks.size)).joinToString(" ")
                val cur = out.trimEnd { c -> c == ' ' || c in PUNCT }
                if (lm.score(alt) / maxOf(1, words(alt).size) > lm.score(cur) / maxOf(1, words(out).size)) {
                    out = alt.trimEnd { c -> c == ' ' || c in PUNCT } + "।"
                }
            }
        }
        if (lm != null) out = collapseRepeats(lm, out)
        val plain = toks.flatMap { norm(it) }
        val joined = plain.joinToString(" ")
        if (out.trimEnd { it == ' ' || it in PUNCT }.isNotEmpty() && TAG_Q.any { joined.endsWith(it) }) {
            return out.trimEnd { it == ' ' || it in PUNCT } + "?"
        }
        val body = out.trimEnd { it == ' ' || it in "।.?!" }
        if (lm != null && body.isNotEmpty() && plain.isNotEmpty() &&
            (plain.any { it in QWORDS } || plain.last() == "ना" || plain.last() == "न")) {
            return if (lm.score("$body?") > lm.score("$body।")) "$body?" else "$body।"
        }
        return out
    }
}
