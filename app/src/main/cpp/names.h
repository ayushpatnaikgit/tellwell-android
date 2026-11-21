// Names for speech recognition: which names and terms to favour, and the text rules that keep that safe.
// A C++ port of the Mac app's TermBoost.swift and ScreenWords.swift (TellWell PR #25), with the same thresholds; see
// that PR for how they were tuned on synthetic and real speech. Pure logic: no model calls, nothing logged or stored.
//
// Text is UTF-8. Matching uses ASCII letters and digits (as on the Mac); the SentencePiece word marker "▁" is 3 bytes
// and is compared bytewise.
#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tw {

inline const std::string kMark = "\xE2\x96\x81";   // "▁", the word-start marker

inline std::string lower(std::string s) {
    for (char& c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
    return s;
}
inline bool startsWith(const std::string& s, const std::string& p) { return s.compare(0, p.size(), p) == 0; }
inline bool isAlnum(char c) { return std::isalnum(static_cast<unsigned char>(c)) != 0; }

/// Lowercase ASCII letters and digits only.
inline std::string letters(const std::string& s) {
    std::string r;
    for (char c : s) if (isAlnum(c) && static_cast<unsigned char>(c) < 128) r += char(std::tolower(static_cast<unsigned char>(c)));
    return r;
}

/// Python's difflib.SequenceMatcher(None, a, b).ratio() for short strings (no autojunk below 200 elements).
inline double ratio(const std::string& a, const std::string& b) {
    if (a.empty() && b.empty()) return 1.0;
    std::unordered_map<char, std::vector<int>> b2j;
    for (int j = 0; j < int(b.size()); j++) b2j[b[j]].push_back(j);
    if (b.size() >= 200) {
        size_t ntest = b.size() / 100 + 1;
        for (auto it = b2j.begin(); it != b2j.end();) it = it->second.size() > ntest ? b2j.erase(it) : std::next(it);
    }
    auto longest = [&](int alo, int ahi, int blo, int bhi) {
        int besti = alo, bestj = blo, bestsize = 0;
        std::unordered_map<int, int> j2len;
        for (int i = alo; i < ahi; i++) {
            std::unordered_map<int, int> next;
            auto f = b2j.find(a[i]);
            if (f != b2j.end()) {
                for (int j : f->second) {
                    if (j < blo) continue;
                    if (j >= bhi) break;
                    auto p = j2len.find(j - 1);
                    int k = (p == j2len.end() ? 0 : p->second) + 1;
                    next[j] = k;
                    if (k > bestsize) { besti = i - k + 1; bestj = j - k + 1; bestsize = k; }
                }
            }
            j2len.swap(next);
        }
        while (besti > alo && bestj > blo && a[besti - 1] == b[bestj - 1]) { besti--; bestj--; bestsize++; }
        while (besti + bestsize < ahi && bestj + bestsize < bhi && a[besti + bestsize] == b[bestj + bestsize]) bestsize++;
        return std::make_tuple(besti, bestj, bestsize);
    };
    int matches = 0;
    std::vector<std::tuple<int, int, int, int>> queue{{0, int(a.size()), 0, int(b.size())}};
    while (!queue.empty()) {
        auto [alo, ahi, blo, bhi] = queue.back();
        queue.pop_back();
        auto [i, j, k] = longest(alo, ahi, blo, bhi);
        if (k > 0) {
            matches += k;
            if (alo < i && blo < j) queue.push_back({alo, i, blo, j});
            if (i + k < ahi && j + k < bhi) queue.push_back({i + k, ahi, j + k, bhi});
        }
    }
    return 2.0 * matches / double(a.size() + b.size());
}

/// Rough sound-alike key: similar consonants merged, vowels after the first letter dropped (LoRA = Laura).
/// `mergeVoicing` also treats d/t, b/p, g/k and v/f as one sound ("Podnamic" ~ Patnaik).
inline std::string sound(const std::string& text, bool mergeVoicing = false) {
    static const std::vector<std::pair<std::string, std::string>> sounds = {
        {"wh", "h"}, {"ph", "f"}, {"ck", "k"}, {"qu", "k"}, {"q", "k"}, {"c", "k"}, {"x", "ks"}, {"z", "s"},
        {"w", "v"}, {"dh", "d"}, {"th", "t"}, {"gh", "g"}};
    std::string s = letters(text);
    for (auto& [a, b] : sounds) {
        std::string out;
        for (size_t i = 0; i < s.size();) {
            if (s.compare(i, a.size(), a) == 0) { out += b; i += a.size(); } else out += s[i++];
        }
        s = out;
    }
    if (s.empty()) return s;
    char first = std::string("aeiou").find(s[0]) != std::string::npos ? 'a' : s[0];
    std::string key(1, first);
    for (size_t i = 1; i < s.size(); i++) if (std::string("aeiouy").find(s[i]) == std::string::npos) key += s[i];
    std::string out;
    for (char c : key) {
        if (mergeVoicing) c = c == 'd' ? 't' : c == 'b' ? 'p' : c == 'g' ? 'k' : c == 'v' ? 'f' : c;
        if (out.empty() || out.back() != c) out += c;
    }
    return out;
}

/// Mean of sound-alike and spelling similarity; either alone matched too much.
inline double similarity(const std::string& word, const std::string& spoken, bool mergeVoicing = false) {
    double byLetters = ratio(letters(word), letters(spoken));
    std::string a = sound(word, mergeVoicing), b = sound(spoken, mergeVoicing);
    double bySound = a.size() >= 2 ? ratio(a, b) : 0.0;
    return (byLetters + bySound) / 2;
}

/// An English word list (one word per line; the Mac uses /usr/share/dict/words, Android ships the same list).
struct Dictionary {
    std::unordered_set<std::string> words;

    bool loaded() const { return words.size() > 1000; }

    bool contains(const std::string& word) const {
        std::string w = lower(word);
        if (words.count(w)) return true;
        static const char* suffixes[] = {"ings", "ing", "ers", "er", "ed", "es", "s", "ly"};
        for (const char* suf : suffixes) {
            std::string s(suf);
            if (w.size() < s.size() || w.compare(w.size() - s.size(), s.size(), s) != 0) continue;
            std::string stem = w.substr(0, w.size() - s.size());
            if (stem.size() < 3) continue;
            if (words.count(stem) || words.count(stem + "e")) return true;
            if (stem.size() > 3 && stem.back() == stem[stem.size() - 2] && words.count(stem.substr(0, stem.size() - 1))) return true;
        }
        return false;
    }
};

inline bool isUpperWord(const std::string& s) {
    bool any = false;
    for (char c : s) {
        if (std::islower(static_cast<unsigned char>(c))) return false;
        if (std::isupper(static_cast<unsigned char>(c))) any = true;
    }
    return any;
}

/// 2 = clearly a name or term (LoRA, MLX, Qwen3, Siddhant), 1 = unknown lowercase word (pgvector), 0 = skip.
inline int strength(const std::string& word, const Dictionary& dict) {
    static const std::set<std::string> skip = {"ok", "id", "ui", "url", "urls", "http", "https", "www", "com", "md", "py",
        "sh", "js", "ts", "json", "html", "png", "jpg", "pdf", "txt", "app", "ctrl", "cmd", "esc", "tab", "pm", "am"};
    int nLetters = 0;
    bool digit = false, camel = false;
    for (size_t i = 0; i < word.size(); i++) {
        unsigned char c = static_cast<unsigned char>(word[i]);
        if (c < 128 && std::isalpha(c)) nLetters++;
        if (std::isdigit(c)) digit = true;
        if (i > 0 && std::islower(static_cast<unsigned char>(word[i - 1])) && std::isupper(c)) camel = true;
    }
    if (nLetters < 2 || skip.count(lower(word)) || word.find('_') != std::string::npos || word.find('.') != std::string::npos) return 0;
    static const char* exts[] = {"md", "sh", "py", "js", "ts", "txt", "json", "html", "swift", "yml", "yaml", "toml"};
    for (const char* e : exts) {      // "READMEmd", "setupsh": a file name with its dot lost
        std::string x(e);
        if (word.size() >= x.size() + 3 && lower(word).compare(word.size() - x.size(), x.size(), x) == 0) {
            std::string stem = word.substr(0, word.size() - x.size());
            bool allAlpha = std::all_of(stem.begin(), stem.end(), [](char c) { return std::isalpha(static_cast<unsigned char>(c)); });
            if (allAlpha && (isUpperWord(stem) || dict.contains(stem))) return 0;
        }
    }
    if (camel || digit) return 2;
    if (isUpperWord(word)) return word.size() <= 8 ? 2 : 0;
    if (nLetters < 4 || dict.contains(word)) return 0;
    if (std::isupper(static_cast<unsigned char>(word[0]))) return 2;
    return nLetters >= 5 ? 1 : 0;
}

/// Words of `text` split on anything but letters, digits and apostrophes.
inline std::vector<std::string> wordsOf(const std::string& text) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : text) {
        if (isAlnum(c) || c == '\'') cur += c;
        else if (!cur.empty()) { out.push_back(cur); cur.clear(); }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

/// Names and terms in text the user can see (the field being typed into): the unusual words, strongest first.
/// Unlike screen OCR, this text is exact, so a word seen once counts.
inline std::vector<std::string> nameWords(const std::string& text, const Dictionary& dict, size_t limit = 40) {
    std::map<std::string, std::pair<int, std::string>> best;   // lowercased -> (strength, spelling)
    std::vector<std::string> order;
    for (auto& raw : wordsOf(text)) {
        std::string w = raw;
        if (w.size() > 2 && w.compare(w.size() - 2, 2, "'s") == 0) w = w.substr(0, w.size() - 2);
        while (!w.empty() && w.back() == '\'') w.pop_back();
        while (!w.empty() && w.front() == '\'') w.erase(w.begin());
        int s = w.empty() ? 0 : strength(w, dict);
        if (s == 0) continue;
        std::string key = lower(w);
        if (!best.count(key)) order.push_back(key);
        if (best[key].first < s) best[key] = {s, w};
    }
    std::stable_sort(order.begin(), order.end(), [&](const std::string& a, const std::string& b) { return best[a].first > best[b].first; });
    std::vector<std::string> out;
    for (auto& k : order) { if (out.size() >= limit) break; out.push_back(best[k].second); }
    return out;
}

/// The list to boost, in priority order; one spelling per term, at least 3 letters or digits, at most `limit`.
inline std::vector<std::string> termList(const std::vector<std::vector<std::string>>& sources, size_t limit = 100) {
    std::vector<std::string> out;
    std::unordered_set<std::string> seen;
    for (auto& source : sources)
        for (auto& t : source) {
            if (letters(t).size() < 3 || !seen.insert(lower(t)).second) continue;
            out.push_back(t);
            if (out.size() >= limit) return out;
        }
    return out;
}

/// Of many candidate names (contacts), the few that sound like a 1-3 word stretch of what was heard, best first.
/// Boosting thousands of names would put names everywhere; this keeps the list to the ones that could have been said.
inline std::vector<std::string> soundingLike(const std::vector<std::string>& names, const std::string& heard,
                                             double threshold, size_t limit) {
    std::vector<std::string> spoken = wordsOf(heard);
    std::vector<std::string> spans;
    for (size_t n = 1; n <= 3; n++)
        for (size_t i = 0; i + n <= spoken.size(); i++) {
            std::string s;
            for (size_t k = i; k < i + n; k++) s += (k > i ? " " : "") + spoken[k];
            spans.push_back(s);
        }
    std::vector<std::pair<double, std::string>> scored;
    for (auto& name : names) {
        std::string w = letters(name);
        if (w.size() < 3) continue;
        double best = 0;
        for (auto& span : spans) {
            std::string l = letters(span);
            if (l == w) { best = 0; break; }                       // already heard as spelled: nothing to fix
            if (std::abs(int(l.size()) - int(w.size())) > std::max(3, int(w.size()) / 2)) continue;
            best = std::max(best, similarity(name, span, true));
        }
        if (best >= threshold) scored.push_back({best, name});
    }
    std::stable_sort(scored.begin(), scored.end(), [](auto& a, auto& b) { return a.first > b.first; });
    std::vector<std::string> out;
    for (auto& s : scored) { if (out.size() >= limit) break; out.push_back(s.second); }
    return out;
}

// ---------------------------------------------------------------- boosting

struct Token { int id; int frame; float prob = 1.f; };   // frame: absolute, in 10 ms feature frames

struct Partial {
    int term, pos;   // which term, and how many bytes of its key have been spelled
    bool operator<(const Partial& o) const { return term != o.term ? term < o.term : pos < o.pos; }
    bool operator==(const Partial& o) const { return term == o.term && pos == o.pos; }
};

/// Names to favour in the boosted pass, with the settings tuned on the Mac (beam 4, bonus 4, gate 0.8).
struct Boost {
    std::vector<std::string> terms;
    std::vector<std::string> keys;         // "▁" + lowercased term, spaces as "▁"
    float beta = 4.0f;                     // log-probability bonus per matching piece
    float floor = -6.0f;                   // pieces less likely than this are never boosted
    float beamFloor = -10.0f;              // in beam search, pieces less likely than this are never tried
    int beam = 4;
    float acceptConfidence = 0.8f;         // only words heard with less confidence may be replaced by a term
    static constexpr double minSimilarity = 0.5;

    explicit Boost(std::vector<std::string> t = {}) {
        for (auto& term : t) {
            if (term.find_first_not_of(' ') == std::string::npos) continue;
            terms.push_back(term);
            std::string k = kMark;
            for (char c : lower(term)) { if (c == ' ') k += kMark; else k += c; }
            keys.push_back(k);
        }
    }
    bool on() const { return !keys.empty() && beta > 0; }

    /// Partials after emitting `piece` (lowercased), or empty if it spells no listed term from `active`.
    std::set<Partial> advance(const std::set<Partial>& active, const std::string& piece) const {
        std::set<Partial> next;
        if (piece.empty()) return next;
        auto tryFrom = [&](const Partial& p) {
            const std::string& key = keys[p.term];
            if (p.pos + piece.size() <= key.size() && key.compare(p.pos, piece.size(), piece) == 0)
                next.insert({p.term, int(p.pos + piece.size())});
        };
        for (auto& p : active) tryFrom(p);
        if (startsWith(piece, kMark)) for (int i = 0; i < int(keys.size()); i++) tryFrom({i, 0});
        return next;
    }
    bool complete(const std::set<Partial>& s) const {
        for (auto& p : s) if (p.pos == int(keys[p.term].size())) return true;
        return false;
    }
};

/// Word-pieces as matching sees them: lowercased (they already mark word starts with "▁").
inline std::vector<std::string> piecesOf(const std::vector<std::string>& vocab) {
    std::vector<std::string> out;
    for (auto& v : vocab) out.push_back(lower(v));
    return out;
}

inline bool startsWord(const Token& t, const std::vector<std::string>& pieces) { return startsWith(pieces[t.id], kMark); }

inline std::string stripMarks(std::string s) {
    for (size_t p; (p = s.find(kMark)) != std::string::npos;) s.replace(p, kMark.size(), " ");
    size_t a = s.find_first_not_of(' '), b = s.find_last_not_of(' ');
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}

/// The ordinary tokens, with each boosted word that spells a whole listed term swapped in for the ordinary words heard
/// at the same time, if they sound like it, the recogniser was unsure of them, and nothing would be dropped.
inline std::vector<Token> acceptTerms(const Boost& boost, const std::vector<Token>& stock, const std::vector<Token>& boosted,
                                      const std::vector<std::string>& pieces, int* swapped = nullptr) {
    struct Word { size_t from, to; int frame; };
    auto words = [&](const std::vector<Token>& v) {
        std::vector<Word> out;
        for (size_t i = 0; i < v.size(); i++) {
            if (i > 0 && !startsWord(v[i], pieces)) continue;
            size_t end = i + 1;
            while (end < v.size() && !startsWord(v[end], pieces)) end++;
            out.push_back({i, end, v[i].frame});
        }
        return out;
    };
    auto spell = [&](const std::vector<Token>& v, size_t a, size_t b) {
        std::string s;
        for (size_t i = a; i < b; i++) s += pieces[v[i].id];
        return s;
    };
    std::set<std::string> whole;
    for (auto& k : boost.keys) whole.insert(k.substr(kMark.size()));
    std::vector<Token> out = stock;
    auto bw = words(boosted);
    for (int k = int(bw.size()) - 1; k >= 0; k--) {
        std::string spelled = spell(boosted, bw[k].from, bw[k].to);
        while (startsWith(spelled, kMark)) spelled = spelled.substr(kMark.size());
        std::string bare = spelled;
        while (!bare.empty() && std::ispunct(static_cast<unsigned char>(bare.back()))) bare.pop_back();
        if (!whole.count(bare)) continue;
        int start = bw[k].frame, end = k + 1 < int(bw.size()) ? bw[k + 1].frame : 1 << 30;
        std::vector<Word> sw;
        for (auto& w : words(out)) if (w.frame >= start - 12 && w.frame < end - 1) sw.push_back(w);   // 0.12 s, 0.01 s
        if (sw.empty()) continue;                                                                       // never into silence
        const Word& last = sw.back();
        // Only the heard words that sound like the term: the most similar span ending at the last word.
        double bestScore = -1;
        size_t bestIndex = 0;
        for (size_t i = 0; i < sw.size(); i++) {
            double s = similarity(stripMarks(spelled), stripMarks(spell(out, sw[i].from, last.to)));
            if (s >= bestScore) { bestScore = s; bestIndex = i; }      // ties: the later (shorter) span, as on the Mac
        }
        if (bestScore < Boost::minSimilarity) continue;
        const Word& first = sw[bestIndex];
        // The replaced words must start where the name does; if the name began inside the word before them
        // ("Ayushpat Nayak" -> "Patnaik"), swapping would type part of it twice ("Ayushpat Patnaik").
        if (first.frame > start + 16) continue;                                                        // 0.16 s
        std::string replaced = spell(out, first.from, last.to);
        while (startsWith(replaced, kMark)) replaced = replaced.substr(kMark.size());
        if (replaced == spelled) continue;
        float minProb = 1.f;
        for (size_t i = first.from; i < last.to; i++) minProb = std::min(minProb, out[i].prob);
        if (minProb >= boost.acceptConfidence) continue;
        // Never drop words: skip if the term was already heard among them, or they hold much more than the term.
        std::string term = letters(spelled);
        size_t heardLetters = 0;
        bool alreadyHeard = false;
        for (auto& w : wordsOf(stripMarks(replaced))) {
            heardLetters += letters(w).size();
            if (letters(w) == term) alreadyHeard = true;
        }
        if (alreadyHeard || heardLetters > term.size() + std::max<size_t>(2, term.size() / 3)) continue;
        std::vector<Token> repl(boosted.begin() + long(bw[k].from), boosted.begin() + long(bw[k].to));
        out.erase(out.begin() + long(first.from), out.begin() + long(last.to));
        out.insert(out.begin() + long(first.from), repl.begin(), repl.end());
        if (swapped) (*swapped)++;
    }
    return out;
}

/// The runs of heard words (first, last word index, counting words as acceptTerms does) a listed term could replace:
/// a run that includes a word heard with less than acceptConfidence, holds not much more than the term, and sounds
/// like it (slightly loosened). The boosted pass runs only around these; anywhere else its result would be discarded.
inline std::vector<std::pair<int, int>> candidateRuns(const Boost& boost, const std::vector<Token>& stock,
                                                      const std::vector<std::string>& pieces) {
    struct Word { std::string text; size_t letters; bool unsure; };
    std::vector<Word> ws;
    for (size_t i = 0; i < stock.size(); i++) {
        bool unsure = stock[i].prob < boost.acceptConfidence;
        if (i == 0 || startsWord(stock[i], pieces)) { ws.push_back({pieces[stock[i].id], 0, unsure}); continue; }
        ws.back().text += pieces[stock[i].id];
        ws.back().unsure |= unsure;
    }
    bool anyUnsure = false;
    for (auto& w : ws) { w.text = " " + stripMarks(w.text); w.letters = letters(w.text).size(); anyUnsure |= w.unsure; }
    std::vector<std::pair<int, int>> runs;
    if (!anyUnsure) return runs;
    struct Term { std::string letters, sound; size_t cap; };
    std::vector<Term> terms;
    size_t longest = 0;
    for (auto& t : boost.terms) {
        std::string l = letters(lower(t));
        terms.push_back({l, sound(lower(t)), l.size() + std::max<size_t>(2, l.size() / 3)});
        longest = std::max(longest, terms.back().cap);
    }
    auto bound = [](size_t a, size_t b) { return a + b == 0 ? 0.0 : 2.0 * double(std::min(a, b)) / double(a + b); };
    for (int i = 0; i < int(ws.size()); i++) {
        size_t count = 0;
        bool unsure = false;
        std::string text;
        for (int j = i; j < int(ws.size()); j++) {
            count += ws[j].letters;
            if (count > longest) break;
            unsure |= ws[j].unsure;
            text += ws[j].text;
            if (!unsure) continue;
            std::string l = letters(text), s = sound(text);
            for (auto& t : terms) {
                if (count > t.cap) continue;
                double soundBound = t.sound.size() >= 2 ? bound(t.sound.size(), s.size()) : 0.0;
                if ((bound(t.letters.size(), l.size()) + soundBound) / 2 < Boost::minSimilarity - 0.05) continue;
                double sim = (ratio(t.letters, l) + (t.sound.size() >= 2 ? ratio(t.sound, s) : 0.0)) / 2;
                if (sim >= Boost::minSimilarity - 0.05) { runs.push_back({i, j}); break; }
            }
        }
    }
    return runs;
}

/// Replaces 1-3 heard words with a listed name when the recogniser was unsure of them (< 0.6), at least one isn't an
/// English word, and they sound like the name (>= 0.7, voicing merged): "Putnamic" -> Patnaik. Spans that already
/// hold a listed name are skipped, so no word is dropped. Does nothing without a loaded dictionary.
inline std::string correctNames(const std::vector<std::pair<std::string, float>>& words, const std::vector<std::string>& terms,
                                const Dictionary& dict, int* fixed = nullptr) {
    constexpr double kSimilarity = 0.7;
    constexpr float kConfidence = 0.6f;
    static const std::set<std::string> fillers = {"uh", "um", "uhm", "hmm", "mm", "er", "ah"};
    std::string text;
    struct Tok { size_t at, len; std::string word; float conf; };
    std::vector<Tok> toks;
    for (size_t i = 0; i < words.size(); i++) {
        if (i > 0) text += " ";
        size_t base = text.size();
        const std::string& w = words[i].first;
        for (size_t j = 0; j < w.size();) {
            if (!(isAlnum(w[j]) || w[j] == '\'')) { j++; continue; }
            size_t k = j;
            while (k < w.size() && (isAlnum(w[k]) || w[k] == '\'')) k++;
            toks.push_back({base + j, k - j, w.substr(j, k - j), words[i].second});
            j = k;
        }
        text += w;
    }
    if (!dict.loaded()) return text;
    struct Cand { double score; int start, end; std::string term; };
    std::vector<Cand> cands;
    for (int n = 1; n <= 3; n++)
        for (int i = 0; i + n <= int(toks.size()); i++) {
            bool foreign = false, filler = false;
            float conf = 1.f;
            std::string spoken;
            for (int k = i; k < i + n; k++) {
                foreign |= !dict.contains(toks[k].word);
                filler |= fillers.count(lower(toks[k].word)) > 0;
                conf = std::min(conf, toks[k].conf);
                spoken += (k > i ? " " : "") + toks[k].word;
            }
            if (!foreign || filler || conf >= kConfidence) continue;
            std::string heard = letters(spoken);
            if (heard.size() < 4) continue;
            for (auto& term : terms) {
                std::string l = letters(term);
                if (l.size() < 3) continue;
                bool holds = false;
                for (int k = i; k < i + n; k++) holds |= letters(toks[k].word) == l;
                if (holds || std::abs(int(l.size()) - int(heard.size())) > std::max(2, int(l.size()) / 3)) continue;
                double s = similarity(term, spoken, true);
                if (s >= kSimilarity) cands.push_back({s, i, i + n, term});
            }
        }
    std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) {
        return std::tie(a.score, a.start, a.end, a.term) > std::tie(b.score, b.start, b.end, b.term);
    });
    std::vector<bool> used(toks.size(), false);
    std::vector<Cand> chosen;
    for (auto& c : cands) {
        bool free = true;
        for (int k = c.start; k < c.end; k++) free &= !used[k];
        if (!free) continue;
        for (int k = c.start; k < c.end; k++) used[k] = true;
        chosen.push_back(c);
    }
    std::sort(chosen.begin(), chosen.end(), [](const Cand& a, const Cand& b) { return a.start > b.start; });
    for (auto& c : chosen) {
        size_t a = toks[c.start].at, b = toks[c.end - 1].at + toks[c.end - 1].len;
        text.replace(a, b - a, c.term);
    }
    if (fixed) *fixed += int(chosen.size());
    return text;
}

/// The listed spelling for any whole word that matches a term ignoring case ("grafana" -> "Grafana").
inline std::string spellTerms(std::string text, std::vector<std::string> terms) {
    std::stable_sort(terms.begin(), terms.end(), [](auto& a, auto& b) { return a.size() > b.size(); });
    auto boundary = [](char c) { return !(isAlnum(c) || c == '_' || c == '-'); };
    for (auto& term : terms) {
        if (term.empty()) continue;
        std::string lt = lower(term);
        std::string lowered = lower(text);
        for (size_t p = lowered.find(lt); p != std::string::npos; p = lowered.find(lt, p + term.size())) {
            bool left = p == 0 || boundary(text[p - 1]);
            bool right = p + term.size() >= text.size() || boundary(text[p + term.size()]);
            if (left && right) { text.replace(p, term.size(), term); lowered = lower(text); }
        }
    }
    return text;
}

}  // namespace tw
