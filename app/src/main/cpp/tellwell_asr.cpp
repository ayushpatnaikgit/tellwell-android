// Speech recognition on the Qualcomm NPU: the Parakeet TDT 0.6B v2 encoder, statically quantised (8-bit weights,
// 16-bit activations) for a fixed 10 s window, run by ONNX Runtime's QNN execution provider. The decoder and joiner
// run on the CPU with the same ONNX Runtime. Features match the model's front end (128 log-mel, per-feature
// normalised). Audio and text only pass through memory; nothing is logged or written except the compiled model.
//
// ONNX Runtime with QNN is a separate library (libonnxruntime_qnn.so) from the one sherpa-onnx uses, opened by path
// and used only through its C API table, so the two never clash. The same code runs the CPU encoder on phones
// without a usable NPU, so both paths decode the same way.
//
// Names (names.h): words in the text field being typed into and, if the user turned it on, contact names that sound
// like what was heard, steer a second, boosted beam-search pass and a correction of unsure words. Both stay on the
// phone (the app has no internet permission) and are never logged.
#include <dlfcn.h>
#ifdef __ANDROID__
#include <android/log.h>
#include <jni.h>
#define TW_WARN(...) __android_log_print(ANDROID_LOG_WARN, "TellWellASR", __VA_ARGS__)
#else
#include <cstdio>
#define TW_WARN(...) (std::fprintf(stderr, __VA_ARGS__), std::fputc('\n', stderr))
#endif

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "kaldi-native-fbank/csrc/online-feature.h"
#include "names.h"
#include "ort/onnxruntime_c_api.h"

namespace {

constexpr int kFrames = 1000;     // encoder window: 10 s of 10 ms feature frames
constexpr int kStride = 700;      // windows overlap by 3 s, as on the Mac
constexpr int kTolerance = 40;    // 0.4 s: the same word's start differs slightly between windows
constexpr int kSubsample = 8;     // feature frames per encoder frame
constexpr int kBlank = 1024;
constexpr int kDurations[] = {0, 1, 2, 3, 4};

std::mutex mu;
const OrtApi* api = nullptr;
OrtEnv* env = nullptr;
OrtSession* encoder = nullptr;
OrtSession* decoder = nullptr;
OrtSession* joiner = nullptr;
OrtMemoryInfo* cpu = nullptr;
std::vector<std::string> tokens;
bool encoderOnNpu = false;
std::vector<std::string> pieces;      // tokens lowercased, for matching names
tw::Dictionary dictionary;            // English words: a name correction never replaces one
std::vector<std::string> contacts;    // contact names, only if the user turned it on; empty otherwise
// Contacts: a name is tried only if it sounds this much like something heard, at most a few per window, and only in the
// boosted pass (gated by the recogniser's confidence), not the correction of unsure words. Measured with ~220 invented
// contact names on 700 public real-speech clips: 3 clips worse (looser settings: 7-15); synthetic names 31 -> 53/120.
double contactSimilarity = 0.75;
size_t contactLimit = 5;
bool correctWithContacts = false;

std::string check(OrtStatus* s) {
    if (!s) return "";
    std::string m = api->GetErrorMessage(s);
    api->ReleaseStatus(s);
    return m.empty() ? "error" : m;
}
std::string check(const std::string& s) { return s; }
#define TRY(x) do { std::string e_ = check(x); if (!e_.empty()) return e_; } while (0)

#ifdef __ANDROID__
std::string str(JNIEnv* e, jstring s) {
    if (!s) return "";
    const char* c = e->GetStringUTFChars(s, nullptr);
    std::string r(c);
    e->ReleaseStringUTFChars(s, c);
    return r;
}
#endif

bool exists(const std::string& p) { return std::ifstream(p).good(); }

std::string loadRuntime(const std::string& lib) {
    if (api) return "";
    void* h = dlopen((lib + "/libonnxruntime_qnn.so").c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) return std::string("dlopen: ") + dlerror();
    auto base = reinterpret_cast<const OrtApiBase* (*)()>(dlsym(h, "OrtGetApiBase"));
    if (!base) return "no OrtGetApiBase";
    api = base()->GetApi(ORT_API_VERSION);
    if (!api) return "ORT API version mismatch";
    TRY(api->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "tellwell", &env));
    TRY(api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &cpu));
    return "";
}

std::string loadDecoder(const std::string& dec, const std::string& joi, const std::string& tok) {
    if (decoder && joiner && tokens.size() > kBlank) return "";   // already loaded (NPU after CPU)
    // Decoder and joiner: small, run once per token, on one CPU thread.
    OrtSessionOptions* so = nullptr;
    TRY(api->CreateSessionOptions(&so));
    TRY(api->SetIntraOpNumThreads(so, 1));
    std::string err = check(api->CreateSession(env, dec.c_str(), so, &decoder));
    if (err.empty()) err = check(api->CreateSession(env, joi.c_str(), so, &joiner));
    api->ReleaseSessionOptions(so);
    if (!err.empty()) return "decoder: " + err;

    std::ifstream in(tok);
    std::string line;
    tokens.clear();
    while (std::getline(in, line)) {
        auto sp = line.rfind(' ');
        if (sp == std::string::npos) continue;
        size_t id = std::stoul(line.substr(sp + 1));
        if (tokens.size() <= id) tokens.resize(id + 1);
        tokens[id] = line.substr(0, sp);
    }
    pieces = tw::piecesOf(tokens);
    return tokens.size() > kBlank ? "" : "bad tokens file";
}

// The speech encoder on the CPU (phones without a usable NPU), with the same decoding as the NPU path.
std::string loadCpu(const std::string& lib, const std::string& enc, const std::string& dec, const std::string& joi,
                    const std::string& tok) {
    TRY(loadRuntime(lib));
    OrtSessionOptions* so = nullptr;
    TRY(api->CreateSessionOptions(&so));
    TRY(api->SetIntraOpNumThreads(so, 2));
    std::string err = check(api->CreateSession(env, enc.c_str(), so, &encoder));
    api->ReleaseSessionOptions(so);
    if (!err.empty()) return "encoder: " + err;
    return loadDecoder(dec, joi, tok);
}

std::string loadAll(const std::string& lib, const std::string& ctx, const std::string& model, const std::string& dec,
                    const std::string& joi, const std::string& tok) {
    TRY(loadRuntime(lib));

    // Encoder on the NPU. A compiled context (made on this phone, ~3 min the first time) loads in about a second.
    OrtSessionOptions* so = nullptr;
    TRY(api->CreateSessionOptions(&so));
    TRY(api->SetIntraOpNumThreads(so, 2));
    TRY(api->AddSessionConfigEntry(so, "session.disable_cpu_ep_fallback", "0"));
    std::string backend = lib + "/libQnnHtp.so";
    const char* keys[] = {"backend_path", "htp_performance_mode", "htp_graph_finalization_optimization_mode"};
    // Finalisation level 1: compiles in about a minute (level 3: ~3 min for ~20% faster runs). The compile needs about
    // 3 GB of memory, so a shorter one is less likely to be stopped by Android.
    const char* vals[] = {backend.c_str(), "power_saver", "1"};
    TRY(api->SessionOptionsAppendExecutionProvider(so, "QNN", keys, vals, 3));
    const std::string* path = &ctx;
    if (!exists(ctx)) {
        if (!exists(model)) { api->ReleaseSessionOptions(so); return "no NPU model"; }
        TRY(api->AddSessionConfigEntry(so, "ep.context_enable", "1"));
        TRY(api->AddSessionConfigEntry(so, "ep.context_file_path", ctx.c_str()));
        path = &model;
    }
    std::string err = check(api->CreateSession(env, path->c_str(), so, &encoder));
    api->ReleaseSessionOptions(so);
    if (!err.empty()) return "encoder: " + err;
    return loadDecoder(dec, joi, tok);
}

std::string loadDictionary(const std::string& path) {
    std::ifstream in(path);
    if (!in) return "no dictionary";
    std::string w;
    dictionary.words.clear();
    while (std::getline(in, w)) {
        while (!w.empty() && (w.back() == '\r' || w.back() == ' ')) w.pop_back();
        if (!w.empty()) dictionary.words.insert(tw::lower(w));
    }
    return dictionary.loaded() ? "" : "dictionary too small";
}

// [T][128] log-mel features, normalised per feature over the whole clip.
std::vector<std::vector<float>> features(const float* audio, int n) {
    knf::FbankOptions o;
    o.frame_opts.dither = 0;
    o.frame_opts.remove_dc_offset = false;
    o.frame_opts.window_type = "hann";
    o.frame_opts.snip_edges = false;
    o.mel_opts.num_bins = 128;
    o.mel_opts.low_freq = 0;
    o.mel_opts.high_freq = 8000;
    o.mel_opts.is_librosa = true;
    knf::OnlineFbank fb(o);
    fb.AcceptWaveform(16000, audio, n);
    fb.InputFinished();
    int t = fb.NumFramesReady();
    std::vector<std::vector<float>> x(t, std::vector<float>(128));
    for (int i = 0; i < t; i++) std::copy_n(fb.GetFrame(i), 128, x[i].begin());
    for (int k = 0; k < 128; k++) {
        double mean = 0, var = 0;
        for (int i = 0; i < t; i++) mean += x[i][k];
        mean /= std::max(t, 1);
        for (int i = 0; i < t; i++) var += (x[i][k] - mean) * (x[i][k] - mean);
        float sd = float(std::sqrt(var / std::max(t, 1))) + 1e-5f;
        for (int i = 0; i < t; i++) x[i][k] = float((x[i][k] - mean) / sd);
    }
    return x;
}

struct Decoder {
    std::vector<float> h = std::vector<float>(2 * 640, 0.f), c = std::vector<float>(2 * 640, 0.f);
    std::vector<float> out = std::vector<float>(640, 0.f);
    std::string step(int32_t token) {
        int32_t len = 1;
        int64_t ts[] = {1, 1}, ls[] = {1}, ss[] = {2, 1, 640};
        OrtValue* in[4] = {};
        TRY(api->CreateTensorWithDataAsOrtValue(cpu, &token, 4, ts, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &in[0]));
        TRY(api->CreateTensorWithDataAsOrtValue(cpu, &len, 4, ls, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &in[1]));
        TRY(api->CreateTensorWithDataAsOrtValue(cpu, h.data(), h.size() * 4, ss, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in[2]));
        TRY(api->CreateTensorWithDataAsOrtValue(cpu, c.data(), c.size() * 4, ss, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in[3]));
        const char* inames[] = {"targets", "target_length", "states.1", "onnx::Slice_3"};
        const char* onames[] = {"outputs", "states", "162"};
        OrtValue* o[3] = {};
        std::string err = check(api->Run(decoder, nullptr, inames, in, 4, onames, 3, o));
        for (auto* v : in) api->ReleaseValue(v);
        if (!err.empty()) return err;
        float* p;
        api->GetTensorMutableData(o[0], (void**)&p); std::copy_n(p, 640, out.begin());
        api->GetTensorMutableData(o[1], (void**)&p); std::copy_n(p, h.size(), h.begin());
        api->GetTensorMutableData(o[2], (void**)&p); std::copy_n(p, c.size(), c.begin());
        for (auto* v : o) api->ReleaseValue(v);
        return "";
    }
};

using Token = tw::Token;
constexpr int kOutputs = kBlank + 6;       // 1,025 token scores, then 5 duration scores
constexpr int kMaxSymbols = 10;

// The joiner's 1,030 scores for encoder frame t and a decoder output.
std::string joint(const float* enc, int encFrames, int t, const std::vector<float>& dec, std::vector<float>& scores) {
    std::vector<float> e(1024);
    for (int k = 0; k < 1024; k++) e[k] = enc[k * encFrames + t];
    int64_t es[] = {1, 1024, 1}, ds[] = {1, 640, 1};
    OrtValue* in[2] = {};
    TRY(api->CreateTensorWithDataAsOrtValue(cpu, e.data(), e.size() * 4, es, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in[0]));
    TRY(api->CreateTensorWithDataAsOrtValue(cpu, const_cast<float*>(dec.data()), dec.size() * 4, ds, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in[1]));
    const char* inames[] = {"encoder_outputs", "decoder_outputs"};
    const char* onames[] = {"outputs"};
    OrtValue* o = nullptr;
    std::string err = check(api->Run(joiner, nullptr, inames, in, 2, onames, 1, &o));
    for (auto* v : in) api->ReleaseValue(v);
    if (!err.empty()) return err;
    float* logits;
    api->GetTensorMutableData(o, (void**)&logits);
    scores.assign(logits, logits + kOutputs);
    api->ReleaseValue(o);
    return "";
}

std::vector<float> logSoftmax(const float* a, int n) {
    float top = *std::max_element(a, a + n);
    double z = 0;
    for (int i = 0; i < n; i++) z += std::exp(double(a[i] - top));
    float lz = top + float(std::log(z));
    std::vector<float> out(a, a + n);
    for (float& x : out) x -= lz;
    return out;
}

// The decoder state just before the greedy pass emitted a word, so the boosted pass can start there.
struct Checkpoint { size_t token; int t; Decoder dec; };

// Greedy TDT decoding of one encoder window. Each token keeps how sure the recogniser was of it. `checkpoints` (if
// given) gets the decoder state at the start of every word.
std::string decodeWindow(const float* enc, int encFrames, int validFrames, int start, std::vector<Token>& out,
                         std::vector<Checkpoint>* checkpoints = nullptr) {
    Decoder d;
    TRY(d.step(kBlank));
    std::vector<float> logits;
    int t = 0, emitted = 0;
    while (t < validFrames) {
        TRY(joint(enc, encFrames, t, d.out, logits));
        int tok = int(std::max_element(logits.begin(), logits.begin() + kBlank + 1) - logits.begin());
        int skip = kDurations[std::max_element(logits.begin() + kBlank + 1, logits.end()) - (logits.begin() + kBlank + 1)];
        if (tok != kBlank) {
            double z = 0;
            for (int i = 0; i <= kBlank; i++) z += std::exp(double(logits[i] - logits[tok]));
            if (checkpoints && (out.empty() || tw::startsWith(pieces[tok], tw::kMark))) checkpoints->push_back({out.size(), t, d});
            out.push_back({tok, start + t * kSubsample, float(1.0 / z)});
            TRY(d.step(tok));
            emitted++;
        }
        if (skip == 0 && (tok == kBlank || emitted >= kMaxSymbols)) skip = 1;
        if (skip > 0) emitted = 0;
        t += skip;
    }
    return "";
}

// TDT beam search with boosting (as on the Mac): keeps boost.beam candidate transcripts and lets a listed name compete
// as a whole instead of losing it at its first piece. Candidates are compared only after consuming the same audio, and
// the bonus for a name started but not finished is taken back. Runs from the greedy pass's state at `from` up to
// encoder frame `to`, and returns the tokens it adds.
std::string decodeBeam(const float* enc, int encFrames, const Checkpoint& from, int to, int start, const tw::Boost& boost,
                       std::vector<Token>& best) {
    struct Hyp {
        std::vector<Token> tokens;
        float score = 0;
        Decoder dec;
        std::set<tw::Partial> active;
        float credit = 0;
        int symbols = 0;
    };
    const int width = boost.beam;
    auto prune = [&](std::vector<Hyp>& hyps) {
        std::map<std::vector<int>, size_t> bestOf;
        std::vector<Hyp> kept;
        for (auto& h : hyps) {
            std::vector<int> key;
            for (auto& t : h.tokens) key.push_back(t.id);
            auto f = bestOf.find(key);
            if (f == bestOf.end()) { bestOf[key] = kept.size(); kept.push_back(std::move(h)); }
            else if (kept[f->second].score < h.score) kept[f->second] = std::move(h);
        }
        std::sort(kept.begin(), kept.end(), [](const Hyp& a, const Hyp& b) { return a.score > b.score; });
        if (int(kept.size()) > width) kept.resize(size_t(width));
        hyps.swap(kept);
    };
    Hyp first;
    first.dec = from.dec;
    std::map<int, std::vector<Hyp>> buckets;
    buckets[from.t].push_back(std::move(first));
    std::vector<Hyp> finished;
    auto place = [&](Hyp h, int frame) {
        if (frame >= to) finished.push_back(std::move(h)); else buckets[frame].push_back(std::move(h));
    };
    std::vector<float> scores;
    while (!buckets.empty()) {
        auto it = buckets.begin();
        int t = it->first;
        std::vector<Hyp> current = std::move(it->second);
        buckets.erase(it);
        prune(current);
        while (!current.empty()) {
            struct Cand { size_t parent; int id, duration; float score; std::set<tw::Partial> active; float credit; };
            std::vector<Cand> cands;
            for (size_t hi = 0; hi < current.size(); hi++) {
                Hyp& hyp = current[hi];
                TRY(joint(enc, encFrames, t, hyp.dec.out, scores));
                auto lp = logSoftmax(scores.data(), kBlank + 1);
                auto dp = logSoftmax(scores.data() + kBlank + 1, 5);
                // Nothing said here: move on by 1-4 frames (the two likeliest).
                std::vector<int> blanks = {1, 2, 3, 4};
                std::sort(blanks.begin(), blanks.end(), [&](int a, int b) { return dp[a] > dp[b]; });
                for (int k = 0; k < 2; k++) {
                    Hyp next = hyp;
                    next.score += lp[kBlank] + dp[blanks[k]];
                    next.symbols = 0;
                    place(std::move(next), t + kDurations[blanks[k]]);
                }
                // Word-pieces: the likeliest ones, plus any plausible piece that continues a listed name.
                std::vector<int> order(kBlank);
                for (int i = 0; i < kBlank; i++) order[i] = i;
                std::partial_sort(order.begin(), order.begin() + width, order.end(), [&](int a, int b) { return lp[a] > lp[b]; });
                std::vector<int> ids(order.begin(), order.begin() + width);
                for (int id = 0; id < kBlank; id++)
                    if (lp[id] >= boost.floor && std::find(ids.begin(), ids.end(), id) == ids.end() &&
                        !boost.advance(hyp.active, pieces[id]).empty()) ids.push_back(id);
                std::vector<int> durations;
                for (int d = 0; d < 5; d++) if (d > 0 || hyp.symbols + 1 < kMaxSymbols) durations.push_back(d);
                std::sort(durations.begin(), durations.end(), [&](int a, int b) { return dp[a] > dp[b]; });
                if (durations.size() > 2) durations.resize(2);
                for (int id : ids) {
                    if (lp[id] < boost.beamFloor) continue;
                    float bonus, credit;
                    std::set<tw::Partial> active = boost.advance(hyp.active, pieces[id]);
                    if (!active.empty()) {
                        // Continuing a name keeps its credit; starting a new one takes back an abandoned one's.
                        bool continuing = false;
                        for (auto& p : active) continuing |= p.pos > int(pieces[id].size());
                        bonus = boost.beta - (continuing ? 0 : hyp.credit);
                        credit = boost.complete(active) ? 0 : (continuing ? hyp.credit : 0) + boost.beta;
                    } else {
                        bonus = -hyp.credit; credit = 0;
                    }
                    for (int d : durations) cands.push_back({hi, id, d, hyp.score + lp[id] + dp[d] + bonus, active, credit});
                }
            }
            // Only the best few word-piece expansions are kept, so the decoder runs a few times per step.
            std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.score > b.score; });
            if (int(cands.size()) > width) cands.resize(size_t(width));
            std::vector<Hyp> same;
            for (auto& c : cands) {
                Hyp next = current[c.parent];
                TRY(next.dec.step(c.id));
                next.tokens.push_back({c.id, start + t * kSubsample, 1.f});
                next.score = c.score;
                next.active = c.active;
                next.credit = c.credit;
                if (kDurations[c.duration] == 0) { next.symbols++; same.push_back(std::move(next)); }
                else { next.symbols = 0; place(std::move(next), t + kDurations[c.duration]); }
            }
            prune(same);
            current.swap(same);
        }
    }
    const Hyp* top = nullptr;
    for (auto& h : finished) if (!top || h.score - h.credit > top->score - top->credit) top = &h;
    best = top ? top->tokens : std::vector<Token>{};
    return "";
}

bool startsWord(const Token& t) { return tokens[t.id].rfind("\xE2\x96\x81", 0) == 0; }

// The word starting at token i, lowercased letters and digits only, for matching across windows.
std::string wordAt(const std::vector<Token>& v, size_t i) {
    std::string w;
    for (size_t j = i; j < v.size() && (j == i || !startsWord(v[j])); j++) w += tokens[v[j].id];
    std::string r;
    for (unsigned char c : w) if (std::isalnum(c)) r += char(std::tolower(c));
    return r;
}

// Joins a window's tokens onto the transcript so far, as the Mac app does: both are cut at the first word starting at
// or after the middle of their overlap, so each word comes from exactly one window.
void stitch(std::vector<Token>& all, std::vector<Token> next, int start) {
    int cutFrame = start + (kFrames - kStride) / 2;
    std::string cutWord;
    bool cut = false;
    for (size_t i = 0; i < all.size(); i++) {
        if (startsWord(all[i]) && all[i].frame >= cutFrame) {
            cutFrame = all[i].frame;
            cutWord = wordAt(all, i);
            all.resize(i);
            cut = true;
            break;
        }
    }
    if (!cut && !all.empty()) {
        // Nothing of the earlier window was after the midpoint (speech ended or paused there): keep all of it, and
        // take only words that start clearly after its last one, or the same word would be typed twice.
        size_t first = next.size();
        for (size_t i = 0; i < next.size(); i++)
            if (startsWord(next[i]) && next[i].frame > all.back().frame + kTolerance) { first = i; break; }
        all.insert(all.end(), next.begin() + long(first), next.end());
        return;
    }
    // Resume at the same word near the cut; otherwise the nearest word start; otherwise the first word after it.
    size_t from = next.size();
    int best = kTolerance + 1;
    bool sameWord = false;
    for (size_t i = 0; i < next.size(); i++) {
        if (!startsWord(next[i])) continue;
        int d = std::abs(next[i].frame - cutFrame);
        if (d > kTolerance) continue;
        bool same = !cutWord.empty() && wordAt(next, i) == cutWord;
        if ((same && !sameWord) || (same == sameWord && d < best)) { from = i; best = d; sameWord = same; }
    }
    if (from == next.size())
        for (size_t i = 0; i < next.size(); i++)
            if (startsWord(next[i]) && next[i].frame >= cutFrame) { from = i; break; }
    all.insert(all.end(), next.begin() + long(from), next.end());
}

// Text of tokens, word-start markers as spaces.
std::string textOf(const std::vector<Token>& v) {
    std::string s;
    for (const Token& t : v) s += tokens[t.id];
    return tw::stripMarks(s);
}

// `context`: text in the field being typed into (its names are favoured). Contact names come from `contacts`.
// `vocabulary`: names to favour as given (none yet in the app; the benchmark passes its own).
std::string transcribe(const float* audio, int n, const std::string& context, std::string& text,
                       const std::vector<std::string>& vocabulary = {}) {
    const std::vector<std::string> fieldNames = tw::nameWords(context, dictionary);
    std::vector<std::string> listed;   // names for the final correction of unsure words
    std::vector<std::string> spelled;  // every name used, for its listed spelling
    auto x = features(audio, n);
    int total = int(x.size());
    std::vector<Token> all;
    std::vector<float> window(128 * kFrames);
    OrtRunOptions* ro = nullptr;
    TRY(api->CreateRunOptions(&ro));
    // Full speed while transcribing, then back to low power.
    api->AddRunConfigEntry(ro, "qnn.htp_perf_mode", "burst");
    api->AddRunConfigEntry(ro, "qnn.htp_perf_mode_post_run", "power_saver");
    std::string err;
    for (int start = 0; err.empty(); start += kStride) {
        int valid = std::min(kFrames, total - start);
        if (valid <= 0) break;
        std::fill(window.begin(), window.end(), 0.f);
        for (int i = 0; i < valid; i++)
            for (int k = 0; k < 128; k++) window[k * kFrames + i] = x[start + i][k];
        int64_t len = valid;
        int64_t as[] = {1, 128, kFrames}, ls[] = {1};
        OrtValue* in[2] = {};
        err = check(api->CreateTensorWithDataAsOrtValue(cpu, window.data(), window.size() * 4, as, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in[0]));
        if (err.empty()) err = check(api->CreateTensorWithDataAsOrtValue(cpu, &len, 8, ls, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &in[1]));
        const char* inames[] = {"audio_signal", "length"};
        const char* onames[] = {"outputs", "encoded_lengths"};
        OrtValue* o[2] = {};
        if (err.empty()) err = check(api->Run(encoder, ro, inames, in, 2, onames, 2, o));
        for (auto* v : in) if (v) api->ReleaseValue(v);
        if (!err.empty()) break;
        float* enc;
        int64_t* encLen;
        api->GetTensorMutableData(o[0], (void**)&enc);
        api->GetTensorMutableData(o[1], (void**)&encLen);
        bool last = start + kFrames >= total;
        std::vector<Token> next;
        std::vector<Checkpoint> checkpoints;
        int validFrames = int(std::min<int64_t>(*encLen, kFrames / kSubsample));
        err = decodeWindow(enc, kFrames / kSubsample, validFrames, start, next, &checkpoints);
        // Names for this window: the field's, then contacts that sound like something heard here.
        std::vector<std::string> terms, contactNames;
        if (err.empty() && !next.empty()) {
            contactNames = tw::soundingLike(contacts, textOf(next), contactSimilarity, contactLimit);
            terms = tw::termList({vocabulary, fieldNames, contactNames});
        }
        if (err.empty() && !terms.empty()) {
            tw::Boost boost(terms);
            // The boosted pass runs only around heard words a listed name could replace: from a word before to two
            // after, starting from the greedy pass's decoder state there. Elsewhere its result would be discarded.
            std::vector<std::pair<int, int>> spans;   // word ranges [from, to), merged
            for (auto [a, b] : tw::candidateRuns(boost, next, pieces)) {
                int from = std::max(0, a - 1), to = std::min(int(checkpoints.size()), b + 3);
                if (!spans.empty() && from <= spans.back().second) spans.back().second = std::max(spans.back().second, to);
                else spans.push_back({from, to});
            }
            if (!spans.empty()) {
                std::vector<Token> boosted;
                size_t copied = 0;
                for (auto [from, to] : spans) {
                    const Checkpoint& begin = checkpoints[size_t(from)];
                    size_t endToken = to < int(checkpoints.size()) ? checkpoints[size_t(to)].token : next.size();
                    int endFrame = to < int(checkpoints.size()) ? checkpoints[size_t(to)].t : validFrames;
                    boosted.insert(boosted.end(), next.begin() + long(copied), next.begin() + long(begin.token));
                    std::vector<Token> part;
                    err = decodeBeam(enc, kFrames / kSubsample, begin, endFrame, start, boost, part);
                    if (!err.empty()) break;
                    boosted.insert(boosted.end(), part.begin(), part.end());
                    copied = endToken;
                }
                boosted.insert(boosted.end(), next.begin() + long(copied), next.end());
                if (err.empty()) next = tw::acceptTerms(boost, next, boosted, pieces);
            }
            for (auto& t : terms) {
                if (std::find(spelled.begin(), spelled.end(), t) == spelled.end()) spelled.push_back(t);
                bool contact = std::find(contactNames.begin(), contactNames.end(), t) != contactNames.end();
                if (contact && !correctWithContacts) continue;
                if (std::find(listed.begin(), listed.end(), t) == listed.end()) listed.push_back(t);
            }
        }
        if (start == 0) all = std::move(next); else stitch(all, std::move(next), start);
        for (auto* v : o) api->ReleaseValue(v);
        if (last) break;
    }
    api->ReleaseRunOptions(ro);
    if (!err.empty()) return err;
    text = textOf(all);
    if (spelled.empty()) return "";
    // Word-pieces match names ignoring case, so a swapped-in name can carry odd capitals ("BJORN"): use the listed one.
    if (listed.empty()) { text = tw::spellTerms(text, spelled); return ""; }
    // Words the recogniser was unsure of that sound like a listed name ("Putnamic" -> Patnaik), then listed spellings.
    std::vector<std::pair<std::string, float>> words;
    for (const Token& t : all) {
        std::string piece = tokens[t.id];
        bool begins = tw::startsWith(piece, tw::kMark);
        if (begins) piece = piece.substr(tw::kMark.size());
        if (begins || words.empty()) words.push_back({piece, t.prob});
        else { words.back().first += piece; words.back().second = std::min(words.back().second, t.prob); }
    }
    words.erase(std::remove_if(words.begin(), words.end(), [](auto& w) { return w.first.empty(); }), words.end());
    text = tw::spellTerms(tw::correctNames(words, listed, dictionary), spelled);
    return "";
}

}  // namespace

#ifdef __ANDROID__
extern "C" JNIEXPORT jstring JNICALL
Java_io_github_ayushpatnaikgit_tellwell_AsrNpu_load(JNIEnv* e, jobject, jstring lib, jstring ctx, jstring model,
                                                    jstring dec, jstring joi, jstring tok) {
    std::lock_guard<std::mutex> g(mu);
    if (encoder && encoderOnNpu) return e->NewStringUTF("");
    // A phone that started on the CPU encoder and now has the NPU one: keep the CPU encoder until the NPU one works.
    OrtSession* previous = encoder;
    encoder = nullptr;
    std::string err = loadAll(str(e, lib), str(e, ctx), str(e, model), str(e, dec), str(e, joi), str(e, tok));
    if (!err.empty()) {
        // Error text only, never user content.
        TW_WARN("NPU speech unavailable: %s", err.c_str());
        if (encoder) api->ReleaseSession(encoder);
        encoder = previous;
    } else {
        encoderOnNpu = true;
        if (previous) api->ReleaseSession(previous);
    }
    return e->NewStringUTF(err.c_str());
}

// The speech encoder on the CPU, decoded the same way. "" when ready, else why not (no user content).
extern "C" JNIEXPORT jstring JNICALL
Java_io_github_ayushpatnaikgit_tellwell_AsrNpu_loadCpu(JNIEnv* e, jobject, jstring lib, jstring enc, jstring dec,
                                                       jstring joi, jstring tok) {
    std::lock_guard<std::mutex> g(mu);
    if (encoder) return e->NewStringUTF("");
    std::string err = loadCpu(str(e, lib), str(e, enc), str(e, dec), str(e, joi), str(e, tok));
    if (!err.empty()) {
        TW_WARN("CPU speech unavailable: %s", err.c_str());
        if (encoder) { api->ReleaseSession(encoder); encoder = nullptr; }
    }
    return e->NewStringUTF(err.c_str());
}

// The English word list for name correction. "" when loaded.
extern "C" JNIEXPORT jstring JNICALL
Java_io_github_ayushpatnaikgit_tellwell_AsrNpu_loadDictionary(JNIEnv* e, jobject, jstring path) {
    std::lock_guard<std::mutex> g(mu);
    return e->NewStringUTF(loadDictionary(str(e, path)).c_str());
}

// Contact names to consider (empty to turn contacts off). Kept in memory only.
extern "C" JNIEXPORT void JNICALL
Java_io_github_ayushpatnaikgit_tellwell_AsrNpu_setContacts(JNIEnv* e, jobject, jobjectArray names) {
    std::lock_guard<std::mutex> g(mu);
    contacts.clear();
    jsize n = names ? e->GetArrayLength(names) : 0;
    for (jsize i = 0; i < n; i++) {
        auto s = static_cast<jstring>(e->GetObjectArrayElement(names, i));
        contacts.push_back(str(e, s));
        e->DeleteLocalRef(s);
    }
}

// The transcript, or null if recognition failed. `context` is the text around the cursor (its names are favoured).
extern "C" JNIEXPORT jstring JNICALL
Java_io_github_ayushpatnaikgit_tellwell_AsrNpu_transcribe(JNIEnv* e, jobject, jfloatArray samples, jstring context) {
    std::lock_guard<std::mutex> g(mu);
    if (!encoder) return nullptr;
    jsize n = e->GetArrayLength(samples);
    std::vector<float> audio(n);
    e->GetFloatArrayRegion(samples, 0, n, audio.data());
    std::string text;
    std::string err = transcribe(audio.data(), n, str(e, context), text);
    if (!err.empty()) {
        TW_WARN("transcription failed: %s", err.c_str());
        return nullptr;
    }
    return e->NewStringUTF(text.c_str());
}
#endif
