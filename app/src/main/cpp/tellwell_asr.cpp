// Speech recognition on the Qualcomm NPU: the Parakeet TDT 0.6B v2 encoder, statically quantised (8-bit weights,
// 16-bit activations) for a fixed 10 s window, run by ONNX Runtime's QNN execution provider. The decoder and joiner
// run on the CPU with the same ONNX Runtime. Features match the model's front end (128 log-mel, per-feature
// normalised). Audio and text only pass through memory; nothing is logged or written except the compiled model.
//
// ONNX Runtime with QNN is a separate library (libonnxruntime_qnn.so) from the one sherpa-onnx uses, opened by path
// and used only through its C API table, so the two never clash.
#include <android/log.h>
#include <dlfcn.h>
#include <jni.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#include "kaldi-native-fbank/csrc/online-feature.h"
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

std::string check(OrtStatus* s) {
    if (!s) return "";
    std::string m = api->GetErrorMessage(s);
    api->ReleaseStatus(s);
    return m.empty() ? "error" : m;
}
std::string check(const std::string& s) { return s; }
#define TRY(x) do { std::string e_ = check(x); if (!e_.empty()) return e_; } while (0)

std::string str(JNIEnv* e, jstring s) {
    const char* c = e->GetStringUTFChars(s, nullptr);
    std::string r(c);
    e->ReleaseStringUTFChars(s, c);
    return r;
}

bool exists(const std::string& p) { return std::ifstream(p).good(); }

std::string loadAll(const std::string& lib, const std::string& ctx, const std::string& model, const std::string& dec,
                    const std::string& joi, const std::string& tok) {
    void* h = dlopen((lib + "/libonnxruntime_qnn.so").c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) return std::string("dlopen: ") + dlerror();
    auto base = reinterpret_cast<const OrtApiBase* (*)()>(dlsym(h, "OrtGetApiBase"));
    if (!base) return "no OrtGetApiBase";
    api = base()->GetApi(ORT_API_VERSION);
    if (!api) return "ORT API version mismatch";
    TRY(api->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "tellwell", &env));
    TRY(api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &cpu));

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

    // Decoder and joiner: small, run once per token, on one CPU thread.
    TRY(api->CreateSessionOptions(&so));
    TRY(api->SetIntraOpNumThreads(so, 1));
    err = check(api->CreateSession(env, dec.c_str(), so, &decoder));
    if (err.empty()) err = check(api->CreateSession(env, joi.c_str(), so, &joiner));
    api->ReleaseSessionOptions(so);
    if (!err.empty()) return "decoder: " + err;

    std::ifstream in(tok);
    std::string line;
    while (std::getline(in, line)) {
        auto sp = line.rfind(' ');
        if (sp == std::string::npos) continue;
        size_t id = std::stoul(line.substr(sp + 1));
        if (tokens.size() <= id) tokens.resize(id + 1);
        tokens[id] = line.substr(0, sp);
    }
    return tokens.size() > kBlank ? "" : "bad tokens file";
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

struct Token { int id; int frame; };  // frame: absolute, in 10 ms feature frames

// Greedy TDT decoding of one encoder window.
std::string decodeWindow(const float* enc, int encFrames, int validFrames, int start, std::vector<Token>& out) {
    Decoder d;
    TRY(d.step(kBlank));
    std::vector<float> e(1024);
    int64_t es[] = {1, 1024, 1}, ds[] = {1, 640, 1};
    int t = 0, emitted = 0;
    while (t < validFrames) {
        for (int k = 0; k < 1024; k++) e[k] = enc[k * encFrames + t];
        OrtValue* in[2] = {};
        TRY(api->CreateTensorWithDataAsOrtValue(cpu, e.data(), e.size() * 4, es, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in[0]));
        TRY(api->CreateTensorWithDataAsOrtValue(cpu, d.out.data(), d.out.size() * 4, ds, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in[1]));
        const char* inames[] = {"encoder_outputs", "decoder_outputs"};
        const char* onames[] = {"outputs"};
        OrtValue* o = nullptr;
        std::string err = check(api->Run(joiner, nullptr, inames, in, 2, onames, 1, &o));
        for (auto* v : in) api->ReleaseValue(v);
        if (!err.empty()) return err;
        float* logits;
        api->GetTensorMutableData(o, (void**)&logits);
        int tok = int(std::max_element(logits, logits + kBlank + 1) - logits);
        int skip = kDurations[std::max_element(logits + kBlank + 1, logits + kBlank + 6) - (logits + kBlank + 1)];
        api->ReleaseValue(o);
        if (tok != kBlank) {
            out.push_back({tok, start + t * kSubsample});
            TRY(d.step(tok));
            emitted++;
        }
        if (skip == 0 && (tok == kBlank || emitted >= 10)) skip = 1;
        if (skip > 0) emitted = 0;
        t += skip;
    }
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

std::string transcribe(const float* audio, int n, std::string& text) {
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
        err = decodeWindow(enc, kFrames / kSubsample, int(std::min<int64_t>(*encLen, kFrames / kSubsample)), start, next);
        if (start == 0) all = std::move(next); else stitch(all, std::move(next), start);
        for (auto* v : o) api->ReleaseValue(v);
        if (last) break;
    }
    api->ReleaseRunOptions(ro);
    if (!err.empty()) return err;
    std::string s;
    for (const Token& t : all) s += tokens[t.id];
    // SentencePiece word marker (U+2581) to spaces.
    const std::string mark = "\xE2\x96\x81";
    for (size_t p; (p = s.find(mark)) != std::string::npos;) s.replace(p, mark.size(), " ");
    size_t a = s.find_first_not_of(' '), b = s.find_last_not_of(' ');
    text = a == std::string::npos ? "" : s.substr(a, b - a + 1);
    return "";
}

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_io_github_ayushpatnaikgit_tellwell_AsrNpu_load(JNIEnv* e, jobject, jstring lib, jstring ctx, jstring model,
                                                    jstring dec, jstring joi, jstring tok) {
    std::lock_guard<std::mutex> g(mu);
    if (encoder) return e->NewStringUTF("");
    std::string err = loadAll(str(e, lib), str(e, ctx), str(e, model), str(e, dec), str(e, joi), str(e, tok));
    if (!err.empty()) {
        // Error text only, never user content.
        __android_log_print(ANDROID_LOG_WARN, "TellWellASR", "NPU speech unavailable: %s", err.c_str());
        if (encoder) { api->ReleaseSession(encoder); encoder = nullptr; }
    }
    return e->NewStringUTF(err.c_str());
}

// The transcript, or null if the NPU failed (the caller then uses the CPU recogniser).
extern "C" JNIEXPORT jstring JNICALL
Java_io_github_ayushpatnaikgit_tellwell_AsrNpu_transcribe(JNIEnv* e, jobject, jfloatArray samples) {
    std::lock_guard<std::mutex> g(mu);
    if (!encoder) return nullptr;
    jsize n = e->GetArrayLength(samples);
    std::vector<float> audio(n);
    e->GetFloatArrayRegion(samples, 0, n, audio.data());
    std::string text;
    std::string err = transcribe(audio.data(), n, text);
    if (!err.empty()) {
        __android_log_print(ANDROID_LOG_WARN, "TellWellASR", "NPU transcription failed: %s", err.c_str());
        return nullptr;
    }
    return e->NewStringUTF(text.c_str());
}
