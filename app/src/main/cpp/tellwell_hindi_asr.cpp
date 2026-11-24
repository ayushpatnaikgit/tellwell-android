// Hindi speech recognition (an optional language): IndicConformer Hindi (AI4Bharat, MIT) on the CPU through ONNX Runtime.
// One ONNX model takes 16 kHz audio and returns Hindi CTC log-probabilities; its feature computation is NeMo's,
// built into the model (tellwell-lab/hindi/apk/export_ctc_audio.py). Greedy CTC decoding. Nothing here logs or
// stores audio or text; warnings carry error messages only.
#include <jni.h>

#include <android/log.h>
#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#include "ort/onnxruntime_c_api.h"

#define TW_WARN(...) __android_log_print(ANDROID_LOG_WARN, "TellWellHindiASR", __VA_ARGS__)

namespace {

constexpr int kRate = 16000;
// Long dictations are recognised in pieces: the model attends over the whole input, so time and memory grow
// quickly with length. Pieces are at most 20 s and end at the quietest 0.1 s of their last 4 s.
constexpr int kMaxPiece = 20 * kRate;
constexpr int kSearch = 4 * kRate;
constexpr int kHop = kRate / 10;

std::mutex mu;
const OrtApi* api = nullptr;
OrtEnv* env = nullptr;
OrtSession* session = nullptr;
OrtMemoryInfo* cpu = nullptr;
std::vector<std::string> pieces;  // SentencePiece pieces; the last one is the CTC blank

std::string check(OrtStatus* s) {
    if (!s) return "";
    std::string m = api->GetErrorMessage(s);
    api->ReleaseStatus(s);
    return m.empty() ? "error" : m;
}
std::string check(const std::string& s) { return s; }
#define TRY(x) do { std::string e_ = check(x); if (!e_.empty()) return e_; } while (0)

std::string str(JNIEnv* e, jstring s) {
    if (!s) return "";
    const char* c = e->GetStringUTFChars(s, nullptr);
    std::string r(c);
    e->ReleaseStringUTFChars(s, c);
    return r;
}

std::string load(const std::string& lib, const std::string& model, const std::string& tokens, int threads) {
    if (!api) {
        // The ONNX Runtime build that ships with the QNN libraries also has the CPU provider used here.
        void* h = dlopen((lib + "/libonnxruntime_qnn.so").c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!h) return std::string("dlopen: ") + dlerror();
        auto base = reinterpret_cast<const OrtApiBase* (*)()>(dlsym(h, "OrtGetApiBase"));
        if (!base) return "no OrtGetApiBase";
        api = base()->GetApi(ORT_API_VERSION);
        if (!api) return "ORT API version mismatch";
        TRY(api->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "tellwell-hindi", &env));
        TRY(api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &cpu));
    }
    if (session) return "";
    std::ifstream in(tokens);
    std::string line;
    pieces.clear();
    while (std::getline(in, line)) pieces.push_back(line);
    if (pieces.size() < 2) return "bad tokens file";
    OrtSessionOptions* so = nullptr;
    TRY(api->CreateSessionOptions(&so));
    TRY(api->SetIntraOpNumThreads(so, threads));
    // No memory pool: a long dictation's working memory is returned afterwards instead of staying reserved.
    TRY(api->DisableCpuMemArena(so));
    std::string err = check(api->CreateSession(env, model.c_str(), so, &session));
    api->ReleaseSessionOptions(so);
    return err.empty() ? "" : "model: " + err;
}

// Greedy CTC over one piece of audio; appends the text.
std::string recognise(const float* audio, int n, std::string& text) {
    if (n < kRate / 10) return "";
    OrtValue* in = nullptr;
    int64_t shape[2] = {1, n};
    TRY(api->CreateTensorWithDataAsOrtValue(cpu, const_cast<float*>(audio), (size_t) n * 4, shape, 2,
                                            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in));
    const char* in_names[] = {"audio"};
    const char* out_names[] = {"logprobs"};
    OrtValue* out = nullptr;
    std::string err = check(api->Run(session, nullptr, in_names, &in, 1, out_names, 1, &out));
    api->ReleaseValue(in);
    if (!err.empty()) return err;
    OrtTensorTypeAndShapeInfo* info = nullptr;
    err = check(api->GetTensorTypeAndShape(out, &info));
    size_t dims = 0;
    int64_t shp[3] = {0, 0, 0};
    if (err.empty()) err = check(api->GetDimensionsCount(info, &dims));
    if (err.empty() && dims == 3) err = check(api->GetDimensions(info, shp, 3));
    if (info) api->ReleaseTensorTypeAndShapeInfo(info);
    float* lp = nullptr;
    if (err.empty()) err = check(api->GetTensorMutableData(out, (void**) &lp));
    if (err.empty() && (dims != 3 || shp[2] != (int64_t) pieces.size())) err = "unexpected output shape";
    if (err.empty()) {
        const int frames = (int) shp[1], vocab = (int) shp[2], blank = vocab - 1;
        int prev = -1;
        for (int t = 0; t < frames; t++) {
            const float* row = lp + (size_t) t * vocab;
            const int best = (int) (std::max_element(row, row + vocab) - row);
            if (best != prev && best != blank && best != 0) text += pieces[best];  // 0 = <unk>
            prev = best;
        }
    }
    api->ReleaseValue(out);
    return err;
}

// Where to end a piece that starts at `from`: the quietest 0.1 s in the last 4 s before from + kMaxPiece.
int pieceEnd(const float* a, int n, int from) {
    if (n - from <= kMaxPiece) return n;
    const int hi = from + kMaxPiece, lo = hi - kSearch;
    int best = hi;
    double quietest = 1e30;
    for (int s = lo; s + kHop <= hi; s += kHop / 2) {
        double e = 0;
        for (int i = s; i < s + kHop; i++) e += (double) a[i] * a[i];
        if (e < quietest) { quietest = e; best = s + kHop / 2; }
    }
    return best;
}

std::string transcribe(const float* a, int n, std::string& out) {
    std::string text;
    for (int from = 0; from < n;) {
        const int to = pieceEnd(a, n, from);
        TRY(recognise(a + from, to - from, text));
        from = to;
    }
    // SentencePiece: "▁" (U+2581) marks a word start.
    const std::string mark = "\xe2\x96\x81";
    for (size_t p = text.find(mark); p != std::string::npos; p = text.find(mark, p + 1)) text.replace(p, mark.size(), " ");
    size_t b = text.find_first_not_of(' '), e = text.find_last_not_of(' ');
    out = b == std::string::npos ? "" : text.substr(b, e - b + 1);
    // Collapse the double spaces left where pieces meet.
    out.erase(std::unique(out.begin(), out.end(), [](char x, char y) { return x == ' ' && y == ' '; }), out.end());
    return "";
}

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_io_github_ayushpatnaikgit_tellwell_AsrHindi_load(JNIEnv* e, jobject, jstring lib, jstring model, jstring tokens,
                                                      jint threads) {
    std::lock_guard<std::mutex> g(mu);
    std::string err = load(str(e, lib), str(e, model), str(e, tokens), threads);
    if (!err.empty()) TW_WARN("Hindi speech unavailable: %s", err.c_str());
    return e->NewStringUTF(err.c_str());
}

// The transcript, or null if recognition failed.
extern "C" JNIEXPORT jstring JNICALL
Java_io_github_ayushpatnaikgit_tellwell_AsrHindi_transcribe(JNIEnv* e, jobject, jfloatArray samples) {
    std::lock_guard<std::mutex> g(mu);
    if (!session) return nullptr;
    jsize n = e->GetArrayLength(samples);
    std::vector<float> audio(n);
    e->GetFloatArrayRegion(samples, 0, n, audio.data());
    std::string text;
    std::string err = transcribe(audio.data(), n, text);
    if (!err.empty()) {
        TW_WARN("transcription failed: %s", err.c_str());
        return nullptr;
    }
    return e->NewStringUTF(text.c_str());
}

// Frees the model (the user removed Hindi).
extern "C" JNIEXPORT void JNICALL
Java_io_github_ayushpatnaikgit_tellwell_AsrHindi_unload(JNIEnv*, jobject) {
    std::lock_guard<std::mutex> g(mu);
    if (session) { api->ReleaseSession(session); session = nullptr; }
    pieces.clear();
}
