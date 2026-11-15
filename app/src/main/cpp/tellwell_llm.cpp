// TellWell's bridge to llama.cpp: load S1-mini once, keep the fixed prompt prefix in the KV cache,
// and clean up one transcript per call. Nothing here logs or stores the text it processes.
#include <jni.h>

#include <android/log.h>

#include <algorithm>
#include <mutex>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "llama.h"

#define TAG "TellWellLLM"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

namespace {

struct State {
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    const llama_vocab *vocab = nullptr;
    int n_prefix = 0;
    std::vector<llama_token> prefix_tokens;
    std::string device;
};

State g;
std::mutex g_mu;

std::string to_string(JNIEnv *env, jstring s) {
    const char *c = env->GetStringUTFChars(s, nullptr);
    std::string out(c);
    env->ReleaseStringUTFChars(s, c);
    return out;
}

std::vector<llama_token> tokenize(const std::string &text) {
    int n = -llama_tokenize(g.vocab, text.c_str(), (int32_t) text.size(), nullptr, 0, false, true);
    std::vector<llama_token> tokens(n);
    if (llama_tokenize(g.vocab, text.c_str(), (int32_t) text.size(), tokens.data(), n, false, true) < 0) {
        tokens.clear();
    }
    return tokens;
}

// Decode tokens in batch-sized chunks. Returns false on failure.
bool decode_all(std::vector<llama_token> &tokens, int n_batch) {
    for (size_t i = 0; i < tokens.size(); i += n_batch) {
        int n = (int) std::min<size_t>(n_batch, tokens.size() - i);
        if (llama_decode(g.ctx, llama_batch_get_one(tokens.data() + i, n)) != 0) {
            return false;
        }
    }
    return true;
}

void release() {
    if (g.ctx) llama_free(g.ctx);
    if (g.model) llama_model_free(g.model);
    g = State();
}

bool try_load(const std::string &path, const std::string &prefix, const std::string &device, int threads) {
    llama_model_params mp = llama_model_default_params();
    ggml_backend_dev_t devs[2] = {nullptr, nullptr};
    if (device != "CPU") {
        devs[0] = ggml_backend_dev_by_name(device.c_str());
        if (!devs[0]) {
            LOGI("device %s not available", device.c_str());
            return false;
        }
        // llama.cpp's OpenCL backend is written for Qualcomm Adreno GPUs; skip other GPUs.
        if (device == "GPUOpenCL" && std::string(ggml_backend_dev_description(devs[0])).find("Adreno") == std::string::npos) {
            LOGI("GPU is not Adreno, skipping");
            return false;
        }
        mp.devices = devs;
        mp.n_gpu_layers = 99;
    } else {
        mp.devices = devs;  // empty list: CPU only
        mp.n_gpu_layers = 0;
    }
    g.model = llama_model_load_from_file(path.c_str(), mp);
    if (!g.model) {
        LOGE("model load failed on %s", device.c_str());
        return false;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 1024;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    g.ctx = llama_init_from_model(g.model, cp);
    if (!g.ctx) {
        LOGE("context init failed on %s", device.c_str());
        release();
        return false;
    }
    g.vocab = llama_model_get_vocab(g.model);
    std::vector<llama_token> p = tokenize(prefix);
    if (p.empty() || !decode_all(p, (int) cp.n_batch)) {
        LOGE("prefix decode failed on %s", device.c_str());
        release();
        return false;
    }
    g.n_prefix = (int) p.size();
    g.prefix_tokens = p;
    g.device = device;
    LOGI("loaded on %s, cached prefix of %d tokens", device.c_str(), g.n_prefix);
    return true;
}

llama_token argmax(const float *logits, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) {
        if (logits[i] > logits[best]) best = i;
    }
    return best;
}

// Copy-ahead drafting: find the most recent earlier occurrence of the last n tokens (n = 4 down to 1) and propose
// up to k tokens that followed it. Cleanup output mostly repeats the transcript, so these guesses are often right.
std::vector<llama_token> lookup(const std::vector<llama_token> &ctx, int k) {
    for (int n = 4; n >= 1; n--) {
        if ((int) ctx.size() <= n) continue;
        const size_t tail = ctx.size() - n;
        for (ptrdiff_t i = (ptrdiff_t) tail - 1; i >= 0; i--) {
            if (std::equal(ctx.begin() + i, ctx.begin() + i + n, ctx.begin() + tail)) {
                const size_t from = i + n, to = std::min(ctx.size(), from + (size_t) k);
                return std::vector<llama_token>(ctx.begin() + from, ctx.begin() + to);
            }
        }
    }
    return {};
}

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_io_github_ayushpatnaikgit_tellwell_LlmNative_load(JNIEnv *env, jobject, jstring jlibdir, jstring jpath, jstring jprefix,
                                                      jstring jdevices, jint threads) {
    std::lock_guard<std::mutex> lock(g_mu);
    release();
    static bool backends_loaded = false;
    if (!backends_loaded) {
        // Backends (CPU variants, OpenCL, Hexagon) are separate libraries loaded at runtime, so a phone
        // without OpenCL or without the NPU simply doesn't get that backend instead of failing to start.
        ggml_backend_load_all_from_path(to_string(env, jlibdir).c_str());
        llama_backend_init();
        backends_loaded = true;
    }
    std::string path = to_string(env, jpath), prefix = to_string(env, jprefix), devices = to_string(env, jdevices);
    // devices: comma-separated preference list, e.g. "HTP0,CPU"
    size_t start = 0;
    while (start <= devices.size()) {
        size_t comma = devices.find(',', start);
        std::string d = devices.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!d.empty() && try_load(path, prefix, d, threads)) {
            return env->NewStringUTF(g.device.c_str());
        }
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return env->NewStringUTF("");
}

// Greedy decoding. With draft_k > 0, each step also checks up to draft_k copied-ahead tokens in the same forward
// pass and keeps the ones the model agrees with, so the text is the same as plain greedy decoding, only faster.
extern "C" JNIEXPORT jstring JNICALL
Java_io_github_ayushpatnaikgit_tellwell_LlmNative_complete(JNIEnv *env, jobject, jstring jsuffix, jint max_tokens,
                                                          jint draft_k) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!g.ctx) return env->NewStringUTF("");
    llama_memory_t mem = llama_get_memory(g.ctx);
    // Forget the previous dictation; keep the cached instructions.
    llama_memory_seq_rm(mem, 0, g.n_prefix, -1);
    std::vector<llama_token> s = tokenize(to_string(env, jsuffix));
    if (s.empty() || !decode_all(s, 512)) return env->NewStringUTF("");

    const int n_vocab = llama_vocab_n_tokens(g.vocab);
    std::vector<llama_token> ctx_toks = g.prefix_tokens;
    ctx_toks.insert(ctx_toks.end(), s.begin(), s.end());
    int n_past = g.n_prefix + (int) s.size();
    std::string out;
    char buf[256];
    int produced = 0;
    bool done = false;
    auto emit = [&](llama_token t) {
        if (llama_vocab_is_eog(g.vocab, t) || produced >= max_tokens) {
            done = true;
            return;
        }
        int n = llama_token_to_piece(g.vocab, t, buf, sizeof(buf), 0, false);
        if (n > 0) out.append(buf, n);
        produced++;
        ctx_toks.push_back(t);
    };

    llama_batch batch = llama_batch_init(1 + std::max(0, (int) draft_k), 0, 1);
    llama_token y = argmax(llama_get_logits_ith(g.ctx, -1), n_vocab);
    emit(y);
    while (!done) {
        // y was just produced and is not in the KV cache yet: feed it with the draft.
        std::vector<llama_token> draft = draft_k > 0 ? lookup(ctx_toks, draft_k) : std::vector<llama_token>();
        batch.n_tokens = 0;
        for (size_t i = 0; i <= draft.size(); i++) {
            const int j = batch.n_tokens++;
            batch.token[j] = i == 0 ? y : draft[i - 1];
            batch.pos[j] = n_past + (int) i;
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j] = 1;
        }
        if (llama_decode(g.ctx, batch) != 0) break;
        size_t a = 0;
        llama_token next;
        while (true) {
            next = argmax(llama_get_logits_ith(g.ctx, (int) a), n_vocab);
            if (a < draft.size() && next == draft[a]) {
                emit(next);
                a++;
                if (done) break;
            } else {
                break;
            }
        }
        // Keep y and the accepted draft tokens in the cache; drop the rejected ones.
        n_past += 1 + (int) a;
        llama_memory_seq_rm(mem, 0, n_past, -1);
        if (done) break;
        emit(next);
        y = next;
    }
    llama_batch_free(batch);
    return env->NewStringUTF(out.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_io_github_ayushpatnaikgit_tellwell_LlmNative_device(JNIEnv *env, jobject) {
    std::lock_guard<std::mutex> lock(g_mu);
    return env->NewStringUTF(g.device.c_str());
}
