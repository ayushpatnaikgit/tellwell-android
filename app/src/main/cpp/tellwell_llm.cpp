// TellWell's bridge to llama.cpp: load S1-mini once, keep the fixed prompt prefix in the KV cache,
// and clean up one transcript per call. Nothing here logs or stores the text it processes.
#include <jni.h>

#include <android/log.h>

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
    g.device = device;
    LOGI("loaded on %s, cached prefix of %d tokens", device.c_str(), g.n_prefix);
    return true;
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

extern "C" JNIEXPORT jstring JNICALL
Java_io_github_ayushpatnaikgit_tellwell_LlmNative_complete(JNIEnv *env, jobject, jstring jsuffix, jint max_tokens) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!g.ctx) return env->NewStringUTF("");
    // Forget the previous dictation; keep the cached instructions.
    llama_memory_seq_rm(llama_get_memory(g.ctx), 0, g.n_prefix, -1);
    std::vector<llama_token> s = tokenize(to_string(env, jsuffix));
    if (s.empty() || !decode_all(s, 512)) return env->NewStringUTF("");

    llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    std::string out;
    char buf[256];
    for (int i = 0; i < max_tokens; i++) {
        llama_token t = llama_sampler_sample(smpl, g.ctx, -1);
        if (llama_vocab_is_eog(g.vocab, t)) break;
        int n = llama_token_to_piece(g.vocab, t, buf, sizeof(buf), 0, false);
        if (n > 0) out.append(buf, n);
        if (llama_decode(g.ctx, llama_batch_get_one(&t, 1)) != 0) break;
    }
    llama_sampler_free(smpl);
    return env->NewStringUTF(out.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_io_github_ayushpatnaikgit_tellwell_LlmNative_device(JNIEnv *env, jobject) {
    std::lock_guard<std::mutex> lock(g_mu);
    return env->NewStringUTF(g.device.c_str());
}
