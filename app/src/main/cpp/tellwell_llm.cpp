// TellWell's bridge to llama.cpp: one cleanup model per slot (0: English, S1-mini; 1: Hindi, Gemma 4 E2B), each loaded
// once with its fixed prompt prefix kept in the KV cache, cleaning up one transcript per call. Nothing here logs or
// stores the text it processes.
#include <jni.h>

#include <android/log.h>

#include <algorithm>
#include <cmath>
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

constexpr int kSlots = 2;
State slots[kSlots];
// The slot being worked on; every entry point selects it under the lock first.
State *cur = &slots[0];
std::mutex g_mu;

bool select_slot(int slot) {
    if (slot < 0 || slot >= kSlots) return false;
    cur = &slots[slot];
    return true;
}

std::string to_string(JNIEnv *env, jstring s) {
    const char *c = env->GetStringUTFChars(s, nullptr);
    std::string out(c);
    env->ReleaseStringUTFChars(s, c);
    return out;
}

std::vector<llama_token> tokenize(const std::string &text) {
    int n = -llama_tokenize(cur->vocab, text.c_str(), (int32_t) text.size(), nullptr, 0, false, true);
    std::vector<llama_token> tokens(n);
    if (llama_tokenize(cur->vocab, text.c_str(), (int32_t) text.size(), tokens.data(), n, false, true) < 0) {
        tokens.clear();
    }
    return tokens;
}

// Decode tokens in batch-sized chunks. Returns false on failure.
bool decode_all(std::vector<llama_token> &tokens, int n_batch) {
    for (size_t i = 0; i < tokens.size(); i += n_batch) {
        int n = (int) std::min<size_t>(n_batch, tokens.size() - i);
        if (llama_decode(cur->ctx, llama_batch_get_one(tokens.data() + i, n)) != 0) {
            return false;
        }
    }
    return true;
}

void release() {
    if (cur->ctx) llama_free(cur->ctx);
    if (cur->model) llama_model_free(cur->model);
    *cur = State();
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
    cur->model = llama_model_load_from_file(path.c_str(), mp);
    if (!cur->model) {
        LOGE("model load failed on %s", device.c_str());
        return false;
    }
    llama_context_params cp = llama_context_default_params();
    // Room for the prompt prefix (Gemma's is ~320 tokens), a long dictation and its cleanup.
    cp.n_ctx = 2048;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    // Sequence 1 scores short texts (Hindi multi-pass choices) next to the cached prompt in sequence 0, sharing the
    // same cache instead of splitting it.
    cp.n_seq_max = 2;
    cp.kv_unified = true;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cur->ctx = llama_init_from_model(cur->model, cp);
    if (!cur->ctx) {
        LOGE("context init failed on %s", device.c_str());
        release();
        return false;
    }
    cur->vocab = llama_model_get_vocab(cur->model);
    std::vector<llama_token> p = tokenize(prefix);
    if (p.empty() || !decode_all(p, (int) cp.n_batch)) {
        LOGE("prefix decode failed on %s", device.c_str());
        release();
        return false;
    }
    cur->n_prefix = (int) p.size();
    cur->prefix_tokens = p;
    cur->device = device;
    LOGI("loaded on %s, cached prefix of %d tokens", device.c_str(), cur->n_prefix);
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
// When the copy reaches the end of the transcript (token t_end), guess the sentence-end mark (end_tok) instead of the
// prompt text that follows: a cleaned sentence usually ends there. Only the speed changes; the text is the same.
std::vector<llama_token> lookup(const std::vector<llama_token> &ctx, int k, int t_end, llama_token end_tok) {
    for (int n = 4; n >= 1; n--) {
        if ((int) ctx.size() <= n) continue;
        const size_t tail = ctx.size() - n;
        for (ptrdiff_t i = (ptrdiff_t) tail - 1; i >= 0; i--) {
            if (std::equal(ctx.begin() + i, ctx.begin() + i + n, ctx.begin() + tail)) {
                const size_t from = i + n, to = std::min(ctx.size(), from + (size_t) k);
                if (t_end >= 0 && end_tok >= 0 && from <= (size_t) t_end && to > (size_t) t_end) {
                    std::vector<llama_token> d(ctx.begin() + from, ctx.begin() + t_end);
                    d.push_back(end_tok);
                    return d;
                }
                return std::vector<llama_token>(ctx.begin() + from, ctx.begin() + to);
            }
        }
    }
    return {};
}

// Index in the suffix tokens where the prompt text after the transcript starts, or -1.
int transcript_end(const std::string &suffix, const std::vector<llama_token> &s) {
    const size_t close = suffix.rfind("\nRAW_TRANSCRIPTION");
    if (close == std::string::npos) return -1;
    size_t off = 0;
    char buf[256];
    for (size_t i = 0; i < s.size(); i++) {
        if (off >= close) return (int) i;
        const int n = llama_token_to_piece(cur->vocab, s[i], buf, sizeof(buf), 0, true);
        off += n > 0 ? n : 0;
    }
    return -1;
}

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_io_github_ayushpatnaikgit_tellwell_LlmNative_load(JNIEnv *env, jobject, jint slot, jstring jlibdir, jstring jpath,
                                                      jstring jprefix, jstring jdevices, jint threads) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!select_slot(slot)) return env->NewStringUTF("");
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
    // devices: comma-separated preference list, e.cur-> "HTP0,CPU"
    size_t start = 0;
    while (start <= devices.size()) {
        size_t comma = devices.find(',', start);
        std::string d = devices.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!d.empty() && try_load(path, prefix, d, threads)) {
            return env->NewStringUTF(cur->device.c_str());
        }
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return env->NewStringUTF("");
}

// Greedy decoding. With draft_k > 0, each step also checks up to draft_k copied-ahead tokens in the same forward
// pass and keeps the ones the model agrees with, so the text is the same as plain greedy decoding, only faster.
extern "C" JNIEXPORT jstring JNICALL
Java_io_github_ayushpatnaikgit_tellwell_LlmNative_complete(JNIEnv *env, jobject, jint slot, jstring jsuffix,
                                                          jint max_tokens, jint draft_k) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!select_slot(slot) || !cur->ctx) return env->NewStringUTF("");
    llama_memory_t mem = llama_get_memory(cur->ctx);
    // Forget the previous dictation; keep the cached instructions.
    llama_memory_seq_rm(mem, 0, cur->n_prefix, -1);
    const std::string suffix = to_string(env, jsuffix);
    std::vector<llama_token> s = tokenize(suffix);
    if (s.empty() || cur->n_prefix + (int) s.size() + max_tokens + draft_k + 1 > 2048 || !decode_all(s, 512)) {
        return env->NewStringUTF("");
    }
    // Sentence-end guess for copy-ahead: "।" for Devanagari text (UTF-8 lead bytes E0 A4/A5), else ".".
    int t_end = -1;
    llama_token end_tok = -1;
    if (draft_k > 0) {
        const int e = transcript_end(suffix, s);
        if (e >= 0) {
            t_end = cur->n_prefix + e;
            const bool deva = suffix.find("\xe0\xa4") != std::string::npos || suffix.find("\xe0\xa5") != std::string::npos;
            std::vector<llama_token> m = tokenize(deva ? "\xe0\xa5\xa4" : ".");
            if (m.size() == 1) end_tok = m[0];
        }
    }

    const int n_vocab = llama_vocab_n_tokens(cur->vocab);
    std::vector<llama_token> ctx_toks = cur->prefix_tokens;
    ctx_toks.insert(ctx_toks.end(), s.begin(), s.end());
    int n_past = cur->n_prefix + (int) s.size();
    std::string out;
    char buf[256];
    int produced = 0;
    bool done = false;
    auto emit = [&](llama_token t) {
        if (llama_vocab_is_eog(cur->vocab, t) || produced >= max_tokens) {
            done = true;
            return;
        }
        int n = llama_token_to_piece(cur->vocab, t, buf, sizeof(buf), 0, false);
        if (n > 0) out.append(buf, n);
        produced++;
        ctx_toks.push_back(t);
    };

    llama_batch batch = llama_batch_init(1 + std::max(0, (int) draft_k), 0, 1);
    llama_token y = argmax(llama_get_logits_ith(cur->ctx, -1), n_vocab);
    emit(y);
    while (!done) {
        // y was just produced and is not in the KV cache yet: feed it with the draft.
        std::vector<llama_token> draft = draft_k > 0 ? lookup(ctx_toks, draft_k, t_end, end_tok) : std::vector<llama_token>();
        batch.n_tokens = 0;
        for (size_t i = 0; i <= draft.size(); i++) {
            const int j = batch.n_tokens++;
            batch.token[j] = i == 0 ? y : draft[i - 1];
            batch.pos[j] = n_past + (int) i;
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j] = 1;
        }
        if (llama_decode(cur->ctx, batch) != 0) break;
        size_t a = 0;
        llama_token next;
        while (true) {
            next = argmax(llama_get_logits_ith(cur->ctx, (int) a), n_vocab);
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

// Log-probability of a short text under the slot's model used as a plain language model (start token, then the
// text), in sequence 1 so the cached prompt in sequence 0 stays. For choices between a few candidate texts; nothing
// is kept afterwards. Returns a very low value on failure.
extern "C" JNIEXPORT jdouble JNICALL
Java_io_github_ayushpatnaikgit_tellwell_LlmNative_score(JNIEnv *env, jobject, jint slot, jstring jtext) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!select_slot(slot) || !cur->ctx) return -1e30;
    std::vector<llama_token> t = tokenize("<bos>" + to_string(env, jtext));
    const int n = (int) t.size();
    if (n < 2 || n > 512 || cur->n_prefix + 1100 + n > 2048) return -1e30;
    llama_memory_t mem = llama_get_memory(cur->ctx);
    llama_memory_seq_rm(mem, 1, -1, -1);
    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        batch.token[i] = t[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 1;
        batch.logits[i] = i < n - 1;
    }
    batch.n_tokens = n;
    double total = -1e30;
    if (llama_decode(cur->ctx, batch) == 0) {
        const int n_vocab = llama_vocab_n_tokens(cur->vocab);
        total = 0;
        for (int i = 0; i < n - 1; i++) {
            const float *row = llama_get_logits_ith(cur->ctx, i);
            float mx = row[0];
            for (int v = 1; v < n_vocab; v++) mx = std::max(mx, row[v]);
            double sum = 0;
            for (int v = 0; v < n_vocab; v++) sum += std::exp((double) row[v] - mx);
            total += (double) row[t[i + 1]] - mx - std::log(sum);
        }
    }
    llama_batch_free(batch);
    llama_memory_seq_rm(mem, 1, -1, -1);
    return total;
}

extern "C" JNIEXPORT jstring JNICALL
Java_io_github_ayushpatnaikgit_tellwell_LlmNative_device(JNIEnv *env, jobject, jint slot) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!select_slot(slot)) return env->NewStringUTF("");
    return env->NewStringUTF(cur->device.c_str());
}

// Frees a slot's model (a language the user removed).
extern "C" JNIEXPORT void JNICALL
Java_io_github_ayushpatnaikgit_tellwell_LlmNative_unload(JNIEnv *, jobject, jint slot) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (select_slot(slot)) release();
}
