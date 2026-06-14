#pragma once
#include <cstddef>
#include <string>
#include <memory>

class BlackwellEngine {
public:
    struct Impl;

    // KV-cache strategy selected at construction:
    //   Continuous - legacy FP32 contiguous cache + layer offloading (default).
    //   Paged      - bf16 paged cache with Copy-on-Write fork / rewind.
    enum class KVCacheMode { Continuous, Paged };

    // num_gpu_layers: how many leading transformer layers keep their weights /
    // KV cache resident in VRAM; the rest are offloaded to pinned host RAM and
    // streamed in asynchronously. Default (SIZE_MAX) keeps everything resident
    // unless the BLACKWELL_GPU_LAYERS environment variable overrides it.
    explicit BlackwellEngine(const std::string& index_path, size_t max_seq_len = 2048,
                             size_t num_gpu_layers = static_cast<size_t>(-1),
                             KVCacheMode kv_mode = KVCacheMode::Continuous);
    ~BlackwellEngine();

    // seq_id selects which sequence to decode (Paged mode; default 0). It is the
    // LAST parameter so existing positional calls -- forward(tok, pos) and
    // forward(tok, pos, temp, top_p) -- keep binding temperature/top_p correctly;
    // putting an int before the float defaults would silently capture them.
    // Continuous mode supports only seq_id 0.
    int forward(int token_id, int pos, float temperature = 0.6f, float top_p = 0.9f,
                int seq_id = 0);
    float forward_eval(int token_id, int pos, int target_token_id, int seq_id = 0);

    // Sequence branching (Paged mode only; throws under Continuous). fork shares
    // the parent's KV pages via CoW; rewind rolls a sequence back to `pos` tokens.
    // A forked child is decoded by passing its id as seq_id to forward().
    void fork(int parent_id, int child_id);
    void rewind(int seq_id, int pos);

    Impl* get_impl() const { return pImpl.get(); }

private:
    std::unique_ptr<Impl> pImpl;
};