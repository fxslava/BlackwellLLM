#pragma once
#include <cstddef>
#include <string>
#include <memory>

class BlackwellEngine {
public:
    struct Impl;

    // num_gpu_layers: how many leading transformer layers keep their weights /
    // KV cache resident in VRAM; the rest are offloaded to pinned host RAM and
    // streamed in asynchronously. Default (SIZE_MAX) keeps everything resident
    // unless the BLACKWELL_GPU_LAYERS environment variable overrides it.
    explicit BlackwellEngine(const std::string& index_path, size_t max_seq_len = 2048,
                             size_t num_gpu_layers = static_cast<size_t>(-1));
    ~BlackwellEngine();

    int forward(int token_id, int pos, float temperature = 0.6f, float top_p = 0.9f);
    float forward_eval(int token_id, int pos, int target_token_id);

    Impl* get_impl() const { return pImpl.get(); }

private:
    std::unique_ptr<Impl> pImpl;
};