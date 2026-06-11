#pragma once
#include <string>
#include <cstddef>
#include "memory_pool.h"
#include "blackwell/config.h"

// Routes linear projections to the correct kernel backend based on ModelConfig::quant_strategy.
// Owns all quantization-related intermediate state (e.g. per-token FP8 scale buffers)
// so that callers remain oblivious to the quantization scheme in use.
class LinearDispatcher {
public:
    LinearDispatcher(const VRAMArena& arena, const ModelConfig& config);
    ~LinearDispatcher();

    LinearDispatcher(const LinearDispatcher&) = delete;
    LinearDispatcher& operator=(const LinearDispatcher&) = delete;

    // Execute one linear projection: d_out = W * d_in.
    //
    // base_name        - weight tensor prefix, e.g. "model.layers.0.self_attn.q_proj"
    // d_in             - raw (unquantized) float input vector on device
    // d_out            - output buffer on device (ignored when d_residual_accum != nullptr)
    // out_features     - output dimension M
    // in_features      - input dimension K (also the quantization domain for ROWWISE_FP8)
    // d_residual_accum - if non-null, result is accumulated into this buffer in-place
    void forward(const std::string& base_name,
                 const float* d_in,
                 float* d_out,
                 size_t out_features,
                 size_t in_features,
                 float* d_residual_accum = nullptr);

private:
    const VRAMArena& m_arena;
    const ModelConfig& m_config;

    // Per-token scale buffer [scale, inv_scale]; owned and allocated only for ROWWISE_FP8.
    float* d_token_scale = nullptr;
};
