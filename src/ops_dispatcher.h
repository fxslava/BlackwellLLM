#pragma once
#include <string>
#include <cstddef>
#include "memory_pool.h"
#include "blackwell/config.h"

// Routes linear projections to FP8 or AWQ kernels based on ModelConfig::quant_bits.
class LinearDispatcher {
public:
    LinearDispatcher(const VRAMArena& arena, const ModelConfig& config)
        : m_arena(arena), m_config(config) {}

    // Runs a linear projection: d_out = W * d_in, with optional residual accumulation.
    //
    // base_name    - weight tensor prefix (e.g. "model.layers.0.self_attn.q_proj")
    // d_in         - normalized input vector on device
    // d_out        - output buffer on device
    // out_features - output dimension M
    // in_features  - input dimension K
    // d_token_scale - per-token quantization scale (FP8 path only)
    // d_residual_accum - if non-null, result is added into this buffer (residual connection)
    void forward(const std::string& base_name,
                 const float* d_in,
                 float* d_out,
                 size_t out_features,
                 size_t in_features,
                 const float* d_token_scale,
                 float* d_residual_accum = nullptr);

private:
    const VRAMArena& m_arena;
    const ModelConfig& m_config;
};
