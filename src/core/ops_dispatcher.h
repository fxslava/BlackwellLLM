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
    // batched_gemm_threshold: the num_tokens crossover at/above which forward()
    // launches the Tensor-Core batched GEMM instead of sweeping the per-row GEMV
    // (RuntimeConfig::batched_gemm_threshold; validated >= 1 upstream). The engine
    // passes the resolved plan value; the default keeps stand-alone construction
    // (tests/tools) on the batch-16 crossover.
    LinearDispatcher(const VRAMArena& arena, const ModelConfig& config,
                     int batched_gemm_threshold = 16);
    ~LinearDispatcher();

    LinearDispatcher(const LinearDispatcher&) = delete;
    LinearDispatcher& operator=(const LinearDispatcher&) = delete;

    // Execute one linear projection: d_out = W * d_in.
    //
    // base_name        - weight tensor prefix, e.g. "model.layers.0.self_attn.q_proj"
    // d_in             - raw (unquantized) float input, [num_tokens, in_features]
    // d_out            - output buffer on device (ignored when d_residual_accum != nullptr),
    //                    [num_tokens, out_features]
    // out_features     - output dimension M
    // in_features      - input dimension K (also the quantization domain for ROWWISE_FP8)
    // d_residual_accum - if non-null, result is accumulated into this buffer in-place,
    //                    [num_tokens, out_features]
    // num_tokens       - number of activation rows. 1 (default) keeps the exact,
    //                    latency-critical batch=1 GEMV paths. > 1 (batched prefill /
    //                    true batch mode) routes the unquantized BF16 path to the
    //                    Tensor-Core batched GEMM; quantized paths fall back to a
    //                    per-row GEMV sweep (bit-identical to batch=1) since no
    //                    batched quantized kernel exists yet.
    void forward(const std::string& base_name,
                 const float* d_in,
                 float* d_out,
                 size_t out_features,
                 size_t in_features,
                 float* d_residual_accum = nullptr,
                 size_t num_tokens = 1);

private:
    // One projection over a single activation row (the exact batch=1 kernels).
    void forward_row(const std::string& base_name,
                     const float* d_in,
                     float* d_out,
                     size_t out_features,
                     size_t in_features,
                     float* d_residual_accum);

    // The low-latency path: forward_row over each of num_tokens rows. Used below
    // the batched-GEMM crossover and for strategies without a batched kernel.
    void sweep_rows(const std::string& base_name,
                    const float* d_in,
                    float* d_out,
                    size_t out_features,
                    size_t in_features,
                    float* d_residual_accum,
                    size_t num_tokens);

    const VRAMArena& m_arena;
    const ModelConfig& m_config;

    // num_tokens crossover: below it forward() loops the batch=1 GEMV, at/above it
    // launches the batched GEMM (see forward()).
    int m_batched_gemm_threshold;

    // Per-token scale buffer [scale, inv_scale]; owned and allocated only for ROWWISE_FP8.
    float* d_token_scale = nullptr;
};
