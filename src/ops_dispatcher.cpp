#include "ops_dispatcher.h"
#include "kernels/fp8_linear.cuh"
#include "kernels/awq_linear.cuh"
#include "kernels/bf16_linear.cuh"
#include "kernels/bf16_linear_residual.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

LinearDispatcher::LinearDispatcher(const VRAMArena& arena, const ModelConfig& config)
    : m_arena(arena), m_config(config)
{
    if (m_config.quant_strategy == QuantStrategy::ROWWISE_FP8) {
        const cudaError_t err = cudaMalloc(&d_token_scale, 2 * sizeof(float));
        if (err != cudaSuccess) {
            d_token_scale = nullptr; // value is unspecified on failure
            throw std::runtime_error(
                "LinearDispatcher: failed to allocate per-token scale buffer: " +
                std::string(cudaGetErrorString(err)));
        }
    }
}

LinearDispatcher::~LinearDispatcher() {
    if (d_token_scale) cudaFree(d_token_scale);
}

void LinearDispatcher::forward(const std::string& base_name,
                               const float* d_in,
                               float* d_out,
                               size_t out_features,
                               size_t in_features,
                               float* d_residual_accum)
{
    switch (m_config.quant_strategy) {

    case QuantStrategy::WEIGHT_ONLY_PACKED: {
        // AWQ / GPTQ: raw float input, no activation scaling needed.
        QuantizedTensorPtrs ptrs = m_arena.get_quantized_pointers(base_name);
        if (d_residual_accum == nullptr) {
            launch_awq_gemv_kernel(ptrs.qweight, ptrs.scales, ptrs.qzeros,
                                   d_in, d_out, out_features, in_features,
                                   m_config.quant_group_size);
        } else {
            launch_awq_gemv_residual_kernel(ptrs.qweight, ptrs.scales, ptrs.qzeros,
                                            d_in, d_residual_accum,
                                            out_features, in_features,
                                            m_config.quant_group_size);
        }
        break;
    }

    case QuantStrategy::ROWWISE_FP8: {
        // Compute per-token activation scale, then dispatch the appropriate FP8 GEMV.
        launch_quantize_per_token_kernel(d_in, d_token_scale, in_features);

        const void* w   = m_arena.get_weight_ptr(base_name + ".weight");
        const void* w_s = m_arena.get_weight_ptr(base_name + ".weight_scale");
        const void* i_s = m_arena.get_weight_ptr_optional(base_name + ".input_scale");

        if (d_residual_accum == nullptr) {
            launch_fp8_gemv_kernel(w, d_in, w_s, i_s, d_token_scale,
                                   d_out, out_features, in_features, 1);
        } else {
            launch_fp8_gemv_residual_kernel(w, d_in, w_s, i_s, d_token_scale,
                                            d_residual_accum, out_features, in_features, 1);
        }
        break;
    }

    case QuantStrategy::NONE: {
        // Unquantized BF16 GEMV. The biased q/k/v projections never reach the
        // dispatcher (engine.cpp routes them to the fused GEMV+bias fast path),
        // so only the plain and residual-accumulating variants are needed here.
        const void* w = m_arena.get_weight_ptr(base_name + ".weight");

        if (d_residual_accum == nullptr) {
            launch_bf16_gemv_kernel(w, d_in, d_out, out_features, in_features);
        } else {
            launch_bf16_gemv_residual_kernel(w, d_in, d_residual_accum,
                                             out_features, in_features);
        }
        break;
    }

    default:
        // Defensive: ModelConfig::quant_strategy is set once by ConfigLoader, but a
        // silent fall-through here would leave d_out untouched and corrupt decoding.
        throw std::logic_error("LinearDispatcher: unhandled QuantStrategy for " + base_name);
    }
}
