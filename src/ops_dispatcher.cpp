#include "ops_dispatcher.h"
#include "kernels/fp8_linear.cuh"
#include "kernels/awq_linear.cuh"
#include <cuda_runtime.h>
#include <stdexcept>

LinearDispatcher::LinearDispatcher(const VRAMArena& arena, const ModelConfig& config)
    : m_arena(arena), m_config(config)
{
    if (m_config.quant_method == "fp8") {
        cudaMalloc(&d_token_scale, 2 * sizeof(float));
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
    const std::string& method = m_config.quant_method;

    if (method == "awq" || method == "gptq") {
        // Weight-only packed quantization: raw float input, no activation scaling.
        QuantizedTensorPtrs ptrs = m_arena.get_quantized_pointers(base_name);
        launch_awq_gemv_kernel(ptrs.qweight, ptrs.scales, ptrs.qzeros,
                               d_in, d_out, out_features, in_features,
                               m_config.quant_group_size);
        // TODO: in-place residual accumulation for AWQ/GPTQ once kernel is implemented.
        (void)d_residual_accum;

    } else if (method == "fp8") {
        // Activation quantization: compute per-token scale, then dispatch FP8 GEMV.
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

    } else {
        // "none" / unquantized: placeholder for future BF16/FP32 GEMV path.
        (void)d_out; (void)d_residual_accum;
        throw std::runtime_error("LinearDispatcher: unsupported quant_method \"" + method + "\"");
    }
}
