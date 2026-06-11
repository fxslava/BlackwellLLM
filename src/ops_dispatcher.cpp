#include "ops_dispatcher.h"
#include "kernels/fp8_linear.cuh"
#include "kernels/awq_linear.cuh"

void LinearDispatcher::forward(const std::string& base_name,
                               const float* d_in,
                               float* d_out,
                               size_t out_features,
                               size_t in_features,
                               const float* d_token_scale,
                               float* d_residual_accum)
{
    if (m_config.quant_bits == 4) {
        QuantizedTensorPtrs ptrs = m_arena.get_quantized_pointers(base_name);
        launch_awq_gemv_kernel(ptrs.qweight, ptrs.scales, ptrs.qzeros,
                               d_in, d_out, out_features, in_features,
                               m_config.quant_group_size);
        // TODO: fused residual add for AWQ path once kernel is implemented.
        (void)d_residual_accum;
    } else {
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
    }
}
