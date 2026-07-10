#include "ops_dispatcher.h"
#include "kernels/fp8_linear.cuh"
#include "kernels/awq_linear.cuh"
#include "kernels/bf16_linear.cuh"
#include "kernels/bf16_linear_residual.cuh"
#include "kernels/batched_bf16_gemm.cuh"
#include "kernels/sym_int4_linear.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
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

// ============================================================================
// Dual-path entry point: batch=1 keeps the exact latency-critical GEMV kernels;
// num_tokens > 1 routes each strategy to its Tensor-Core batched GEMM (BF16 /
// AWQ int4 / FP8 E4M3). Only COMPRESSED_TENSORS_INT4 still falls back to a
// per-row GEMV sweep (no batched kernel yet). The batch=1 GEMV kernels
// themselves are UNCHANGED.
// ============================================================================
void LinearDispatcher::forward(const std::string& base_name,
                               const float* d_in,
                               float* d_out,
                               size_t out_features,
                               size_t in_features,
                               float* d_residual_accum,
                               size_t num_tokens)
{
    if (num_tokens == 1) {
        forward_row(base_name, d_in, d_out, out_features, in_features, d_residual_accum);
        return;
    }

    const bool residual = (d_residual_accum != nullptr);

    switch (m_config.quant_strategy) {

    case QuantStrategy::NONE: {
        // Unquantized BF16: one Tensor-Core GEMM over all rows. The biased q/k/v
        // projections never reach the dispatcher (engine.cpp fast path), so only
        // plain and residual variants are needed, matching forward_row.
        const void* w = m_arena.get_weight_ptr(base_name + ".weight");
        if (residual)
            launch_bf16_gemm_residual_batched(w, d_in, d_residual_accum,
                                              out_features, in_features, num_tokens);
        else
            launch_bf16_gemm_batched(w, d_in, d_out, out_features, in_features, num_tokens);
        return;
    }

    case QuantStrategy::WEIGHT_ONLY_PACKED: {
        // AWQ / GPTQ int4: unpack + dequant tiles into shared memory, Tensor-Core GEMM.
        QuantizedTensorPtrs ptrs = m_arena.get_quantized_pointers(base_name);
        if (residual)
            launch_batched_awq_gemm_residual(ptrs.qweight, ptrs.scales, ptrs.qzeros,
                                             d_in, d_residual_accum, out_features,
                                             in_features, m_config.quant_group_size, num_tokens);
        else
            launch_batched_awq_gemm(ptrs.qweight, ptrs.scales, ptrs.qzeros,
                                    d_in, d_out, out_features, in_features,
                                    m_config.quant_group_size, num_tokens);
        return;
    }

    case QuantStrategy::ROWWISE_FP8: {
        // FP8 E4M3: row-wise weight scale + optional per-tensor activation scale,
        // applied uniformly across the batch (matches the GEMV; see the kernel).
        const void* w   = m_arena.get_weight_ptr(base_name + ".weight");
        const void* w_s = m_arena.get_weight_ptr(base_name + ".weight_scale");
        const void* i_s = m_arena.get_weight_ptr_optional(base_name + ".input_scale");
        if (residual)
            launch_batched_fp8_gemm_residual(w, d_in, w_s, i_s, d_token_scale,
                                             d_residual_accum, out_features, in_features,
                                             /*scale_stride=*/1, num_tokens);
        else
            launch_batched_fp8_gemm(w, d_in, w_s, i_s, d_token_scale, d_out,
                                    out_features, in_features, /*scale_stride=*/1, num_tokens);
        return;
    }

    case QuantStrategy::COMPRESSED_TENSORS_INT4:
    default: {
        // No batched symmetric-int4 kernel yet — sweep the rows through the exact
        // batch=1 path. d_in / d_out / d_residual_accum are [num_tokens, *].
        for (size_t t = 0; t < num_tokens; ++t) {
            const float* in_row = d_in + t * in_features;
            float* out_row      = d_out ? d_out + t * out_features : nullptr;
            float* accum_row    = d_residual_accum ? d_residual_accum + t * out_features : nullptr;
            forward_row(base_name, in_row, out_row, out_features, in_features, accum_row);
        }
        return;
    }
    }
}

void LinearDispatcher::forward_row(const std::string& base_name,
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

    case QuantStrategy::COMPRESSED_TENSORS_INT4: {
        // Symmetric int4 pack-quantized (compressed-tensors): weight_packed (int32,
        // 8 nibbles each) + weight_scale (bf16, per group of group_size inputs),
        // no zero-point. Dequant GEMV is validated for self-consistency in
        // test_sym_int4_gemv.cu (cos 1.0); exact-checkpoint parity is pinned by the
        // golden-dump integration test.
        const auto* packed = static_cast<const int32_t*>(
            m_arena.get_weight_ptr(base_name + ".weight_packed"));
        const auto* scales = static_cast<const __nv_bfloat16*>(
            m_arena.get_weight_ptr(base_name + ".weight_scale"));
        if (d_residual_accum == nullptr) {
            launch_sym_int4_gemv(packed, scales, d_in, d_out,
                                 static_cast<int>(out_features), static_cast<int>(in_features),
                                 m_config.quant_group_size);
        } else {
            launch_sym_int4_gemv_residual(packed, scales, d_in, d_residual_accum,
                                          static_cast<int>(out_features), static_cast<int>(in_features),
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
