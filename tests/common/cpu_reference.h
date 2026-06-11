#pragma once
// CPU reference implementations of every engine kernel. Validation suites
// assert GPU results against these; they favour clarity and FP64 accumulation
// over speed. Also consumed by src/experiments via the compatibility shim
// tests/reference/cpu_models.h.
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_bf16.h>

// Several kernels deliberately snap intermediate values onto the bfloat16
// grid for bitwise parity with the PyTorch BF16 reference implementation;
// their CPU references below replicate that with the same RNE conversion.
inline float cpu_bf16_round(float v) {
    return __bfloat162float(__float2bfloat16(v));
}

// The embedding table is a [vocab_size, hidden_dim] matrix: take each token id
// and copy the corresponding row into the output buffer.
inline void cpu_embedding_lookup(const int* tokens,
                                 const float* embed_table,
                                 float* output,
                                 size_t seq_len,
                                 size_t hidden_dim) {
    for (size_t s = 0; s < seq_len; ++s) {
        int token_id = tokens[s];
        const float* src_row = embed_table + (size_t)token_id * hidden_dim;
        float* dst_row = output + s * hidden_dim;
        for (size_t h = 0; h < hidden_dim; ++h) {
            dst_row[h] = src_row[h];
        }
    }
}

// Fused residual-add + RMS normalization, matching the GPU kernel's HF/BF16
// semantics exactly (see rmsnorm.cu): the updated residual stream is snapped
// onto the bf16 grid, the sum of squares accumulates in double, weights are
// stored as bf16, and the output is bf16(bf16(res * rsqrt) * weight).
// x_out    - input vector X (overwritten with the normalized output Y)
// residual - accumulator buffer (overwritten with bf16(residual + X))
// weight   - normalization layer weights (g) as bf16
inline void cpu_rmsnorm_residual(float* x_out,
                                 float* residual,
                                 const __nv_bfloat16* weight,
                                 size_t seq_len,
                                 size_t hidden_dim,
                                 float eps = 1e-5f) {
    for (size_t s = 0; s < seq_len; ++s) {
        float* cur_x = x_out + s * hidden_dim;
        float* cur_res = residual + s * hidden_dim;

        // 1. Residual add (snapped to bf16) and double sum of squares.
        double sum_sq = 0.0;
        for (size_t h = 0; h < hidden_dim; ++h) {
            cur_res[h] = cpu_bf16_round(cur_res[h] + cur_x[h]);
            sum_sq += (double)cur_res[h] * (double)cur_res[h];
        }

        // 2. Inverse RMS.
        float rsqrt = (float)(1.0 / std::sqrt(sum_sq / hidden_dim + (double)eps));

        // 3. Normalize, cast to bf16, multiply by bf16 weight, store as bf16.
        for (size_t h = 0; h < hidden_dim; ++h) {
            float norm_bf16 = cpu_bf16_round(cur_res[h] * rsqrt);
            cur_x[h] = cpu_bf16_round(norm_bf16 * __bfloat162float(weight[h]));
        }
    }
}

// Direct RMS normalization with the same HF/BF16 semantics; the input is
// read in full float precision and left untouched.
inline void cpu_rmsnorm_direct(const float* input,
                               float* output,
                               const __nv_bfloat16* weight,
                               size_t seq_len,
                               size_t hidden_dim,
                               float eps = 1e-5f)
{
    for (size_t s = 0; s < seq_len; ++s) {
        const float* cur_input = input + s * hidden_dim;
        float* cur_output = output + s * hidden_dim;

        double sum_sq = 0.0;
        for (size_t h = 0; h < hidden_dim; ++h) {
            sum_sq += (double)cur_input[h] * (double)cur_input[h];
        }

        float rsqrt = (float)(1.0 / std::sqrt(sum_sq / hidden_dim + (double)eps));

        for (size_t h = 0; h < hidden_dim; ++h) {
            float norm_bf16 = cpu_bf16_round(cur_input[h] * rsqrt);
            cur_output[h] = cpu_bf16_round(norm_bf16 * __bfloat162float(weight[h]));
        }
    }
}

// Exact unpacking of one FP8 (E4M3) byte into a standard float.
inline float cpu_unpack_fp8_e4m3(uint8_t byte_val) {
    // Signed zero.
    if ((byte_val & 0x7F) == 0) {
        return (byte_val & 0x80) ? -0.0f : 0.0f;
    }

    int sign = (byte_val & 0x80) ? -1 : 1;
    int exp  = (byte_val & 0x78) >> 3;
    int mant = byte_val & 0x07;

    if (exp == 0) {
        // Subnormals (exponent fixed at 1 - bias = -6).
        return sign * std::ldexp(static_cast<float>(mant) / 8.0f, -6);
    }

    // Normals (bias = 7).
    return sign * std::ldexp(1.0f + static_cast<float>(mant) / 8.0f, exp - 7);
}

// Reference quantized FP8 matrix-vector product.
// W_fp8  - [M, K] matrix of raw E4M3 bytes
// X      - activation vector [K]
// scales - per-row scale vector [M]
// Y_out  - output vector [M]
inline void cpu_fp8_gemv(const uint8_t* W_fp8,
                         const float* X,
                         const float* scales,
                         float* Y_out,
                         size_t M,
                         size_t K) {
    for (size_t i = 0; i < M; ++i) {
        double row_sum = 0.0;
        const uint8_t* cur_W_row = W_fp8 + i * K;

        for (size_t j = 0; j < K; ++j) {
            float w_float = cpu_unpack_fp8_e4m3(cur_W_row[j]);
            row_sum += w_float * X[j];
        }

        Y_out[i] = static_cast<float>(row_sum) * scales[i];
    }
}

// Apply RoPE to Q and K while simultaneously writing K and V into the cache.
// Operates on a single token at position pos.
inline void cpu_fused_rope_kv_append(
    float* Q,               // In/out query vector [q_heads * head_dim]
    float* K,               // Input key vector [kv_heads * head_dim]
    const float* V,         // Input value vector [kv_heads * head_dim]
    float* K_cache,         // Global key cache
    float* V_cache,         // Global value cache
    int pos,                // Current token position
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,        // Per-head dimension (typically 128)
    size_t max_seq_len,     // Cache capacity
    float rope_theta = 500000.0f)
{
    // The GPU kernel implements the Hugging Face rotate_half convention
    // (see rope.cu): channel k pairs with channel k + head_dim/2, and the
    // frequency for pair k is theta^(-2k/head_dim).
    const size_t half_dim = head_dim / 2;

    // 1. Queries (Q).
    for (size_t h = 0; h < q_heads; ++h) {
        float* cur_q = Q + h * head_dim;

        for (size_t k = 0; k < half_dim; ++k) {
            float freq = 1.0f / std::pow(rope_theta, static_cast<float>(2 * k) / head_dim);
            float angle = pos * freq;
            float cos_val = std::cos(angle);
            float sin_val = std::sin(angle);

            float q0 = cur_q[k];
            float q1 = cur_q[k + half_dim];

            cur_q[k]            = q0 * cos_val - q1 * sin_val;
            cur_q[k + half_dim] = q0 * sin_val + q1 * cos_val;
        }
    }

    // 2. Keys (K) and K/V cache append. The cache layout is linear:
    // [kv_heads, max_seq_len, head_dim].
    for (size_t h = 0; h < kv_heads; ++h) {
        float* cur_k = K + h * head_dim;
        const float* cur_v = V + h * head_dim;

        float* k_slot = K_cache + (h * max_seq_len + pos) * head_dim;
        float* v_slot = V_cache + (h * max_seq_len + pos) * head_dim;

        for (size_t k = 0; k < half_dim; ++k) {
            float freq = 1.0f / std::pow(rope_theta, static_cast<float>(2 * k) / head_dim);
            float angle = pos * freq;
            float cos_val = std::cos(angle);
            float sin_val = std::sin(angle);

            float k0 = cur_k[k];
            float k1 = cur_k[k + half_dim];

            float k0_rot = k0 * cos_val - k1 * sin_val;
            float k1_rot = k0 * sin_val + k1 * cos_val;

            // Overwrite local K and append to the cache.
            cur_k[k]            = k0_rot;
            cur_k[k + half_dim] = k1_rot;

            k_slot[k]            = k0_rot;
            k_slot[k + half_dim] = k1_rot;

            // V is copied into its cache untouched.
            v_slot[k]            = cur_v[k];
            v_slot[k + half_dim] = cur_v[k + half_dim];
        }
    }
}

// Reference attention for the decoding phase (single-token generation).
// Q       - query vector of the current token [q_heads, head_dim]
// K_cache - global key cache [kv_heads, max_seq_len, head_dim]
// V_cache - global value cache [kv_heads, max_seq_len, head_dim]
// O_out   - output vector [q_heads, head_dim]
inline void cpu_attention_decoding(
    const float* Q,
    const float* K_cache,
    const float* V_cache,
    float* O_out,
    int pos,                // Current position (tokens already in the cache)
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t max_seq_len)
{
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    size_t gqa_ratio = q_heads / kv_heads;

    for (size_t qh = 0; qh < q_heads; ++qh) {
        size_t kvh = qh / gqa_ratio;  // K/V head shared by this query head

        const float* cur_q = Q + qh * head_dim;
        std::vector<float> scores(pos + 1, 0.0f);
        float max_score = -INFINITY;

        // 1. Q * K^T over every cached token from 0 to pos.
        for (int t = 0; t <= pos; ++t) {
            const float* cur_k = K_cache + (kvh * max_seq_len + t) * head_dim;
            float dot = 0.0f;
            for (size_t i = 0; i < head_dim; ++i) {
                dot += cur_q[i] * cur_k[i];
            }
            float score = dot * scale;
            scores[t] = score;
            if (score > max_score) {
                max_score = score;
            }
        }

        // 2. Softmax (max-subtracted for stability).
        float sum_exp = 0.0f;
        for (int t = 0; t <= pos; ++t) {
            scores[t] = std::exp(scores[t] - max_score);
            sum_exp += scores[t];
        }

        // 3. Weighted sum over V.
        float* cur_o = O_out + qh * head_dim;
        for (size_t i = 0; i < head_dim; ++i) {
            cur_o[i] = 0.0f;
        }

        for (int t = 0; t <= pos; ++t) {
            float prob = scores[t] / sum_exp;
            const float* cur_v = V_cache + (kvh * max_seq_len + t) * head_dim;
            for (size_t i = 0; i < head_dim; ++i) {
                cur_o[i] += prob * cur_v[i];
            }
        }
    }
}

// Reference fused SwiGLU, matching the GPU kernel's BF16-parity semantics
// (see swiglu.cu): inputs are truncated to bf16, F.silu(gate) is cast to
// bf16 before multiplying by up, and the output lands on the bf16 grid.
// num_elements is the total element count (for decoding this is simply
// intermediate_dim).
inline void cpu_fused_swiglu(const float* gate,
                             const float* up,
                             float* output,
                             size_t num_elements)
{
    for (size_t i = 0; i < num_elements; ++i) {
        float g = cpu_bf16_round(gate[i]);
        float u = cpu_bf16_round(up[i]);

        // SiLU / Swish activation.
        float sigmoid = 1.0f / (1.0f + std::exp(-g));
        float swish_bf16 = cpu_bf16_round(g * sigmoid);

        output[i] = cpu_bf16_round(swish_bf16 * u);
    }
}

// Reference argmax over logits.
inline void cpu_argmax(const float* logits, int* out_token_id, size_t vocab_size) {
    float max_val = logits[0];
    int max_idx = 0;

    for (size_t i = 1; i < vocab_size; ++i) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = static_cast<int>(i);
        }
    }

    *out_token_id = max_idx;
}
