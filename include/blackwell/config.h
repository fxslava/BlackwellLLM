#pragma once

#include <string>
#include <vector>

// Strongly-typed representation of the quantization strategy in use.
// Mapped from raw JSON strings once at load time in ConfigLoader; the rest of the
// codebase branches on this enum and never inspects raw method strings or bit widths.
enum class QuantStrategy {
    NONE,                   // Unquantized / BF16 / FP32 weights
    ROWWISE_FP8,            // Per-row FP8 weight quantization with per-token activation scaling
    WEIGHT_ONLY_PACKED,     // Weight-only packed int4 (AWQ / GPTQ) — asymmetric, qweight/qzeros/scales
    COMPRESSED_TENSORS_INT4 // compressed-tensors "pack-quantized" — symmetric int4, weight_packed/weight_scale (no zero-point)
};

// Per-layer attention kind for hybrid stacks (Qwen3.5: interleaved linear-attention
// SSM blocks and periodic full softmax-attention layers). Legacy uniform models
// leave ModelConfig::layer_types empty, which every consumer must treat as "all Full".
enum class AttnKind { Full, Linear };

struct ModelConfig {
    size_t hidden_dim;
    size_t intermediate_dim;
    size_t num_layers;
    size_t num_attention_heads;
    size_t num_key_value_heads;
    size_t vocab_size;
    size_t head_dim;
    size_t rotary_dim;            // channels per head that receive RoPE (head_dim * partial_rotary_factor)

    float rope_theta;
    float rms_norm_eps;

    bool has_qkv_bias;
    bool tie_word_embeddings;
    bool has_qk_norm;             // full-attention layers carry per-head q_norm/k_norm (Qwen3 family)
    bool attn_output_gate;        // gated attention output (Qwen3.5)

    // Hybrid layer topology. Empty == legacy uniform full-attention (Qwen2.5 flat).
    std::vector<AttnKind> layer_types;
    int full_attention_interval;  // 0 when not applicable / uniform

    // Geometry of the linear-attention (SSM) blocks. Only meaningful when
    // layer_types contains AttnKind::Linear entries.
    struct LinearAttnConfig {
        size_t num_key_heads;
        size_t num_value_heads;
        size_t key_head_dim;
        size_t value_head_dim;
        size_t conv_kernel_dim;
    } linear;

    QuantStrategy quant_strategy; // primary: use this for all control-flow decisions
    std::string quant_method;     // kept for VRAMArena internal weight-routing logic
    int quant_bits;
    int quant_group_size;

    // Tensor-name prefix for the decoder body. Flat checkpoints (Qwen2.5/Llama)
    // use "model."; multimodal Qwen3.5 nests the language model, so its weights
    // are "model.language_model.*". Embeddings/final-norm hang off this prefix;
    // lm_head is always top-level "lm_head.weight".
    std::string weight_prefix;
};

class ConfigLoader {
public:
    static ModelConfig load_from_json(const std::string& json_path);
};
