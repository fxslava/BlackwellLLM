#pragma once

#include <string>

// Strongly-typed representation of the quantization strategy in use.
// Mapped from raw JSON strings once at load time in ConfigLoader; the rest of the
// codebase branches on this enum and never inspects raw method strings or bit widths.
enum class QuantStrategy {
    NONE,               // Unquantized / BF16 / FP32 weights
    ROWWISE_FP8,        // Per-row FP8 weight quantization with per-token activation scaling
    WEIGHT_ONLY_PACKED, // Weight-only packed int4 (AWQ / GPTQ)
};

struct ModelConfig {
    size_t hidden_dim;
    size_t intermediate_dim;
    size_t num_layers;
    size_t num_attention_heads;
    size_t num_key_value_heads;
    size_t vocab_size;
    size_t head_dim;

    float rope_theta;
    float rms_norm_eps;

    bool has_qkv_bias;
    bool tie_word_embeddings;

    QuantStrategy quant_strategy; // primary: use this for all control-flow decisions
    std::string quant_method;     // kept for VRAMArena internal weight-routing logic
    int quant_bits;
    int quant_group_size;
};

class ConfigLoader {
public:
    static ModelConfig load_from_json(const std::string& json_path);
};
