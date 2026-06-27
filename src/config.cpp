#include "blackwell/config.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>

// Map the raw JSON quantization fields to an internal QuantStrategy.
// All ambiguity is resolved here so no other code ever inspects strings or bit widths.
static QuantStrategy resolve_quant_strategy(const std::string& method, int bits) {
    // Weight-only packed int4: AWQ and GPTQ both use the same packed-weight path.
    if (method == "awq" || method == "gptq") {
        if (bits != 4)
            throw std::runtime_error(
                "ConfigLoader: quant_method \"" + method + "\" requires bits=4, got " +
                std::to_string(bits));
        return QuantStrategy::WEIGHT_ONLY_PACKED;
    }

    // Per-row FP8 weights with per-token activation scaling.
    if (method == "fp8") {
        return QuantStrategy::ROWWISE_FP8;
    }

    // 🎯 HUGGINGFACE/NEURAL MAGIC MUTANT CHECKPOINT WORKAROUND:
    // Some official vLLM/Neural Magic export scripts for FP8 models generate a corrupted
    // quantization_config block where "quant_method" is set to "compressed-tensors", but
    // "bits" mistakenly retains the baseline model's value (16) instead of 8.
    // Since "compressed-tensors" with bits=4 is handled as AWQ/Packed, any other bit width
    // (8 or mislabelled 16) in the Neural Magic ecosystem signifies a row-wise FP8 execution path.
    //
    // NOTE: this legacy heuristic only fires for the FLAT compressed-tensors manifest (a
    // top-level "bits" field). The newer config_groups / pack-quantized schema is detected
    // ahead of this call and routed to COMPRESSED_TENSORS_INT4 directly.
    if (method == "compressed-tensors") {
        if (bits == 4) {
            return QuantStrategy::WEIGHT_ONLY_PACKED;
        } else {
            // Fallback for both proper 8-bit and mislabelled 16-bit FP8 manifests
            return QuantStrategy::ROWWISE_FP8;
        }
    }

    // Explicit unquantized path.
    if (method == "none") {
        return QuantStrategy::NONE;
    }

    throw std::runtime_error(
        "ConfigLoader: unrecognised quantization configuration "
        "(quant_method=\"" + method + "\", bits=" + std::to_string(bits) + "). "
        "Supported combinations: awq/4, gptq/4, fp8/8, compressed-tensors/8, none.");
}

ModelConfig ConfigLoader::load_from_json(const std::string& json_path) {
    std::ifstream f(json_path);
    if (!f.is_open())
        throw std::runtime_error("ConfigLoader: cannot open " + json_path);

    nlohmann::json j;
    f >> j;

    // Multimodal / hybrid checkpoints (Qwen3.5: "Qwen3_5ForConditionalGeneration")
    // nest every language-model hyperparameter under "text_config". Legacy flat
    // checkpoints (Qwen2.5) keep them at the document root. Resolve a single "text
    // root" so the rest of the parser is layout-agnostic.
    const nlohmann::json& t =
        (j.contains("text_config") && j.at("text_config").is_object()) ? j.at("text_config") : j;

    const std::string model_type = t.value("model_type", j.value("model_type", std::string("unknown")));

    ModelConfig cfg{};

    cfg.hidden_dim          = t.at("hidden_size").get<size_t>();
    cfg.intermediate_dim    = t.at("intermediate_size").get<size_t>();
    cfg.num_layers          = t.at("num_hidden_layers").get<size_t>();
    cfg.num_attention_heads = t.at("num_attention_heads").get<size_t>();
    cfg.num_key_value_heads = t.at("num_key_value_heads").get<size_t>();
    cfg.vocab_size          = t.at("vocab_size").get<size_t>();

    if (cfg.hidden_dim == 0 || cfg.intermediate_dim == 0 || cfg.num_layers == 0 ||
        cfg.num_attention_heads == 0 || cfg.num_key_value_heads == 0 || cfg.vocab_size == 0)
        throw std::runtime_error("ConfigLoader: " + json_path +
                                 " declares a zero-sized model dimension");
    if (cfg.num_attention_heads % cfg.num_key_value_heads != 0)
        throw std::runtime_error(
            "ConfigLoader: num_attention_heads (" + std::to_string(cfg.num_attention_heads) +
            ") is not a multiple of num_key_value_heads (" +
            std::to_string(cfg.num_key_value_heads) + "); GQA grouping would be broken");

    cfg.head_dim            = t.value("head_dim", cfg.hidden_dim / cfg.num_attention_heads);
    if (cfg.head_dim == 0)
        throw std::runtime_error("ConfigLoader: head_dim resolved to zero");

    cfg.rms_norm_eps        = t.value("rms_norm_eps", 1e-6f);
    cfg.tie_word_embeddings = j.value("tie_word_embeddings", t.value("tie_word_embeddings", false));
    cfg.attn_output_gate    = t.value("attn_output_gate", false);

    // RoPE. Qwen3.5 wraps the parameters in a "rope_parameters" object and applies a
    // PARTIAL rotary (only head_dim * partial_rotary_factor channels are rotated).
    // Flat checkpoints expose a top-level "rope_theta" and rotate the full head.
    if (t.contains("rope_parameters") && t.at("rope_parameters").is_object()) {
        const auto& rp = t.at("rope_parameters");
        cfg.rope_theta = rp.value("rope_theta", 10000.0f);
        const float prf = rp.value("partial_rotary_factor", 1.0f);
        cfg.rotary_dim = static_cast<size_t>(static_cast<double>(cfg.head_dim) * prf);
        if (cfg.rotary_dim == 0 || cfg.rotary_dim > cfg.head_dim)
            cfg.rotary_dim = cfg.head_dim;
        if (rp.value("mrope_interleaved", false))
            std::cerr << "[ConfigLoader] WARNING: config requests interleaved M-RoPE "
                      << "(mrope_section) but the RoPE kernel applies vanilla 1D rotate_half; "
                      << "multimodal positional encoding will be WRONG.\n";
    } else {
        cfg.rope_theta = t.value("rope_theta", 10000.0f);
        cfg.rotary_dim = cfg.head_dim;
    }

    // Qwen2-family checkpoints (incl. Qwen2.5-Coder) hardcode q/k/v bias in the
    // modeling code and OMIT "attention_bias" from config.json entirely, so for
    // model_type "qwen2" the key's absence means true, not false. Qwen3/Qwen3.5
    // dropped the bias (and added q_norm/k_norm instead), so they set it false.
    cfg.has_qkv_bias = t.value("attention_bias", model_type == "qwen2");

    // q_norm/k_norm (per-head RMSNorm before RoPE) ships with the Qwen3 family.
    // This is a config-level heuristic; the weight binder makes the authoritative
    // decision by probing for the q_norm tensor and may override this flag.
    cfg.has_qk_norm = (model_type.rfind("qwen3", 0) == 0);

    // ----- Hybrid layer topology (Qwen3.5: interleaved linear-attention + full) ----
    // "layer_types" enumerates each layer as "linear_attention" or "full_attention".
    // Absent => legacy uniform full-attention; layer_types stays empty and downstream
    // treats every layer as AttnKind::Full.
    cfg.full_attention_interval = t.value("full_attention_interval", 0);
    if (t.contains("layer_types") && t.at("layer_types").is_array()) {
        cfg.layer_types.reserve(cfg.num_layers);
        for (const auto& lt : t.at("layer_types")) {
            const std::string s = lt.get<std::string>();
            cfg.layer_types.push_back(s == "full_attention" ? AttnKind::Full : AttnKind::Linear);
        }
        if (cfg.layer_types.size() != cfg.num_layers)
            throw std::runtime_error(
                "ConfigLoader: layer_types length (" + std::to_string(cfg.layer_types.size()) +
                ") != num_hidden_layers (" + std::to_string(cfg.num_layers) + ")");
    }

    // Linear-attention (SSM) block geometry. Only consulted for AttnKind::Linear
    // layers; defaults are harmless for pure-Full models.
    cfg.linear.num_key_heads   = t.value("linear_num_key_heads",   size_t{0});
    cfg.linear.num_value_heads = t.value("linear_num_value_heads", size_t{0});
    cfg.linear.key_head_dim    = t.value("linear_key_head_dim",    size_t{0});
    cfg.linear.value_head_dim  = t.value("linear_value_head_dim",  size_t{0});
    cfg.linear.conv_kernel_dim = t.value("linear_conv_kernel_dim", size_t{0});

    // The RoPE kernel implements vanilla rotate_half only; a rope_scaling block
    // (llama3 / YaRN / linear) is NOT honored. The angles coincide at pos=0 but
    // diverge for every later position, so warn loudly instead of failing hard.
    if (t.contains("rope_scaling") && !t.at("rope_scaling").is_null()) {
        const std::string rope_type =
            t.at("rope_scaling").value("rope_type", t.at("rope_scaling").value("type", "unknown"));
        std::cerr << "[ConfigLoader] WARNING: config declares rope_scaling (rope_type=\""
                  << rope_type << "\") but the RoPE kernel applies vanilla rotate_half "
                  << "frequencies; positional encoding is WRONG for pos >= 1.\n";
    }

    // The decode attention kernel attends over the full causal prefix.
    if (t.value("use_sliding_window", false))
        throw std::runtime_error(
            "ConfigLoader: use_sliding_window=true is not supported by the attention kernel.");

    // ----- Quantization -----------------------------------------------------------
    // Two manifest schemas are supported:
    //   (a) flat: quantization_config.{quant_method,bits,group_size}   (AWQ / GPTQ / NM-FP8)
    //   (b) compressed-tensors "pack-quantized": the real bit width / group size live
    //       inside config_groups.<group>.weights.{num_bits,group_size,symmetric}.
    if (j.contains("quantization_config")) {
        const auto& qc = j.at("quantization_config");
        const std::string method =
            qc.value("quant_type", qc.value("quant_method", std::string("none")));
        const std::string format = qc.value("format", std::string());

        if (format == "pack-quantized" && qc.contains("config_groups") &&
            qc.at("config_groups").is_object() && !qc.at("config_groups").empty()) {
            // (b) compressed-tensors PACK-quantized int4. Read the first config
            // group's weight spec — every Linear target in this checkpoint shares one
            // group. Only "pack-quantized" routes here; other compressed-tensors
            // formats (e.g. FP8 "naive-quantized", whose weights carry null group_size
            // and type "float") deliberately fall through to the flat legacy path
            // below so the existing ROWWISE_FP8 routing is preserved byte-for-byte.
            const auto& weights = qc.at("config_groups").begin().value().at("weights");
            // null-safe: a present-but-null field must behave like a missing one.
            auto int_or = [](const nlohmann::json& o, const char* k, int d) {
                auto it = o.find(k);
                return (it != o.end() && it->is_number_integer()) ? it->get<int>() : d;
            };
            cfg.quant_bits       = int_or(weights, "num_bits", 4);
            cfg.quant_group_size = int_or(weights, "group_size", 128);
            cfg.quant_method     = method.empty() ? "compressed-tensors" : method;

            const bool symmetric = weights.value("symmetric", true);
            if (!symmetric)
                throw std::runtime_error(
                    "ConfigLoader: asymmetric compressed-tensors pack-quant is not supported "
                    "(only symmetric int4 weight_packed/weight_scale).");
            if (cfg.quant_bits != 4)
                throw std::runtime_error(
                    "ConfigLoader: compressed-tensors pack-quant expected num_bits=4, got " +
                    std::to_string(cfg.quant_bits));

            cfg.quant_strategy = QuantStrategy::COMPRESSED_TENSORS_INT4;
        } else {
            // (a) flat manifest — legacy path.
            cfg.quant_bits       = qc.value("bits", 16);
            cfg.quant_group_size = qc.value("group_size", 128);
            cfg.quant_strategy   = resolve_quant_strategy(method, cfg.quant_bits);
            cfg.quant_method     = method; // kept for VRAMArena weight-routing (memory_pool.cpp)
        }
    } else {
        cfg.quant_bits       = 16;
        cfg.quant_group_size = 128;
        cfg.quant_strategy   = QuantStrategy::NONE;
        cfg.quant_method     = "none";
    }

    return cfg;
}
