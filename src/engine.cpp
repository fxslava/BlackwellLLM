#include "engine_impl.h"
#include "blackwell/engine.h"
#include "common.h"
#include "kernels/embedding.cuh"
#include "kernels/rmsnorm.cuh"
#include "kernels/bf16_linear.cuh"
#include "kernels/bf16_linear_bias.cuh"
#include "kernels/bias.cuh"
#include "kernels/rope.cuh"
#include "kernels/attention.cuh"
#include "kernels/swiglu.cuh"
#include "kernels/sampling.cuh"
#include "kv_cache/continuous_kv_manager.h"
#include "kv_cache/paged_kv_manager.h"
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <stdexcept>

// SIZE_MAX (= "all resident") may be overridden by BLACKWELL_GPU_LAYERS so the
// VRAM split is tunable without touching any call site.
static size_t resolve_num_gpu_layers(size_t requested) {
    if (requested != static_cast<size_t>(-1)) return requested;
    if (const char* env = std::getenv("BLACKWELL_GPU_LAYERS")) {
        char* end = nullptr;
        const unsigned long long v = std::strtoull(env, &end, 10);
        if (end != env && *end == '\0') return static_cast<size_t>(v);
        throw std::invalid_argument(
            "BLACKWELL_GPU_LAYERS is not a valid non-negative integer: " + std::string(env));
    }
    return static_cast<size_t>(-1);
}

// Initializer list mirrors the declaration order in engine_impl.h: members are
// constructed in declaration order regardless of the list, and arena consumes
// both loader and m_config, so the textual order must not suggest otherwise.
BlackwellEngine::Impl::Impl(const std::string& index_path, size_t max_seq_len, size_t num_gpu_layers,
                            BlackwellEngine::KVCacheMode kv_mode)
    : m_config(ConfigLoader::load_from_json(
          (std::filesystem::path(index_path).parent_path() / "config.json").string())),
      loader(index_path),
      arena(index_path, loader, m_config, max_seq_len, resolve_num_gpu_layers(num_gpu_layers)),
      dispatcher(arena, m_config)
{
    // 1. Bind core activation buffers from the arena
    d_X_accum = arena.get_activation_buffer_A();
    d_X_norm  = arena.get_activation_buffer_B();

    // 2. Allocate layer-scoped compute buffers once
    CUDA_CHECK(cudaMalloc(&d_Q,        m_config.num_attention_heads * m_config.head_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K,        m_config.num_key_value_heads * m_config.head_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V,        m_config.num_key_value_heads * m_config.head_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Attn_out, m_config.hidden_dim * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_Gate,       m_config.intermediate_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Up,         m_config.intermediate_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Swiglu_out, m_config.intermediate_dim * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_logits, m_config.vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_next_token, sizeof(int)));

    // Resident pool may be empty when every layer is offloaded to host RAM.
    if (arena.get_k_cache_size() > 0) {
        CUDA_CHECK(cudaMemset(arena.get_k_cache(), 0, arena.get_k_cache_size()));
        CUDA_CHECK(cudaMemset(arena.get_v_cache(), 0, arena.get_v_cache_size()));
    }

    // KV-cache strategy. Continuous (default) wraps the existing VRAMArena flow
    // (behavior-preserving). Paged uses the bf16 CoW cache + paged-flash kernel.
    if (kv_mode == BlackwellEngine::KVCacheMode::Paged) {
        kv_mgr = std::make_unique<blackwell::PagedKVManager>(m_config, max_seq_len);
    } else {
        kv_mgr = std::make_unique<blackwell::ContinuousKVManager>(arena, m_config);
    }
}

BlackwellEngine::Impl::~Impl() {
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_Attn_out); cudaFree(d_Gate); cudaFree(d_Up); cudaFree(d_Swiglu_out);
    cudaFree(d_logits); cudaFree(d_next_token);
}

// ============================================================================
// STAGE 1: Embedding
// ============================================================================
// AWQ/GPTQ checkpoints store every non-quantized tensor (embeddings, norm
// weights, lm_head, biases) in FP16; BF16/FP8 checkpoints use bfloat16. The
// same heuristic already routes the QKV bias dtype below.
static bool half_weights_are_fp16(const ModelConfig& cfg) {
    return cfg.quant_strategy == QuantStrategy::WEIGHT_ONLY_PACKED;
}

void BlackwellEngine::Impl::step_embedding(int token_id) {
    // d_next_token doubles as the persistent device staging slot for the current
    // token id; a CudaVector here would cost a cudaMalloc/cudaFree pair on every
    // decode step.
    CUDA_CHECK(cudaMemcpy(d_next_token, &token_id, sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_X_accum, 0, m_config.hidden_dim * sizeof(float)));

    const void* d_embed_table = arena.get_weight_ptr("model.embed_tokens.weight");
    if (half_weights_are_fp16(m_config)) {
        launch_fp16_embedding_kernel(d_next_token, d_embed_table, d_X_accum, 1, m_config.hidden_dim);
    } else {
        launch_bf16_embedding_kernel(d_next_token, d_embed_table, d_X_accum, 1, m_config.hidden_dim);
    }
}

// ============================================================================
// STAGE 2: Granular Attention
// ============================================================================
void BlackwellEngine::Impl::step_attention_norm(int layer_idx) {
    // Offloaded layers: make the compute stream wait until the layer's weight
    // block is staged in VRAM (no-op for resident layers). Every step repeats
    // the call so the integration tests, which drive steps directly, stay safe.
    arena.ensure_layer_ready(layer_idx);
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";
    const void* d_w = arena.get_weight_ptr(prefix + "input_layernorm.weight");
    if (half_weights_are_fp16(m_config)) {
        launch_rmsnorm_fp16_kernel(d_X_accum, d_X_norm, d_w, 1, m_config.hidden_dim, m_config.rms_norm_eps);
    } else {
        launch_rmsnorm_kernel(d_X_accum, d_X_norm, d_w, 1, m_config.hidden_dim, m_config.rms_norm_eps);
    }
}

void BlackwellEngine::Impl::step_attention_qkv_projections(int layer_idx) {
    arena.ensure_layer_ready(layer_idx);
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".self_attn.";

    const size_t q_dim  = m_config.num_attention_heads * m_config.head_dim;
    const size_t kv_dim = m_config.num_key_value_heads * m_config.head_dim;

    // The QKV bias must land BEFORE step_attention_math: HF computes
    // (x @ W^T + b) and only then applies RoPE / appends to the KV cache.
    const void* d_bias_q = nullptr;
    const void* d_bias_k = nullptr;
    const void* d_bias_v = nullptr;
    if (m_config.has_qkv_bias) {
        d_bias_q = arena.get_weight_ptr_optional(prefix + "q_proj.bias");
        d_bias_k = arena.get_weight_ptr_optional(prefix + "k_proj.bias");
        d_bias_v = arena.get_weight_ptr_optional(prefix + "v_proj.bias");
        if ((d_bias_q != nullptr) != (d_bias_k != nullptr) ||
            (d_bias_q != nullptr) != (d_bias_v != nullptr))
            throw std::runtime_error(
                "BlackwellEngine: QKV bias tensors partially missing at layer " +
                std::to_string(layer_idx) + " (Qwen2 requires all three or none)");
    }

    if (m_config.quant_strategy == QuantStrategy::NONE) {
        // Unquantized fast path: fused BF16 GEMV + bias epilogue (bit-exact with
        // the plain GEMV when bias is nullptr).
        launch_bf16_gemv_bias_kernel(arena.get_weight_ptr(prefix + "q_proj.weight"),
                                     d_X_norm, d_bias_q, d_Q, q_dim, m_config.hidden_dim);
        launch_bf16_gemv_bias_kernel(arena.get_weight_ptr(prefix + "k_proj.weight"),
                                     d_X_norm, d_bias_k, d_K, kv_dim, m_config.hidden_dim);
        launch_bf16_gemv_bias_kernel(arena.get_weight_ptr(prefix + "v_proj.weight"),
                                     d_X_norm, d_bias_v, d_V, kv_dim, m_config.hidden_dim);
        return;
    }

    dispatcher.forward(prefix + "q_proj", d_X_norm, d_Q, q_dim,  m_config.hidden_dim);
    dispatcher.forward(prefix + "k_proj", d_X_norm, d_K, kv_dim, m_config.hidden_dim);
    dispatcher.forward(prefix + "v_proj", d_X_norm, d_V, kv_dim, m_config.hidden_dim);

    if (d_bias_q) {
        // AutoAWQ checkpoints keep biases in half; BF16/FP8 checkpoints in bf16.
        const BiasDType bias_dtype = (m_config.quant_strategy == QuantStrategy::WEIGHT_ONLY_PACKED)
                                         ? BiasDType::FP16
                                         : BiasDType::BF16;
        launch_fused_qkv_bias_kernel(d_Q, d_K, d_V, d_bias_q, d_bias_k, d_bias_v,
                                     q_dim, kv_dim, bias_dtype);
    }
}

void BlackwellEngine::Impl::step_attention_math(int layer_idx, int pos) {
    // Delegated to the active KV-cache strategy. The continuous adapter runs the
    // exact legacy sequence (prepare_layer_kv -> fused RoPE+append -> decode
    // attention -> commit_layer_kv); the paged adapter routes through the block
    // table + paged-flash kernel. The single virtual call sits at per-layer
    // granularity and is monomorphic, so it is free relative to the kernel
    // launches it wraps.
    kv_mgr->attention_decode(layer_idx, pos, d_Q, d_K, d_V, d_Attn_out);
}

void BlackwellEngine::Impl::step_attention_out(int layer_idx) {
    arena.ensure_layer_ready(layer_idx);
    std::string base = "model.layers." + std::to_string(layer_idx) + ".self_attn.o_proj";

    dispatcher.forward(base, d_Attn_out, nullptr,
                       m_config.hidden_dim, m_config.hidden_dim, d_X_accum);
}

// ============================================================================
// STAGE 3: Granular MLP
// ============================================================================
void BlackwellEngine::Impl::step_mlp_norm(int layer_idx) {
    arena.ensure_layer_ready(layer_idx);
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";
    const void* d_w = arena.get_weight_ptr(prefix + "post_attention_layernorm.weight");
    if (half_weights_are_fp16(m_config)) {
        launch_rmsnorm_fp16_kernel(d_X_accum, d_X_norm, d_w, 1, m_config.hidden_dim, m_config.rms_norm_eps);
    } else {
        launch_rmsnorm_kernel(d_X_accum, d_X_norm, d_w, 1, m_config.hidden_dim, m_config.rms_norm_eps);
    }
}

void BlackwellEngine::Impl::step_mlp_projections(int layer_idx) {
    arena.ensure_layer_ready(layer_idx);
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".mlp.";

    dispatcher.forward(prefix + "gate_proj", d_X_norm, d_Gate,
                       m_config.intermediate_dim, m_config.hidden_dim);
    dispatcher.forward(prefix + "up_proj",   d_X_norm, d_Up,
                       m_config.intermediate_dim, m_config.hidden_dim);
}

void BlackwellEngine::Impl::step_mlp_out(int layer_idx) {
    arena.ensure_layer_ready(layer_idx);
    std::string base = "model.layers." + std::to_string(layer_idx) + ".mlp.down_proj";

    launch_fused_swiglu_kernel(d_Gate, d_Up, d_Swiglu_out, m_config.intermediate_dim);

    dispatcher.forward(base, d_Swiglu_out, nullptr,
                       m_config.hidden_dim, m_config.intermediate_dim, d_X_accum);
}

// ============================================================================
// STAGE 4: Final Operations
// ============================================================================
void BlackwellEngine::Impl::step_final_ops() {
    const void* d_w = arena.get_weight_ptr("model.norm.weight");
    const bool fp16_w = half_weights_are_fp16(m_config);
    if (fp16_w) {
        launch_rmsnorm_fp16_kernel(d_X_accum, d_X_norm, d_w, 1, m_config.hidden_dim, m_config.rms_norm_eps);
    } else {
        launch_rmsnorm_kernel(d_X_accum, d_X_norm, d_w, 1, m_config.hidden_dim, m_config.rms_norm_eps);
    }

    // Tied checkpoints omit lm_head.weight entirely; the output head shares the
    // embedding matrix (both are [vocab_size, hidden_dim], so the GEMV row-major
    // W @ x contract holds unchanged).
    const void* d_head_w = m_config.tie_word_embeddings
        ? arena.get_weight_ptr("model.embed_tokens.weight")
        : arena.get_weight_ptr("lm_head.weight");
    if (fp16_w) {
        launch_fp16_gemv_kernel(d_head_w, d_X_norm, d_logits, m_config.vocab_size, m_config.hidden_dim);
    } else {
        launch_bf16_gemv_kernel(d_head_w, d_X_norm, d_logits, m_config.vocab_size, m_config.hidden_dim);
    }
}

// Реализация фасада BlackwellEngine
BlackwellEngine::BlackwellEngine(const std::string& index_path, size_t max_seq_len, size_t num_gpu_layers,
                                 KVCacheMode kv_mode)
    : pImpl(std::make_unique<Impl>(index_path, max_seq_len, num_gpu_layers, kv_mode)) {}

BlackwellEngine::~BlackwellEngine() = default;

// Sequence branching delegates to the active KV-cache strategy. Continuous
// throws (CoW unsupported); Paged shares / rolls back KV pages.
void BlackwellEngine::fork(int parent_id, int child_id) {
    pImpl->kv_mgr->fork(parent_id, child_id);
}

void BlackwellEngine::rewind(int seq_id, int pos) {
    pImpl->kv_mgr->rewind(seq_id, pos);
}

// ============================================================================
// Shared decoder pipeline: embedding -> N transformer layers -> final norm/head.
// Leaves the logits for token `pos` in impl->d_logits.
// ============================================================================
static void run_decoder_stack(BlackwellEngine::Impl* impl, int token_id, int pos, int seq_id) {
    const size_t max_seq_len = impl->arena.get_max_seq_len();
    if (pos < 0 || static_cast<size_t>(pos) >= max_seq_len)
        throw std::out_of_range(
            "BlackwellEngine: pos " + std::to_string(pos) +
            " exceeds KV cache capacity " + std::to_string(max_seq_len) +
            " (the RoPE/KV append kernel would write out of bounds)");

    impl->step_embedding(token_id);

    // Per-token KV control plane: latch the target sequence before the layer
    // sweep. The paged manager resolves the CoW append slot, stages this
    // sequence's block table, and (groundwork) ensures it is GPU-resident -- so
    // decoding any forked branch is just forward(..., seq_id). The continuous
    // manager accepts only seq_id 0.
    impl->kv_mgr->prepare_decode_step(seq_id, pos);

    const int num_layers = static_cast<int>(impl->m_config.num_layers);
    for (int i = 0; i < num_layers; ++i) {
        // Async pipeline: enqueue layer i+1's weight block and KV prefix on the
        // transfer stream NOW, so PCIe traffic overlaps layer i's kernels on
        // the compute stream (no-op when i+1 is VRAM-resident).
        impl->arena.prefetch_layer(i + 1, pos);

        impl->step_attention_norm(i);
        impl->step_attention_qkv_projections(i);
        impl->step_attention_math(i, pos);
        impl->step_attention_out(i);

        impl->step_mlp_norm(i);
        impl->step_mlp_projections(i);
        impl->step_mlp_out(i);
    }

    impl->step_final_ops();
}

// ============================================================================
// Full Engine Inference
// ============================================================================
int BlackwellEngine::forward(int token_id, int pos, float temperature, float top_p, int seq_id) {
    auto* impl = pImpl.get();
    run_decoder_stack(impl, token_id, pos, seq_id);
    return sample_top_p(impl->d_logits, impl->m_config.vocab_size, temperature, top_p);
}

// ============================================================================
// Evaluation Inference (Для расчета Перплексии)
// ============================================================================
float BlackwellEngine::forward_eval(int token_id, int pos, int target_token_id, int seq_id) {
    auto* impl = pImpl.get();
    run_decoder_stack(impl, token_id, pos, seq_id);
    return compute_log_prob(impl->d_logits, impl->m_config.vocab_size, target_token_id);
}