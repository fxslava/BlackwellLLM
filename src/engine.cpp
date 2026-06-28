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
#include "kernels/full_attention.cuh"
#include "kernels/swiglu.cuh"
#include "kernels/sampling.cuh"
#include "kernels/ssm_kernels.cuh"
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
        // Match the weight-offload split: layers [num_gpu_layers, num_layers) keep
        // their paged KV in the pinned host mirror, not VRAM. Without this the
        // paged pool sizes for ALL layers and silently maxes VRAM despite the
        // arena reporting offloading active.
        kv_mgr = std::make_unique<blackwell::PagedKVManager>(m_config, max_seq_len,
                                                             arena.num_gpu_layers());
    } else {
        kv_mgr = std::make_unique<blackwell::ContinuousKVManager>(arena, m_config);
    }

    // Derive model capabilities from the parsed layer topology. An empty
    // layer_types is the legacy uniform-Full layout (every layer softmax attn).
    int linear = 0, full = 0;
    for (AttnKind k : m_config.layer_types)
        (k == AttnKind::Linear ? linear : full)++;
    if (m_config.layer_types.empty())
        full = static_cast<int>(m_config.num_layers);

    m_caps.num_linear_attention_layers = linear;
    m_caps.num_full_attention_layers   = full;
    m_caps.requires_ssm_subsystem      = linear > 0;
    m_caps.is_hybrid                   = linear > 0 && full > 0;
    // CoW branching needs BOTH a branching-capable cache (paged) AND the absence
    // of a recurrent SSM state we cannot snapshot. Linear layers veto branching.
    m_caps.supports_cow_branching      = kv_mgr->supports_branching() && linear == 0;

    // Map absolute layer -> linear ordinal (-1 for full-attention layers), and
    // allocate the SSM recurrent state for hybrid models. No CoW => a single
    // active sequence (branching is vetoed above), so the pool holds one slot.
    m_linear_layer_index.assign(m_config.num_layers, -1);
    if (m_caps.requires_ssm_subsystem) {
        int ord = 0;
        for (size_t i = 0; i < m_config.layer_types.size(); ++i)
            if (m_config.layer_types[i] == AttnKind::Linear)
                m_linear_layer_index[i] = ord++;
        ssm_state = std::make_unique<blackwell::ssm::SsmStatePool>(
            blackwell::ssm::SsmGeometry::from_config(m_config), /*max_sequences=*/1);

        const auto& L = m_config.linear;
        const size_t conv_dim = 2 * L.num_key_heads * L.key_head_dim
                                  + L.num_value_heads * L.value_head_dim;
        const size_t v_dim    = L.num_value_heads * L.value_head_dim;   // H * Dv
        const size_t H        = L.num_value_heads;
        CUDA_CHECK(cudaMalloc(&d_ssm_qkv,      conv_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ssm_qkv_conv, conv_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ssm_z,        v_dim    * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ssm_q,        v_dim    * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ssm_k,        v_dim    * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ssm_v,        v_dim    * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ssm_core,     v_dim    * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ssm_o,        v_dim    * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ssm_a,        H        * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ssm_b,        H        * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dt_bias_f32,  H        * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_A_log_f32,    H        * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_norm_f32,     L.value_head_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_conv_w_f32,   conv_dim * L.conv_kernel_dim * sizeof(float)));
    }

    // Qwen3.5 hybrid gated full-attention layers (head_dim 256). These cannot use
    // the shared 128-wide attention/KV path, so map absolute layer -> full ordinal
    // and give them a dedicated continuous FP32 KV cache. attn_output_gate flags
    // this architecture; Qwen2.5/Llama keep it false and the generic path.
    m_full_layer_index.assign(m_config.num_layers, -1);
    if (m_config.attn_output_gate && m_caps.num_full_attention_layers > 0) {
        int ord = 0;
        for (size_t i = 0; i < m_config.num_layers; ++i) {
            const bool is_full = m_config.layer_types.empty() ||
                                 m_config.layer_types[i] == AttnKind::Full;
            if (is_full) m_full_layer_index[i] = ord++;
        }
        const size_t kv_dim = m_config.num_key_value_heads * m_config.head_dim;
        m_full_kv_layer_stride = kv_dim * arena.get_max_seq_len();
        const size_t total = (size_t)m_caps.num_full_attention_layers * m_full_kv_layer_stride;
        CUDA_CHECK(cudaMalloc(&d_QG,    m_config.num_attention_heads * m_config.head_dim * 2 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_gate,  m_config.num_attention_heads * m_config.head_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_full_k_cache, total * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_full_v_cache, total * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_full_k_cache, 0, total * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_full_v_cache, 0, total * sizeof(float)));
    }
}

BlackwellEngine::Impl::~Impl() {
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_Attn_out); cudaFree(d_Gate); cudaFree(d_Up); cudaFree(d_Swiglu_out);
    cudaFree(d_logits); cudaFree(d_next_token);
    cudaFree(d_ssm_qkv); cudaFree(d_ssm_z); cudaFree(d_ssm_qkv_conv);
    cudaFree(d_ssm_q); cudaFree(d_ssm_k); cudaFree(d_ssm_v);
    cudaFree(d_ssm_a); cudaFree(d_ssm_b); cudaFree(d_ssm_core); cudaFree(d_ssm_o);
    cudaFree(d_dt_bias_f32); cudaFree(d_A_log_f32); cudaFree(d_norm_f32); cudaFree(d_conv_w_f32);
    cudaFree(d_QG); cudaFree(d_gate); cudaFree(d_full_k_cache); cudaFree(d_full_v_cache);
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

    const void* d_embed_table = arena.get_weight_ptr(m_config.weight_prefix + "embed_tokens.weight");
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
    std::string prefix = m_config.weight_prefix + "layers." + std::to_string(layer_idx) + ".";
    const void* d_w = arena.get_weight_ptr(prefix + "input_layernorm.weight");
    if (half_weights_are_fp16(m_config)) {
        launch_rmsnorm_fp16_kernel(d_X_accum, d_X_norm, d_w, 1, m_config.hidden_dim, m_config.rms_norm_eps, m_config.norm_add_unit_offset);
    } else {
        launch_rmsnorm_kernel(d_X_accum, d_X_norm, d_w, 1, m_config.hidden_dim, m_config.rms_norm_eps, m_config.norm_add_unit_offset);
    }
}

void BlackwellEngine::Impl::step_attention_qkv_projections(int layer_idx) {
    arena.ensure_layer_ready(layer_idx);
    std::string prefix = m_config.weight_prefix + "layers." + std::to_string(layer_idx) + ".self_attn.";

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
    std::string base = m_config.weight_prefix + "layers." + std::to_string(layer_idx) + ".self_attn.o_proj";

    dispatcher.forward(base, d_Attn_out, nullptr,
                       m_config.hidden_dim, m_config.hidden_dim, d_X_accum);
}

// ============================================================================
// STAGE 2b: Linear-attention (SSM) layer — replaces softmax attention for the
// AttnKind::Linear layers of a hybrid model. Bypasses the KV cache entirely; the
// per-layer recurrent state lives in SsmStatePool and evolves in place.
// ============================================================================
void BlackwellEngine::Impl::step_linear_attention(int layer_idx, int pos) {
    arena.ensure_layer_ready(layer_idx);
    const int li = m_linear_layer_index[layer_idx];
    const std::string base = m_config.weight_prefix + "layers." + std::to_string(layer_idx) + ".";
    const std::string la   = base + "linear_attn.";

    // Per-(sequence, layer) recurrent state. Sequence 0 only: hybrid models forbid
    // forks (ModelCapabilities::supports_cow_branching == false).
    float* d_state = ssm_state->rec_state(0, li);
    float* d_conv  = ssm_state->conv_state(0, li);
    (void)pos;   // recurrence is position-implicit (state carries history)

    const auto& L = m_config.linear;
    const int Kh = (int)L.num_key_heads, Hh = (int)L.num_value_heads;
    const int Dk = (int)L.key_head_dim,  Dv = (int)L.value_head_dim, Kw = (int)L.conv_kernel_dim;
    const size_t conv_dim = 2 * (size_t)Kh * Dk + (size_t)Hh * Dv;     // q|k|v
    const size_t v_dim    = (size_t)Hh * Dv;                            // H * Dv

    // 1. input RMSNorm (bf16 weight; this checkpoint's norms are bf16) -> d_X_norm.
    launch_rmsnorm_kernel(d_X_accum, d_X_norm,
                          arena.get_weight_ptr(base + "input_layernorm.weight"),
                          1, m_config.hidden_dim, m_config.rms_norm_eps, m_config.norm_add_unit_offset);

    // 2. in-projections. qkv/z are symmetric int4 (dispatcher); a/b are bf16 GEMVs
    //    (per-head dt / beta sources). a feeds dt = softplus(a + dt_bias);
    //    b feeds beta = sigmoid(b).
    dispatcher.forward(la + "in_proj_qkv", d_X_norm, d_ssm_qkv, conv_dim, m_config.hidden_dim);
    dispatcher.forward(la + "in_proj_z",   d_X_norm, d_ssm_z,   v_dim,    m_config.hidden_dim);
    launch_bf16_gemv_kernel(arena.get_weight_ptr(la + "in_proj_a.weight"), d_X_norm, d_ssm_a, Hh, m_config.hidden_dim);
    launch_bf16_gemv_kernel(arena.get_weight_ptr(la + "in_proj_b.weight"), d_X_norm, d_ssm_b, Hh, m_config.hidden_dim);
    blackwell::ssm::launch_sigmoid_inplace(d_ssm_b, Hh);              // b -> beta

    // 3. causal depthwise conv1d (SiLU) over the full qkv channels, advancing this
    //    layer's ring buffer. conv1d.weight is bf16 [conv_dim,1,Kw] == [conv_dim,Kw].
    blackwell::ssm::launch_bf16_to_f32(arena.get_weight_ptr(la + "conv1d.weight"),
                                       d_conv_w_f32, (int)conv_dim * Kw);
    blackwell::ssm::launch_causal_conv1d_update(d_ssm_qkv, d_conv, d_conv_w_f32, /*bias=*/nullptr,
                                                d_ssm_qkv_conv, (int)conv_dim, Kw, /*silu=*/true);

    // 4. split conv'd qkv -> per-head q,k,v; L2-normalize q,k; GQA-broadcast Kh->Hh.
    blackwell::ssm::launch_ssm_split_norm_broadcast(d_ssm_qkv_conv, d_ssm_q, d_ssm_k, d_ssm_v,
                                                    Kh, Hh, Dk);

    // 5. GatedDeltaNet recurrent write/read into this layer's state.
    blackwell::ssm::launch_bf16_to_f32(arena.get_weight_ptr(la + "dt_bias"), d_dt_bias_f32, Hh);
    blackwell::ssm::launch_bf16_to_f32(arena.get_weight_ptr(la + "A_log"),   d_A_log_f32,   Hh);
    blackwell::ssm::launch_selective_scan_update(d_ssm_q, d_ssm_k, d_ssm_v, /*z=*/nullptr,
                                 d_ssm_a, d_dt_bias_f32, d_A_log_f32, d_ssm_b,
                                 d_state, d_ssm_core, Hh, Dk, Dv, /*gate_silu=*/false);

    // 6. per-head gated RMSNorm: o = rmsnorm(core * silu(z)) * norm.weight[Dv].
    blackwell::ssm::launch_bf16_to_f32(arena.get_weight_ptr(la + "norm.weight"), d_norm_f32, Dv);
    blackwell::ssm::launch_gated_rmsnorm_per_head(d_ssm_core, d_ssm_z, d_norm_f32, d_ssm_o,
                                                  Hh, Dv, m_config.rms_norm_eps);

    // 7. out_proj (symmetric int4), accumulating into the residual stream.
    dispatcher.forward(la + "out_proj", d_ssm_o, nullptr, m_config.hidden_dim, v_dim, d_X_accum);
}

// ============================================================================
// STAGE 2c: Qwen3.5 hybrid gated FULL-attention layer (head_dim 256). Periodic
// softmax-attention layers of the hybrid stack. The generic step_attention_* path
// cannot serve them: q_proj emits a per-head [query|gate] pair (2*head_dim) and
// head_dim 256 exceeds the shared attention/KV kernel's 128-wide block. This is an
// unoptimized "make it work" path over a dedicated continuous FP32 KV cache.
// ============================================================================
void BlackwellEngine::Impl::step_full_attention(int layer_idx, int pos) {
    arena.ensure_layer_ready(layer_idx);
    const std::string base = m_config.weight_prefix + "layers." + std::to_string(layer_idx) + ".";
    const std::string sa   = base + "self_attn.";

    const int Hq = (int)m_config.num_attention_heads;   // 16
    const int Hkv = (int)m_config.num_key_value_heads;  // 4
    const int Dh = (int)m_config.head_dim;              // 256
    const int rot = (int)m_config.rotary_dim;           // 64 (partial)
    const size_t q_dim  = (size_t)Hq * Dh;
    const size_t kv_dim = (size_t)Hkv * Dh;

    // 1. input RMSNorm (Gemma-style 1+weight, bf16 weight) -> d_X_norm.
    launch_rmsnorm_kernel(d_X_accum, d_X_norm,
                          arena.get_weight_ptr(base + "input_layernorm.weight"),
                          1, m_config.hidden_dim, m_config.rms_norm_eps, m_config.norm_add_unit_offset);

    // 2. projections. q_proj emits [Hq, 2*Dh] (query|gate); k/v emit [Hkv, Dh].
    dispatcher.forward(sa + "q_proj", d_X_norm, d_QG, q_dim * 2, m_config.hidden_dim);
    dispatcher.forward(sa + "k_proj", d_X_norm, d_K,  kv_dim,    m_config.hidden_dim);
    dispatcher.forward(sa + "v_proj", d_X_norm, d_V,  kv_dim,    m_config.hidden_dim);

    // 3. de-interleave the per-head query|gate pair.
    launch_qg_split(d_QG, d_Q, d_gate, Hq, Dh);

    // 4. per-head q_norm / k_norm (RMSNorm over head_dim, same 1+weight offset).
    launch_rmsnorm_kernel(d_Q, d_Q, arena.get_weight_ptr(sa + "q_norm.weight"),
                          Hq, Dh, m_config.rms_norm_eps, m_config.norm_add_unit_offset);
    launch_rmsnorm_kernel(d_K, d_K, arena.get_weight_ptr(sa + "k_norm.weight"),
                          Hkv, Dh, m_config.rms_norm_eps, m_config.norm_add_unit_offset);

    // 5. partial rotate_half RoPE on Q and K (identity at pos 0). NOTE: the config
    //    requests interleaved M-RoPE, which this 1D kernel does not implement, so
    //    positions >= 1 are NOT faithful (already warned at config load).
    launch_rope_partial_inplace(d_Q, pos, Hq,  Dh, rot, m_config.rope_theta);
    launch_rope_partial_inplace(d_K, pos, Hkv, Dh, rot, m_config.rope_theta);

    // 6. append K/V into this full-attn layer's dedicated cache, then decode.
    const int fo = m_full_layer_index[layer_idx];
    float* d_k_cache = d_full_k_cache + (size_t)fo * m_full_kv_layer_stride;
    float* d_v_cache = d_full_v_cache + (size_t)fo * m_full_kv_layer_stride;
    const int msl = (int)arena.get_max_seq_len();
    launch_kv_append(d_K, d_V, d_k_cache, d_v_cache, pos, Hkv, Dh, msl);
    launch_full_attention_decode(d_Q, d_k_cache, d_v_cache, d_Attn_out, pos,
                                 Hq, Hkv, Dh, msl);

    // 7. gate the context: attn_out *= sigmoid(gate), then o_proj into residual.
    launch_gate_sigmoid_mul(d_Attn_out, d_gate, (int)q_dim);
    dispatcher.forward(sa + "o_proj", d_Attn_out, nullptr, m_config.hidden_dim, q_dim, d_X_accum);
}

// ============================================================================
// STAGE 3: Granular MLP
// ============================================================================
void BlackwellEngine::Impl::step_mlp_norm(int layer_idx) {
    arena.ensure_layer_ready(layer_idx);
    std::string prefix = m_config.weight_prefix + "layers." + std::to_string(layer_idx) + ".";
    const void* d_w = arena.get_weight_ptr(prefix + "post_attention_layernorm.weight");
    if (half_weights_are_fp16(m_config)) {
        launch_rmsnorm_fp16_kernel(d_X_accum, d_X_norm, d_w, 1, m_config.hidden_dim, m_config.rms_norm_eps, m_config.norm_add_unit_offset);
    } else {
        launch_rmsnorm_kernel(d_X_accum, d_X_norm, d_w, 1, m_config.hidden_dim, m_config.rms_norm_eps, m_config.norm_add_unit_offset);
    }
}

void BlackwellEngine::Impl::step_mlp_projections(int layer_idx) {
    arena.ensure_layer_ready(layer_idx);
    std::string prefix = m_config.weight_prefix + "layers." + std::to_string(layer_idx) + ".mlp.";

    dispatcher.forward(prefix + "gate_proj", d_X_norm, d_Gate,
                       m_config.intermediate_dim, m_config.hidden_dim);
    dispatcher.forward(prefix + "up_proj",   d_X_norm, d_Up,
                       m_config.intermediate_dim, m_config.hidden_dim);
}

void BlackwellEngine::Impl::step_mlp_out(int layer_idx) {
    arena.ensure_layer_ready(layer_idx);
    std::string base = m_config.weight_prefix + "layers." + std::to_string(layer_idx) + ".mlp.down_proj";

    launch_fused_swiglu_kernel(d_Gate, d_Up, d_Swiglu_out, m_config.intermediate_dim);

    dispatcher.forward(base, d_Swiglu_out, nullptr,
                       m_config.hidden_dim, m_config.intermediate_dim, d_X_accum);
}

// ============================================================================
// STAGE 4: Final Operations
// ============================================================================
void BlackwellEngine::Impl::step_final_ops() {
    const void* d_w = arena.get_weight_ptr(m_config.weight_prefix + "norm.weight");
    const bool fp16_w = half_weights_are_fp16(m_config);
    if (fp16_w) {
        launch_rmsnorm_fp16_kernel(d_X_accum, d_X_norm, d_w, 1, m_config.hidden_dim, m_config.rms_norm_eps, m_config.norm_add_unit_offset);
    } else {
        launch_rmsnorm_kernel(d_X_accum, d_X_norm, d_w, 1, m_config.hidden_dim, m_config.rms_norm_eps, m_config.norm_add_unit_offset);
    }

    // Tied checkpoints omit lm_head.weight entirely; the output head shares the
    // embedding matrix (both are [vocab_size, hidden_dim], so the GEMV row-major
    // W @ x contract holds unchanged).
    const void* d_head_w = m_config.tie_word_embeddings
        ? arena.get_weight_ptr(m_config.weight_prefix + "embed_tokens.weight")
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

ModelCapabilities BlackwellEngine::get_capabilities() const {
    return pImpl->m_caps;
}

// Sequence branching delegates to the active KV-cache strategy, but only after a
// capability gate: hybrid SSM models (Qwen3.5) and Continuous mode cannot snapshot
// their state, so fork/rewind fail cleanly here instead of corrupting decode.
static void require_branching(const ModelCapabilities& caps, const char* op) {
    if (!caps.supports_cow_branching)
        throw std::runtime_error(
            std::string("BlackwellEngine::") + op + ": the loaded model does not support "
            "CoW branching (" +
            (caps.requires_ssm_subsystem
                 ? "hybrid linear-attention/SSM checkpoint — recurrent state is not "
                   "snapshot-able; run linear ReAct only"
                 : "Continuous KV mode — construct with KVCacheMode::Paged for branching") +
            ").");
}

void BlackwellEngine::fork(int parent_id, int child_id) {
    require_branching(pImpl->m_caps, "fork");
    pImpl->kv_mgr->fork(parent_id, child_id);
}

void BlackwellEngine::rewind(int seq_id, int pos) {
    require_branching(pImpl->m_caps, "rewind");
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

        // Hybrid dispatch: an empty layer_types means the legacy uniform-Full
        // layout, so this is a plain `if` per layer with no effect on existing
        // models. Linear layers bypass the KV cache and run the SSM path.
        const bool is_linear =
            !impl->m_config.layer_types.empty() &&
            impl->m_config.layer_types[i] == AttnKind::Linear;

        if (is_linear) {
            impl->step_linear_attention(i, pos);
        } else if (impl->m_config.attn_output_gate) {
            // Qwen3.5 hybrid: gated head_dim-256 full-attention (dedicated path).
            impl->step_full_attention(i, pos);
        } else {
            impl->step_attention_norm(i);
            impl->step_attention_qkv_projections(i);
            impl->step_attention_math(i, pos);
            impl->step_attention_out(i);
        }

        // The MLP block is identical for both layer kinds.
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