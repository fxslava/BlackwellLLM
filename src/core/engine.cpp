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
#include "rope_config.h"                   // rope_scaling_from(ModelConfig)
#include "paging/cuda_tier_backend.h"      // SmVramPool, CudaTierBackend
#include "paging/prefix_cache_manager.h"   // TieredMemoryPager, PrefixCacheManager
#include "engine_prefill_coordinator.h"
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>

using blackwell::EngineStatus;

// RUNTIME-tier CUDA check (Hybrid error doctrine, Roadmap #2): the decode hot
// loop never throws -- a CUDA failure is logged once and becomes an
// EngineStatus that ripples up by return value (ENGINE_TRY), translated to an
// HRESULT at the COM boundary or to an engine_error by the facade wrappers.
// INIT paths (ctor) use CUDA_CHECK_THROW from common.h instead.
#define CUDA_CHECK_RETURN(call)                                                    \
    do {                                                                           \
        cudaError_t err__ = (call);                                                \
        if (err__ != cudaSuccess) {                                                \
            std::cerr << "[blackwell_core] CUDA error: " << cudaGetErrorString(err__) \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";            \
            return (err__ == cudaErrorMemoryAllocation)                            \
                       ? EngineStatus::OutOfVram                                   \
                       : EngineStatus::CudaRuntimeError;                           \
        }                                                                          \
    } while (0)

// Propagate a non-Success status up the runtime call chain (zero-cost on the
// happy path; one predictable branch per step).
#define ENGINE_TRY(expr)                                                \
    do {                                                                \
        const EngineStatus st__ = (expr);                               \
        if (st__ != EngineStatus::Success) return st__;                 \
    } while (0)

// Stable identity of (checkpoint x KV geometry) for the prefix cache and the
// .bkv serializer: a page written under one hash must never be grafted into an
// engine whose pages mean something else. FNV-1a over the index path plus the
// geometry fields that define a page record's byte layout. (A content digest of
// the weights would be stronger; the path+geometry pair matches what the
// warmup manifest / spill records actually need to agree on.)
static uint64_t compute_kv_model_hash(const std::string& index_path, const ModelConfig& c) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](uint64_t v) { h = (h ^ v) * 1099511628211ull; };
    for (char ch : index_path) mix(static_cast<unsigned char>(ch));
    mix(c.num_layers);
    mix(c.num_key_value_heads);
    mix(c.head_dim);
    mix(c.hidden_dim);
    mix(c.vocab_size);
    return h;
}

// Batched-forward width from the resolved plan. Batched prefill (paged + dense,
// where the coordinator sweeps prompt deltas) processes up to kPrefillTile tokens
// per chunk; true-batch decode needs max_sequences rows. The tile is bounded so
// the widened activation/scratch buffers stay small — they scale linearly with
// it. Batch=1 continuous decode (no coordinator, single sequence) stays at 1 so
// nothing is over-allocated and the decode path is byte-for-byte as before.
// Hybrid SSM models can never batch (recurrent state), so they stay at 1 too.
static size_t resolve_token_capacity(const ModelConfig& config,
                                     const ModelCapabilities& caps,
                                     const blackwell::RuntimeConfig& rt) {
    constexpr size_t kPrefillTile = 64;
    if (caps.requires_ssm_subsystem) {
        // Hybrid True Batched Prefill (chunked delta rule) widens the residual
        // stream to a token tile the same way the dense path does, but only under
        // Paged mode (the branching-capable configuration prefill/fork target).
        // The chunk kernel's own tile is 64, so the tile matches it. Continuous /
        // batch=1 hybrid decode keeps width 1 (no batched buffers allocated).
        if (rt.kv_mode == BlackwellEngine::KVCacheMode::Paged)
            return std::max<size_t>(1, std::min(rt.max_seq_len, kPrefillTile));
        return 1;
    }
    const bool paged_dense = rt.kv_mode == BlackwellEngine::KVCacheMode::Paged &&
                             !config.attn_output_gate;
    const bool will_batch = rt.max_sequences > 1 || paged_dense;
    if (!will_batch) return 1;
    return std::max(rt.max_sequences, std::min(rt.max_seq_len, kPrefillTile));
}

// Initializer list mirrors the declaration order in engine_impl.h: members are
// constructed in declaration order regardless of the list. The tier-3 RuntimeConfig
// (m_runtime) is built and validated FIRST so `arena` and the KV manager can size
// themselves from a single resolved plan instead of loose ad-hoc args.
BlackwellEngine::Impl::Impl(const std::string& index_path, const blackwell::InferenceConfig& request,
                            const blackwell::RuntimeOverrides& overrides)
    : m_config(ConfigLoader::load_from_json(
          (std::filesystem::path(index_path).parent_path() / "config.json").string())),
      m_caps(blackwell::derive_capabilities(m_config)),
      m_runtime(blackwell::build_and_validate_runtime(m_config, m_caps, request, overrides)),
      m_token_capacity(resolve_token_capacity(m_config, m_caps, m_runtime)),
      loader(index_path),
      arena(index_path, loader, m_config, m_runtime.max_seq_len, m_runtime.num_gpu_layers,
            overrides.load_progress, m_token_capacity),
      dispatcher(arena, m_config, m_runtime.batched_gemm_threshold)
{
    // 1. Bind core activation buffers from the arena
    d_X_accum = arena.get_activation_buffer_A();
    d_X_norm  = arena.get_activation_buffer_B();

    // 2. Allocate layer-scoped compute buffers once (RAII: freed by the members
    // themselves, in reverse declaration order -- no hand-maintained list). Each
    // holds one row per active token: width == m_token_capacity * per-row size, so
    // the batched-prefill / true-batch path can stage [num_tokens, *] contiguously.
    // m_token_capacity == 1 reproduces the exact batch=1 decode allocation.
    const size_t tc = m_token_capacity;
    d_Q.allocate(tc * m_config.num_attention_heads * m_config.head_dim);
    d_K.allocate(tc * m_config.num_key_value_heads * m_config.head_dim);
    d_V.allocate(tc * m_config.num_key_value_heads * m_config.head_dim);
    d_Attn_out.allocate(tc * m_config.hidden_dim);

    // GLM-4 ships gate and up as one tensor, so the activation buffer is one
    // [tc, 2*I] block and the separate halves are never materialized. Exactly one
    // of the two layouts is allocated -- see the members' declaration comments.
    if (m_config.mlp_fused_gate_up) {
        d_GateUp.allocate(tc * 2 * m_config.intermediate_dim);
    } else {
        d_Gate.allocate(tc * m_config.intermediate_dim);
        d_Up.allocate(tc * m_config.intermediate_dim);
    }
    d_Swiglu_out.allocate(tc * m_config.intermediate_dim);

    // Sandwich norms need a landing buffer for the un-normalized sub-layer output
    // (see the member declaration); one hidden-width row per active token.
    if (m_config.has_sandwich_norms)
        d_sublayer_out.allocate(tc * m_config.hidden_dim);

    // One logits row per active token: batched decode leaves [batch, vocab] here
    // and samples each row; every single-row path (run_token, run_chunk,
    // last_token_probability) uses row 0, so widening is transparent to them.
    d_logits.allocate(tc * m_config.vocab_size);
    d_next_token.allocate(tc);

    // Resident pool may be empty when every layer is offloaded to host RAM.
    // INIT tier: a failure here throws and unwinds the RAII members above.
    if (arena.get_k_cache_size() > 0) {
        CUDA_CHECK_THROW(cudaMemset(arena.get_k_cache(), 0, arena.get_k_cache_size()));
        CUDA_CHECK_THROW(cudaMemset(arena.get_v_cache(), 0, arena.get_v_cache_size()));
    }

    // KV-cache strategy from the resolved plan. Continuous (default) wraps the
    // existing VRAMArena flow (behavior-preserving). Paged uses the bf16 CoW cache
    // + paged-flash kernel, with its host-mirror headroom from m_runtime.
    if (m_runtime.kv_mode == BlackwellEngine::KVCacheMode::Paged) {
        // Match the weight-offload split: layers [num_gpu_layers, num_layers) keep
        // their paged KV in the pinned host mirror, not VRAM. Without this the
        // paged pool sizes for ALL layers and silently maxes VRAM despite the
        // arena reporting offloading active.
        kv_mgr = std::make_unique<blackwell::PagedKVManager>(m_config, m_runtime.max_seq_len,
                                                             arena.num_gpu_layers(),
                                                             m_runtime.paged_branch_factor,
                                                             m_runtime.kv_vram_cache_pages);
    } else {
        kv_mgr = std::make_unique<blackwell::ContinuousKVManager>(
            arena, m_config, m_runtime.attention_split_k_max);
    }

    // Finalize the one capability the topology alone could not decide: CoW
    // branching needs a branching-capable cache (paged). Hybrid SSM models now
    // qualify too -- fork() PHYSICALLY snapshots the recurrent state
    // (SsmStatePool::fork_sequence) and the dedicated gated full-attention KV
    // cache, so the old "dense only" restriction is lifted. Continuous mode still
    // has no fork. derive_capabilities() filled the rest.
    m_caps.supports_cow_branching = kv_mgr->supports_branching();

    // Concurrent-branch capacity for the physically-snapshotted hybrid state
    // stores (the SSM recurrent/conv pool + the gated full-attention cache).
    // Only branching-capable (paged) engines pay the multi-sequence VRAM; it
    // mirrors the paged pool's own CoW branch headroom so the paged KV branches
    // and the snapshotted stores share one budget. Continuous stays single-seq.
    m_branch_capacity = m_caps.supports_cow_branching
                            ? std::max(1, m_runtime.paged_branch_factor) : 1;

    // ---- Prefix-cache substrate composition root (Phase 3 wiring) ----------
    // SmVramPool -> CudaTierBackend -> TieredMemoryPager -> PrefixCacheManager
    // -> EnginePrefillCoordinator, all over the PagedKVManager's OWN
    // SequenceManager, so the pages the radix tree indexes are the very pages
    // the paged-flash attention kernels read and write (zero-copy reuse).
    // Restricted to dense uniform full-attention models: hybrid SSM state and
    // the dedicated gated full-attention cache live outside the paged pools,
    // so a radix "prefix hit" would silently skip state those models need.
    if (m_runtime.kv_mode == BlackwellEngine::KVCacheMode::Paged &&
        !m_caps.requires_ssm_subsystem && !m_config.attn_output_gate) {
        auto& pkm = static_cast<blackwell::PagedKVManager&>(*kv_mgr);
        auto& sm  = pkm.sequence_manager();

        kv_vram_pool = std::make_unique<blackwell::paging::SmVramPool>(sm);

        // Tier sizing from the resolved RuntimeConfig (docs/TIERED_KV_AND_AOT.md
        // §5.1 knobs). kMirrorDevicePool is the one late-bound value: it becomes
        // the actual device-pool page count, known only now.
        blackwell::paging::TieredMemoryPager::Config pager_cfg;
        pager_cfg.ram_slots =
            (m_runtime.kv_ram_slots == blackwell::RuntimeConfig::kMirrorDevicePool)
                ? sm.total_pages()
                : m_runtime.kv_ram_slots;
        pager_cfg.disk_slots = m_runtime.kv_disk_slots;

        kv_tier_backend = std::make_unique<blackwell::paging::CudaTierBackend>(
            sm, pager_cfg.ram_slots, pager_cfg.disk_slots, m_runtime.kv_spill_path);
        kv_pager = std::make_unique<blackwell::paging::TieredMemoryPager>(
            *kv_vram_pool, *kv_tier_backend, pager_cfg);
        prefix_cache = std::make_unique<blackwell::paging::PrefixCacheManager>(
            sm, *kv_pager, compute_kv_model_hash(index_path, m_config));
        prefill = std::make_unique<blackwell::EnginePrefillCoordinator>(
            *this, pkm, *prefix_cache);
    }

    // Map absolute layer -> linear ordinal (-1 for full-attention layers), and
    // allocate the SSM recurrent state for hybrid models. The pool holds
    // m_branch_capacity sequence slots so fork() can physically snapshot the
    // parent's recurrent state into a sibling branch (1 slot when branching is off).
    m_linear_layer_index.assign(m_config.num_layers, -1);
    if (m_caps.requires_ssm_subsystem) {
        int ord = 0;
        for (size_t i = 0; i < m_config.layer_types.size(); ++i)
            if (m_config.layer_types[i] == AttnKind::Linear)
                m_linear_layer_index[i] = ord++;
        ssm_state = std::make_unique<blackwell::ssm::SsmStatePool>(
            blackwell::ssm::SsmGeometry::from_config(m_config),
            /*max_sequences=*/m_branch_capacity);

        const auto& L = m_config.linear;
        const size_t conv_dim = 2 * L.num_key_heads * L.key_head_dim
                                  + L.num_value_heads * L.value_head_dim;
        const size_t v_dim    = L.num_value_heads * L.value_head_dim;   // H * Dv
        const size_t H        = L.num_value_heads;
        d_ssm_qkv.allocate(conv_dim);
        d_ssm_qkv_conv.allocate(conv_dim);
        d_ssm_z.allocate(v_dim);
        d_ssm_q.allocate(v_dim);
        d_ssm_k.allocate(v_dim);
        d_ssm_v.allocate(v_dim);
        d_ssm_core.allocate(v_dim);
        d_ssm_o.allocate(v_dim);
        d_ssm_a.allocate(H);
        d_ssm_b.allocate(H);
        d_dt_bias_f32.allocate(H);
        d_A_log_f32.allocate(H);
        d_norm_f32.allocate(L.value_head_dim);
        d_conv_w_f32.allocate(conv_dim * L.conv_kernel_dim);

        // Chunked-prefill staging. Sized to the token tile (m_token_capacity) so
        // run_chunk_hybrid holds a whole chunk: the batched in_proj outputs
        // (qkv/z), the token-major q/k/v/o chunk buffers the delta kernel reads,
        // the per-(token,head) decay/write scalars, and the per-head UT-solve
        // scratch (64 == the kernel's intra-chunk tile). Width 1 when batching is
        // off, so this is a negligible allocation for the batch=1 decode build.
        constexpr size_t kDeltaChunk = 64;
        const size_t kdim = L.key_head_dim, vdim = L.value_head_dim;
        d_ssm_qkv_batch.allocate(tc * conv_dim);
        d_ssm_z_batch.allocate(tc * v_dim);
        d_ssm_o_batch.allocate(tc * v_dim);
        d_ssm_chunk_q.allocate(tc * H * kdim);
        d_ssm_chunk_k.allocate(tc * H * kdim);
        d_ssm_chunk_v.allocate(tc * H * vdim);
        d_ssm_chunk_o.allocate(tc * H * vdim);
        d_ssm_chunk_logdecay.allocate(tc * H);
        d_ssm_chunk_beta.allocate(tc * H);
        d_ssm_u_scratch.allocate(H * kDeltaChunk * vdim);
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
        // Per-sequence slice spans all full-attn layers; the cache is laid out
        // [seq][full-layer][pos][kv] so a hybrid fork D2D-copies one seq stride
        // and step_full_attention offsets by seq_id * m_full_kv_seq_stride.
        m_full_kv_seq_stride = (size_t)m_caps.num_full_attention_layers * m_full_kv_layer_stride;
        const size_t total = (size_t)m_branch_capacity * m_full_kv_seq_stride;
        // d_QG holds the [num_heads, 2*head_dim] q_proj output; widened to the
        // token tile so run_chunk_hybrid stages the whole chunk's query|gate pair
        // batched (per-token qg_split then reads row t). d_gate stays single-row —
        // the full-attn chunk sweep gates one token at a time.
        d_QG.allocate(m_token_capacity * m_config.num_attention_heads * m_config.head_dim * 2);
        d_gate.allocate(m_config.num_attention_heads * m_config.head_dim);
        d_full_k_cache.allocate(total);
        d_full_v_cache.allocate(total);
        // The KV cache is read at positions beyond what has been written yet
        // (masked lanes) -- it must start zeroed, unlike the per-step scratch.
        d_full_k_cache.zero();
        d_full_v_cache.zero();
    }
}

// Every owned device buffer is a DeviceBuffer member: they free themselves in
// reverse declaration order. d_X_accum / d_X_norm are non-owning arena views.
BlackwellEngine::Impl::~Impl() = default;

// ============================================================================
// STAGE 1: Embedding
// ============================================================================
// AWQ/GPTQ checkpoints store every non-quantized tensor (embeddings, norm
// weights, lm_head, biases) in FP16; BF16/FP8 checkpoints use bfloat16. The
// same heuristic already routes the QKV bias dtype below.
static bool half_weights_are_fp16(const ModelConfig& cfg) {
    return cfg.quant_strategy == QuantStrategy::WEIGHT_ONLY_PACKED;
}

EngineStatus BlackwellEngine::Impl::step_embedding(int token_id) {
    // d_next_token doubles as the persistent device staging slot for the current
    // token id; a CudaVector here would cost a cudaMalloc/cudaFree pair on every
    // decode step.
    CUDA_CHECK_RETURN(cudaMemcpy(d_next_token, &token_id, sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK_RETURN(cudaMemset(d_X_accum, 0, m_config.hidden_dim * sizeof(float)));

    const void* d_embed_table = arena.get_weight_ptr(m_config.weight_prefix + "embed_tokens.weight");
    if (half_weights_are_fp16(m_config)) {
        launch_fp16_embedding_kernel(d_next_token, d_embed_table, d_X_accum, 1, m_config.hidden_dim);
    } else {
        launch_bf16_embedding_kernel(d_next_token, d_embed_table, d_X_accum, 1, m_config.hidden_dim);
    }
    return EngineStatus::Success;
}

// ============================================================================
// STAGE 2: Granular Attention
// ============================================================================
EngineStatus BlackwellEngine::Impl::step_attention_norm(int layer_idx) {
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
    return EngineStatus::Success;
}

EngineStatus BlackwellEngine::Impl::step_attention_qkv_projections(int layer_idx) {
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
            (d_bias_q != nullptr) != (d_bias_v != nullptr)) {
            // Checkpoint-integrity violation discovered mid-decode: runtime
            // tier, so log the detail and report by status, not by throw.
            std::cerr << "[blackwell_core] QKV bias tensors partially missing at layer "
                      << layer_idx << " (Qwen2 requires all three or none)\n";
            return EngineStatus::InvalidConfig;
        }
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
        return EngineStatus::Success;
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
    return EngineStatus::Success;
}

EngineStatus BlackwellEngine::Impl::step_attention_math(int layer_idx, int pos) {
    // Delegated to the active KV-cache strategy. The continuous adapter runs the
    // exact legacy sequence (prepare_layer_kv -> fused RoPE+append -> decode
    // attention -> commit_layer_kv); the paged adapter routes through the block
    // table + paged-flash kernel. The single virtual call sits at per-layer
    // granularity and is monomorphic, so it is free relative to the kernel
    // launches it wraps.
    kv_mgr->attention_decode(layer_idx, pos, d_Q, d_K, d_V, d_Attn_out);
    return EngineStatus::Success;
}

// GLM-4-0414 sandwich norm: d_X_accum += post_*_layernorm(d_sublayer_out).
void BlackwellEngine::Impl::sandwich_norm_accum(const std::string& weight_name,
                                                size_t num_tokens) {
    const void* d_w = arena.get_weight_ptr(weight_name);
    if (half_weights_are_fp16(m_config))
        launch_rmsnorm_accum_fp16_kernel(d_sublayer_out, d_X_accum, d_w, num_tokens,
                                         m_config.hidden_dim, m_config.rms_norm_eps,
                                         m_config.norm_add_unit_offset);
    else
        launch_rmsnorm_accum_kernel(d_sublayer_out, d_X_accum, d_w, num_tokens,
                                    m_config.hidden_dim, m_config.rms_norm_eps,
                                    m_config.norm_add_unit_offset);
}

EngineStatus BlackwellEngine::Impl::step_attention_out(int layer_idx) {
    arena.ensure_layer_ready(layer_idx);
    const std::string prefix = m_config.weight_prefix + "layers." + std::to_string(layer_idx) + ".";
    const std::string base = prefix + "self_attn.o_proj";

    if (m_config.has_sandwich_norms) {
        // h = x + post_self_attn_layernorm(o_proj(context)): the norm sits BETWEEN
        // the projection and the residual add, so the projection cannot fuse the
        // accumulate -- it would add the un-normalized output.
        dispatcher.forward(base, d_Attn_out, d_sublayer_out,
                           m_config.hidden_dim, m_config.hidden_dim, nullptr);
        sandwich_norm_accum(prefix + "post_self_attn_layernorm.weight", 1);
        return EngineStatus::Success;
    }

    dispatcher.forward(base, d_Attn_out, nullptr,
                       m_config.hidden_dim, m_config.hidden_dim, d_X_accum);
    return EngineStatus::Success;
}

// ============================================================================
// STAGE 2b: Linear-attention (SSM) layer — replaces softmax attention for the
// AttnKind::Linear layers of a hybrid model. Bypasses the KV cache entirely; the
// per-layer recurrent state lives in SsmStatePool and evolves in place.
// ============================================================================
EngineStatus BlackwellEngine::Impl::step_linear_attention(int layer_idx, int pos, int seq_id) {
    arena.ensure_layer_ready(layer_idx);
    const int li = m_linear_layer_index[layer_idx];
    const std::string base = m_config.weight_prefix + "layers." + std::to_string(layer_idx) + ".";
    const std::string la   = base + "linear_attn.";

    // Per-(sequence, layer) recurrent state. seq_id selects this branch's slice:
    // a forked child owns an independent physical copy (SsmStatePool::fork_sequence),
    // so decoding it never disturbs the parent's recurrent state.
    float* d_state = ssm_state->rec_state(seq_id, li);
    float* d_conv  = ssm_state->conv_state(seq_id, li);
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
    return EngineStatus::Success;
}

// ============================================================================
// STAGE 2c: Qwen3.5 hybrid gated FULL-attention layer (head_dim 256). Periodic
// softmax-attention layers of the hybrid stack. The generic step_attention_* path
// cannot serve them: q_proj emits a per-head [query|gate] pair (2*head_dim) and
// head_dim 256 exceeds the shared attention/KV kernel's 128-wide block. This is an
// unoptimized "make it work" path over a dedicated continuous FP32 KV cache.
// ============================================================================
EngineStatus BlackwellEngine::Impl::step_full_attention(int layer_idx, int pos, int seq_id) {
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
    launch_rope_partial_inplace(d_Q, pos, Hq,  Dh, rot, m_config.rope_theta, rope_scaling_from(m_config));
    launch_rope_partial_inplace(d_K, pos, Hkv, Dh, rot, m_config.rope_theta, rope_scaling_from(m_config));

    // 6. append K/V into this full-attn layer's dedicated cache, then decode.
    // Offset by this sequence's per-branch slice so forked branches keep
    // independent full-attention histories (the dedicated cache is NOT the paged
    // pool, so fork() snapshots it separately; see Impl::fork).
    const int fo = m_full_layer_index[layer_idx];
    const size_t seq_off = (size_t)seq_id * m_full_kv_seq_stride;
    float* d_k_cache = d_full_k_cache + seq_off + (size_t)fo * m_full_kv_layer_stride;
    float* d_v_cache = d_full_v_cache + seq_off + (size_t)fo * m_full_kv_layer_stride;
    const int msl = (int)arena.get_max_seq_len();
    launch_kv_append(d_K, d_V, d_k_cache, d_v_cache, pos, Hkv, Dh, msl);
    launch_full_attention_decode(d_Q, d_k_cache, d_v_cache, d_Attn_out, pos,
                                 Hq, Hkv, Dh, msl);

    // 7. gate the context: attn_out *= sigmoid(gate), then o_proj into residual.
    launch_gate_sigmoid_mul(d_Attn_out, d_gate, (int)q_dim);
    dispatcher.forward(sa + "o_proj", d_Attn_out, nullptr, m_config.hidden_dim, q_dim, d_X_accum);
    return EngineStatus::Success;
}

// ============================================================================
// STAGE 3: Granular MLP
// ============================================================================
EngineStatus BlackwellEngine::Impl::step_mlp_norm(int layer_idx) {
    arena.ensure_layer_ready(layer_idx);
    std::string prefix = m_config.weight_prefix + "layers." + std::to_string(layer_idx) + ".";
    const void* d_w = arena.get_weight_ptr(prefix + "post_attention_layernorm.weight");
    if (half_weights_are_fp16(m_config)) {
        launch_rmsnorm_fp16_kernel(d_X_accum, d_X_norm, d_w, 1, m_config.hidden_dim, m_config.rms_norm_eps, m_config.norm_add_unit_offset);
    } else {
        launch_rmsnorm_kernel(d_X_accum, d_X_norm, d_w, 1, m_config.hidden_dim, m_config.rms_norm_eps, m_config.norm_add_unit_offset);
    }
    return EngineStatus::Success;
}

EngineStatus BlackwellEngine::Impl::step_mlp_projections(int layer_idx) {
    arena.ensure_layer_ready(layer_idx);
    std::string prefix = m_config.weight_prefix + "layers." + std::to_string(layer_idx) + ".mlp.";

    if (m_config.mlp_fused_gate_up) {
        // One projection for both halves: [2*I] rows, gate first (see ModelConfig).
        dispatcher.forward(prefix + "gate_up_proj", d_X_norm, d_GateUp,
                           2 * m_config.intermediate_dim, m_config.hidden_dim);
        return EngineStatus::Success;
    }

    dispatcher.forward(prefix + "gate_proj", d_X_norm, d_Gate,
                       m_config.intermediate_dim, m_config.hidden_dim);
    dispatcher.forward(prefix + "up_proj",   d_X_norm, d_Up,
                       m_config.intermediate_dim, m_config.hidden_dim);
    return EngineStatus::Success;
}

EngineStatus BlackwellEngine::Impl::step_mlp_out(int layer_idx) {
    arena.ensure_layer_ready(layer_idx);
    const std::string prefix = m_config.weight_prefix + "layers." + std::to_string(layer_idx) + ".";
    const std::string base = prefix + "mlp.down_proj";

    if (m_config.mlp_fused_gate_up) {
        // Single token: the gate and up halves ARE two adjacent contiguous runs of
        // d_GateUp, so the plain elementwise SwiGLU reads them in place.
        launch_fused_swiglu_kernel(d_GateUp, d_GateUp + m_config.intermediate_dim,
                                   d_Swiglu_out, m_config.intermediate_dim);
    } else {
        launch_fused_swiglu_kernel(d_Gate, d_Up, d_Swiglu_out, m_config.intermediate_dim);
    }

    if (m_config.has_sandwich_norms) {
        // h = h + post_mlp_layernorm(down_proj(swiglu)) -- same reason as o_proj.
        dispatcher.forward(base, d_Swiglu_out, d_sublayer_out,
                           m_config.hidden_dim, m_config.intermediate_dim, nullptr);
        sandwich_norm_accum(prefix + "post_mlp_layernorm.weight", 1);
        return EngineStatus::Success;
    }

    dispatcher.forward(base, d_Swiglu_out, nullptr,
                       m_config.hidden_dim, m_config.intermediate_dim, d_X_accum);
    return EngineStatus::Success;
}

// ============================================================================
// STAGE 4: Final Operations
// ============================================================================
EngineStatus BlackwellEngine::Impl::step_final_ops() {
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
    return EngineStatus::Success;
}

// Реализация фасада BlackwellEngine.
// Legacy constructor: map the loose positional args onto a tier-2 request plus
// low-level overrides (explicit kv_mode + residency split), preserving the exact
// prior behavior. require_branching stays false -- callers that want branching use
// the InferenceConfig constructor below.
BlackwellEngine::BlackwellEngine(const std::string& index_path, size_t max_seq_len, size_t num_gpu_layers,
                                 KVCacheMode kv_mode) {
    blackwell::InferenceConfig request;
    request.max_context_length = max_seq_len;
    blackwell::RuntimeOverrides overrides;
    overrides.kv_mode = kv_mode;
    overrides.num_gpu_layers = num_gpu_layers;   // kAllLayersResident => env / all-resident
    pImpl = std::make_unique<Impl>(index_path, request, overrides);
}

// Tier-2 constructor: the validated builder picks kv_mode (Paged iff branching).
BlackwellEngine::BlackwellEngine(const std::string& index_path, const blackwell::InferenceConfig& request)
    : pImpl(std::make_unique<Impl>(index_path, request, blackwell::RuntimeOverrides{})) {}

// Tier-2 + explicit low-level overrides (kv_mode, offloading, tiered KV sizing).
BlackwellEngine::BlackwellEngine(const std::string& index_path, const blackwell::InferenceConfig& request,
                                 const blackwell::RuntimeOverrides& overrides)
    : pImpl(std::make_unique<Impl>(index_path, request, overrides)) {}

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
            "CoW branching (Continuous KV mode — construct with KVCacheMode::Paged for "
            "branching; hybrid SSM models fork via physical state snapshot, but still "
            "require Paged mode).");
}

// Clone every per-sequence state store parent owns into child. This is the single
// place the three storage worlds a hybrid model keeps are snapshotted together, so
// one logical fork() branches all of them atomically. A dense model exercises only
// the paged KV path; the SSM and full-attention snapshots are no-ops it never has.
void BlackwellEngine::Impl::fork(int parent_id, int child_id) {
    // The hybrid physical stores are indexed directly by sequence id and sized to
    // m_branch_capacity; reject ids that don't fit BEFORE any partial clone (a
    // half-forked child would be worse than a clean failure).
    if (ssm_state || d_full_k_cache) {
        const auto in_range = [&](int id) { return id >= 0 && id < m_branch_capacity; };
        if (!in_range(parent_id) || !in_range(child_id))
            throw std::runtime_error(
                "BlackwellEngine::fork: sequence id out of hybrid branch capacity [0, " +
                std::to_string(m_branch_capacity) +
                ") -- raise paged_branch_factor to hold more concurrent Tree-of-Thoughts "
                "branches");
    }

    // 1. Attention KV pages: paged CoW share (O(blocks), no copy until first write).
    //    For a dense model this is the whole fork; for a hybrid model these pages
    //    are the (unused) placeholder the paged manager tracks per sequence.
    kv_mgr->fork(parent_id, child_id);

    // 2. Recurrent SSM state: physical device-to-device snapshot of the dense,
    //    context-independent slice.
    if (ssm_state) ssm_state->fork_sequence(parent_id, child_id);

    // 3. Dedicated gated full-attention KV cache: the paged fork above does NOT
    //    cover it (these head_dim-256 layers keep their own continuous cache), so
    //    snapshot the parent's per-sequence slice into the child explicitly. D2D
    //    on the default stream, ordered with the SSM copy and the later decode.
    if (d_full_k_cache) {
        const size_t off_p = (size_t)parent_id * m_full_kv_seq_stride;
        const size_t off_c = (size_t)child_id  * m_full_kv_seq_stride;
        const size_t bytes = m_full_kv_seq_stride * sizeof(float);
        CUDA_CHECK_THROW(cudaMemcpy(d_full_k_cache + off_c, d_full_k_cache + off_p,
                                    bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK_THROW(cudaMemcpy(d_full_v_cache + off_c, d_full_v_cache + off_p,
                                    bytes, cudaMemcpyDeviceToDevice));
    }
}

void BlackwellEngine::fork(int parent_id, int child_id) {
    require_branching(pImpl->m_caps, "fork");
    pImpl->fork(parent_id, child_id);
}

void BlackwellEngine::rewind(int seq_id, int pos) {
    require_branching(pImpl->m_caps, "rewind");
    // Recurrent SSM state cannot be rolled back to an arbitrary position: it
    // accumulates with every decode and keeps no per-position history to restore
    // (fork() snapshots forward; rewind() would need the inverse, which does not
    // exist). Reject on hybrid models rather than silently rewinding only the
    // (unused) paged pages and leaving the recurrent state ahead of `pos`.
    if (pImpl->m_caps.requires_ssm_subsystem)
        throw std::runtime_error(
            "BlackwellEngine::rewind: unsupported for hybrid linear-attention/SSM "
            "models — the recurrent state cannot be rolled back to an arbitrary "
            "position (use fork() to branch, or reset_state() to restart at 0).");
    pImpl->kv_mgr->rewind(seq_id, pos);
}

void BlackwellEngine::reset_state(int seq_id) {
    auto* impl = pImpl.get();
    // Attention KV (continuous / paged / the dedicated full-attention cache) is
    // POSITION-ADDRESSED: re-decoding from pos 0 overwrites stale slots, so it
    // self-heals on reprefill and needs nothing here. The recurrent linear-
    // attention (SSM) state is the exception -- it accumulates with every forward()
    // and cannot be rewound -- so it must be zeroed explicitly on a sequence
    // restart. Dense models have no SSM state: this is then a no-op.
    if (!impl->ssm_state) return;
    // Hybrid models are multi-sequence now (fork() snapshots the recurrent state):
    // reset the requested branch's slice. The pool bounds-checks seq_id against
    // its branch capacity. reset() is stream-0 (the compute stream) ordered, so it
    // composes with the subsequent decode kernels without an extra device sync.
    impl->ssm_state->reset(seq_id);
}

void BlackwellEngine::release_sequence(int seq_id) {
    // Same gate as fork/rewind: recycling a branch id only means anything on the
    // branching-capable paged cache. Under Continuous / hybrid-without-branching
    // there are no fork children to free.
    require_branching(pImpl->m_caps, "release_sequence");
    // Only the paged pool holds per-branch allocations to reclaim. The hybrid
    // physical stores (SSM recurrent/conv slice, gated full-attention KV slice)
    // are statically sized to m_branch_capacity and indexed directly by seq_id,
    // so nothing is freed there -- the next fork() into this id overwrites the
    // slice wholesale (fork_sequence / the D2D copy in Impl::fork). Leaving stale
    // bytes behind is harmless: a slot is only ever decoded after a fork seeds it.
    pImpl->kv_mgr->release_sequence(seq_id);
}

int BlackwellEngine::branch_capacity() const noexcept { return pImpl->m_branch_capacity; }

// ============================================================================
// Shared decoder pipeline: embedding -> N transformer layers -> final norm/head.
// With want_logits, leaves the logits for token `pos` in d_logits; without, the
// pass ends after the last layer's MLP — the prefill sweep's fast path, where
// only the KV appended along the way matters and the vocab-size lm_head GEMV
// (the single largest GEMV in the model) is skipped per prompt token.
// ============================================================================
EngineStatus BlackwellEngine::Impl::run_token(int token_id, int pos, int seq_id, bool want_logits) {
    auto* impl = this;
    const size_t max_seq_len = impl->arena.get_max_seq_len();
    if (pos < 0 || static_cast<size_t>(pos) >= max_seq_len) {
        // Caller error on the runtime tier: report by status (the RoPE/KV
        // append kernel would write out of bounds).
        std::cerr << "[blackwell_core] pos " << pos << " exceeds KV cache capacity "
                  << max_seq_len << "\n";
        return EngineStatus::InvalidArgument;
    }

    ENGINE_TRY(impl->step_embedding(token_id));

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
            ENGINE_TRY(impl->step_linear_attention(i, pos, seq_id));
        } else if (impl->m_config.attn_output_gate) {
            // Qwen3.5 hybrid: gated head_dim-256 full-attention (dedicated path).
            ENGINE_TRY(impl->step_full_attention(i, pos, seq_id));
        } else {
            ENGINE_TRY(impl->step_attention_norm(i));
            ENGINE_TRY(impl->step_attention_qkv_projections(i));
            ENGINE_TRY(impl->step_attention_math(i, pos));
            ENGINE_TRY(impl->step_attention_out(i));
        }

        // The MLP block is identical for both layer kinds.
        ENGINE_TRY(impl->step_mlp_norm(i));
        ENGINE_TRY(impl->step_mlp_projections(i));
        ENGINE_TRY(impl->step_mlp_out(i));
    }

    if (want_logits)
        ENGINE_TRY(impl->step_final_ops());
    return EngineStatus::Success;
}

// ============================================================================
// Batched prefill: one forward pass over num_tokens prompt tokens. Mirrors
// run_token's generic dense path but every stage is widened to num_tokens rows,
// staged contiguously in the m_token_capacity-sized activation/scratch buffers.
// The heavy projections run through the dispatcher's num_tokens path (Tensor-
// Core GEMM for BF16, per-row GEMV sweep for the quantized checkpoints); the
// paged-flash attention runs the full BLOCK_M-tile prefill kernel. Only the
// small per-channel epilogues (QKV bias) loop over rows.
//
// This is the drop-in the coordinator (run_delta) calls instead of sweeping
// run_token token-by-token; numerically it reproduces that sweep (the quantized
// projections are the SAME GEMV per row, RoPE/append/attention the same kernels
// per position), so a batched prompt and a single-token prompt land on the same
// logits. Dense uniform full-attention runs the pipeline below; SSM / gated-full
// hybrid models delegate to run_chunk_hybrid (chunked delta rule) at the top.
// ============================================================================
EngineStatus BlackwellEngine::Impl::run_chunk(const int* token_ids, int start_pos,
                                              int num_tokens, int seq_id,
                                              bool want_logits) {
    const size_t H   = m_config.hidden_dim;
    const size_t msl = arena.get_max_seq_len();
    if (num_tokens <= 0 || token_ids == nullptr) return EngineStatus::InvalidArgument;
    if (static_cast<size_t>(num_tokens) > m_token_capacity) {
        std::cerr << "[blackwell_core] run_chunk: num_tokens " << num_tokens
                  << " exceeds token capacity " << m_token_capacity
                  << " (caller must tile the delta)\n";
        return EngineStatus::InvalidArgument;
    }
    if (start_pos < 0 || static_cast<size_t>(start_pos + num_tokens) > msl) {
        std::cerr << "[blackwell_core] run_chunk: positions [" << start_pos << ", "
                  << (start_pos + num_tokens) << ") exceed KV capacity " << msl << "\n";
        return EngineStatus::InvalidArgument;
    }
    // Hybrid (SSM / gated full-attention) models take the dedicated chunked path:
    // the generic dense body below cannot express the linear-attention recurrence
    // or the head_dim-256 gated cache. Dense uniform full-attention falls through.
    if (m_caps.requires_ssm_subsystem || m_config.attn_output_gate)
        return run_chunk_hybrid(token_ids, start_pos, num_tokens, seq_id, want_logits);

    const bool fp16_w = half_weights_are_fp16(m_config);

    // 1. Batched embedding: num_tokens ids -> d_X_accum [num_tokens, H].
    CUDA_CHECK_RETURN(cudaMemcpy(d_next_token, token_ids, num_tokens * sizeof(int),
                                 cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemset(d_X_accum, 0, num_tokens * H * sizeof(float)));
    {
        const void* d_embed = arena.get_weight_ptr(m_config.weight_prefix + "embed_tokens.weight");
        if (fp16_w) launch_fp16_embedding_kernel(d_next_token, d_embed, d_X_accum, num_tokens, H);
        else        launch_bf16_embedding_kernel(d_next_token, d_embed, d_X_accum, num_tokens, H);
    }

    // 2. Reserve the chunk's pages + stage the block table (may throw on a KV gap;
    // the coordinator's run_delta owns the noexcept boundary).
    kv_mgr->prepare_prefill_step(seq_id, start_pos, num_tokens);

    const size_t q_dim  = m_config.num_attention_heads * m_config.head_dim;
    const size_t kv_dim = m_config.num_key_value_heads * m_config.head_dim;
    const int    L      = static_cast<int>(m_config.num_layers);

    for (int i = 0; i < L; ++i) {
        arena.prefetch_layer(i + 1, start_pos + num_tokens - 1);
        const std::string lp = m_config.weight_prefix + "layers." + std::to_string(i) + ".";
        const std::string sa = lp + "self_attn.";
        arena.ensure_layer_ready(i);

        // --- attention RMSNorm (batched) ---
        {
            const void* w = arena.get_weight_ptr(lp + "input_layernorm.weight");
            if (fp16_w) launch_rmsnorm_fp16_kernel(d_X_accum, d_X_norm, w, num_tokens, H,
                                                   m_config.rms_norm_eps, m_config.norm_add_unit_offset);
            else        launch_rmsnorm_kernel(d_X_accum, d_X_norm, w, num_tokens, H,
                                              m_config.rms_norm_eps, m_config.norm_add_unit_offset);
        }

        // --- QKV projections (batched) + per-row bias epilogue ---
        const void* d_bias_q = nullptr;
        const void* d_bias_k = nullptr;
        const void* d_bias_v = nullptr;
        if (m_config.has_qkv_bias) {
            d_bias_q = arena.get_weight_ptr_optional(sa + "q_proj.bias");
            d_bias_k = arena.get_weight_ptr_optional(sa + "k_proj.bias");
            d_bias_v = arena.get_weight_ptr_optional(sa + "v_proj.bias");
            if ((d_bias_q != nullptr) != (d_bias_k != nullptr) ||
                (d_bias_q != nullptr) != (d_bias_v != nullptr)) {
                std::cerr << "[blackwell_core] run_chunk: QKV bias partially missing at layer "
                          << i << "\n";
                return EngineStatus::InvalidConfig;
            }
        }
        dispatcher.forward(sa + "q_proj", d_X_norm, d_Q, q_dim,  H, nullptr, num_tokens);
        dispatcher.forward(sa + "k_proj", d_X_norm, d_K, kv_dim, H, nullptr, num_tokens);
        dispatcher.forward(sa + "v_proj", d_X_norm, d_V, kv_dim, H, nullptr, num_tokens);
        if (d_bias_q) {
            const BiasDType bt = (m_config.quant_strategy == QuantStrategy::WEIGHT_ONLY_PACKED)
                                     ? BiasDType::FP16 : BiasDType::BF16;
            for (int t = 0; t < num_tokens; ++t)
                launch_fused_qkv_bias_kernel(d_Q + t * q_dim, d_K + t * kv_dim, d_V + t * kv_dim,
                                             d_bias_q, d_bias_k, d_bias_v, q_dim, kv_dim, bt);
        }

        // --- batched paged-flash prefill attention (RoPE + append + attention) ---
        kv_mgr->attention_prefill(i, start_pos, num_tokens, d_Q, d_K, d_V, d_Attn_out);

        // --- o_proj, accumulate into the residual stream (batched) ---
        // Sandwich norms (GLM-4-0414) break the fused accumulate for the same
        // reason they do in step_attention_out: the norm sits in between.
        if (m_config.has_sandwich_norms) {
            dispatcher.forward(sa + "o_proj", d_Attn_out, d_sublayer_out, H, H, nullptr, num_tokens);
            sandwich_norm_accum(lp + "post_self_attn_layernorm.weight",
                                static_cast<size_t>(num_tokens));
        } else {
            dispatcher.forward(sa + "o_proj", d_Attn_out, nullptr, H, H, d_X_accum, num_tokens);
        }

        // --- MLP (batched) ---
        {
            const void* w = arena.get_weight_ptr(lp + "post_attention_layernorm.weight");
            if (fp16_w) launch_rmsnorm_fp16_kernel(d_X_accum, d_X_norm, w, num_tokens, H,
                                                   m_config.rms_norm_eps, m_config.norm_add_unit_offset);
            else        launch_rmsnorm_kernel(d_X_accum, d_X_norm, w, num_tokens, H,
                                              m_config.rms_norm_eps, m_config.norm_add_unit_offset);
        }
        if (m_config.mlp_fused_gate_up) {
            dispatcher.forward(lp + "mlp.gate_up_proj", d_X_norm, d_GateUp,
                               2 * m_config.intermediate_dim, H, nullptr, num_tokens);
            // A row of d_GateUp is [gate | up], so the halves are interleaved BY ROW
            // once num_tokens > 1 -- the strided launcher, not a pointer pair.
            launch_fused_swiglu_gate_up_kernel(d_GateUp, d_Swiglu_out,
                                               static_cast<size_t>(num_tokens),
                                               m_config.intermediate_dim);
        } else {
            dispatcher.forward(lp + "mlp.gate_proj", d_X_norm, d_Gate, m_config.intermediate_dim, H,
                               nullptr, num_tokens);
            dispatcher.forward(lp + "mlp.up_proj",   d_X_norm, d_Up,   m_config.intermediate_dim, H,
                               nullptr, num_tokens);
            launch_fused_swiglu_kernel(d_Gate, d_Up, d_Swiglu_out,
                                       static_cast<size_t>(num_tokens) * m_config.intermediate_dim);
        }
        if (m_config.has_sandwich_norms) {
            dispatcher.forward(lp + "mlp.down_proj", d_Swiglu_out, d_sublayer_out, H,
                               m_config.intermediate_dim, nullptr, num_tokens);
            sandwich_norm_accum(lp + "post_mlp_layernorm.weight",
                                static_cast<size_t>(num_tokens));
        } else {
            dispatcher.forward(lp + "mlp.down_proj", d_Swiglu_out, nullptr, H,
                               m_config.intermediate_dim, d_X_accum, num_tokens);
        }
    }

    // 3. Final norm + lm_head for the LAST token only (its logits drive sampling).
    if (want_logits) {
        float* last_hidden = d_X_accum + static_cast<size_t>(num_tokens - 1) * H;
        const void* d_w = arena.get_weight_ptr(m_config.weight_prefix + "norm.weight");
        if (fp16_w) launch_rmsnorm_fp16_kernel(last_hidden, d_X_norm, d_w, 1, H,
                                               m_config.rms_norm_eps, m_config.norm_add_unit_offset);
        else        launch_rmsnorm_kernel(last_hidden, d_X_norm, d_w, 1, H,
                                          m_config.rms_norm_eps, m_config.norm_add_unit_offset);
        const void* d_head_w = m_config.tie_word_embeddings
            ? arena.get_weight_ptr(m_config.weight_prefix + "embed_tokens.weight")
            : arena.get_weight_ptr("lm_head.weight");
        if (fp16_w) launch_fp16_gemv_kernel(d_head_w, d_X_norm, d_logits, m_config.vocab_size, H);
        else        launch_bf16_gemv_kernel(d_head_w, d_X_norm, d_logits, m_config.vocab_size, H);
    }
    return EngineStatus::Success;
}

// ============================================================================
// HYBRID True Batched Prefill. run_chunk delegates here for SSM / gated-full
// models: the generic dense body cannot express the linear-attention recurrence
// (chunked delta rule) or the head_dim-256 gated cache. Structure mirrors the
// dense run_chunk -- batched embedding, per-layer mixer, batched MLP, final norm
// + lm_head for the last token -- but the per-layer mixer dispatches to the two
// chunk steps below. Numerically it reproduces run_token's token-by-token sweep
// (each per-token/per-head kernel does the identical math), so a batched prompt
// and a single-token sweep land on the same logits, to fp32 rounding.
// ============================================================================
EngineStatus BlackwellEngine::Impl::run_chunk_hybrid(const int* token_ids, int start_pos,
                                                     int num_tokens, int seq_id,
                                                     bool want_logits) {
    const size_t H = m_config.hidden_dim;
    const bool fp16_w = half_weights_are_fp16(m_config);
    const int N = num_tokens;

    // 1. Batched embedding: N ids -> d_X_accum [N, H].
    CUDA_CHECK_RETURN(cudaMemcpy(d_next_token, token_ids, N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemset(d_X_accum, 0, (size_t)N * H * sizeof(float)));
    {
        const void* d_embed = arena.get_weight_ptr(m_config.weight_prefix + "embed_tokens.weight");
        if (fp16_w) launch_fp16_embedding_kernel(d_next_token, d_embed, d_X_accum, N, H);
        else        launch_bf16_embedding_kernel(d_next_token, d_embed, d_X_accum, N, H);
    }

    // 2. Keep the paged manager's per-seq length consistent with the prefill
    //    positions. The linear/full-attn layers hold their own state stores, but a
    //    later decode/fork on this seq resolves its slot through the paged length
    //    bookkeeping. May throw on a forward KV gap -- the caller (coordinator or
    //    test) owns that exception boundary, exactly as the dense run_chunk does.
    kv_mgr->prepare_prefill_step(seq_id, start_pos, num_tokens);

    const int L = static_cast<int>(m_config.num_layers);
    for (int i = 0; i < L; ++i) {
        arena.prefetch_layer(i + 1, start_pos + num_tokens - 1);
        arena.ensure_layer_ready(i);
        const std::string lp = m_config.weight_prefix + "layers." + std::to_string(i) + ".";

        // --- attention mixer over the chunk (linear recurrence or gated cache) ---
        const bool is_linear = !m_config.layer_types.empty() &&
                               m_config.layer_types[i] == AttnKind::Linear;
        if (is_linear) {
            ENGINE_TRY(step_linear_attention_chunk(i, start_pos, num_tokens, seq_id));
        } else if (m_config.attn_output_gate) {
            ENGINE_TRY(step_full_attention_chunk(i, start_pos, num_tokens, seq_id));
        } else {
            // A plain dense full-attention layer inside a hybrid stack is not a
            // configuration Qwen3.5 produces (every layer is linear or gated); fail
            // loudly rather than silently miscompute through the wrong path.
            std::cerr << "[blackwell_core] run_chunk_hybrid: unexpected dense "
                         "full-attention layer " << i << " in a hybrid model\n";
            return EngineStatus::InvalidConfig;
        }

        // --- MLP (batched), byte-for-byte the dense run_chunk MLP block ---
        {
            const void* w = arena.get_weight_ptr(lp + "post_attention_layernorm.weight");
            if (fp16_w) launch_rmsnorm_fp16_kernel(d_X_accum, d_X_norm, w, N, H,
                                                   m_config.rms_norm_eps, m_config.norm_add_unit_offset);
            else        launch_rmsnorm_kernel(d_X_accum, d_X_norm, w, N, H,
                                              m_config.rms_norm_eps, m_config.norm_add_unit_offset);
        }
        dispatcher.forward(lp + "mlp.gate_proj", d_X_norm, d_Gate, m_config.intermediate_dim, H,
                           nullptr, N);
        dispatcher.forward(lp + "mlp.up_proj",   d_X_norm, d_Up,   m_config.intermediate_dim, H,
                           nullptr, N);
        launch_fused_swiglu_kernel(d_Gate, d_Up, d_Swiglu_out,
                                   static_cast<size_t>(N) * m_config.intermediate_dim);
        dispatcher.forward(lp + "mlp.down_proj", d_Swiglu_out, nullptr, H,
                           m_config.intermediate_dim, d_X_accum, N);
    }

    // 3. Final norm + lm_head for the LAST token only (its logits drive sampling).
    if (want_logits) {
        float* last_hidden = d_X_accum + static_cast<size_t>(N - 1) * H;
        const void* d_w = arena.get_weight_ptr(m_config.weight_prefix + "norm.weight");
        if (fp16_w) launch_rmsnorm_fp16_kernel(last_hidden, d_X_norm, d_w, 1, H,
                                               m_config.rms_norm_eps, m_config.norm_add_unit_offset);
        else        launch_rmsnorm_kernel(last_hidden, d_X_norm, d_w, 1, H,
                                          m_config.rms_norm_eps, m_config.norm_add_unit_offset);
        const void* d_head_w = m_config.tie_word_embeddings
            ? arena.get_weight_ptr(m_config.weight_prefix + "embed_tokens.weight")
            : arena.get_weight_ptr("lm_head.weight");
        if (fp16_w) launch_fp16_gemv_kernel(d_head_w, d_X_norm, d_logits, m_config.vocab_size, H);
        else        launch_bf16_gemv_kernel(d_head_w, d_X_norm, d_logits, m_config.vocab_size, H);
    }
    return EngineStatus::Success;
}

// ----------------------------------------------------------------------------
// Linear-attention layer over a CHUNK. Mirrors step_linear_attention but widens
// the whole layer to num_tokens rows: batched norm/in-proj, a per-token sweep for
// the strictly-sequential conv1d ring + split/normalize (scattered token-major
// into the chunk buffers), ONE chunked delta-rule launch to carry the recurrent
// state across the chunk, then batched gated-RMSNorm + out_proj. seq_id selects
// the branch's recurrent/conv slice; the state carries across chunks via the pool.
// ----------------------------------------------------------------------------
EngineStatus BlackwellEngine::Impl::step_linear_attention_chunk(int layer_idx, int start_pos,
                                                                int num_tokens, int seq_id) {
    arena.ensure_layer_ready(layer_idx);
    const int li = m_linear_layer_index[layer_idx];
    const std::string base = m_config.weight_prefix + "layers." + std::to_string(layer_idx) + ".";
    const std::string la   = base + "linear_attn.";
    const int N = num_tokens;
    (void)start_pos;   // recurrence is position-implicit (state carries history)

    float* d_state = ssm_state->rec_state(seq_id, li);
    float* d_conv  = ssm_state->conv_state(seq_id, li);

    const auto& Lc = m_config.linear;
    const int Kh = (int)Lc.num_key_heads, Hh = (int)Lc.num_value_heads;
    const int Dk = (int)Lc.key_head_dim,  Dv = (int)Lc.value_head_dim, Kw = (int)Lc.conv_kernel_dim;
    const size_t conv_dim = 2 * (size_t)Kh * Dk + (size_t)Hh * Dv;
    const size_t v_dim    = (size_t)Hh * Dv;
    const size_t hidden   = m_config.hidden_dim;

    // 1. batched input RMSNorm (bf16 weight, exactly as step_linear_attention).
    launch_rmsnorm_kernel(d_X_accum, d_X_norm,
                          arena.get_weight_ptr(base + "input_layernorm.weight"),
                          N, hidden, m_config.rms_norm_eps, m_config.norm_add_unit_offset);

    // 2. batched in-projections (qkv/z via the dispatcher; a/b are small bf16 GEMVs
    //    swept per row below). Per-layer params cast to fp32 once for the chunk.
    dispatcher.forward(la + "in_proj_qkv", d_X_norm, d_ssm_qkv_batch, conv_dim, hidden, nullptr, N);
    dispatcher.forward(la + "in_proj_z",   d_X_norm, d_ssm_z_batch,   v_dim,    hidden, nullptr, N);
    blackwell::ssm::launch_bf16_to_f32(arena.get_weight_ptr(la + "conv1d.weight"),
                                       d_conv_w_f32, (int)conv_dim * Kw);
    blackwell::ssm::launch_bf16_to_f32(arena.get_weight_ptr(la + "dt_bias"), d_dt_bias_f32, Hh);
    blackwell::ssm::launch_bf16_to_f32(arena.get_weight_ptr(la + "A_log"),   d_A_log_f32,   Hh);

    // 3. per token: conv1d ring advance (sequential -- the ring carries within AND
    //    across chunks), split+L2-normalize, decay a and write-strength beta,
    //    scattered token-major into the chunk buffers the delta kernel consumes.
    //    Everything runs on the default stream, so the async copies are ordered
    //    with the kernels that produce the single-token d_ssm_* they read.
    for (int t = 0; t < N; ++t) {
        const float* xn = d_X_norm + (size_t)t * hidden;
        launch_bf16_gemv_kernel(arena.get_weight_ptr(la + "in_proj_a.weight"), xn, d_ssm_a, Hh, (int)hidden);
        launch_bf16_gemv_kernel(arena.get_weight_ptr(la + "in_proj_b.weight"), xn, d_ssm_b, Hh, (int)hidden);
        blackwell::ssm::launch_sigmoid_inplace(d_ssm_b, Hh);                  // b -> beta
        blackwell::ssm::launch_gated_delta_logdecay(d_ssm_a, d_dt_bias_f32, d_A_log_f32,
                                                    d_ssm_chunk_logdecay + (size_t)t * Hh, Hh);
        CUDA_CHECK_RETURN(cudaMemcpyAsync(d_ssm_chunk_beta + (size_t)t * Hh, d_ssm_b,
                                          (size_t)Hh * sizeof(float), cudaMemcpyDeviceToDevice));

        float* qkv_row = d_ssm_qkv_batch + (size_t)t * conv_dim;
        blackwell::ssm::launch_causal_conv1d_update(qkv_row, d_conv, d_conv_w_f32, /*bias=*/nullptr,
                                                    d_ssm_qkv_conv, (int)conv_dim, Kw, /*silu=*/true);
        blackwell::ssm::launch_ssm_split_norm_broadcast(d_ssm_qkv_conv, d_ssm_q, d_ssm_k, d_ssm_v,
                                                        Kh, Hh, Dk);
        CUDA_CHECK_RETURN(cudaMemcpyAsync(d_ssm_chunk_q + (size_t)t * Hh * Dk, d_ssm_q,
                                          (size_t)Hh * Dk * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK_RETURN(cudaMemcpyAsync(d_ssm_chunk_k + (size_t)t * Hh * Dk, d_ssm_k,
                                          (size_t)Hh * Dk * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK_RETURN(cudaMemcpyAsync(d_ssm_chunk_v + (size_t)t * Hh * Dv, d_ssm_v,
                                          (size_t)Hh * Dv * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    // 4. ONE chunked delta-rule launch carries S over the chunk. h_prev == h_out ==
    //    the pool slice, so the kernel seeds from and writes back the same state.
    blackwell::ssm::launch_gated_delta_chunked_prefill(
        d_ssm_chunk_k, d_ssm_chunk_q, d_ssm_chunk_v, d_ssm_chunk_logdecay, d_ssm_chunk_beta,
        d_state, d_ssm_chunk_o, d_state, d_ssm_u_scratch, Hh, N, Dk, Dv);

    // 5. batched per-head gated RMSNorm: o = rmsnorm(o * silu(z)) * norm.weight[Dv].
    //    chunk-o and z_batch are both [N*Hh][Dv] token-major, so a single call over
    //    N*Hh "heads" (gamma repeats per Dv channel) reproduces the per-token norm.
    blackwell::ssm::launch_bf16_to_f32(arena.get_weight_ptr(la + "norm.weight"), d_norm_f32, Dv);
    blackwell::ssm::launch_gated_rmsnorm_per_head(d_ssm_chunk_o, d_ssm_z_batch, d_norm_f32,
                                                  d_ssm_o_batch, N * Hh, Dv, m_config.rms_norm_eps);

    // 6. batched out_proj, accumulating into the residual stream.
    dispatcher.forward(la + "out_proj", d_ssm_o_batch, nullptr, hidden, v_dim, d_X_accum, N);
    return EngineStatus::Success;
}

// ----------------------------------------------------------------------------
// Gated full-attention layer over a CHUNK (Qwen3.5 head_dim 256). Batched
// projections, then a per-token sweep for the inherently position-serial stages
// (de-interleave query|gate, q/k norm, partial RoPE at the token's position,
// append into the dedicated cache, decode over 0..pos, gate) -- each reproducing
// step_full_attention exactly -- then batched o_proj. seq_id selects the branch's
// slice of the dedicated cache.
// ----------------------------------------------------------------------------
EngineStatus BlackwellEngine::Impl::step_full_attention_chunk(int layer_idx, int start_pos,
                                                              int num_tokens, int seq_id) {
    arena.ensure_layer_ready(layer_idx);
    const std::string base = m_config.weight_prefix + "layers." + std::to_string(layer_idx) + ".";
    const std::string sa   = base + "self_attn.";
    const int N = num_tokens;

    const int Hq  = (int)m_config.num_attention_heads;
    const int Hkv = (int)m_config.num_key_value_heads;
    const int Dh  = (int)m_config.head_dim;
    const int rot = (int)m_config.rotary_dim;
    const size_t q_dim  = (size_t)Hq * Dh;
    const size_t kv_dim = (size_t)Hkv * Dh;
    const size_t hidden = m_config.hidden_dim;

    // 1. batched input RMSNorm (bf16 weight, exactly as step_full_attention).
    launch_rmsnorm_kernel(d_X_accum, d_X_norm,
                          arena.get_weight_ptr(base + "input_layernorm.weight"),
                          N, hidden, m_config.rms_norm_eps, m_config.norm_add_unit_offset);

    // 2. batched projections. q_proj emits [N, 2*q_dim] (query|gate); k/v [N, kv_dim].
    dispatcher.forward(sa + "q_proj", d_X_norm, d_QG, q_dim * 2, hidden, nullptr, N);
    dispatcher.forward(sa + "k_proj", d_X_norm, d_K,  kv_dim,    hidden, nullptr, N);
    dispatcher.forward(sa + "v_proj", d_X_norm, d_V,  kv_dim,    hidden, nullptr, N);

    // Per-branch slice of the dedicated gated full-attention KV cache.
    const int fo = m_full_layer_index[layer_idx];
    const size_t seq_off = (size_t)seq_id * m_full_kv_seq_stride;
    float* d_k_cache = d_full_k_cache + seq_off + (size_t)fo * m_full_kv_layer_stride;
    float* d_v_cache = d_full_v_cache + seq_off + (size_t)fo * m_full_kv_layer_stride;
    const int msl = (int)arena.get_max_seq_len();

    // 3. per token: the position-serial stages, each identical to the decode step.
    for (int t = 0; t < N; ++t) {
        const int pos = start_pos + t;
        float* qg = d_QG       + (size_t)t * q_dim * 2;
        float* qr = d_Q        + (size_t)t * q_dim;
        float* kr = d_K        + (size_t)t * kv_dim;
        float* vr = d_V        + (size_t)t * kv_dim;
        float* ar = d_Attn_out + (size_t)t * q_dim;

        launch_qg_split(qg, qr, d_gate, Hq, Dh);
        launch_rmsnorm_kernel(qr, qr, arena.get_weight_ptr(sa + "q_norm.weight"),
                              Hq, Dh, m_config.rms_norm_eps, m_config.norm_add_unit_offset);
        launch_rmsnorm_kernel(kr, kr, arena.get_weight_ptr(sa + "k_norm.weight"),
                              Hkv, Dh, m_config.rms_norm_eps, m_config.norm_add_unit_offset);
        launch_rope_partial_inplace(qr, pos, Hq,  Dh, rot, m_config.rope_theta, rope_scaling_from(m_config));
        launch_rope_partial_inplace(kr, pos, Hkv, Dh, rot, m_config.rope_theta, rope_scaling_from(m_config));
        launch_kv_append(kr, vr, d_k_cache, d_v_cache, pos, Hkv, Dh, msl);
        launch_full_attention_decode(qr, d_k_cache, d_v_cache, ar, pos, Hq, Hkv, Dh, msl);
        launch_gate_sigmoid_mul(ar, d_gate, (int)q_dim);
    }

    // 4. batched o_proj, accumulating into the residual stream.
    dispatcher.forward(sa + "o_proj", d_Attn_out, nullptr, hidden, q_dim, d_X_accum, N);
    return EngineStatus::Success;
}

// ============================================================================
// TRUE-batch decode: one forward pass over batch_size INDEPENDENT sequences,
// one new token each. The wavefront primitive for parallel code-agent
// orchestration (Tree-of-Thoughts): N branches advance together instead of via
// N sequential run_token passes, so the weight reads amortize over the batch
// through the dispatcher's num_tokens>1 Tensor-Core GEMM path.
//
// Contrast with run_chunk: run_chunk widens over consecutive positions of ONE
// sequence (shared block table, causal tile); this widens over SEQUENCES, each
// carrying its own block table, length and position -- so RoPE, KV append and
// attention all route through the sequence-aware batched kernels
// (attention_decode_batch), while the projections/MLP/embedding reuse the exact
// same batched primitives run_chunk uses. Every per-row/per-seq kernel does the
// identical math its batch=1 counterpart does, so a batched step reproduces the
// sequential decode loop's logits (BatchedDecodeMatchesSequential).
// ============================================================================
EngineStatus BlackwellEngine::Impl::run_decode_batch(const int* token_ids, const int* seqs,
                                                     const int* positions, int batch_size,
                                                     int* out_next_tokens) {
    const size_t H   = m_config.hidden_dim;
    const size_t V   = m_config.vocab_size;
    const size_t msl = arena.get_max_seq_len();
    if (batch_size <= 0 || !token_ids || !seqs || !positions || !out_next_tokens)
        return EngineStatus::InvalidArgument;
    if (static_cast<size_t>(batch_size) > m_token_capacity) {
        std::cerr << "[blackwell_core] run_decode_batch: batch_size " << batch_size
                  << " exceeds token capacity " << m_token_capacity << "\n";
        return EngineStatus::InvalidArgument;
    }
    // Batched decode is the generic dense full-attention pipeline only (same
    // restriction as run_chunk: SSM recurrent state and the gated full-attention
    // cache live outside the batched paged path).
    if (!m_config.layer_types.empty() || m_config.attn_output_gate || ssm_state) {
        std::cerr << "[blackwell_core] run_decode_batch: batched decode is unsupported "
                     "for SSM / gated full-attention models\n";
        return EngineStatus::InvalidConfig;
    }
    int max_pos = 0;
    for (int b = 0; b < batch_size; ++b) {
        if (positions[b] < 0 || static_cast<size_t>(positions[b]) >= msl) {
            std::cerr << "[blackwell_core] run_decode_batch: pos " << positions[b]
                      << " (row " << b << ") exceeds KV capacity " << msl << "\n";
            return EngineStatus::InvalidArgument;
        }
        max_pos = std::max(max_pos, positions[b]);
    }

    const bool fp16_w = half_weights_are_fp16(m_config);

    // 1. Batched embedding: batch_size ids -> d_X_accum [batch, H].
    CUDA_CHECK_RETURN(cudaMemcpy(d_next_token, token_ids, batch_size * sizeof(int),
                                 cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemset(d_X_accum, 0, static_cast<size_t>(batch_size) * H * sizeof(float)));
    {
        const void* d_embed = arena.get_weight_ptr(m_config.weight_prefix + "embed_tokens.weight");
        if (fp16_w) launch_fp16_embedding_kernel(d_next_token, d_embed, d_X_accum, batch_size, H);
        else        launch_bf16_embedding_kernel(d_next_token, d_embed, d_X_accum, batch_size, H);
    }

    // 2. Latch the batched control plane: resolve each sequence's append slot (CoW
    // fork-shared boundary pages) + stage the per-sequence device arrays. May
    // throw on an unknown seq id / KV gap -- the forward_batch facade owns that
    // exception-tier boundary (like run_chunk delegating to the coordinator).
    kv_mgr->prepare_decode_batch(seqs, positions, batch_size);

    const size_t q_dim  = m_config.num_attention_heads * m_config.head_dim;
    const size_t kv_dim = m_config.num_key_value_heads * m_config.head_dim;
    const int    L      = static_cast<int>(m_config.num_layers);

    for (int i = 0; i < L; ++i) {
        arena.prefetch_layer(i + 1, max_pos);
        const std::string lp = m_config.weight_prefix + "layers." + std::to_string(i) + ".";
        const std::string sa = lp + "self_attn.";
        arena.ensure_layer_ready(i);

        // --- attention RMSNorm (batched over rows) ---
        {
            const void* w = arena.get_weight_ptr(lp + "input_layernorm.weight");
            if (fp16_w) launch_rmsnorm_fp16_kernel(d_X_accum, d_X_norm, w, batch_size, H,
                                                   m_config.rms_norm_eps, m_config.norm_add_unit_offset);
            else        launch_rmsnorm_kernel(d_X_accum, d_X_norm, w, batch_size, H,
                                              m_config.rms_norm_eps, m_config.norm_add_unit_offset);
        }

        // --- QKV projections (batched) + per-row bias epilogue ---
        const void* d_bias_q = nullptr;
        const void* d_bias_k = nullptr;
        const void* d_bias_v = nullptr;
        if (m_config.has_qkv_bias) {
            d_bias_q = arena.get_weight_ptr_optional(sa + "q_proj.bias");
            d_bias_k = arena.get_weight_ptr_optional(sa + "k_proj.bias");
            d_bias_v = arena.get_weight_ptr_optional(sa + "v_proj.bias");
            if ((d_bias_q != nullptr) != (d_bias_k != nullptr) ||
                (d_bias_q != nullptr) != (d_bias_v != nullptr)) {
                std::cerr << "[blackwell_core] run_decode_batch: QKV bias partially missing at layer "
                          << i << "\n";
                return EngineStatus::InvalidConfig;
            }
        }
        dispatcher.forward(sa + "q_proj", d_X_norm, d_Q, q_dim,  H, nullptr, batch_size);
        dispatcher.forward(sa + "k_proj", d_X_norm, d_K, kv_dim, H, nullptr, batch_size);
        dispatcher.forward(sa + "v_proj", d_X_norm, d_V, kv_dim, H, nullptr, batch_size);
        if (d_bias_q) {
            const BiasDType bt = (m_config.quant_strategy == QuantStrategy::WEIGHT_ONLY_PACKED)
                                     ? BiasDType::FP16 : BiasDType::BF16;
            for (int b = 0; b < batch_size; ++b)
                launch_fused_qkv_bias_kernel(d_Q + b * q_dim, d_K + b * kv_dim, d_V + b * kv_dim,
                                             d_bias_q, d_bias_k, d_bias_v, q_dim, kv_dim, bt);
        }

        // --- batched paged-flash decode (per-seq RoPE + append + attention) ---
        kv_mgr->attention_decode_batch(i, batch_size, d_Q, d_K, d_V, d_Attn_out);

        // --- o_proj, accumulate into the residual stream (batched) ---
        if (m_config.has_sandwich_norms) {
            dispatcher.forward(sa + "o_proj", d_Attn_out, d_sublayer_out, H, H, nullptr, batch_size);
            sandwich_norm_accum(lp + "post_self_attn_layernorm.weight",
                                static_cast<size_t>(batch_size));
        } else {
            dispatcher.forward(sa + "o_proj", d_Attn_out, nullptr, H, H, d_X_accum, batch_size);
        }

        // --- MLP (batched) ---
        {
            const void* w = arena.get_weight_ptr(lp + "post_attention_layernorm.weight");
            if (fp16_w) launch_rmsnorm_fp16_kernel(d_X_accum, d_X_norm, w, batch_size, H,
                                                   m_config.rms_norm_eps, m_config.norm_add_unit_offset);
            else        launch_rmsnorm_kernel(d_X_accum, d_X_norm, w, batch_size, H,
                                              m_config.rms_norm_eps, m_config.norm_add_unit_offset);
        }
        if (m_config.mlp_fused_gate_up) {
            dispatcher.forward(lp + "mlp.gate_up_proj", d_X_norm, d_GateUp,
                               2 * m_config.intermediate_dim, H, nullptr, batch_size);
            launch_fused_swiglu_gate_up_kernel(d_GateUp, d_Swiglu_out,
                                               static_cast<size_t>(batch_size),
                                               m_config.intermediate_dim);
        } else {
            dispatcher.forward(lp + "mlp.gate_proj", d_X_norm, d_Gate, m_config.intermediate_dim, H,
                               nullptr, batch_size);
            dispatcher.forward(lp + "mlp.up_proj",   d_X_norm, d_Up,   m_config.intermediate_dim, H,
                               nullptr, batch_size);
            launch_fused_swiglu_kernel(d_Gate, d_Up, d_Swiglu_out,
                                       static_cast<size_t>(batch_size) * m_config.intermediate_dim);
        }
        if (m_config.has_sandwich_norms) {
            dispatcher.forward(lp + "mlp.down_proj", d_Swiglu_out, d_sublayer_out, H,
                               m_config.intermediate_dim, nullptr, batch_size);
            sandwich_norm_accum(lp + "post_mlp_layernorm.weight",
                                static_cast<size_t>(batch_size));
        } else {
            dispatcher.forward(lp + "mlp.down_proj", d_Swiglu_out, nullptr, H,
                               m_config.intermediate_dim, d_X_accum, batch_size);
        }
    }

    // 3. Batched final norm over ALL rows -> d_X_norm [batch, H].
    {
        const void* d_w = arena.get_weight_ptr(m_config.weight_prefix + "norm.weight");
        if (fp16_w) launch_rmsnorm_fp16_kernel(d_X_accum, d_X_norm, d_w, batch_size, H,
                                               m_config.rms_norm_eps, m_config.norm_add_unit_offset);
        else        launch_rmsnorm_kernel(d_X_accum, d_X_norm, d_w, batch_size, H,
                                          m_config.rms_norm_eps, m_config.norm_add_unit_offset);
    }

    // 4. lm_head + argmax PER ROW. The head GEMV stays per-row (not the dispatcher,
    // exactly like run_token/run_chunk): one row's head projection is bit-identical
    // to the single-sequence decode, which is what makes batched logits match the
    // sequential loop exactly. Each row's greedy token lands in d_next_token[b].
    const void* d_head_w = m_config.tie_word_embeddings
        ? arena.get_weight_ptr(m_config.weight_prefix + "embed_tokens.weight")
        : arena.get_weight_ptr("lm_head.weight");
    for (int b = 0; b < batch_size; ++b) {
        float* row_hidden = d_X_norm + static_cast<size_t>(b) * H;
        float* row_logits = d_logits + static_cast<size_t>(b) * V;
        if (fp16_w) launch_fp16_gemv_kernel(d_head_w, row_hidden, row_logits, V, H);
        else        launch_bf16_gemv_kernel(d_head_w, row_hidden, row_logits, V, H);
        launch_argmax_kernel(row_logits, d_next_token + b, V);
    }
    CUDA_CHECK_RETURN(cudaMemcpy(out_next_tokens, d_next_token, batch_size * sizeof(int),
                                 cudaMemcpyDeviceToHost));
    return EngineStatus::Success;
}

// ============================================================================
// Full Engine Inference -- runtime status tier ONLY (the throwing wrappers
// were purged; no exception leaves these endpoints).
// ============================================================================
// Convert a stray exception from an unmigrated subsystem (paged manager,
// arena, kernels lib) into a status: called ONLY from a catch context inside
// the noexcept endpoints below. This is what makes them safely noexcept.
static EngineStatus status_from_current_exception(const char* op) noexcept {
    try {
        throw;
    } catch (const blackwell::cuda_error& e) {
        std::cerr << "[blackwell_core] " << op << ": " << e.what() << "\n";
        return (e.code() == cudaErrorMemoryAllocation) ? EngineStatus::OutOfVram
                                                       : EngineStatus::CudaRuntimeError;
    } catch (const std::invalid_argument& e) {
        std::cerr << "[blackwell_core] " << op << ": " << e.what() << "\n";
        return EngineStatus::InvalidArgument;
    } catch (const std::exception& e) {
        // Subsystem contract violations (unknown seq id, hibernated arena...)
        // surface as runtime_errors today; StateMismatch is the closest status.
        std::cerr << "[blackwell_core] " << op << ": " << e.what() << "\n";
        return EngineStatus::StateMismatch;
    } catch (...) {
        std::cerr << "[blackwell_core] " << op << ": unknown exception\n";
        return EngineStatus::CudaRuntimeError;
    }
}

blackwell::EngineStatus BlackwellEngine::forward_status(int token_id, int pos,
                                                        float temperature, float top_p,
                                                        int seq_id,
                                                        int* next_token) noexcept {
    if (!next_token) return EngineStatus::InvalidArgument;
    try {
        auto* impl = pImpl.get();
        ENGINE_TRY(impl->run_token(token_id, pos, seq_id));
        *next_token = sample_top_p(impl->d_logits, impl->m_config.vocab_size, temperature, top_p);
        return EngineStatus::Success;
    } catch (...) {
        return status_from_current_exception("forward_status");
    }
}

// ============================================================================
// Prompt prefill. See the contract on BlackwellEngine::prefill_status
// (include/blackwell/engine.h): equivalent to a forward_status() sweep over the
// same tokens, minus the lm_head work for the tokens whose logits nobody reads,
// plus chunk-tiling where the resolved token capacity permits it.
// ============================================================================
blackwell::EngineStatus BlackwellEngine::prefill_status(const int* token_ids, int num_tokens,
                                                        int start_pos, float temperature,
                                                        float top_p, int seq_id,
                                                        int* next_token) noexcept {
    if (!token_ids || !next_token || num_tokens <= 0) return EngineStatus::InvalidArgument;
    try {
        auto* impl = pImpl.get();
        // capacity 1 (Continuous, or batch=1 dense) => the run_token branch below
        // for every token; > 1 (Paged dense) => 64-token tiles through run_chunk.
        const size_t cap = std::max<size_t>(1, impl->m_token_capacity);
        int consumed = 0;
        while (consumed < num_tokens) {
            const int n = static_cast<int>(
                std::min<size_t>(cap, static_cast<size_t>(num_tokens - consumed)));
            // Only the final token of the final tile needs logits; every earlier
            // pass skips step_final_ops entirely.
            const bool want_logits = (consumed + n == num_tokens);
            if (n == 1) {
                ENGINE_TRY(impl->run_token(token_ids[consumed], start_pos + consumed, seq_id,
                                           want_logits));
            } else {
                ENGINE_TRY(impl->run_chunk(token_ids + consumed, start_pos + consumed, n,
                                           seq_id, want_logits));
            }
            consumed += n;
        }
        *next_token = sample_top_p(impl->d_logits, impl->m_config.vocab_size, temperature,
                                   top_p);
        return EngineStatus::Success;
    } catch (...) {
        return status_from_current_exception("prefill_status");
    }
}

// ============================================================================
// Evaluation Inference (Для расчета Перплексии)
// ============================================================================
blackwell::EngineStatus BlackwellEngine::forward_eval_status(int token_id, int pos,
                                                             int target_token_id,
                                                             int seq_id,
                                                             float* log_prob) noexcept {
    if (!log_prob) return EngineStatus::InvalidArgument;
    try {
        auto* impl = pImpl.get();
        ENGINE_TRY(impl->run_token(token_id, pos, seq_id));
        *log_prob = compute_log_prob(impl->d_logits, impl->m_config.vocab_size, target_token_id);
        return EngineStatus::Success;
    } catch (...) {
        return status_from_current_exception("forward_eval_status");
    }
}

// TRUE-batch decode facade (exception tier, like fork/rewind -- an orchestration
// entry point, not the noexcept single-stream hot loop). Capability-gates, then
// delegates to Impl::run_decode_batch and repackages the greedy tokens as
// DecodeResults in request order.
std::vector<blackwell::DecodeResult> BlackwellEngine::forward_batch(
    const std::vector<blackwell::DecodeRequest>& requests) {
    // Same gate as fork/rewind: batched decode needs the paged CoW cache (the
    // sibling sequences come from fork()), and a snapshot-able (non-SSM) state.
    require_branching(pImpl->m_caps, "forward_batch");

    const int batch_size = static_cast<int>(requests.size());
    if (batch_size == 0)
        throw std::invalid_argument("BlackwellEngine::forward_batch: empty request batch");

    // Unpack into contiguous host arrays; reject duplicate seq_ids in the same
    // batch (two rows targeting one sequence would double-append it). O(n^2) is
    // fine -- a decode wavefront is a handful of branches.
    std::vector<int> token_ids(batch_size), seqs(batch_size), positions(batch_size);
    for (int b = 0; b < batch_size; ++b) {
        token_ids[b] = requests[b].token_id;
        seqs[b]      = requests[b].seq_id;
        positions[b] = requests[b].pos;
        for (int p = 0; p < b; ++p)
            if (seqs[p] == seqs[b])
                throw std::invalid_argument(
                    "BlackwellEngine::forward_batch: duplicate seq_id " +
                    std::to_string(seqs[b]) + " in the batch (each sequence may "
                    "appear at most once per step)");
    }

    std::vector<int> next_tokens(batch_size, -1);
    const EngineStatus st = pImpl->run_decode_batch(token_ids.data(), seqs.data(),
                                                    positions.data(), batch_size,
                                                    next_tokens.data());
    if (st != EngineStatus::Success)
        throw blackwell::engine_error(
            st, std::string("BlackwellEngine::forward_batch: ") + blackwell::to_string(st));

    std::vector<blackwell::DecodeResult> results(batch_size);
    for (int b = 0; b < batch_size; ++b) {
        results[b].seq_id        = seqs[b];
        results[b].next_token_id = next_tokens[b];
    }
    return results;
}

float BlackwellEngine::last_token_probability(int token_id) const {
    const auto* impl = pImpl.get();
    if (token_id < 0 || static_cast<size_t>(token_id) >= impl->m_config.vocab_size) {
        return 0.0f;
    }
    // compute_log_prob reads the current d_logits (no forward pass) and returns
    // ln P(token). Exponentiate back to a probability and clamp against tiny FP
    // drift so the heatmap always gets a clean [0, 1].
    const float p = std::exp(compute_log_prob(impl->d_logits, impl->m_config.vocab_size, token_id));
    if (p < 0.0f) return 0.0f;
    if (p > 1.0f) return 1.0f;
    return p;
}

// ============================================================================
// Prefix-cache substrate accessors (Paged mode, dense models only).
// ============================================================================
bool BlackwellEngine::has_prefix_cache() const noexcept {
    return pImpl->prefix_cache != nullptr;
}

static void require_prefix_cache(const BlackwellEngine::Impl* impl, const char* op) {
    if (impl->prefix_cache) return;
    throw std::runtime_error(
        std::string("BlackwellEngine::") + op + ": no prefix cache on this engine (" +
        (impl->m_runtime.kv_mode != BlackwellEngine::KVCacheMode::Paged
             ? "construct with KVCacheMode::Paged"
             : "hybrid SSM / gated full-attention models keep state outside the "
               "paged pools and cannot serve cached prefixes") +
        "). Check has_prefix_cache() first.");
}

blackwell::paging::PrefixCacheManager& BlackwellEngine::prefix_cache() {
    require_prefix_cache(pImpl.get(), "prefix_cache");
    return *pImpl->prefix_cache;
}

blackwell::EnginePrefillCoordinator& BlackwellEngine::prefill_driver() {
    require_prefix_cache(pImpl.get(), "prefill_driver");
    return *pImpl->prefill;
}

// ============================================================================
// Inactivity lifecycle (see include/blackwell/engine.h for the contract).
// ============================================================================
int BlackwellEngine::spill_kv_cache() {
    // Deliberately NOT require_prefix_cache: a lifecycle sweep over a model
    // with no substrate is a benign no-op, not a caller error.
    if (!pImpl->prefix_cache) return 0;
    return pImpl->prefix_cache->spill_all();
}

void BlackwellEngine::hibernate() {
    pImpl->arena.hibernate();
}

void BlackwellEngine::wakeup() {
    pImpl->arena.wakeup();
}

bool BlackwellEngine::hibernated() const noexcept {
    return pImpl->arena.hibernated();
}