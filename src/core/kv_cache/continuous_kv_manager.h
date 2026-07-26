#pragma once
#include "ikv_cache_manager.h"
#include "memory_pool.h"        // VRAMArena
#include "blackwell/config.h"   // ModelConfig

namespace blackwell {

// Legacy adapter: wraps the FP32 contiguous KV cache + VRAMArena layer
// offloading. attention_decode reproduces the original step_attention_math
// kernel sequence byte-for-byte, so routing BlackwellEngine through
// IKVCacheManager is a no-behavior-change refactor and stays the FP32 parity
// reference. CoW branching (fork) and chunked prefill are unsupported on the
// contiguous cache and throw std::runtime_error; rewind() IS supported as a
// single-sequence linear rollback (delegates to VRAMArena::truncate_kv to
// reconcile the offload high-water marks with the rewound position).
class ContinuousKVManager : public IKVCacheManager {
public:
    ContinuousKVManager(VRAMArena& arena, const ModelConfig& config)
        : m_arena(arena), m_config(config) {}

    void prepare_decode_step(SeqId seq, int pos) override;
    void prepare_prefill_step(SeqId seq, int start_pos, int num_tokens) override;

    void attention_decode(int layer_idx, int pos,
                          float* d_Q, float* d_K, float* d_V, float* d_O) override;
    void attention_prefill(int layer_idx, int start_pos, int num_tokens,
                           float* d_Q, float* d_K, float* d_V, float* d_O) override;

    void* get_layer_k_ptr(int layer_idx) override { return m_arena.get_layer_k_cache(layer_idx); }
    void* get_layer_v_ptr(int layer_idx) override { return m_arena.get_layer_v_cache(layer_idx); }

    void fork(SeqId parent, SeqId child) override;
    void rewind(SeqId seq, int target_pos) override;

    const char* name() const override { return "continuous"; }
    bool supports_branching() const override { return false; }

private:
    VRAMArena&         m_arena;
    const ModelConfig& m_config;
};

} // namespace blackwell
