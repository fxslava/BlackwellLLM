#pragma once
#include "ikv_cache_manager.h"
#include "blackwell/config.h"          // ModelConfig
#include "paging/paged_kv_cache.h"     // paging::SequenceManager
#include <unordered_map>
#include <memory>

namespace blackwell {

// Paged adapter: wraps the verified bf16 paging::SequenceManager + the
// paged-flash-attention kernel, adding Copy-on-Write fork / rewind for ReAct
// tree-search. Decode flow (per the IKVCacheManager contract):
//   prepare_decode_step(seq,pos)  -> reserve the append slot (CoW if the target
//                                    page is fork-shared) and stage the block
//                                    table; LATCH the per-token context.
//   attention_decode(L,...) x N   -> RoPE(Q,K) -> paged append -> paged-flash
//                                    attention over the block table, per layer.
//
// EngineSeqId (caller-assigned, public) is bridged to the SequenceManager's
// internally-generated ids via m_id_map; engine seq 0 is created at construction.
class PagedKVManager : public IKVCacheManager {
public:
    // num_gpu_layers: leading layers whose paged KV stays resident in VRAM; the
    // rest are KV-offloaded to a pinned host mirror inside the SequenceManager and
    // streamed per layer during decode. SIZE_MAX (default) keeps every layer
    // resident (the original behaviour). Mirrors the weight-offload split the
    // VRAMArena applies, so paged mode matches continuous mode's VRAM ceiling.
    // branch_factor inflates the host-mirror page pool to leave headroom for
    // concurrent CoW fork branches (RuntimeConfig::paged_branch_factor).
    // min_total_pages is a FLOOR on the pool (RuntimeConfig::kv_vram_cache_pages):
    // pages beyond the sequence budget stay available to the prefix cache, so
    // cached branches can remain VRAM-resident instead of being demoted the
    // moment live sequences fill their share. 0 = sequence budget only.
    PagedKVManager(const ModelConfig& config, size_t max_seq_len,
                   size_t num_gpu_layers = static_cast<size_t>(-1),
                   int branch_factor = 4, int min_total_pages = 0);

    void prepare_decode_step(SeqId seq, int pos) override;
    void prepare_prefill_step(SeqId seq, int start_pos, int num_tokens) override;

    void attention_decode(int layer_idx, int pos,
                          float* d_Q, float* d_K, float* d_V, float* d_O) override;
    void attention_prefill(int layer_idx, int start_pos, int num_tokens,
                           float* d_Q, float* d_K, float* d_V, float* d_O) override;

    void* get_layer_k_ptr(int layer_idx) override;
    void* get_layer_v_ptr(int layer_idx) override;

    void fork(SeqId parent, SeqId child) override;
    void rewind(SeqId seq, int target_pos) override;

    const char* name() const override { return "paged"; }
    bool supports_branching() const override { return true; }

    // --- external sequence adoption (prefix-cache seam) -------------------
    // PrefixCacheManager::acquire() creates sequences DIRECTLY in the
    // SequenceManager (seeded with radix-tree prefix pages), bypassing this
    // adapter's engine-id map. bind_external() grafts such an internal id into
    // the map so the standard decode/prefill control plane can drive it, and
    // returns the engine id to pass as forward()'s seq_id. Ids are allocated
    // from a RESERVED NEGATIVE range (descending from -2) so they can never
    // collide with caller-assigned fork ids, which must be >= 0.
    // unbind_external() drops the mapping only — destroying the sequence
    // remains PrefixCacheManager::release()'s job (it also unpins/unlocks).
    SeqId bind_external(paging::SeqId internal);
    void  unbind_external(SeqId engine_seq);

    // The shared page pool + block tables. The prefix-cache substrate
    // (TieredMemoryPager, PrefixCacheManager, KVBranchSerializer) MUST be built
    // over this exact instance: its pages are the ones the attention kernels
    // read and write.
    paging::SequenceManager& sequence_manager() { return *m_seqmgr; }

private:
    paging::SeqId internal_id(SeqId engine_seq) const;

    // --- KV-cache offloading groundwork ----------------------------------
    // Each managed sequence is either GPU-RESIDENT (its pages live in the device
    // pool) or SWAPPED (pages mirrored to host RAM, device pages returned to the
    // allocator for reuse by active sequences). The decode path routes through
    // ensure_resident() BEFORE touching any device page, so a suspended /
    // low-priority sequence is faulted back in rather than read through a stale
    // block table -- the seam that keeps multi-sequence execution from issuing
    // CUDA faults on swapped state. The actual page<->host transfers
    // (swap_out / swap_in + free-list interaction) are the next phase; today
    // every sequence stays Resident and ensure_resident is the guard.
    enum class SeqResidency { Resident, Swapped };
    void ensure_resident(paging::SeqId internal);

    const ModelConfig& m_config;
    std::unique_ptr<paging::SequenceManager> m_seqmgr;
    std::unordered_map<SeqId, paging::SeqId>     m_id_map;     // engine id -> internal id
    std::unordered_map<paging::SeqId, SeqResidency> m_residency;
    SeqId m_next_external = -2;   // bind_external id source (reserved: < 0)

    // Cached geometry (avoids re-reading m_config on the hot path).
    int   m_num_q_heads;
    int   m_num_kv_heads;
    int   m_head_dim;
    float m_rope_theta;

    // Per-token context latched by prepare_decode_step, consumed per layer.
    paging::SeqId  m_active        = -1;
    const int32_t* m_block_table   = nullptr;
    int            m_seq_len       = 0;     // tokens incl. the current one
    int            m_append_page   = -1;
    int            m_append_slot   = -1;
};

} // namespace blackwell
