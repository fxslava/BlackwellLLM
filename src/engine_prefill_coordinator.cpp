#include "engine_prefill_coordinator.h"
#include "engine_impl.h"
#include "kv_cache/paged_kv_manager.h"
#include "kernels/sampling.cuh"

namespace blackwell {

namespace {

// Unwind helper for the BIND state: guarantees the engine-id mapping dies on
// any exit path (the mapping is pure host bookkeeping; the sequence itself is
// owned by PrefixCacheManager / the AOT warmer).
class ScopedBinding {
public:
    ScopedBinding(PagedKVManager& kv, paging::SeqId internal)
        : m_kv(kv), m_engine_seq(kv.bind_external(internal)) {}
    ~ScopedBinding() {
        if (m_armed) m_kv.unbind_external(m_engine_seq);
    }
    ScopedBinding(const ScopedBinding&) = delete;
    ScopedBinding& operator=(const ScopedBinding&) = delete;

    SeqId id() const noexcept { return m_engine_seq; }
    // Session survives the scope (prefill_prompt success path).
    SeqId keep() noexcept { m_armed = false; return m_engine_seq; }

private:
    PagedKVManager& m_kv;
    SeqId           m_engine_seq;
    bool            m_armed = true;
};

int delta_blocks(int total_tokens, int cached_tokens) {
    const int total_pages  = (total_tokens + paging::PAGE_SIZE - 1) / paging::PAGE_SIZE;
    const int cached_pages = cached_tokens / paging::PAGE_SIZE;  // page-aligned by contract
    return total_pages - cached_pages;
}

} // namespace

EnginePrefillCoordinator::EnginePrefillCoordinator(BlackwellEngine::Impl& impl,
                                                   PagedKVManager& kv,
                                                   paging::PrefixCacheManager& pc)
    : m_impl(impl), m_kv(kv), m_pc(pc) {
    // The engine only constructs a coordinator on the paged + dense path, but
    // assert the invariants here too so a future composition-root change fails
    // at startup, not as silently-wrong prefix hits.
    if (impl.m_runtime.kv_mode != BlackwellEngine::KVCacheMode::Paged)
        throw std::logic_error(
            "EnginePrefillCoordinator requires KVCacheMode::Paged (the prefix cache "
            "indexes paged KV pool pages)");
    if (impl.m_caps.requires_ssm_subsystem || impl.m_config.attn_output_gate)
        throw std::logic_error(
            "EnginePrefillCoordinator requires a dense uniform full-attention model: "
            "recurrent SSM state / the dedicated full-attention cache live outside "
            "the paged pools, so a cached prefix would skip state the model needs");
}

EnginePrefillCoordinator::~EnginePrefillCoordinator() {
    // Abandoned sessions (caller never called finish()): drop the pins/locks so
    // the tree pages become demotable/evictable again instead of leaking as
    // permanently VRAM-locked.
    for (auto& [engine_seq, internal] : m_sessions) {
        try {
            m_kv.unbind_external(engine_seq);
            m_pc.release(internal);
        } catch (...) {
            // Teardown path: the pools are being destroyed right after us.
        }
    }
}

// ---------------------------------------------------------------------------
// COMPUTE — the delta forward pass. Orchestration per token (all launches on
// the compute stream, in order):
//   1. run_token -> step_embedding: token id -> d_X_accum (FP16 AWQ table).
//   2. prepare_decode_step: reserve the block-table append slot for `pos`
//      (allocates the physical page when the position crosses into a fresh
//      block — the delta's page allocation happens HERE, page by page, after
//      BUDGET guaranteed the allocator can satisfy every one of them).
//   3. per layer: RMSNorm -> AWQ int4 GEMV q/k/v projections (+fused bias) ->
//      attention_decode (RoPE(Q,K), scatter K/V into the reserved page slot,
//      paged-flash attention over the block table) -> o_proj accumulate ->
//      MLP RMSNorm -> AWQ gate/up GEMV -> fused SwiGLU -> down_proj accumulate.
//   4. last token only: final RMSNorm + lm_head GEMV -> d_logits.
// The KV tensors are written straight into the pool pages the block table
// names — commit() later just wraps those pages in VPids. Zero copies.
// ---------------------------------------------------------------------------
void EnginePrefillCoordinator::run_delta(SeqId engine_seq, const TokenId* tokens,
                                         int n, int start_pos, bool want_logits) {
    for (int pos = start_pos; pos < n; ++pos) {
        const bool last = (pos + 1 == n);
        m_impl.run_token(static_cast<int>(tokens[pos]), pos, engine_seq,
                         want_logits && last);
    }
}

EnginePrefillCoordinator::Result
EnginePrefillCoordinator::prefill_prompt(const TokenId* tokens, int n,
                                         cudaStream_t compute_stream) {
    if (tokens == nullptr || n <= 0)
        throw std::invalid_argument("EnginePrefillCoordinator::prefill_prompt: empty prompt");
    if (static_cast<size_t>(n) > m_impl.m_runtime.max_seq_len)
        throw std::invalid_argument(
            "EnginePrefillCoordinator::prefill_prompt: prompt of " + std::to_string(n) +
            " tokens exceeds max_seq_len " + std::to_string(m_impl.m_runtime.max_seq_len));

    // [ACQUIRE] Longest radix prefix restored to VRAM + pinned; sequence seeded
    // with those pages. Graceful by contract: VRAM pressure shrinks
    // cached_tokens instead of failing (we just prefill more).
    const paging::PrefixCacheManager::Acquired a =
        m_pc.acquire(tokens, n, compute_stream);

    try {
        // [BUDGET] The whole delta's pages must be obtainable BEFORE the sweep
        // writes anything: demote cold tree pages, then LRU-evict branches.
        // (acquire() pinned and locked OUR prefix, so it cannot be a victim.)
        const int needed = delta_blocks(n, a.cached_tokens);
        if (!m_pc.ensure_free_pages(needed))
            throw std::runtime_error(
                "EnginePrefillCoordinator: KV OOM — prompt needs " +
                std::to_string(needed) + " free pages beyond the cached prefix, and "
                "the tree is already drained (raise max_seq_len headroom / "
                "paged_branch_factor, or release live sequences)");

        // [BIND] + [COMPUTE] + [COMMIT]
        ScopedBinding binding(m_kv, a.seq);
        run_delta(binding.id(), tokens, n, a.cached_tokens, /*want_logits=*/true);

        // Publish the completed full pages (partial tail stays private). After
        // this the prompt is a zero-prefill hit for every future branch.
        m_pc.commit(a.seq, tokens, n);

        Result r;
        r.engine_seq      = binding.keep();
        r.total_tokens    = n;
        r.cached_tokens   = a.cached_tokens;
        r.computed_tokens = n - a.cached_tokens;
        m_sessions.emplace(r.engine_seq, a.seq);
        return r;
    } catch (...) {
        // Unwind ACQUIRE: unpin + unlock + destroy. Pages the aborted sweep
        // already filled die with the sequence's references; the tree holds
        // only pre-existing state. (~ScopedBinding already unbound the id.)
        m_pc.release(a.seq);
        throw;
    }
}

int EnginePrefillCoordinator::sample_last_logits(float temperature, float top_p) {
    return sample_top_p(m_impl.d_logits, m_impl.m_config.vocab_size, temperature, top_p);
}

void EnginePrefillCoordinator::finish(SeqId engine_seq) {
    auto it = m_sessions.find(engine_seq);
    if (it == m_sessions.end())
        throw std::invalid_argument(
            "EnginePrefillCoordinator::finish: unknown session id " +
            std::to_string(engine_seq));
    m_kv.unbind_external(engine_seq);
    m_pc.release(it->second);
    m_sessions.erase(it);
}

// ---------------------------------------------------------------------------
// IPrefillDriver facet (AOTCacheWarmer / warmup CLI).
// ---------------------------------------------------------------------------
std::vector<EnginePrefillCoordinator::TokenId>
EnginePrefillCoordinator::tokenize(const std::string& text, bool add_special) {
    if (!m_tokenize)
        throw std::logic_error(
            "EnginePrefillCoordinator::tokenize: no tokenizer bound — call "
            "set_tokenizer() with the embedder's tokenizer before compiling a "
            "WarmupSpec (prefill_prompt() takes token ids and does not need one)");
    return m_tokenize(text, add_special);
}

void EnginePrefillCoordinator::prefill(paging::SeqId seq, const TokenId* tokens,
                                       int num_tokens, int start_pos) {
    if (num_tokens <= start_pos) return;   // fully cached path segment
    if (static_cast<size_t>(num_tokens) > m_impl.m_runtime.max_seq_len)
        throw std::invalid_argument(
            "EnginePrefillCoordinator::prefill: path of " + std::to_string(num_tokens) +
            " tokens exceeds max_seq_len " + std::to_string(m_impl.m_runtime.max_seq_len));

    // The warmer acquired/budgets nothing beyond its own pins — make room for
    // the tail's pages the same way prefill_prompt does.
    const int needed = delta_blocks(num_tokens, start_pos);
    if (!m_pc.ensure_free_pages(needed))
        throw std::runtime_error(
            "EnginePrefillCoordinator::prefill: KV OOM while compiling a warmup "
            "branch (" + std::to_string(needed) + " pages needed)");

    // The warmer never samples: skip the lm_head GEMV for the whole tail.
    ScopedBinding binding(m_kv, seq);
    run_delta(binding.id(), tokens, num_tokens, start_pos, /*want_logits=*/false);
}

} // namespace blackwell
