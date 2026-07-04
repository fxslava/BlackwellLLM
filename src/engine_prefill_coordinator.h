#pragma once
#include "blackwell/engine.h"
#include "kv_cache/ikv_cache_manager.h"      // blackwell::SeqId (engine ids)
#include "paging/prefix_cache_manager.h"     // PrefixCacheManager, paging::SeqId/TokenId
#include "paging/aot_cache_warmer.h"         // paging::IPrefillDriver

#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// ============================================================================
// EnginePrefillCoordinator — the production IPrefillDriver: prompt tokens in,
// decode-ready sequence out, prefix cache consulted and repaid.
// ============================================================================
// This is the Phase-3 seam docs/TIERED_KV_AND_AOT.md §5 calls "engine wiring +
// production IPrefillDriver": it binds the paging substrate (PrefixCacheManager
// over the TieredMemoryPager) to the live BlackwellEngine decode pipeline, so a
// prompt pays GPU time only for the tokens the radix tree has never seen.
//
// One prompt = one pass of this state machine (prefill_prompt):
//
//   ┌─────────┐  radix match + page-fault + pin      (PrefixCacheManager)
//   │ ACQUIRE │  cached_tokens = page-aligned prefix now VRAM-resident;
//   └────┬────┘  STRICTLY < n — the last token always re-runs for logits.
//   ┌────▼────┐  ensure_free_pages(delta blocks): demote cold tree pages,
//   │ BUDGET  │  then LRU-evict branches; false => real OOM, throw BEFORE
//   └────┬────┘  any partial KV is written.
//   ┌────▼────┐  graft the acquire()d internal sequence into PagedKVManager's
//   │  BIND   │  id map (reserved negative engine id) so the standard
//   └────┬────┘  prepare_decode_step / attention control plane can drive it.
//   ┌────▼────┐  per delta token: embedding -> AWQ QKV GEMV -> RoPE + paged
//   │ COMPUTE │  append + paged-flash attention -> o_proj -> SwiGLU MLP.
//   └────┬────┘  Final norm + lm_head GEMV run ONLY for the last token.
//   ┌────▼────┐  commit(): wrap the new full pages in VPids, insert into the
//   │ COMMIT  │  radix tree (duplicate spans dedup + gc), re-lock the deeper
//   └────┬────┘  node. The prompt is now zero-prefill for future branches.
//        ▼
//     Result{engine_seq,...} — decode continues via forward(tok, pos, ..,
//     engine_seq); logits for tokens[n-1] are live (sample_last_logits()).
//     finish(engine_seq) unbinds + releases (unpin/unlock/destroy) when the
//     generation is over; committed pages survive in the tree.
//
// Failure contract: any throw after ACQUIRE unwinds BIND and releases the
// sequence (unpin + unlock + destroy) before rethrowing — no leaked pins, no
// dangling engine-id mappings, and the tree keeps only what commit() published
// (nothing, on failure). The KV pages the aborted sweep already wrote die with
// the sequence's page references, exactly like an abandoned decode.
//
// COMPUTE today is a per-token sweep over the engine's single-token GEMV
// pipeline (Impl::run_token): correct and zero-copy, but O(n) kernel launches.
// The batched Tensor-Core prefill (prepare_prefill_step + attention_prefill
// on IKVCacheManager) is a drop-in replacement inside run_delta() once the
// multi-token activation buffers and kernels exist — callers see no change.
//
// Scope guards (checked at construction): Paged KV mode on a DENSE uniform
// full-attention model. Hybrid SSM checkpoints keep recurrent state outside
// the paged pools and the gated head_dim-256 layers use a dedicated continuous
// cache — for both, a radix "prefix hit" would silently skip state the model
// needs, so the engine simply does not construct a coordinator for them.
//
// Threading: single-threaded host control plane, like the whole substrate.
// ----------------------------------------------------------------------------
namespace blackwell {

class PagedKVManager;

class EnginePrefillCoordinator final : public paging::IPrefillDriver {
public:
    using TokenId = paging::TokenId;
    // Bound by the embedder (playground adapter / overlay / warmup CLI) — the
    // core engine deliberately does not own a tokenizer instance.
    using TokenizeFn =
        std::function<std::vector<TokenId>(const std::string& text, bool add_special)>;

    // All three referents outlive the coordinator: they are siblings inside
    // BlackwellEngine::Impl, which constructs the coordinator last.
    EnginePrefillCoordinator(BlackwellEngine::Impl& impl, PagedKVManager& kv,
                             paging::PrefixCacheManager& pc);

    EnginePrefillCoordinator(const EnginePrefillCoordinator&) = delete;
    EnginePrefillCoordinator& operator=(const EnginePrefillCoordinator&) = delete;
    ~EnginePrefillCoordinator() override;

    // -----------------------------------------------------------------------
    // The prompt-level state machine (see the header comment).
    // -----------------------------------------------------------------------
    struct Result {
        SeqId engine_seq      = -1;  // pass as forward()'s seq_id to decode
        int   total_tokens    = 0;   // n
        int   cached_tokens   = 0;   // served from the radix tree (zero GPU work)
        int   computed_tokens = 0;   // delta that went through the forward pass
    };

    // Run ACQUIRE..COMMIT for tokens[0..n). On return the sequence is live
    // (pinned prefix, private tail) and d_logits holds the distribution for
    // tokens[n-1]. compute_stream: non-null lets page-fault DMA overlap the
    // delta's kernels (GPU-side fence); null blocks the host on the fault.
    Result prefill_prompt(const TokenId* tokens, int n,
                          cudaStream_t compute_stream = nullptr);

    // Sample the first generated token from the logits prefill_prompt left
    // behind — avoids re-running tokens[n-1] through forward().
    int sample_last_logits(float temperature = 0.6f, float top_p = 0.9f);

    // End a prefill_prompt() session: unbind the engine id and release the
    // sequence (unpin + unlock + destroy). Committed pages stay in the tree at
    // whatever tier the pager later decides. Idempotent per id; throws on ids
    // this coordinator never issued.
    void finish(SeqId engine_seq);

    // -----------------------------------------------------------------------
    // IPrefillDriver — the AOTCacheWarmer / warmup-CLI facet. The warmer owns
    // acquire/commit/release itself and hands us the INTERNAL paging seq id;
    // we only bind it, run the delta sweep (no logits needed), and unbind.
    // -----------------------------------------------------------------------
    std::vector<TokenId> tokenize(const std::string& text, bool add_special) override;
    void prefill(paging::SeqId seq, const TokenId* tokens, int num_tokens,
                 int start_pos) override;

    void set_tokenizer(TokenizeFn fn) { m_tokenize = std::move(fn); }
    bool has_tokenizer() const noexcept { return static_cast<bool>(m_tokenize); }

    paging::PrefixCacheManager& prefix_cache() noexcept { return m_pc; }

private:
    // COMPUTE: sweep tokens[start_pos..n) through the decode pipeline on the
    // bound engine sequence. want_logits gates the final-norm + lm_head GEMV
    // (only the last token of an interactive prompt needs them).
    void run_delta(SeqId engine_seq, const TokenId* tokens, int n, int start_pos,
                   bool want_logits);

    BlackwellEngine::Impl&      m_impl;
    PagedKVManager&             m_kv;
    paging::PrefixCacheManager& m_pc;
    TokenizeFn                  m_tokenize;

    // Live prefill_prompt sessions: engine id -> internal paging id, so
    // finish() can release without the caller tracking both.
    std::unordered_map<SeqId, paging::SeqId> m_sessions;
};

} // namespace blackwell
