#pragma once
#include "blackwell/engine.h"
#include "kv_cache/ikv_cache_manager.h"      // blackwell::SeqId (engine ids)
#include "paging/prefix_cache_manager.h"     // PrefixCacheManager, paging::SeqId/TokenId
#include "paging/prefill_driver.h"           // paging::IPrefillDriver

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
// Failure contract (hybrid, mirrors the engine's error doctrine):
//   - COMPUTE failures are STATUS-tier: run_delta() never throws — a failed
//     decode step gracefully halts the sweep, the coordinator unwinds BIND /
//     rolls back exactly as below, and the failure comes back as an
//     EngineStatus in the Result/UpdateStats/EngineSequence it returns. The
//     live UI layer (overlay tracker) sees a faulted stream, never an unwind.
//   - Session-admin failures (ACQUIRE/BUDGET/COMMIT: KV OOM, dead handles)
//     stay exception-tier: any throw after ACQUIRE unwinds BIND and releases
//     the sequence (unpin + unlock + destroy) before rethrowing — no leaked
//     pins, no dangling engine-id mappings, and the tree keeps only what
//     commit() published (nothing, on failure). The KV pages an aborted sweep
//     already wrote die with the sequence's page references either way.
//
// COMPUTE processes the uncached delta in BATCHED Tensor-Core chunks
// (Impl::run_chunk): per forward pass it widens embedding / RMSNorm /
// projections / paged-flash attention / MLP to up to Impl::m_token_capacity
// token rows, tiling longer deltas. It reproduces the old per-token
// Impl::run_token sweep numerically (the quantized projections are the same
// per-row GEMV, RoPE / KV-append / attention the same kernels per position), so
// callers see identical logits — just far fewer kernel launches. run_delta falls
// back to the run_token sweep only for a degenerate token capacity of 1.
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
        SeqId engine_seq      = -1;  // pass as forward_status()'s seq_id; -1 => faulted
        int   total_tokens    = 0;   // n
        int   cached_tokens   = 0;   // served from the radix tree (zero GPU work)
        int   computed_tokens = 0;   // delta that went through the forward pass
        // COMPUTE outcome (status tier). On failure engine_seq stays -1 and the
        // sequence was already released -- nothing to finish().
        EngineStatus status = EngineStatus::Success;
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
    // Continuous Speculative Tracking (micro-rewinds).
    //
    // BPE tokenizers re-segment as characters arrive ("t" -> "th" -> "the"
    // produce different token boundaries each keystroke), so a live tracker
    // cannot append — it must DIFF the fresh tokenization against what the KV
    // cache holds, roll back to the divergence point, and recompute only the
    // new suffix. EngineSequence is the session handle for that loop: it
    // carries the host-side token mirror the diff runs against (the KV cache
    // itself has no token identity — the mirror IS the source of truth for
    // what each position holds).
    //
    // Policy: update_sequence does NOT commit to the radix tree. Keystroke
    // states are transient — publishing every intermediate tokenization would
    // fill the tree with pages that are superseded milliseconds later and
    // survive as evictable-but-resident garbage. Call commit_sequence() at
    // stable boundaries (debounce timeout, word break, request submit); the
    // tree's duplicate-span dedup makes repeated commits of the same prefix
    // free.
    // -----------------------------------------------------------------------
    struct EngineSequence {
        SeqId engine_seq = -1;         // forward_status()'s seq_id while live
        std::vector<TokenId> tokens;   // host mirror of the KV cache content
        // Last COMPUTE outcome for this stream (status tier). A failed
        // begin_sequence returns an invalid handle carrying the reason here.
        EngineStatus status = EngineStatus::Success;
        bool valid() const noexcept { return engine_seq != -1; }
    };

    struct UpdateStats {
        int reused_tokens    = 0;  // longest common prefix kept in the KV cache
        int truncated_tokens = 0;  // old tail rolled back
        int computed_tokens  = 0;  // new suffix run through the forward pass
        int freed_pages      = 0;  // physical pages the rollback returned
        // COMPUTE outcome. On failure the sequence was rolled back to the
        // common prefix and the mirror shrunk to match: the session stays
        // coherent and usable (self-heals on the next reconcile).
        EngineStatus status = EngineStatus::Success;
    };

    // prefill_prompt + a tracking handle (the initial prompt IS committed —
    // it is the stable prefix the tracker diverges from).
    EngineSequence begin_sequence(const std::vector<TokenId>& tokens,
                                  cudaStream_t compute_stream = nullptr);

    // Reconcile the live sequence with a fresh tokenization:
    //   diff -> truncate_sequence(divergence point) -> budget -> delta compute.
    // The last token ALWAYS re-runs (even on a pure append or an identical
    // retokenization) so d_logits matches new_tokens on return, mirroring
    // acquire()'s cached_tokens < n rule. On failure the sequence is rolled
    // back to the common prefix and the mirror shrunk to match — the session
    // stays coherent and usable. Throws std::invalid_argument on an empty /
    // oversized tokenization or a dead handle.
    UpdateStats update_sequence(EngineSequence& seq,
                                const std::vector<TokenId>& new_tokens);

    // Publish the sequence's CURRENT tokens to the radix tree (full pages
    // only, exactly like prefill_prompt's COMMIT phase). Call at stable
    // tracking boundaries.
    void commit_sequence(const EngineSequence& seq);

    // finish() for a tracking handle; poisons it against reuse.
    void finish(EngineSequence& seq);

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
    // Status tier: NEVER throws -- a failed step halts the sweep and returns
    // its status (stray subsystem exceptions are converted inside).
    EngineStatus run_delta(SeqId engine_seq, const TokenId* tokens, int n,
                           int start_pos, bool want_logits) noexcept;

    BlackwellEngine::Impl&      m_impl;
    PagedKVManager&             m_kv;
    paging::PrefixCacheManager& m_pc;
    TokenizeFn                  m_tokenize;

    // Live prefill_prompt sessions: engine id -> internal paging id, so
    // finish() can release without the caller tracking both.
    std::unordered_map<SeqId, paging::SeqId> m_sessions;
};

} // namespace blackwell
