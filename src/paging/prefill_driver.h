#pragma once
#include <cstdint>
#include <string>
#include <vector>

// ============================================================================
// IPrefillDriver — the engine seam for everything that compiles prompts into
// KV cache (AOTCacheWarmer, the warmup CLI, tests with mocks).
// ============================================================================
// Split out of aot_cache_warmer.h so consumers of the interface (notably
// engine_prefill_coordinator.h, which every engine embedder includes) do not
// inherit the warmer's nlohmann/json + <filesystem> baggage. The warmer's
// header re-includes this one, so existing includers are unaffected.
//
// Production binding: EnginePrefillCoordinator (tokenizer injected + per-token
// prefill sweep over the paged decode path). Tests inject a mock and never
// link CUDA.
// ----------------------------------------------------------------------------
namespace blackwell { namespace paging {

// Same aliases as paged_kv_cache.h / radix_tree.h (consistent redeclaration).
using SeqId   = int32_t;
using TokenId = int32_t;

class IPrefillDriver {
public:
    virtual ~IPrefillDriver() = default;

    // Tokenize a text segment. add_special is true only for the first segment
    // of a root path (BOS etc.) — continuation segments must tokenize as
    // continuations, or the child path's tokens would not extend the parent's.
    virtual std::vector<TokenId> tokenize(const std::string& text,
                                          bool add_special) = 0;

    // Run the engine's chunked prefill for seq over positions
    // [start_pos, num_tokens): prepare_prefill_step + per-layer
    // attention_prefill, writing KV into the pages the PrefixCacheManager
    // pre-seeded/reserved. `tokens` is the FULL path (indexable from 0).
    virtual void prefill(SeqId seq, const TokenId* tokens, int num_tokens,
                         int start_pos) = 0;
};

}} // namespace blackwell::paging
