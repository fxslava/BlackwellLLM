#pragma once
// The COM-style binary boundary of blackwell_core.dll.
//
// This header is the ONLY thing an out-of-tree consumer needs: pure-virtual
// interfaces + C factory exports + POD descriptor structs. It deliberately
// contains no CUDA, no engine internals, no third-party types -- nothing but
// <windows.h> and fixed-width integers -- so any MSVC-built binary (different
// CRT, different exception model) can sit on the other side of the DLL edge.
//
// Boundary contract:
//   - Every method returns an HRESULT; payload comes back through out-params.
//     Internal C++ exceptions NEVER cross the edge -- the implementation
//     (src/core/engine_com.cpp) catches and maps them.
//   - Interfaces are single-owner: Release() destroys the object (no AddRef /
//     shared refcounting -- the engine control plane is single-threaded, and a
//     shareable handle would only invite violations of that doctrine).
//   - The threading doctrine crosses the boundary unchanged: exactly one
//     thread may touch an IBlackwellEngine after creation.
//   - Array/string outputs use the classic two-call protocol: pass a null
//     buffer to receive the required element count via *pCount (S_OK), then
//     call again with a buffer of at least that capacity. A too-small buffer
//     fails with HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER) and still sets
//     *pCount. Strings are UTF-8, length excludes any terminator; a NUL is
//     appended only when the capacity allows it.

#include <windows.h>
#include <cstdint>

#if defined(BLACKWELL_CORE_EXPORTS)
#define BLACKWELL_API __declspec(dllexport)
#else
#define BLACKWELL_API __declspec(dllimport)
#endif

// KV-cache strategy selected at engine creation (mirrors
// BlackwellEngine::KVCacheMode; values are ABI-frozen).
enum BLACKWELL_KV_MODE : int32_t {
    BLACKWELL_KV_MODE_CONTINUOUS = 0,  // legacy FP32 contiguous cache + offloading
    BLACKWELL_KV_MODE_PAGED      = 1,  // bf16 paged cache with CoW fork / rewind
};

// num_gpu_layers sentinel: keep every layer's weights/KV resident in VRAM
// (subject to the BLACKWELL_GPU_LAYERS environment override, like the C++ ctor).
constexpr uint32_t BLACKWELL_ALL_LAYERS_RESIDENT = 0xFFFFFFFFu;

// Engine creation descriptor (the D3D DESC idiom: a POD, not an interface).
struct BLACKWELL_ENGINE_DESC {
    const char* index_path;          // UTF-8 path to model.safetensors.index.json (required)
    uint32_t max_context_length;     // 0 => engine default
    uint32_t num_gpu_layers;         // BLACKWELL_ALL_LAYERS_RESIDENT => all resident
    BLACKWELL_KV_MODE kv_mode;       // Continuous unless branching/prefix cache is needed
};

// ABI-safe mirror of ModelCapabilities (BOOL, not bool -- fixed 4-byte layout).
struct BLACKWELL_CAPABILITIES {
    BOOL supports_cow_branching;      // Fork()/Rewind() permitted
    BOOL requires_ssm_subsystem;      // model has >= 1 linear-attention layer
    BOOL is_hybrid;                   // mixes full and linear attention layers
    int32_t num_full_attention_layers;
    int32_t num_linear_attention_layers;
};

// The engine boundary. Semantics of each method are those of the matching
// BlackwellEngine member (include/blackwell/engine.h) -- capability gating
// included: Fork/Rewind on a non-branching model fail with an HRESULT instead
// of throwing.
struct IBlackwellEngine {
    // Destroys the engine (weights leave VRAM). Single-owner: no AddRef.
    virtual ULONG STDMETHODCALLTYPE Release() = 0;

    virtual HRESULT STDMETHODCALLTYPE GetCapabilities(BLACKWELL_CAPABILITIES* pCaps) = 0;

    // One decode step: sample the next token for `token_id` at `pos`.
    virtual HRESULT STDMETHODCALLTYPE Forward(int32_t token_id, int32_t pos,
                                              float temperature, float top_p,
                                              int32_t seq_id, int32_t* pNextToken) = 0;
    // Evaluation step: ln P(target_token_id | context) for perplexity math.
    virtual HRESULT STDMETHODCALLTYPE ForwardEval(int32_t token_id, int32_t pos,
                                                  int32_t target_token_id, int32_t seq_id,
                                                  float* pLogProb) = 0;
    // P(token_id) from the CURRENT logits; no forward pass (heatmap probe).
    virtual HRESULT STDMETHODCALLTYPE LastTokenProbability(int32_t token_id,
                                                           float* pProbability) = 0;

    // CoW branching (Paged mode on dense models only -- check GetCapabilities).
    virtual HRESULT STDMETHODCALLTYPE Fork(int32_t parent_id, int32_t child_id) = 0;
    virtual HRESULT STDMETHODCALLTYPE Rewind(int32_t seq_id, int32_t pos) = 0;

    // Zero the recurrent (SSM) state before re-prefilling a fresh prompt on a
    // hybrid model. No-op for dense models.
    virtual HRESULT STDMETHODCALLTYPE ResetState(int32_t seq_id) = 0;

    // Inactivity lifecycle. SpillKvCache is a benign no-op (0 pages) on models
    // without a prefix-cache substrate.
    virtual HRESULT STDMETHODCALLTYPE SpillKvCache(int32_t* pPagesSpilled) = 0;
    virtual HRESULT STDMETHODCALLTYPE Hibernate() = 0;
    virtual HRESULT STDMETHODCALLTYPE Wakeup() = 0;
    virtual HRESULT STDMETHODCALLTYPE IsHibernated(BOOL* pHibernated) = 0;
};

// The tokenizer boundary: everything a chat/eval loop needs. Configured
// entirely from the checkpoint directory, like blackwell::TokenizerFactory.
struct IBlackwellTokenizer {
    virtual ULONG STDMETHODCALLTYPE Release() = 0;

    // Text -> ids (two-call protocol; add_special applies the checkpoint's
    // post-processor template, e.g. Llama-3's BOS).
    virtual HRESULT STDMETHODCALLTYPE Encode(const char* pTextUtf8, BOOL add_special,
                                             int32_t* pIds, uint32_t capacity,
                                             uint32_t* pCount) = 0;

    // Incremental chat encoding for a persistent-KV loop: prelude once
    // (BOS + system block), then one message per turn + the generation prompt.
    virtual HRESULT STDMETHODCALLTYPE EncodeChatPrelude(const char* pSystemPromptUtf8,
                                                        int32_t* pIds, uint32_t capacity,
                                                        uint32_t* pCount) = 0;
    virtual HRESULT STDMETHODCALLTYPE EncodeChatMessage(const char* pRoleUtf8,
                                                        const char* pContentUtf8,
                                                        int32_t* pIds, uint32_t capacity,
                                                        uint32_t* pCount) = 0;
    virtual HRESULT STDMETHODCALLTYPE EncodeGenerationPrompt(int32_t* pIds, uint32_t capacity,
                                                             uint32_t* pCount) = 0;

    // Id -> UTF-8 piece (two-call protocol; *pLength excludes the terminator).
    virtual HRESULT STDMETHODCALLTYPE DecodeToken(int32_t token_id, BOOL render_special,
                                                  char* pTextUtf8, uint32_t capacity,
                                                  uint32_t* pLength) = 0;

    virtual HRESULT STDMETHODCALLTYPE IsStop(int32_t token_id, BOOL* pIsStop) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetVocabSize(uint32_t* pSize) = 0;
};

// C factory exports -- the DLL's ONLY named entry points. Creation is
// initialization (weights load inside CreateBlackwellEngine, exactly like the
// C++ constructor); failure returns an HRESULT and *ppOut stays null.
extern "C" {
BLACKWELL_API HRESULT WINAPI CreateBlackwellEngine(const BLACKWELL_ENGINE_DESC* pDesc,
                                                   IBlackwellEngine** ppEngine);
BLACKWELL_API HRESULT WINAPI CreateBlackwellTokenizer(const char* pModelDirUtf8,
                                                      IBlackwellTokenizer** ppTokenizer);
}
