// The DLL side of the COM-style boundary declared in
// include/blackwell/iblackwell_engine.h: concrete IBlackwellEngine /
// IBlackwellTokenizer wrappers over the C++ facade, plus the exported C
// factories. This is the ONLY translation unit compiled into the SHARED
// target itself (everything else arrives via blackwell_core_obj), and the
// only place where __declspec(dllexport) appears.
//
// Boundary rule: no exception escapes a method on these interfaces. Every
// body runs under boundary(), which catches and maps to an HRESULT; the
// mapping is intentionally coarse for now.
#include "blackwell/iblackwell_engine.h"

#include "blackwell/chat_template.h"
#include "blackwell/engine.h"
#include "blackwell/engine_status.h"
#include "blackwell/tokenizer.h"

#include "common.h"  // blackwell::cuda_error (init-tier CUDA failures)

#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

// Hybrid error doctrine: RUNTIME-tier statuses map here, once, at the edge.
HRESULT hresult_from_status(blackwell::EngineStatus status, const char* op) noexcept {
    switch (status) {
        case blackwell::EngineStatus::Success:
            return S_OK;
        default:
            std::cerr << "[blackwell_core] " << op << ": "
                      << blackwell::to_string(status) << "\n";
            switch (status) {
                case blackwell::EngineStatus::OutOfVram:        return E_OUTOFMEMORY;
                case blackwell::EngineStatus::InvalidArgument:  return E_INVALIDARG;
                case blackwell::EngineStatus::InvalidConfig:    return E_INVALIDARG;
                case blackwell::EngineStatus::StateMismatch:    return E_NOT_VALID_STATE;
                case blackwell::EngineStatus::CudaRuntimeError:
                default:                                        return BLACKWELL_E_CUDA_RUNTIME;
            }
    }
}

// Rethrow-and-map: called ONLY from a catch context, for the INIT tier
// (factories, lifecycle) and as the last-resort panic net around the status
// tier. Logs to stderr because the boundary has no richer channel yet.
HRESULT map_current_exception(const char* op) noexcept {
    try {
        throw;
    } catch (const blackwell::cuda_error& e) {
        // INIT-tier CUDA failure (CUDA_CHECK_THROW: DeviceBuffer/ctor paths).
        std::cerr << "[blackwell_core] " << op << ": " << e.what() << "\n";
        return (e.code() == cudaErrorMemoryAllocation) ? E_OUTOFMEMORY
                                                       : BLACKWELL_E_CUDA_RUNTIME;
    } catch (const blackwell::engine_error& e) {
        // A status-tier failure re-raised by an exception-tier wrapper below us.
        std::cerr << "[blackwell_core] " << op << ": " << e.what() << "\n";
        return hresult_from_status(e.status(), op);
    } catch (const std::bad_alloc&) {
        std::cerr << "[blackwell_core] " << op << ": out of memory\n";
        return E_OUTOFMEMORY;
    } catch (const std::invalid_argument& e) {
        std::cerr << "[blackwell_core] " << op << ": " << e.what() << "\n";
        return E_INVALIDARG;
    } catch (const std::out_of_range& e) {
        std::cerr << "[blackwell_core] " << op << ": " << e.what() << "\n";
        return E_BOUNDS;
    } catch (const std::exception& e) {
        std::cerr << "[blackwell_core] " << op << ": " << e.what() << "\n";
        return E_FAIL;
    } catch (...) {
        std::cerr << "[blackwell_core] " << op << ": unknown exception\n";
        return E_FAIL;
    }
}

template <typename Body>
HRESULT boundary(const char* op, Body&& body) noexcept {
    try {
        return body();
    } catch (...) {
        return map_current_exception(op);
    }
}

// Two-call protocol for id arrays (see the header contract): null buffer =>
// size query; short buffer => ERROR_INSUFFICIENT_BUFFER; *pCount always set.
HRESULT copy_ids(const std::vector<int>& src, int32_t* pIds, uint32_t capacity,
                 uint32_t* pCount) {
    if (!pCount) return E_POINTER;
    *pCount = static_cast<uint32_t>(src.size());
    if (!pIds) return S_OK;
    if (capacity < src.size()) return HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER);
    for (size_t i = 0; i < src.size(); ++i) pIds[i] = static_cast<int32_t>(src[i]);
    return S_OK;
}

// Same protocol for UTF-8 strings; NUL is appended only when it fits.
HRESULT copy_utf8(const std::string& src, char* pText, uint32_t capacity,
                  uint32_t* pLength) {
    if (!pLength) return E_POINTER;
    *pLength = static_cast<uint32_t>(src.size());
    if (!pText) return S_OK;
    if (capacity < src.size()) return HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER);
    std::memcpy(pText, src.data(), src.size());
    if (capacity > src.size()) pText[src.size()] = '\0';
    return S_OK;
}

// ---------------------------------------------------------------------------
// IBlackwellEngine over the C++ facade. Single-owner (Release deletes); the
// single-threaded control-plane doctrine is inherited verbatim.
// ---------------------------------------------------------------------------
class EngineCom final : public IBlackwellEngine {
public:
    explicit EngineCom(std::unique_ptr<BlackwellEngine> engine)
        : engine_(std::move(engine)) {}

    ULONG STDMETHODCALLTYPE Release() override {
        delete this;
        return 0;
    }

    HRESULT STDMETHODCALLTYPE GetCapabilities(BLACKWELL_CAPABILITIES* pCaps) override {
        if (!pCaps) return E_POINTER;
        return boundary("GetCapabilities", [&] {
            const ModelCapabilities c = engine_->get_capabilities();
            pCaps->supports_cow_branching      = c.supports_cow_branching ? TRUE : FALSE;
            pCaps->requires_ssm_subsystem      = c.requires_ssm_subsystem ? TRUE : FALSE;
            pCaps->is_hybrid                   = c.is_hybrid ? TRUE : FALSE;
            pCaps->num_full_attention_layers   = c.num_full_attention_layers;
            pCaps->num_linear_attention_layers = c.num_linear_attention_layers;
            return S_OK;
        });
    }

    HRESULT STDMETHODCALLTYPE Forward(int32_t token_id, int32_t pos, float temperature,
                                      float top_p, int32_t seq_id,
                                      int32_t* pNextToken) override {
        if (!pNextToken) return E_POINTER;
        // RUNTIME tier: the status returned by the hot loop translates straight
        // to an HRESULT -- no exception round-trip. boundary() remains as the
        // last-resort panic net (unmigrated subsystems may still throw).
        return boundary("Forward", [&] {
            int next = -1;
            const auto st = engine_->forward_status(token_id, pos, temperature, top_p,
                                                    seq_id, &next);
            if (st != blackwell::EngineStatus::Success)
                return hresult_from_status(st, "Forward");
            *pNextToken = next;
            return S_OK;
        });
    }

    HRESULT STDMETHODCALLTYPE ForwardEval(int32_t token_id, int32_t pos,
                                          int32_t target_token_id, int32_t seq_id,
                                          float* pLogProb) override {
        if (!pLogProb) return E_POINTER;
        return boundary("ForwardEval", [&] {
            float log_prob = 0.0f;
            const auto st = engine_->forward_eval_status(token_id, pos, target_token_id,
                                                         seq_id, &log_prob);
            if (st != blackwell::EngineStatus::Success)
                return hresult_from_status(st, "ForwardEval");
            *pLogProb = log_prob;
            return S_OK;
        });
    }

    HRESULT STDMETHODCALLTYPE LastTokenProbability(int32_t token_id,
                                                   float* pProbability) override {
        if (!pProbability) return E_POINTER;
        return boundary("LastTokenProbability", [&] {
            *pProbability = engine_->last_token_probability(token_id);
            return S_OK;
        });
    }

    HRESULT STDMETHODCALLTYPE Fork(int32_t parent_id, int32_t child_id) override {
        return boundary("Fork", [&] {
            engine_->fork(parent_id, child_id);
            return S_OK;
        });
    }

    HRESULT STDMETHODCALLTYPE Rewind(int32_t seq_id, int32_t pos) override {
        return boundary("Rewind", [&] {
            engine_->rewind(seq_id, pos);
            return S_OK;
        });
    }

    HRESULT STDMETHODCALLTYPE ResetState(int32_t seq_id) override {
        return boundary("ResetState", [&] {
            engine_->reset_state(seq_id);
            return S_OK;
        });
    }

    HRESULT STDMETHODCALLTYPE SpillKvCache(int32_t* pPagesSpilled) override {
        if (!pPagesSpilled) return E_POINTER;
        return boundary("SpillKvCache", [&] {
            *pPagesSpilled = engine_->spill_kv_cache();
            return S_OK;
        });
    }

    HRESULT STDMETHODCALLTYPE Hibernate() override {
        return boundary("Hibernate", [&] {
            engine_->hibernate();
            return S_OK;
        });
    }

    HRESULT STDMETHODCALLTYPE Wakeup() override {
        return boundary("Wakeup", [&] {
            engine_->wakeup();
            return S_OK;
        });
    }

    HRESULT STDMETHODCALLTYPE IsHibernated(BOOL* pHibernated) override {
        if (!pHibernated) return E_POINTER;
        *pHibernated = engine_->hibernated() ? TRUE : FALSE;
        return S_OK;
    }

private:
    std::unique_ptr<BlackwellEngine> engine_;
};

// ---------------------------------------------------------------------------
// IBlackwellTokenizer over blackwell::ITokenizer.
// ---------------------------------------------------------------------------
class TokenizerCom final : public IBlackwellTokenizer {
public:
    explicit TokenizerCom(std::unique_ptr<blackwell::ITokenizer> tok)
        : tok_(std::move(tok)) {}

    ULONG STDMETHODCALLTYPE Release() override {
        delete this;
        return 0;
    }

    HRESULT STDMETHODCALLTYPE Encode(const char* pTextUtf8, BOOL add_special,
                                     int32_t* pIds, uint32_t capacity,
                                     uint32_t* pCount) override {
        if (!pTextUtf8) return E_INVALIDARG;
        return boundary("Encode", [&] {
            return copy_ids(tok_->encode(pTextUtf8, add_special != FALSE), pIds, capacity,
                            pCount);
        });
    }

    HRESULT STDMETHODCALLTYPE EncodeChatPrelude(const char* pSystemPromptUtf8,
                                                int32_t* pIds, uint32_t capacity,
                                                uint32_t* pCount) override {
        if (!pSystemPromptUtf8) return E_INVALIDARG;
        return boundary("EncodeChatPrelude", [&] {
            return copy_ids(tok_->encode_chat_prelude(pSystemPromptUtf8), pIds, capacity,
                            pCount);
        });
    }

    HRESULT STDMETHODCALLTYPE EncodeChatMessage(const char* pRoleUtf8,
                                                const char* pContentUtf8, int32_t* pIds,
                                                uint32_t capacity,
                                                uint32_t* pCount) override {
        if (!pRoleUtf8 || !pContentUtf8) return E_INVALIDARG;
        return boundary("EncodeChatMessage", [&] {
            return copy_ids(tok_->encode_chat_message({pRoleUtf8, pContentUtf8}), pIds,
                            capacity, pCount);
        });
    }

    HRESULT STDMETHODCALLTYPE EncodeGenerationPrompt(int32_t* pIds, uint32_t capacity,
                                                     uint32_t* pCount) override {
        return boundary("EncodeGenerationPrompt", [&] {
            return copy_ids(tok_->encode_generation_prompt(), pIds, capacity, pCount);
        });
    }

    HRESULT STDMETHODCALLTYPE DecodeToken(int32_t token_id, BOOL render_special,
                                          char* pTextUtf8, uint32_t capacity,
                                          uint32_t* pLength) override {
        return boundary("DecodeToken", [&] {
            return copy_utf8(tok_->decode(token_id, render_special != FALSE), pTextUtf8,
                             capacity, pLength);
        });
    }

    HRESULT STDMETHODCALLTYPE IsStop(int32_t token_id, BOOL* pIsStop) override {
        if (!pIsStop) return E_POINTER;
        *pIsStop = tok_->is_stop(token_id) ? TRUE : FALSE;
        return S_OK;
    }

    HRESULT STDMETHODCALLTYPE GetVocabSize(uint32_t* pSize) override {
        if (!pSize) return E_POINTER;
        *pSize = static_cast<uint32_t>(tok_->vocab_size());
        return S_OK;
    }

private:
    std::unique_ptr<blackwell::ITokenizer> tok_;
};

} // namespace

// ---------------------------------------------------------------------------
// Exported C factories -- the DLL's only named entry points.
// ---------------------------------------------------------------------------
extern "C" HRESULT WINAPI CreateBlackwellEngine(const BLACKWELL_ENGINE_DESC* pDesc,
                                                IBlackwellEngine** ppEngine) {
    if (!ppEngine) return E_POINTER;
    *ppEngine = nullptr;
    if (!pDesc || !pDesc->index_path) return E_INVALIDARG;

    return boundary("CreateBlackwellEngine", [&]() -> HRESULT {
        // Creation IS initialization: the C++ ctor loads weights (RAII), so a
        // failed load surfaces here as an HRESULT, never as a half-built engine.
        const size_t max_seq_len =
            pDesc->max_context_length ? pDesc->max_context_length : 2048;
        const size_t num_gpu_layers =
            (pDesc->num_gpu_layers == BLACKWELL_ALL_LAYERS_RESIDENT)
                ? static_cast<size_t>(-1)
                : pDesc->num_gpu_layers;
        const auto kv_mode = (pDesc->kv_mode == BLACKWELL_KV_MODE_PAGED)
                                 ? BlackwellEngine::KVCacheMode::Paged
                                 : BlackwellEngine::KVCacheMode::Continuous;

        auto engine = std::make_unique<BlackwellEngine>(pDesc->index_path, max_seq_len,
                                                        num_gpu_layers, kv_mode);
        *ppEngine = new EngineCom(std::move(engine));
        return S_OK;
    });
}

extern "C" HRESULT WINAPI CreateBlackwellTokenizer(const char* pModelDirUtf8,
                                                   IBlackwellTokenizer** ppTokenizer) {
    if (!ppTokenizer) return E_POINTER;
    *ppTokenizer = nullptr;
    if (!pModelDirUtf8) return E_INVALIDARG;

    return boundary("CreateBlackwellTokenizer", [&]() -> HRESULT {
        auto tok = blackwell::TokenizerFactory::create(pModelDirUtf8);
        *ppTokenizer = new TokenizerCom(std::move(tok));
        return S_OK;
    });
}
