// =============================================================================
// bridge/engine_api.cpp — C-ABI implementation of include/bridge/engine_api.h.
//
// Every entry point: (1) validates the opaque handle(s), (2) delegates to the
// internal impl class / IEngineControl seam, (3) translates any C++ exception or
// blackwell::EngineStatus into a BridgeStatus. NO C++ exception ever crosses the
// extern "C" boundary. The producer path (push_pcm) is lock-free and does no
// try/catch (the ring is noexcept) — only a null check stands between the socket
// thread and the atomics.
// =============================================================================
#include "bridge/engine_api.h"

#include <exception>
#include <new>       // std::bad_alloc

#include "bridge/bridge_internal.hpp"
#include "common.h"  // blackwell::cuda_error

using blackwell::EngineStatus;

namespace {

// EngineStatus -> BridgeStatus (1:1 with the doctrine's vocabulary).
BridgeStatus map_status(EngineStatus s) noexcept {
    switch (s) {
        case EngineStatus::Success:          return BRIDGE_OK;
        case EngineStatus::CudaRuntimeError: return BRIDGE_ERR_CUDA;
        case EngineStatus::OutOfVram:        return BRIDGE_ERR_OUT_OF_MEMORY;
        case EngineStatus::InvalidArgument:  return BRIDGE_ERR_INVALID_ARG;
        case EngineStatus::InvalidConfig:    return BRIDGE_ERR_INVALID_ARG;
        case EngineStatus::StateMismatch:    return BRIDGE_ERR_STATE;
    }
    return BRIDGE_ERR_INTERNAL;
}

// Run `fn` (returns BridgeStatus) inside the single exception firewall used by
// every allocating/marshalling entry point. cuda_error carries a cudaError_t so
// VRAM exhaustion maps precisely; engine_error carries an EngineStatus.
template <class Fn>
BridgeStatus guarded(Fn&& fn) noexcept {
    try {
        return fn();
    } catch (const blackwell::cuda_error& e) {
        return (e.code() == cudaErrorMemoryAllocation) ? BRIDGE_ERR_OUT_OF_MEMORY
                                                       : BRIDGE_ERR_CUDA;
    } catch (const blackwell::engine_error& e) {
        return map_status(e.status());
    } catch (const std::bad_alloc&) {
        return BRIDGE_ERR_OUT_OF_MEMORY;
    } catch (const std::exception&) {
        return BRIDGE_ERR_INTERNAL;
    } catch (...) {
        return BRIDGE_ERR_INTERNAL;
    }
}

}  // namespace

namespace blackwell::bridge {

EngineHandle bridge_wrap_engine(IEngineControl* control) {
    if (!control) return nullptr;
    return new BridgeEngineOpaque(control);  // INIT tier: allocation is fine here
}

void bridge_release_engine(EngineHandle handle) noexcept {
    // Deletes the bridge object (and its stream registry). Does NOT delete the
    // streams themselves — the app owns those and must destroy them first.
    delete handle;
}

}  // namespace blackwell::bridge

// -----------------------------------------------------------------------------
// C-ABI entry points.
// -----------------------------------------------------------------------------
extern "C" {

BridgeStatus engine_create_audio_stream(EngineHandle handle, AudioStreamHandle* stream_out) {
    if (!handle || !stream_out) return BRIDGE_ERR_INVALID_HANDLE;
    *stream_out = nullptr;

    return guarded([&]() -> BridgeStatus {
        blackwell::bridge::IEngineControl* ctrl = handle->control;
        if (!ctrl) return BRIDGE_ERR_INVALID_HANDLE;

        // Pinned ring allocated once here (INIT tier; may throw cuda_error ->
        // OUT_OF_MEMORY via guarded()).
        auto* stream = new BridgeAudioStreamOpaque(ctrl->audio_ring_capacity_samples());
        stream->engine = handle;
        {
            std::lock_guard<std::mutex> lk(handle->admin_mutex);  // admin path, not hot path
            handle->streams.insert(stream);
        }
        *stream_out = stream;
        return BRIDGE_OK;
    });
}

BridgeStatus engine_destroy_audio_stream(AudioStreamHandle stream) {
    if (!stream) return BRIDGE_ERR_INVALID_HANDLE;
    // Refuse to tear down a stream a generation is still draining.
    if (stream->generating.load(std::memory_order_acquire)) return BRIDGE_ERR_STATE;

    if (BridgeEngineOpaque* eng = stream->engine) {
        std::lock_guard<std::mutex> lk(eng->admin_mutex);
        eng->streams.erase(stream);
    }
    delete stream;  // frees the pinned ring in ~AudioRingBuffer
    return BRIDGE_OK;
}

BridgeStatus audio_stream_push_pcm(AudioStreamHandle stream, const float* pcm_data,
                                   size_t num_samples) {
    // HOT PRODUCER PATH: no lock, no try/catch, no allocation. push_samples is
    // noexcept and wait-free; a null check is the only overhead.
    if (!stream) return BRIDGE_ERR_INVALID_HANDLE;
    if (!pcm_data && num_samples != 0) return BRIDGE_ERR_INVALID_ARG;
    if (num_samples == 0) return BRIDGE_OK;

    return stream->ring.push_samples(pcm_data, num_samples) ? BRIDGE_OK
                                                            : BRIDGE_ERR_BUFFER_FULL;
}

BridgeStatus engine_generate_multimodal_async(EngineHandle handle, AudioStreamHandle stream,
                                              const char* prompt_text, CallbackFn callback,
                                              void* user_data) {
    if (!handle || !stream) return BRIDGE_ERR_INVALID_HANDLE;
    if (!prompt_text || !callback) return BRIDGE_ERR_INVALID_ARG;

    return guarded([&]() -> BridgeStatus {
        blackwell::bridge::IEngineControl* ctrl = handle->control;
        if (!ctrl) return BRIDGE_ERR_INVALID_HANDLE;

        // Single-in-flight guard (single-thread engine doctrine at the edge).
        if (ctrl->generation_in_flight()) return BRIDGE_ERR_STATE;
        bool expected = false;
        if (!stream->generating.compare_exchange_strong(expected, true,
                                                        std::memory_order_acq_rel)) {
            return BRIDGE_ERR_STATE;  // this stream already has a job in flight
        }

        // Marshal onto the engine thread. The engine snapshots `ring`, runs DSP +
        // projector once, and streams tokens back via the sink. The `generating`
        // flag is cleared by the engine on the final callback (out of scope here);
        // if submit itself fails, clear it now so the stream stays usable.
        const EngineStatus st = ctrl->submit_multimodal(
            stream->ring, prompt_text,
            blackwell::bridge::TokenSink{callback, user_data});
        if (st != EngineStatus::Success) {
            stream->generating.store(false, std::memory_order_release);
            return map_status(st);
        }
        return BRIDGE_OK;
    });
}

const char* bridge_status_to_string(BridgeStatus status) {
    switch (status) {
        case BRIDGE_OK:                 return "BRIDGE_OK";
        case BRIDGE_ERR_INVALID_HANDLE: return "BRIDGE_ERR_INVALID_HANDLE";
        case BRIDGE_ERR_INVALID_ARG:    return "BRIDGE_ERR_INVALID_ARG";
        case BRIDGE_ERR_BUFFER_FULL:    return "BRIDGE_ERR_BUFFER_FULL";
        case BRIDGE_ERR_OUT_OF_MEMORY:  return "BRIDGE_ERR_OUT_OF_MEMORY";
        case BRIDGE_ERR_STATE:          return "BRIDGE_ERR_STATE";
        case BRIDGE_ERR_CUDA:           return "BRIDGE_ERR_CUDA";
        case BRIDGE_ERR_INTERNAL:       return "BRIDGE_ERR_INTERNAL";
    }
    return "BRIDGE_ERR_UNKNOWN";
}

}  // extern "C"
