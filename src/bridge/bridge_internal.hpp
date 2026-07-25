#pragma once
// =============================================================================
// bridge/bridge_internal.hpp — INTERNAL glue behind the engine_api.h C-ABI.
//
// NOT a public header (lives in src/, not include/). It defines the concrete
// types the opaque EngineHandle / AudioStreamHandle point at, and the seam
// (IEngineControl) by which the real single-thread engine is plugged in. The
// engine assembly calls bridge_wrap_engine() to mint a C EngineHandle; the C
// entry points in engine_api.cpp then adapt to IEngineControl. This keeps the
// bridge free of any concrete engine/CUDA-loop dependency (it depends only on an
// interface), and keeps the engine free of C-ABI concerns.
// =============================================================================
#include <atomic>
#include <cstddef>
#include <mutex>
#include <unordered_set>

#include "bridge/audio_ring_buffer.hpp"
#include "bridge/engine_api.h"        // BridgeStatus, CallbackFn, handle typedefs
#include "blackwell/engine_status.h"  // blackwell::EngineStatus

namespace blackwell::bridge {

// A C token callback bound to its user context — a trivially-copyable POD passed
// by value onto the engine thread (no std::function heap allocation).
struct TokenSink {
    CallbackFn fn = nullptr;
    void* user = nullptr;

    void emit(const char* utf8, int32_t index, int32_t is_final, BridgeStatus status) const {
        if (fn) fn(user, utf8, index, is_final, status);
    }
};

// The seam to the real engine. Implemented by the engine assembly (out of this
// subsystem's scope); the bridge only ever holds a non-owning pointer to it and
// marshals across it. All calls originate from the C entry points.
class IEngineControl {
public:
    virtual ~IEngineControl() = default;

    // Capacity (samples) for a new stream's pinned ring — sized from the engine's
    // resolved plan (max audio seconds * sample_rate), so the ring is allocated
    // once at stream creation and never resized.
    virtual size_t audio_ring_capacity_samples() const = 0;

    // Is a generation already running on the single engine thread? Enforces the
    // single-in-flight rule at the bridge edge before we marshal another job.
    virtual bool generation_in_flight() const = 0;

    // Marshal a multimodal generate job onto the engine thread and return
    // immediately. The engine snapshots the audio buffered in `ring` at job
    // start, runs Whisper DSP + projector once (shared across Draft/Target per
    // ProjectionSpaceId), and streams tokens via `sink` from the engine thread.
    // Returns a runtime-tier EngineStatus (Success once the job is queued).
    virtual blackwell::EngineStatus submit_multimodal(AudioRingBuffer& ring,
                                                       const char* prompt_utf8,
                                                       TokenSink sink) = 0;

    // ---- Speculative speech-streaming control plane -------------------------
    // The marshalling boundary onto the single engine thread. The audio/VAD/event
    // thread only ever calls these — it NEVER touches CUDA, the KV cache, or the
    // decode loop directly (single-thread engine doctrine). Each op is tagged
    // with a monotone `gen` so the engine thread applies latest-wins semantics.

    // Barge-in: latest-wins cancel SIGNAL — immediate, NOT queued. Publishes the
    // live epoch so any in-flight decode with an older gen aborts at its next
    // checkpoint. Callable from any thread; allocation-free; noexcept.
    virtual void cancel_generation(uint64_t gen) noexcept = 0;

    // KV micro-rewind: marshal a "drop KV/SSM state past `keep_prompt_tokens`
    // back to the verified prefix" onto the engine thread. Tagged with `gen`; a
    // superseded rewind is dropped. Enqueue is allocation-free; noexcept.
    virtual blackwell::EngineStatus rewind_kv(AudioStreamHandle stream,
                                              uint32_t keep_prompt_tokens,
                                              uint64_t gen) noexcept = 0;

    // Speculative warming prefill of the audio buffered so far — grows the KV
    // cache while the user is still speaking (TrackUpdate analogue). Marshaled,
    // tagged with `gen`; noexcept.
    virtual blackwell::EngineStatus warm_prefill(AudioStreamHandle stream,
                                                 uint64_t gen) noexcept = 0;

    // Commit soft-tokens + begin decode: snapshot audio, run the projector once,
    // inject_audio_embeddings splice, run the decode loop; tokens stream via
    // `sink` from the engine thread (TriggerGeneration analogue). Marshaled,
    // tagged with `gen`; noexcept.
    virtual blackwell::EngineStatus commit_and_decode(AudioStreamHandle stream,
                                                      TokenSink sink,
                                                      uint64_t gen) noexcept = 0;
};

}  // namespace blackwell::bridge

// ---- Concrete opaque types (tags MUST match the C typedefs in engine_api.h) --

// One audio input stream: its pinned SPSC ring + a back-pointer + an in-flight
// guard. Single-owner; the app manages its lifetime (single-thread doctrine).
struct BridgeAudioStreamOpaque {
    explicit BridgeAudioStreamOpaque(size_t capacity_samples)
        : ring(capacity_samples) {}

    blackwell::bridge::AudioRingBuffer ring;
    struct BridgeEngineOpaque* engine = nullptr;   // owning engine (non-owning back-ref)
    std::atomic<bool> generating{false};           // set while a job using this stream runs
};

// The engine-side bridge object: adapts the C-ABI to IEngineControl and tracks
// live streams for validation/teardown. The `admin_mutex` guards only the
// create/destroy bookkeeping (INIT/admin tier) — never the lock-free push path.
struct BridgeEngineOpaque {
    explicit BridgeEngineOpaque(blackwell::bridge::IEngineControl* ctrl) : control(ctrl) {}

    blackwell::bridge::IEngineControl* control = nullptr;   // non-owning seam to the engine
    std::mutex admin_mutex;
    std::unordered_set<BridgeAudioStreamOpaque*> streams;
};

namespace blackwell::bridge {

// Engine-assembly entry point: wrap an IEngineControl as a C EngineHandle.
// (Declared here, defined in engine_api.cpp — the only place opaque types are
// constructed.) Returns nullptr only if `control` is null.
EngineHandle bridge_wrap_engine(IEngineControl* control);

// Destroy an engine handle minted above (releases the bridge object; does NOT
// own or destroy the underlying IEngineControl).
void bridge_release_engine(EngineHandle handle) noexcept;

}  // namespace blackwell::bridge
