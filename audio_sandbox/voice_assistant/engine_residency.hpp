#pragma once
// -----------------------------------------------------------------------------
// engine_residency.hpp — whether the local model's weights are ON THE CARD right
// now, as a three-state machine the UI can watch and the transport must consult.
//
// WHY THIS IS HIBERNATION AND NOT A TEARDOWN. The obvious reading of "unload the
// weights" is to destroy the engine and rebuild it later. That is unimplementable
// here without a rewrite, and the reason is worth stating once: the control
// pointer is BORROWED BY EVERYTHING. AppContext holds it, the speech mode holds
// it and pumps it from the engine thread, ConversationRouter holds it,
// LocalEngineTransport holds it, and main() captured the concrete
// RealEngineControl* into the generate callable at startup. Destroying the stack
// under those would leave five dangling pointers and a PCM tap dereferencing a
// freed mode on the next 10 ms block.
//
// BlackwellEngine already has the primitive that avoids all of it:
//
//   spill_kv_cache()  stage 1 -- demote every unpinned KV page down the tier
//                     waterfall, freeing KV VRAM. The radix index survives and
//                     pages fault back in on the next prefill.
//   hibernate()       stage 2 -- evacuate the multi-GB weight arena to a PINNED
//                     HOST STASH and free the device allocation. The engine
//                     object, the name->pointer registry and every dynamic pool
//                     stay alive.
//   wakeup()          re-allocate, DMA the stash back, REBASE the registry.
//
// So the object graph is untouched and every borrowed pointer stays valid; what
// goes away is the VRAM, which is the entire point of the toggle.
//
// WHAT THIS COSTS, stated plainly because it is a real trade. The stash is
// pinned host RAM, so ~5.3 GB of system memory stays committed while unloaded --
// VRAM is freed, RAM is not. In exchange wakeup() is a single bulk H2D burst
// rather than a re-read of the checkpoint from disk, which is what makes turning
// local inference back on feel like a toggle instead of a restart.
//
// THREADING. Every one of the three engine calls carries the same contract as
// forward(): single engine-owning thread, engine idle. So nothing here touches
// them directly -- request_load()/request_unload() are callable from ANY thread
// (in practice the UI thread, from the settings fan-out) and marshal onto the
// engine thread with post_engine_task, which runs them at a command-batch
// boundary with no decode in flight. That is also what satisfies the task's "do
// NOT block the main UI or audio pipeline thread": the caller returns
// immediately and watches `state()`.
//
// ERROR TIER: RUNTIME. Nothing here throws. A refused request reports why
// through its out-parameter; a failed wakeup lands the machine in Unloaded with
// `failure()` set, because a half-resident engine is not a state anything above
// can reason about.
// -----------------------------------------------------------------------------
#include <atomic>
#include <functional>
#include <mutex>
#include <string>

// BlackwellEngine is at GLOBAL scope, not in namespace blackwell -- see the
// declaration in engine.h, which closes the namespace just above it. Spelled
// ::BlackwellEngine throughout so a reader inside namespace rt does not have to
// know that.
#include "blackwell/engine.h"          // ::BlackwellEngine
#include "engine_control_bridge.hpp"   // blackwell::bridge::EngineControlBridge

namespace rt {

// The three states the task names, and there are exactly three because the
// middle one is observable: a wakeup is a multi-second PCIe burst, and a UI that
// only knew loaded/unloaded would show a dead toggle for the whole of it.
enum class ResidencyState {
    Unloaded,   // VRAM released. Local dispatch MUST be refused.
    Loading,    // a wakeup (or the first cold load) is in flight on the engine thread.
    Ready,      // weights are on the card; local dispatch is legal.
};

const char* to_string(ResidencyState s) noexcept;

class EngineResidency {
public:
    // What the page is told. `percent` is deliberately coarse (0/35/80/100): the
    // underlying DMA reports nothing, so a smooth bar would be a lie. It exists
    // to prove the request was accepted and is moving, not to time it.
    struct Progress {
        ResidencyState state = ResidencyState::Ready;
        int            percent = 100;
        std::string    detail;
    };
    using ProgressFn = std::function<void(const Progress&)>;

    // `control` is BORROWED and must outlive this object; `engine` may be NULL,
    // which is the simulated backend and means there is no VRAM to release.
    //
    // `weights_serve_asr` is the CAPABILITY GATE (CLAUDE.md extension pattern
    // #1). True on the ultravox_legacy pipeline, where the backbone is also the
    // speech recogniser -- audio soft-tokens are spliced into ITS KV -- so
    // evicting its weights does not merely stop replies, it makes the app deaf.
    // Rather than guessing, unload is refused there with a message naming the
    // thing to do instead. False on whisper_cascade, where whisper.cpp
    // transcribes independently and the backbone is purely the reply generator.
    EngineResidency(blackwell::bridge::EngineControlBridge* control,
                    ::BlackwellEngine* engine, bool weights_serve_asr) noexcept;

    EngineResidency(const EngineResidency&) = delete;
    EngineResidency& operator=(const EngineResidency&) = delete;

    // ANY thread. Fired on the ENGINE thread as the state advances, so the
    // implementation must only marshal (post_event is the intended sink).
    void set_progress_sink(ProgressFn fn);

    // ---- requests (ANY thread; they return immediately) ---------------------
    // Release the VRAM. Returns false and fills `why_refused` when this launch
    // cannot honour it -- the simulated backend, the legacy pipeline's ASR
    // dependency, or a request that raced another one.
    bool request_unload(std::string* why_refused) noexcept;
    // Put the weights back. Same refusal contract.
    bool request_load(std::string* why_refused) noexcept;

    // Drive residency from the settings toggle. `local_inference` true means the
    // user wants the local leg, so the weights must be resident; false is
    // "Remote Network" and releases them. Idempotent -- calling it with the
    // state already correct is a no-op, which is what lets the settings fan-out
    // call it unconditionally on every save.
    void apply_local_inference(bool local_inference) noexcept;

    // ---- observation (ANY thread) -------------------------------------------
    [[nodiscard]] ResidencyState state() const noexcept {
        return state_.load(std::memory_order_acquire);
    }
    // THE PREDICATE THE TRANSPORT ASKS. Anything but Ready must not reach a
    // decode: get_weight_ptr() throws while hibernated, and a throw crossing the
    // engine thread's pump loop would take the app down.
    [[nodiscard]] bool ready() const noexcept {
        return state() == ResidencyState::Ready;
    }
    // Whether an unload is even possible on this launch. The UI uses it to
    // explain the checkbox rather than to hide it.
    [[nodiscard]] bool can_release_vram() const noexcept {
        return engine_ != nullptr && !weights_serve_asr_;
    }
    // Empty unless the last transition failed.
    [[nodiscard]] std::string failure() const;

private:
    // Runs ON THE ENGINE THREAD. `load` selects wakeup vs. spill+hibernate.
    void run_transition(bool load) noexcept;
    void publish(ResidencyState s, int percent, std::string detail) noexcept;

    blackwell::bridge::EngineControlBridge* control_ = nullptr;   // borrowed
    ::BlackwellEngine*             engine_ = nullptr;    // borrowed, may be null
    bool                                    weights_serve_asr_ = false;

    std::atomic<ResidencyState> state_{ResidencyState::Ready};
    // Set for the whole window between accepting a request and the engine thread
    // finishing it. Guards against a double-click queueing two wakeups, which
    // would be harmless (both are idempotent) but would report nonsense progress.
    std::atomic<bool>           in_flight_{false};

    mutable std::mutex mu_;        // guards failure_ and sink_
    std::string        failure_;
    ProgressFn         sink_;
};

}  // namespace rt
