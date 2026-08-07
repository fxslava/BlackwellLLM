// -----------------------------------------------------------------------------
// engine_residency.cpp — the three-state machine's transitions. See the header
// for why this is hibernation rather than a teardown.
// -----------------------------------------------------------------------------
#include "engine_residency.hpp"

#include <cstdio>
#include <exception>
#include <utility>

namespace rt {

const char* to_string(ResidencyState s) noexcept {
    switch (s) {
        case ResidencyState::Unloaded: return "unloaded";
        case ResidencyState::Loading:  return "loading";
        case ResidencyState::Ready:    return "ready";
    }
    return "ready";
}

EngineResidency::EngineResidency(blackwell::bridge::EngineControlBridge* control, EngineFn engine,
                                 bool weights_serve_asr) noexcept
    : control_(control), engine_fn_(std::move(engine)),
      weights_serve_asr_(weights_serve_asr) {
    // SEEDED FROM WHAT IS ACTUALLY THERE. On an eager launch the weights are
    // already on the card by the time this object exists, and starting in
    // Unloaded would make the first ready() check refuse a perfectly resident
    // engine. On a LAZY launch there is no engine at all, and starting in Ready
    // would let a local intent through to a control with nothing behind it.
    const bool resident = engine_fn_ && engine_fn_() != nullptr;
    state_.store(resident ? ResidencyState::Ready : ResidencyState::Unloaded,
                 std::memory_order_release);
}

void EngineResidency::enable_cold_load(ColdLoadFn load, std::function<void()> after_load) {
    cold_load_ = std::move(load);
    after_cold_load_ = std::move(after_load);
    // Explicit, per the task's wording, and it is also already true: the seed
    // above resolved to Unloaded because engine_fn_() returned null.
    state_.store(ResidencyState::Unloaded, std::memory_order_release);
}

void EngineResidency::set_progress_sink(ProgressFn fn) {
    std::lock_guard<std::mutex> lk(mu_);
    sink_ = std::move(fn);
}

std::string EngineResidency::failure() const {
    std::lock_guard<std::mutex> lk(mu_);
    return failure_;
}

void EngineResidency::publish(ResidencyState s, int percent, std::string detail) noexcept {
    state_.store(s, std::memory_order_release);
    ProgressFn sink;
    {
        std::lock_guard<std::mutex> lk(mu_);
        sink = sink_;
    }
    if (!sink) return;
    Progress p;
    p.state = s;
    p.percent = percent;
    p.detail = std::move(detail);
    // The sink is a page post (queue + PostMessage), so it cannot block the
    // engine thread -- but a throwing sink would, and this runs inside a task
    // the pump loop has no handler for.
    try {
        sink(p);
    } catch (...) {
    }
}

bool EngineResidency::request_unload(std::string* why_refused) noexcept {
    auto refuse = [&](const char* why) {
        if (why_refused != nullptr) *why_refused = why;
        return false;
    };
    if (!engine_fn_ || engine_fn_() == nullptr) {
        // Not a failure and not worth a banner: with no engine there is nothing
        // holding VRAM, so "release it" is already true. Covers both the
        // simulated backend and a lazy launch that has not loaded yet.
        return true;
    }
    // THE CAPABILITY GATE. Phrased the way require_branching() phrases its
    // refusals (CLAUDE.md extension pattern #1): say what to do instead, because
    // the user CAN have this -- on the other pipeline.
    if (weights_serve_asr_) {
        return refuse("the ultravox_legacy pipeline transcribes speech THROUGH the backbone, "
                      "so releasing its weights would stop the app hearing you. Switch the "
                      "pipeline to whisper_cascade to run remote-only with the GPU released.");
    }
    if (state() == ResidencyState::Unloaded) return true;    // already there
    bool expected = false;
    if (!in_flight_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        return refuse("a load or unload is already in flight");
    }
    if (!control_->post_engine_task([this] { run_transition(/*load=*/false); })) {
        in_flight_.store(false, std::memory_order_release);
        return refuse("could not reach the engine thread");
    }
    return true;
}

bool EngineResidency::request_load(std::string* why_refused) noexcept {
    auto refuse = [&](const char* why) {
        if (why_refused != nullptr) *why_refused = why;
        return false;
    };
    // NO ENGINE plus NO COLD LOADER is the simulated backend: there is nothing
    // to load and never was. With a cold loader this is the lazy launch's first
    // load, which is the whole point of the toggle.
    const bool have_engine = engine_fn_ && engine_fn_() != nullptr;
    if (!have_engine && !cold_load_) {
        return refuse("this launch has no local checkpoint to load");
    }
    if (state() == ResidencyState::Ready) return true;       // already there
    bool expected = false;
    if (!in_flight_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        return refuse("a load or unload is already in flight");
    }
    // PUBLISHED HERE, on the CALLER's thread, not inside the task. The engine
    // thread may be several hundred milliseconds from the next command boundary
    // (it is parked in pump_engine), and a toggle that shows nothing for that
    // long reads as a click that missed.
    publish(ResidencyState::Loading, 5, "queued behind the engine thread");
    if (!control_->post_engine_task([this] { run_transition(/*load=*/true); })) {
        in_flight_.store(false, std::memory_order_release);
        publish(ResidencyState::Unloaded, 0, "could not reach the engine thread");
        return refuse("could not reach the engine thread");
    }
    return true;
}

void EngineResidency::apply_local_inference(bool local_inference) noexcept {
    // Deliberately swallows the refusal. This is the unconditional call from the
    // settings fan-out, and the two refusals it can hit (simulated backend, the
    // legacy-ASR gate) are both permanent properties of the launch rather than
    // events -- reporting them on every save would be noise. The UI asks
    // can_release_vram() once and explains the checkbox itself.
    std::string ignored;
    if (local_inference) {
        request_load(&ignored);
    } else if (can_release_vram()) {
        request_unload(&ignored);
    }
}

void EngineResidency::run_transition(bool load) noexcept {
    // ON THE ENGINE THREAD, at a command-batch boundary: no decode is in flight,
    // which is exactly the idleness hibernate()/wakeup() require.
    {
        std::lock_guard<std::mutex> lk(mu_);
        failure_.clear();
    }
    try {
        if (load) {
            // TWO WAYS TO BECOME RESIDENT, and which one applies is decided by
            // whether an engine exists rather than by a flag:
            //
            //   COLD  no engine at all (a remote-only launch never built one).
            //         Read the checkpoint, build the engine, attach it to the
            //         control every borrower already holds, then lay the system
            //         prefix down -- the startup prefill was skipped because
            //         there was nothing to prefill into. Seconds, and it is disk
            //         and PCIe bound.
            //   WARM  the engine exists and is hibernated. One bulk H2D burst
            //         from the pinned stash; no disk, no re-parse.
            if (engine_fn_() == nullptr) {
                publish(ResidencyState::Loading, 20, "loading the checkpoint");
                cold_load_();
                publish(ResidencyState::Loading, 85, "freezing the system prompt");
                if (after_cold_load_) after_cold_load_();
                publish(ResidencyState::Ready, 100, "local model loaded");
                std::printf("[residency] cold load complete -- local inference available\n");
            } else {
                publish(ResidencyState::Loading, 35, "restoring weights over PCIe");
                engine_fn_()->wakeup();
                publish(ResidencyState::Ready, 100, "local model resident");
                std::printf("[residency] weights resident -- local inference available\n");
            }
        } else {
            publish(ResidencyState::Loading, 35, "spilling KV cache");
            // STAGE 1 FIRST, and the order matters: the KV pool is freed by
            // demoting pages down the tier waterfall, which is engine work that
            // needs the weights' arena registry intact. Hibernating first would
            // leave the pages stranded in VRAM behind a hibernated engine.
            const int spilled = engine_fn_()->spill_kv_cache();
            publish(ResidencyState::Loading, 80, "releasing weight arena");
            engine_fn_()->hibernate();
            publish(ResidencyState::Unloaded, 0, "VRAM released");
            std::printf("[residency] VRAM released (%d KV pages spilled) -- remote only\n",
                        spilled);
        }
    } catch (const std::exception& e) {
        // FAIL TO Unloaded IN BOTH DIRECTIONS, which looks wrong for a failed
        // wakeup and is not. A wakeup that threw leaves the arena in whatever
        // state it got to, and the ONE thing that must not happen is `ready()`
        // returning true over it -- that would send a decode at weights which
        // are not there. Unloaded refuses local dispatch, which is the safe
        // reading of "we do not know", and a second toggle retries cleanly
        // because wakeup() is idempotent.
        {
            std::lock_guard<std::mutex> lk(mu_);
            failure_ = e.what();
        }
        publish(ResidencyState::Unloaded, 0, e.what());
        std::fprintf(stderr, "[residency] %s failed: %s\n", load ? "load" : "unload", e.what());
    } catch (...) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            failure_ = "unknown error";
        }
        publish(ResidencyState::Unloaded, 0, "unknown error");
    }
    in_flight_.store(false, std::memory_order_release);
}

}  // namespace rt
