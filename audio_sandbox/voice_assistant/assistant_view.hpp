#pragma once
// -----------------------------------------------------------------------------
// assistant_view.hpp — the pipeline's UI-facing model. It owns no widgets and
// draws nothing: every producer edge turns one pipeline event into one JSON
// message and hands it to a sink. The Chromium view on the other side of that
// sink is what renders a messenger out of it.
//
// WHY IT IS NOT A RENDERER ANY MORE. The previous version called ImGui directly
// and painted two "cards" plus a bar of commit-gate counters -- a debug panel
// wearing an app's name. Splitting the model from the renderer is what let the
// whole main screen become an HTML messenger without touching a line of the
// pipeline underneath it.
//
// THREADING is the reason this class exists at all. Events arrive from THREE
// threads:
//     local tokens        engine thread (TokenSink)
//     pipeline state      audio/VAD thread or engine thread
//     remote deltas       dispatcher thread
// None of them may touch a window, COM, or WebView2 (which is STA: creating
// thread only). So every producer here is a pure function of its arguments into
// a JSON string, and the sink's ONLY job is to marshal that string to the UI
// thread. The sink is therefore required to be callable from any thread and must
// not block -- see AssistantWindow::post_event, which appends to a queue and
// PostMessage()s a wake-up.
//
// WHERE THE TELEMETRY WENT. The commit-gate counters are still published, but as
// a `stats` event the Settings/diagnostics view consumes -- never on the main
// screen. One of them still matters enough to survive that move: a TokenCap
// truncation produces NO cloud call and NO error, so `local.final` carries the
// termination reason and the user-facing bubble marks a truncated turn as
// incomplete. That is a state a person can act on ("say it again"), not a metric.
// -----------------------------------------------------------------------------
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>

#include <nlohmann/json.hpp>

#include "bridge/speculative_bridge_api.h"  // SpeechPipelineState
#include "intent_commit.hpp"                // TerminationReason, IntentCommitQueue

namespace rt {

class AssistantView {
public:
    // What the pipeline is doing, in the user's vocabulary rather than the state
    // machine's. Mapped from SpeechPipelineState + the gate's verdict.
    enum class Phase : int { Idle, Listening, Generating, BargedIn, Committed, Dispatching };

    // Called with one complete JSON message, from ANY thread. Must not block and
    // must not touch the UI directly -- marshal.
    using EventSink = std::function<void(std::string)>;

    // Bind (or rebind) the sink. A null sink is legal and makes every producer a
    // no-op, which is what lets a headless test drive the same pipeline with no
    // window at all.
    //
    // GUARDED, and not as a formality: the window is created AFTER the audio,
    // engine and dispatcher threads are already running, so this assignment
    // genuinely races their emit() calls -- a VAD transition arriving during the
    // std::function assignment was aborting the process about half the time.
    void set_sink(EventSink sink) {
        std::lock_guard<std::mutex> lk(sink_mu_);
        sink_ = std::move(sink);
    }

    // ---- producer edges (any thread) ---------------------------------------

    // Engine thread: one detokenized piece of the LOCAL model's answer -- what the
    // on-device model heard and is turning into an intent. In the messenger this
    // is the USER's live bubble: it is the transcription of what they just said.
    void on_local_token(const char* utf8, std::uint64_t gen) {
        if (utf8 == nullptr || *utf8 == '\0') return;
        bool restart = false;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (gen != local_gen_) {   // a new generation replaces the old draft
                local_gen_ = gen;
                restart = true;
            }
        }
        nlohmann::json j;
        j["type"] = "local.delta";
        j["gen"] = gen;
        j["restart"] = restart;   // the view clears the live bubble on a new epoch
        j["text"] = utf8;
        emit(j);
    }

    // Engine thread: the local generation ended. `reason` is the gate's verdict,
    // and it is the difference between "that was your whole sentence" and "that
    // got cut off and was never sent anywhere".
    void on_local_final(blackwell::bridge::TerminationReason reason) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            last_reason_ = reason;
            switch (reason) {
                case blackwell::bridge::TerminationReason::Eos:
                    phase_ = Phase::Committed;
                    break;
                case blackwell::bridge::TerminationReason::BargeIn:
                    phase_ = Phase::BargedIn;
                    break;
                default:
                    phase_ = Phase::Idle;
                    break;
            }
        }
        nlohmann::json j;
        j["type"] = "local.final";
        j["reason"] = blackwell::bridge::to_string(reason);
        // The single fact the bubble needs: was this a finished thought (and so
        // on its way to the assistant), or a fragment that goes nowhere?
        j["dispatched"] = blackwell::bridge::is_dispatchable(reason);
        j["phase"] = phase_label(phase());
        emit(j);
    }

    // Audio/engine thread: raw pipeline state transition.
    void on_pipeline_state(SpeechPipelineState s) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            pipeline_state_ = s;
            // Only advance the *user-facing* phase for states the gate has no
            // opinion about; Committed/BargedIn are set by on_local_final and must
            // survive the IDLE transition that immediately follows them.
            if (s == SPEECH_STATE_PREFILL_SPEAKING) phase_ = Phase::Listening;
            else if (s == SPEECH_STATE_DECODE_TRANSLATING) phase_ = Phase::Generating;
            else if (s == SPEECH_STATE_INTERRUPTION_REWIND) phase_ = Phase::BargedIn;
        }
        nlohmann::json j;
        j["type"] = "phase";
        j["phase"] = phase_label(phase());
        emit(j);
    }

    // UI thread: the user typed a message instead of speaking it. The bubble is
    // painted from HERE rather than optimistically in the page, so the transcript
    // has exactly one source of truth on both input paths. The round trip is a
    // PostMessage on the thread that is already in the message loop.
    //
    // It also advances the phase: a typed turn never touches the speech pipeline,
    // so no SpeechPipelineState transition will announce that work has started.
    void on_user_text(const std::string& text) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            phase_ = Phase::Generating;
        }
        nlohmann::json j;
        j["type"] = "user.text";
        j["text"] = text;
        j["phase"] = phase_label(phase());
        emit(j);
    }

    // Dispatcher thread: an intent left the gate and is being sent.
    void on_dispatch_start(std::uint64_t sequence) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            phase_ = Phase::Dispatching;
        }
        nlohmann::json j;
        j["type"] = "remote.start";
        j["seq"] = sequence;
        emit(j);
    }

    // Dispatcher thread: one piece of the REMOTE model's reply.
    void on_remote_token(std::string_view utf8) {
        if (utf8.empty()) return;
        nlohmann::json j;
        j["type"] = "remote.delta";
        j["text"] = std::string(utf8);
        emit(j);
    }

    // Dispatcher thread: the remote exchange finished.
    void on_remote_final(bool ok, const std::string& detail) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            phase_ = Phase::Idle;
        }
        nlohmann::json j;
        j["type"] = "remote.final";
        j["ok"] = ok;
        j["detail"] = detail;
        emit(j);
    }

    // `live` drives the cost indicator: a live transport spends money per commit,
    // an offline one does not, and that difference must stay legible even after
    // the debug bar is gone. It renders as one small pill, not a metric.
    void set_transport(const char* name, bool live) {
        nlohmann::json j;
        j["type"] = "transport";
        j["name"] = name != nullptr ? name : "";
        j["live"] = live;
        emit(j);
    }

    // Engine bring-up result, so the view can say "simulated" or "8B on CUDA 0"
    // in Settings rather than leaving the user guessing which backend answered.
    void set_backend(const std::string& name, const std::string& detail, bool audio_ready) {
        nlohmann::json j;
        j["type"] = "backend";
        j["name"] = name;
        j["detail"] = detail;
        j["audio_ready"] = audio_ready;
        emit(j);
    }

    // ---- diagnostics (Settings view only, never the chat) -------------------
    // The gate counters, polled on the UI thread. They are plain uint64 written by
    // the engine thread: a torn read misreports one poll and self-corrects on the
    // next, which is why they are not atomic and why they are not allowed on a
    // screen the user reads while talking.
    void publish_stats(const blackwell::bridge::IntentCommitQueue& q) {
        nlohmann::json j;
        j["type"] = "stats";
        j["committed"] = q.committed();
        j["barge_in"] = q.dropped_barge_in();
        j["token_cap"] = q.dropped_token_cap();
        j["queue_full"] = q.dropped_queue_full();
        j["fault"] = q.dropped_fault();
        j["phase"] = phase_label(phase());
        j["last_reason"] = blackwell::bridge::to_string(last_reason());
        emit(j);
    }

    // ---- readback (any thread) ----------------------------------------------
    Phase phase() const {
        std::lock_guard<std::mutex> lk(mu_);
        return phase_;
    }
    blackwell::bridge::TerminationReason last_reason() const {
        std::lock_guard<std::mutex> lk(mu_);
        return last_reason_;
    }

    static const char* phase_label(Phase p) noexcept {
        switch (p) {
            case Phase::Idle:        return "idle";
            case Phase::Listening:   return "listening";
            case Phase::Generating:  return "generating";
            case Phase::BargedIn:    return "interrupted";
            case Phase::Committed:   return "committed";
            case Phase::Dispatching: return "dispatching";
        }
        return "idle";
    }

private:
    // NOTHROW, by contract. Every caller is a callback on the audio, engine or
    // dispatcher thread, and an exception escaping one of those is std::terminate
    // -- a UI event is never worth the process.
    //
    // Two things can throw here and both are real:
    //   * dump() raises type_error.316 on invalid UTF-8, and a detokenized token
    //     piece routinely splits a multi-byte sequence (any Cyrillic or CJK
    //     reply). `replace` substitutes U+FFFD instead, so a torn code point
    //     shows as one bad glyph for one frame and is healed by the next delta.
    //   * the sink allocates (queue push).
    //
    // sink_mu_ is its OWN lock, not mu_: emit() is called from methods that have
    // just released mu_, and reusing it would make the ordering a standing
    // deadlock hazard for anyone adding a producer later.
    void emit(const nlohmann::json& j) const noexcept {
        try {
            std::string payload =
                j.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
            std::lock_guard<std::mutex> lk(sink_mu_);
            if (sink_) sink_(std::move(payload));
        } catch (...) {
            // Dropping one UI event is the correct failure here: the pipeline is
            // authoritative, and the next event re-establishes the view's state.
        }
    }

    mutable std::mutex mu_;
    mutable std::mutex sink_mu_;
    EventSink sink_;
    std::uint64_t local_gen_ = ~0ull;
    Phase phase_ = Phase::Idle;
    SpeechPipelineState pipeline_state_ = SPEECH_STATE_IDLE;
    blackwell::bridge::TerminationReason last_reason_ =
        blackwell::bridge::TerminationReason::None;
};

}  // namespace rt
