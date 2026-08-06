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
#include <vector>

#include <nlohmann/json.hpp>

#include "bridge/speculative_bridge_api.h"  // SpeechPipelineState
#include "intent_commit.hpp"                // TerminationReason, IntentCommitQueue
#include "session_store.hpp"                // SessionSummary + ChatTurn (the sidebar's data)
#include "utf8_stream.hpp"                  // Utf8StreamAssembler (remote chunk splits)

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
            remote_utf8_.reset();   // no fragment may survive into a new reply
        }
        nlohmann::json j;
        j["type"] = "remote.start";
        j["seq"] = sequence;
        emit(j);
    }

    // Dispatcher thread: one piece of the REMOTE model's reply.
    //
    // A TRANSPORT CHUNK IS NOT A CHARACTER. Whatever is on the other end delivers
    // bytes on its own schedule -- an SSE text_delta, a simulated slice, a future
    // transport nobody has written yet -- and none of them owes this view a whole
    // code point. Reassembling here rather than trusting each transport is what
    // makes that a non-issue: the assembler holds a trailing partial sequence
    // until the bytes completing it arrive, so a Cyrillic or CJK reply cannot be
    // torn no matter who produced it. (It matters: OfflineTransport sliced its
    // echo every 12 BYTES and cut two-byte Cyrillic characters in half, and each
    // orphaned byte then reached the page as its own U+FFFD.)
    void on_remote_token(std::string_view utf8) {
        if (utf8.empty()) return;
        std::string whole;
        {
            std::lock_guard<std::mutex> lk(mu_);
            whole = remote_utf8_.push(utf8);
        }
        if (whole.empty()) return;   // this chunk was only part of a character
        nlohmann::json j;
        j["type"] = "remote.delta";
        j["text"] = std::move(whole);
        emit(j);
    }

    // Dispatcher thread: the remote exchange finished.
    //
    // `cancelled` separates "this went wrong" from "you stopped it". They arrive
    // through the same non-Ok status and must not be drawn the same way: a red
    // failure note on an answer the user themselves interrupted reads as a bug
    // in the app, and it is the one outcome the user already knows about.
    // Defaulted, because every producer other than the dispatcher's completion
    // edge is reporting a genuine failure.
    void on_remote_final(bool ok, const std::string& detail, bool cancelled = false) {
        std::string tail;
        {
            std::lock_guard<std::mutex> lk(mu_);
            phase_ = Phase::Idle;
            // Non-empty only if the reply genuinely stopped mid-character (a
            // truncated or aborted stream). Surrender it as a last delta rather
            // than dropping bytes the model did produce -- one visible glyph beats
            // silently losing a character.
            tail = remote_utf8_.flush();
        }
        if (!tail.empty()) {
            nlohmann::json t;
            t["type"] = "remote.delta";
            t["text"] = std::move(tail);
            emit(t);
        }
        nlohmann::json j;
        j["type"] = "remote.final";
        j["ok"] = ok;
        j["detail"] = detail;
        j["cancelled"] = cancelled;
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

    // ---- sessions (UI thread) -----------------------------------------------
    // Both of these are APP-level producers, like set_transport/set_backend
    // below: the pipeline has no concept of a session, and neither of these is
    // reachable from the audio, engine or dispatcher threads.

    // The sidebar's contents. Newest first (SessionStore::list_summaries orders
    // it), with `active` naming the conversation the app is currently writing
    // into -- which may legitimately be absent from the list: a brand-new chat is
    // not persisted until its first turn is answered, exactly like every other
    // messenger. The page draws that case as a pending row rather than showing
    // nothing selected.
    //
    // `ephemeral` is true when the local leg is answering, i.e. nothing said from
    // now on will be written to disk. It is pushed rather than inferred in the
    // page because the page would have to derive it from a settings field, and a
    // "your conversation is not being saved" banner must not be one stale copy
    // away from lying.
    void set_sessions(const std::vector<blackwell::cloud::SessionSummary>& list,
                      const std::string& active, bool ephemeral) {
        nlohmann::json arr = nlohmann::json::array();
        for (const blackwell::cloud::SessionSummary& s : list) {
            nlohmann::json e;
            e["id"] = s.id;
            e["updated_at"] = s.updated_at;
            e["turns"] = s.turn_count;
            e["preview"] = s.preview;
            arr.push_back(std::move(e));
        }
        nlohmann::json j;
        j["type"] = "session.list";
        j["sessions"] = std::move(arr);
        j["active"] = active;
        j["ephemeral"] = ephemeral;
        emit(j);
    }

    // REPAINT THE TRANSCRIPT. Sent when a session is opened -- at startup for the
    // restored default, and on every click in the sidebar.
    //
    // The whole transcript in ONE event, not a replay of user.text/remote.delta
    // per turn. Those producers advance the phase, freeze live bubbles and drive
    // the typing indicator; replaying them would animate a conversation that
    // already happened and would leave the status line describing a turn that is
    // not running. This event says "the transcript IS this" and the page rebuilds
    // from scratch, which is also the only shape that can render an EMPTY session
    // (a new chat) without a special case.
    void on_session_restored(const std::string& id,
                             const std::vector<blackwell::cloud::ChatTurn>& turns) {
        nlohmann::json arr = nlohmann::json::array();
        for (const blackwell::cloud::ChatTurn& t : turns) {
            arr.push_back({{"user", t.user}, {"assistant", t.assistant}});
        }
        {
            // A restore ends whatever the view thought was in flight: the bubbles
            // those producers were filling have just been deleted, so a delta
            // arriving afterwards must not be appended to a handle the page no
            // longer has. Resetting the assembler is the same reasoning as
            // on_dispatch_start -- no fragment may survive into a new transcript.
            std::lock_guard<std::mutex> lk(mu_);
            phase_ = Phase::Idle;
            remote_utf8_.reset();
            local_gen_ = ~0ull;
        }
        nlohmann::json j;
        j["type"] = "session.restore";
        j["id"] = id;
        j["turns"] = std::move(arr);
        j["phase"] = phase_label(phase());
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
    //   * dump() raises type_error.316 on invalid UTF-8. Torn code points are now
    //     healed UPSTREAM -- Utf8StreamAssembler holds a partial sequence in the
    //     decode loop until the token completing it arrives, so a Cyrillic or CJK
    //     reply reaches this point already whole. `replace` stays as the backstop
    //     for what the assembler deliberately does not fix: genuinely malformed
    //     model output, and the tail it surrenders at flush() when a generation
    //     stops mid-character. One U+FFFD beats losing the event.
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
    // Heals code points split across transport chunks. Touched only by the three
    // remote_* producers, all of which run on the single dispatcher thread -- mu_
    // guards it anyway, because "only one thread calls these" is a property of the
    // dispatcher that this class cannot enforce.
    blackwell::bridge::Utf8StreamAssembler remote_utf8_;
    std::uint64_t local_gen_ = ~0ull;
    Phase phase_ = Phase::Idle;
    SpeechPipelineState pipeline_state_ = SPEECH_STATE_IDLE;
    blackwell::bridge::TerminationReason last_reason_ =
        blackwell::bridge::TerminationReason::None;
};

}  // namespace rt
