#pragma once
// -----------------------------------------------------------------------------
// simulated_engine_control.hpp — a GPU-free stand-in for the multimodal engine that
// still drives the REAL speculative control plane (EngineControlBridge: the SPSC
// command ring, the monotone barge-in epoch, the 0%-CPU wait/pump loop, and the
// wait-free decode cancel). Only the execute hooks are faked, so audio_translator
// exercises the production streaming/interruption logic end to end with no FP8
// Llama weights present (STEP 5's simulated-engine mode).
//
// It also honours the system-prompt prefix-cache invariant: prefill_system_prompt
// freezes the prefix length as the KV rewind floor (set_system_prefix_tokens), so
// every barge-in rewind (do_rewind) clamps UP to that floor and can never truncate
// the cached system prefix (STEP 2).
//
// THREADING: do_* run on the single engine thread (drained by wait_and_pump); the
// fake decode loop checks cancelled(gen) before every emitted piece, so a barge-in
// raised on the audio/VAD thread aborts it within one piece — exactly the real
// contract, minus CUDA.
// -----------------------------------------------------------------------------
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>

#include "engine_control_bridge.hpp"        // EngineControlBridge, Command
#include "bridge/engine_api.h"              // BRIDGE_OK for the token sink

namespace rt {

class SimulatedEngineControl : public blackwell::bridge::EngineControlBridge {
public:
    using Config = blackwell::bridge::EngineControlBridge::Config;

    explicit SimulatedEngineControl(const Config& cfg = {})
        : blackwell::bridge::EngineControlBridge(/*engine=*/nullptr, cfg) {}

    // Engine-thread setup: "prefill" the frozen system prompt. There is no CUDA
    // KV here, so we tokenize by whitespace (a representative token count) and
    // freeze it as the rewind floor. Returns N_system_prefix_tokens.
    uint32_t prefill_system_prompt(const std::string& prompt) {
        const uint32_t n = count_tokens(prompt);
        set_system_prefix_tokens(n);
        return n;
    }

    // The keep count the most recent barge-in rewind resolved to (after the floor
    // clamp). Read by the UI for the prefix-cache invariant readout.
    uint32_t last_rewind_keep() const noexcept {
        return last_rewind_keep_.load(std::memory_order_acquire);
    }

    void set_piece_delay_ms(uint32_t ms) noexcept { piece_delay_ms_ = ms; }

protected:
    // Barge-in micro-rewind. No real KV to drop; the point exercised here is the
    // frozen-prefix floor: effective_keep_tokens clamps UP to the system prefix,
    // so the cached prompt is never truncated no matter what the controller asks.
    blackwell::EngineStatus do_rewind(const Command& cmd) override {
        const uint32_t keep = effective_keep_tokens(cmd.keep_prompt_tokens);
        last_rewind_keep_.store(keep, std::memory_order_release);
        return blackwell::EngineStatus::Success;
    }

    // Speculative warming: a no-op here (the real path runs Whisper -> the
    // Ultravox projector -> inject_audio_embeddings -> Llama prefill). Report
    // success so the state machine keeps warming.
    blackwell::EngineStatus do_warm_prefill(const Command& /*cmd*/) override {
        return blackwell::EngineStatus::Success;
    }

    // Commit + decode: stream a canned "[Speech] ... | [Translation] ..." response
    // word by word, checking cancelled(gen) before each piece so a barge-in aborts
    // within one word. Clears the in-flight guard and emits the final piece on exit
    // (real EngineControlBridge::run_decode_loop contract, faked payload).
    blackwell::EngineStatus do_commit_decode(const Command& cmd) override {
        using blackwell::bridge::TerminationReason;

        // TYPED TURN (submit_text): the same faked decode loop, over the user's
        // own words instead of a canned utterance. Deliberately NOT a separate
        // path -- the point of this stand-in is that text and voice reach the
        // gate identically, and a second loop here would make that untrue offline
        // while it stayed true on the GPU.
        std::string full;
        if (cmd.text_turn) {
            std::string typed;
            if (!take_text_turn(typed) || typed.empty()) {
                clear_in_flight();
                cmd.sink.emit("", 0, /*is_final=*/1, BRIDGE_OK);
                return blackwell::EngineStatus::Success;
            }
            full = "You said: " + typed + " — noted.";
        } else {
            const std::size_t idx =
                utterance_counter_.fetch_add(1, std::memory_order_relaxed) % kNumUtterances;
            const Utterance& u = kUtterances[idx];
            full = std::string("[Speech] ") + u.speech + " | [Translation] " + u.translation;
        }

        // Running the canned utterance to its end IS this stand-in's EOS: the
        // model stopped because it had nothing left to say, which is exactly the
        // condition the commit gate accepts. Every other exit below overwrites it.
        TerminationReason reason = TerminationReason::Eos;
        std::string reply;
        std::int32_t emitted = 0;
        std::size_t pos = 0;
        while (pos < full.size()) {
            if (cancelled(cmd.gen)) {          // wait-free barge-in abort
                reason = TerminationReason::BargeIn;
                break;
            }
            // Demo/test seam: a piece ceiling that forces the TokenCap path so the
            // truncation telemetry can be exercised offline, with no GPU and no
            // 256-token real decode to sit through.
            const std::int32_t cap = max_pieces_.load(std::memory_order_acquire);
            if (cap > 0 && emitted >= cap) {
                reason = TerminationReason::TokenCap;
                break;
            }
            const std::size_t space = full.find(' ', pos);
            const std::size_t end = (space == std::string::npos) ? full.size() : space + 1;
            const std::string piece = full.substr(pos, end - pos);
            cmd.sink.emit(piece.c_str(), emitted, /*is_final=*/0, BRIDGE_OK);
            reply += piece;
            pos = end;
            ++emitted;
            std::this_thread::sleep_for(std::chrono::milliseconds(piece_delay_ms_));
        }

        clear_in_flight();
        cmd.sink.emit("", emitted, /*is_final=*/1, BRIDGE_OK);

        // The gate. Offered on EVERY path, including the rejected ones -- that is
        // what makes dropped_barge_in()/dropped_token_cap() move offline, and it
        // is also what publishes last_reason() for the UI.
        (void)publish_intent(reason, cmd.gen, std::move(reply),
                             static_cast<uint32_t>(emitted));
        return blackwell::EngineStatus::Success;
    }

public:
    // 0 = unlimited (the canned utterance runs to its natural end -> Eos).
    // > 0 = force a TokenCap after N pieces, for exercising the truncation path.
    void set_max_pieces(std::int32_t n) noexcept {
        max_pieces_.store(n, std::memory_order_release);
    }

    // last_reason() is inherited from EngineControlBridge (published by
    // publish_intent), so the UI reads it identically here and on the real path.

protected:

private:
    struct Utterance {
        const char* speech;       // ASCII transcript
        const char* translation;  // UTF-8 Russian (source is compiled with /utf-8)
    };

    // Whitespace token count — a GPU-free stand-in for the real tokenizer's prefill
    // length. Collapses runs of spaces; good enough to freeze a representative
    // N_system_prefix_tokens.
    static uint32_t count_tokens(const std::string& s) noexcept {
        uint32_t n = 0;
        bool in_word = false;
        for (const char c : s) {
            const bool ws = (c == ' ' || c == '\t' || c == '\n' || c == '\r');
            if (!ws && !in_word) { ++n; in_word = true; }
            else if (ws) { in_word = false; }
        }
        return n;
    }

    static constexpr std::size_t kNumUtterances = 3;
    static constexpr Utterance kUtterances[kNumUtterances] = {
        {"Hello, how are you today?", "Здравствуйте, как ваши дела сегодня?"},
        {"The weather is nice this morning.", "Погода сегодня утром прекрасная."},
        {"Please send me the report by tomorrow.", "Пожалуйста, пришлите мне отчёт к завтрашнему дню."},
    };

    std::atomic<uint32_t> last_rewind_keep_{0};
    std::atomic<std::size_t> utterance_counter_{0};
    std::atomic<std::int32_t> max_pieces_{0};   // 0 = unlimited (natural Eos)
    uint32_t piece_delay_ms_ = 70;  // pace so streaming + barge-in are observable
};

}  // namespace rt
