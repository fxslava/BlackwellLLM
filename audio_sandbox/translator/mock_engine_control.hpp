#pragma once
// -----------------------------------------------------------------------------
// mock_engine_control.hpp — a GPU-free stand-in for the multimodal engine that
// still drives the REAL speculative control plane (EngineControlBridge: the SPSC
// command ring, the monotone barge-in epoch, the 0%-CPU wait/pump loop, and the
// wait-free decode cancel). Only the execute hooks are faked, so audio_translator
// exercises the production streaming/interruption logic end to end with no FP8
// Llama weights present (STEP 5's "Mock Engine mode").
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

class MockEngineControl : public blackwell::bridge::EngineControlBridge {
public:
    using Config = blackwell::bridge::EngineControlBridge::Config;

    explicit MockEngineControl(const Config& cfg = {})
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

    // Speculative warming: a no-op in the mock (the real path runs Whisper -> the
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
        const std::size_t idx =
            utterance_counter_.fetch_add(1, std::memory_order_relaxed) % kNumUtterances;
        const Utterance& u = kUtterances[idx];
        const std::string full = std::string("[Speech] ") + u.speech +
                                 " | [Translation] " + u.translation;

        std::int32_t emitted = 0;
        std::size_t pos = 0;
        while (pos < full.size()) {
            if (cancelled(cmd.gen)) break;  // wait-free barge-in abort
            const std::size_t space = full.find(' ', pos);
            const std::size_t end = (space == std::string::npos) ? full.size() : space + 1;
            const std::string piece = full.substr(pos, end - pos);
            cmd.sink.emit(piece.c_str(), emitted, /*is_final=*/0, BRIDGE_OK);
            pos = end;
            ++emitted;
            std::this_thread::sleep_for(std::chrono::milliseconds(piece_delay_ms_));
        }

        clear_in_flight();
        cmd.sink.emit("", emitted, /*is_final=*/1, BRIDGE_OK);
        return blackwell::EngineStatus::Success;
    }

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
    uint32_t piece_delay_ms_ = 70;  // pace so streaming + barge-in are observable
};

}  // namespace rt
