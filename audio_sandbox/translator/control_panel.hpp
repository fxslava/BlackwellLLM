#pragma once
// -----------------------------------------------------------------------------
// translator/control_panel.hpp — the "Translator Settings" ImGui panel for
// audio_translator: context-mode toggle (Live Translator / Voice Assistant),
// forced source/target language dropdowns, the auto-commit silence-timeout
// slider, and the push-to-talk hotkey.
//
// THREADING (single-thread engine doctrine, CLAUDE.md)
//   draw() runs ONLY on the UI thread inside the ImGui frame. It never touches
//   the engine, CUDA, or the KV cache: every control lands either in a
//   RealEngineControl atomic (read by the engine thread at turn boundaries) or
//   in a lock-free speech_pipeline_* C call (documented callable from any
//   thread). The push-to-talk key maps 1:1 onto the pipeline's existing
//   boundary events: press -> on_speech_start (barge-in if decoding), release ->
//   on_silence_timeout (immediate commit + decode) — both no-op safely when the
//   state machine is not in a matching state.
// -----------------------------------------------------------------------------
#include <cstdint>

#include "imgui.h"

#include "bridge/speculative_bridge_api.h"  // speech_pipeline_* boundary events
#include "language_table.hpp"               // rt::kLanguages
#include "real_engine_control.hpp"          // RealEngineControl atomics (setters)

namespace rt {

class ControlPanel {
public:
    // Both non-owning; the app guarantees they outlive the UI loop. The initial
    // widget values are read back from the control's atomics so CLI defaults
    // (--context-mode/--src-lang/...) show up pre-selected.
    ControlPanel(RealEngineControl* control, SpeechPipelineHandle pipe,
                 int initial_silence_ms) noexcept
        : control_(control), pipe_(pipe), silence_ms_(initial_silence_ms) {
        mode_bounded_ = control_->context_mode() ==
                        RealEngineControl::ContextMode::BoundedHistory;
        history_budget_ = control_->history_budget_tokens();
        src_idx_ = control_->source_language_index();
        tgt_idx_ = control_->target_language_index();
    }

    // UI thread, once per ImGui frame.
    void draw() {
        ImGui::SetNextWindowPos(ImVec2(444.0f, 12.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(400.0f, 300.0f), ImGuiCond_FirstUseEver);
        ImGui::Begin("Translator Settings");

        draw_mode_section();
        ImGui::Separator();
        draw_language_section();
        ImGui::Separator();
        draw_timing_section();
        ImGui::Separator();
        draw_hotkey_section();

        ImGui::End();

        // The hotkey must work even when this window is not focused; poll it at
        // frame scope, but never while a text widget owns the keyboard.
        poll_push_to_talk();
    }

private:
    void draw_mode_section() {
        ImGui::TextUnformatted("Context mode");
        bool changed = false;
        if (ImGui::RadioButton("Live Translator (stateless)", !mode_bounded_)) {
            mode_bounded_ = false;
            changed = true;
        }
        if (ImGui::RadioButton("Voice Assistant (bounded memory)", mode_bounded_)) {
            mode_bounded_ = true;
            changed = true;
        }
        if (changed) {
            control_->set_context_mode(mode_bounded_
                                           ? RealEngineControl::ContextMode::BoundedHistory
                                           : RealEngineControl::ContextMode::Stateless);
        }
        if (mode_bounded_) {
            ImGui::SetNextItemWidth(160.0f);
            if (ImGui::SliderInt("History budget (tokens)", &history_budget_, 64, 1024)) {
                control_->set_history_budget_tokens(history_budget_);
            }
        }
        // Live invariant readout: where the KV floor currently sits.
        ImGui::TextDisabled("KV floor: sys=%u  base=%d",
                            control_->system_prefix_tokens(),
                            control_->history_base_pos());
    }

    void draw_language_section() {
        ImGui::TextUnformatted("Languages (apply from the next utterance)");
        ImGui::SetNextItemWidth(160.0f);
        const bool s = ImGui::Combo("Source", &src_idx_, kLanguages, kLanguageCount);
        ImGui::SetNextItemWidth(160.0f);
        const bool t = ImGui::Combo("Target", &tgt_idx_, kLanguages, kLanguageCount);
        if (s || t) control_->set_languages(src_idx_, tgt_idx_);
    }

    void draw_timing_section() {
        ImGui::TextUnformatted("Auto-generation");
        ImGui::SetNextItemWidth(200.0f);
        if (ImGui::SliderInt("Silence timeout (ms)", &silence_ms_, 200, 2000)) {
            (void)speech_pipeline_set_silence_hangover_ms(
                pipe_, static_cast<uint32_t>(silence_ms_));
        }
        if (ImGui::Checkbox("Manual only (disable auto-commit)", &manual_only_)) {
            (void)speech_pipeline_set_silence_hangover_ms(
                pipe_, manual_only_ ? 0u : static_cast<uint32_t>(silence_ms_));
        }
    }

    void draw_hotkey_section() {
        ImGui::TextUnformatted("Push-to-talk");
        ImGui::SetNextItemWidth(120.0f);
        ImGui::Combo("Hotkey", &hotkey_idx_, kHotkeyNames, kHotkeyCount);
        ImGui::TextDisabled(ptt_down_ ? "[%s] HELD — release to translate"
                                      : "hold [%s] to capture, release to translate; "
                                        "tap to flush now",
                            kHotkeyNames[hotkey_idx_]);
    }

    void poll_push_to_talk() {
        ImGuiIO& io = ImGui::GetIO();
        if (io.WantTextInput) return;  // a text widget owns the keyboard
        const ImGuiKey key = kHotkeys[hotkey_idx_];
        const bool down = ImGui::IsKeyDown(key);
        if (down && !ptt_down_) {
            // Press: open (or barge into) an utterance boundary NOW.
            (void)speech_pipeline_on_speech_start(pipe_);
        } else if (!down && ptt_down_) {
            // Release: flush the buffered audio and decode immediately. A no-op
            // unless the pipeline is PREFILL_SPEAKING, so a stray tap is safe.
            (void)speech_pipeline_on_silence_timeout(pipe_);
        }
        ptt_down_ = down;
    }

    static constexpr const char* kHotkeyNames[] = {"Space", "F2", "F4", "RCtrl"};
    static constexpr ImGuiKey kHotkeys[] = {ImGuiKey_Space, ImGuiKey_F2, ImGuiKey_F4,
                                            ImGuiKey_RightCtrl};
    static constexpr int kHotkeyCount =
        static_cast<int>(sizeof(kHotkeys) / sizeof(kHotkeys[0]));

    RealEngineControl*  control_ = nullptr;  // non-owning
    SpeechPipelineHandle pipe_ = nullptr;    // non-owning

    // UI-thread-only widget state.
    bool mode_bounded_ = false;
    int  history_budget_ = 256;
    int  src_idx_ = 0;
    int  tgt_idx_ = 0;
    int  silence_ms_ = 800;
    bool manual_only_ = false;
    int  hotkey_idx_ = 0;
    bool ptt_down_ = false;
};

}  // namespace rt
