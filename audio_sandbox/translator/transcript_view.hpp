#pragma once
// -----------------------------------------------------------------------------
// transcript_view.hpp — the ImGui panel over TranscriptModel for audio_translator.
//
// This file is now PRESENTATION ONLY. The utterance storage, the identity rules
// and the scrollback cap moved to transcript_model.hpp (STL-only, unit-tested);
// what stays here is the pipeline-state badge and the drawing.
//
// The public surface is deliberately UNCHANGED — on_token / on_final / set_state
// are still the methods the speech-pipeline callbacks in main.cpp bind to, and
// they still mean exactly what they did. The split is invisible to callers.
//
// THREADING
//   * on_token()/on_final() are called FROM the engine thread and forward
//     straight into the model (short mutex, no ImGui).
//   * set_state() is called from whichever thread performs a VAD/decode
//     transition (audio-VAD thread or engine thread); it stores one atomic.
//   * draw() runs on the UI thread inside the ImGui frame. It snapshots the model
//     under the lock, then renders outside it.
//   No ImGui call ever happens under the lock, and the engine/audio threads never
//   touch ImGui — the UI thread is the only ImGui caller (ImGui is not reentrant).
// -----------------------------------------------------------------------------
#include <cstdint>
#include <string>

#include <atomic>

#include "imgui.h"

#include "bridge/speculative_bridge_api.h"  // SpeechPipelineState

#include "transcript_model.hpp"

namespace rt {

class TranscriptView {
public:
    // ---- engine thread: streamed pieces of the current utterance --------------
    void on_token(const char* text, uint64_t gen) { model_.on_token(text, gen); }

    // Utterance finished cleanly (EOS/cap): commit the live line to history.
    void on_final(uint64_t gen) { model_.on_final(gen); }

    // ---- any thread: pipeline status badge ------------------------------------
    void set_state(SpeechPipelineState s) noexcept {
        state_.store(s, std::memory_order_release);
    }

    // The storage, for consumers that need to resolve an utterance id (the TTS
    // request path). Exposed as a reference because the model carries its own
    // synchronisation — there is no lock for a caller to get wrong.
    TranscriptModel& model() noexcept { return model_; }
    const TranscriptModel& model() const noexcept { return model_; }

    // ---- UI thread: render the "Live Transcription & Translation" window -------
    void draw() {
        // Snapshot under the lock (inside the model), render outside it. snap_ is
        // a member so its buffers survive across frames and the per-frame copy
        // does not allocate.
        model_.snapshot(snap_);

        ImGui::SetNextWindowPos(ImVec2(12.0f, 324.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(560.0f, 300.0f), ImGuiCond_FirstUseEver);
        ImGui::Begin("Live Transcription & Translation");

        draw_status_badge();
        ImGui::Checkbox("Auto-scroll", &autoscroll_);
        ImGui::Separator();

        ImGui::BeginChild("transcript_scroll", ImVec2(0.0f, 0.0f), /*border=*/true,
                          ImGuiWindowFlags_HorizontalScrollbar);
        for (const Utterance& u : snap_.history) {
            // Scope every committed line to its STABLE id, not its position, so
            // per-utterance widgets keep their ImGui identity as the cap trims
            // older entries out from under them.
            ImGui::PushID(static_cast<int>(u.id));
            draw_utterance(u.text, /*dim=*/true);
            ImGui::PopID();
        }
        if (!snap_.live.empty()) draw_utterance(snap_.live, /*dim=*/false);
        if (autoscroll_ && ImGui::GetScrollY() >= ImGui::GetScrollMaxY()) {
            ImGui::SetScrollHereY(1.0f);
        }
        ImGui::EndChild();

        ImGui::End();
    }

private:
    void draw_status_badge() const {
        const SpeechPipelineState s = state_.load(std::memory_order_acquire);
        const char* label = "IDLE";
        ImVec4 color(0.65f, 0.65f, 0.65f, 1.0f);
        switch (s) {
            case SPEECH_STATE_IDLE:
                label = "IDLE";
                color = ImVec4(0.65f, 0.65f, 0.65f, 1.0f);
                break;
            case SPEECH_STATE_PREFILL_SPEAKING:
                label = "LISTENING";
                color = ImVec4(0.35f, 0.85f, 0.40f, 1.0f);
                break;
            case SPEECH_STATE_DECODE_TRANSLATING:
                label = "TRANSLATING";
                color = ImVec4(0.40f, 0.80f, 1.00f, 1.0f);
                break;
            case SPEECH_STATE_INTERRUPTION_REWIND:
                label = "INTERRUPTED";
                color = ImVec4(1.00f, 0.45f, 0.35f, 1.0f);
                break;
        }
        ImGui::TextUnformatted("Status:");
        ImGui::SameLine();
        ImGui::TextColored(color, "[%s]", label);
    }

    // Render one utterance, splitting the "[Speech] ... | [Translation] ..." format
    // into a source line and a Russian-translation line when the delimiter exists.
    static void draw_utterance(const std::string& utt, bool dim) {
        const float a = dim ? 0.60f : 1.0f;
        const std::string::size_type cut = utt.find(kDelimiter);
        if (cut == std::string::npos) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.90f, 0.90f, 0.90f, a));
            ImGui::TextWrapped("%s", utt.c_str());
            ImGui::PopStyleColor();
            return;
        }
        const std::string speech = utt.substr(0, cut);
        const std::string translation = utt.substr(cut + 3);  // skip " | "
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.90f, 0.90f, 0.90f, a));
        ImGui::TextWrapped("%s", speech.c_str());
        ImGui::PopStyleColor();
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.55f, 0.85f, 1.00f, a));
        ImGui::TextWrapped("%s", translation.c_str());
        ImGui::PopStyleColor();
    }

    static constexpr const char* kDelimiter = " | ";  // between [Speech] and [Translation]

    TranscriptModel model_;
    TranscriptModel::Snapshot snap_;            // UI-thread only; reused per frame

    std::atomic<SpeechPipelineState> state_{SPEECH_STATE_IDLE};
    bool autoscroll_ = true;                    // UI-thread only
};

}  // namespace rt
