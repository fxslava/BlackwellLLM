#pragma once
// -----------------------------------------------------------------------------
// transcript_view.hpp — thread-safe streaming transcript/translation buffer +
// its ImGui panel for audio_translator.
//
// THREADING
//   * on_token()/on_final() are called FROM the engine thread (via the speech
//     pipeline's token callback). They only append under a short mutex.
//   * set_state() is called from whichever thread performs a VAD/decode
//     transition (audio-VAD thread or engine thread); it stores one atomic.
//   * draw() runs on the UI thread inside the ImGui frame. It copies the text
//     under the lock, then renders outside it.
//   No ImGui call ever happens under the lock, and the engine/audio threads never
//   touch ImGui — the UI thread is the only ImGui caller (ImGui is not reentrant).
// -----------------------------------------------------------------------------
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include <atomic>

#include "imgui.h"

#include "bridge/speculative_bridge_api.h"  // SpeechPipelineState

namespace rt {

class TranscriptView {
public:
    // ---- engine thread: streamed pieces of the current utterance --------------
    // Each utterance is tagged with its generation id; a barge-in bumps the gen,
    // so a piece from a new gen flushes the (interrupted) previous utterance to
    // history and starts a fresh live line.
    void on_token(const char* text, uint64_t gen) {
        std::lock_guard<std::mutex> lk(m_);
        if (gen != live_gen_) {
            flush_live_locked();
            live_gen_ = gen;
        }
        if (text != nullptr) live_ += text;
    }

    // Utterance finished cleanly (EOS/cap): commit the live line to history.
    void on_final(uint64_t gen) {
        std::lock_guard<std::mutex> lk(m_);
        if (gen == live_gen_) flush_live_locked();
    }

    // ---- any thread: pipeline status badge ------------------------------------
    void set_state(SpeechPipelineState s) noexcept {
        state_.store(s, std::memory_order_release);
    }

    // ---- UI thread: render the "Live Transcription & Translation" window -------
    void draw() {
        // Snapshot the text under the lock; render outside it.
        std::vector<std::string> lines;
        std::string live;
        {
            std::lock_guard<std::mutex> lk(m_);
            lines = history_;
            live = live_;
        }

        ImGui::SetNextWindowPos(ImVec2(12.0f, 324.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(560.0f, 300.0f), ImGuiCond_FirstUseEver);
        ImGui::Begin("Live Transcription & Translation");

        draw_status_badge();
        ImGui::Checkbox("Auto-scroll", &autoscroll_);
        ImGui::Separator();

        ImGui::BeginChild("transcript_scroll", ImVec2(0.0f, 0.0f), /*border=*/true,
                          ImGuiWindowFlags_HorizontalScrollbar);
        for (const std::string& utt : lines) draw_utterance(utt, /*dim=*/true);
        if (!live.empty()) draw_utterance(live, /*dim=*/false);
        if (autoscroll_ && ImGui::GetScrollY() >= ImGui::GetScrollMaxY()) {
            ImGui::SetScrollHereY(1.0f);
        }
        ImGui::EndChild();

        ImGui::End();
    }

private:
    // Move the in-progress line into history (drops nothing; an interrupted
    // partial is preserved so the barge-in is visible). Caller holds m_.
    void flush_live_locked() {
        if (!live_.empty()) {
            history_.push_back(live_);
            if (history_.size() > kMaxHistory) history_.erase(history_.begin());
            live_.clear();
        }
    }

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
    static constexpr std::size_t kMaxHistory = 200;   // cap the scrollback

    mutable std::mutex m_;
    std::vector<std::string> history_;          // completed/interrupted utterances
    std::string live_;                          // current utterance accumulation
    uint64_t live_gen_ = UINT64_MAX;            // gen of the live_ line (UINT64_MAX = none)

    std::atomic<SpeechPipelineState> state_{SPEECH_STATE_IDLE};
    bool autoscroll_ = true;                    // UI-thread only
};

}  // namespace rt
