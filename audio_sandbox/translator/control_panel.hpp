#pragma once
// -----------------------------------------------------------------------------
// translator/control_panel.hpp — the "Translator Settings" ImGui panel for
// audio_translator. Grouped into collapsing sections, each one a subsystem:
// VAD & audio gating, streaming & eviction, context & task, languages, hotkey.
//
// NO MAGIC NUMBERS HERE. Every streaming/eviction widget reads its default AND
// its slider bounds from ContinuousStreamingConfig's own constants, so a value
// the UI can produce is by construction a value the pipeline accepts. Adding a
// knob means adding it there, not here.
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
#include "continuous_streaming_config.hpp"  // ContinuousStreamingConfig + Live publisher
#include "language_table.hpp"               // rt::kLanguages
#include "real_engine_control.hpp"          // RealEngineControl atomics (setters)
#include "silero_vad.hpp"                   // blackwell::vad::SileroVAD (live readout)

namespace rt {

class ControlPanel {
public:
    // All non-owning; the app guarantees they outlive the UI loop. The initial
    // widget values are read back from the control's atomics so CLI defaults
    // (--context-mode/--src-lang/...) show up pre-selected.
    //
    // `vad` + `vad_fn` are the neural-VAD seam and may be null/null (the app
    // failed to load the model, or --no-neural-vad): the panel then reports that
    // the built-in RMS threshold detector is running and disables the controls.
    // The panel only READS the SileroVAD's published atomics — it never calls
    // feed() or reset_state(), which belong to the DSP worker thread.
    // `streaming` is the re-translation config publisher and may be null (the
    // pipeline is not wired yet): the section then says so instead of offering
    // sliders that go nowhere. `context_length` / `max_new_tokens` are only used
    // for the headroom readout; pass 0 to skip that check.
    ControlPanel(RealEngineControl* control, SpeechPipelineHandle pipe,
                 int initial_silence_ms, blackwell::vad::SileroVAD* vad = nullptr,
                 SpeechVadScoreFn vad_fn = nullptr,
                 blackwell::bridge::LiveStreamingConfig* streaming = nullptr,
                 int context_length = 0, int max_new_tokens = 0) noexcept
        : control_(control), pipe_(pipe), vad_(vad), vad_fn_(vad_fn),
          streaming_(streaming), context_length_(context_length),
          max_new_tokens_(max_new_tokens) {
        neural_vad_on_ = (vad_ != nullptr && vad_fn_ != nullptr);
        if (vad_ != nullptr) vad_threshold_ = vad_->threshold();
        mode_bounded_ = control_->context_mode() ==
                        RealEngineControl::ContextMode::BoundedHistory;
        history_budget_ = control_->history_budget_tokens();
        src_idx_ = control_->source_language_index();
        tgt_idx_ = control_->target_language_index();
        task_transcribe_ = control_->task_transcribe();
        task_translate_ = control_->task_translate();
        // Seed the widget mirror from the publisher, so CLI-supplied values show
        // up pre-selected exactly like the language/mode dropdowns do.
        if (streaming_ != nullptr) cfg_ = streaming_->load();
        // The hangover is ONE knob: the segmenter's release window and the
        // pipeline's silence timeout are the same boundary. The CLI still owns the
        // initial value, so adopt it rather than overwriting it from the default.
        if (initial_silence_ms > 0) cfg_.hangover_ms = initial_silence_ms;
        cfg_.clamp();
        publish_streaming();
    }

    // UI thread, once per ImGui frame.
    void draw() {
        ImGui::SetNextWindowPos(ImVec2(444.0f, 12.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(430.0f, 560.0f), ImGuiCond_FirstUseEver);
        ImGui::Begin("Translator Settings");

        if (ImGui::CollapsingHeader("VAD & Audio Gating", ImGuiTreeNodeFlags_DefaultOpen)) {
            draw_vad_section();
            ImGui::Spacing();
            draw_gating_section();
        }
        if (ImGui::CollapsingHeader("Streaming & Eviction", ImGuiTreeNodeFlags_DefaultOpen)) {
            draw_retranslation_section();
            ImGui::Spacing();
            draw_encoder_streaming_section();
        }
        if (ImGui::CollapsingHeader("Context & Task")) {
            draw_mode_section();
            ImGui::Spacing();
            draw_task_section();
        }
        if (ImGui::CollapsingHeader("Languages")) {
            draw_language_section();
        }
        if (ImGui::CollapsingHeader("Push-to-talk", ImGuiTreeNodeFlags_DefaultOpen)) {
            draw_hotkey_section();
        }

        ImGui::End();

        // The hotkey must work even when this window is not focused; poll it at
        // frame scope, but never while a text widget owns the keyboard.
        poll_push_to_talk();
    }

private:
    // Single publish path. clamp() runs FIRST, so the widget mirror the next frame
    // draws from is already legal — the UI cannot display a configuration the
    // pipeline would reject, and the target <= high_water invariant is repaired in
    // the mirror rather than only in the publisher.
    void publish_streaming() {
        cfg_.clamp();
        if (streaming_ != nullptr) streaming_->store(cfg_);
    }
    void draw_mode_section() {
        ImGui::SeparatorText("Context mode");
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

    void draw_encoder_streaming_section() {
        ImGui::SeparatorText("Encoder mode (applies from the next utterance)");
        if (!control_->center_slice_available()) {
            // The CenterSlice geometry is resolved at launch (tier-3 plan); without
            // it the toggle would be a lie, so say how to arm it instead.
            ImGui::TextDisabled("center-slice plan not armed — launch with"
                                " --streaming --stream-mode center");
        } else {
            bool center = control_->live_center_slice();
            if (ImGui::RadioButton("Whole utterance (encode at pause)", !center)) {
                control_->set_live_center_slice(false);
            }
            if (ImGui::RadioButton("Center-slice streaming (append-only)", center)) {
                control_->set_live_center_slice(true);
            }
        }
        // Live latency readout: both commit paths record it, so flipping the mode
        // above gives a direct A/B on the very next utterance.
        const float ttft = control_->last_ttft_ms();
        if (ttft > 0.0f) {
            ImGui::TextDisabled("last TTFT (VAD -> first token): %.0f ms", ttft);
        } else {
            ImGui::TextDisabled("last TTFT: n/a (no utterance yet)");
        }
    }

    // Voice activity detection: which detector decides "is this speech?", and how
    // readily it says yes. Everything downstream of that verdict — the silence
    // timeout below, push-to-talk, background listening, the ping-pong swap
    // policy — is unchanged by this choice and keeps running either way.
    //
    // The toggle is live: installing or clearing the scorer is one atomic store
    // in the pipeline, effective on the next 10 ms block, so the two detectors
    // can be A/B'd mid-conversation.
    void draw_vad_section() {
        ImGui::SeparatorText("Detector");
        if (vad_ == nullptr || vad_fn_ == nullptr) {
            ImGui::TextDisabled("neural VAD unavailable — RMS threshold detector active");
            ImGui::TextDisabled("(launch without --no-neural-vad, and check "
                                "--vad-model points at silero_vad.onnx)");
            return;
        }

        if (ImGui::Checkbox("Neural VAD (Silero)", &neural_vad_on_)) {
            (void)speech_pipeline_set_vad_scorer(pipe_, neural_vad_on_ ? vad_fn_ : nullptr,
                                                 neural_vad_on_ ? vad_ : nullptr);
        }

        // SENSITIVITY. The slider is the probability a block must exceed to count
        // as a speech onset. The pipeline derives the release threshold 0.15
        // below it, so utterances in progress survive a dip without chattering.
        if (!neural_vad_on_) ImGui::BeginDisabled();
        ImGui::SetNextItemWidth(200.0f);
        if (ImGui::SliderFloat("VAD sensitivity", &vad_threshold_, 0.1f, 0.9f, "%.2f")) {
            vad_->set_threshold(vad_threshold_);
            (void)speech_pipeline_set_vad_threshold(pipe_, vad_threshold_);
        }
        ImGui::TextDisabled("lower = triggers on quieter/less certain speech; "
                            "higher = rejects more non-speech");

        // Live readout. A sensitivity slider is untunable without seeing what it
        // is being compared against, so show the probability and where the
        // threshold sits on the same bar.
        const float p = vad_->last_probability();
        ImGui::ProgressBar(p, ImVec2(200.0f, 0.0f));
        ImGui::SameLine();
        ImGui::TextUnformatted(p > vad_threshold_ ? "SPEECH" : "silence");
        if (!neural_vad_on_) ImGui::EndDisabled();

        const uint64_t errors = vad_->inference_errors();
        if (errors != 0) {
            // Non-zero means inferences are failing and the last-good probability
            // is being reported instead — the detector is degraded, not dead.
            ImGui::TextDisabled("WARNING: %llu failed inference(s); running degraded",
                                static_cast<unsigned long long>(errors));
        } else if (!neural_vad_on_) {
            ImGui::TextDisabled("off: the built-in RMS threshold detector decides boundaries");
        }
    }

    // Narrowing the task is the cheapest cognitive-load cut available for the 8B
    // backbone: one output instead of two. The toggles only flip atomics the
    // engine thread reads when it builds the NEXT user turn's prefix — nothing
    // cached is invalidated (the task directive never enters the frozen system
    // prefix), so a flip costs nothing.
    void draw_task_section() {
        ImGui::SeparatorText("Task (applies from the next utterance)");
        bool changed = ImGui::Checkbox("Transcribe", &task_transcribe_);
        ImGui::SameLine();
        changed = ImGui::Checkbox("Translate", &task_translate_) || changed;
        if (changed) {
            // The control coerces "neither" to transcribe-only; mirror that back
            // into the widgets so the panel never displays an impossible state.
            if (!task_transcribe_ && !task_translate_) task_transcribe_ = true;
            control_->set_tasks(task_transcribe_, task_translate_);
        }
        // The exact tags the model is being held to — the live answer to "do the
        // expected output tags match the selected mode?".
        ImGui::TextDisabled("expects: %s", control_->expected_output_format().c_str());
    }

    void draw_language_section() {
        ImGui::SeparatorText("Languages (apply from the next utterance)");
        ImGui::SetNextItemWidth(160.0f);
        const bool s = ImGui::Combo("Source", &src_idx_, kLanguages, kLanguageCount);
        ImGui::SetNextItemWidth(160.0f);
        const bool t = ImGui::Combo("Target", &tgt_idx_, kLanguages, kLanguageCount);
        if (s || t) control_->set_languages(src_idx_, tgt_idx_);
    }

    // WHERE an utterance's edges fall. The pre-roll and the hangover are the two
    // halves of that question, and both are segmenter policy — nothing here can
    // change WHETHER a block is speech, only how much audio surrounds the verdict.
    void draw_gating_section() {
        ImGui::SeparatorText("Utterance boundaries");
        using Cfg = blackwell::bridge::ContinuousStreamingConfig;

        ImGui::SetNextItemWidth(200.0f);
        if (ImGui::SliderInt("Pre-roll (ms)", &cfg_.pre_roll_ms,
                             Cfg::kMinPreRollMs, Cfg::kMaxPreRollMs)) {
            publish_streaming();
        }
        ImGui::TextDisabled("audio kept BEFORE the onset: the encoder is weakest at a");
        ImGui::TextDisabled("window's left edge, so never start it on the first phoneme");

        // ONE knob for the release window. The segmenter's hangover and the
        // pipeline's silence timeout are the same boundary, so the slider drives
        // both rather than leaving two controls to disagree.
        ImGui::SetNextItemWidth(200.0f);
        if (ImGui::SliderInt("Hangover / silence (ms)", &cfg_.hangover_ms,
                             Cfg::kMinHangoverMs, Cfg::kMaxHangoverMs)) {
            publish_streaming();
            (void)speech_pipeline_set_silence_hangover_ms(
                pipe_, static_cast<uint32_t>(cfg_.hangover_ms));
        }
        ImGui::TextDisabled("silence that ends an utterance and triggers the commit");

        // Manual mode mutes the WHOLE auto-VAD (onset + barge-in + auto-commit)
        // in the pipeline, not just the hangover: state transitions then come
        // exclusively from the push-to-talk press/release events, so a threshold
        // trigger can never race a hotkey mid-utterance.
        if (ImGui::Checkbox("Manual only (push-to-talk, auto-VAD muted)", &manual_only_)) {
            (void)speech_pipeline_set_manual_mode(pipe_, manual_only_);
        }
    }

    // The re-translation loop's budget (docs/CONTINUOUS_STREAMING.md). These do
    // not change WHAT is translated, only how often the draft is rebuilt and how
    // much committed history survives behind the commit pointer.
    void draw_retranslation_section() {
        ImGui::SeparatorText("Re-translation (draft & commit)");
        if (streaming_ == nullptr) {
            ImGui::TextDisabled("continuous streaming not wired — these knobs are inert");
            return;
        }
        using Cfg = blackwell::bridge::ContinuousStreamingConfig;

        ImGui::SetNextItemWidth(200.0f);
        if (ImGui::SliderInt("Redraft cadence (ms)", &cfg_.partial_cadence_ms,
                             Cfg::kMinPartialCadenceMs, Cfg::kMaxPartialCadenceMs)) {
            publish_streaming();
        }
        ImGui::TextDisabled("every redraft re-prefills the utterance and re-decodes it:");
        ImGui::TextDisabled("lower = fresher partial text, higher = more GPU headroom");

        ImGui::SetNextItemWidth(200.0f);
        if (ImGui::SliderInt("Max utterance (ms)", &cfg_.max_utterance_ms,
                             Cfg::kMinMaxUtteranceMs, Cfg::kMaxMaxUtteranceMs)) {
            publish_streaming();
        }
        ImGui::TextDisabled("forced commit for a speaker who never pauses");

        ImGui::SeparatorText("KV eviction watermarks");
        // GC-style watermarks: evict AT high water, reclaim down TO target. The
        // gap between them is the hysteresis, so a wide gap means rare, large
        // compactions instead of a compaction on every commit.
        ImGui::SetNextItemWidth(200.0f);
        if (ImGui::DragInt("High water (tokens)", &cfg_.eviction_high_water_mark, 8.0f,
                           Cfg::kMinHighWaterTokens, Cfg::kMaxHighWaterTokens)) {
            publish_streaming();   // clamp() pulls target down if it dragged under
        }
        ImGui::SetNextItemWidth(200.0f);
        // Bounded by the CURRENT high water, so the invariant is enforced at the
        // widget as well as in clamp() — the slider cannot even express the
        // illegal state, rather than silently snapping back a frame later.
        if (ImGui::DragInt("Target / low water (tokens)", &cfg_.eviction_target_tokens, 8.0f,
                           Cfg::kMinTargetTokens, cfg_.eviction_high_water_mark)) {
            publish_streaming();
        }
        ImGui::TextDisabled("evict at high water, reclaim down to target (frees %d tokens)",
                            cfg_.eviction_high_water_mark - cfg_.eviction_target_tokens);

        draw_headroom_readout();
    }

    // The draft zone lives ABOVE the high water mark and eviction never touches
    // it, so its worst case is headroom the context length has to cover. Getting
    // this wrong does not misbehave gracefully: a redraft runs off the end of the
    // cache between two evictions. Show the arithmetic rather than assume it.
    void draw_headroom_readout() {
        if (context_length_ <= 0) return;
        const std::uint32_t draft = cfg_.draft_headroom_tokens(max_new_tokens_);
        const long long need =
            static_cast<long long>(cfg_.eviction_high_water_mark) + static_cast<long long>(draft);
        ImGui::TextDisabled("headroom: %d high water + %u draft = %lld of %d context",
                            cfg_.eviction_high_water_mark, draft, need, context_length_);
        if (!cfg_.fits_context(context_length_, max_new_tokens_)) {
            ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.35f, 1.0f),
                               "WARNING: exceeds context by %lld tokens — lower the high",
                               need - static_cast<long long>(context_length_));
            ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.35f, 1.0f),
                               "water mark or the max utterance, or raise --max-context");
        }
    }

    void draw_hotkey_section() {
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
            // Press: mute the auto-VAD FIRST, then open (or barge into) an
            // utterance boundary. Ordering matters — while the key is held the
            // hold is exclusively key-driven, so a threshold onset/auto-commit
            // can never fire in parallel and race this press (even in auto mode).
            (void)speech_pipeline_set_manual_mode(pipe_, true);
            (void)speech_pipeline_on_speech_start(pipe_);
        } else if (!down && ptt_down_) {
            // Release: flush the buffered audio and decode immediately (a no-op
            // unless the pipeline is PREFILL_SPEAKING, so a stray tap is safe),
            // THEN restore the checkbox's mode — the release event itself must
            // still be the one that commits, never a revived auto-trigger.
            (void)speech_pipeline_on_silence_timeout(pipe_);
            (void)speech_pipeline_set_manual_mode(pipe_, manual_only_);
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
    // Neural VAD seam (both null when it is unavailable). The panel reads the
    // detector's atomics and installs/clears it; the DSP worker owns feeding it.
    blackwell::vad::SileroVAD* vad_ = nullptr;  // non-owning
    SpeechVadScoreFn vad_fn_ = nullptr;
    // Re-translation config publisher (null until the pipeline is wired). The
    // panel owns the widget MIRROR below and publishes into this; it never reads
    // back mid-session, so a drag cannot fight the engine thread for the value.
    blackwell::bridge::LiveStreamingConfig* streaming_ = nullptr;  // non-owning
    int context_length_ = 0;    // readout only (headroom check)
    int max_new_tokens_ = 0;    // readout only (draft worst case)

    // UI-thread-only widget state.
    blackwell::bridge::ContinuousStreamingConfig cfg_{};
    bool mode_bounded_ = false;
    int  history_budget_ = 256;
    int  src_idx_ = 0;
    int  tgt_idx_ = 0;
    bool task_transcribe_ = true;
    bool task_translate_ = true;
    bool manual_only_ = false;
    int  hotkey_idx_ = 0;
    bool ptt_down_ = false;
    bool  neural_vad_on_ = false;    // seeded in the ctor from whether the seam exists
    float vad_threshold_ = 0.5f;     // seeded from the detector's own threshold
};

}  // namespace rt
