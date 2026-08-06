// web_ui_bridge.cpp — see web_ui_bridge.hpp.
//
// MOVED, NOT CHANGED. The callback bodies, the live/restart tier split, the mute
// edge, the device-reload gate and every log line are byte-for-byte what main()
// ran. What changed is only WHERE they read their state from: a member reference
// instead of a lambda capture.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include "web_ui_bridge.hpp"

#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <utility>

namespace rt {

WebUIBridge::WebUIBridge(AssistantSettings& settings, AppContext& ctx,
                         AppLifecycleManager& lifecycle, AudioPipelineBinder& audio,
                         ConversationRouter& router, AssistantView& view)
    : settings_(settings),
      ctx_(ctx),
      lifecycle_(lifecycle),
      audio_(audio),
      router_(router),
      view_(view),
      applied_audio_task_(settings.audio_task_prompt),
      applied_speech_language_(settings.speech_language) {
    // The view's ONLY output is JSON; the window's post_event is the only thing
    // that ever crosses back to the UI thread.
    view_.set_sink([this](std::string json) { window_.post_event(std::move(json)); });
    window_.set_settings(settings_);

    // Where the router reports a system-prompt rebuild's verdict. Bound here
    // because the window is what has to show it.
    router_.set_prompt_applied_sink(
        [this](bool ok, unsigned tokens, const std::string& detail) {
            window_.post_system_prompt_applied(ok, tokens, detail);
        });

    install_callbacks();
}

WebUIBridge::~WebUIBridge() { stop_diagnostics_poller(); }

void WebUIBridge::install_callbacks() {
    AssistantWindowCallbacks cb;

    cb.on_send_text = [this](const std::string& text) { router_.submit_typed_turn(text); };

    cb.on_mic_toggle = [this](bool listening) {
        // Manual mode mutes the VAD for ALL transitions while PCM keeps flowing,
        // which is exactly "mic off" without tearing the stream down. No explicit
        // on_speech_start follows, so nothing commits. Through the INTERFACE: both
        // modes implement it, with the same meaning on each (Mode C's
        // WhisperCascadeMode::set_manual_mode says what it does and does not do).
        lifecycle_.mode()->set_manual_mode(!listening);
    };

    cb.on_cancel = [this] { router_.cancel_in_flight(); };

    cb.on_session_list_request = [this] { router_.push_session_list(); };
    cb.on_session_select = [this](const std::string& id) { router_.select_session(id); };
    cb.on_session_new = [this] { router_.new_session(); };
    cb.on_session_delete = [this](const std::string& id) { router_.delete_session(id); };

    cb.on_settings_apply = [this](const AssistantSettings& next, bool live_only) {
        on_settings_apply(next, live_only);
    };
    cb.on_system_prompt_apply = [this](const std::string& persona) {
        router_.apply_system_prompt(persona);
    };
    cb.on_audio_hot_update = [this](const AudioHotUpdate& u) { on_audio_hot_update(u); };

    cb.on_test_tone = [this] {
#if defined(VOICE_ASSISTANT_HAS_TTS)
        // Through the live TTS pipeline, not a second private device: the point of
        // the button is to prove the endpoint the ASSISTANT speaks through makes
        // sound. A tone on its own device could pass while speech stayed silent,
        // which is exactly the confusion it exists to end.
        if (ctx_.tts != nullptr) ctx_.tts->PlayTestTone();
#endif
    };

    cb.on_restart = [this] {
        restart_requested_.store(true, std::memory_order_release);
        if (window_.hwnd() != nullptr) PostMessageW(window_.hwnd(), WM_CLOSE, 0, 0);
    };

    window_.set_callbacks(std::move(cb));
}

// Live-tier settings, applied to the RUNNING system. Everything here is an atomic
// store read at the next VAD block or turn boundary -- no marshaling, no engine
// work, doctrine intact (see settings_store.hpp) -- with the ONE documented
// exception of the audio task prefix, which is a KV rebuild and is marshaled.
void WebUIBridge::apply_live_settings(const AssistantSettings& s) {
    // Through the INTERFACE, not through Mode A's pipeline handle. Each mode
    // honours the knobs it actually has and ignores the rest (see ISpeechMode) --
    // which is the only arrangement that works now that there are two modes with
    // different sets of them.
    ISpeechMode* mode = lifecycle_.mode();
    mode->set_vad_threshold(s.vad_threshold);
    mode->set_silence_hangover_ms(static_cast<std::uint32_t>(s.silence_hangover_ms));
    mode->set_warm_prefill_interval_ms(static_cast<std::uint32_t>(s.warm_prefill_interval_ms));
    // The pre-roll goes to the CONTROL, not the pipeline: the flush it sizes
    // happens on the engine thread, which is the only consumer allowed to move the
    // ring's read cursor. It is on the base bridge, so it applies to whichever
    // backend is live without a branch.
    lifecycle_.control()->set_pre_roll_ms(s.pre_roll_ms);

    // The AEC enable flag and the mute-folded speaker volume, plus the mic gain.
    audio_.apply_live_settings(s);

    // The sampling knobs and the reply ceiling exist on BOTH controls (the
    // simulated one honours the ceiling and stores the rest), so they are applied
    // without asking which backend is live -- the only branch left is the handful
    // of settings that are genuinely real-engine-only.
    if (RealEngineControl* rc = lifecycle_.real_control(); rc != nullptr) {
        rc->set_sampling(s.temperature, s.top_p);
        rc->set_max_new_tokens(s.max_new_tokens);
        rc->set_context_mode(s.context_mode == "bounded"
                                 ? RealEngineControl::ContextMode::BoundedHistory
                                 : RealEngineControl::ContextMode::Stateless);
        rc->set_history_budget_tokens(s.history_budget_tokens);
        rc->set_live_center_slice(s.live_streaming);
        // The task TOGGLES stay a plain store: they only change the generated
        // directives in the per-turn user block.
        rc->set_tasks(/*transcribe=*/s.speech_task != "translate",
                      /*translate=*/s.speech_task != "transcribe");

        // The AUDIO TASK PROMPT and the SPOKEN LANGUAGE compose the transcription
        // session's frozen system prefix, so editing either is a KV rebuild on that
        // sequence, not a store -- exactly like the persona prompt on the chat
        // sequence. Marshal it (CUDA work belongs to the engine thread) and only
        // when one of them actually changed: a settings push happens on every panel
        // edit, and re-prefilling the prefix on each one would drop a live
        // utterance's audio for nothing. Without isolation there is no second
        // prefix, both values ride in the per-turn user block, and the stores below
        // are the whole operation.
        if (lifecycle_.real_stack().isolated_sessions) {
            if (s.audio_task_prompt != applied_audio_task_ ||
                s.speech_language != applied_speech_language_) {
                applied_audio_task_ = s.audio_task_prompt;
                applied_speech_language_ = s.speech_language;
                audio_prefix_dirty_ = true;
            }
            if (audio_prefix_dirty_) {
                const std::string prompt = applied_audio_task_;
                const std::string language = applied_speech_language_;
                if (lifecycle_.control()->post_engine_task([rc, prompt, language] {
                        try {
                            // ORDER: the language is folded INTO the text the
                            // rebuild prefills, so it has to be current before the
                            // rebuild reads it.
                            rc->set_speech_language(language);
                            rc->rebuild_audio_task_prompt(prompt);
                        } catch (const std::exception& e) {
                            std::fprintf(stderr,
                                         "[session] audio task prefix rebuild failed: %s\n",
                                         e.what());
                        }
                    })) {
                    audio_prefix_dirty_ = false;
                }
                // Otherwise the task queue was full: stay dirty so the next push
                // retries rather than silently running the old prefix forever.
            }
        } else {
            rc->set_audio_task_prompt(s.audio_task_prompt);
            rc->set_speech_language(s.speech_language);
        }
    } else if (SimulatedEngineControl* sc = lifecycle_.simulated_control(); sc != nullptr) {
        sc->set_sampling(s.temperature, s.top_p);
        sc->set_max_new_tokens(s.max_new_tokens);
    }

    // Where the NEXT intent gets answered, and the badge that has to follow it.
    router_.set_use_local(s.local_inference);
}

void WebUIBridge::on_settings_apply(const AssistantSettings& next, bool live_only) {
    // THE HOT-RELOAD CHECK, before `settings_` is overwritten -- it is a comparison
    // against the values currently in force, so it has to run while those are still
    // readable.
    //
    // Reopening two ma_devices takes tens of milliseconds and touches no VRAM: the
    // engine keeps its 5.3 GB of weights, the KV pool stays allocated and the F5
    // ONNX session is not reloaded. That is the entire reason these settings are
    // their own tier.
    const bool audio_moved = requires_audio_reload(settings_, next);
    settings_ = next;
    if (!save_settings(settings_)) {
        std::fprintf(stderr, "[settings] WARN: could not write %s\n", settings_path().c_str());
    }
    // Live settings apply on EVERY save, restart-tier or not: refusing to update
    // the VAD threshold because an unrelated checkpoint path also changed would be
    // a worse answer than doing what can be done now. `live_only` decides whether
    // the app also restarts, nothing else.
    apply_live_settings(settings_);
    if (audio_moved) audio_.apply_audio_reload(settings_);
    (void)live_only;
    // Restart-tier changes are persisted here and picked up by the next process;
    // the app keeps running on the OLD engine until then, which is the honest state
    // and is what the modal's banner says.
}

void WebUIBridge::on_audio_hot_update(const AudioHotUpdate& u) {
    // Merge into the live settings. ONLY these fields are touched, so whatever the
    // settings modal is holding cannot be dragged along -- which is exactly the bug
    // this path exists to remove.
    if (u.output_device_index != AudioHotUpdate::kNoIndex) {
        settings_.output_device_index = u.output_device_index;
    }
    if (u.input_device_index != AudioHotUpdate::kNoIndex) {
        settings_.input_device_index = u.input_device_index;
    }
    if (u.has_output_name) settings_.output_device_name = u.output_device_name;
    if (u.has_input_name)  settings_.input_device_name  = u.input_device_name;
    if (u.tts_volume >= 0.0f) settings_.tts_volume = u.tts_volume;
    if (u.mic_gain   >= 0.0f) settings_.mic_gain   = u.mic_gain;
    // Mute is session state and is deliberately NOT merged into `settings_` -- see
    // AudioHotUpdate::tts_muted on why it must not survive a restart. It lives on
    // ctx_, which the save below therefore cannot reach.
    if (u.tts_muted != AudioHotUpdate::Tri::Absent) {
        const bool now_muted = (u.tts_muted == AudioHotUpdate::Tri::On);
        const bool was_muted = ctx_.tts_muted.exchange(now_muted, std::memory_order_acq_rel);
#if defined(VOICE_ASSISTANT_HAS_TTS)
        // THE MUTE EDGE. Everything above stops speech that has not STARTED yet; a
        // mute tapped mid-sentence has to stop the sentence in the air, and only
        // barge-in can -- it aborts the solver mid-step, drops the buffered text,
        // and bumps the speak epoch so the device drops the queued audio on its
        // next pull instead of finishing the phrase at zero gain.
        //
        // On the EDGE, not on every muted update: a volume drag arrives as dozens
        // of messages, and re-cancelling on each is only harmless because Cancel is
        // idempotent -- which is not a reason to do it. Unmuting is deliberately
        // NOT the mirror image: the answer that was cancelled is gone, and speech
        // re-arms at the next turn's dispatch-start rather than resurrecting a
        // reply from the middle of a word.
        if (now_muted && !was_muted && ctx_.tts != nullptr) ctx_.tts->BargeIn();
#else
        (void)was_muted;
#endif
    }
    clamp_settings(settings_);

    // Gains first, and unconditionally: they are atomic stores, they are what the
    // user is listening to while dragging, and they must not wait behind a device
    // reopen that may not even be needed.
    //
    // THE EFFECTIVE GAIN is the slider folded with the mute. It goes to the speaker
    // and nowhere else: the canceller observes the endpoint through a loopback, so
    // it sees this change by itself, at the same instant the room does, with no
    // second setter to keep in step.
#if defined(VOICE_ASSISTANT_HAS_TTS)
    if (ctx_.tts != nullptr) {
        ctx_.tts->SetVolume(audio_.effective_tts_volume(settings_.tts_volume));
    }
#endif
    audio_.set_input_gain(settings_.mic_gain);

    // The device reopen happens ONLY when an endpoint actually moved. A slider drag
    // arrives as dozens of messages; reopening WASAPI on each would turn a volume
    // change into an audible stutter.
    if (u.needs_device_reload()) {
        const auto t0 = std::chrono::steady_clock::now();
        audio_.apply_audio_reload(settings_);
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
        // The line that proves the claim: a device swap, timed, with no engine or
        // arena activity between the two timestamps.
        std::printf("[audio-hot-reload] Swapped WASAPI device in %lld ms (engine untouched, "
                    "VRAM unchanged)\n",
                    static_cast<long long>(ms));
        std::fflush(stdout);
    }

    // Persisted so the choice survives a restart -- but note the ordering: the
    // device is ALREADY live by here. A failed write costs the user the setting next
    // launch, not the sound they just fixed.
    if (!save_settings(settings_)) {
        std::fprintf(stderr, "[settings] WARN: could not persist audio settings\n");
    }
}

void WebUIBridge::create(int client_w, int client_h) {
    if (!window_.create(client_w, client_h)) {
        throw std::runtime_error("failed to create the assistant window");
    }
    view_.set_transport(router_.transport_name(), router_.transport_is_live());
}

void WebUIBridge::run_message_loop() { window_.run_message_loop(); }

void WebUIBridge::start_diagnostics_poller() {
    if (polling_.exchange(true, std::memory_order_acq_rel)) return;
    stats_thread_ = std::thread([this] {
        // An exception escaping a std::thread is std::terminate. A telemetry poller
        // is the last thing that should be allowed to take the app down, so it
        // swallows and keeps going.
        //
        // The gate counters move at conversation pace; the mic meter has to move at
        // speech pace or it reads as broken. So the loop runs at the METER's rate
        // and the counters are published every Nth pass.
        constexpr int kMeterMs = 80;              // ~12 fps, smooth enough
        constexpr int kStatsEvery = 1000 / kMeterMs;
        int tick = 0;
        while (polling_.load(std::memory_order_acquire)) {
            try {
                if (tick % kStatsEvery == 0) view_.publish_stats(router_.commit_queue());
                window_.post_audio_level(audio_.input_level());
            } catch (...) {
            }
            ++tick;
            std::this_thread::sleep_for(std::chrono::milliseconds(kMeterMs));
        }
    });
}

void WebUIBridge::stop_diagnostics_poller() noexcept {
    if (!polling_.exchange(false, std::memory_order_acq_rel)) return;
    if (stats_thread_.joinable()) stats_thread_.join();
}

}  // namespace rt
