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

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <utility>

namespace rt {

WebUIBridge::WebUIBridge(AssistantSettings& settings, AppContext& ctx,
                         AppLifecycleManager& lifecycle, AudioPipelineBinder& audio,
                         ConversationRouter& router, AssistantView& view, LogBuffer* log)
    : settings_(settings),
      ctx_(ctx),
      lifecycle_(lifecycle),
      audio_(audio),
      router_(router),
      view_(view),
      log_(log),
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

    // Residency transitions fire on the ENGINE thread, so the sink does the one
    // thing a callback from that thread may do: hand JSON to the window's queue.
    lifecycle_.residency().set_progress_sink(
        [this](const EngineResidency::Progress& p) { publish_residency(p); });

    install_subscribers();
    install_callbacks();
}

WebUIBridge::~WebUIBridge() { stop_diagnostics_poller(); }

// THE OBSERVER REGISTRY. Every consumer of a live setting is registered here,
// once, and the order is the order they are notified in -- which matters in
// exactly one place and is called out there.
void WebUIBridge::install_subscribers() {
    // FIRST, because the residency decides whether the local leg can serve an
    // intent at all, and the router below is about to be told to route to it.
    // Publishing them the other way round would open a window -- microseconds
    // wide, but real -- in which the router dispatches locally at weights that
    // are on their way out.
    bus_.subscribe("residency", [this](const AssistantSettings& s) {
        lifecycle_.residency().apply_local_inference(s.local_inference);
    });
    bus_.subscribe("router", [this](const AssistantSettings& s) {
        // Where the NEXT intent gets answered, and the badge that has to follow it.
        router_.set_use_local(s.local_inference);
    });
    bus_.subscribe("engine", [this](const AssistantSettings& s) { apply_engine_settings(s); });
    bus_.subscribe("audio", [this](const AssistantSettings& s) {
        // The AEC enable flag and the mute-folded speaker volume, plus the mic gain.
        audio_.apply_live_settings(s);
    });
    bus_.subscribe("console", [this](const AssistantSettings& s) { console_.apply_settings(s); });
    bus_.subscribe("window", [this](const AssistantSettings& s) {
        window_.apply_window_settings(s);
    });
}

void WebUIBridge::apply_live_settings(const AssistantSettings& s) { bus_.publish(s); }

ValidationContext WebUIBridge::validation_context() const {
    ValidationContext vc;
    // What actually came up, not what was asked for: real_control() is non-null
    // exactly when a checkpoint is resident, which is the condition under which
    // a VRAM budget means anything.
    vc.have_gpu = lifecycle_.real_control() != nullptr;
    vc.cascade = settings_.pipeline_mode == "whisper_cascade";
#if defined(VOICE_ASSISTANT_HAS_TTS)
    vc.tts_enabled = ctx_.tts != nullptr;
#endif
    vc.whisper_model = settings_.whisper_model_path;
    // The two-sequence topology doubles the per-token KV cost. Read off the
    // stack rather than assumed, so a launch that failed to isolate is budgeted
    // as the single sequence it actually got.
    vc.branch_factor = lifecycle_.real_stack().isolated_sessions ? 2 : 1;
    return vc;
}

void WebUIBridge::publish_diagnostics(const ValidationReport& report) {
    // Straight to the page, one entry per finding, keyed by field so the modal
    // can attach each message to the input that produced it rather than dumping
    // everything into one banner.
    nlohmann::json j;
    j["type"] = "settings.diagnostics";
    j["items"] = nlohmann::json::array();
    for (const Diagnostic& d : report.items) {
        j["items"].push_back({{"field", d.field},
                              {"severity", to_string(d.severity)},
                              {"message", d.message}});
    }
    window_.post_event(j.dump());
    // AND to the log, because the console is where a user looks when the modal
    // is closed -- and because errors belong on stderr regardless of whether a
    // page is up to receive them.
    report.print("settings");
}

void WebUIBridge::publish_residency(const EngineResidency::Progress& p) {
    nlohmann::json j;
    j["type"] = "residency";
    j["state"] = to_string(p.state);
    j["percent"] = p.percent;
    j["detail"] = p.detail;
    // Whether the checkbox can free VRAM at all on this launch, so the page can
    // explain the toggle rather than hide it.
    j["can_release"] = lifecycle_.residency().can_release_vram();
    window_.post_event(j.dump());
}

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
        // DestroyWindow, not WM_CLOSE. WM_CLOSE is intercepted now and hides to
        // the tray, so a restart posted through it would put the app in the
        // notification area and never relaunch. This is the real quit path, and
        // it still runs WM_DESTROY -- where the hotkeys, the WebView2 controller
        // and the tray icon are released.
        if (window_.hwnd() != nullptr) DestroyWindow(window_.hwnd());
    };

    cb.on_toggle_console = [this] { console_.toggle(); };
    // Nothing to do beyond letting the window run its real close path: main()'s
    // shutdown ordering happens after run_message_loop() returns and must not be
    // short-circuited from a menu handler.
    cb.on_exit = [] {};

    window_.set_callbacks(std::move(cb));
}

// Live-tier settings, applied to the RUNNING system. Everything here is an atomic
// store read at the next VAD block or turn boundary -- no marshaling, no engine
// work, doctrine intact (see settings_store.hpp) -- with the ONE documented
// exception of the audio task prefix, which is a KV rebuild and is marshaled.
//
// A BUS SUBSCRIBER like every other consumer; it is a named method only because
// it is the one with real policy in it. The audio and router pokes that used to
// sit at the two ends of this function are now their own subscribers, so what is
// left here is exactly the engine-facing set.
void WebUIBridge::apply_engine_settings(const AssistantSettings& s) {
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
    // VALIDATED BEFORE IT IS ADOPTED, so the value that reaches `settings_`, the
    // subsystems and the file on disk is the CORRECTED one -- a max_context the
    // card cannot afford is reduced here, once, rather than being persisted and
    // then quietly clamped again at the next bring-up.
    //
    // A copy, because validate_settings corrects in place and `next` is the
    // page's payload.
    AssistantSettings vetted = next;
    const ValidationReport report = validate_settings(vetted, validation_context());
    publish_diagnostics(report);
    // NOT REFUSED ON ERRORS, deliberately. A settings dialog that will not close
    // is a worse failure than a setting that was corrected: the report tells the
    // user exactly which field is wrong and what it will do instead, and every
    // Error here describes something inert (a dead hotkey, an unusable key)
    // rather than something dangerous.
    const bool audio_moved = requires_audio_reload(settings_, vetted);
    settings_ = vetted;
    // The modal is re-seeded from what was ACTUALLY stored, so a corrected field
    // shows its new value instead of the rejected one the user typed.
    window_.set_settings(settings_);
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
    // The console, on THIS thread -- which is the UI thread, so its messages are
    // dispatched by the same GetMessage loop the main window's are and it needs
    // no pump of its own.
    //
    // A failure here is a DEGRADATION, not an error: the app is fully usable
    // without a log overlay and taking startup down over one would be the wrong
    // trade. console_.create() prints the reason itself.
    if (console_.create(settings_, log_)) {
        std::printf("[console] press %s for the log overlay\n", settings_.hotkey_console.c_str());
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
