// -----------------------------------------------------------------------------
// voice_assistant — the Local Router voice loop as a standalone app.
//
//   listen -> pause -> local generate -> [barge-in aborts]
//                                     -> [EOS commits] -> remote model
//
// THIS FILE IS THE ORCHESTRATION AND NOTHING ELSE. It resolves the launch
// configuration, constructs five components in the one order that works, and
// tears them down in the reverse. Every behaviour lives in the component that owns
// it:
//
//   app_lifecycle.hpp        COM apartment, the VRAM guard, the engine stack, the
//                            neural VAD, the speech mode, the engine thread.
//   audio_pipeline_binder.h  microphone -> DSP -> the PCM tap -> the mode, plus
//                            speech output, the loopback reference and the AEC.
//   conversation_router.hpp  the commit gate, both legs, the dispatcher, the
//                            reply splitter, chat history and the session store.
//   web_ui_bridge.hpp        the Chromium window, every page-message binding, the
//                            settings fan-out and the diagnostics poller.
//   cli_options.hpp          this app's own flags, and the two self-tests that
//                            exist only to serve one.
//
// CONSTRUCTION ORDER IS THE CONTRACT, and it is the reason those five are plain
// locals in this function rather than members of something:
//
//   AppContext        first -- the speech and answer callbacks reach state
//                     through it, and both of the next two publish into it.
//   AssistantView     the JSON model. No widgets, thread-safe by construction.
//   AudioPipelineBinder  owns the WhisperDSP the ENGINE borrows, so it must
//                     outlive the engine -- which declaring it here gives.
//   AppLifecycleManager  the engine, and the thread that is allowed to touch it.
//   ConversationRouter   holds the control; destroyed before it.
//   WebUIBridge       last, so its callbacks can reach all four.
//
// THE SINGLE-THREADED ENGINE DOCTRINE (CLAUDE.md) holds throughout: exactly one
// thread calls into the control after load, and every other thread marshals onto
// it with post_engine_task.
//
// THREADS (five, unchanged by the decomposition)
//   UI          window message loop + WebView2 (STA). Touches the window only.
//   DSP worker  mic -> log-mel -> spectrogram, and the PCM tap into the mode.
//   ENGINE      the ONLY thread that calls into the control.
//   DISPATCHER  owned by IntentDispatcher. Blocks on the gate, then on the
//               transport. Never touches the engine.
//   STATS       a poller for the Diagnostics panel; touches nothing else.
//
// CONFIGURATION PRECEDENCE. persisted settings < CLI flags. A flag typed on this
// launch wins and is not written back (see settings_store.hpp).
// -----------------------------------------------------------------------------
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include "build_features.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "audio_devices.h"

#include "app_context.hpp"
#include "app_lifecycle.hpp"
#include "asset_paths.hpp"
#include "assistant_view.hpp"
#include "audio_pipeline_binder.hpp"
#include "cli_config.hpp"          // rt::TranslatorArgs, rt::parse_cli
#include "cli_options.hpp"         // rt::VoiceArgs, rt::parse_voice_args, the self-tests
#include "conversation_router.hpp"
#include "full_stack_test.hpp"
#include "settings_store.hpp"
#include "web_ui_bridge.hpp"

int main(int argc, char** argv) {
    // THE APARTMENT, claimed FIRST and before anything else can claim a different
    // one. RAII, so it is released on every exit path including a throw -- see
    // ComApartment for why STA specifically, and what happens when miniaudio wins
    // this race instead.
    std::unique_ptr<rt::ComApartment> com;
    try {
        com = std::make_unique<rt::ComApartment>();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 2;
    }

    // The console renders in the OEM codepage (866/850) unless told otherwise, so
    // every Cyrillic transcript this app logs -- the decode trace, the commit-gate
    // lines, the system-prefix readout -- comes out as mojibake or question marks
    // in a terminal while the SAME string renders perfectly in the WebView. That
    // asymmetry is worth naming: it makes the log look like the bug when the
    // pipeline is fine.
    SetConsoleOutputCP(CP_UTF8);

    // ---- configuration: persisted settings, then typed flags on top ----------
    rt::AssistantSettings settings = rt::load_settings();

    rt::TranslatorArgs args;
    std::vector<char*> shared_argv;
    const rt::VoiceArgs vargs = rt::parse_voice_args(argc, argv, shared_argv);

    // Answered BEFORE anything else is parsed, loaded or opened: this is the flag
    // someone reaches for precisely because the app will not start, or because it
    // started on the wrong speaker. Requiring a valid checkpoint to find out what
    // the audio endpoints are called would defeat it.
    if (vargs.list_audio_devices) {
        rt::print_audio_devices();
        return 0;
    }

    try {
        args = rt::parse_cli(static_cast<int>(shared_argv.size()), shared_argv.data());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 2;
    }

    const rt::TypedFlags& typed = vargs.typed;
    if (typed.model_dir)      { settings.model_dir = args.model_dir; settings.simulated = false; }
    if (typed.real)           { settings.model_dir = args.model_dir; settings.simulated = false; }
    if (typed.simulated)      settings.simulated = true;
    if (typed.audio_head)     settings.audio_head = args.audio_head;
    if (typed.projector_path) settings.projector_path = args.projector_path;
    if (typed.data_dir)       settings.data_dir = args.data_dir;
    if (typed.device)         settings.device_id = vargs.device_id;
    if (typed.vad_threshold)  settings.vad_threshold = args.vad_threshold;
    if (typed.no_neural_vad)  settings.neural_vad = false;
    if (typed.context_mode)   settings.context_mode = args.context_mode;
    if (typed.history_budget) settings.history_budget_tokens = args.history_budget_tokens;
    if (typed.output_device)  settings.output_device_name = vargs.output_device;
    if (typed.input_device)   settings.input_device_name = vargs.input_device;
    if (typed.output_device_index) settings.output_device_index = vargs.output_device_index;
    if (typed.input_device_index)  settings.input_device_index = vargs.input_device_index;
    if (typed.tts_volume)     settings.tts_volume = vargs.tts_volume;
    // Before clamp_settings, deliberately: an unrecognised --pipeline value has to
    // hit the SAME validator a hand-edited settings file does, and fall back the
    // same way (to legacy, never to the newer path).
    if (typed.pipeline_mode)     settings.pipeline_mode = vargs.pipeline_mode;
    if (typed.whisper_model)     settings.whisper_model_path = vargs.whisper_model;
    if (typed.whisper_language)  settings.whisper_language = vargs.whisper_language;
    rt::clamp_settings(settings);
    // Typed but rejected: clamp_settings silently reverts an unknown value, which
    // is right for a settings file and wrong for a flag somebody just typed --
    // launching on the legacy path after asking for the cascade must not be quiet.
    if (typed.pipeline_mode && settings.pipeline_mode != vargs.pipeline_mode) {
        std::fprintf(stderr,
                     "--pipeline '%s' is not a pipeline (expected ultravox_legacy or "
                     "whisper_cascade) -- using %s\n",
                     vargs.pipeline_mode.c_str(), settings.pipeline_mode.c_str());
    }

    // After the settings resolve (so it honours a configured output device) but
    // before anything heavy loads: this test needs an audio endpoint and nothing
    // else -- no checkpoint, no GPU, no window.
    if (vargs.check_volume) {
#if defined(VOICE_ASSISTANT_HAS_TTS)
        return rt::run_volume_check(settings);
#else
        std::fprintf(stderr, "--check-volume needs the speech-output stack, which this build "
                             "does not have (blackwell_tts_f5 absent).\n");
        return 2;
#endif
    }

    // Same placement and the same reasoning as --check-volume, one tier heavier:
    // this one loads the F5 graphs onto the GPU, but still no checkpoint, no engine
    // and no window.
    if (!vargs.say.empty()) {
#if defined(VOICE_ASSISTANT_HAS_TTS)
        return rt::run_say_check(settings, vargs.say);
#else
        std::fprintf(stderr, "--say needs the speech-output stack, which this build does not "
                             "have (blackwell_tts_f5 absent).\n");
        return 2;
#endif
    }

    // ---- WHICH SPEECH-TO-TEXT PIPELINE --------------------------------------
    // Resolved HERE, before anything is loaded, because it decides what gets
    // ALLOCATED: the Ultravox audio head, the backbone checkpoint, and whether a
    // ~1.6 GB GGML model is pulled onto the GPU.
    bool cascade = (settings.pipeline_mode == "whisper_cascade");
#if !defined(VOICE_ASSISTANT_HAS_WHISPER)
    if (cascade) {
        // Refused, LOUDLY, and clamped back to the path that works. A build with no
        // whisper.cpp cannot honour this setting, and coming up silently on the
        // legacy pipeline would leave the user believing they were testing the
        // cascade.
        std::fprintf(stderr,
                     "[cascade] pipeline_mode=whisper_cascade, but this build has no "
                     "whisper.cpp (-DUSE_WHISPER_CPP=OFF) -- running the legacy Ultravox "
                     "path instead.\n");
        cascade = false;
        settings.pipeline_mode = "ultravox_legacy";
    }
#endif

    // The GGML model, resolved the same three ways the Silero model is: the
    // configured path, then whatever CMake baked in, then a copy next to the exe.
    std::string whisper_model = settings.whisper_model_path;
#if defined(VOICE_ASSISTANT_HAS_WHISPER)
    if (whisper_model.empty()) {
#if defined(BLACKWELL_WHISPER_MODEL_PATH)
        whisper_model = BLACKWELL_WHISPER_MODEL_PATH;
#endif
    }
    whisper_model = rt::resolve_asset_file(whisper_model, "Whisper-Turbo-Platinum-F16.bin");
#endif

    // CASCADE'S BACKBONE. With no projector in the graph the backbone width is free
    // (see kDefaultCascadeModelDir), so an unconfigured cascade brings up Qwen
    // rather than falling through to the simulated stand-in -- but only if the
    // checkpoint is actually there. Guessing a path that does not exist would turn
    // "no model configured" into a startup crash, which is strictly worse than the
    // GPU-free backend it would have used.
    if (cascade && settings.model_dir.empty() && rt::path_exists(rt::kDefaultCascadeModelDir)) {
        settings.model_dir = rt::kDefaultCascadeModelDir;
        settings.simulated = false;
        std::printf("[cascade] no backbone configured -- defaulting to %s\n",
                    rt::kDefaultCascadeModelDir);
    }

    // THE decision. Everything downstream is written against the base type.
    const bool use_real = !settings.simulated && !settings.model_dir.empty();

    // Fold the resolved settings back into the args the engine bootstrap reads, so
    // there is exactly ONE description of the launch configuration from here on.
    args.model_dir = settings.model_dir;
    args.have_model_dir = !settings.model_dir.empty();
    if (!settings.audio_head.empty())     args.audio_head = settings.audio_head;
    if (!settings.projector_path.empty()) args.projector_path = settings.projector_path;
    // Assets: prefer the copies the build deployed next to the exe, so the app
    // launches identically from F5, Explorer, and a copied output folder.
    args.data_dir = rt::resolve_asset_dir(settings.data_dir, "mel_filters.bin");
    args.vad_model = rt::resolve_asset_file(args.vad_model, "silero_vad.onnx");
    args.vad_threshold = settings.vad_threshold;
    args.neural_vad = settings.neural_vad;
    args.context_mode = settings.context_mode;
    args.history_budget_tokens = settings.history_budget_tokens;
    // Reflect what will actually run, so the Settings modal never shows a path the
    // engine is not using.
    settings.audio_head = args.audio_head;
    settings.projector_path = args.projector_path;

    std::printf("=== voice_assistant (Local Router) ===\n");
    std::printf("  commit rule : dispatch IFF the local generation reaches EOS\n");
    // FIRST, because it is what every line below means something different under. A
    // reader diagnosing "why is there no [audio] line" needs this one before the
    // ones that would otherwise look like the failure.
    // Keyed off the SETTING, not off `cascade`, so a recognised-but-unsupported
    // pipeline (Mode B -- see bring_up_speech_mode) is named honestly for the two
    // lines before it is refused, rather than reported as the legacy path it is
    // about to not run.
    std::printf("  pipeline    : %s\n",
                cascade ? "whisper_cascade (whisper.cpp -> text -> backbone; NO audio head)"
                : settings.pipeline_mode == "simultaneous"
                    ? "simultaneous (Mode B -- NOT BUILT INTO THIS BINARY; will refuse)"
                    : "ultravox_legacy (audio -> soft-tokens -> backbone)");
    if (cascade) {
        std::printf("  whisper     : %s\n",
                    whisper_model.empty() ? "(NOT CONFIGURED -- cascade will refuse to arm)"
                                          : whisper_model.c_str());
    }
    if (use_real) {
        std::printf("  engine      : RealEngineControl (CUDA device %d)\n", settings.device_id);
        std::printf("  model-dir   : %s\n", args.model_dir.c_str());
        if (!cascade) std::printf("  audio-head  : %s\n", args.audio_head.c_str());
    } else {
        std::printf("  engine      : SimulatedEngineControl (no GPU, no checkpoint)%s\n",
                    settings.simulated ? "  [Simulated selected]"
                                       : "  [no model directory configured]");
    }
    // Reports the selector that will actually be CONSULTED, applying the same
    // index-beats-name precedence resolve_device_selection does. Printing both when
    // both are set would be honest about the config and misleading about the
    // behaviour -- and this line exists to predict the behaviour.
    const auto endpoint_desc = [](const std::string& name, int index) {
        if (index >= 0) return "index " + std::to_string(index);
        return name.empty() ? std::string("(system default)") : name;
    };
    std::printf("  audio in    : %s\n",
                endpoint_desc(settings.input_device_name, settings.input_device_index).c_str());
    std::printf("  audio out   : %s  @ %.0f%% volume\n",
                endpoint_desc(settings.output_device_name, settings.output_device_index).c_str(),
                static_cast<double>(settings.tts_volume) * 100.0);
    std::printf("  settings    : %s\n", rt::settings_path().c_str());

    // The endpoint list, at launch, every launch. It is a few lines on a typical
    // box and it is the difference between "the assistant is silent" being a
    // ten-second fix and an evening: the [N] printed here is EXACTLY what
    // output_device_index / input_device_index select with, and the names are
    // EXACTLY the strings output_device_name / input_device_name accept.
    //
    // AFTER the two lines above, deliberately. Those say which endpoint is
    // configured; this says what the endpoints are. Read in that order, a mismatch
    // is visible on one screen without scrolling back.
    rt::print_audio_devices();

    try {
        // ---- the shared handle, and the JSON model ---------------------------
        rt::AppContext app_ctx;
        rt::AssistantView view;
        app_ctx.view = &view;

        // ---- 1. the audio front-end -----------------------------------------
        // Declared FIRST of the components because it owns the WhisperDSP the
        // engine borrows for the live commit path -- so it must outlive the engine,
        // which is exactly what this position gives.
        //
        // On the LEGACY real path the mel geometry MUST match the projector the
        // audio head was trained with; 128 (Whisper large-v3-turbo) is the
        // simulated default. Resolving it here also fails fast on a bad projector
        // path, before the mic is opened. Cascade mode takes the default and does
        // not consult the projector at all.
        const int n_mels = (use_real && !cascade)
                               ? rt::resolve_projector_params(args.projector_path).num_mel_bins
                               : 128;
        rt::AudioPipelineBinder audio(settings, n_mels, args.data_dir, app_ctx);

        // ---- 2. the engine, and the thread allowed to touch it ---------------
        rt::AppLifecycleManager::Config lcfg;
        lcfg.settings = settings;
        lcfg.args = args;
        lcfg.cascade = cascade;
        lcfg.whisper_model = whisper_model;
        lcfg.use_real = use_real;
        lcfg.sample_rate = audio.sample_rate();
        rt::AppLifecycleManager lifecycle(lcfg, audio.dsp(), app_ctx);

        // ---- 3. the Local Router ---------------------------------------------
        // The local generator lives on the CONCRETE control (that is where the
        // decode loop is), so it arrives as a callable -- the same seam
        // prefill_system_prompt uses -- and LocalEngineTransport marshals it onto
        // the engine thread.
        rt::LocalEngineTransport::GenerateFn generate_locally;
        rt::ConversationRouter::RebuildSystemPromptFn rebuild_prompt;
        if (rt::RealEngineControl* rc = lifecycle.real_control(); rc != nullptr) {
            generate_locally = [rc](const std::string& intent,
                                    const std::function<void(std::string_view)>& emit,
                                    blackwell::bridge::TerminationReason* reason) {
                return rc->generate_local_reply(intent, emit, reason);
            };
            rebuild_prompt = [rc](const std::string& p) { return rc->rebuild_system_prompt(p); };
        } else {
            auto* sc = lifecycle.simulated_control();
            generate_locally = [sc](const std::string& intent,
                                    const std::function<void(std::string_view)>& emit,
                                    blackwell::bridge::TerminationReason* reason) {
                return sc->generate_local_reply(intent, emit, reason);
            };
            rebuild_prompt = [sc](const std::string& p) { return sc->rebuild_system_prompt(p); };
        }

        rt::ConversationRouter::Config rcfg;
        rcfg.persona = settings.system_prompt;
        rcfg.remote_api_key = settings.remote_api_key;
        rcfg.remote_api_url = settings.remote_api_url;
        rcfg.remote_model = settings.remote_model;
        rcfg.max_new_tokens = settings.max_new_tokens;
        rcfg.local_inference = settings.local_inference;
        rt::ConversationRouter router(rcfg, app_ctx, view, lifecycle.control(),
                                      std::move(generate_locally), std::move(rebuild_prompt));
        router.load_persisted_history(settings.local_inference);
        lifecycle.control()->set_commit_queue(&router.commit_queue());
        router.start();

        // ---- 4. start the engine, then the speech output ---------------------
        // The engine thread first, because start_speech_output is built AFTER it so
        // a TTS failure cannot delay the thing the app is actually for -- and
        // BEFORE the UI, so the first reply can be spoken.
        lifecycle.start_engine_thread();
        audio.start_speech_output(settings, settings.device_id);
        // The last GPU consumer to load, so this line is the WHOLE budget: if it
        // says OVER BUDGET, the box is already paging and every latency number
        // after it is meaningless. The reducible items are max_context, the fp32
        // audio head, and F5's max_frames.
        if (use_real) rt::report_vram("+ TTS (full stack)");

        // Install the PCM tap and start the DSP worker. Last of the audio wiring,
        // because the tap dereferences the mode on every block.
        audio.bind_speech_mode(lifecycle.mode());

        // ---- 5. the UI, and every page-message binding -----------------------
        rt::WebUIBridge ui(settings, app_ctx, lifecycle, audio, router, view);
        ui.apply_live_settings(settings);
        ui.create(/*client_w=*/860, /*client_h=*/720);
        view.set_backend(use_real ? "Local model" : "Simulated",
                         use_real ? args.model_dir : std::string("no checkpoint loaded"),
                         use_real && lifecycle.real_stack().audio_head_ready);
        router.seed_startup_view();
        ui.start_diagnostics_poller();

        // ---- the optional full-stack diagnostic ------------------------------
        std::thread test_thread;
        if (!vargs.test_llm_tts.empty()) {
            test_thread = std::thread([&] {
                rt::run_full_stack_test(vargs.test_llm_tts, settings, use_real, app_ctx, audio,
                                        router, ui.window());
            });
        }

        std::printf("running... speak, then pause. Interrupt mid-answer to test barge-in.\n");
        std::fflush(stdout);
        ui.run_message_loop();
        if (test_thread.joinable()) test_thread.join();

        // ---- shutdown (reverse dependency order) -----------------------------
        // Each step names what it must precede. This is the ONE ordering in the app
        // that a destructor cannot express, because the objects are independent and
        // the constraint is between their THREADS.
        audio.stop();          // the PCM tap calls into the mode the engine pumps
        lifecycle.shutdown();  // joins the engine thread
        ui.stop_diagnostics_poller();
        router.stop();         // after the engine, so no commit can arrive post-join

        router.print_shutdown_summary();
        lifecycle.print_shutdown_summary();
        audio.print_shutdown_summary();

        // Relaunched LAST, after this process has released the CUDA context, the
        // capture device and the WebView2 user-data folder -- starting the new
        // instance any earlier would have two processes fighting over all three.
        const bool restart = ui.restart_requested();
        if (restart && !rt::relaunch_self()) {
            std::fprintf(stderr, "[settings] WARN: could not relaunch -- start the app again "
                                 "to apply the new settings.\n");
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        MessageBoxA(nullptr, e.what(), "Voice Assistant", MB_ICONERROR | MB_OK);
        return 1;
    }
}
