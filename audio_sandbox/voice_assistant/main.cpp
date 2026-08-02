// -----------------------------------------------------------------------------
// voice_assistant — the Local Router voice loop as a standalone app.
//
//   listen -> pause -> local generate -> [barge-in aborts]
//                                     -> [EOS commits] -> remote model
//
// THE UI IS A CHROMIUM MESSENGER (AssistantWindow + web/). The main window hosts
// a WebView2 whose whole client area is the chat: bubbles, a text box, a mic
// button, a status pill. It carries NO inference telemetry -- no TTFT, no KV
// watermarks, no gate counters. Every technical knob lives behind the Settings
// modal, and the counters that used to occupy the main screen are now a
// collapsed Diagnostics section inside it. The pipeline below is untouched by
// that change: the UI is a JSON sink on one side and a text/settings source on
// the other.
//
// TWO ENGINE BACKENDS, ONE PIPELINE
//   model directory set   RealEngineControl: real CUDA decode on a real
//                         checkpoint, audio head armed, genuine EOS from the
//                         tokenizer's stop set.
//   no model directory    SimulatedEngineControl: no GPU, no checkpoint, canned
//   / Simulated checked   replies. Everything ABOVE the control is identical --
//                         the same SPSC command ring, the same barge-in epoch,
//                         the same commit gate, the same dispatcher.
//
//   Both derive from EngineControlBridge, so everything below holds ONE base
//   pointer and never branches on which backend is live. That is the property
//   that makes the offline path a real rehearsal of the GPU path rather than a
//   parallel implementation of it.
//
// TEXT AND VOICE ARE THE SAME PATH. A typed message goes through
// EngineControlBridge::submit_text -> the SAME command ring -> the SAME decode
// loop -> the SAME IntentCommitQueue under the SAME EOS rule. There is no second
// route to a billed cloud call.
//
// CONFIGURATION PRECEDENCE. persisted settings < CLI flags. A flag typed on this
// launch wins and is not written back (see settings_store.hpp).
//
// THREADS (five, and the single-engine-thread doctrine holds trivially)
//   UI          window message loop + WebView2 (STA). Touches the window only.
//   DSP worker  mic -> log-mel -> spectrogram, and the PCM tap into the mode.
//   ENGINE      the ONLY thread that calls into the control. Runs the decode
//               loop and offers finished turns to the commit gate. Selects the
//               CUDA device for ITSELF -- the current device is per-thread.
//   DISPATCHER  owned by IntentDispatcher. Blocks on the gate, then blocks on
//               the transport. Never touches the engine.
//   STATS       a 1 Hz poller for the Diagnostics panel. Reads gate counters and
//               posts them to the UI queue; touches nothing else.
// -----------------------------------------------------------------------------
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <objbase.h>   // CoInitializeEx / COINIT_APARTMENTTHREADED (WIN32_LEAN_AND_MEAN
                       // drops <ole2.h> from <windows.h>, so pull it in explicitly)

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "audio_capture.h"
#include "audio_recorder.h"   // realtime_dsp.h only forward-declares AudioRecorder
#include "realtime_dsp.h"
#include "whisper_dsp.h"

#include "assistant_view.hpp"           // rt::AssistantView (JSON model, no widgets)
#include "assistant_window.hpp"         // rt::AssistantWindow (Chromium host)
#include "settings_store.hpp"           // rt::AssistantSettings, load/save
#include "cli_config.hpp"               // rt::TranslatorArgs, rt::parse_cli
#include "conversational_mode.hpp"      // rt::ConversationalMode (Mode A, reused as-is)
#include "engine_bootstrap.hpp"         // rt::RealEngineStack, bring_up_real_engine
#include "simulated_engine_control.hpp" // rt::SimulatedEngineControl

#include "intent_commit.hpp"          // blackwell::bridge::IntentCommitQueue
#include "intent_dispatcher.hpp"      // blackwell::cloud::IntentDispatcher
#include "offline_transport.hpp"      // blackwell::cloud::OfflineTransport

#if defined(BLACKWELL_HAVE_CLOUD_CLIENT)
#include "claude_stream_client.hpp"
#include "claude_transport.hpp"
#endif

#if defined(BLACKWELL_VAD_MODEL_PATH)
#include "silero_vad.hpp"
#define VOICE_ASSISTANT_HAS_SILERO 1
#endif

namespace {

// ---- which flags the user actually TYPED on this launch ----------------------
// Load-bearing for the precedence rule: parse_cli RESOLVES a value for several
// fields (a config.json in the CWD, its built-in default paths) whether or not
// anything was typed, so "args.model_dir is non-empty" says nothing about user
// intent. Only a flag that appeared in argv may override the persisted setting.
struct TypedFlags {
    bool model_dir = false;
    bool audio_head = false;
    bool projector_path = false;
    bool data_dir = false;
    bool device = false;
    bool simulated = false;
    bool real = false;
    bool vad_threshold = false;
    bool no_neural_vad = false;
    bool context_mode = false;
    bool history_budget = false;
};

// voice_assistant's OWN flags, layered over the shared rt::parse_cli. rt::parse_cli
// THROWS on an unknown argument (a good property -- a typo'd flag must not be
// silently ignored), so the flags that exist only here are removed before it runs.
struct VoiceArgs {
    int device_id = 0;   // --device <id>
    TypedFlags typed;
};

VoiceArgs parse_voice_args(int argc, char** argv, std::vector<char*>& passthrough) {
    VoiceArgs v;
    passthrough.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        char* a = argv[i];
        const bool is_flag = a[0] == '-';
        if (std::strcmp(a, "--device") == 0 && i + 1 < argc) {
            v.device_id = std::atoi(argv[++i]);
            v.typed.device = true;
        } else if (std::strcmp(a, "--simulated") == 0 || std::strcmp(a, "--mock") == 0) {
            // --mock kept as a deprecated alias so existing scripts keep working;
            // the codebase's own vocabulary is "simulated".
            v.typed.simulated = true;
        } else if (std::strcmp(a, "--real") == 0) {
            // Opt into parse_cli's resolved default checkpoint without naming a
            // path. Ours alone -- it would be rejected downstream.
            v.typed.real = true;
        } else {
            // Record what the shared parser is about to consume, then hand it on.
            if (std::strcmp(a, "--model-dir") == 0)      v.typed.model_dir = true;
            else if (std::strcmp(a, "--audio-head") == 0 ||
                     std::strcmp(a, "--audio-tower-path") == 0) v.typed.audio_head = true;
            else if (std::strcmp(a, "--projector-path") == 0)  v.typed.projector_path = true;
            else if (std::strcmp(a, "--vad-threshold") == 0)   v.typed.vad_threshold = true;
            else if (std::strcmp(a, "--no-neural-vad") == 0)   v.typed.no_neural_vad = true;
            else if (std::strcmp(a, "--context-mode") == 0)    v.typed.context_mode = true;
            else if (std::strcmp(a, "--history-budget") == 0)  v.typed.history_budget = true;
            else if (!is_flag && i > 0)                        v.typed.data_dir = true;
            passthrough.push_back(a);
        }
    }
    return v;
}

// ---- speech-pipeline callbacks (engine / VAD threads) -----------------------
// They may only hand results to the thread-safe view -- never a window, COM,
// CUDA, or a re-entrant pipeline call.
struct AppContext {
    rt::AssistantView* view = nullptr;
    // The BASE type: this is what makes the callbacks backend-agnostic.
    blackwell::bridge::EngineControlBridge* control = nullptr;
};

void on_token(void* user, const SpeechTokenEvent* event, std::uint64_t gen_id) {
    auto* ctx = static_cast<AppContext*>(user);
    if (ctx == nullptr || ctx->view == nullptr || event == nullptr) return;
    ctx->view->on_local_token(event->text, gen_id);
}

void on_state(void* user, SpeechPipelineState /*prev*/, SpeechPipelineState next,
              std::uint64_t /*gen_id*/) {
    auto* ctx = static_cast<AppContext*>(user);
    if (ctx == nullptr || ctx->view == nullptr) return;
    ctx->view->on_pipeline_state(next);
    // The return to IDLE is the turn boundary (SpeechTokenEvent carries no final
    // flag). The gate has already run by then, so the control's published verdict
    // is the authoritative reason this generation ended -- and it is what the UI
    // must show, because BargeIn and Eos both land on IDLE.
    if (next == SPEECH_STATE_IDLE) ctx->view->on_local_final(ctx->control->last_reason());
}

// TYPED-TURN sink. A typed message is already on screen as the user's bubble, so
// the local model's per-token output is NOT painted -- what the UI needs from
// this path is the single is_final edge carrying the gate's verdict, which is
// what decides whether the message was actually sent anywhere.
void on_text_sink(void* user, const char* /*utf8*/, std::int32_t /*index*/,
                  std::int32_t is_final, BridgeStatus /*status*/) {
    auto* ctx = static_cast<AppContext*>(user);
    if (ctx == nullptr || ctx->view == nullptr || is_final == 0) return;
    ctx->view->on_local_final(ctx->control->last_reason());
}

#if defined(VOICE_ASSISTANT_HAS_SILERO)
// Runs on the DSP worker, once per 10 ms block. That thread is the SINGLE owner
// of the SileroVAD instance (the class's threading contract); the UI thread only
// reads its published atomics.
float vad_score(void* user, const float* block, size_t count) {
    return static_cast<blackwell::vad::SileroVAD*>(user)->feed(block, count);
}
#endif

// ---- asset resolution: the exe's own directory, not the caller's CWD ---------
// A relative path like "data" is resolved against whoever launched us, so the
// app used to start under F5 (where CMake anchors the debugger's working
// directory) and die from Explorer with `cannot open: data/mel_filters.bin`.
// The build deploys every asset next to the binary, so prefer THAT copy and fall
// back to the CWD-relative one -- which keeps an explicitly passed --data-dir,
// and a developer running out of the source tree, working exactly as before.

std::string exe_dir() {
    wchar_t buf[MAX_PATH] = {};
    if (GetModuleFileNameW(nullptr, buf, MAX_PATH) == 0) return {};
    std::wstring w(buf);
    if (const size_t slash = w.find_last_of(L"\\/"); slash != std::wstring::npos) {
        w.resize(slash + 1);
    }
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), static_cast<int>(w.size()),
                                      nullptr, 0, nullptr, nullptr);
    std::string out(static_cast<size_t>(n), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), static_cast<int>(w.size()), out.data(), n,
                        nullptr, nullptr);
    return out;
}

bool path_exists(const std::string& p) {
    return GetFileAttributesA(p.c_str()) != INVALID_FILE_ATTRIBUTES;
}

// `dir` is usable if it holds `probe`; otherwise try <exe dir>/<dir>. Returns
// `dir` unchanged when neither works, so the caller still reports the original
// path in its error rather than a rewritten one the user never typed.
std::string resolve_asset_dir(const std::string& dir, const char* probe) {
    if (path_exists(dir + "/" + probe)) return dir;
    const std::string beside = exe_dir() + dir;
    if (path_exists(beside + "/" + probe)) return beside;
    return dir;
}

// Same idea for a single file (the VAD model, whose default is a configure-time
// absolute path that does not survive being copied to another machine).
std::string resolve_asset_file(const std::string& path, const char* fallback_name) {
    if (!path.empty() && path_exists(path)) return path;
    const std::string beside = exe_dir() + fallback_name;
    if (path_exists(beside)) return beside;
    return path;
}

// Relaunch this executable with NO arguments and let the current process exit.
//
// Restart-tier settings choose what gets ALLOCATED at bring-up, so applying them
// means a new engine -- and the honest way to get one is a new process, not a
// hot-swap of a 5.3 GB weight set under a live decode loop. Deliberately argv-
// free: every setting is persisted by now, and replaying the old CLI flags would
// re-override the very values the user just saved.
bool relaunch_self() {
    wchar_t exe[MAX_PATH] = {};
    if (GetModuleFileNameW(nullptr, exe, MAX_PATH) == 0) return false;
    STARTUPINFOW si{sizeof(si)};
    PROCESS_INFORMATION pi{};
    std::wstring cmd = L"\"" + std::wstring(exe) + L"\"";
    if (!CreateProcessW(exe, cmd.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr,
                        &si, &pi)) {
        return false;
    }
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    // ---- COM apartment: STA, claimed FIRST, before anything else ------------
    // THE FIRST CALLER WINS, PERMANENTLY. A thread's apartment cannot be changed
    // once set: a later CoInitializeEx with a different model returns
    // RPC_E_CHANGED_MODE and leaves the thread where it was.
    //
    // WebView2 is STA-only. miniaudio, meanwhile, initializes COM on whatever
    // thread first opens a device and defaults to COINIT_MULTITHREADED -- and
    // capture.start() runs on THIS thread, below, well before the window is
    // created. That ordering put the main thread in the MTA and made
    // CreateCoreWebView2EnvironmentWithOptions fail with an error the UI then
    // reported as a missing runtime, sending anyone who hit it off to reinstall
    // a runtime that was already installed.
    //
    // So the apartment is claimed here, at the top, where it is a stated decision
    // rather than a side effect of whichever subsystem happened to boot first.
    // miniaudio's own CoInitializeEx then returns RPC_E_CHANGED_MODE, which it
    // tolerates -- WASAPI works fine from an STA host, and its device thread has
    // its own apartment regardless.
    const HRESULT com_hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    if (FAILED(com_hr)) {
        std::fprintf(stderr, "FATAL: CoInitializeEx(STA) failed: 0x%08lX\n",
                     static_cast<unsigned long>(com_hr));
        return 2;
    }

    // ---- configuration: persisted settings, then typed flags on top ---------
    rt::AssistantSettings settings = rt::load_settings();

    rt::TranslatorArgs args;
    std::vector<char*> shared_argv;
    const VoiceArgs vargs = parse_voice_args(argc, argv, shared_argv);
    try {
        args = rt::parse_cli(static_cast<int>(shared_argv.size()), shared_argv.data());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 2;
    }

    const TypedFlags& typed = vargs.typed;
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
    rt::clamp_settings(settings);

    // THE decision. Everything downstream is written against the base type.
    const bool use_real = !settings.simulated && !settings.model_dir.empty();

    // Fold the resolved settings back into the args the engine bootstrap reads,
    // so there is exactly ONE description of the launch configuration from here on.
    args.model_dir = settings.model_dir;
    args.have_model_dir = !settings.model_dir.empty();
    if (!settings.audio_head.empty())     args.audio_head = settings.audio_head;
    if (!settings.projector_path.empty()) args.projector_path = settings.projector_path;
    // Assets: prefer the copies the build deployed next to the exe, so the app
    // launches identically from F5, Explorer, and a copied output folder.
    args.data_dir = resolve_asset_dir(settings.data_dir, "mel_filters.bin");
    args.vad_model = resolve_asset_file(args.vad_model, "silero_vad.onnx");
    args.vad_threshold = settings.vad_threshold;
    args.neural_vad = settings.neural_vad;
    args.context_mode = settings.context_mode;
    args.history_budget_tokens = settings.history_budget_tokens;
    // Reflect what will actually run, so the Settings modal never shows a path
    // the engine is not using.
    settings.audio_head = args.audio_head;
    settings.projector_path = args.projector_path;

    std::printf("=== voice_assistant (Local Router) ===\n");
    std::printf("  commit rule : dispatch IFF the local generation reaches EOS\n");
    if (use_real) {
        std::printf("  engine      : RealEngineControl (CUDA device %d)\n", settings.device_id);
        std::printf("  model-dir   : %s\n", args.model_dir.c_str());
        std::printf("  audio-head  : %s\n", args.audio_head.c_str());
    } else {
        std::printf("  engine      : SimulatedEngineControl (no GPU, no checkpoint)%s\n",
                    settings.simulated ? "  [Simulated selected]"
                                       : "  [no model directory configured]");
    }
    std::printf("  settings    : %s\n", rt::settings_path().c_str());

    try {
        // ---- audio front-end ------------------------------------------------
        whisper::DspConfig cfg;
        // On the real path the mel geometry MUST match the projector the audio
        // head was trained with; 128 (Whisper large-v3-turbo) is the simulated
        // default. Resolving it here also fails fast on a bad projector path,
        // before the mic is opened.
        cfg.n_mels = use_real ? rt::resolve_projector_params(args.projector_path).num_mel_bins
                              : 128;
        whisper::WhisperDSP dsp(cfg, args.data_dir + "/mel_filters.bin");
        // The spectrogram no longer has a viewer -- it is kept because
        // RealTimeDSP writes into it unconditionally, and shrinking it is a DSP
        // change, not a UI one.
        rt::SpectrogramBuffer spectrogram(cfg.n_mels, /*max_frames=*/1000);
        rt::AudioRecorder recorder(cfg.sample_rate, /*out_dir=*/"recordings");

        rt::AudioCapture capture;
        capture.start(settings.loopback_capture ? rt::CaptureMode::Loopback
                                                : rt::CaptureMode::Microphone);
        std::printf("capture started (%s, 16 kHz mono f32)\n", capture.backend_name().c_str());
        rt::RealTimeDSP realtime(dsp, capture.ring(), spectrogram, &recorder);

        // ---- the control plane ----------------------------------------------
        // Exactly one backend is constructed; `control` is the only handle
        // anything below uses.
        rt::RealEngineStack real_stack;
        std::unique_ptr<rt::SimulatedEngineControl> simulated;
        blackwell::bridge::EngineControlBridge* control = nullptr;

        if (use_real) {
            real_stack = rt::bring_up_real_engine(args, dsp, settings.max_context,
                                                  settings.device_id,
                                                  /*arm_streaming_plan=*/true);
            control = real_stack.bridge();
        } else {
            simulated = std::make_unique<rt::SimulatedEngineControl>();
            control = simulated.get();
        }

        rt::AssistantView view;

        // ---- the commit gate -------------------------------------------------
        // Capacity 4: an intent is one finished utterance. A deep backlog of
        // stale intents is worse than refusing new ones -- see intent_commit.hpp.
        // IDENTICAL on both backends: the gate does not know or care which decode
        // loop produced the verdict it is given.
        blackwell::bridge::IntentCommitQueue commit_queue(4);
        control->set_commit_queue(&commit_queue);

        // ---- the remote transport -------------------------------------------
        blackwell::cloud::OfflineTransport offline_transport;
        blackwell::cloud::IIntentTransport* transport = &offline_transport;

#if defined(BLACKWELL_HAVE_CLOUD_CLIENT)
        std::unique_ptr<blackwell::cloud::ClaudeStreamClient> live_client;
        std::unique_ptr<blackwell::cloud::ClaudeTransport> live_transport;
        if (const char* key = std::getenv("ANTHROPIC_API_KEY"); key != nullptr && *key != '\0') {
            blackwell::cloud::ClaudeStreamClient::Config ccfg;
            ccfg.api_key = key;
            ccfg.beta = "server-side-fallback-2026-07-01";
            live_client = std::make_unique<blackwell::cloud::ClaudeStreamClient>(std::move(ccfg));
            live_transport = std::make_unique<blackwell::cloud::ClaudeTransport>(*live_client);
            transport = live_transport.get();
            std::printf("[cloud] ANTHROPIC_API_KEY present -- LIVE transport armed\n");
        } else {
            std::printf("[cloud] no ANTHROPIC_API_KEY -- Mode: Offline (Simulated)\n");
        }
#else
        std::printf("[cloud] built without BUILD_CLOUD_CLIENT -- Mode: Offline (Simulated)\n");
#endif

        // ---- the dispatcher: gate -> remote ---------------------------------
        const std::string system_prompt = settings.system_prompt;
        blackwell::cloud::IntentDispatcher dispatcher(
            commit_queue, *transport,
            [&system_prompt] {
                blackwell::cloud::RequestContext c;
                // FROZEN. Nothing volatile may appear here: a timestamp or a
                // session id in this string drives the prompt-cache hit rate to
                // zero, silently and expensively. See intent_request.hpp.
                c.instructions = system_prompt;
                c.glossary = "This is a spoken-language voice assistant session.";
                c.committed_prefix = "";
                return c;
            });
        dispatcher.set_on_dispatch_start([&view](const blackwell::bridge::IntentRecord& r) {
            view.on_dispatch_start(r.sequence);
        });
        dispatcher.set_on_text([&view](std::string_view s) { view.on_remote_token(s); });
        dispatcher.set_on_complete(
            [&view](const blackwell::bridge::IntentRecord&, const blackwell::cloud::Result& r) {
                view.on_remote_final(r.status == blackwell::cloud::Status::Ok,
                                     r.error_detail.empty()
                                         ? std::string(blackwell::cloud::to_string(r.status))
                                         : r.error_detail);
            });
        dispatcher.start();

        // ---- optional Silero neural VAD --------------------------------------
        // Constructed BEFORE the mode and outliving it: the ORT session is a
        // startup-cost resource. Non-fatal, like the audio head.
        SpeechVadScoreFn vad_fn = nullptr;
        void* vad_user = nullptr;
#if defined(VOICE_ASSISTANT_HAS_SILERO)
        std::unique_ptr<blackwell::vad::SileroVAD> neural_vad;
        if (args.neural_vad && !args.vad_model.empty()) {
            try {
                neural_vad = std::make_unique<blackwell::vad::SileroVAD>(args.vad_model);
                neural_vad->set_threshold(args.vad_threshold);
                vad_fn = &vad_score;
                vad_user = neural_vad.get();
                std::printf("[vad] Silero neural VAD armed (threshold %.2f)\n",
                            args.vad_threshold);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[vad] WARN: %s -- falling back to the RMS detector\n",
                             e.what());
            }
        }
#endif
        if (vad_fn == nullptr) std::printf("[vad] built-in RMS threshold detector\n");

        // ---- the speech mode (Mode A, reused unchanged) ---------------------
        AppContext app_ctx{&view, control};
        rt::ConversationalMode::Config mcfg;
        mcfg.sample_rate = static_cast<std::uint32_t>(cfg.sample_rate);
        mcfg.vad_probability_threshold = settings.vad_threshold;
        mcfg.system_prompt = system_prompt;

        // prefill_system_prompt lives on the CONCRETE control, not the bridge
        // (how a frozen prefix is laid down genuinely differs per backend), so it
        // arrives as a callable rather than a virtual -- the same seam
        // audio_translator uses.
        rt::ConversationalMode::PrefillSystemPromptFn prefill;
        if (use_real) {
            auto* rc = real_stack.control.get();
            prefill = [rc](const std::string& p) { return rc->prefill_system_prompt(p); };
        } else {
            auto* sc = simulated.get();
            prefill = [sc](const std::string& p) { return sc->prefill_system_prompt(p); };
        }

        rt::ConversationalMode speech_mode(control, prefill, vad_fn, vad_user, mcfg, &on_token,
                                           &on_state, &app_ctx);
        rt::ISpeechMode* active = &speech_mode;

        // Live-tier settings, applied to the RUNNING pipeline. Everything here is
        // an atomic store read at the next VAD block or turn boundary -- no
        // marshaling, no engine work, doctrine intact (see settings_store.hpp).
        auto apply_live_settings = [&](const rt::AssistantSettings& s) {
            (void)speech_pipeline_set_vad_threshold(speech_mode.pipeline(), s.vad_threshold);
            if (real_stack.control) {
                real_stack.control->set_context_mode(
                    s.context_mode == "bounded"
                        ? rt::RealEngineControl::ContextMode::BoundedHistory
                        : rt::RealEngineControl::ContextMode::Stateless);
                real_stack.control->set_history_budget_tokens(s.history_budget_tokens);
                real_stack.control->set_live_center_slice(s.live_streaming);
            }
        };

        // ---- engine thread ---------------------------------------------------
        std::atomic<bool> running{true};
        std::atomic<bool> prefilled{false};
        std::atomic<bool> engine_failed{false};
        std::string engine_error;
        const int  device_id = settings.device_id;
        const bool real = use_real;
        std::thread engine_thread([&] {
            try {
                // CUDA's current device is PER-THREAD. This thread launches every
                // kernel, so it must select the same device the weights were
                // allocated on -- doing it only in main() would silently run
                // kernels on device 0 against device-N memory.
                if (real) rt::select_cuda_device(device_id);
                active->start_on_engine_thread();   // system-prompt prefill (INIT tier: throws)
                prefilled.store(true, std::memory_order_release);
                while (running.load(std::memory_order_acquire)) {
                    active->pump_engine();  // 0% CPU until a boundary event arrives
                }
            } catch (const std::exception& e) {
                // A throw here used to hang main() forever on the `prefilled`
                // spin. Publish the failure, then release the latch.
                engine_error = e.what();
                engine_failed.store(true, std::memory_order_release);
                prefilled.store(true, std::memory_order_release);
            }
        });
        while (!prefilled.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (engine_failed.load(std::memory_order_acquire)) {
            running.store(false, std::memory_order_release);
            active->stop();
            engine_thread.join();
            dispatcher.stop();
            throw std::runtime_error("engine thread failed during startup: " + engine_error);
        }
        std::printf("[system-prefix] frozen %u tokens (KV rewind floor)\n",
                    speech_mode.frozen_prefix_tokens());
        apply_live_settings(settings);

        realtime.set_pcm_tap(
            [&active](const float* s, std::size_t n) { active->on_pcm_block(s, n); });
        realtime.start();

        // ---- UI: the Chromium messenger --------------------------------------
        rt::AssistantWindow window(L"Assistant");
        // The view's ONLY output is JSON; the window's post_event is the only
        // thing that ever crosses back to the UI thread.
        view.set_sink([&window](std::string json) { window.post_event(std::move(json)); });
        window.set_settings(settings);

        const blackwell::bridge::TokenSink text_sink{&on_text_sink, &app_ctx};
        std::atomic<bool> restart_requested{false};

        rt::AssistantWindowCallbacks cb;
        cb.on_send_text = [&](const std::string& text) {
            // Same ring, same decode loop, same gate as a spoken turn.
            view.on_user_text(text);
            if (control->submit_text(text, text_sink) != blackwell::EngineStatus::Success) {
                view.on_remote_final(false, "The assistant is busy — try again in a moment.");
            }
        };
        cb.on_mic_toggle = [&](bool listening) {
            // Manual mode mutes the VAD for ALL transitions while PCM keeps
            // flowing, which is exactly "mic off" without tearing the stream
            // down. No explicit on_speech_start follows, so nothing commits.
            (void)speech_pipeline_set_manual_mode(speech_mode.pipeline(), !listening);
        };
        cb.on_settings_apply = [&](const rt::AssistantSettings& next, bool live_only) {
            settings = next;
            if (!rt::save_settings(settings)) {
                std::fprintf(stderr, "[settings] WARN: could not write %s\n",
                             rt::settings_path().c_str());
            }
            if (live_only) apply_live_settings(settings);
            // Restart-tier changes are persisted here and picked up by the next
            // process; the app keeps running on the OLD engine until then, which
            // is the honest state and is what the modal's banner says.
        };
        cb.on_restart = [&] {
            restart_requested.store(true, std::memory_order_release);
            if (window.hwnd() != nullptr) PostMessageW(window.hwnd(), WM_CLOSE, 0, 0);
        };
        window.set_callbacks(std::move(cb));

        if (!window.create(/*client_w=*/860, /*client_h=*/720)) {
            throw std::runtime_error("failed to create the assistant window");
        }

        view.set_transport(transport->name(), transport->is_live());
        view.set_backend(use_real ? "Local model" : "Simulated",
                         use_real ? args.model_dir : std::string("no checkpoint loaded"),
                         use_real && real_stack.audio_head_ready);

        // Diagnostics poller: the gate counters, once a second, into the Settings
        // panel. Deliberately its own thread and deliberately NOT on the chat
        // screen -- see the file preamble.
        std::thread stats_thread([&] {
            // An exception escaping a std::thread is std::terminate. A telemetry
            // poller is the last thing that should be allowed to take the app
            // down, so it swallows and keeps going.
            while (running.load(std::memory_order_acquire)) {
                try {
                    view.publish_stats(commit_queue);
                } catch (...) {
                }
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        });

        std::printf("running... speak, then pause. Interrupt mid-answer to test barge-in.\n");
        std::fflush(stdout);
        window.run_message_loop();

        // ---- shutdown (reverse dependency order) -----------------------------
        realtime.stop();
        capture.stop();
        running.store(false, std::memory_order_release);
        active->stop();
        engine_thread.join();
        stats_thread.join();
        dispatcher.stop();   // after the engine, so no commit can arrive post-join

        std::printf("\n=== commit gate summary ===\n");
        std::printf("  committed (dispatched) : %llu\n",
                    static_cast<unsigned long long>(commit_queue.committed()));
        std::printf("  dropped barge-in       : %llu\n",
                    static_cast<unsigned long long>(commit_queue.dropped_barge_in()));
        std::printf("  dropped token-cap      : %llu%s\n",
                    static_cast<unsigned long long>(commit_queue.dropped_token_cap()),
                    commit_queue.dropped_token_cap() > 0
                        ? "   <-- truncated before EOS; raise the token ceiling"
                        : "");
        std::printf("  dropped queue-full     : %llu\n",
                    static_cast<unsigned long long>(commit_queue.dropped_queue_full()));

        // Relaunched LAST, after this process has released the CUDA context, the
        // capture device and the WebView2 user-data folder -- starting the new
        // instance any earlier would have two processes fighting over all three.
        if (restart_requested.load(std::memory_order_acquire) && !relaunch_self()) {
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
