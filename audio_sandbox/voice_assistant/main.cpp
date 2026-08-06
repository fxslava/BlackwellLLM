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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "audio_capture.h"
#include "audio_devices.h"
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

#if defined(VOICE_ASSISTANT_HAS_WHISPER)
// Mode C, the Cascade. Present only when the build fetched whisper.cpp
// (-DUSE_WHISPER_CPP=ON, the default); without it pipeline_mode="whisper_cascade"
// is refused at startup with a message and the app runs the legacy path.
#include "whisper_cascade_mode.hpp"     // rt::WhisperCascadeMode
#endif

#include "intent_commit.hpp"          // blackwell::bridge::IntentCommitQueue
#include "intent_dispatcher.hpp"      // blackwell::cloud::IntentDispatcher
#include "offline_transport.hpp"      // blackwell::cloud::OfflineTransport
#include "local_transport.hpp"        // rt::LocalEngineTransport, rt::RoutedTransport

#if defined(BLACKWELL_HAVE_CLOUD_CLIENT)
#include "claude_stream_client.hpp"
#include "claude_transport.hpp"
#include "openai_stream_client.hpp"
#include "openai_transport.hpp"
#endif

#if defined(BLACKWELL_VAD_MODEL_PATH)
#include "silero_vad.hpp"
#define VOICE_ASSISTANT_HAS_SILERO 1
#endif

#if defined(BLACKWELL_HAVE_TTS_F5)
// F5-TTS speech output. BLACKWELL_HAVE_TTS_F5 is blackwell_tts_f5's own PUBLIC
// define, which exists only when the ONNXRuntime CUDA execution provider was
// fetched (-DBLACKWELL_ORT_GPU=ON): the DiT measures ~0.2x realtime on CPU, so a
// CPU-provider build would produce audio slower than it plays -- worse than no
// speech at all. Without it the app is text-only and every use site below
// compiles out.
//
// Do NOT gate this on BLACKWELL_HAVE_ORT_CUDA: that define belongs to
// blackwell::onnxruntime, which blackwell_tts_f5 links PRIVATE, so it never
// reaches here and the whole integration silently vanishes from the binary.
#include "tts_runtime.hpp"
// The capture-side half of full duplex. Gated with the TTS because it is the
// loudspeaker that creates the problem it solves: with no speech output there is
// no echo, and no far-end reference to cancel one with.
#include "aec_capture_filter.hpp"
#if defined(BLACKWELL_HAVE_AEC3)
// PIMPL'd: this pulls in no WebRTC header, only the seam and a unique_ptr.
#include "aec3_echo_canceller.hpp"
#endif
#define VOICE_ASSISTANT_HAS_TTS 1
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
    bool output_device = false;
    bool input_device = false;
    bool output_device_index = false;
    bool input_device_index = false;
    bool tts_volume = false;
    bool pipeline_mode = false;
    bool whisper_model = false;
    bool whisper_language = false;
};

// voice_assistant's OWN flags, layered over the shared rt::parse_cli. rt::parse_cli
// THROWS on an unknown argument (a good property -- a typo'd flag must not be
// silently ignored), so the flags that exist only here are removed before it runs.
struct VoiceArgs {
    int device_id = 0;   // --device <id>
    // Audio ENDPOINTS -- unrelated to device_id above, which is a CUDA device.
    // The names are deliberately unambiguous (--audio-output / --audio-input)
    // because "--device" already means something else here.
    std::string output_device;
    std::string input_device;
    // The index form of the same two endpoints, -1 = unset. Kept separate from
    // the names rather than overloading one flag with "a number means an index":
    // an endpoint genuinely called "2" would then be unaddressable by name.
    int output_device_index = -1;
    int input_device_index = -1;
    float tts_volume = 1.0f;
    bool list_audio_devices = false;   // print the endpoints and exit
    bool check_volume = false;         // measure the software gain and exit
    std::string say;                   // --say "<text>": speak it and exit
    // --test-llm-tts ["<prompt>"]: the FULL stack, one injected turn, then exit.
    // Empty string = not requested; the flag supplies a default prompt.
    std::string test_llm_tts;
    // --pipeline <ultravox_legacy|whisper_cascade> and the cascade's two paths.
    // Restart-tier settings all three, so a flag is the ONLY way to try the
    // cascade without opening Settings, saving, and being restarted -- which
    // matters most on exactly the launch where it is being brought up for the
    // first time. Validated by clamp_settings, not here: one validator.
    std::string pipeline_mode;
    std::string whisper_model;
    std::string whisper_language;
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
        } else if (std::strcmp(a, "--list-audio-devices") == 0) {
            v.list_audio_devices = true;
        } else if (std::strcmp(a, "--check-volume") == 0) {
            v.check_volume = true;
        } else if (std::strcmp(a, "--say") == 0 && i + 1 < argc) {
            v.say = argv[++i];
        } else if (std::strcmp(a, "--test-llm-tts") == 0) {
            // Optional argument: the next token is the prompt only if it is not
            // itself a flag, so `--test-llm-tts --real` does not silently
            // swallow --real and then generate a reply to the word "--real".
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                v.test_llm_tts = argv[++i];
            } else {
                v.test_llm_tts = "Привет! Как дела?";
            }
        } else if (std::strcmp(a, "--audio-output") == 0 && i + 1 < argc) {
            v.output_device = argv[++i];
            v.typed.output_device = true;
        } else if (std::strcmp(a, "--audio-input") == 0 && i + 1 < argc) {
            v.input_device = argv[++i];
            v.typed.input_device = true;
        } else if (std::strcmp(a, "--audio-output-index") == 0 && i + 1 < argc) {
            v.output_device_index = std::atoi(argv[++i]);
            v.typed.output_device_index = true;
        } else if (std::strcmp(a, "--audio-input-index") == 0 && i + 1 < argc) {
            v.input_device_index = std::atoi(argv[++i]);
            v.typed.input_device_index = true;
        } else if (std::strcmp(a, "--tts-volume") == 0 && i + 1 < argc) {
            v.tts_volume = static_cast<float>(std::atof(argv[++i]));
            v.typed.tts_volume = true;
        } else if (std::strcmp(a, "--pipeline") == 0 && i + 1 < argc) {
            v.pipeline_mode = argv[++i];
            v.typed.pipeline_mode = true;
        } else if (std::strcmp(a, "--cascade") == 0) {
            // The shorthand, because this is the flag anyone bringing the feature
            // up will type twenty times in a row.
            v.pipeline_mode = "whisper_cascade";
            v.typed.pipeline_mode = true;
        } else if (std::strcmp(a, "--whisper-model") == 0 && i + 1 < argc) {
            v.whisper_model = argv[++i];
            v.typed.whisper_model = true;
        } else if (std::strcmp(a, "--whisper-language") == 0 && i + 1 < argc) {
            v.whisper_language = argv[++i];
            v.typed.whisper_language = true;
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

#if defined(VOICE_ASSISTANT_HAS_TTS)
// ---- --check-volume: does the volume setting actually change the loudness? ---
//
// A self-test rather than a claim. Software volume is one multiply in an audio
// callback, which is exactly the kind of code that is "obviously correct" and
// silently does nothing -- the gain never reaching the callback, or reaching it
// once and never updating, both look identical from the outside and neither
// shows up in a build log.
//
// So this MEASURES it, through the production path and nothing else: the real
// AudioPlayback on the configured endpoint, with WASAPI loopback recording that
// same endpoint's output. It plays a tone at full gain, changes the volume WHILE
// PLAYING (which is the part that must work without a restart), and reports the
// measured ratio of the two.
//
// It makes an audible sound, which is why it is opt-in and brief.
struct ToneState {
    double phase = 0.0;
    double step = 0.0;
};

// A generator can never starve, so every frame it returns is a real one.
std::size_t tone_pull(void* user, float* dst, std::size_t frames) {
    auto* t = static_cast<ToneState*>(user);
    for (std::size_t i = 0; i < frames; ++i) {
        dst[i] = static_cast<float>(0.2 * std::sin(t->phase));
        t->phase += t->step;
        if (t->phase > 6.283185307179586) t->phase -= 6.283185307179586;
    }
    return frames;
}

// Measures the amplitude of the 1 kHz TONE specifically, by Goertzel, rather
// than the broadband RMS of everything the endpoint is playing.
//
// This is not fussiness. Loopback captures the whole system mix, so a broadband
// measurement adds a noise floor `n` in quadrature to both readings: at unity it
// is invisible (sqrt(A^2 + n^2) ~ A), but at a quarter gain it dominates
// sqrt((A/4)^2 + n^2) and inflates the ratio. Measured that way this test
// reported 0.302 for a gain of exactly 0.250 -- correct hardware, misleading
// number, and the one thing a volume self-test must never do is make working
// volume look broken. A single-bin DFT at the tone's own frequency rejects
// everything else in the mix.
//
// Drains first: the ring holds audio produced at the PREVIOUS gain, and
// averaging across the change would report the mean of two answers.
double measure_tone(rt::AudioCapture& cap, int ms) {
    // Capture is fixed at 16 kHz (audio_capture.h), regardless of the rate the
    // tone was synthesised at.
    constexpr double kCaptureRate = 16000.0;
    constexpr double kToneHz = 1000.0;
    const double coeff = 2.0 * std::cos(6.283185307179586 * kToneHz / kCaptureRate);

    std::vector<float> buf(4096);
    while (cap.ring().pop(buf.data(), buf.size()) != 0) {}
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));

    double s1 = 0.0, s2 = 0.0;
    std::size_t n = 0;
    for (;;) {
        const std::size_t got = cap.ring().pop(buf.data(), buf.size());
        if (got == 0) break;
        for (std::size_t i = 0; i < got; ++i) {
            const double s = static_cast<double>(buf[i]) + coeff * s1 - s2;
            s2 = s1;
            s1 = s;
        }
        n += got;
    }
    if (n == 0) return 0.0;
    const double power = s1 * s1 + s2 * s2 - coeff * s1 * s2;
    return 2.0 * std::sqrt(power < 0.0 ? 0.0 : power) / static_cast<double>(n);
}

int run_volume_check(const rt::AssistantSettings& settings) {
    std::printf("=== volume self-test (a 1 kHz tone will play for ~3 seconds) ===\n");
    try {
        rt::AudioPlayback playback;
        ToneState tone;
        tone.step = 6.283185307179586 * 1000.0 /
                    static_cast<double>(blackwell::tts::kF5SampleRate);
        playback.set_volume(1.0f);
        playback.start(blackwell::tts::kF5SampleRate, &tone_pull, &tone,
                       settings.output_device_name, settings.output_device_index);
        std::printf("  output   : %s\n", playback.device_name().empty()
                                             ? "(system default)"
                                             : playback.device_name().c_str());

        // Loopback on the SAME endpoint -- measuring a different speaker would
        // measure nothing. In loopback mode both selectors refer to a PLAYBACK
        // device, which is why the OUTPUT name and index are the ones passed
        // here; the input_* settings would name the wrong list entirely.
        rt::AudioCapture cap;
        cap.start(rt::CaptureMode::Loopback, settings.output_device_name,
                  settings.output_device_index);

        std::this_thread::sleep_for(std::chrono::milliseconds(400));   // device settle
        const double full = measure_tone(cap, 700);

        // THE POINT OF THE TEST: changed while the device is running, with no
        // restart and no re-synthesis.
        constexpr float kQuiet = 0.25f;
        playback.set_volume(kQuiet);
        // Longer than the endpoint's buffer: samples already handed to WASAPI
        // carry the OLD gain and are still going to be emitted.
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        const double quiet = measure_tone(cap, 700);

        cap.stop();
        playback.stop();

        std::printf("  1 kHz level at 100%%: %.5f\n  1 kHz level at  25%%: %.5f\n", full,
                    quiet);
        if (full < 1e-5) {
            std::printf("  INCONCLUSIVE: nothing was captured. The endpoint may not "
                        "support loopback, or it is muted at the OS level.\n");
            return 1;
        }
        const double ratio = quiet / full;
        std::printf("  measured ratio: %.3f (expected %.3f)\n", ratio,
                    static_cast<double>(kQuiet));
        // Still a physical measurement through a shared endpoint, so the band is
        // wider than the instrument: the question is "does the gain apply, live",
        // not "is it accurate to a percent".
        const bool ok = ratio > 0.18 && ratio < 0.33;
        std::printf("  %s\n", ok ? "PASS -- the volume setting takes effect immediately."
                                 : "FAIL -- the gain did not track the setting.");
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "  volume self-test could not run: %s\n", e.what());
        return 1;
    }
}

// --say "<text>": drive the SPEECH HALF of a turn with no LLM and no microphone.
//
// WHAT IT PROVES, AND WHAT IT DOES NOT. It pushes text through exactly the calls
// the dispatcher's answer stream makes -- Resume(), PushToken(), EndOfTurn() --
// so everything downstream of "the answer exists as text" is under test:
// chunker, F5 tokenizer, CUDA synthesis, the speaker ring, the device.
//
// It does NOT test the LLM -> dispatcher -> here wiring, because it stands in
// for the LLM. That half has no headless entry point: a real turn needs a
// microphone or the typed box in the UI. So a PASS here plus a
// "[answer->tts] pushed ... (N chars)" line in a live run is what covers the
// whole path; this alone covers the half where the failure usually is, and is
// the only half that can be checked without a person talking.
int run_say_check(const rt::AssistantSettings& settings, const std::string& text) {
    std::printf("=== speech self-test: synthesising %zu chars ===\n", text.size());
    try {
        rt::TtsRuntimeConfig tcfg;
        tcfg.ckpt_dir            = settings.tts_ckpt_dir;
        tcfg.vocab_path          = settings.tts_vocab_path;
        tcfg.ref_audio           = settings.tts_ref_audio;
        tcfg.ref_text            = settings.tts_ref_text;
        tcfg.nfe_step            = settings.tts_nfe_step;
        tcfg.split_on_commas     = settings.tts_split_on_commas;
        tcfg.min_chunk_chars     = settings.tts_min_chunk_chars;
        tcfg.max_chunk_chars     = settings.tts_max_chunk_chars;
        tcfg.output_device       = settings.output_device_name;
        tcfg.output_device_index = settings.output_device_index;
        tcfg.volume              = settings.tts_volume;

        rt::TtsRuntime tts(tcfg);
        std::printf("  output   : %s\n", tts.output_device().empty()
                                             ? "(system default)"
                                             : tts.output_device().c_str());

        // The dispatcher's exact call sequence. Resume() first for the same
        // reason it is on the dispatch-start edge: without it a barge-in latch
        // left set from a previous turn silently eats every token.
        tts.Resume();
        tts.PushToken(text);
        tts.EndOfTurn();

        // Wait for the worker to drain, bounded. Synthesis runs several times
        // faster than realtime, so anything past this is a hang, not slowness.
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        while (std::chrono::steady_clock::now() < deadline) {
            if (tts.chunks_spoken() > 0 && !tts.speaking()) break;
            if (tts.synthesis_errors() > 0) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        // Let the tail actually reach the speaker before the dtor stops the
        // device -- otherwise a PASS would be reported for audio nobody heard.
        std::this_thread::sleep_for(std::chrono::milliseconds(400));

        const bool ok = tts.chunks_spoken() > 0 && tts.synthesis_errors() == 0;
        std::printf("  chunks spoken: %llu | cancelled: %llu | errors: %llu\n",
                    static_cast<unsigned long long>(tts.chunks_spoken()),
                    static_cast<unsigned long long>(tts.chunks_cancelled()),
                    static_cast<unsigned long long>(tts.synthesis_errors()));
        std::printf("  %s\n", ok ? "PASS -- text handed to the TTS was synthesised and played."
                                 : "FAIL -- nothing reached the speaker; see the [tts-worker] "
                                   "lines above for where it stopped.");
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "  speech self-test could not run: %s\n", e.what());
        return 1;
    }
}
#endif  // VOICE_ASSISTANT_HAS_TTS

// ---- speech-pipeline callbacks (engine / VAD threads) -----------------------
// They may only hand results to the thread-safe view -- never a window, COM,
// CUDA, or a re-entrant pipeline call.
struct AppContext {
    rt::AssistantView* view = nullptr;
    // The BASE type: this is what makes the callbacks backend-agnostic.
    blackwell::bridge::EngineControlBridge* control = nullptr;
#if defined(VOICE_ASSISTANT_HAS_TTS)
    // Null when speech output is unavailable (no models, no GPU, ctor threw).
    // Every use below is guarded, because "the assistant cannot speak" must
    // never become "the assistant cannot answer".
    rt::TtsRuntime* tts = nullptr;
    // The echo canceller sitting in the capture path. Null exactly when `tts` is
    // null: with no loudspeaker there is no echo and no reference to cancel it
    // with. It is a POINTER here rather than being reached through `tts` because
    // it belongs to the capture side -- the settings handler and the PCM tap
    // touch it, and neither of them has any business in the TTS stack.
    blackwell::audio_rt::AecCaptureFilter* aec = nullptr;
    // The loopback device supplying that filter's far end. A POINTER for the
    // same reason `aec` is one: apply_audio_reload has to move it when the
    // OUTPUT endpoint changes, and it is declared long after that lambda.
    rt::AudioCapture* loopback = nullptr;

    // The answer text as it is handed to the speaker, accumulated purely to be
    // LOGGED once the turn ends. It is not the source of what gets spoken --
    // PushToken already streamed every fragment as it arrived, which is what
    // gives time-to-first-audio; re-pushing this at the end would say everything
    // twice.
    //
    // THREADING. on_text runs on the ENGINE thread for the local leg and the
    // DISPATCHER thread for the remote one, and on_complete on the dispatcher
    // thread either way. For the local leg the dispatcher is parked inside
    // send() for the whole generation, so the two never actually overlap -- but
    // "never overlaps" here is a property of another header's blocking
    // behaviour, and a mutex costs nothing on a per-token path that already
    // takes one inside PushToken.
    std::mutex answer_mu;
    std::string answer_text;
#endif
};

// THIS STREAM IS THE USER'S OWN WORDS. Do not connect it to the TTS.
//
// Everything arriving here is decoded by session B, the ephemeral audio
// sequence (RealEngineControl's `audio_`), whose job is to write down what the
// user just said. publish_turn() runs extract_transcript() over it and offers
// the result to the commit gate as an INTENT; the reply to that intent is a
// separate generation entirely, and it surfaces through the dispatcher
// callbacks above -- which is where speech output is wired.
//
// The earlier version gated this on event->is_translation, which reads as if it
// separated the two. It does not: speech_pipeline_controller.cpp sets that flag
// to a hardcoded `true` for ALL decode-loop output, because at that layer there
// is only one decode loop and it cannot see which session ran it. So the gate
// was always open and the assistant recited the user's sentence back at them.
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
#if defined(VOICE_ASSISTANT_HAS_TTS)
    if (ctx->tts != nullptr) {
        switch (next) {
            case SPEECH_STATE_PREFILL_SPEAKING:
                // THE BARGE-IN EDGE. The user has started talking. Whether we are
                // mid-answer or idle, anything still unspoken is now unwanted:
                // this aborts the solver mid-step, drops buffered text, and marks
                // queued audio stale so the device drops it on its next pull.
                ctx->tts->BargeIn();
                break;
            case SPEECH_STATE_INTERRUPTION_REWIND:
                // The pipeline's own barge-in path. Idempotent with the above.
                ctx->tts->BargeIn();
                break;
            case SPEECH_STATE_DECODE_TRANSLATING:
                // NOT a Resume point. This state means the TRANSCRIPTION decode
                // has started -- the user's words are being written down. The
                // answer does not exist yet and may never (the commit gate can
                // reject the turn). Clearing the barge-in latch here would arm
                // the speaker for a generation that is not this one; it is done
                // on the dispatcher's dispatch-start edge instead.
                break;
            case SPEECH_STATE_IDLE:
            default:
                break;
        }
    }
#endif
    // The return to IDLE is the turn boundary (SpeechTokenEvent carries no final
    // flag). The gate has already run by then, so the control's published verdict
    // is the authoritative reason this generation ended -- and it is what the UI
    // must show, because BargeIn and Eos both land on IDLE.
    if (next == SPEECH_STATE_IDLE) {
        ctx->view->on_local_final(ctx->control->last_reason());
        // NO EndOfTurn() here. IDLE is the end of the TRANSCRIPTION pass, which
        // happens BEFORE the intent is gated and long before the answer starts
        // streaming. Flushing the chunker at this point would push the tail of
        // whatever was buffered -- in practice nothing, now that the transcript
        // no longer feeds it -- and would then arrive a second time, mid-answer,
        // as a spurious boundary. The answer's real end is the dispatcher's
        // completion edge, which is where the flush now lives.
    }
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
// SELF-BARGE-IN IS ANSWERED IN THE DSP, NOT HERE.
//
// The loudspeaker feeds the microphone and Silero scores the assistant's own
// voice as speech -- correctly, because it IS speech. Two guards were tried in
// this function and both were removed: a raised probability bar (a statistical
// bound that fails whenever the reference drifts out of alignment) and a hard
// mute keyed on playback state (deterministic, but it starved a recurrent model
// of blocks and withheld barge-in without asking).
//
// Both made the DETECTOR lie about what it heard. The answer now lives one layer
// down, where the problem actually is: the assistant's own voice is SUBTRACTED
// from the capture stream by the echo canceller, fed a WASAPI loopback reference
// of what the speaker is really emitting (see the capture tap below). This
// function does one thing and does it unconditionally.
struct VadContext {
    blackwell::vad::SileroVAD* vad = nullptr;
};

float vad_score(void* user, const float* block, size_t count) {
    auto* ctx = static_cast<VadContext*>(user);
    if (ctx == nullptr || ctx->vad == nullptr) return 0.0f;
    // EVERY BLOCK, unconditionally. Silero is recurrent -- it carries an LSTM
    // state across blocks -- so skipping blocks to suppress a verdict leaves that
    // state describing a signal the microphone never produced.
    return ctx->vad->feed(block, count);
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

    // The console renders in the OEM codepage (866/850) unless told otherwise, so
    // every Cyrillic transcript this app logs -- the decode trace, the commit-gate
    // lines, the system-prefix readout -- comes out as mojibake or question marks
    // in a terminal while the SAME string renders perfectly in the WebView. That
    // asymmetry is worth naming: it makes the log look like the bug when the
    // pipeline is fine. src/apps/main.cpp and benchmark.cpp already do this; this
    // app was the one that did not.
    SetConsoleOutputCP(CP_UTF8);

    // ---- configuration: persisted settings, then typed flags on top ---------
    rt::AssistantSettings settings = rt::load_settings();

    rt::TranslatorArgs args;
    std::vector<char*> shared_argv;
    const VoiceArgs vargs = parse_voice_args(argc, argv, shared_argv);

    // Answered BEFORE anything else is parsed, loaded or opened: this is the
    // flag someone reaches for precisely because the app will not start, or
    // because it started on the wrong speaker. Requiring a valid checkpoint to
    // find out what the audio endpoints are called would defeat it.
    if (vargs.list_audio_devices) {
        rt::print_audio_devices();
        CoUninitialize();
        return 0;
    }

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
        const int rc = run_volume_check(settings);
        CoUninitialize();
        return rc;
#else
        std::fprintf(stderr,
                     "--check-volume needs the speech-output stack, which this build "
                     "does not have (blackwell_tts_f5 absent).\n");
        CoUninitialize();
        return 2;
#endif
    }

    // Same placement and the same reasoning as --check-volume, one tier heavier:
    // this one loads the F5 graphs onto the GPU, but still no checkpoint, no
    // engine and no window.
    if (!vargs.say.empty()) {
#if defined(VOICE_ASSISTANT_HAS_TTS)
        const int rc = run_say_check(settings, vargs.say);
        CoUninitialize();
        return rc;
#else
        std::fprintf(stderr,
                     "--say needs the speech-output stack, which this build does not "
                     "have (blackwell_tts_f5 absent).\n");
        CoUninitialize();
        return 2;
#endif
    }

    // ---- WHICH SPEECH-TO-TEXT PIPELINE ------------------------------------
    // Resolved HERE, before anything is loaded, because it decides what gets
    // ALLOCATED: the Ultravox audio head, the backbone checkpoint, and whether a
    // ~1.6 GB GGML model is pulled onto the GPU.
    bool cascade = (settings.pipeline_mode == "whisper_cascade");
#if !defined(VOICE_ASSISTANT_HAS_WHISPER)
    if (cascade) {
        // Refused, LOUDLY, and clamped back to the path that works. A build with
        // no whisper.cpp cannot honour this setting, and coming up silently on
        // the legacy pipeline would leave the user believing they were testing
        // the cascade.
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
    whisper_model = resolve_asset_file(whisper_model, "Whisper-Turbo-Platinum-F16.bin");
#endif

    // CASCADE'S BACKBONE. With no projector in the graph the backbone width is
    // free (see kDefaultCascadeModelDir), so an unconfigured cascade brings up
    // Qwen rather than falling through to the simulated stand-in -- but only if
    // the checkpoint is actually there. Guessing a path that does not exist would
    // turn "no model configured" into a startup crash, which is strictly worse
    // than the GPU-free backend it would have used.
    if (cascade && settings.model_dir.empty() && path_exists(rt::kDefaultCascadeModelDir)) {
        settings.model_dir = rt::kDefaultCascadeModelDir;
        settings.simulated = false;
        std::printf("[cascade] no backbone configured -- defaulting to %s\n",
                    rt::kDefaultCascadeModelDir);
    }

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
    // FIRST, because it is what every line below means something different
    // under. A reader diagnosing "why is there no [audio] line" needs this one
    // before the ones that would otherwise look like the failure.
    std::printf("  pipeline    : %s\n",
                cascade ? "whisper_cascade (whisper.cpp -> text -> backbone; NO audio head)"
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
    // index-beats-name precedence resolve_device_selection does. Printing both
    // when both are set would be honest about the config and misleading about
    // the behaviour -- and this line exists to predict the behaviour.
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
    // configured; this says what the endpoints are. Read in that order, a
    // mismatch is visible on one screen without scrolling back.
    rt::print_audio_devices();

    try {
        // ---- audio front-end ------------------------------------------------
        whisper::DspConfig cfg;
        // On the LEGACY real path the mel geometry MUST match the projector the
        // audio head was trained with; 128 (Whisper large-v3-turbo) is the
        // simulated default. Resolving it here also fails fast on a bad projector
        // path, before the mic is opened.
        //
        // Cascade mode takes the default and does not consult the projector at
        // all: whisper.cpp computes its own log-mel internally from the PCM it is
        // handed, so this DSP feeds nothing on that path. It is still constructed
        // because RealTimeDSP is what HOSTS the PCM tap both modes read from.
        cfg.n_mels = (use_real && !cascade)
                         ? rt::resolve_projector_params(args.projector_path).num_mel_bins
                         : 128;
        whisper::WhisperDSP dsp(cfg, args.data_dir + "/mel_filters.bin");
        // The spectrogram no longer has a viewer -- it is kept because
        // RealTimeDSP writes into it unconditionally, and shrinking it is a DSP
        // change, not a UI one.
        rt::SpectrogramBuffer spectrogram(cfg.n_mels, /*max_frames=*/1000);
        rt::AudioRecorder recorder(cfg.sample_rate, /*out_dir=*/"recordings");

        rt::AudioCapture capture;
        capture.start(settings.loopback_capture ? rt::CaptureMode::Loopback
                                                : rt::CaptureMode::Microphone,
                      settings.input_device_name, settings.input_device_index);
        std::printf("capture started (%s, 16 kHz mono f32) on %s\n",
                    capture.backend_name().c_str(),
                    capture.device_name().empty() ? "the system default device"
                                                  : capture.device_name().c_str());
        rt::RealTimeDSP realtime(dsp, capture.ring(), spectrogram, &recorder);

        // ---- the control plane ----------------------------------------------
        // Exactly one backend is constructed; `control` is the only handle
        // anything below uses.
        rt::RealEngineStack real_stack;
        std::unique_ptr<rt::SimulatedEngineControl> simulated;
        blackwell::bridge::EngineControlBridge* control = nullptr;

        if (use_real) {
            // isolated_sessions: THIS app is the one that needs it. It runs both
            // jobs -- transcribe the speech, then answer it -- and until now ran
            // them on one sequence, so the transcription inherited the assistant
            // persona and the conversation inherited the transcript's tag format.
            // Two engine sequences (native CoW branching) is the fix; see
            // RealEngineControl's Session block.
            // ISOLATED SESSIONS ARE A LEGACY-PATH NEED, not a general one. They
            // exist because Mode A runs TWO jobs on one engine -- transcribe the
            // speech, then answer it -- and a shared linear context let the
            // transcription inherit the assistant persona. A cascade has only
            // ONE job for the backbone (answer), because whisper.cpp did the
            // transcribing outside the engine entirely. So there is no second
            // sequence to isolate, and asking for branching would only inflate
            // the paged host-mirror pool for a branch nothing forks.
            real_stack = rt::bring_up_real_engine(args, dsp, settings.max_context,
                                                  settings.device_id,
                                                  /*arm_streaming_plan=*/true,
                                                  /*isolated_sessions=*/!cascade,
                                                  /*load_audio_head=*/!cascade);
            control = real_stack.bridge();
            if (!cascade) {
                // Seed the transcription session's prefix source BEFORE the engine
                // thread freezes it: prefill_system_prompt lays BOTH prefixes and
                // reads this one from the control.
                // Both halves of that prefix: the task text and the forced spoken
                // language are composed into one system prompt for seq 1.
                real_stack.control->set_audio_task_prompt(settings.audio_task_prompt);
                real_stack.control->set_speech_language(settings.speech_language);
            }
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
        blackwell::cloud::IIntentTransport* remote = &offline_transport;

#if defined(BLACKWELL_HAVE_CLOUD_CLIENT)
        // PRECEDENCE: the configured OpenAI-compatible endpoint wins, then
        // ANTHROPIC_API_KEY, then the offline stand-in. The settings file is
        // checked first because it is the surface the user can actually SEE --
        // an environment variable silently overriding what the Settings modal
        // shows would be the worst of both.
        std::unique_ptr<blackwell::cloud::OpenAiStreamClient> openai_client;
        std::unique_ptr<blackwell::cloud::OpenAiTransport>    openai_transport;
        std::unique_ptr<blackwell::cloud::ClaudeStreamClient> live_client;
        std::unique_ptr<blackwell::cloud::ClaudeTransport>    live_transport;

        // The key may come from the Settings modal (persisted per machine under
        // %LOCALAPPDATA%) or from OPENAI_API_KEY. It is deliberately NOT
        // checked into settings_store.hpp's defaults: a default in a tracked
        // header enters git history permanently and is compiled into every
        // binary built from the tree. The settings file wins when both are set,
        // because it is the one the user can see.
        std::string remote_key = settings.remote_api_key;
        if (remote_key.empty()) {
            if (const char* env = std::getenv("OPENAI_API_KEY"); env != nullptr) {
                remote_key = env;
            }
        }

        if (!remote_key.empty() && !settings.remote_api_url.empty()) {
            // INIT tier: a curl handle we cannot create is fatal for this leg
            // only, so it degrades to offline rather than taking the app down.
            // The user still has a working assistant and a log line saying why.
            try {
                blackwell::cloud::OpenAiStreamClient::Config ocfg;
                ocfg.api_key = remote_key;
                ocfg.base_url = settings.remote_api_url;
                openai_client =
                    std::make_unique<blackwell::cloud::OpenAiStreamClient>(std::move(ocfg));

                // max_tokens mirrors the LOCAL per-turn cap so one number bounds
                // a turn wherever it is answered -- a remote leg with no ceiling
                // is an unbounded bill on a runaway generation.
                blackwell::cloud::OpenAiRequestOptions oopt;
                oopt.max_tokens = settings.max_new_tokens;
                openai_transport = std::make_unique<blackwell::cloud::OpenAiTransport>(
                    *openai_client, settings.remote_model, oopt);
                remote = openai_transport.get();
                // The KEY IS NEVER PRINTED. The endpoint is, because a wrong
                // base URL is the likeliest misconfiguration and otherwise only
                // surfaces as a 404 on the first real utterance.
                std::printf("[cloud] remote leg: %s (model %s) -- LIVE\n",
                            openai_client->endpoint().c_str(), settings.remote_model.c_str());
            } catch (const std::exception& e) {
                openai_transport.reset();
                openai_client.reset();
                std::fprintf(stderr, "[cloud] remote leg failed to arm (%s) -- falling back\n",
                             e.what());
            }
        }

        if (remote == &offline_transport) {
            if (const char* key = std::getenv("ANTHROPIC_API_KEY");
                key != nullptr && *key != '\0') {
                blackwell::cloud::ClaudeStreamClient::Config ccfg;
                ccfg.api_key = key;
                ccfg.beta = "server-side-fallback-2026-07-01";
                live_client =
                    std::make_unique<blackwell::cloud::ClaudeStreamClient>(std::move(ccfg));
                live_transport =
                    std::make_unique<blackwell::cloud::ClaudeTransport>(*live_client);
                remote = live_transport.get();
                std::printf("[cloud] ANTHROPIC_API_KEY present -- LIVE transport armed\n");
            } else {
                std::printf("[cloud] no remote API key (Settings -> Remote API endpoint, or "
                            "OPENAI_API_KEY) -- Mode: Offline (Simulated)\n");
            }
        }
#else
        std::printf("[cloud] built without BUILD_CLOUD_CLIENT -- Mode: Offline (Simulated)\n");
#endif

        // ---- fully local inference: the second leg of the router -------------
        // The same committed intent, answered by the local backbone instead of
        // sent anywhere. The generator lives on the CONCRETE control (that is
        // where the decode loop is), so it arrives as a callable -- the same seam
        // prefill_system_prompt uses -- and LocalEngineTransport marshals it onto
        // the engine thread. See local_transport.hpp for why this is a transport.
        rt::LocalEngineTransport::GenerateFn generate_locally;
        if (use_real) {
            auto* rc = real_stack.control.get();
            generate_locally = [rc](const std::string& intent,
                                    const std::function<void(std::string_view)>& emit,
                                    blackwell::bridge::TerminationReason* reason) {
                return rc->generate_local_reply(intent, emit, reason);
            };
        } else {
            auto* sc = simulated.get();
            generate_locally = [sc](const std::string& intent,
                                    const std::function<void(std::string_view)>& emit,
                                    blackwell::bridge::TerminationReason* reason) {
                return sc->generate_local_reply(intent, emit, reason);
            };
        }
        rt::LocalEngineTransport local_transport(control, std::move(generate_locally));

        // The dispatcher binds ONE transport for its lifetime, so the router is
        // what makes "answer locally" a live toggle rather than a restart.
        rt::RoutedTransport router(&local_transport, remote);
        router.set_use_local(settings.local_inference);
        blackwell::cloud::IIntentTransport* transport = &router;

        // Declared HERE, above the dispatcher, because the dispatcher's reply
        // callbacks are the TTS tap (see set_on_text below) and they need a
        // handle that is still empty at this point -- app_ctx.tts is filled in
        // once the engine is up. The speech-pipeline callbacks further down take
        // the same object.
        AppContext app_ctx{&view, control};

        // ---- the dispatcher: gate -> remote ---------------------------------
        // The system prompt is EDITABLE at runtime, and the dispatcher thread
        // reads it on every commit -- so it cannot be a plain captured string.
        // A mutex is right here rather than an atomic: the value is a std::string
        // (no lock-free store exists for one), the reader runs once per cloud
        // call, and the writer is a person clicking Save.
        std::mutex prompt_mu;
        std::string system_prompt = settings.system_prompt;
        blackwell::cloud::IntentDispatcher dispatcher(
            commit_queue, *transport,
            [&prompt_mu, &system_prompt] {
                blackwell::cloud::RequestContext c;
                // FROZEN BETWEEN EDITS. Nothing volatile may appear here: a
                // timestamp or a session id in this string drives the prompt-cache
                // hit rate to zero, silently and expensively (intent_request.hpp).
                // A user editing the system prompt DOES invalidate that cache --
                // correctly, since it is a different prompt -- but only once, and
                // only when they asked for it.
                {
                    std::lock_guard<std::mutex> lk(prompt_mu);
                    c.instructions = system_prompt;
                }
                c.glossary = "This is a spoken-language voice assistant session.";
                c.committed_prefix = "";
                return c;
            });
        // ---------------------------------------------------------------------
        // THE SPEECH TAP LIVES HERE, on the ANSWER stream, and nowhere else.
        //
        // These three callbacks are the only place the ASSISTANT'S REPLY exists
        // as text. Everything the speech pipeline emits through on_token is the
        // TRANSCRIPT of what the USER said -- session B (the ephemeral audio
        // sequence) decodes the user's words, publish_turn extracts them, and
        // the gate hands them here as an intent. Tapping on_token therefore
        // reads the user their own sentence back; tapping the dispatcher reads
        // them the answer.
        //
        // It also means local and cloud replies are spoken by the same code:
        // RoutedTransport picks the leg, and neither branch is visible from
        // here (local_transport.hpp's whole argument).
        // ---------------------------------------------------------------------
        dispatcher.set_on_dispatch_start([&view, &app_ctx](const blackwell::bridge::IntentRecord& r) {
            view.on_dispatch_start(r.sequence);
#if defined(VOICE_ASSISTANT_HAS_TTS)
            // A new answer is starting: clear the cancelled latch that the
            // barge-in edge set when the user began speaking, so the tokens
            // about to arrive are actually spoken. This is the ANSWER's start,
            // which is why it is here and not on a pipeline state -- the
            // pipeline's DECODE state belongs to the transcription pass.
            if (app_ctx.tts != nullptr) app_ctx.tts->Resume();
            {
                const std::lock_guard<std::mutex> lk(app_ctx.answer_mu);
                app_ctx.answer_text.clear();
            }
#endif
        });
        dispatcher.set_on_text([&view, &app_ctx](std::string_view s) {
            view.on_remote_token(s);
#if defined(VOICE_ASSISTANT_HAS_TTS)
            // Runs on the ENGINE thread for the local leg and the dispatcher
            // thread for the remote one; PushToken appends under a short mutex
            // and returns, so neither is blocked and all synthesis stays on the
            // TTS worker.
            if (app_ctx.tts != nullptr) app_ctx.tts->PushToken(s);
            {
                const std::lock_guard<std::mutex> lk(app_ctx.answer_mu);
                app_ctx.answer_text.append(s);
            }
#endif
        });
        dispatcher.set_on_complete(
            [&view, &app_ctx](const blackwell::bridge::IntentRecord&,
                              const blackwell::cloud::Result& r) {
                view.on_remote_final(r.status == blackwell::cloud::Status::Ok,
                                     r.error_detail.empty()
                                         ? std::string(blackwell::cloud::to_string(r.status))
                                         : r.error_detail);
#if defined(VOICE_ASSISTANT_HAS_TTS)
                // EOS / stop / error -- whichever ended the generation, the turn
                // is over. Flush so a tail shorter than min_chunk_chars is still
                // spoken. Unconditional on status: a failed answer may still have
                // streamed a partial sentence, and leaving it buffered would
                // splice it onto the FRONT of the next reply.
                if (app_ctx.tts != nullptr) app_ctx.tts->EndOfTurn();

                // THE HANDOFF LINE. This is the seam people go looking for when
                // the assistant answers on screen but says nothing, so it prints
                // the text that reached the speaker and how much of it there was.
                //
                // It is logged HERE, on the answer stream's completion, and not
                // at the commit gate. The gate sits one stage EARLIER and on the
                // other sequence: it decides whether the user's TRANSCRIPT
                // becomes an intent worth dispatching. Its "EOS -> DISPATCHED (N
                // tokens)" line counts transcript tokens, and the answer that
                // follows deliberately never re-enters it -- see the block above
                // on_token(), and generate_local_reply(), which calls
                // finalize_turn() and pointedly not publish_turn().
                //
                // A 0-char line here means the LLM emitted nothing (look up at
                // [Decode Stop]); a non-zero line with no [tts-worker] line
                // after it means the text died between the chunker and the
                // synthesiser, which is the next place to look.
                std::string spoken;
                {
                    const std::lock_guard<std::mutex> lk(app_ctx.answer_mu);
                    spoken.swap(app_ctx.answer_text);
                }
                if (app_ctx.tts != nullptr) {
                    std::printf("[answer->tts] pushed LLM response to TTS (%zu chars): \"%s\"\n",
                                spoken.size(), spoken.c_str());
                    std::fflush(stdout);
                }
#endif
            });
        dispatcher.start();

        // ---- optional Silero neural VAD --------------------------------------
        // Constructed BEFORE the mode and outliving it: the ORT session is a
        // startup-cost resource. Non-fatal, like the audio head.
        // Outlives the pipeline that reads it through vad_user; see vad_score.
        VadContext vad_ctx;
        SpeechVadScoreFn vad_fn = nullptr;
        void* vad_user = nullptr;
#if defined(VOICE_ASSISTANT_HAS_SILERO)
        std::unique_ptr<blackwell::vad::SileroVAD> neural_vad;
        if (args.neural_vad && !args.vad_model.empty()) {
            try {
                neural_vad = std::make_unique<blackwell::vad::SileroVAD>(args.vad_model);
                neural_vad->set_threshold(args.vad_threshold);
                vad_fn = &vad_score;
                vad_ctx.vad = neural_vad.get();
                vad_user = &vad_ctx;
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
        // app_ctx is declared above the dispatcher (it is the TTS tap's handle).
        rt::ConversationalMode::Config mcfg;
        mcfg.sample_rate = static_cast<std::uint32_t>(cfg.sample_rate);
        mcfg.vad_probability_threshold = settings.vad_threshold;
        mcfg.silence_hangover_ms =
            static_cast<std::uint32_t>(settings.silence_hangover_ms);
        mcfg.warm_prefill_interval_ms =
            static_cast<std::uint32_t>(settings.warm_prefill_interval_ms);
        mcfg.pre_roll_ms = settings.pre_roll_ms;
        mcfg.system_prompt = settings.system_prompt;

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

        // ONE of the two is constructed, never both -- they are two answers to the
        // same question and each wants the VRAM the other is holding. `active` is
        // the only handle the runner, the PCM tap and the shutdown path use; the
        // typed handles below exist solely for the handful of call sites that
        // need something the ISpeechMode interface deliberately does not carry.
        std::unique_ptr<rt::ConversationalMode> conv_mode;
#if defined(VOICE_ASSISTANT_HAS_WHISPER)
        std::unique_ptr<rt::WhisperCascadeMode> cascade_mode;
#endif
        rt::ISpeechMode* active = nullptr;

#if defined(VOICE_ASSISTANT_HAS_WHISPER)
        if (cascade) {
            if (whisper_model.empty() || !path_exists(whisper_model)) {
                // REFUSED, not degraded. Falling back to the legacy path here
                // would hand the user a working-looking assistant running a
                // pipeline they did not select, on a model they did not choose.
                throw std::runtime_error(
                    "cascade mode needs a GGML Whisper model, and none was found. Set "
                    "Settings -> Whisper model (or -DBLACKWELL_WHISPER_MODEL at configure "
                    "time, or drop the .bin next to the exe). Looked at: '" +
                    (whisper_model.empty() ? std::string("(nothing configured)") : whisper_model) +
                    "'");
            }

            rt::WhisperAsrConfig acfg;
            acfg.model_path = whisper_model;
            acfg.language   = settings.whisper_language;
            acfg.n_threads  = settings.whisper_threads;
            acfg.use_gpu    = true;   // forced false at compile time on a CPU build
            acfg.gpu_device = settings.device_id;

            rt::WhisperCascadeMode::Config ccfg;
            ccfg.sample_rate      = cfg.sample_rate;
            ccfg.onset_threshold  = settings.vad_threshold;
            ccfg.hangover_ms      = settings.silence_hangover_ms;
            ccfg.preroll_ms       = settings.pre_roll_ms;
            ccfg.max_utterance_ms = settings.whisper_max_utterance_ms;
            ccfg.system_prompt    = settings.system_prompt;

            // The transcript IS the user's bubble, exactly as Mode A's token
            // stream is -- so it lands on the same two view calls, in the same
            // order. The difference is that it arrives whole rather than token by
            // token, and that the gate's verdict comes with it instead of being
            // read back off the control (which never ran a decode loop here).
            //
            // TWO EDGES, NOT ONE, and the mode fires them either side of the
            // commit gate on purpose: on_text must reach the page before the
            // offer wakes the dispatcher, or the assistant's bubble is created
            // first and the answer renders above the question. See the note over
            // WhisperCascadeMode::publish().
            rt::WhisperCascadeMode::TranscriptCallbacks on_transcript;
            on_transcript.on_text = [&view](const std::string& text, std::uint32_t id) {
                view.on_local_token(text.c_str(), id);
            };
            on_transcript.on_verdict = [&view](std::uint32_t /*id*/, bool committed) {
                view.on_local_final(committed
                                        ? blackwell::bridge::TerminationReason::Eos
                                        : blackwell::bridge::TerminationReason::None);
            };

            cascade_mode = std::make_unique<rt::WhisperCascadeMode>(
                control, prefill, vad_fn, vad_user, acfg, ccfg, &on_state,
                std::move(on_transcript), &app_ctx);
            active = cascade_mode.get();
        }
#endif
        if (active == nullptr) {
            conv_mode = std::make_unique<rt::ConversationalMode>(
                control, prefill, vad_fn, vad_user, mcfg, &on_token, &on_state, &app_ctx);
            active = conv_mode.get();
        }

        // Live-tier settings, applied to the RUNNING pipeline. Everything here is
        // an atomic store read at the next VAD block or turn boundary -- no
        // marshaling, no engine work, doctrine intact (see settings_store.hpp).
        //
        // The persona system prompt is the ONE live setting absent from this
        // function: it is a KV cache rebuild, not a store, and goes through
        // post_engine_task (see cb.on_system_prompt_apply below). Under session
        // isolation the AUDIO TASK prompt joins it in that category -- it is the
        // transcription sequence's frozen prefix -- so these track the values the
        // engine is actually running, to rebuild only on a real edit.
        //
        // TWO settings compose that ONE prefix (the task text and the forced
        // spoken language), and the retry is a FLAG rather than a poisoned shadow
        // copy: clearing the copies cannot express "retry" when the value the user
        // just chose is itself the empty string, which is exactly what "no forced
        // language" is.
        std::string applied_audio_task = settings.audio_task_prompt;
        std::string applied_speech_language = settings.speech_language;
        bool audio_prefix_dirty = false;

        // THE SPEAKER MUTE, and the reason it is a local rather than a field of
        // `settings`: it must not be persisted (an app that starts up silent
        // because of a tap three days ago reads as broken), and settings.save
        // physically cannot reach a variable that is not in the struct.
        //
        // It folds with tts_volume at every application site -- the two are never
        // applied independently, because the AEC reference has to describe what
        // the speaker actually emitted and a muted speaker emits nothing.
        //
        // UI-thread only: both writers are window callbacks, which the message
        // loop serialises.
        bool tts_muted = false;
        const auto effective_tts_volume = [&](float slider) noexcept {
            return tts_muted ? 0.0f : slider;
        };
        auto apply_live_settings = [&](const rt::AssistantSettings& s) {
            // Through the INTERFACE, not through Mode A's pipeline handle. Each
            // mode honours the knobs it actually has and ignores the rest (see
            // ISpeechMode) -- which is the only arrangement that works now that
            // there are two modes with different sets of them.
            active->set_vad_threshold(s.vad_threshold);
            active->set_silence_hangover_ms(
                static_cast<std::uint32_t>(s.silence_hangover_ms));
            active->set_warm_prefill_interval_ms(
                static_cast<std::uint32_t>(s.warm_prefill_interval_ms));
            // The pre-roll goes to the CONTROL, not the pipeline: the flush it
            // sizes happens on the engine thread, which is the only consumer
            // allowed to move the ring's read cursor. It is on the base bridge, so
            // it applies to whichever backend is live without a branch.
            control->set_pre_roll_ms(s.pre_roll_ms);

#if defined(VOICE_ASSISTANT_HAS_TTS)
            // Live-toggleable because the filter bypasses in place. Null until
            // the TTS stack is up -- this lambda also runs once BEFORE that, so
            // the launch value is applied at construction instead.
            if (app_ctx.aec != nullptr) app_ctx.aec->SetEnabled(s.aec_enabled);
            // VOLUME GOES TO THE SPEAKER ONLY. The canceller is no longer told
            // about it: its reference is a loopback of the endpoint, so the gain
            // is already baked into what it observes. The pre-gain/post-gain
            // correction this used to need -- and the re-convergence every time
            // the slider moved -- went away with the playback tap.
            //
            // Folded with the dock's mute: a Save while deafened must not turn
            // the sound back on behind the icon, which would leave the UI
            // claiming a state the speaker does not have.
            if (app_ctx.tts != nullptr) app_ctx.tts->SetVolume(effective_tts_volume(s.tts_volume));
#endif
            // The capture-side twin of tts_volume, and live for the same reason:
            // the callback reads it once per block.
            capture.set_input_gain(s.mic_gain);

            // The sampling knobs and the reply ceiling exist on BOTH controls (the
            // simulated one honours the ceiling and stores the rest), so they are
            // applied without asking which backend is live -- the only branch left
            // is the handful of settings that are genuinely real-engine-only.
            if (real_stack.control) {
                real_stack.control->set_sampling(s.temperature, s.top_p);
                real_stack.control->set_max_new_tokens(s.max_new_tokens);
                real_stack.control->set_context_mode(
                    s.context_mode == "bounded"
                        ? rt::RealEngineControl::ContextMode::BoundedHistory
                        : rt::RealEngineControl::ContextMode::Stateless);
                real_stack.control->set_history_budget_tokens(s.history_budget_tokens);
                real_stack.control->set_live_center_slice(s.live_streaming);
                // The task TOGGLES stay a plain store: they only change the
                // generated directives in the per-turn user block.
                real_stack.control->set_tasks(
                    /*transcribe=*/s.speech_task != "translate",
                    /*translate=*/s.speech_task != "transcribe");

                // The AUDIO TASK PROMPT and the SPOKEN LANGUAGE compose the
                // transcription session's frozen system prefix, so editing either
                // is a KV rebuild on that sequence, not a store -- exactly like
                // the persona prompt on the chat sequence. Marshal it (CUDA work
                // belongs to the engine thread) and only when one of them actually
                // changed: a settings push happens on every panel edit, and
                // re-prefilling the prefix on each one would drop a live
                // utterance's audio for nothing. Without isolation there is no
                // second prefix, both values ride in the per-turn user block, and
                // the stores below are the whole operation.
                if (real_stack.isolated_sessions) {
                    if (s.audio_task_prompt != applied_audio_task ||
                        s.speech_language != applied_speech_language) {
                        applied_audio_task = s.audio_task_prompt;
                        applied_speech_language = s.speech_language;
                        audio_prefix_dirty = true;
                    }
                    if (audio_prefix_dirty) {
                        auto* rc = real_stack.control.get();
                        const std::string prompt = applied_audio_task;
                        const std::string language = applied_speech_language;
                        if (control->post_engine_task([rc, prompt, language] {
                                try {
                                    // ORDER: the language is folded INTO the text
                                    // the rebuild prefills, so it has to be
                                    // current before the rebuild reads it.
                                    rc->set_speech_language(language);
                                    rc->rebuild_audio_task_prompt(prompt);
                                } catch (const std::exception& e) {
                                    std::fprintf(stderr,
                                                 "[session] audio task prefix rebuild failed: "
                                                 "%s\n", e.what());
                                }
                            })) {
                            audio_prefix_dirty = false;
                        }
                        // Otherwise the task queue was full: stay dirty so the
                        // next push retries rather than silently running the old
                        // prefix forever.
                    }
                } else {
                    real_stack.control->set_audio_task_prompt(s.audio_task_prompt);
                    real_stack.control->set_speech_language(s.speech_language);
                }
            } else if (simulated) {
                simulated->set_sampling(s.temperature, s.top_p);
                simulated->set_max_new_tokens(s.max_new_tokens);
            }

            // Where the NEXT intent gets answered. One atomic; an answer already
            // streaming finishes on the leg it started on (local_transport.hpp).
            router.set_use_local(s.local_inference);
            // The badge must follow the routing, or a user who switched to local
            // still sees "billed" on a turn that never leaves the machine.
            view.set_transport(router.name(), router.is_live());
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
        // Not on ISpeechMode: the number is a KV floor, which only means anything
        // to a caller that already knows which mode laid the prefix down and how
        // many prefixes it laid. Branching here is cheaper than a virtual that
        // would have to explain itself.
        std::printf("[system-prefix] frozen %u tokens (KV rewind floor)\n",
#if defined(VOICE_ASSISTANT_HAS_WHISPER)
                    cascade_mode ? cascade_mode->frozen_prefix_tokens() :
#endif
                                 conv_mode->frozen_prefix_tokens());
        apply_live_settings(settings);

        // ---- audio hot reload -------------------------------------------------
        // Moves the two ma_devices to whatever endpoints the settings now name,
        // and NOTHING else. Declared here because it needs `capture` (above) and
        // `app_ctx.tts` (below, via the pointer), and is called from the settings
        // handler at the bottom.
        //
        // ORDER: playback first, then capture, then the AEC reset. Playback is
        // what the canceller's reference describes, so resetting the filter
        // before the new speaker exists would just make it converge on the old
        // one for another few hundred milliseconds.
        auto apply_audio_reload = [&](const rt::AssistantSettings& s) {
#if defined(VOICE_ASSISTANT_HAS_TTS)
            if (app_ctx.tts != nullptr) {
                const bool ok = app_ctx.tts->HotReloadOutput(s.output_device_name,
                                                             s.output_device_index);
                std::printf("[audio] output hot reload %s -> %s\n", ok ? "ok" : "FAILED",
                            app_ctx.tts->output_device().empty()
                                ? "(system default)"
                                : app_ctx.tts->output_device().c_str());
            }
#endif
            // Loopback capture taps a PLAYBACK endpoint, so it follows the OUTPUT
            // selection -- the same trap the name form has always had, restated
            // here because getting it wrong yields MA_NO_DEVICE with no hint.
            const bool loopback = s.loopback_capture;
            const bool cap_ok = capture.hot_reload(
                loopback ? rt::CaptureMode::Loopback : rt::CaptureMode::Microphone,
                loopback ? s.output_device_name : s.input_device_name,
                loopback ? s.output_device_index : s.input_device_index);
            std::printf("[audio] input hot reload %s -> %s\n", cap_ok ? "ok" : "FAILED",
                        capture.device_name().empty() ? "(system default)"
                                                      : capture.device_name().c_str());
            capture.set_input_gain(s.mic_gain);
#if defined(VOICE_ASSISTANT_HAS_TTS)
            // THE canceller's learned impulse response describes the OLD room
            // path -- old speaker, old microphone, old latency between them. Left
            // alone it would spend a few hundred milliseconds actively
            // subtracting the wrong signal, which is worse than not cancelling.
            // The loopback reference follows the OUTPUT endpoint, because that
            // is the speaker whose echo it exists to describe. Moved BEFORE the
            // filter is reset, so the reset lands on a reference that is already
            // pointing at the new room.
            if (app_ctx.loopback != nullptr) {
                const bool lb_ok = app_ctx.loopback->hot_reload(
                    rt::CaptureMode::Loopback, s.output_device_name, s.output_device_index);
                std::printf("[aec] loopback reference hot reload %s -> %s\n",
                            lb_ok ? "ok" : "FAILED",
                            app_ctx.loopback->device_name().empty()
                                ? "(system default output)"
                                : app_ctx.loopback->device_name().c_str());
            }
            if (app_ctx.aec != nullptr) app_ctx.aec->Reset();
#endif
            std::fflush(stdout);
        };

        // ---- speech output ---------------------------------------------------
        // Built AFTER the engine so a TTS failure cannot delay the thing the app
        // is actually for, and BEFORE the UI so the first reply can be spoken.
        // INIT tier: a throw here disables speech and the assistant runs on --
        // the same posture already taken for the neural VAD and the audio head.
#if defined(VOICE_ASSISTANT_HAS_TTS)
        std::optional<rt::TtsRuntime> tts;

        // ---- THE FAR END: a WASAPI loopback of the render endpoint ----------
        // The canceller's reference is what the SPEAKER emits, taken from the
        // speaker rather than from us. A second AudioCapture in Loopback mode
        // taps the output endpoint; miniaudio hands it back at the same 16 kHz
        // mono f32 the microphone runs at, so there is NO resampler on this path
        // at all -- the 24 kHz -> 16 kHz conversion the old reference needed is
        // simply gone.
        //
        // WHAT THIS BUYS over tapping PullForPlayback. The reference is
        // post-mix and post-volume: it already contains the software gain (so
        // the canceller never has to be told about it, and never re-converges
        // when the slider moves) and it contains audio THIS PROCESS DID NOT
        // PRODUCE -- a browser, a notification, anything else on the endpoint --
        // all of which used to reach the microphone as uncancellable echo.
        //
        // WHAT IT DOES NOT BUY, stated so nobody looks for it: the microphone
        // and the render endpoint are still two devices on two independent
        // clocks, so reference/mic drift is unchanged and AecCaptureFilter's
        // backlog policy is still doing that job.
        //
        // Loopback runs continuously and returns SILENCE when nothing is
        // playing, which is exactly what the synchroniser wants -- a reference
        // stream with no gaps in it.
        rt::AudioCapture loopback_capture;
        // The far-end ring the filter reads. ~3 s at 16 kHz: it only ever holds
        // the drift between the loopback callback and the DSP worker, and the
        // filter's own ceiling discards anything older than max_lag_ms.
        blackwell::audio_rt::SpscRing<float> loopback_ref(48000);

        // DECLARED AFTER both, DELIBERATELY. It holds a reference into
        // `loopback_ref`, so it must be destroyed BEFORE the ring it points at --
        // which reverse declaration order is exactly what gives.
        // (CLAUDE.md extension pattern #3, applied to a function scope.)
        std::optional<blackwell::audio_rt::AecCaptureFilter> aec;
        if (!settings.tts_ckpt_dir.empty()) {
            rt::TtsRuntimeConfig tcfg;
            tcfg.ckpt_dir   = settings.tts_ckpt_dir;
            tcfg.vocab_path = settings.tts_vocab_path;
            tcfg.ref_audio  = settings.tts_ref_audio;
            tcfg.ref_text   = settings.tts_ref_text;
            tcfg.nfe_step   = settings.tts_nfe_step;
            tcfg.device_id  = device_id;
            tcfg.split_on_commas  = settings.tts_split_on_commas;
            tcfg.min_chunk_chars  = settings.tts_min_chunk_chars;
            tcfg.max_chunk_chars  = settings.tts_max_chunk_chars;
            tcfg.output_device       = settings.output_device_name;
            tcfg.output_device_index = settings.output_device_index;
            tcfg.volume           = settings.tts_volume;

            // THE PAIR MUST MATCH. F5 treats generation as infilling against
            // (reference audio, reference text), so a transcript that is not
            // what the clip says makes the model invent content to reconcile
            // them -- the reference bleeds into every utterance, which reads as
            // a broken model rather than a misconfiguration.
            //
            // The shipped default is now a REAL matched pair (a FLEURS clip and
            // its ground-truth transcript), so "equals the default" is no longer
            // the fault condition -- it is the good case. What is still worth
            // catching is an empty transcript, and the half-edited state: a
            // custom clip still carrying the shipped text. Both are silent
            // otherwise, and both cost a 1.3 GB graph load to discover by ear.
            const rt::AssistantSettings kDefaults{};
            if (settings.tts_ref_text.empty()) {
                std::fprintf(stderr,
                             "[tts] WARNING: reference transcript is EMPTY. Set it to the "
                             "literal text spoken in %s (Settings -> Audio -> TTS), or the "
                             "voice will be wrong.\n",
                             tcfg.ref_audio.c_str());
            } else if (settings.tts_ref_audio != kDefaults.tts_ref_audio &&
                       settings.tts_ref_text == kDefaults.tts_ref_text) {
                std::fprintf(stderr,
                             "[tts] WARNING: reference clip was changed to %s but the "
                             "transcript is still the one shipped for %s. They must "
                             "describe the SAME audio (Settings -> Audio -> TTS).\n",
                             tcfg.ref_audio.c_str(), kDefaults.tts_ref_audio.c_str());
            }
            try {
                tts.emplace(tcfg);
                app_ctx.tts = &tts.value();
                std::printf("[tts] ready: %s (nfe=%d, volume %.0f%%) on %s\n",
                            tcfg.ckpt_dir.c_str(), tcfg.nfe_step,
                            static_cast<double>(tts->volume()) * 100.0,
                            tts->output_device().empty() ? "the system default device"
                                                         : tts->output_device().c_str());
                // The last GPU consumer to load, so this line is the WHOLE
                // budget: if it says OVER BUDGET, the box is already paging and
                // every latency number after it is meaningless.
                //
                // MEASURED, so nobody re-litigates it from guesses: the 1.31 GB
                // of fp32 DiT weights are allocated THROUGH the CUDA EP arena,
                // not beside it. A sweep of the old gpu_mem_limit_mb cap showed
                // 2048 MiB costing 2219 MiB committed, 1792 costing the same
                // 2219, and 1536 failing to initialise outright.
                //
                // That sweep is why the cap is now 0 (unbounded) by default. It
                // only ever measured the WEIGHTS -- a cap of 2048 leaves the
                // graph ~700 MiB for its attention intermediates, which is
                // enough to load and enough to synthesise a SHORT utterance, so
                // the sweep read as "no headroom left" rather than as what it
                // was: a ceiling one long sentence away from failing inside
                // Softmax. Confirmed since -- a 335-token utterance asks one
                // attention node for 385 MB and dies against the cap, and
                // succeeds without it. See F5TtsConfig::gpu_mem_limit_mb.
                //
                // So this line is the WHOLE budget in a stronger sense than
                // before: nothing bounds F5 but the box. The reducible items are
                // max_context, the fp32 audio head, and F5's max_frames.
                if (use_real) rt::report_vram("+ TTS (full stack)");

                // ---- the capture-side half of full duplex --------------------
                // Built here and not earlier because it needs the far-end ring,
                // which only exists once the runtime does. It is the SOLE
                // consumer of that ring, per its SPSC contract.
                //
                // ITS OWN try/catch, and that is not tidiness. Speech output is
                // already live and app_ctx.tts is already published by this
                // point, so letting a throw fall into the handler below would
                // print "[tts] disabled" about a TTS stack that is running --
                // and would leave it running with no canceller, which is the one
                // configuration that self-triggers. The two failures are
                // different and have to say so.
                try {
                    // THE LOOPBACK REFERENCE, opened on the OUTPUT endpoint --
                    // both selectors address the PLAYBACK list in this mode
                    // (audio_capture.h says so, and it is the trap here: an
                    // index that looks like a capture index is not one).
                    //
                    // Started BEFORE the filter exists so a failure to open is
                    // reported as "no reference" rather than as a canceller that
                    // silently subtracts nothing.
                    loopback_capture.start(rt::CaptureMode::Loopback, settings.output_device_name,
                                   settings.output_device_index);

                    blackwell::audio_rt::AecCaptureFilterConfig acfg;
                    acfg.near_rate = static_cast<int>(cfg.sample_rate);
                    // SAME RATE, so the polyphase resampler degenerates to a
                    // copy: miniaudio already delivers the loopback stream at the
                    // capture rate. The old 24 kHz far end needed a 2/3 converter
                    // and paid its group delay out of the filter's tail budget.
                    acfg.far_rate = acfg.near_rate;
                    // Clamped, not trusted: this arrives from a hand-editable
                    // settings file, and a tail of zero produces a canceller
                    // that runs and cancels nothing.
                    acfg.aec.filter_tail_samples =
                        static_cast<std::size_t>(std::clamp(settings.aec_tail_ms, 64, 1000)) *
                        static_cast<std::size_t>(acfg.near_rate) / 1000u;
                    // ---- the subtractor backend ---------------------------
                    // AEC3 when the package was found at configure time, the
                    // built-in partitioned-block filter otherwise. The choice is
                    // made HERE rather than inside AecCaptureFilter because
                    // blackwell_audio_rt is dependency-free by construction and
                    // must not learn the name of a WebRTC type.
                    //
                    // A THROW HERE IS NOT FATAL and deliberately falls through
                    // to the built-in filter: an AEC3 that will not construct is
                    // a reason to cancel worse, not a reason to have no speech.
                    std::unique_ptr<blackwell::audio_rt::IEchoCanceller> backend;
                    const char* backend_name = "built-in block-FDAF";
#if defined(BLACKWELL_HAVE_AEC3)
                    try {
                        blackwell::audio_rt::Aec3Config a3;
                        a3.sample_rate_hz = acfg.near_rate;
                        // A loopback reference is tapped at the endpoint, so the
                        // true speaker->mic delay is the render buffer plus the
                        // capture buffer -- tens of milliseconds, not a room's
                        // worth. AEC3 re-estimates regardless; this only saves it
                        // the first second of searching.
                        a3.initial_delay_ms = 30;
                        backend = std::make_unique<blackwell::audio_rt::Aec3EchoCanceller>(a3);
                        backend_name = "WebRTC AEC3";
                    } catch (const std::exception& e) {
                        std::fprintf(stderr,
                                     "[aec] AEC3 unavailable (%s) -- falling back to the "
                                     "built-in canceller.\n",
                                     e.what());
                        backend.reset();
                    }
#endif
                    aec.emplace(loopback_ref, acfg, std::move(backend));
                    aec->SetEnabled(settings.aec_enabled);
                    // NO SetReferenceGain. The loopback stream is tapped after
                    // the mix, so it already carries whatever gain the user set
                    // -- the correction the playback tap needed (and the
                    // re-convergence every time the slider moved) is retired.
                    app_ctx.aec = &aec.value();
                    app_ctx.loopback = &loopback_capture;
                    std::printf("[aec] %s (%s): %d ms tail, %zu-sample capture latency, "
                                "far end = WASAPI loopback @ %d Hz on %s\n",
                                settings.aec_enabled ? "on" : "BYPASSED (diagnostic)",
                                backend_name,
                                settings.aec_tail_ms, aec->latency_samples(),
                                acfg.far_rate,
                                loopback_capture.device_name().empty()
                                    ? "(system default output)"
                                    : loopback_capture.device_name().c_str());
                    if (loopback_capture.device_fallback()) {
                        // THE failure that produces a canceller which subtracts
                        // the wrong room: the reference must come from the
                        // endpoint the assistant is SPEAKING through, and a
                        // silent fallback to a different one is uncancellable.
                        std::fprintf(stderr,
                                     "[aec] WARNING: the loopback reference fell back to a "
                                     "different endpoint than the one speech plays on -- "
                                     "cancellation will not work until they match.\n");
                    }
                    if (!settings.aec_enabled) {
                        // Said out loud because it is the difference between a
                        // demo that works and one that talks over itself: with
                        // the canceller bypassed, the assistant's own voice
                        // reaches the mic on open speakers, Silero scores it as
                        // speech, and the pipeline barges in on the answer it is
                        // currently giving.
                        std::fprintf(stderr,
                                     "[aec] WARNING: echo cancellation is OFF. Use "
                                     "HEADPHONES, or the assistant will interrupt itself.\n");
                    }
                } catch (const std::exception& e) {
                    // Speech still works; what is lost is the ability to survive
                    // hearing it. Degrade loudly rather than silently.
                    std::fprintf(stderr,
                                 "[aec] DISABLED: %s\n"
                                 "[aec] The assistant will hear its own voice and may "
                                 "interrupt itself. Use HEADPHONES.\n",
                                 e.what());
                }
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[tts] disabled: %s\n", e.what());
            }
        } else {
            std::printf("[tts] disabled (no --tts-ckpt-dir / persisted setting)\n");
        }
#endif

        // THE CAPTURE TAP -- and the whole of what full duplex means here.
        //
        // The microphone is still NEVER muted, gated, zeroed or paused. Every
        // block the capture device produces reaches the AEC, the level meter and
        // the ring, exactly as before: the PCM path is untouched, so pre-roll,
        // metering and speculative warming all keep working while the assistant
        // talks. There is no state flag anywhere on this path and no verdict is
        // masked -- what removes the assistant's voice is a SUBTRACTION, and the
        // thing being subtracted is a loopback of the speaker itself.
        //
        // Runs on the DSP worker, which is this pipeline's single producer, so
        // the scratch buffers need no synchronisation. They are sized ONCE here:
        // RealTimeDSP pops at most 4096 samples per call, so the resizes below
        // never fire again after the first block and the tap stays
        // allocation-free on the path it shares with the capture drain.
#if defined(VOICE_ASSISTANT_HAS_TTS)
        std::vector<float> aec_out(4096, 0.0f);
        std::vector<float> far_scratch(4096, 0.0f);
#endif
        realtime.set_pcm_tap(
            [&](const float* s, std::size_t n) {
#if defined(VOICE_ASSISTANT_HAS_TTS)
                if (app_ctx.aec != nullptr) {
                    // ---- pump the far end ----------------------------------
                    // The loopback device writes into its own SampleRing on its
                    // own callback thread; the filter reads a SpscRing. This
                    // moves one from the other, and it happens HERE because this
                    // thread is the filter's only consumer -- so the SpscRing
                    // ends up written and read by the same thread, which trivially
                    // satisfies its SPSC contract.
                    //
                    // Drained to EMPTY rather than n samples: the two devices
                    // deliver on independent schedules, and leaving a residue
                    // would let the reference fall progressively behind the
                    // microphone -- the one direction of misalignment no causal
                    // filter can represent (aec_capture_filter.hpp).
                    for (;;) {
                        const std::size_t got =
                            loopback_capture.ring().pop(far_scratch.data(), far_scratch.size());
                        if (got == 0) break;
                        // write_or_drop: this is a real-time-ish path and the
                        // filter's backlog ceiling would discard the excess
                        // anyway. A ring that is full means the filter is not
                        // keeping up, which its own resync counter reports.
                        (void)loopback_ref.write_or_drop(far_scratch.data(), got);
                        if (got < far_scratch.size()) break;
                    }

                    if (aec_out.size() < n) aec_out.resize(n);
                    app_ctx.aec->Process(s, n, aec_out.data());
                    active->on_pcm_block(aec_out.data(), n);
                    return;
                }
#endif
                // No canceller in this build or this launch: pass the microphone
                // through untouched.
                active->on_pcm_block(s, n);
            });
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
            // Through the interface: both modes implement it, with the same
            // meaning on each (Mode C's WhisperCascadeMode::set_manual_mode says
            // what it does and does not do).
            active->set_manual_mode(!listening);
        };
        cb.on_cancel = [&] {
            // The cancel hotkey IS a barge-in, minus the speech: bump the epoch
            // and the in-flight decode loop aborts at its next token check. Same
            // mechanism the VAD uses, so a cancelled turn lands as BargeIn and is
            // correctly NOT dispatched.
            control->cancel_generation(control->active_generation() + 1);
#if defined(VOICE_ASSISTANT_HAS_TTS)
            // The button must also stop the SOUND, not just the decode. Without
            // this the engine halts while the speaker keeps playing everything
            // already synthesised -- which reads as the button not working.
            if (app_ctx.tts != nullptr) app_ctx.tts->BargeIn();
#endif
        };
        cb.on_settings_apply = [&](const rt::AssistantSettings& next, bool live_only) {
            // THE HOT-RELOAD CHECK, before `settings` is overwritten -- it is a
            // comparison against the values currently in force, so it has to run
            // while those are still readable.
            //
            // Reopening two ma_devices takes tens of milliseconds and touches no
            // VRAM: the engine keeps its 5.3 GB of weights, the KV pool stays
            // allocated and the F5 ONNX session is not reloaded. That is the
            // entire reason these settings are their own tier.
            const bool audio_moved = rt::requires_audio_reload(settings, next);
            settings = next;
            if (!rt::save_settings(settings)) {
                std::fprintf(stderr, "[settings] WARN: could not write %s\n",
                             rt::settings_path().c_str());
            }
            // Live settings apply on EVERY save, restart-tier or not: refusing to
            // update the VAD threshold because an unrelated checkpoint path also
            // changed would be a worse answer than doing what can be done now.
            // `live_only` decides whether the app also restarts, nothing else.
            apply_live_settings(settings);
            if (audio_moved) apply_audio_reload(settings);
            (void)live_only;
            // Restart-tier changes are persisted here and picked up by the next
            // process; the app keeps running on the OLD engine until then, which
            // is the honest state and is what the modal's banner says.
        };
        cb.on_system_prompt_apply = [&](const std::string& prompt) {
            // The cloud side first, because it is a plain string swap and must not
            // be left describing the old prompt if the GPU rebuild fails.
            {
                std::lock_guard<std::mutex> lk(prompt_mu);
                system_prompt = prompt;
            }
            // THE marshal. This runs on the UI thread; prefill is CUDA work on a
            // 5.3 GB weight set and belongs to the engine thread alone (CLAUDE.md).
            // post_engine_task runs it at the next command-batch boundary -- after
            // any turn already decoding, before any turn not yet started.
            const bool queued = control->post_engine_task([&, prompt] {
                try {
                    const std::uint32_t n =
                        real_stack.control
                            ? real_stack.control->rebuild_system_prompt(prompt)
                            : simulated->rebuild_system_prompt(prompt);
                    std::printf("[system-prefix] rebuilt: %u tokens frozen (KV rewind floor)\n",
                                n);
                    std::fflush(stdout);
                    window.post_system_prompt_applied(true, n, {});
                } catch (const std::exception& e) {
                    // A failed rebuild leaves a SHORTER but valid prefix (see
                    // prefill_system_prompt), so the app keeps working -- the user
                    // just has to be told the prompt is not what they typed.
                    std::fprintf(stderr, "[system-prefix] rebuild FAILED: %s\n", e.what());
                    window.post_system_prompt_applied(false, 0, e.what());
                }
            });
            if (!queued) {
                window.post_system_prompt_applied(
                    false, 0, "Could not reach the engine thread — the prompt was saved "
                              "but not applied. Restart to load it.");
            }
        };
        cb.on_audio_hot_update = [&](const rt::AudioHotUpdate& u) {
            // Merge into the live settings. ONLY these fields are touched, so
            // whatever the settings modal is holding cannot be dragged along --
            // which is exactly the bug this path exists to remove.
            if (u.output_device_index != rt::AudioHotUpdate::kNoIndex) {
                settings.output_device_index = u.output_device_index;
            }
            if (u.input_device_index != rt::AudioHotUpdate::kNoIndex) {
                settings.input_device_index = u.input_device_index;
            }
            if (u.has_output_name) settings.output_device_name = u.output_device_name;
            if (u.has_input_name)  settings.input_device_name  = u.input_device_name;
            if (u.tts_volume >= 0.0f) settings.tts_volume = u.tts_volume;
            if (u.mic_gain   >= 0.0f) settings.mic_gain   = u.mic_gain;
            // Mute is session state and is deliberately NOT merged into
            // `settings` -- see AudioHotUpdate::tts_muted on why it must not
            // survive a restart. It lives in this one bool, which the save
            // below therefore cannot reach.
            if (u.tts_muted != rt::AudioHotUpdate::Tri::Absent) {
                tts_muted = (u.tts_muted == rt::AudioHotUpdate::Tri::On);
            }
            rt::clamp_settings(settings);

            // Gains first, and unconditionally: they are atomic stores, they are
            // what the user is listening to while dragging, and they must not
            // wait behind a device reopen that may not even be needed.
            //
            // THE EFFECTIVE GAIN is the slider folded with the mute. It goes to
            // the speaker and nowhere else: the canceller observes the endpoint
            // through a loopback, so it sees this change by itself, at the same
            // instant the room does, with no second setter to keep in step.
            const float effective_volume = effective_tts_volume(settings.tts_volume);
#if defined(VOICE_ASSISTANT_HAS_TTS)
            if (app_ctx.tts != nullptr) app_ctx.tts->SetVolume(effective_volume);
#endif
            capture.set_input_gain(settings.mic_gain);

            // The device reopen happens ONLY when an endpoint actually moved. A
            // slider drag arrives as dozens of messages; reopening WASAPI on each
            // would turn a volume change into an audible stutter.
            if (u.needs_device_reload()) {
                const auto t0 = std::chrono::steady_clock::now();
                apply_audio_reload(settings);
                const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::steady_clock::now() - t0).count();
                // The line that proves the claim: a device swap, timed, with no
                // engine or arena activity between the two timestamps.
                std::printf("[audio-hot-reload] Swapped WASAPI device in %lld ms "
                            "(engine untouched, VRAM unchanged)\n",
                            static_cast<long long>(ms));
                std::fflush(stdout);
            }

            // Persisted so the choice survives a restart -- but note the
            // ordering: the device is ALREADY live by here. A failed write costs
            // the user the setting next launch, not the sound they just fixed.
            if (!rt::save_settings(settings)) {
                std::fprintf(stderr, "[settings] WARN: could not persist audio settings\n");
            }
        };
        cb.on_test_tone = [&] {
#if defined(VOICE_ASSISTANT_HAS_TTS)
            // Through the live TTS pipeline, not a second private device: the
            // point of the button is to prove the endpoint the ASSISTANT speaks
            // through makes sound. A tone on its own device could pass while
            // speech stayed silent, which is exactly the confusion it exists to
            // end.
            if (app_ctx.tts != nullptr) app_ctx.tts->PlayTestTone();
#endif
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
            // The gate counters move at conversation pace; the mic meter has to
            // move at speech pace or it reads as broken. So the loop runs at the
            // METER's rate and the counters are published every Nth pass.
            constexpr int kMeterMs = 80;              // ~12 fps, smooth enough
            constexpr int kStatsEvery = 1000 / kMeterMs;
            int tick = 0;
            while (running.load(std::memory_order_acquire)) {
                try {
                    if (tick % kStatsEvery == 0) view.publish_stats(commit_queue);
                    window.post_audio_level(capture.input_level());
                } catch (...) {
                }
                ++tick;
                std::this_thread::sleep_for(std::chrono::milliseconds(kMeterMs));
            }
        });

        // ---- --test-llm-tts: one injected turn on the COMPLETE stack ----------
        // Everything is already constructed at this point -- engine, arena, audio
        // head, Silero, AEC, F5 -- so this measures the loaded machine, which is
        // the entire point. It exists because --say does NOT: --say stands in for
        // the LLM, so it runs F5 on an otherwise idle GPU and therefore cannot
        // reproduce anything caused by decode and synthesis sharing the device.
        //
        // The prompt goes in through submit_text, the SAME entry the typed box
        // uses and therefore the same ring, gate, dispatcher and answer stream a
        // spoken turn takes. Reaching into generate_local_reply() directly would
        // have been fewer lines and would have tested a path no user can take.
        std::thread test_thread;
        if (!vargs.test_llm_tts.empty()) {
            test_thread = std::thread([&] {
                try {
                    const std::string prompt = vargs.test_llm_tts;
                    std::printf("\n=== full-stack LLM+TTS test ===\n  prompt: \"%s\"\n",
                                prompt.c_str());
#if defined(VOICE_ASSISTANT_HAS_TTS)
                    if (tts.has_value()) tts->reset_playback_stats();
#endif
                    // Let the device settle so the first buffers are not counted
                    // as starvation from before there was anything to play.
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));

                    view.on_user_text(prompt);
                    if (control->submit_text(prompt, text_sink) !=
                        blackwell::EngineStatus::Success) {
                        std::printf("  FAIL -- submit_text was refused (command ring full).\n");
                    } else {
                        // Sample WHILE the turn runs: a summary printed after the
                        // fact cannot show whether the ring ran dry during decode,
                        // which is the whole question.
                        const auto deadline =
                            std::chrono::steady_clock::now() + std::chrono::seconds(120);
                        bool spoke = false;
                        while (std::chrono::steady_clock::now() < deadline) {
                            std::this_thread::sleep_for(std::chrono::milliseconds(500));
#if defined(VOICE_ASSISTANT_HAS_TTS)
                            if (!tts.has_value()) break;
                            const auto ps = tts->playback_stats();
                            std::printf("[audio-playback] WASAPI requested %u frames | "
                                        "RingBuffer available: %u frames | starved %llu/%llu "
                                        "callbacks | ring underruns %llu | chunks %llu\n",
                                        ps.last_requested, ps.last_available,
                                        static_cast<unsigned long long>(ps.starved_callbacks),
                                        static_cast<unsigned long long>(ps.callbacks),
                                        static_cast<unsigned long long>(tts->speaker_underruns()),
                                        static_cast<unsigned long long>(tts->chunks_spoken()));
                            std::fflush(stdout);
                            if (tts->chunks_spoken() > 0) spoke = true;
                            // Done when the speaker has drained after speaking.
                            if (spoke && !tts->speaking()) break;
#else
                            break;
#endif
                        }
                    }
#if defined(VOICE_ASSISTANT_HAS_TTS)
                    std::this_thread::sleep_for(std::chrono::milliseconds(300));
                    if (tts.has_value()) {
                        const auto ps = tts->playback_stats();
                        std::printf("\n  --- playback under load ---\n");
                        std::printf("  callbacks              : %llu\n",
                                    static_cast<unsigned long long>(ps.callbacks));
                        std::printf("  frames requested/served: %llu / %llu\n",
                                    static_cast<unsigned long long>(ps.frames_requested),
                                    static_cast<unsigned long long>(ps.frames_served));
                        std::printf("  starved callbacks      : %llu\n",
                                    static_cast<unsigned long long>(ps.starved_callbacks));
                        std::printf("  speaker-ring underruns : %llu\n",
                                    static_cast<unsigned long long>(tts->speaker_underruns()));
                        std::printf("  chunks spoken/cancelled: %llu / %llu   errors %llu\n",
                                    static_cast<unsigned long long>(tts->chunks_spoken()),
                                    static_cast<unsigned long long>(tts->chunks_cancelled()),
                                    static_cast<unsigned long long>(tts->synthesis_errors()));
                        std::printf("  %s\n",
                                    tts->chunks_spoken() > 0 && tts->synthesis_errors() == 0
                                        ? "PASS -- audio was synthesised and played while the "
                                          "model was decoding."
                                        : "FAIL -- nothing reached the speaker during the turn.");
                    }
#endif
#if defined(VOICE_ASSISTANT_HAS_TTS)
                    // ---- hot swap, on the loaded stack -----------------------
                    // The acceptance test for the AudioHotReload tier: move the
                    // endpoint with 8 GB of weights resident and show that the
                    // VRAM figure does not move. A reload would be unmissable
                    // here -- the arena alone is gigabytes.
                    if (tts.has_value()) {
                        std::printf("\n  --- audio hot swap (no model reload) ---\n");
                        if (use_real) rt::report_vram("  before swap");
                        rt::AssistantSettings swapped = settings;
                        // Deliberately to a DIFFERENT endpoint than the one in
                        // use, so a no-op cannot pass as a success.
                        swapped.output_device_index =
                            (settings.output_device_index == 4) ? 0 : 4;
                        swapped.output_device_name.clear();
                        apply_audio_reload(swapped);
                        std::printf("  now on: %s\n", tts->output_device().empty()
                                                          ? "(system default)"
                                                          : tts->output_device().c_str());
                        if (use_real) rt::report_vram("  after swap ");
                        // Prove the NEW device actually carries audio, through
                        // the same pipeline speech uses.
                        tts->PlayTestTone();
                        std::this_thread::sleep_for(std::chrono::milliseconds(900));
                        std::printf("  test tone on the new device: %s\n",
                                    tts->chunks_spoken() > 0 ? "published" : "NOT published");
                    }
#endif
                    std::fflush(stdout);
                } catch (...) {
                    std::printf("  FAIL -- the test threw.\n");
                }
                // Close the window so the process exits and the summaries print.
                if (window.hwnd() != nullptr) PostMessageW(window.hwnd(), WM_CLOSE, 0, 0);
            });
        }

        std::printf("running... speak, then pause. Interrupt mid-answer to test barge-in.\n");
        std::fflush(stdout);
        window.run_message_loop();
        if (test_thread.joinable()) test_thread.join();

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

#if defined(VOICE_ASSISTANT_HAS_WHISPER)
        // The cascade's own stage, reported separately for the same reason the
        // speech-output summary below is: it answers a DIFFERENT question. The
        // gate's numbers say whether transcripts became intents; these say
        // whether speech became transcripts at all -- and when the assistant
        // "did not hear" something, this is the block that says which.
        if (cascade_mode) {
            std::printf("\n=== cascade ASR summary ===\n");
            std::printf("  transcripts published  : %llu\n",
                        static_cast<unsigned long long>(cascade_mode->published()));
            if (const rt::WhisperAsr* a = cascade_mode->asr(); a != nullptr) {
                std::printf("  utterances transcribed : %llu\n",
                            static_cast<unsigned long long>(a->utterances()));
                std::printf("  encode failures        : %llu%s\n",
                            static_cast<unsigned long long>(a->failures()),
                            a->failures() > 0 ? "   <-- VRAM or a bad model" : "");
                std::printf("  last encode            : %.1f ms (%s)\n",
                            a->last_encode_ms(), a->gpu() ? "GPU" : "CPU");
            }
            std::printf("  dropped (ASR behind)   : %llu%s\n",
                        static_cast<unsigned long long>(cascade_mode->dropped_overrun()),
                        cascade_mode->dropped_overrun() > 0
                            ? "   <-- the encode is slower than the speaker"
                            : "");
        }
#endif

#if defined(VOICE_ASSISTANT_HAS_TTS)
        // A SEPARATE SUMMARY, because these count a different stage and confusing
        // the two is what makes "it went silent" hard to place. The gate's
        // numbers are about the user's TRANSCRIPT becoming an intent; these are
        // about the ANSWER becoming sound, which happens after and independently.
        //
        // `cancelled` is the one to read first when the assistant answered on
        // screen but said nothing or stopped mid-sentence: every count is a
        // barge-in that killed a chunk.
        //
        // WHAT A NON-ZERO COUNT MEANS. Either somebody interrupted, or the
        // canceller is not removing enough of the assistant's own voice for
        // Silero to stop scoring it as speech. There is no state flag to suspect
        // any more -- the ONLY thing standing between the loudspeaker and a
        // barge-in is the subtraction. So read it next to the [aec] ERLE line:
        // a healthy loopback reference and a converged filter give a quiet
        // microphone, and if the ERLE is low the reference is the thing to
        // check first (wrong endpoint, or a fallback warned about at startup).
        if (tts.has_value()) {
            std::printf("\n=== speech output summary ===\n");
            std::printf("  chunks spoken          : %llu\n",
                        static_cast<unsigned long long>(tts->chunks_spoken()));
            std::printf("  chunks cancelled       : %llu%s\n",
                        static_cast<unsigned long long>(tts->chunks_cancelled()),
                        tts->chunks_cancelled() > 0
                            ? "   <-- barge-in killed these; if nobody interrupted, "
                              "read the ERLE below"
                            : "");
            std::printf("  synthesis errors       : %llu\n",
                        static_cast<unsigned long long>(tts->synthesis_errors()));
        }
        // THE NUMBER THAT EXPLAINS THE ONE ABOVE. ERLE is how many dB of the
        // assistant's own voice the canceller is actually removing; resyncs are
        // how often the reference had to be discarded to catch up with the
        // microphone, which is the clock-drift term loopback does NOT fix.
        // Reference underruns mean the loopback device stopped delivering, which
        // makes the filter subtract silence and cancel nothing.
        if (aec.has_value()) {
            std::printf("  AEC erle               : %.1f dB\n",
                        static_cast<double>(aec->erle_db()));
            std::printf("  AEC resyncs            : %llu (reference discarded to catch up)\n",
                        static_cast<unsigned long long>(aec->resyncs()));
            std::printf("  AEC ref underruns      : %llu%s\n",
                        static_cast<unsigned long long>(aec->reference_underruns()),
                        aec->reference_underruns() > 0
                            ? "   <-- the loopback reference had gaps; cancellation "
                              "was blind for those samples"
                            : "");
        }
#endif

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
