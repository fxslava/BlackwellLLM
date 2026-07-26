// -----------------------------------------------------------------------------
// translator/main.cpp — audio_translator: real-time speech transcription +
// Russian translation, wiring the audio_sandbox capture/DSP/GUI to the live
// speech pipeline (include/bridge/speculative_bridge_api.h) over a REAL
// BlackwellEngine (8B AWQ backbone).
//
// THREADS (single-thread engine doctrine, CLAUDE.md)
//   miniaudio callback ─push─▶ SampleRing ─pop─▶ RealTimeDSP worker ──┬─▶ Spectrogram
//   (audio thread)                              (DSP/VAD producer)    ├─▶ AudioRecorder
//                                                                     └─▶ speech_pipeline_push_pcm
//                                                                          (internal VAD ->
//                                                                           marshals engine work)
//   Engine thread:  prefill system prompt (freeze KV prefix) -> wait_and_pump loop
//                   (drains the SPSC command ring; runs the REAL decode; emits tokens)
//   UI thread:      Direct2D spectrogram + ImGui control panel + live transcript
//
//   The DSP worker is the SINGLE producer of both the audio ring AND the control
//   ring. The engine thread is the SOLE consumer and the ONLY thread that touches
//   CUDA. push_pcm / on_* never touch the engine — they only append to lock-free
//   rings and bump the barge-in epoch.
//
// STARTUP: CLI + config validation BEFORE any CUDA. --model-dir names the HF
// checkpoint; its config.json is parsed for hidden_size/quant_method/vocab_size/
// rope_scaling and the Ultravox projector's output width is validated against the
// backbone hidden_size (validate_dimensions) so a mismatch aborts with a clear
// message instead of crashing inside a kernel. Only then is the ~5.3 GB engine
// constructed. See real_engine_control.hpp for the audio-encoder seam.
// -----------------------------------------------------------------------------
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <string>
#include <thread>

#include "audio_capture.h"
#include "audio_recorder.h"
#include "realtime_dsp.h"
#include "whisper_dsp.h"
#include "window_d2d.h"

#include "blackwell/engine.h"        // BlackwellEngine
#include "blackwell/tokenizer.h"     // blackwell::TokenizerFactory / ITokenizer

#include "bridge/engine_api.h"
#include "bridge/speculative_bridge_api.h"
#include "bridge_internal.hpp"       // bridge_wrap_engine / bridge_release_engine

#include "cli_config.hpp"
#include "real_engine_control.hpp"
#include "transcript_view.hpp"

namespace {

// The frozen system instruction. Prefilled ONCE at startup; its KV prefix is never
// truncated by a barge-in rewind (the prefix-cache floor).
constexpr const char* kSystemPrompt =
    "You are a real-time speech translator. Transcribe the audio verbatim and "
    "provide its translation into Russian in the format: "
    "[Speech] <transcript> | [Translation] <translation>";

// The persistent KV budget (tokens). Utterances are short; a small context keeps
// VRAM near the ~5.3 GB AWQ weight footprint and decode latency low.
constexpr size_t kMaxContext = 2048;

// ---- speech-pipeline callbacks (fire on the engine / VAD threads) -------------
// They only hand results off to the thread-safe TranscriptView (never touch ImGui,
// COM, CUDA, or re-enter the pipeline API), per the callback contract.
void on_token(void* user, const SpeechTokenEvent* event, uint64_t gen_id) {
    auto* view = static_cast<rt::TranscriptView*>(user);
    if (view != nullptr && event != nullptr) view->on_token(event->text, gen_id);
}

void on_state(void* user, SpeechPipelineState /*prev*/, SpeechPipelineState next,
              uint64_t gen_id) {
    auto* view = static_cast<rt::TranscriptView*>(user);
    if (view == nullptr) return;
    view->set_state(next);
    if (next == SPEECH_STATE_IDLE) view->on_final(gen_id);
}

}  // namespace

int main(int argc, char** argv) {
    // ---- STEP 1-2: CLI + config parse + STRICT validation (no CUDA yet) --------
    rt::TranslatorArgs args;
    rt::BackboneConfig backbone;
    rt::ProjectorParams projector;
    try {
        args = rt::parse_cli(argc, argv);
        if (!args.have_model_dir) {
            std::fprintf(stderr,
                "FATAL: no model directory. Pass --model-dir <path> (e.g. "
                "--model-dir F:/AI/llama-3.1-8B-Instruct-AWQ-INT4) or provide a "
                "config.json with a \"model_dir\" key.\n");
            return 2;
        }
        backbone  = rt::parse_backbone_config(args.model_dir);
        projector = rt::resolve_projector_params(args.projector_path);
        // Abort BEFORE allocating CUDA / constructing the engine on a mismatch.
        rt::validate_dimensions(projector, backbone, args.model_dir);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 2;
    }

    std::printf("=== Real-time speech translator ===\n");
    std::printf("  model-dir      : %s\n", args.model_dir.c_str());
    std::printf("  audio-head     : %s\n", args.audio_head.c_str());
    std::printf("  backbone       : hidden_size=%d vocab_size=%d quant=%s rope=%s\n",
                backbone.hidden_size, backbone.vocab_size, backbone.quant_method.c_str(),
                backbone.has_rope_scaling ? backbone.rope_type.c_str() : "none");
    std::printf("  projector      : output_dim=%d num_mel_bins=%d stack_factor=%d\n",
                projector.output_dim, projector.num_mel_bins, projector.stack_factor);
    std::printf("  [validated] projector.output_dim == backbone.hidden_size (%d)\n\n",
                backbone.hidden_size);

    std::printf("Capture source:\n  [M] Microphone\n  [L] System loopback (what you hear)\n> ");
    std::fflush(stdout);
    const int ch = std::getchar();
    const rt::CaptureMode mode =
        (ch == 'l' || ch == 'L') ? rt::CaptureMode::Loopback : rt::CaptureMode::Microphone;
    std::printf("selected: %s\n",
                mode == rt::CaptureMode::Loopback ? "system loopback" : "microphone");

    try {
        // ---- audio front-end (identical geometry to the parity sandbox) --------
        whisper::DspConfig cfg;
        cfg.n_mels = projector.num_mel_bins;   // 128 (Whisper large-v3-turbo)
        whisper::WhisperDSP dsp(cfg, args.data_dir + "/mel_filters.bin");
        rt::SpectrogramBuffer spectrogram(cfg.n_mels, /*max_frames=*/1000);
        rt::AudioRecorder recorder(cfg.sample_rate, /*out_dir=*/"recordings");

        rt::AudioCapture capture;
        capture.start(mode);
        std::printf("capture started (backend: %s, 16 kHz mono f32)\n",
                    capture.backend_name().c_str());

        rt::RealTimeDSP realtime(dsp, capture.ring(), spectrogram, &recorder);

        // ---- STEP 3: REAL engine + tokenizer (the ~5.3 GB AWQ load) ------------
        std::printf("[engine] loading tokenizer + 8B AWQ backbone from %s ...\n",
                    args.model_dir.c_str());
        std::fflush(stdout);
        std::unique_ptr<blackwell::ITokenizer> tokenizer =
            blackwell::TokenizerFactory::create(args.model_dir);
        const std::string index_path = args.model_dir + "/model.safetensors.index.json";
        BlackwellEngine engine(index_path, kMaxContext);   // INIT tier: throws on OOM/bad index
        std::printf("[engine] loaded (hidden=%d vocab=%d, ~5.3 GB VRAM resident)\n",
                    backbone.hidden_size, backbone.vocab_size);

        // ---- speech pipeline over the REAL engine (real control plane) ---------
        rt::RealEngineControl control(&engine, tokenizer.get(),
                                      static_cast<int>(kMaxContext));

        // ---- STEP 3b: audio head (Whisper encoder + Ultravox projector) --------
        // Load the audio frontend weights from --audio-head so the double-buffered
        // encode->project->prefill path is armed. Non-fatal: a missing/failed audio
        // head just leaves the app in text-only mode (the live mel feed is the one
        // remaining seam, so nothing calls prefill_audio yet regardless).
        std::printf("[audio] loading audio head from %s ...\n", args.audio_head.c_str());
        std::fflush(stdout);
        try {
            control.load_audio_head(args.audio_head);
            std::printf("[audio] encoder + projector loaded (%d audio soft-tokens/frame)\n",
                        control.audio_out_frames());
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[audio] WARN: audio head not loaded (%s) — text-only mode\n",
                         e.what());
        }
        EngineHandle eng_handle = blackwell::bridge::bridge_wrap_engine(&control);
        if (eng_handle == nullptr) throw std::runtime_error("bridge_wrap_engine failed");

        AudioStreamHandle stream = nullptr;
        if (BridgeStatus s = engine_create_audio_stream(eng_handle, &stream); s != BRIDGE_OK) {
            blackwell::bridge::bridge_release_engine(eng_handle);
            throw std::runtime_error(std::string("engine_create_audio_stream: ") +
                                     bridge_status_to_string(s));
        }

        SpeechPipelineConfig scfg{};
        scfg.engine = eng_handle;
        scfg.audio_stream = stream;
        scfg.sample_rate = static_cast<uint32_t>(cfg.sample_rate);
        scfg.vad_threshold_db = -40.0f;   // onset
        scfg.vad_release_db = -45.0f;     // hysteresis release (<= onset)
        scfg.silence_hangover_ms = 800;   // stable-boundary hangover -> commit decode
        scfg.warm_prefill_interval_ms = 320;  // speculative warm-prefill throttle

        SpeechPipelineHandle pipe = nullptr;
        if (BridgeStatus s = speech_pipeline_create(&scfg, &pipe); s != BRIDGE_OK) {
            engine_destroy_audio_stream(stream);
            blackwell::bridge::bridge_release_engine(eng_handle);
            throw std::runtime_error(std::string("speech_pipeline_create: ") +
                                     bridge_status_to_string(s));
        }

        rt::TranscriptView transcript;
        speech_pipeline_register_callbacks(pipe, &on_token, &on_state, &transcript);

        // ---- engine thread: prefill the frozen prefix, then drain the ring -----
        std::atomic<bool> running{true};
        std::atomic<bool> prefilled{false};
        std::thread engine_thread([&] {
            const uint32_t n = control.prefill_system_prompt(kSystemPrompt);
            std::printf("[system-prefix] frozen %u tokens (KV rewind floor)\n", n);
            std::fflush(stdout);
            prefilled.store(true, std::memory_order_release);
            while (running.load(std::memory_order_acquire)) {
                control.wait_and_pump();  // 0% CPU until a boundary event arrives
            }
        });

        // Don't feed audio until the system prefix is frozen (so the first barge-in
        // rewind already has a valid floor).
        while (!prefilled.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        // PCM tap: the DSP worker becomes the pipeline's single producer.
        realtime.set_pcm_tap([pipe](const float* samples, std::size_t count) {
            (void)speech_pipeline_push_pcm(pipe, samples, count);
        });
        realtime.start();

        // ---- UI: spectrogram + control panel + live transcript window ----------
        rt::WindowD2D window(spectrogram, recorder, L"Real-time Speech Translator");
        window.set_extra_panel([&transcript] { transcript.draw(); });
        if (!window.create(/*client_w=*/1000, /*client_h=*/640)) {
            throw std::runtime_error("failed to create Direct2D window");
        }
        std::printf("running... speak into the source; close the window to quit.\n");
        std::fflush(stdout);
        window.run_message_loop();

        // ---- shutdown (reverse dependency order) -------------------------------
        realtime.stop();   // join the DSP worker: no more push_pcm after this
        capture.stop();

        running.store(false, std::memory_order_release);
        // Abort any in-flight decode, then wake the pump so the thread exits.
        control.cancel_generation(control.active_generation() + 1);
        control.stop();
        engine_thread.join();

        speech_pipeline_destroy(pipe);
        engine_destroy_audio_stream(stream);
        blackwell::bridge::bridge_release_engine(eng_handle);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
}
