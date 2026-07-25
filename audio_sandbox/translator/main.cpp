// -----------------------------------------------------------------------------
// translator/main.cpp — audio_translator: real-time speech transcription +
// Russian translation, wiring the audio_sandbox capture/DSP/GUI to the live
// speculative_bridge_api speech pipeline (include/bridge/speculative_bridge_api.h).
//
// THREADS (single-thread engine doctrine, CLAUDE.md)
//   miniaudio callback ─push─▶ SampleRing ─pop─▶ RealTimeDSP worker ──┬─▶ Spectrogram
//   (audio thread)                              (DSP/VAD producer)    ├─▶ AudioRecorder
//                                                                     └─▶ speech_pipeline_push_pcm
//                                                                          (internal VAD ->
//                                                                           marshals engine work)
//   Engine thread:  prefill system prompt (freeze KV prefix) -> wait_and_pump loop
//                   (drains the SPSC command ring; runs the fake decode; emits tokens)
//   UI thread:      Direct2D spectrogram + ImGui control panel + live transcript
//
//   The DSP worker is the SINGLE producer of both the audio ring AND the control
//   ring (its internal VAD raises speech-start / silence-timeout). The engine
//   thread is the SOLE consumer. push_pcm / on_* never touch CUDA — they only
//   append to lock-free rings and bump the barge-in epoch.
//
// ENGINE: a GPU-free MockEngineControl drives the REAL EngineControlBridge control
// plane (STEP 5). The FP8 Llama backbone + Ultravox projector assembly is not
// wired here (no backbone weights), so no real audio->logits path runs; the mock
// streams a canned translation so the full UI + interruption path is validated now.
// -----------------------------------------------------------------------------
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <string>
#include <thread>

#include "audio_capture.h"
#include "audio_recorder.h"
#include "realtime_dsp.h"
#include "whisper_dsp.h"
#include "window_d2d.h"

#include "bridge/engine_api.h"
#include "bridge/speculative_bridge_api.h"
#include "bridge_internal.hpp"  // blackwell::bridge::bridge_wrap_engine / bridge_release_engine

#include "mock_engine_control.hpp"
#include "transcript_view.hpp"

namespace {

// The frozen system instruction. Prefilled ONCE at startup; its KV prefix is never
// truncated by a barge-in rewind (the prefix-cache floor, STEP 2).
constexpr const char* kSystemPrompt =
    "You are a real-time speech translator. Transcribe the audio verbatim and "
    "provide its translation into Russian in the format: "
    "[Speech] <transcript> | [Translation] <translation>";

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
    // A clean return to IDLE means the utterance finished — commit its live line.
    if (next == SPEECH_STATE_IDLE) view->on_final(gen_id);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string data_dir = (argc > 1) ? argv[1] : "data";

    std::printf("=== Real-time speech translator (mock engine) ===\n");
    std::printf("Capture source:\n");
    std::printf("  [M] Microphone\n");
    std::printf("  [L] System loopback (what you hear)\n");
    std::printf("> ");
    std::fflush(stdout);

    const int ch = std::getchar();
    const rt::CaptureMode mode =
        (ch == 'l' || ch == 'L') ? rt::CaptureMode::Loopback : rt::CaptureMode::Microphone;
    std::printf("selected: %s\n",
                mode == rt::CaptureMode::Loopback ? "system loopback" : "microphone");

    try {
        // ---- audio front-end (identical geometry to the parity sandbox) --------
        whisper::DspConfig cfg;
        whisper::WhisperDSP dsp(cfg, data_dir + "/mel_filters.bin");
        rt::SpectrogramBuffer spectrogram(cfg.n_mels, /*max_frames=*/1000);
        rt::AudioRecorder recorder(cfg.sample_rate, /*out_dir=*/"recordings");

        rt::AudioCapture capture;
        capture.start(mode);
        std::printf("capture started (backend: %s, 16 kHz mono f32)\n",
                    capture.backend_name().c_str());

        rt::RealTimeDSP realtime(dsp, capture.ring(), spectrogram, &recorder);

        // ---- speech pipeline over the mock engine (real control plane) ---------
        rt::MockEngineControl control;
        EngineHandle engine = blackwell::bridge::bridge_wrap_engine(&control);
        if (engine == nullptr) throw std::runtime_error("bridge_wrap_engine failed");

        AudioStreamHandle stream = nullptr;
        if (BridgeStatus s = engine_create_audio_stream(engine, &stream); s != BRIDGE_OK) {
            blackwell::bridge::bridge_release_engine(engine);
            throw std::runtime_error(std::string("engine_create_audio_stream: ") +
                                     bridge_status_to_string(s));
        }

        SpeechPipelineConfig scfg{};
        scfg.engine = engine;
        scfg.audio_stream = stream;
        scfg.sample_rate = static_cast<uint32_t>(cfg.sample_rate);
        scfg.vad_threshold_db = -40.0f;   // onset
        scfg.vad_release_db = -45.0f;     // hysteresis release (<= onset)
        scfg.silence_hangover_ms = 800;   // stable-boundary hangover -> commit decode
        scfg.warm_prefill_interval_ms = 320;  // speculative warm-prefill throttle

        SpeechPipelineHandle pipe = nullptr;
        if (BridgeStatus s = speech_pipeline_create(&scfg, &pipe); s != BRIDGE_OK) {
            engine_destroy_audio_stream(stream);
            blackwell::bridge::bridge_release_engine(engine);
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
        // Abort any in-flight mock decode within one piece, then wake the pump.
        control.cancel_generation(control.active_generation() + 1);
        control.stop();
        engine_thread.join();

        speech_pipeline_destroy(pipe);
        engine_destroy_audio_stream(stream);
        blackwell::bridge::bridge_release_engine(engine);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
