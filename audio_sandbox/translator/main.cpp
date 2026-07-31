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
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "device_buffer.h"           // blackwell::DeviceBuffer (--wav log-mel upload)

#include "audio_capture.h"
#include "audio_recorder.h"
#include "realtime_dsp.h"
#include "whisper_dsp.h"
#include "window_d2d.h"

#include "blackwell/engine.h"        // BlackwellEngine
#include "blackwell/runtime_config.h" // blackwell::InferenceConfig (streaming opt-in)
#include "blackwell/tokenizer.h"     // blackwell::TokenizerFactory / ITokenizer

#include "bridge/engine_api.h"
#include "bridge/speculative_bridge_api.h"
#include "bridge_internal.hpp"       // bridge_wrap_engine / bridge_release_engine

#include "cli_config.hpp"
#include "control_panel.hpp"
#include "conversational_mode.hpp"   // rt::ConversationalMode (Mode A behind ISpeechMode)
#include "real_engine_control.hpp"
#include "silero_vad.hpp"            // blackwell::vad::SileroVAD (neural VAD scorer)
#include "speech_mode.hpp"           // rt::ISpeechMode
#include "transcript_view.hpp"

namespace {

// The frozen system instruction. Prefilled ONCE at startup; its KV prefix is never
// truncated by a barge-in rewind (the prefix-cache floor). Deliberately
// LANGUAGE- AND TASK-NEUTRAL: the source/target languages, the task selection
// (transcribe / translate) and the exact output tags are all runtime-selectable,
// so their directives live in the per-turn user prefix
// (RealEngineControl::build_user_instruction), never in this frozen prefix. A
// format hardcoded here would contradict the panel the moment a task is switched
// off — and it cannot be re-prefilled without invalidating the frozen floor.
constexpr const char* kSystemPrompt =
    "You are a real-time speech transcriber and translator. Each user turn states "
    "its task and its exact output format: follow them literally, transcribe "
    "verbatim, and output nothing else.";

// The persistent KV budget (tokens). Sized for CONTINUOUS STREAMING: the default
// eviction high water mark is 3000 tokens and a maximal draft adds ~222 (one
// 15 s utterance's soft tokens plus the decode cap), so 2048 could not hold the
// steady state — a redraft would run off the end of the cache between two
// evictions. See ContinuousStreamingConfig::fits_context, which the control
// panel evaluates live and warns about.
//
// Cost: the FP32 KV pool is ~256 KB/token at 8B geometry (32 layers x 8 kv heads
// x 128 head_dim x 4 B x K+V), so 4096 tokens is ~1.05 GB on top of the ~5.3 GB
// AWQ weight footprint.
constexpr size_t kMaxContext = 4096;

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

// ---- neural-VAD scorer trampoline (SpeechVadScoreFn) -------------------------
// Stays HERE, with the app that owns the SileroVAD instance, rather than moving
// into ConversationalMode: the mode takes a SpeechVadScoreFn + user pointer, so
// nothing downstream of this file has to know that Silero is what scores — which
// also keeps ONNXRuntime out of every consumer of conversational_mode.hpp.
//
// Runs on the DSP worker thread, inside push_pcm, once per 10 ms block. feed()
// accumulates to Silero's 512-sample chunk and runs an inference every 32 ms,
// returning the latest probability in between — measured at ~0.1 ms per
// inference in a Debug build, a ~0.3% duty cycle on this thread.
//
// The DSP worker is the SINGLE owner of the SileroVAD (the class's threading
// contract): it is the only thread that ever calls feed(). The UI thread only
// reads the published atomics (last_probability / inference_errors).
//
// NO reset_state() in this path, deliberately. The mic feed is continuous — even
// in push-to-talk, where the ring is gated but VAD blocks keep flowing — so the
// recurrent state never sees a discontinuity to recover from. A reset issued
// from the UI thread would be a data race on the ORT session; if a future path
// does introduce a gap, it must marshal the reset onto this thread.
float vad_score(void* user, const float* block, size_t count) {
    auto* vad = static_cast<blackwell::vad::SileroVAD*>(user);
    return vad->feed(block, count);
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

    // ---- Headless one-shot: audio/log-mel -> text (no GUI / no mic) -------------
    // --wav <file>       : WhisperDSP log-mel -> encoder -> projector -> decode.
    // --features <bin>   : feed a precomputed log-mel [n_mels,3000] (isolates the
    //                      encoder/projector/decode path from the DSP).
    if (!args.wav_path.empty() || !args.features_path.empty()) {
        try {
            constexpr int kConvFrames = 3000;                 // Whisper large-v3-turbo 30 s
            const int nmb = projector.num_mel_bins;
            std::vector<float> mel_in(static_cast<size_t>(nmb) * kConvFrames, 0.0f);
            int real_frames = kConvFrames;   // unpadded log-mel length (for --streaming)

            if (!args.features_path.empty()) {
                std::ifstream f(args.features_path, std::ios::binary | std::ios::ate);
                if (!f) throw std::runtime_error("cannot read --features file: " + args.features_path);
                const std::streamsize nb = f.tellg(); f.seekg(0);
                if (static_cast<size_t>(nb) / sizeof(float) != mel_in.size())
                    throw std::runtime_error("--features size mismatch (expected n_mels*3000 floats)");
                f.read(reinterpret_cast<char*>(mel_in.data()), nb);
                std::printf("[features] loaded %s [%d,%d]\n", args.features_path.c_str(), nmb, kConvFrames);
            } else {
                whisper::DspConfig dcfg;
                dcfg.n_mels = nmb;
                whisper::WhisperDSP dsp(dcfg, args.data_dir + "/mel_filters.bin");

                // WAV -> mono PCM, padded/trimmed to 30 s so the DSP emits 3000 frames.
                std::vector<float> pcm = rt::load_wav_mono16k(args.wav_path);
                const size_t orig_pcm_samples = pcm.size();   // pre-pad: real audio length
                {   // PCM range check: normalized f32 audio must sit in [-1, 1].
                    float pmin = 0.f, pmax = 0.f; double psum = 0.0;
                    for (float s : pcm) { pmin = std::min(pmin, s); pmax = std::max(pmax, s); psum += std::abs(s); }
                    std::printf("[wav] %s: %zu samples (%.2f s), pcm min=%.4f max=%.4f mean|.|=%.5f\n",
                                args.wav_path.c_str(), pcm.size(),
                                static_cast<double>(pcm.size()) / dcfg.sample_rate, pmin, pmax,
                                pcm.empty() ? 0.0 : psum / pcm.size());
                }
                const size_t k30s = static_cast<size_t>(dcfg.sample_rate) * 30;
                pcm.resize(k30s, 0.0f);
                const whisper::LogMel mel = dsp.process(pcm);
                const int nf = std::min(mel.n_frames, kConvFrames);
                // The unpadded mel length --streaming slides over (Whisper hop = 160
                // samples/frame). Clamp to nf so a >30 s clip still fits the buffer.
                real_frames = std::min(nf, static_cast<int>(orig_pcm_samples / 160) + 1);
                for (int m = 0; m < nmb; ++m)
                    std::copy(mel.data.begin() + static_cast<size_t>(m) * mel.n_frames,
                              mel.data.begin() + static_cast<size_t>(m) * mel.n_frames + nf,
                              mel_in.begin() + static_cast<size_t>(m) * kConvFrames);
                {   // log-mel sanity vs the golden input_features.bin.
                    float lmin = mel_in[0], lmax = mel_in[0];
                    for (float v : mel_in) { lmin = std::min(lmin, v); lmax = std::max(lmax, v); }
                    std::printf("[log-mel] shape [%d,%d] (nf=%d) min=%.4f max=%.4f\n",
                                nmb, kConvFrames, mel.n_frames, lmin, lmax);
                    std::ifstream gf(args.data_dir + "/input_features.bin", std::ios::binary | std::ios::ate);
                    if (gf) {
                        const std::streamsize nb = gf.tellg(); gf.seekg(0);
                        std::vector<float> golden(static_cast<size_t>(nb) / sizeof(float));
                        gf.read(reinterpret_cast<char*>(golden.data()), nb);
                        if (golden.size() == mel_in.size()) {
                            double dot = 0, na = 0, nb2 = 0;
                            for (size_t i = 0; i < mel_in.size(); ++i) {
                                dot += (double)mel_in[i] * golden[i];
                                na += (double)mel_in[i] * mel_in[i];
                                nb2 += (double)golden[i] * golden[i];
                            }
                            std::printf("[log-mel] cosine vs golden input_features.bin = %.6f\n",
                                        dot / (std::sqrt(na) * std::sqrt(nb2) + 1e-12));
                        }
                    }
                }
            }

            std::printf("[engine] loading tokenizer + backbone from %s ...\n", args.model_dir.c_str());
            std::fflush(stdout);
            std::unique_ptr<blackwell::ITokenizer> tokenizer =
                blackwell::TokenizerFactory::create(args.model_dir);
            // --streaming arms the sliding-window plan (reconcile or center-slice
            // per --stream-mode) in the engine's resolved RuntimeConfig; default
            // stays whole-utterance.
            blackwell::InferenceConfig req;
            req.max_context_length = kMaxContext;
            rt::apply_streaming_flags(args, req);
            req.source_language = args.src_lang;
            req.target_language = args.tgt_lang;
            BlackwellEngine engine(args.model_dir + "/model.safetensors.index.json", req);
            rt::RealEngineControl control(&engine, tokenizer.get(), static_cast<int>(kMaxContext));
            control.set_context_mode(args.context_mode == "bounded"
                                         ? rt::RealEngineControl::ContextMode::BoundedHistory
                                         : rt::RealEngineControl::ContextMode::Stateless);
            control.set_history_budget_tokens(args.history_budget_tokens);
            std::printf("[audio] loading audio head from %s ...\n", args.audio_head.c_str());
            std::fflush(stdout);
            control.load_audio_head(args.audio_head);
            std::printf("[audio] loaded (%d soft-tokens/frame)\n", control.audio_out_frames());

            blackwell::DeviceBuffer<float> d_mel(mel_in.size());
            if (cudaMemcpy(d_mel.get(), mel_in.data(), mel_in.size() * sizeof(float),
                           cudaMemcpyHostToDevice) != cudaSuccess)
                throw std::runtime_error("cudaMemcpy log-mel H2D failed");

            const uint32_t nsys = control.prefill_system_prompt(kSystemPrompt);
            std::printf("[system-prefix] %u tokens; running audio prefill + decode (%s) ...\n",
                        nsys, args.streaming ? "streaming/reconciled" : "whole-utterance");
            std::fflush(stdout);
            const std::string text = args.streaming
                ? control.transcribe_streaming(d_mel.get(), nmb, real_frames, args.max_new_tokens)
                : control.transcribe(d_mel.get(), args.max_new_tokens);
            std::printf("\n===== AUDIO -> TEXT =====\n%s\n=========================\n", text.c_str());
            return text.empty() ? 3 : 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "FATAL (--wav): %s\n", e.what());
            return 1;
        }
    }

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
        // Tier-2 request: context budget + streaming plan + the forced-language
        // startup defaults (the ImGui dropdowns override them live). INIT tier:
        // throws on OOM/bad index.
        blackwell::InferenceConfig req;
        req.max_context_length = kMaxContext;
        rt::apply_streaming_flags(args, req);
        // The LIVE translator is the streaming product: the plan is ALWAYS armed
        // here (unlike headless --wav, where --streaming stays an opt-in) so the
        // panel's mode toggle is functional and the commit can never silently
        // fall back to the multi-second whole-utterance re-prefill just because
        // a launch flag was missing. --stream-mode still picks the initial mode
        // (center by default); the whole-utterance radio remains the live opt-out.
        req.audio_streaming.enable = true;
        std::printf("[stream] plan armed: mode=%s window=%d ms hop=%d ms edges L=%d/K=%d ms "
                    "(toggle live in Translator Settings)\n",
                    args.stream_mode.c_str(), args.stream_window_ms, args.stream_hop_ms,
                    args.stream_left_edge_ms, args.stream_right_edge_ms);
        req.source_language = args.src_lang;
        req.target_language = args.tgt_lang;
        BlackwellEngine engine(index_path, req);
        std::printf("[engine] loaded (hidden=%d vocab=%d, ~5.3 GB VRAM resident)\n",
                    backbone.hidden_size, backbone.vocab_size);

        // ---- speech pipeline over the REAL engine (real control plane) ---------
        rt::RealEngineControl control(&engine, tokenizer.get(),
                                      static_cast<int>(kMaxContext));
        control.set_context_mode(args.context_mode == "bounded"
                                     ? rt::RealEngineControl::ContextMode::BoundedHistory
                                     : rt::RealEngineControl::ContextMode::Stateless);
        control.set_history_budget_tokens(args.history_budget_tokens);
        std::printf("[context] mode=%s history-budget=%d src=%s tgt=%s\n",
                    args.context_mode.c_str(), args.history_budget_tokens,
                    args.src_lang.c_str(), args.tgt_lang.c_str());

        // ---- STEP 3b: audio head (Whisper encoder + Ultravox projector) --------
        // Load the audio frontend weights from --audio-head so the double-buffered
        // encode->project->prefill path is armed. Non-fatal: a missing/failed audio
        // head just leaves the app in text-only mode (the live mel feed is the one
        // remaining seam, so nothing calls prefill_audio yet regardless).
        std::printf("[audio] loading audio head from %s ...\n", args.audio_head.c_str());
        std::fflush(stdout);
        try {
            control.load_audio_head(args.audio_head);
            control.set_dsp(&dsp);  // engine-thread log-mel for the live commit path
            std::printf("[audio] encoder + projector loaded (%d audio soft-tokens/frame); "
                        "live audio->text ARMED\n", control.audio_out_frames());
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[audio] WARN: audio head not loaded (%s) — text-only mode\n",
                         e.what());
        }
        rt::TranscriptView transcript;

        // ---- STEP 3c: Silero neural VAD (the boundary detector) ---------------
        // Installed as the pipeline's VAD scorer, replacing the RMS threshold as
        // the answer to "is this block speech?" — and ONLY that. The hangover,
        // the warm-prefill throttle, push-to-talk muting and background listening
        // all keep running on top of its verdict.
        //
        // Constructed BEFORE the mode and outliving it: the Silero session is a
        // startup-cost resource that must survive a mode switch rather than be
        // reloaded on every flip (ISpeechMode borrows it).
        //
        // Non-fatal, like the audio head: a missing or unloadable model leaves
        // the built-in threshold detector in place (the panel then says so)
        // rather than taking the app down over a 2.3 MB file.
        std::unique_ptr<blackwell::vad::SileroVAD> neural_vad;
        if (args.neural_vad && !args.vad_model.empty()) {
            try {
                neural_vad = std::make_unique<blackwell::vad::SileroVAD>(args.vad_model);
                neural_vad->set_threshold(args.vad_threshold);
                std::printf("[vad] Silero neural VAD armed (threshold %.2f) from %s\n",
                            args.vad_threshold, args.vad_model.c_str());
            } catch (const std::exception& e) {
                std::fprintf(stderr,
                             "[vad] WARN: neural VAD not loaded (%s) — falling back to the "
                             "RMS threshold detector\n", e.what());
                neural_vad.reset();
            }
        } else {
            std::printf("[vad] neural VAD disabled — RMS threshold detector\n");
        }

        // ---- the active speech mode (ISpeechMode) ------------------------------
        // Mode A. Every config value below was previously spelled out inline at the
        // speech_pipeline_create call site; the mode owns the bridge plumbing and
        // unwinds it in reverse order, exactly as the old shutdown block did.
        rt::ConversationalMode::Config mcfg;
        mcfg.sample_rate = static_cast<uint32_t>(cfg.sample_rate);
        mcfg.vad_probability_threshold = args.vad_threshold;
        mcfg.system_prompt = kSystemPrompt;
        rt::ConversationalMode speech_mode(
            &control,
            [&control](const std::string& p) { return control.prefill_system_prompt(p); },
            neural_vad ? &vad_score : nullptr, neural_vad.get(),
            mcfg, &on_token, &on_state, &transcript);
        rt::ISpeechMode* active = &speech_mode;

        // ---- engine thread: ONE runner, mode-agnostic -------------------------
        // The doctrine holds trivially: this is still the only thread that calls
        // into the engine, whichever mode is active.
        std::atomic<bool> running{true};
        std::atomic<bool> prefilled{false};
        std::thread engine_thread([&] {
            active->start_on_engine_thread();
            std::printf("[system-prefix] frozen %u tokens (KV rewind floor)\n",
                        speech_mode.frozen_prefix_tokens());
            std::fflush(stdout);
            prefilled.store(true, std::memory_order_release);
            while (running.load(std::memory_order_acquire)) {
                active->pump_engine();   // 0% CPU until a boundary event arrives
            }
        });

        // Don't feed audio until the system prefix is frozen (so the first barge-in
        // rewind already has a valid floor).
        while (!prefilled.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        // PCM tap: the DSP worker becomes the active mode's single producer. ONE
        // indirection, no per-callback branching on which mode is live.
        realtime.set_pcm_tap([&active](const float* samples, std::size_t count) {
            active->on_pcm_block(samples, count);
        });
        realtime.start();

        // ---- UI: spectrogram + control panel + live transcript window ----------
        // The settings panel talks ONLY to RealEngineControl atomics and lock-free
        // speech_pipeline_* calls (never the engine) — UI thread stays doctrine-clean.
        // Re-translation tunables (docs/CONTINUOUS_STREAMING.md). The publisher is
        // the UI->workers seam: the panel stores, and the DSP/engine threads will
        // load it once the continuous pipeline is wired (T4). It is live from the
        // first frame so the eviction headroom check is visible now — with
        // kMaxContext at 2048 the default 3000-token high water does NOT fit, and
        // the panel says so rather than letting a redraft run off the cache.
        blackwell::bridge::LiveStreamingConfig streaming_cfg;
        rt::ControlPanel settings(&control, speech_mode.pipeline(),
                                  static_cast<int>(mcfg.silence_hangover_ms),
                                  neural_vad.get(),
                                  neural_vad ? &vad_score : nullptr,
                                  &streaming_cfg, static_cast<int>(kMaxContext),
                                  args.max_new_tokens);
        rt::WindowD2D window(spectrogram, recorder, L"Real-time Speech Translator");
        window.set_extra_panel([&transcript, &settings] {
            transcript.draw();
            settings.draw();
        });
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
        active->stop();
        engine_thread.join();

        // The mode's destructor unwinds pipe -> stream -> engine handle in the
        // same reverse order this block used to spell out.
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
}
