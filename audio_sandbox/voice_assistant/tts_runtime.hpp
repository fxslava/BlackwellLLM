#pragma once
// -----------------------------------------------------------------------------
// TtsRuntime — the whole speech-output stack as one object with one lifetime.
//
// Owns, in construction order (which IS destruction order reversed, and the
// order matters -- see the member block): the F5 engine, its ISynthesizer
// adapter, the tokenizer, the two rings, the duplex bridge, the playback device
// and the TTS worker thread. main.cpp holds ONE optional<TtsRuntime> and calls
// four methods; everything else about speech output is in here.
//
// =============================================================================
// GRACEFUL DEGRADATION IS THE POINT OF THE CTOR CONTRACT
// =============================================================================
// INIT tier: the constructor THROWS if the models, the vocab, or the reference
// clip are missing or malformed. main.cpp catches, reports, and runs without
// speech output -- exactly the posture this app already takes for the neural
// VAD and the audio head. A missing 1.3 GB checkpoint must not stop the
// assistant from being an assistant.
//
// =============================================================================
// THE SELF-TRIGGER PROBLEM, STATED HONESTLY
// =============================================================================
// The loudspeaker feeds the microphone. Silero scores the assistant's own voice
// as speech -- correctly, because it IS speech -- which fires speech-onset,
// which is barge-in, which cancels the generation currently being spoken. The
// system's own correctness works against it.
//
// There are exactly two mitigations and they are mutually exclusive today:
//
//   AEC (real fix)   subtract the far-end reference from the mic. The bridge
//                    already produces a correctly time-aligned reference at
//                    aec_reference(). NOTHING CONSUMES IT YET -- there is no
//                    echo canceller in this repo (docs/TTS_INTEGRATION_AUDIT.md
//                    Phase 4, not started). The tap is wired and correct so that
//                    landing an AEC is a connection, not a redesign.
//
//   Mic gating (V1)  drop mic blocks while the speaker is active plus a reverb
//                    tail. Cheap and effective, and it COSTS BARGE-IN: the user
//                    cannot interrupt by talking over the assistant, only with
//                    the cancel button. That is a real regression against the
//                    Conversational mode's design intent.
//
// mic_gate_enabled defaults to FALSE, so barge-in-while-speaking works as the
// pipeline intends. The consequence you are accepting is that on OPEN SPEAKERS
// the assistant can hear itself and cut itself off. Use headphones, or set the
// gate. There is no third option until AEC lands, and pretending otherwise in a
// comment would be worse than saying it here.
// -----------------------------------------------------------------------------
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "audio_playback.h"

#include "f5_engine_synthesizer.hpp"
#include "f5_mel_extractor.hpp"
#include "f5_tokenizer.hpp"
#include "f5_tts_engine.hpp"
#include "spsc_ring.hpp"
#include "text_chunker.hpp"
#include "tts_duplex_bridge.hpp"

namespace rt {

struct TtsRuntimeConfig {
    std::string ckpt_dir;      // holds f5_tts_dit.onnx + f5_tts_vocoder.onnx
    std::string vocab_path;    // the vocab that shipped WITH the checkpoint
    std::string ref_audio;     // reference clip, 24 kHz mono WAV
    std::string ref_text;      // its transcript -- NOT optional, see below
    int  nfe_step  = 16;
    int  device_id = 0;
    float speed    = 1.0f;

    // Chunking. Defaults match settings_store.hpp; the ranges are enforced in
    // MakeChunkerConfig rather than trusted, because these arrive from a UI.
    bool split_on_commas = true;
    int  min_chunk_chars = 20;    // clamped to [5, 50]
    int  max_chunk_chars = 150;   // clamped to [50, 300]

    bool mic_gate_enabled = true;   // see the header block before disabling
    std::uint32_t mic_gate_tail_ms = 200;   // reverb tail after the last sample
};

class TtsRuntime {
public:
    // INIT tier: throws std::runtime_error. See the header.
    explicit TtsRuntime(const TtsRuntimeConfig& cfg)
        : cfg_(cfg),
          engine_(MakeEngineConfig(cfg), nullptr),
          synth_(engine_),
          tokenizer_(cfg.vocab_path),
          // ~2 s at 24 kHz. Sized from the deadline to survive, not the steady
          // state: this is how long synthesis may stall before the speaker
          // starves, and synthesis runs several times faster than realtime.
          speaker_(48000),
          aec_ref_(48000),
          bridge_(synth_, tokenizer_, speaker_, aec_ref_, MakeChunkerConfig(cfg),
                  MakeDuplexConfig()) {
        LoadReference();

        bridge_.SetCancelSignal(&cancel_);

        // Playback LAST: once the device is running its callback is live, and it
        // must never observe a half-built bridge.
        playback_.start(blackwell::tts::kF5SampleRate, &PullThunk, this);

        worker_ = std::thread([this] { WorkerLoop(); });
    }

    ~TtsRuntime() {
        // Order is load-bearing. Stop producing, then stop consuming, then tear
        // down what both were touching.
        running_.store(false, std::memory_order_release);
        cancel_.store(true, std::memory_order_release);   // unblock a solve in flight
        if (worker_.joinable()) worker_.join();
        playback_.stop();                                  // joins the audio callback
    }

    TtsRuntime(const TtsRuntime&) = delete;
    TtsRuntime& operator=(const TtsRuntime&) = delete;

    // ---- LLM stream (engine thread) -----------------------------------------
    // One streamed token. Chunking, UTF-8 reassembly and latency policy are the
    // chunker's problem.
    void PushToken(std::string_view token) { bridge_.PushToken(token); }

    // The turn is complete: flush whatever is buffered so the tail is spoken.
    void EndOfTurn() { bridge_.EndOfStream(); }

    // ---- barge-in (any thread) ----------------------------------------------
    // The user started speaking (or hit cancel) while we were talking. Stops
    // synthesis mid-solver, drops unspoken text, and marks queued audio stale.
    void BargeIn() {
        cancel_.store(true, std::memory_order_release);
        bridge_.Cancel();
        gate_until_ = std::chrono::steady_clock::now();   // release the mic at once
    }

    // The user's turn ended; speak again on the next reply.
    void Resume() {
        cancel_.store(false, std::memory_order_release);
        bridge_.Resume();
    }

    // ---- mic interlock (DSP worker) -----------------------------------------
    // True while the speaker is active, plus a reverb tail. Only meaningful when
    // mic_gate_enabled -- see the header for what enabling it costs.
    bool MicShouldBeGated() const noexcept {
        if (!cfg_.mic_gate_enabled) return false;
        return std::chrono::steady_clock::now() < gate_until_;
    }

    // ---- observers -----------------------------------------------------------
    bool speaking() const noexcept { return bridge_.speaking(); }
    std::uint64_t chunks_spoken() const noexcept { return bridge_.chunks_spoken(); }
    std::uint64_t chunks_cancelled() const noexcept { return bridge_.chunks_cancelled(); }
    std::uint64_t synthesis_errors() const noexcept { return bridge_.synthesis_errors(); }

    // The far-end reference for a future echo canceller. Correctly time-aligned
    // with the speaker (tapped at playback), and currently UNCONSUMED -- see the
    // header block.
    blackwell::audio_rt::SpscRing<float>& aec_reference() noexcept { return aec_ref_; }

    const std::string& reference_text() const noexcept { return cfg_.ref_text; }

private:
    static blackwell::tts::F5TtsConfig MakeEngineConfig(const TtsRuntimeConfig& c) {
        blackwell::tts::F5TtsConfig e;
        e.dit_model_path     = c.ckpt_dir + "/f5_tts_dit.onnx";
        e.vocoder_model_path = c.ckpt_dir + "/f5_tts_vocoder.onnx";
        e.device_id          = c.device_id;
        e.nfe_step           = c.nfe_step;
        e.speed              = c.speed;
        return e;
    }

    static blackwell::tts::ChunkerConfig MakeChunkerConfig(const TtsRuntimeConfig& c) {
        // Sentence splits are never disabled: a reply with no sentence breaks at
        // all would buffer until end-of-turn, which is precisely the latency
        // chunking exists to remove.
        blackwell::tts::ChunkerConfig k;
        k.split_on_sentence_ends = true;
        k.split_on_commas = c.split_on_commas;

        // CLAMPED, not trusted. These come from a settings file a user can hand
        // edit, and out-of-range values fail in ways that look like model bugs:
        // a tiny minimum fragments every reply into clipped barks, and a maximum
        // below the minimum would make the chunker split on every codepoint.
        k.min_chunk_chars = static_cast<std::size_t>(std::clamp(c.min_chunk_chars, 5, 50));
        k.max_chunk_chars = static_cast<std::size_t>(std::clamp(c.max_chunk_chars, 50, 300));
        return k;
    }

    static blackwell::tts::DuplexConfig MakeDuplexConfig() {
        blackwell::tts::DuplexConfig d;
        // Playback tap: the reference must be aligned to what the SPEAKER emits,
        // not to when synthesis happened -- synthesis runs several times faster
        // than realtime and would put the reference seconds ahead of the mic.
        d.aec_tap = blackwell::tts::AecTap::Playback;
        return d;
    }

    // Reference clip -> mel -> engine, plus the byte geometry F5's duration
    // formula needs. Done once at startup; a voice change re-runs it.
    void LoadReference() {
        if (cfg_.ref_text.empty()) {
            throw std::runtime_error(
                "TtsRuntime: --tts-ref-text is required. F5 conditions on an "
                "(audio, TEXT) pair and treats generation as infilling; a missing or "
                "wrong transcript does not degrade gracefully, it produces confident "
                "nonsense.");
        }
        std::vector<float> pcm = LoadWav24kMono(cfg_.ref_audio);

        // F5 normalises the reference to RMS 0.1 before taking its mel. The mel
        // is a LOG magnitude, so skipping this shifts every conditioning value
        // by ln(gain) -- far outside anything the model saw in training.
        double sumsq = 0.0;
        for (const float v : pcm) sumsq += static_cast<double>(v) * v;
        const double rms = std::sqrt(sumsq / static_cast<double>(pcm.size()));
        constexpr double kTargetRms = 0.1;
        if (rms > 0.0 && rms < kTargetRms) {
            const double g = kTargetRms / rms;
            for (float& v : pcm) v = static_cast<float>(v * g);
        }

        blackwell::tts::F5MelExtractor mel_ex{blackwell::tts::F5MelConfig{}};
        const std::vector<float> mel = mel_ex.Compute(pcm);
        const std::size_t frames = mel_ex.FrameCount(pcm.size());

        std::size_t unknown = 0;
        const std::vector<std::int32_t> ids32 = tokenizer_.Tokenize(cfg_.ref_text, &unknown);
        if (unknown != 0) {
            throw std::runtime_error(
                "TtsRuntime: " + std::to_string(unknown) + " character(s) of the reference "
                "transcript are not in this voice's alphabet. The vocab and the checkpoint "
                "must come from the same release.");
        }
        const std::vector<std::int64_t> ids(ids32.begin(), ids32.end());

        const blackwell::tts::TtsStatus st = engine_.SetReferenceMel(mel.data(), frames, ids);
        if (st != blackwell::tts::TtsStatus::Success) {
            throw std::runtime_error(std::string("TtsRuntime: SetReferenceMel failed: ") +
                                     blackwell::tts::to_string(st));
        }

        // F5 appends a space to ref_text when it ends in a 1-byte character, and
        // measures BOTH texts in UTF-8 bytes. The adapter reproduces the formula;
        // it just needs the same string F5 would have measured.
        std::string ref_measured = cfg_.ref_text;
        if (!ref_measured.empty() &&
            static_cast<unsigned char>(ref_measured.back()) < 0x80u) {
            ref_measured += ' ';
        }
        synth_.SetReferenceGeometry(pcm.size(), ref_measured, cfg_.speed);
    }

    // Minimal RIFF reader: 16-bit PCM and 32-bit float, mono or downmixed.
    // Deliberately not dr_wav -- pulling a third-party amalgamation into this
    // header's TU would drag its /W4 exemptions along with it, for a file format
    // whose header is forty lines.
    static std::vector<float> LoadWav24kMono(const std::string& path);

    void WorkerLoop() {
        while (running_.load(std::memory_order_acquire)) {
            const blackwell::tts::TtsStatus st = bridge_.PumpOnce();
            if (st == blackwell::tts::TtsStatus::Success) continue;  // more may be queued
            // Nothing to do (or a fault already counted). Park briefly rather
            // than spin: this thread is idle for most of a session, and a hot
            // loop here would steal a core from the decode it must not disturb.
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    static void PullThunk(void* user, float* dst, std::size_t frames) {
        static_cast<TtsRuntime*>(user)->OnPull(dst, frames);
    }

    // THE audio callback. Lock-free and allocation-free, per audio_playback.h.
    void OnPull(float* dst, std::size_t frames) noexcept {
        const std::size_t real = bridge_.PullForPlayback(dst, frames);
        if (real != 0) {
            // Hold the mic gate open while sound is actually going out, plus a
            // tail for the room. steady_clock::now() on the audio thread is a
            // vDSO/QPC read -- no lock, no syscall.
            gate_until_ = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(cfg_.mic_gate_tail_ms);
        }
    }

    TtsRuntimeConfig cfg_;

    // DECLARATION ORDER IS DEPENDENCY ORDER (CLAUDE.md extension pattern #3).
    //   engine_    -> synth_ holds a reference into it
    //   tokenizer_ -> bridge_ holds a reference into it
    //   rings      -> bridge_ holds references into them
    //   bridge_    -> the worker and the audio callback drive it
    // Reordering any pair turns teardown into a use-after-free that only
    // reproduces on the destruction path.
    blackwell::tts::F5TtsEngine          engine_;
    blackwell::tts::F5EngineSynthesizer  synth_;
    blackwell::tts::F5Tokenizer          tokenizer_;
    blackwell::audio_rt::SpscRing<float> speaker_;
    blackwell::audio_rt::SpscRing<float> aec_ref_;
    blackwell::tts::TTSDuplexBridge      bridge_;

    AudioPlayback     playback_;
    std::thread       worker_;
    std::atomic<bool> running_{true};
    std::atomic<bool> cancel_{false};

    // Written by the audio callback, read by the DSP worker. Not atomic because
    // a time_point is not lock-free everywhere; a torn read costs one 10 ms
    // block of gating either way, which is below the tail it is guarding.
    std::chrono::steady_clock::time_point gate_until_{};
};

}  // namespace rt
