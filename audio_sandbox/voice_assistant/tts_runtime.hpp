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
// THE SELF-TRIGGER PROBLEM, AND WHAT NOW SOLVES IT
// =============================================================================
// The loudspeaker feeds the microphone. Silero scores the assistant's own voice
// as speech -- correctly, because it IS speech -- which fires speech-onset,
// which is barge-in, which cancels the generation currently being spoken. The
// system's own correctness works against it.
//
// This used to be answered by GATING THE MICROPHONE while the speaker was live.
// That worked and it cost the whole feature: during exactly the window in which
// a user would interrupt, the assistant was deaf, so barge-in-while-speaking
// could not happen at all and the cancel button was the only way out. The gate
// is GONE -- config, state and interlock -- and nothing here mutes, zeroes or
// pauses capture for any reason.
//
// What replaced it is acoustic echo cancellation
// (src/audio_rt/echo_canceller.hpp), fed the far-end reference this class
// publishes at aec_reference(). The microphone runs continuously; the
// assistant's own voice is SUBTRACTED from it rather than the microphone being
// switched off. A user talking over the assistant reaches the VAD in the same
// block they would have if nothing were playing.
//
// This class's remaining share of that contract is the REFERENCE, and it has
// exactly one property to protect: it is tapped at PullForPlayback, so it is
// aligned with what the speaker is emitting rather than with when synthesis
// happened (see tts_duplex_bridge.hpp -- synthesis runs several times faster
// than realtime and would put the reference seconds ahead of the microphone).
// The consumer side -- rate conversion, backlog bounding, cancellation -- is
// AecCaptureFilter, on the capture thread, where it belongs.
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

    // Output endpoint: name (empty = system default) and zero-based index
    // (-1 = not selected by index, and the index wins when both are set).
    // Both are forwarded verbatim to AudioPlayback::start, which owns the
    // precedence rule and the fallback behaviour. See audio_playback.h.
    std::string output_device;
    int output_device_index = -1;
    float volume = 1.0f;          // initial playback gain, [0, 1]
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

        // Volume BEFORE the device starts, so the very first buffer is already
        // at the configured level rather than briefly at full scale.
        playback_.set_volume(cfg.volume);

        // Playback LAST: once the device is running its callback is live, and it
        // must never observe a half-built bridge.
        playback_.start(blackwell::tts::kF5SampleRate, &PullThunk, this, cfg.output_device,
                        cfg.output_device_index);

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
    }

    // ---- audio hot swap (settings thread) -----------------------------------
    // Moves speech output to a different endpoint WITHOUT touching the F5
    // session, the arena or anything else on the GPU. The synthesiser, its ONNX
    // graphs and the worker thread all keep running; only the ma_device is
    // closed and reopened underneath the same speaker ring.
    //
    // The AEC reference needs NO rebinding across this, and that is a property
    // of where it is tapped rather than luck: PullForPlayback writes the
    // reference as it hands samples to the device, so it describes "what we are
    // emitting" independently of which endpoint emits it. What the swap DOES
    // invalidate is the canceller's learned impulse response, which belongs to
    // the old speaker -- the caller resets the filter (main.cpp does).
    bool HotReloadOutput(const std::string& device_name, int device_index) {
        return playback_.hot_reload(device_name, device_index);
    }

    // ---- "Check sound" (any thread) -----------------------------------------
    // A short tone straight into the speaker ring, bypassing the synthesiser
    // entirely. THROUGH THE REAL PIPELINE ON PURPOSE: it goes out the configured
    // endpoint, at the configured volume, and is tapped into the AEC reference
    // exactly like speech -- so it tests the path that was actually silent,
    // rather than a second path that might work when the first does not.
    //
    // Short and gentle: 0.35 s of 660 Hz at -18 dBFS with raised-cosine edges,
    // because a hard-edged tone at full scale is what makes test buttons
    // unpleasant to press twice.
    void PlayTestTone() {
        constexpr int kRate = blackwell::tts::kF5SampleRate;
        constexpr std::size_t kSamples = static_cast<std::size_t>(kRate * 0.35);
        constexpr double kFreq = 660.0;
        constexpr std::size_t kFade = kRate / 100;   // 10 ms in and out

        std::vector<float> pcm(kSamples);
        for (std::size_t i = 0; i < kSamples; ++i) {
            const double t = static_cast<double>(i) / kRate;
            double env = 0.125;                       // -18 dBFS
            if (i < kFade) {
                env *= 0.5 * (1.0 - std::cos(3.14159265358979 * static_cast<double>(i) / kFade));
            } else if (i + kFade > kSamples) {
                const std::size_t k = kSamples - i;
                env *= 0.5 * (1.0 - std::cos(3.14159265358979 * static_cast<double>(k) / kFade));
            }
            pcm[i] = static_cast<float>(env * std::sin(6.283185307179586 * kFreq * t));
        }
        // Resume first: a barge-in latch left set from an earlier turn would
        // otherwise swallow the tone, and "the test button does nothing" is the
        // worst possible outcome for a button whose job is to prove sound works.
        Resume();
        bridge_.PushPcm(pcm);
    }

    // The user's turn ended; speak again on the next reply.
    void Resume() {
        cancel_.store(false, std::memory_order_release);
        bridge_.Resume();
    }

    // ---- volume (any thread) -------------------------------------------------
    // Effective on the next audio buffer. THE CALLER MUST ALSO give the same
    // value to AecCaptureFilter::SetReferenceGain(), or the canceller's learned
    // response goes stale by exactly this ratio -- audio_playback.h says why.
    void SetVolume(float v) noexcept { playback_.set_volume(v); }
    float volume() const noexcept { return playback_.volume(); }

    // ---- observers -----------------------------------------------------------
    bool speaking() const noexcept { return bridge_.speaking(); }
    // The output endpoint actually opened ("" = system default), and whether a
    // requested one was silently swapped for it.
    const std::string& output_device() const noexcept { return playback_.device_name(); }
    bool output_device_fallback() const noexcept { return playback_.device_fallback(); }
    std::uint64_t chunks_spoken() const noexcept { return bridge_.chunks_spoken(); }
    std::uint64_t chunks_cancelled() const noexcept { return bridge_.chunks_cancelled(); }
    std::uint64_t synthesis_errors() const noexcept { return bridge_.synthesis_errors(); }
    std::uint64_t speaker_underruns() const noexcept { return bridge_.speaker_underruns(); }
    // The device-side view of the same question; see AudioPlayback::stats().
    AudioPlayback::PlaybackStats playback_stats() const noexcept { return playback_.stats(); }
    void reset_playback_stats() noexcept { playback_.reset_stats(); }

    // THE far-end reference, consumed by AecCaptureFilter on the capture thread.
    // Time-aligned with the speaker because it is tapped at playback, and
    // continuous because PullForPlayback writes every sample it hands the device
    // including silence -- a gap here would shift the whole stream and the
    // canceller would be subtracting the wrong milliseconds.
    //
    // SINGLE CONSUMER, per the ring's SPSC contract: exactly one AecCaptureFilter
    // may read it.
    blackwell::audio_rt::SpscRing<float>& aec_reference() noexcept { return aec_ref_; }

    // The rate that reference is in (F5's output rate). The capture side is at
    // 16 kHz, so somebody has to convert; this is what tells them by how much.
    static constexpr int far_sample_rate() noexcept { return blackwell::tts::kF5SampleRate; }

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
        // NO reference published from here. The canceller's far end is a WASAPI
        // LOOPBACK capture of the render endpoint (main.cpp), which observes what
        // the speaker actually emits rather than what this class handed the
        // device -- post-mix, post-volume, and inclusive of audio this process
        // never produced.
        //
        // The Playback tap it replaces was correct about ALIGNMENT and wrong
        // about CONTENT: it saw the pre-gain signal, so the canceller had to be
        // told the volume separately and re-converged whenever it moved.
        d.aec_tap = blackwell::tts::AecTap::None;
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

    static std::size_t PullThunk(void* user, float* dst, std::size_t frames) {
        return static_cast<TtsRuntime*>(user)->OnPull(dst, frames);
    }

    // THE audio callback. Lock-free and allocation-free, per audio_playback.h.
    // PullForPlayback does the reference tap itself; there is nothing else for
    // this to do, and nothing here may touch the microphone.
    //
    // Its return value -- real frames as opposed to padding -- is now forwarded
    // rather than discarded, because that is the number the starvation counters
    // are built from.
    std::size_t OnPull(float* dst, std::size_t frames) noexcept {
        return bridge_.PullForPlayback(dst, frames);
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
};

}  // namespace rt
