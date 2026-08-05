// -----------------------------------------------------------------------------
// Aec3EchoCanceller implementation — see the header for what AEC3 buys and for
// the two "switches" in the spec that are not switches.
//
// THE ONLY TU IN THIS REPO THAT SEES WEBRTC HEADERS. Everything below stays
// behind the PIMPL so no consumer inherits WebRTC's include path, its abseil
// dependency or its compile definitions -- the same containment silero_vad.cpp
// applies to ONNXRuntime.
// -----------------------------------------------------------------------------
#include "aec3_echo_canceller.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

// WebRTC's own headers want these on MSVC before anything else pulls <windows.h>
// in transitively. The vcpkg package carries WEBRTC_WIN itself, but NOMINMAX is
// ours to set: several AEC3 headers use std::min/std::max unqualified in
// templates, and the Win32 macros turn those into syntax errors.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

// =============================================================================
// WHY THIS DRIVES AudioProcessing AND NOT EchoControl DIRECTLY
// =============================================================================
// AEC3's own interface (EchoControl::AnalyzeRender/ProcessCapture) takes
// webrtc::AudioBuffer, and AudioBuffer is not usable from a packaged WebRTC:
// modules/audio_processing/audio_buffer.h is installed, but the
// common_audio/channel_buffer.h it includes on its second line is NOT -- the
// vcpkg port ships a curated subset of the tree's ~30k headers and common_audio
// is not in it. That is a property of every prebuilt WebRTC, not of this port:
// AudioBuffer is an internal type, and internal types travel with the source
// tree or not at all.
//
// AudioProcessing IS the supported public entry point, it is entirely present
// in the installed set, and it takes plain deinterleaved float* -- so it needs
// no internal type at all. It builds the AudioBuffers on our behalf.
//
// The cost is one wrapper object around AEC3. It buys back more than it costs:
// every submodule in AudioProcessing::Config defaults to enabled = false, so
// switching them off is the DEFAULT rather than a list we maintain, and the one
// we turn on is echo_canceller. No gain control, no noise suppression, no
// transient suppression -- the near-end signal reaching the VAD is the
// microphone minus the echo, and nothing else has touched it.
// =============================================================================
#include "api/audio/audio_processing.h"
#include "api/audio/builtin_audio_processing_builder.h"
#include "api/audio/echo_canceller3_config.h"
// The Environment is a REQUIRED constructor argument as of the WebRTC generation
// this links against: Build() takes one and there is no overload without it. It
// is the injection point for the clock, the task-queue factory, the field trials
// and the event log -- CreateEnvironment() supplies production defaults for all
// four, which is exactly what this wants. Copyable, refcounted, thread-safe.
#include "api/environment/environment.h"
#include "api/environment/environment_factory.h"
#include "api/scoped_refptr.h"

namespace blackwell::audio_rt {
namespace {

// AEC3's frame is 10 ms, always, at every supported rate.
constexpr int kFrameMs = 10;

bool RateIsSupported(int hz) noexcept {
    return hz == 16000 || hz == 32000 || hz == 48000;
}

// The tuning. Everything not touched here keeps WebRTC's default, which is the
// right posture for a canceller whose defaults are the product of far more
// measurement than this repo can do.
webrtc::EchoCanceller3Config MakeConfig(const Aec3Config& cfg) {
    webrtc::EchoCanceller3Config c;

    // THE DELAY HINT, not a delay setting -- AEC3 re-estimates and will override
    // it. Expressed in blocks of 4 ms (AEC3's internal block), which is what
    // `default_delay` counts.
    if (cfg.initial_delay_ms > 0) {
        c.delay.default_delay =
            static_cast<size_t>(std::max(1, cfg.initial_delay_ms / 4));
    }

    // THE ONE PLACE WE DEVIATE FROM DEFAULTS, and the reason is this app's
    // consumer rather than a human listener.
    //
    // AEC3's defaults are tuned so a PERSON on the far end of a call hears a
    // natural near-end talker: they deliberately leave some residual rather than
    // over-suppress, because over-suppression sounds pumped and clips syllables.
    // Our consumer is a neural VAD, which does not care about naturalness and
    // does care very much about residual speech -- every dB left in is a dB that
    // can score as speech and cancel the answer being spoken.
    //
    // So the normal-path suppressor is asked to work harder than it would for a
    // phone call. It is a deliberate trade of near-end fidelity for a quieter
    // residual, and it is the knob to back off first if the assistant starts
    // missing quiet interruptions.
    c.suppressor.normal_tuning.mask_lf.enr_transparent = 0.2f;
    c.suppressor.normal_tuning.mask_lf.enr_suppress = 0.3f;
    c.suppressor.normal_tuning.mask_hf.enr_transparent = 0.06f;
    c.suppressor.normal_tuning.mask_hf.enr_suppress = 0.1f;

    return c;
}

// The APM configuration. Stated positively and in full rather than left to the
// defaults, because "which submodules are running" is the single thing about
// this object a reader needs to be able to check at a glance.
webrtc::AudioProcessing::Config MakeApmConfig() {
    webrtc::AudioProcessing::Config c;
    c.echo_canceller.enabled = true;
    // The one submodule AEC3 asks for on its own behalf: it wants the DC/rumble
    // energy out of both paths before it correlates them. Left at its default
    // true, spelled out because it is the only other thing switched on here.
    c.echo_canceller.enforce_high_pass_filtering = true;
    // Everything below already defaults to false. Restated because the value of
    // this backend to the pipeline is precisely that nothing except the echo is
    // removed -- a noise suppressor or an AGC in this path would be editing the
    // signal the VAD scores and the ASR transcribes, and neither was asked for.
    c.noise_suppression.enabled = false;
    c.gain_controller1.enabled = false;
    c.gain_controller2.enabled = false;
    c.transient_suppression.enabled = false;
    return c;
}

// Builds an APM carrying AEC3 with our tuning. Returns null on failure; the
// callers decide what that means (throw at construction, keep the old one on
// reset).
webrtc::scoped_refptr<webrtc::AudioProcessing> BuildApm(
    const webrtc::Environment& env, const Aec3Config& cfg) {
    // SetEchoCancellerConfig, NOT SetEchoControlFactory: the builder documents
    // that an injected factory makes the config a no-op, and the config IS the
    // point -- the suppressor tuning above is the whole deviation from stock.
    webrtc::BuiltinAudioProcessingBuilder builder;
    builder.SetConfig(MakeApmConfig())
        .SetEchoCancellerConfig(MakeConfig(cfg), std::nullopt);
    return builder.Build(env);
}

}  // namespace

struct Aec3EchoCanceller::Impl {
    explicit Impl(const Aec3Config& c)
        : cfg(c),
          frame(static_cast<std::size_t>(c.sample_rate_hz) *
                static_cast<std::size_t>(kFrameMs) / 1000u),
          stream_cfg(c.sample_rate_hz, static_cast<size_t>(c.num_capture_channels)),
          env(webrtc::CreateEnvironment()) {
        apm = BuildApm(env, c);
        if (apm == nullptr) {
            throw std::runtime_error("AEC3: BuiltinAudioProcessingBuilder::Build returned null");
        }
        // The delay HINT. APM forwards it to AEC3 as the audio-buffer delay,
        // which AEC3 then re-estimates and overrides -- see the header.
        if (c.initial_delay_ms > 0) apm->set_stream_delay_ms(c.initial_delay_ms);

        near_fifo.assign(frame, 0.0f);
        far_fifo.assign(frame, 0.0f);
        // The output delay line. Process() must hand back `count` samples for
        // `count` in, so the first frame's worth is silence and everything after
        // is one frame late -- that IS latency_samples().
        out_fifo.assign(frame, 0.0f);
        scratch.assign(frame, 0.0f);
    }

    // Runs AEC3 over one complete frame held in near_fifo/far_fifo, leaving the
    // cancelled result in scratch.
    void ProcessFrame() noexcept {
        // Deinterleaved-channel arrays; mono is one entry each. ProcessStream
        // writes in place when src and dest name the same buffer, but the near
        // end is kept intact and the result lands in scratch so a failed call
        // leaves something defined behind.
        const float* const render_in[1] = {far_fifo.data()};
        float* const render_out[1] = {far_fifo.data()};
        const float* const capture_in[1] = {near_fifo.data()};
        float* const capture_out[1] = {scratch.data()};

        // ORDER IS THE CONTRACT. The render frame must be submitted before the
        // capture frame it is meant to cancel from, or AEC3 aligns against a
        // reference it has not seen yet and the first frames cancel nothing.
        apm->ProcessReverseStream(render_in, stream_cfg, stream_cfg, render_out);
        const int err =
            apm->ProcessStream(capture_in, stream_cfg, stream_cfg, capture_out);
        if (err != webrtc::AudioProcessing::kNoError) {
            // Pass the near end through unchanged rather than emitting whatever
            // scratch happens to hold. Uncancelled audio is a degradation; a
            // stale or partial frame is a glitch the VAD would score.
            std::memcpy(scratch.data(), near_fifo.data(), frame * sizeof(float));
            ++process_errors;
        }

        ++frames_processed;
        // GetStatistics() is not free, and its own docs say the aggregation
        // window becomes one second after the first call -- so sampling it
        // faster than that would return the same numbers at a cost.
        // 100 frames = 1 s.
        if (frames_processed % 100 == 0) {
            const webrtc::AudioProcessingStats st = apm->GetStatistics();
            // Both are optional and stay empty until AEC3 has converged enough
            // to have an opinion; the previous value is kept when they do.
            if (st.echo_return_loss_enhancement.has_value()) {
                erle_db = static_cast<float>(*st.echo_return_loss_enhancement);
            }
            if (st.delay_ms.has_value()) {
                delay_ms = static_cast<int>(*st.delay_ms);
            }
        }
    }

    Aec3Config cfg;
    std::size_t frame = 160;
    webrtc::StreamConfig stream_cfg{16000, 1};
    // DECLARED BEFORE `apm`, deliberately: the APM holds a copy of the
    // Environment's refcounted storage, so ours must outlive it. Declaration
    // order is destruction order in reverse -- this is the same rule
    // BlackwellEngine::Impl documents at each of its member sites.
    webrtc::Environment env;
    webrtc::scoped_refptr<webrtc::AudioProcessing> apm;

    std::vector<float> near_fifo, far_fifo, out_fifo, scratch;
    std::size_t fill = 0;                 // samples staged toward the next frame
    std::uint64_t frames_processed = 0;
    // Frames APM refused. Nonzero means the near end is passing through
    // UNCANCELLED, which otherwise looks exactly like a filter that will not
    // converge -- worth being able to tell apart.
    std::uint64_t process_errors = 0;

    float erle_db = 0.0f;
    int delay_ms = -1;
};

Aec3EchoCanceller::Aec3EchoCanceller(const Aec3Config& cfg) {
    if (!RateIsSupported(cfg.sample_rate_hz)) {
        throw std::invalid_argument(
            "AEC3 supports 16000, 32000 or 48000 Hz; got " +
            std::to_string(cfg.sample_rate_hz));
    }
    if (cfg.num_render_channels != 1 || cfg.num_capture_channels != 1) {
        // Deliberately narrow: the capture path here is mono end to end, and a
        // multichannel implementation that is never exercised is a liability.
        throw std::invalid_argument("AEC3 backend: only mono is wired up");
    }
    impl_ = std::make_unique<Impl>(cfg);
}

Aec3EchoCanceller::~Aec3EchoCanceller() = default;

void Aec3EchoCanceller::Process(const float* near_end, const float* far_end,
                                float* out, std::size_t count) noexcept {
    if (near_end == nullptr || far_end == nullptr || out == nullptr || count == 0) {
        return;
    }
    Impl& s = *impl_;

    // `out` may alias `near_end` (the seam allows it), so every sample is read
    // out of the input before anything is written back.
    for (std::size_t i = 0; i < count; ++i) {
        const float near_sample = near_end[i];
        const float far_sample = far_end[i];

        // Emit the oldest delayed sample first, THEN stage the new one into its
        // slot -- that is the one-frame delay line, done in place.
        const float emitted = s.out_fifo[s.fill];
        s.near_fifo[s.fill] = near_sample;
        s.far_fifo[s.fill] = far_sample;
        out[i] = emitted;

        if (++s.fill == s.frame) {
            s.fill = 0;
            s.ProcessFrame();
            // The cancelled frame becomes what the NEXT frame's worth of calls
            // emits. Nothing here allocates: both vectors are frame-sized and
            // were sized in the constructor.
            std::memcpy(s.out_fifo.data(), s.scratch.data(), s.frame * sizeof(float));
        }
    }
}

void Aec3EchoCanceller::Reset() noexcept {
    Impl& s = *impl_;
    // AEC3 has no reset entry point: the supported way to forget a room is to
    // build a new canceller. Everything else (the FIFOs, the metrics) is ours.
    //
    // noexcept, so a failure here leaves the OLD canceller in place rather than
    // taking the process down -- a stale echo path is bad, no echo path at all
    // is worse, and this runs on a device swap where the app must keep going.
    try {
        webrtc::scoped_refptr<webrtc::AudioProcessing> fresh = BuildApm(s.env, s.cfg);
        if (fresh != nullptr) {
            s.apm = std::move(fresh);
            if (s.cfg.initial_delay_ms > 0) {
                s.apm->set_stream_delay_ms(s.cfg.initial_delay_ms);
            }
        }
    } catch (...) {
        // Keep the existing canceller; it describes the wrong room but it works.
    }

    std::fill(s.near_fifo.begin(), s.near_fifo.end(), 0.0f);
    std::fill(s.far_fifo.begin(), s.far_fifo.end(), 0.0f);
    std::fill(s.out_fifo.begin(), s.out_fifo.end(), 0.0f);
    s.fill = 0;
    s.frames_processed = 0;
    s.process_errors = 0;
    s.erle_db = 0.0f;
    s.delay_ms = -1;
}

float Aec3EchoCanceller::erle_db() const noexcept { return impl_->erle_db; }

std::size_t Aec3EchoCanceller::latency_samples() const noexcept {
    return impl_->frame;
}

int Aec3EchoCanceller::estimated_delay_ms() const noexcept {
    return impl_->delay_ms;
}

}  // namespace blackwell::audio_rt
