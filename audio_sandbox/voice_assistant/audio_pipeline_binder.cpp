// audio_pipeline_binder.cpp — see audio_pipeline_binder.hpp.
//
// MOVED, NOT CHANGED. The construction order, the two-phase split, the try/catch
// nesting around the AEC, the tap's drain-to-empty policy and every warning string
// are byte-for-byte what main() did.
#include "audio_pipeline_binder.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>

namespace rt {
namespace {

whisper::DspConfig make_dsp_config(int n_mels) {
    whisper::DspConfig c;
    c.n_mels = n_mels;
    return c;
}

}  // namespace

// The member-init list follows DECLARATION order, which is the order main() built
// these in. capture_ and the speech-output members are deliberately absent from it
// -- they are default-constructed in place, and capture_ is therefore already
// alive when realtime_ takes its ring by reference on the next line.
AudioPipelineBinder::AudioPipelineBinder(const AssistantSettings& settings, int n_mels,
                                         const std::string& data_dir, AppContext& ctx)
    : cfg_(make_dsp_config(n_mels)),
      ctx_(ctx),
      dsp_(cfg_, data_dir + "/mel_filters.bin"),
      spectrogram_(n_mels, /*max_frames=*/1000),
      recorder_(cfg_.sample_rate, /*out_dir=*/"recordings"),
      realtime_(dsp_, capture_.ring(), spectrogram_, &recorder_)
#if defined(VOICE_ASSISTANT_HAS_TTS)
      ,
      aec_out_(4096, 0.0f),
      far_scratch_(4096, 0.0f)
#endif
{
    capture_.start(settings.loopback_capture ? CaptureMode::Loopback : CaptureMode::Microphone,
                   settings.input_device_name, settings.input_device_index);
    std::printf("capture started (%s, 16 kHz mono f32) on %s\n", capture_.backend_name().c_str(),
                capture_.device_name().empty() ? "the system default device"
                                               : capture_.device_name().c_str());
    capture_.set_input_gain(settings.mic_gain);
}

AudioPipelineBinder::~AudioPipelineBinder() { stop(); }

void AudioPipelineBinder::stop() noexcept {
    if (stopped_) return;
    stopped_ = true;
    // The worker first: it is what calls into the speech mode, and the mode is
    // about to be torn down by whoever owns it.
    realtime_.stop();
    capture_.stop();
}

float AudioPipelineBinder::effective_tts_volume(float slider) const noexcept {
    return ctx_.tts_muted.load(std::memory_order_acquire) ? 0.0f : slider;
}

void AudioPipelineBinder::apply_live_settings(const AssistantSettings& s) {
#if defined(VOICE_ASSISTANT_HAS_TTS)
    // Live-toggleable because the filter bypasses in place. Null until the TTS
    // stack is up -- this also runs once BEFORE that, so the launch value is
    // applied at construction instead.
    if (ctx_.aec != nullptr) ctx_.aec->SetEnabled(s.aec_enabled);
    // VOLUME GOES TO THE SPEAKER ONLY. The canceller is not told about it: its
    // reference is a loopback of the endpoint, so the gain is already baked into
    // what it observes. The pre-gain/post-gain correction this used to need -- and
    // the re-convergence every time the slider moved -- went away with the
    // playback tap.
    //
    // Folded with the dock's mute: a Save while deafened must not turn the sound
    // back on behind the icon, which would leave the UI claiming a state the
    // speaker does not have.
    if (ctx_.tts != nullptr) ctx_.tts->SetVolume(effective_tts_volume(s.tts_volume));
#endif
    capture_.set_input_gain(s.mic_gain);
}

void AudioPipelineBinder::start_speech_output([[maybe_unused]] const AssistantSettings& settings,
                                              [[maybe_unused]] int device_id) {
#if defined(VOICE_ASSISTANT_HAS_TTS)
    if (settings.tts_ckpt_dir.empty()) {
        std::printf("[tts] disabled (no --tts-ckpt-dir / persisted setting)\n");
        return;
    }

    TtsRuntimeConfig tcfg;
    tcfg.ckpt_dir   = settings.tts_ckpt_dir;
    tcfg.vocab_path = settings.tts_vocab_path;
    tcfg.ref_audio  = settings.tts_ref_audio;
    tcfg.ref_text   = settings.tts_ref_text;
    tcfg.nfe_step   = settings.tts_nfe_step;
    tcfg.device_id  = device_id;
    tcfg.split_on_commas = settings.tts_split_on_commas;
    tcfg.min_chunk_chars = settings.tts_min_chunk_chars;
    tcfg.expand_numbers      = settings.tts_expand_numbers;
    tcfg.stress_marks        = settings.tts_stress_marks;
    tcfg.stress_dictionary   = settings.tts_stress_dictionary;
    tcfg.output_device       = settings.output_device_name;
    tcfg.output_device_index = settings.output_device_index;
    tcfg.volume              = settings.tts_volume;

    // THE PAIR MUST MATCH. F5 treats generation as infilling against (reference
    // audio, reference text), so a transcript that is not what the clip says makes
    // the model invent content to reconcile them -- the reference bleeds into
    // every utterance, which reads as a broken model rather than a
    // misconfiguration.
    //
    // The shipped default is a REAL matched pair (a FLEURS clip and its
    // ground-truth transcript), so "equals the default" is not the fault condition
    // -- it is the good case. What is still worth catching is an empty transcript,
    // and the half-edited state: a custom clip still carrying the shipped text.
    // Both are silent otherwise, and both cost a 1.3 GB graph load to discover by
    // ear.
    const AssistantSettings kDefaults{};
    if (settings.tts_ref_text.empty()) {
        std::fprintf(stderr,
                     "[tts] WARNING: reference transcript is EMPTY. Set it to the literal "
                     "text spoken in %s (Settings -> Audio -> TTS), or the voice will be "
                     "wrong.\n",
                     tcfg.ref_audio.c_str());
    } else if (settings.tts_ref_audio != kDefaults.tts_ref_audio &&
               settings.tts_ref_text == kDefaults.tts_ref_text) {
        std::fprintf(stderr,
                     "[tts] WARNING: reference clip was changed to %s but the transcript is "
                     "still the one shipped for %s. They must describe the SAME audio "
                     "(Settings -> Audio -> TTS).\n",
                     tcfg.ref_audio.c_str(), kDefaults.tts_ref_audio.c_str());
    }

    try {
        tts_.emplace(tcfg);
        ctx_.tts = &tts_.value();
        std::printf("[tts] ready: %s (nfe=%d, volume %.0f%%) on %s\n", tcfg.ckpt_dir.c_str(),
                    tcfg.nfe_step, static_cast<double>(tts_->volume()) * 100.0,
                    tts_->output_device().empty() ? "the system default device"
                                                  : tts_->output_device().c_str());

        // ---- the capture-side half of full duplex ---------------------------
        // Built here and not earlier because it needs the far-end ring, which only
        // matters once the runtime does. It is the SOLE consumer of that ring, per
        // its SPSC contract.
        //
        // ITS OWN try/catch, and that is not tidiness. Speech output is already
        // live and ctx_.tts is already published by this point, so letting a throw
        // fall into the handler below would print "[tts] disabled" about a TTS
        // stack that is running -- and would leave it running with no canceller,
        // which is the one configuration that self-triggers. The two failures are
        // different and have to say so.
        try {
            // THE LOOPBACK REFERENCE, opened on the OUTPUT endpoint -- both
            // selectors address the PLAYBACK list in this mode (audio_capture.h
            // says so, and it is the trap here: an index that looks like a capture
            // index is not one).
            //
            // Started BEFORE the filter exists so a failure to open is reported as
            // "no reference" rather than as a canceller that silently subtracts
            // nothing.
            loopback_capture_.start(CaptureMode::Loopback, settings.output_device_name,
                                    settings.output_device_index);

            blackwell::audio_rt::AecCaptureFilterConfig acfg;
            acfg.near_rate = static_cast<int>(cfg_.sample_rate);
            // SAME RATE, so the polyphase resampler degenerates to a copy:
            // miniaudio already delivers the loopback stream at the capture rate.
            // The old 24 kHz far end needed a 2/3 converter and paid its group
            // delay out of the filter's tail budget.
            acfg.far_rate = acfg.near_rate;
            // Clamped, not trusted: this arrives from a hand-editable settings
            // file, and a tail of zero produces a canceller that runs and cancels
            // nothing.
            acfg.aec.filter_tail_samples =
                static_cast<std::size_t>(std::clamp(settings.aec_tail_ms, 64, 1000)) *
                static_cast<std::size_t>(acfg.near_rate) / 1000u;

            // ---- the subtractor backend --------------------------------------
            // AEC3 when the package was found at configure time, the built-in
            // partitioned-block filter otherwise. The choice is made HERE rather
            // than inside AecCaptureFilter because blackwell_audio_rt is
            // dependency-free by construction and must not learn the name of a
            // WebRTC type.
            //
            // A THROW HERE IS NOT FATAL and deliberately falls through to the
            // built-in filter: an AEC3 that will not construct is a reason to
            // cancel worse, not a reason to have no speech.
            std::unique_ptr<blackwell::audio_rt::IEchoCanceller> backend;
            const char* backend_name = "built-in block-FDAF";
#if defined(BLACKWELL_HAVE_AEC3)
            try {
                blackwell::audio_rt::Aec3Config a3;
                a3.sample_rate_hz = acfg.near_rate;
                // A loopback reference is tapped at the endpoint, so the true
                // speaker->mic delay is the render buffer plus the capture buffer
                // -- tens of milliseconds, not a room's worth. AEC3 re-estimates
                // regardless; this only saves it the first second of searching.
                a3.initial_delay_ms = 30;
                backend = std::make_unique<blackwell::audio_rt::Aec3EchoCanceller>(a3);
                backend_name = "WebRTC AEC3";
            } catch (const std::exception& e) {
                std::fprintf(stderr,
                             "[aec] AEC3 unavailable (%s) -- falling back to the built-in "
                             "canceller.\n",
                             e.what());
                backend.reset();
            }
#endif
            aec_.emplace(loopback_ref_, acfg, std::move(backend));
            aec_->SetEnabled(settings.aec_enabled);
            // NO SetReferenceGain. The loopback stream is tapped after the mix, so
            // it already carries whatever gain the user set -- the correction the
            // playback tap needed (and the re-convergence every time the slider
            // moved) is retired.
            ctx_.aec = &aec_.value();
            ctx_.loopback = &loopback_capture_;
            std::printf("[aec] %s (%s): %d ms tail, %zu-sample capture latency, far end = "
                        "WASAPI loopback @ %d Hz on %s\n",
                        settings.aec_enabled ? "on" : "BYPASSED (diagnostic)", backend_name,
                        settings.aec_tail_ms, aec_->latency_samples(), acfg.far_rate,
                        loopback_capture_.device_name().empty()
                            ? "(system default output)"
                            : loopback_capture_.device_name().c_str());
            if (loopback_capture_.device_fallback()) {
                // THE failure that produces a canceller which subtracts the wrong
                // room: the reference must come from the endpoint the assistant is
                // SPEAKING through, and a silent fallback to a different one is
                // uncancellable.
                std::fprintf(stderr,
                             "[aec] WARNING: the loopback reference fell back to a different "
                             "endpoint than the one speech plays on -- cancellation will not "
                             "work until they match.\n");
            }
            if (!settings.aec_enabled) {
                // Said out loud because it is the difference between a demo that
                // works and one that talks over itself: with the canceller
                // bypassed, the assistant's own voice reaches the mic on open
                // speakers, Silero scores it as speech, and the pipeline barges in
                // on the answer it is currently giving.
                std::fprintf(stderr,
                             "[aec] WARNING: echo cancellation is OFF. Use HEADPHONES, or "
                             "the assistant will interrupt itself.\n");
            }
        } catch (const std::exception& e) {
            // Speech still works; what is lost is the ability to survive hearing
            // it. Degrade loudly rather than silently.
            std::fprintf(stderr,
                         "[aec] DISABLED: %s\n"
                         "[aec] The assistant will hear its own voice and may interrupt "
                         "itself. Use HEADPHONES.\n",
                         e.what());
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[tts] disabled: %s\n", e.what());
    }
#endif
}

void AudioPipelineBinder::bind_speech_mode(ISpeechMode* mode) {
    mode_ = mode;

    // THE CAPTURE TAP -- and the whole of what full duplex means here.
    //
    // The microphone is NEVER muted, gated, zeroed or paused. Every block the
    // capture device produces reaches the AEC, the level meter and the ring: the
    // PCM path is untouched, so pre-roll, metering and speculative warming all
    // keep working while the assistant talks. There is no state flag anywhere on
    // this path and no verdict is masked -- what removes the assistant's voice is
    // a SUBTRACTION, and the thing being subtracted is a loopback of the speaker
    // itself.
    realtime_.set_pcm_tap([this](const float* s, std::size_t n) {
#if defined(VOICE_ASSISTANT_HAS_TTS)
        if (ctx_.aec != nullptr) {
            // ---- pump the far end ------------------------------------------
            // The loopback device writes into its own SampleRing on its own
            // callback thread; the filter reads a SpscRing. This moves one from
            // the other, and it happens HERE because this thread is the filter's
            // only consumer -- so the SpscRing ends up written and read by the
            // same thread, which trivially satisfies its SPSC contract.
            //
            // Drained to EMPTY rather than n samples: the two devices deliver on
            // independent schedules, and leaving a residue would let the reference
            // fall progressively behind the microphone -- the one direction of
            // misalignment no causal filter can represent
            // (aec_capture_filter.hpp).
            for (;;) {
                const std::size_t got =
                    loopback_capture_.ring().pop(far_scratch_.data(), far_scratch_.size());
                if (got == 0) break;
                // write_or_drop: this is a real-time-ish path and the filter's
                // backlog ceiling would discard the excess anyway. A ring that is
                // full means the filter is not keeping up, which its own resync
                // counter reports.
                (void)loopback_ref_.write_or_drop(far_scratch_.data(), got);
                if (got < far_scratch_.size()) break;
            }

            if (aec_out_.size() < n) aec_out_.resize(n);
            ctx_.aec->Process(s, n, aec_out_.data());
            mode_->on_pcm_block(aec_out_.data(), n);
            return;
        }
#endif
        // No canceller in this build or this launch: pass the microphone through
        // untouched.
        mode_->on_pcm_block(s, n);
    });
    realtime_.start();
}

void AudioPipelineBinder::apply_audio_reload(const AssistantSettings& s) {
#if defined(VOICE_ASSISTANT_HAS_TTS)
    if (ctx_.tts != nullptr) {
        const bool ok = ctx_.tts->HotReloadOutput(s.output_device_name, s.output_device_index);
        std::printf("[audio] output hot reload %s -> %s\n", ok ? "ok" : "FAILED",
                    ctx_.tts->output_device().empty() ? "(system default)"
                                                      : ctx_.tts->output_device().c_str());
    }
#endif
    // Loopback capture taps a PLAYBACK endpoint, so it follows the OUTPUT
    // selection -- the same trap the name form has always had, restated here
    // because getting it wrong yields MA_NO_DEVICE with no hint.
    const bool loopback = s.loopback_capture;
    const bool cap_ok =
        capture_.hot_reload(loopback ? CaptureMode::Loopback : CaptureMode::Microphone,
                            loopback ? s.output_device_name : s.input_device_name,
                            loopback ? s.output_device_index : s.input_device_index);
    std::printf("[audio] input hot reload %s -> %s\n", cap_ok ? "ok" : "FAILED",
                capture_.device_name().empty() ? "(system default)"
                                               : capture_.device_name().c_str());
    capture_.set_input_gain(s.mic_gain);
#if defined(VOICE_ASSISTANT_HAS_TTS)
    // THE canceller's learned impulse response describes the OLD room path -- old
    // speaker, old microphone, old latency between them. Left alone it would spend
    // a few hundred milliseconds actively subtracting the wrong signal, which is
    // worse than not cancelling. The loopback reference follows the OUTPUT
    // endpoint, because that is the speaker whose echo it exists to describe. Moved
    // BEFORE the filter is reset, so the reset lands on a reference that is already
    // pointing at the new room.
    if (ctx_.loopback != nullptr) {
        const bool lb_ok = ctx_.loopback->hot_reload(CaptureMode::Loopback,
                                                     s.output_device_name,
                                                     s.output_device_index);
        std::printf("[aec] loopback reference hot reload %s -> %s\n", lb_ok ? "ok" : "FAILED",
                    ctx_.loopback->device_name().empty()
                        ? "(system default output)"
                        : ctx_.loopback->device_name().c_str());
    }
    if (ctx_.aec != nullptr) ctx_.aec->Reset();
#endif
    std::fflush(stdout);
}

void AudioPipelineBinder::print_shutdown_summary() const {
#if defined(VOICE_ASSISTANT_HAS_TTS)
    // A SEPARATE SUMMARY, because these count a different stage and confusing the
    // two is what makes "it went silent" hard to place. The commit gate's numbers
    // are about the user's TRANSCRIPT becoming an intent; these are about the
    // ANSWER becoming sound, which happens after and independently.
    //
    // `cancelled` is the one to read first when the assistant answered on screen
    // but said nothing or stopped mid-sentence: every count is a barge-in that
    // killed a chunk.
    //
    // WHAT A NON-ZERO COUNT MEANS. Either somebody interrupted, or the canceller is
    // not removing enough of the assistant's own voice for Silero to stop scoring
    // it as speech. There is no state flag to suspect any more -- the ONLY thing
    // standing between the loudspeaker and a barge-in is the subtraction. So read
    // it next to the [aec] ERLE line: a healthy loopback reference and a converged
    // filter give a quiet microphone, and if the ERLE is low the reference is the
    // thing to check first (wrong endpoint, or a fallback warned about at startup).
    if (tts_.has_value()) {
        std::printf("\n=== speech output summary ===\n");
        std::printf("  chunks spoken          : %llu\n",
                    static_cast<unsigned long long>(tts_->chunks_spoken()));
        std::printf("  chunks cancelled       : %llu%s\n",
                    static_cast<unsigned long long>(tts_->chunks_cancelled()),
                    tts_->chunks_cancelled() > 0
                        ? "   <-- barge-in killed these; if nobody interrupted, read the "
                          "ERLE below"
                        : "");
        std::printf("  synthesis errors       : %llu\n",
                    static_cast<unsigned long long>(tts_->synthesis_errors()));
    }
    // THE NUMBER THAT EXPLAINS THE ONE ABOVE. ERLE is how many dB of the
    // assistant's own voice the canceller is actually removing; resyncs are how
    // often the reference had to be discarded to catch up with the microphone,
    // which is the clock-drift term loopback does NOT fix. Reference underruns mean
    // the loopback device stopped delivering, which makes the filter subtract
    // silence and cancel nothing.
    if (aec_.has_value()) {
        std::printf("  AEC erle               : %.1f dB\n",
                    static_cast<double>(aec_->erle_db()));
        std::printf("  AEC resyncs            : %llu (reference discarded to catch up)\n",
                    static_cast<unsigned long long>(aec_->resyncs()));
        std::printf("  AEC ref underruns      : %llu%s\n",
                    static_cast<unsigned long long>(aec_->reference_underruns()),
                    aec_->reference_underruns() > 0
                        ? "   <-- the loopback reference had gaps; cancellation was blind "
                          "for those samples"
                        : "");
    }
#endif
}

}  // namespace rt
