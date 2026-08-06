// cli_options.cpp — see cli_options.hpp. Moved verbatim out of main.cpp: the
// flag table, the precedence bookkeeping and both self-tests are byte-for-byte
// what main() ran.
#include "cli_options.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

#include "audio_capture.h"
#include "audio_devices.h"

namespace rt {

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
            // itself a flag, so `--test-llm-tts --real` does not silently swallow
            // --real and then generate a reply to the word "--real".
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
namespace {

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

// Measures the amplitude of the 1 kHz TONE specifically, by Goertzel, rather than
// the broadband RMS of everything the endpoint is playing.
//
// This is not fussiness. Loopback captures the whole system mix, so a broadband
// measurement adds a noise floor `n` in quadrature to both readings: at unity it
// is invisible (sqrt(A^2 + n^2) ~ A), but at a quarter gain it dominates
// sqrt((A/4)^2 + n^2) and inflates the ratio. Measured that way this test reported
// 0.302 for a gain of exactly 0.250 -- correct hardware, misleading number, and
// the one thing a volume self-test must never do is make working volume look
// broken. A single-bin DFT at the tone's own frequency rejects everything else in
// the mix.
//
// Drains first: the ring holds audio produced at the PREVIOUS gain, and averaging
// across the change would report the mean of two answers.
double measure_tone(AudioCapture& cap, int ms) {
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

}  // namespace

int run_volume_check(const AssistantSettings& settings) {
    std::printf("=== volume self-test (a 1 kHz tone will play for ~3 seconds) ===\n");
    try {
        AudioPlayback playback;
        ToneState tone;
        tone.step =
            6.283185307179586 * 1000.0 / static_cast<double>(blackwell::tts::kF5SampleRate);
        playback.set_volume(1.0f);
        playback.start(blackwell::tts::kF5SampleRate, &tone_pull, &tone,
                       settings.output_device_name, settings.output_device_index);
        std::printf("  output   : %s\n", playback.device_name().empty()
                                             ? "(system default)"
                                             : playback.device_name().c_str());

        // Loopback on the SAME endpoint -- measuring a different speaker would
        // measure nothing. In loopback mode both selectors refer to a PLAYBACK
        // device, which is why the OUTPUT name and index are the ones passed here;
        // the input_* settings would name the wrong list entirely.
        AudioCapture cap;
        cap.start(CaptureMode::Loopback, settings.output_device_name,
                  settings.output_device_index);

        std::this_thread::sleep_for(std::chrono::milliseconds(400));   // device settle
        const double full = measure_tone(cap, 700);

        // THE POINT OF THE TEST: changed while the device is running, with no
        // restart and no re-synthesis.
        constexpr float kQuiet = 0.25f;
        playback.set_volume(kQuiet);
        // Longer than the endpoint's buffer: samples already handed to WASAPI carry
        // the OLD gain and are still going to be emitted.
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        const double quiet = measure_tone(cap, 700);

        cap.stop();
        playback.stop();

        std::printf("  1 kHz level at 100%%: %.5f\n  1 kHz level at  25%%: %.5f\n", full, quiet);
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

int run_say_check(const AssistantSettings& settings, const std::string& text) {
    std::printf("=== speech self-test: synthesising %zu chars ===\n", text.size());
    try {
        TtsRuntimeConfig tcfg;
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

        TtsRuntime tts(tcfg);
        std::printf("  output   : %s\n", tts.output_device().empty()
                                             ? "(system default)"
                                             : tts.output_device().c_str());

        // The dispatcher's exact call sequence. Resume() first for the same reason
        // it is on the dispatch-start edge: without it a barge-in latch left set
        // from a previous turn silently eats every token.
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
        // Let the tail actually reach the speaker before the dtor stops the device
        // -- otherwise a PASS would be reported for audio nobody heard.
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

}  // namespace rt
