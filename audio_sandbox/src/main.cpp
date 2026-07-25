// -----------------------------------------------------------------------------
// main.cpp — real-time Whisper log-mel spectrogram visualiser.
//
//   audio_realtime [data_dir]        (data_dir default: "data")
//     <data_dir>/mel_filters.bin     the VERIFIED Whisper filterbank (loaded, not
//                                    recomputed — same as the parity sandbox).
//
// Pipeline (concerns split across threads):
//   miniaudio callback ─push─▶ SampleRing ─pop─▶ RealTimeDSP worker ─┬─push─▶
//   SpectrogramBuffer ─snapshot─▶ Direct2D + ImGui UI (this thread)  │
//                                                                    └─▶ AudioRecorder (VAD)
//
// The offline parity tester is preserved as `parity_check` (src/parity_check.cpp).
// -----------------------------------------------------------------------------
#include <cctype>
#include <cstdio>
#include <exception>
#include <string>

#include "audio_capture.h"
#include "audio_recorder.h"
#include "realtime_dsp.h"
#include "whisper_dsp.h"
#include "window_d2d.h"

int main(int argc, char** argv) {
    const std::string data_dir = (argc > 1) ? argv[1] : "data";

    std::printf("=== Whisper real-time log-mel spectrogram ===\n");
    std::printf("Capture source:\n");
    std::printf("  [M] Microphone\n");
    std::printf("  [L] System loopback (what you hear)\n");
    std::printf("> ");
    std::fflush(stdout);

    int ch = std::getchar();
    const rt::CaptureMode mode =
        (ch == 'l' || ch == 'L') ? rt::CaptureMode::Loopback : rt::CaptureMode::Microphone;
    std::printf("selected: %s\n",
                mode == rt::CaptureMode::Loopback ? "system loopback" : "microphone");

    try {
        // Same verified filterbank + DSP geometry as the parity sandbox.
        whisper::DspConfig cfg;
        whisper::WhisperDSP dsp(cfg, data_dir + "/mel_filters.bin");

        rt::SpectrogramBuffer spectrogram(cfg.n_mels, /*max_frames=*/1000);

        rt::AudioRecorder recorder(cfg.sample_rate, /*out_dir=*/"recordings");

        rt::AudioCapture capture;
        capture.start(mode);
        std::printf("capture started (backend: %s, 16 kHz mono f32)\n",
                    capture.backend_name().c_str());

        rt::RealTimeDSP realtime(dsp, capture.ring(), spectrogram, &recorder);
        realtime.start();

        rt::WindowD2D window(spectrogram, recorder, L"Whisper Log-Mel (real-time)");
        if (!window.create(/*client_w=*/1000, /*client_h=*/512)) {
            throw std::runtime_error("failed to create Direct2D window");
        }
        std::printf("rendering... close the window to quit.\n");
        window.run_message_loop();

        // Stop feeding the recorder (join the worker) before it is destroyed.
        realtime.stop();
        capture.stop();
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
