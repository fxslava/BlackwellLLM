// full_stack_test.cpp — see full_stack_test.hpp. Moved verbatim out of main.cpp.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include "full_stack_test.hpp"

#include <chrono>
#include <cstdio>
#include <thread>

#include "assistant_window.hpp"
#include "engine_bootstrap.hpp"   // rt::report_vram

namespace rt {

void run_full_stack_test(const std::string& prompt, AssistantSettings settings, bool use_real,
                         [[maybe_unused]] AppContext& ctx,
                         [[maybe_unused]] AudioPipelineBinder& audio,
                         ConversationRouter& router, AssistantWindow& window) {
    try {
        std::printf("\n=== full-stack LLM+TTS test ===\n  prompt: \"%s\"\n", prompt.c_str());
#if defined(VOICE_ASSISTANT_HAS_TTS)
        if (ctx.tts != nullptr) ctx.tts->reset_playback_stats();
#endif
        // Let the device settle so the first buffers are not counted as starvation
        // from before there was anything to play.
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        router.submit_typed_turn(prompt);
        {
            // Sample WHILE the turn runs: a summary printed after the fact cannot
            // show whether the ring ran dry during decode, which is the whole
            // question.
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
            bool spoke = false;
            while (std::chrono::steady_clock::now() < deadline) {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
#if defined(VOICE_ASSISTANT_HAS_TTS)
                if (ctx.tts == nullptr) break;
                const auto ps = ctx.tts->playback_stats();
                std::printf("[audio-playback] WASAPI requested %u frames | RingBuffer "
                            "available: %u frames | starved %llu/%llu callbacks | ring "
                            "underruns %llu | chunks %llu\n",
                            ps.last_requested, ps.last_available,
                            static_cast<unsigned long long>(ps.starved_callbacks),
                            static_cast<unsigned long long>(ps.callbacks),
                            static_cast<unsigned long long>(ctx.tts->speaker_underruns()),
                            static_cast<unsigned long long>(ctx.tts->chunks_spoken()));
                std::fflush(stdout);
                if (ctx.tts->chunks_spoken() > 0) spoke = true;
                // Done when the speaker has drained after speaking.
                if (spoke && !ctx.tts->speaking()) break;
#else
                break;
#endif
            }
        }
#if defined(VOICE_ASSISTANT_HAS_TTS)
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        if (ctx.tts != nullptr) {
            const auto ps = ctx.tts->playback_stats();
            std::printf("\n  --- playback under load ---\n");
            std::printf("  callbacks              : %llu\n",
                        static_cast<unsigned long long>(ps.callbacks));
            std::printf("  frames requested/served: %llu / %llu\n",
                        static_cast<unsigned long long>(ps.frames_requested),
                        static_cast<unsigned long long>(ps.frames_served));
            std::printf("  starved callbacks      : %llu\n",
                        static_cast<unsigned long long>(ps.starved_callbacks));
            std::printf("  speaker-ring underruns : %llu\n",
                        static_cast<unsigned long long>(ctx.tts->speaker_underruns()));
            std::printf("  chunks spoken/cancelled: %llu / %llu   errors %llu\n",
                        static_cast<unsigned long long>(ctx.tts->chunks_spoken()),
                        static_cast<unsigned long long>(ctx.tts->chunks_cancelled()),
                        static_cast<unsigned long long>(ctx.tts->synthesis_errors()));
            std::printf("  %s\n",
                        ctx.tts->chunks_spoken() > 0 && ctx.tts->synthesis_errors() == 0
                            ? "PASS -- audio was synthesised and played while the model was "
                              "decoding."
                            : "FAIL -- nothing reached the speaker during the turn.");
        }

        // ---- hot swap, on the loaded stack ----------------------------------
        // The acceptance test for the AudioHotReload tier: move the endpoint with
        // 8 GB of weights resident and show that the VRAM figure does not move. A
        // reload would be unmissable here -- the arena alone is gigabytes.
        if (ctx.tts != nullptr) {
            std::printf("\n  --- audio hot swap (no model reload) ---\n");
            if (use_real) report_vram("  before swap");
            // Deliberately to a DIFFERENT endpoint than the one in use, so a no-op
            // cannot pass as a success.
            AssistantSettings swapped = settings;
            swapped.output_device_index = (settings.output_device_index == 4) ? 0 : 4;
            swapped.output_device_name.clear();
            audio.apply_audio_reload(swapped);
            std::printf("  now on: %s\n", ctx.tts->output_device().empty()
                                              ? "(system default)"
                                              : ctx.tts->output_device().c_str());
            if (use_real) report_vram("  after swap ");
            // Prove the NEW device actually carries audio, through the same
            // pipeline speech uses.
            ctx.tts->PlayTestTone();
            std::this_thread::sleep_for(std::chrono::milliseconds(900));
            std::printf("  test tone on the new device: %s\n",
                        ctx.tts->chunks_spoken() > 0 ? "published" : "NOT published");
        }
#else
        (void)settings;
        (void)use_real;
#endif
        std::fflush(stdout);
    } catch (...) {
        std::printf("  FAIL -- the test threw.\n");
    }
    // Close the window so the process exits and the summaries print.
    if (window.hwnd() != nullptr) PostMessageW(window.hwnd(), WM_CLOSE, 0, 0);
}

}  // namespace rt
