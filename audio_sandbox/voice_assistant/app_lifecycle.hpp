#pragma once
// -----------------------------------------------------------------------------
// app_lifecycle.hpp — the startup ORDER and the teardown order, as an object
// whose declaration order enforces both.
//
// WHAT IT OWNS, top to bottom, which is also the order it builds them in:
//
//   1. the VRAM budget      resolved BEFORE anything allocates, so an
//                           unaffordable max_context is clamped (or refused)
//                           while the answer is still free. See vram_budget.hpp.
//   2. the engine stack     RealEngineControl on a real checkpoint, or
//                           SimulatedEngineControl with no GPU at all. Both
//                           derive from EngineControlBridge, so `control()` is
//                           the only handle anything else needs and nothing
//                           downstream branches on which one is live.
//   3. the neural VAD       a startup-cost ORT session, non-fatal, outliving the
//                           mode that reads it.
//   4. the speech mode      Mode A (Ultravox legacy) or Mode C (whisper cascade).
//                           EXACTLY ONE is constructed -- they are two answers to
//                           the same question and each wants the VRAM the other
//                           is holding.
//   5. the engine thread    the ONLY thread that calls into the control.
//
// WHY THE ORDER IS LOAD-BEARING, in the two places it is not obvious:
//   * the VAD is built BEFORE the mode and outlives it, because the mode borrows
//     the scorer through the C ABI's SpeechVadScoreFn seam;
//   * the mode is built AFTER the engine stack and destroyed BEFORE it, because
//     it holds the control pointer and pumps it.
//
// THE SINGLE-THREADED ENGINE DOCTRINE (CLAUDE.md, "the most important rule") is
// what the engine thread here exists to satisfy. It selects the CUDA device for
// ITSELF -- the current device is PER-THREAD, so doing it only in main() would
// silently run kernels on device 0 against device-N memory -- and it is the only
// thread permitted to call into the control after load. Everything else marshals
// via post_engine_task.
//
// ERROR TIER: INIT. The constructor and start_engine_thread() throw; main's
// handler reports and exits. A throw on the engine thread is CAUGHT there,
// published, and re-thrown from start_engine_thread() on the caller's thread --
// before that existed, a failed prefill hung main() forever on the latch.
// -----------------------------------------------------------------------------
#include "build_features.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>

#include "conversational_mode.hpp"      // rt::ConversationalMode (Mode A)
#include "engine_bootstrap.hpp"         // rt::RealEngineStack, bring_up_real_engine
#include "simulated_engine_control.hpp" // rt::SimulatedEngineControl
#include "speech_mode.hpp"              // rt::ISpeechMode

#include "app_context.hpp"
#include "cli_config.hpp"               // rt::TranslatorArgs
#include "engine_residency.hpp"         // rt::EngineResidency
#include "reply_split.hpp"              // rt::compose_system_prompt (both modes' prefix)
#include "settings_store.hpp"

namespace rt {

// The COM apartment, claimed FIRST and released LAST.
//
// THE FIRST CALLER WINS, PERMANENTLY. A thread's apartment cannot be changed once
// set: a later CoInitializeEx with a different model returns RPC_E_CHANGED_MODE
// and leaves the thread where it was.
//
// WebView2 is STA-only. miniaudio, meanwhile, initializes COM on whatever thread
// first opens a device and defaults to COINIT_MULTITHREADED -- and capture.start()
// runs on the main thread, well before the window is created. That ordering put
// the main thread in the MTA and made CreateCoreWebView2EnvironmentWithOptions
// fail with an error the UI then reported as a missing runtime, sending anyone who
// hit it off to reinstall a runtime that was already installed.
//
// So the apartment is claimed here, at the top of main, where it is a stated
// decision rather than a side effect of whichever subsystem happened to boot
// first. miniaudio's own CoInitializeEx then returns RPC_E_CHANGED_MODE, which it
// tolerates -- WASAPI works fine from an STA host, and its device thread has its
// own apartment regardless.
//
// RAII rather than paired calls: the previous arrangement uninitialized on the
// early-return paths and leaked the apartment on every throw, which is exactly the
// asymmetry a guard type removes.
class ComApartment {
public:
    ComApartment();   // INIT tier: throws if STA cannot be claimed
    ~ComApartment();
    ComApartment(const ComApartment&) = delete;
    ComApartment& operator=(const ComApartment&) = delete;

private:
    bool owned_ = false;
};

class AppLifecycleManager {
public:
    struct Config {
        // Resolved settings and args -- ONE description of the launch, already
        // folded together by main (persisted settings < typed flags).
        AssistantSettings settings;
        TranslatorArgs    args;
        // Which speech-to-text pipeline. Decides what gets ALLOCATED: the Ultravox
        // audio head and its dimension check, or a ~1.6 GB GGML model.
        bool        cascade = false;
        std::string whisper_model;
        // THE decision. False means SimulatedEngineControl: no GPU, no checkpoint.
        bool use_real = false;
        // Which leg the app BOOTS on. False (remote/cloud) is what makes the
        // launch LAZY: the control is constructed detached and no weights are
        // read from disk until the user turns local inference on.
        //
        // It does NOT gate the legacy pipeline, and that exception is structural
        // rather than a convenience: on ultravox_legacy the backbone is also the
        // speech recogniser, so a launch with no weights is a launch that cannot
        // hear. Deferring there would trade a few seconds of startup for an app
        // that silently does not work. See bring_up_engine.
        bool local_inference = false;
        // The capture rate, from the audio pipeline's DspConfig rather than from
        // settings: it is the rate the microphone is ACTUALLY delivering at, and
        // the speech modes segment against it. Passing the configured value
        // instead would let the two disagree the moment the DSP clamps one.
        std::uint32_t sample_rate = 16000;
    };

    // `dsp` is BORROWED and must outlive this object (the control holds a pointer
    // to it for the live commit path), which is what declaring the audio pipeline
    // before this in main() gives. `ctx` is likewise borrowed -- the speech-mode
    // callbacks reach state through it.
    //
    // INIT tier: throws on an unaffordable VRAM budget, a bad checkpoint, or a
    // cascade with no GGML model.
    AppLifecycleManager(const Config& cfg, whisper::WhisperDSP& dsp, AppContext& ctx);
    ~AppLifecycleManager();

    AppLifecycleManager(const AppLifecycleManager&) = delete;
    AppLifecycleManager& operator=(const AppLifecycleManager&) = delete;

    // Spawn the engine thread and BLOCK until its system-prompt prefill has run
    // (or failed). Throws with the engine thread's own message on failure, having
    // already joined it.
    void start_engine_thread();

    // Stop the engine thread and join it. Idempotent. MUST be called after the
    // audio pipeline has stopped (its PCM tap calls into the mode this pumps) and
    // BEFORE the dispatcher is stopped (so no commit can arrive post-join).
    void shutdown() noexcept;

    // ---- accessors -----------------------------------------------------------
    // The base type. Never null once the constructor returns.
    blackwell::bridge::EngineControlBridge* control() const noexcept { return control_; }
    // The concrete real control, or null on the simulated backend. Only the
    // handful of genuinely real-engine-only settings need this.
    RealEngineControl* real_control() const noexcept { return real_stack_.control.get(); }
    SimulatedEngineControl* simulated_control() const noexcept { return simulated_.get(); }
    const RealEngineStack& real_stack() const noexcept { return real_stack_; }
    ISpeechMode* mode() const noexcept { return active_; }

    // Whether the local model's weights are on the card, and the handle that
    // moves them. Never null once the constructor returns -- on the simulated
    // backend it is a residency that permanently reports Ready and refuses to
    // release anything, so callers never branch on which backend is live.
    EngineResidency& residency() noexcept { return *residency_; }

    // The context the engine was ACTUALLY brought up with -- which may be less
    // than the configured max_context if the VRAM guard clamped it.
    int granted_context() const noexcept { return granted_context_; }

    // The frozen system prefix, in tokens: the KV rewind floor. Not on ISpeechMode
    // because the number only means anything to a caller that already knows which
    // mode laid the prefix down and how many prefixes it laid -- branching here is
    // cheaper than a virtual that would have to explain itself.
    std::uint32_t frozen_prefix_tokens() const;

    // The cascade's own ASR counters, reported separately because they answer a
    // DIFFERENT question from the commit gate's: the gate says whether transcripts
    // became intents, these say whether speech became transcripts at all. No-op on
    // the legacy path.
    void print_shutdown_summary() const;

private:
    // Throws for a pipeline this BINARY cannot serve (Mode B). Runs before any
    // load: the answer does not depend on the machine or the checkpoint.
    static void refuse_unsupported_mode(const Config& cfg);
    void bring_up_engine(const Config& cfg, whisper::WhisperDSP& dsp);
    void bring_up_vad(const Config& cfg);
    void bring_up_speech_mode(const Config& cfg, AppContext& ctx);

    // ---- DECLARATION ORDER IS DESTRUCTION ORDER (reversed). Do not reorder. ----

    // Captured so the engine thread can select the same device the weights were
    // allocated on.
    int  device_id_ = 0;
    bool use_real_ = false;
    int  granted_context_ = 0;
    // True when the engine was NOT loaded at bring-up (remote-only launch). The
    // control exists and is detached; the residency starts UNLOADED and its
    // cold-load path is what fills the stack in.
    bool lazy_ = false;
    bool cascade_ = false;

    // Everything the DEFERRED load needs, captured at construction because the
    // Config that carried it is gone by the time the user flips the toggle.
    // `dsp` is borrowed on the same terms the constructor borrows it.
    TranslatorArgs      load_args_;
    whisper::WhisperDSP* dsp_ = nullptr;

    // Exactly one of these is constructed; control_ is the only handle anything
    // outside this class uses.
    RealEngineStack                         real_stack_;
    std::unique_ptr<SimulatedEngineControl> simulated_;
    blackwell::bridge::EngineControlBridge*  control_ = nullptr;

    // AFTER the stack it borrows the engine and the control from, and therefore
    // destroyed BEFORE them. It queues work onto the engine thread, so it must
    // also outlive nothing -- shutdown() joins that thread before this dies.
    std::unique_ptr<EngineResidency> residency_;

#if defined(VOICE_ASSISTANT_HAS_SILERO)
    // Constructed BEFORE the mode and outliving it: the ORT session is a
    // startup-cost resource. Non-fatal, like the audio head.
    std::unique_ptr<blackwell::vad::SileroVAD> neural_vad_;
    VadContext                                 vad_ctx_;
#endif
    // The C ABI's own seam -- null leaves the pipeline on its built-in RMS
    // threshold detector, exactly as a failed model load does.
    SpeechVadScoreFn vad_fn_ = nullptr;
    void*            vad_user_ = nullptr;

    // ONE of the two is constructed, never both. `active_` is the only handle the
    // runner, the PCM tap and the shutdown path use; the typed handles exist for
    // the handful of call sites that need something ISpeechMode deliberately does
    // not carry.
    std::unique_ptr<ConversationalMode> conv_mode_;
#if defined(VOICE_ASSISTANT_HAS_WHISPER)
    std::unique_ptr<WhisperCascadeMode> cascade_mode_;
#endif
    ISpeechMode* active_ = nullptr;

    std::atomic<bool> running_{true};
    std::atomic<bool> prefilled_{false};
    std::atomic<bool> engine_failed_{false};
    std::string       engine_error_;

    // LAST, so it is joined and destroyed FIRST.
    std::thread engine_thread_;
    bool        stopped_ = false;
};

}  // namespace rt
