// app_lifecycle.cpp — see app_lifecycle.hpp.
//
// MOVED, NOT CHANGED, with one addition: the VRAM budget now runs here, between
// selecting the device and constructing the engine, because this is the object
// that owns "what gets allocated, in what order". Everything else -- the bring-up
// arguments, the isolated-sessions rationale, the cascade refusal, the engine
// thread's device selection and the prefill latch -- is byte-for-byte what main()
// did.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <objbase.h>   // CoInitializeEx / COINIT_APARTMENTTHREADED

#include "app_lifecycle.hpp"

#include <chrono>
#include <cstdio>
#include <stdexcept>

#include "asset_paths.hpp"
#include "vram_budget.hpp"
#if defined(VOICE_ASSISTANT_HAS_WHISPER)
#include "whisper_asr.hpp"   // rt::WhisperAsr, for the cascade's shutdown counters
#endif

namespace rt {

// ---- ComApartment -----------------------------------------------------------

ComApartment::ComApartment() {
    const HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    if (FAILED(hr)) {
        throw std::runtime_error("CoInitializeEx(STA) failed: 0x" + [hr] {
            char buf[16] = {};
            std::snprintf(buf, sizeof(buf), "%08lX", static_cast<unsigned long>(hr));
            return std::string(buf);
        }());
    }
    // S_FALSE means "already initialized on this thread, with a COMPATIBLE model"
    // -- we still own a reference and must still balance it.
    owned_ = true;
}

ComApartment::~ComApartment() {
    if (owned_) CoUninitialize();
}

// ---- AppLifecycleManager ----------------------------------------------------

AppLifecycleManager::AppLifecycleManager(const Config& cfg, whisper::WhisperDSP& dsp,
                                         AppContext& ctx)
    : device_id_(cfg.settings.device_id),
      use_real_(cfg.use_real),
      granted_context_(cfg.settings.max_context) {
    // FIRST, before a single byte is loaded. The refusal below is a property of
    // this BINARY, not of the machine or the checkpoint, so it is knowable before
    // any work is done -- and answering it after a 5.3 GB weight load would make
    // an instantly-decidable configuration error cost thirty seconds.
    // Captured for the DEFERRED load: the Config that carries them is gone by the
    // time the user flips the toggle, and `dsp` is borrowed on exactly the terms
    // the constructor borrows it (it outlives this object).
    load_args_ = cfg.args;
    dsp_ = &dsp;
    cascade_ = cfg.cascade;

    refuse_unsupported_mode(cfg);
    bring_up_engine(cfg, dsp);
    bring_up_vad(cfg);
    bring_up_speech_mode(cfg, ctx);
    ctx.control = control_;

    // AFTER the engine exists, because it borrows it -- and unconditionally, so
    // that `residency()` is never null and no caller has to branch on the
    // backend. On the simulated path `engine` is null, which the residency reads
    // as "no VRAM to release" and answers every request with a refusal while
    // permanently reporting Ready.
    //
    // weights_serve_asr is `!cascade`: on the legacy pipeline the backbone IS
    // the speech recogniser, so its weights cannot be released without making
    // the app deaf. See the capability gate in engine_residency.hpp.
    // A CALLABLE, not real_stack_.engine.get(): on a lazy launch the engine is
    // null now and becomes real later, and a pointer captured here would never
    // see it. Reading through the stack keeps one source of truth.
    residency_ = std::make_unique<EngineResidency>(
        control_, [this] { return real_stack_.engine.get(); },
        /*weights_serve_asr=*/!cfg.cascade);

    if (lazy_) {
        // The deferred half of bring_up_engine, as a closure the residency runs
        // ON THE ENGINE THREAD when the user turns local inference on. It fills
        // in the SAME stack whose control everything is already borrowing, so
        // nothing downstream is re-pointed and nothing is rebuilt.
        residency_->enable_cold_load(
            [this] {
                load_engine_into(real_stack_, load_args_, *dsp_, granted_context_, device_id_,
                                 /*arm_streaming_plan=*/true,
                                 /*isolated_sessions=*/!cascade_,
                                 /*load_audio_head=*/!cascade_);
            },
            // The prefill the speech mode skipped at startup. It has to happen
            // AFTER the engine is attached and BEFORE the first intent is
            // answered, which is exactly this seam -- the residency does not
            // publish Ready until it returns.
            [this] {
                if (active_ != nullptr) active_->start_on_engine_thread();
            });
    }
}

AppLifecycleManager::~AppLifecycleManager() { shutdown(); }

void AppLifecycleManager::bring_up_engine(const Config& cfg, whisper::WhisperDSP& dsp) {
    if (!cfg.use_real) {
        simulated_ = std::make_unique<SimulatedEngineControl>();
        control_ = simulated_.get();
        return;
    }

    // ---- THE VRAM GUARD, before a single byte is committed ------------------
    // max_context is a persisted number with a 512..131072 range and no
    // relationship to the card it lands on. Nothing downstream re-checks it, and
    // on WDDM an over-commitment does not even FAIL -- it pages to system RAM over
    // PCIe and the app merely appears to freeze. So the budget is resolved HERE,
    // between selecting the device and constructing the engine, which is the last
    // moment at which the answer is still free.
    //
    // The two topology flags below are the SAME ones handed to
    // bring_up_real_engine further down, deliberately: a budget computed for a
    // different topology than the one about to be allocated is worse than no
    // budget at all.
    //
    // The budget queries cudaMemGetInfo, which requires a device to have been
    // selected on THIS thread. bring_up_real_engine does it again; cudaSetDevice
    // is idempotent and the second call is free.
    select_cuda_device(cfg.settings.device_id);

    VramBudgetInputs vin;
    vin.model_dir          = cfg.args.model_dir;
    vin.audio_head_dir     = cfg.args.audio_head;
    vin.load_audio_head    = !cfg.cascade;
    vin.whisper_model_path = cfg.cascade ? cfg.whisper_model : std::string{};
#if defined(VOICE_ASSISTANT_HAS_TTS)
    // The F5 stack is built much LATER (after the engine), but it is built on the
    // SAME card and its cost is not negotiable by then -- so it is charged to the
    // budget now, while the KV size is still the variable. An empty ckpt dir is the
    // same test the construction site uses.
    vin.tts_enabled = !cfg.settings.tts_ckpt_dir.empty();
#endif
    vin.requested_context = cfg.settings.max_context;
    // Isolated sessions fork a second branch and the paged pool is sized per
    // branch, so the KV term is exactly doubled. Mirrors the paged_branch_factor
    // engine_bootstrap sets for the same condition.
    vin.branch_factor = cfg.cascade ? 1 : 2;

    const VramBudget budget = plan_vram_budget(vin);
    log_vram_budget(budget);
    if (!budget.fits) {
        // FAIL CLOSED. Not even the minimum context is affordable, so there is no
        // smaller number to fall back to -- proceeding would either OOM inside a
        // constructor after a multi-GB load or, on WDDM, succeed into a paging
        // collapse. Thrown rather than exit()'d so main's handler gives the user a
        // message box with the reason in it.
        throw std::runtime_error("VRAM budget: " + budget.failure);
    }
    granted_context_ = budget.granted_context;

    // ISOLATED SESSIONS ARE A LEGACY-PATH NEED, not a general one. They exist
    // because Mode A runs TWO jobs on one engine -- transcribe the speech, then
    // answer it -- and a shared linear context let the transcription inherit the
    // assistant persona. A cascade has only ONE job for the backbone (answer),
    // because whisper.cpp did the transcribing outside the engine entirely. So
    // there is no second sequence to isolate, and asking for branching would only
    // inflate the paged host-mirror pool for a branch nothing forks.
    //
    // granted_context_, NOT settings.max_context: the guard above may have clamped
    // it. The persisted setting is deliberately left alone -- a budget is a
    // property of THIS machine on THIS launch, and writing it back would silently
    // lower the user's ceiling forever the first time they ran with a browser open.
    // ---- THE LAZY BRANCH ----------------------------------------------------
    // Booting on the cloud leg means no local model is going to answer anything,
    // so nothing is read from disk and no VRAM is committed: the control is
    // constructed DETACHED and the residency starts UNLOADED. Turning local
    // inference on later runs load_engine_into() against this same control (see
    // EngineResidency's cold-load path), which is why the control is created here
    // rather than at load time -- its identity is what AppContext, the speech
    // mode, the router and the transport all borrow.
    //
    // NOT ON THE LEGACY PIPELINE. There the backbone is also the speech
    // recogniser, so deferring its weights produces an app that cannot hear
    // rather than one that starts faster. Same capability gate the unload path
    // applies, arrived at from the other direction.
    lazy_ = !cfg.local_inference && cfg.cascade;
    if (lazy_) {
        real_stack_ = make_detached_real_control(cfg.args, granted_context_);
        control_ = real_stack_.bridge();
        std::printf("[engine] remote-only launch: no weights loaded, no KV pool allocated "
                    "(turn on local inference in Settings to load them)\n");
        std::fflush(stdout);
        return;
    }

    real_stack_ = bring_up_real_engine(cfg.args, dsp, granted_context_, cfg.settings.device_id,
                                       /*arm_streaming_plan=*/true,
                                       /*isolated_sessions=*/!cfg.cascade,
                                       /*load_audio_head=*/!cfg.cascade);
    control_ = real_stack_.bridge();
    if (!cfg.cascade) {
        // Seed the transcription session's prefix source BEFORE the engine thread
        // freezes it: prefill_system_prompt lays BOTH prefixes and reads this one
        // from the control. Both halves of that prefix -- the task text and the
        // forced spoken language -- are composed into one system prompt for seq 1.
        real_stack_.control->set_audio_task_prompt(cfg.settings.audio_task_prompt);
        real_stack_.control->set_speech_language(cfg.settings.speech_language);
    }
}

void AppLifecycleManager::bring_up_vad([[maybe_unused]] const Config& cfg) {
#if defined(VOICE_ASSISTANT_HAS_SILERO)
    if (cfg.args.neural_vad && !cfg.args.vad_model.empty()) {
        try {
            neural_vad_ = std::make_unique<blackwell::vad::SileroVAD>(cfg.args.vad_model);
            neural_vad_->set_threshold(cfg.args.vad_threshold);
            vad_fn_ = &vad_score;
            vad_ctx_.vad = neural_vad_.get();
            vad_user_ = &vad_ctx_;
            std::printf("[vad] Silero neural VAD armed (threshold %.2f)\n",
                        cfg.args.vad_threshold);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[vad] WARN: %s -- falling back to the RMS detector\n",
                         e.what());
        }
    }
#endif
    if (vad_fn_ == nullptr) std::printf("[vad] built-in RMS threshold detector\n");
}

// ---- MODE B: recognised, and refused ---------------------------------------
// ISpeechMode names three modes; this binary builds two. Mode B (SIMULTANEOUS --
// continuous rolling re-translation over SpeechSegmenter -> AbsoluteAudioRing ->
// RetranslationSession, docs/CONTINUOUS_STREAMING.md) is built into
// audio_translator and is not linked here at all.
//
// AN EXPLICIT THROW, not a silent clamp and not a dead branch. The previous
// arrangement had neither a Mode B case nor a rejection: clamp_settings reverted
// the name to legacy indistinguishably from a typo, and every reader of
// ISpeechMode was left to work out from the absence of a construction site that
// one third of the documented interface is unreachable in this executable. Both
// are worse than a sentence saying so.
//
// Reached only via a hand-edited settings.json or `--pipeline simultaneous`;
// clamp_settings lets the name through precisely so this can answer it.
void AppLifecycleManager::refuse_unsupported_mode(const Config& cfg) {
    if (cfg.settings.pipeline_mode == "simultaneous") {
        throw std::runtime_error(
            "Mode B unsupported in voice_assistant binary. Mode B (simultaneous "
            "re-translation) is a continuous re-drafting pipeline with no commit gate and "
            "no bot voice, which is the opposite of this app's listen -> pause -> answer "
            "loop; it ships in audio_translator. Set Settings -> Pipeline to "
            "ultravox_legacy or whisper_cascade.");
    }
}

void AppLifecycleManager::bring_up_speech_mode(const Config& cfg, AppContext& ctx) {
    // prefill_system_prompt lives on the CONCRETE control, not the bridge (how a
    // frozen prefix is laid down genuinely differs per backend), so it arrives as a
    // callable rather than a virtual -- the same seam audio_translator uses.
    ConversationalMode::PrefillSystemPromptFn prefill;
    if (real_stack_.control) {
        auto* rc = real_stack_.control.get();
        prefill = [rc](const std::string& p) { return rc->prefill_system_prompt(p); };
    } else {
        auto* sc = simulated_.get();
        prefill = [sc](const std::string& p) { return sc->prefill_system_prompt(p); };
    }

#if defined(VOICE_ASSISTANT_HAS_WHISPER)
    if (cfg.cascade) {
        if (cfg.whisper_model.empty() || !path_exists(cfg.whisper_model)) {
            // REFUSED, not degraded. Falling back to the legacy path here would
            // hand the user a working-looking assistant running a pipeline they did
            // not select, on a model they did not choose.
            throw std::runtime_error(
                "cascade mode needs a GGML Whisper model, and none was found. Set "
                "Settings -> Whisper model (or -DBLACKWELL_WHISPER_MODEL at configure "
                "time, or drop the .bin next to the exe). Looked at: '" +
                (cfg.whisper_model.empty() ? std::string("(nothing configured)")
                                           : cfg.whisper_model) +
                "'");
        }

        WhisperAsrConfig acfg;
        acfg.model_path = cfg.whisper_model;
        acfg.language   = cfg.settings.whisper_language;
        acfg.n_threads  = cfg.settings.whisper_threads;
        acfg.use_gpu    = true;   // forced false at compile time on a CPU build
        acfg.gpu_device = cfg.settings.device_id;

        WhisperCascadeMode::Config ccfg;
        ccfg.sample_rate      = static_cast<int>(cfg.sample_rate);
        ccfg.onset_threshold  = cfg.settings.vad_threshold;
        ccfg.hangover_ms      = cfg.settings.silence_hangover_ms;
        ccfg.preroll_ms       = cfg.settings.pre_roll_ms;
        ccfg.max_utterance_ms = cfg.settings.whisper_max_utterance_ms;
        // Composed, like Mode A's below: this mode's ONE frozen prefix is the
        // persona, so it is also where the output contract has to land.
        ccfg.system_prompt = compose_system_prompt(cfg.settings.system_prompt);

        // The transcript IS the user's bubble, exactly as Mode A's token stream is
        // -- so it lands on the same two view calls, in the same order. The
        // difference is that it arrives whole rather than token by token, and that
        // the gate's verdict comes with it instead of being read back off the
        // control (which never ran a decode loop here).
        //
        // TWO EDGES, NOT ONE, and the mode fires them either side of the commit
        // gate on purpose: on_text must reach the page before the offer wakes the
        // dispatcher, or the assistant's bubble is created first and the answer
        // renders above the question. See the note over WhisperCascadeMode::publish().
        WhisperCascadeMode::TranscriptCallbacks on_transcript;
        AssistantView* view = ctx.view;
        on_transcript.on_text = [view](const std::string& text, std::uint32_t id) {
            view->on_local_token(text.c_str(), id);
        };
        on_transcript.on_verdict = [view](std::uint32_t /*id*/, bool committed) {
            view->on_local_final(committed ? blackwell::bridge::TerminationReason::Eos
                                           : blackwell::bridge::TerminationReason::None);
        };

        cascade_mode_ = std::make_unique<WhisperCascadeMode>(
            control_, prefill, vad_fn_, vad_user_, acfg, ccfg, &on_state,
            std::move(on_transcript), &ctx);
        active_ = cascade_mode_.get();
        return;
    }
#endif

    ConversationalMode::Config mcfg;
    mcfg.sample_rate = cfg.sample_rate;
    mcfg.vad_probability_threshold = cfg.settings.vad_threshold;
    mcfg.silence_hangover_ms =
        static_cast<std::uint32_t>(cfg.settings.silence_hangover_ms);
    mcfg.warm_prefill_interval_ms =
        static_cast<std::uint32_t>(cfg.settings.warm_prefill_interval_ms);
    mcfg.pre_roll_ms = cfg.settings.pre_roll_ms;
    mcfg.system_prompt = compose_system_prompt(cfg.settings.system_prompt);

    conv_mode_ = std::make_unique<ConversationalMode>(control_, prefill, vad_fn_, vad_user_,
                                                      mcfg, &on_token, &on_state, &ctx);
    active_ = conv_mode_.get();
}

void AppLifecycleManager::start_engine_thread() {
    const int  device_id = device_id_;
    const bool real = use_real_;
    const bool lazy = lazy_;
    ISpeechMode* active = active_;
    engine_thread_ = std::thread([this, device_id, real, lazy, active] {
        try {
            // CUDA's current device is PER-THREAD. This thread launches every
            // kernel, so it must select the same device the weights were allocated
            // on -- doing it only in main() would silently run kernels on device 0
            // against device-N memory.
            //
            // SKIPPED on a lazy launch: nothing has been allocated yet, and
            // creating a CUDA context here would commit the very VRAM the
            // remote-only path exists to avoid. The cold load selects the device
            // itself (load_engine_into calls select_cuda_device) and it runs on
            // this same thread.
            if (real && !lazy) select_cuda_device(device_id);
            // THE STARTUP PREFILL, and the reason it is conditional. Freezing a
            // system prefix is a forward pass; with no engine attached it would
            // reach an arena that does not exist. On a lazy launch the residency
            // runs this exact call after the cold load instead (see
            // enable_cold_load's after_load).
            if (!lazy) {
                active->start_on_engine_thread();   // INIT tier: throws
            } else {
                std::printf("[system-prefix] deferred -- no local model loaded yet\n");
                std::fflush(stdout);
            }
            prefilled_.store(true, std::memory_order_release);
            while (running_.load(std::memory_order_acquire)) {
                active->pump_engine();   // 0% CPU until a boundary event arrives
            }
        } catch (const std::exception& e) {
            // A throw here used to hang main() forever on the `prefilled` spin.
            // Publish the failure, THEN release the latch.
            engine_error_ = e.what();
            engine_failed_.store(true, std::memory_order_release);
            prefilled_.store(true, std::memory_order_release);
        }
    });

    while (!prefilled_.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (engine_failed_.load(std::memory_order_acquire)) {
        shutdown();
        throw std::runtime_error("engine thread failed during startup: " + engine_error_);
    }
    std::printf("[system-prefix] frozen %u tokens (KV rewind floor)\n",
                frozen_prefix_tokens());
}

void AppLifecycleManager::shutdown() noexcept {
    if (stopped_) return;
    stopped_ = true;
    running_.store(false, std::memory_order_release);
    if (active_ != nullptr) active_->stop();
    if (engine_thread_.joinable()) engine_thread_.join();
}

std::uint32_t AppLifecycleManager::frozen_prefix_tokens() const {
#if defined(VOICE_ASSISTANT_HAS_WHISPER)
    if (cascade_mode_) return cascade_mode_->frozen_prefix_tokens();
#endif
    return conv_mode_ ? conv_mode_->frozen_prefix_tokens() : 0u;
}

void AppLifecycleManager::print_shutdown_summary() const {
#if defined(VOICE_ASSISTANT_HAS_WHISPER)
    // The cascade's own stage, reported separately for the same reason the speech
    // output summary is: it answers a DIFFERENT question. The commit gate's numbers
    // say whether transcripts became intents; these say whether speech became
    // transcripts at all -- and when the assistant "did not hear" something, this is
    // the block that says which.
    if (!cascade_mode_) return;
    std::printf("\n=== cascade ASR summary ===\n");
    std::printf("  transcripts published  : %llu\n",
                static_cast<unsigned long long>(cascade_mode_->published()));
    if (const WhisperAsr* a = cascade_mode_->asr(); a != nullptr) {
        std::printf("  utterances transcribed : %llu\n",
                    static_cast<unsigned long long>(a->utterances()));
        std::printf("  encode failures        : %llu%s\n",
                    static_cast<unsigned long long>(a->failures()),
                    a->failures() > 0 ? "   <-- VRAM or a bad model" : "");
        std::printf("  last encode            : %.1f ms (%s)\n", a->last_encode_ms(),
                    a->gpu() ? "GPU" : "CPU");
    }
    std::printf("  dropped (ASR behind)   : %llu%s\n",
                static_cast<unsigned long long>(cascade_mode_->dropped_overrun()),
                cascade_mode_->dropped_overrun() > 0
                    ? "   <-- the encode is slower than the speaker"
                    : "");
#endif
}

}  // namespace rt
