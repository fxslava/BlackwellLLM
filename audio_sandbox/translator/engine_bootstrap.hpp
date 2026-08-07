#pragma once
// -----------------------------------------------------------------------------
// engine_bootstrap.hpp — the REAL GPU stack, brought up once and owned in one
// place: CUDA device -> tokenizer -> BlackwellEngine -> RealEngineControl ->
// audio head (Whisper encoder + Ultravox projector) -> live DSP.
//
// WHY THIS FILE EXISTS. audio_translator and voice_assistant both need exactly
// this sequence, and it is the kind of code that rots the instant it is copied:
// the dimension validation must happen BEFORE any CUDA allocation, the engine
// must outlive the control that points at it, and the audio head is
// non-fatal-optional. Two divergent copies of those rules is how one app ends up
// validating and the other segfaulting.
//
// OWNERSHIP AND DESTRUCTION ORDER (the reason this is a struct, not a tuple):
// members are declared engine-before-control, so the compiler-generated
// destructor tears down control-then-engine. RealEngineControl holds a raw
// BlackwellEngine* and a raw ITokenizer*; reversing these two declarations is a
// use-after-free at shutdown that no test would catch. Same rule as
// BlackwellEngine::Impl (CLAUDE.md extension pattern #3).
//
// ERROR TIER: INIT. Everything here throws on failure (bad checkpoint, OOM,
// dimension mismatch) and the caller reports and exits. The one deliberate
// exception is the audio head: a missing/failed audio frontend leaves the app in
// text-only mode with a warning, exactly as audio_translator has always done,
// because a 15 GB download being absent should not take the binary down.
// -----------------------------------------------------------------------------
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include "blackwell/engine.h"
#include "blackwell/tokenizer.h"

#include "cli_config.hpp"           // rt::TranslatorArgs, parse/validate helpers
#include "real_engine_control.hpp"  // rt::RealEngineControl
#include "whisper_dsp.h"            // whisper::WhisperDSP

namespace rt {

// Everything the real pipeline needs to stay alive for the session.
struct RealEngineStack {
    // DECLARATION ORDER IS DESTRUCTION ORDER (reversed). Do not reorder.
    std::unique_ptr<blackwell::ITokenizer> tokenizer;  // outlives control
    std::unique_ptr<BlackwellEngine>       engine;     // outlives control
    std::unique_ptr<RealEngineControl>     control;    // destroyed FIRST

    BackboneConfig  backbone{};
    // Left at its defaults when the audio head was not requested (cascade mode):
    // nothing resolved it, so nothing may read it. audio_head_ready is the flag
    // that says whether it means anything.
    ProjectorParams projector{};
    bool            audio_head_ready = false;
    // Whether the two-sequence topology actually came up (see
    // bring_up_real_engine's isolated_sessions). False means every task shares
    // one linear context, the pre-isolation behaviour.
    bool            isolated_sessions = false;

    // The control, as the base type ConversationalMode and the UI hold. Never
    // null once bring_up_real_engine() returns.
    blackwell::bridge::EngineControlBridge* bridge() const noexcept { return control.get(); }
};

// Select the CUDA device for the CALLING thread.
//
// CUDA's current device is PER-THREAD, so this must be called on every thread
// that touches the engine -- the bootstrap thread that allocates, and the engine
// thread that decodes. Calling it once in main() and forgetting the engine
// thread is the classic version of this bug: allocations land on device N,
// kernels launch on device 0, and it only reproduces on a multi-GPU box.
inline void select_cuda_device(int device_id) {
    const cudaError_t rc = cudaSetDevice(device_id);
    if (rc != cudaSuccess) {
        throw std::runtime_error("cudaSetDevice(" + std::to_string(device_id) +
                                 ") failed: " + cudaGetErrorString(rc));
    }
}

// ---------------------------------------------------------------------------
// VRAM accounting. MEASURED, not estimated -- cudaMemGetInfo reports what the
// driver has actually committed, including allocations this process did not
// make (the desktop compositor, other apps) and the CUDA context itself.
//
// WHY THIS IS PRINTED RATHER THAN JUST BUDGETED. On Windows/WDDM, exceeding
// VRAM does NOT fail an allocation: the driver silently migrates pages to
// system RAM over PCIe. Throughput collapses by an order of magnitude and
// nothing anywhere reports an error -- the app just appears to freeze. The
// only way to catch it is to watch the committed total cross the line, which
// is what these lines are for. Compare against `nvidia-smi` "Memory-Usage";
// if Windows Task Manager shows "Shared GPU memory" above 0, paging has
// already started and the numbers below will say why.
// ---------------------------------------------------------------------------
inline void report_vram(const char* stage) {
    std::size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return;
    const double used_gb  = static_cast<double>(total_b - free_b) / (1024.0 * 1024.0 * 1024.0);
    const double total_gb = static_cast<double>(total_b) / (1024.0 * 1024.0 * 1024.0);
    std::printf("[vram] %-22s %5.2f / %5.2f GB committed (%.2f GB free)%s\n",
                stage, used_gb, total_gb,
                static_cast<double>(free_b) / (1024.0 * 1024.0 * 1024.0),
                // The practical ceiling is BELOW the nameplate: WDDM keeps a
                // reserve for the desktop, so paging begins before `free` hits
                // zero. 87.5% is where a 12 GB card starts spilling in practice.
                used_gb > 0.875 * total_gb ? "  <-- OVER BUDGET: WDDM paging likely" : "");
    std::fflush(stdout);
}

// Bring up the full GPU pipeline. `dsp` is borrowed and must outlive the stack
// (the control holds a pointer to it for the live commit path).
//
// max_context caps persistent KV growth across utterances; at 8B geometry the
// FP32 KV pool is ~256 KB/token, so 4096 tokens is ~1.05 GB on top of the
// ~5.3 GB AWQ weights.
//
// isolated_sessions asks for the TWO-SESSION topology (see
// RealEngineControl::enable_isolated_sessions): transcription and chat get their
// own engine sequence instead of sharing one linear context. That is a request
// for the engine's NATIVE CoW branching, and branching exists only under the
// paged KV cache -- ContinuousKVManager::supports_branching() is false and
// fork() throws there, so this flag necessarily switches the whole app from the
// continuous cache to the paged one. Non-fatal: a model that cannot branch
// (hybrid SSM) or a paged geometry the flash kernel cannot serve degrades to the
// single-session path with a warning, exactly like the audio head.
//
// load_audio_head=false is CASCADE MODE, and it is the reason that flag exists:
// whisper.cpp answers "what did the user say?" on its own, so the Ultravox
// encoder + projector are dead weight -- and expensive dead weight, because
// whisper_encoder.h carries its weights as `const float*` and inflates an fp16
// checkpoint 2x on upload. Skipping it is what pays for the ~1.6 GB GGML model
// the cascade loads instead.
//
// IT ALSO SKIPS THE PROJECTOR/BACKBONE DIMENSION CHECK, and that is not a
// shortcut -- it is the point. The projector's output width is fixed by the
// checkpoint Ultravox was trained against (4096, Llama-3.1-8B), so validating it
// against a DIFFERENT backbone is validating a tensor nothing will multiply.
// Cascade mode is precisely what makes a backbone of another width usable here
// (Qwen2.5-7B is 3584), and running the check anyway would reject it for a
// mismatch that has no consumer.
// ---------------------------------------------------------------------------
// THE DETACHED CONTROL — a RealEngineControl with no engine behind it.
//
// This is the whole of what a remote-only launch constructs: no tokenizer, no
// safetensors read, no KV pool, no arena, no CUDA context. What it buys is a
// STABLE control pointer for everything downstream to borrow (AppContext, the
// speech mode, the router, the transport) so that loading an engine LATER is an
// attach rather than a teardown and a re-pointing of five borrowers.
//
// The stack it returns has `control` set and `engine`/`tokenizer` null, which is
// exactly the state load_engine_into() fills in.
// ---------------------------------------------------------------------------
inline RealEngineStack make_detached_real_control(const TranslatorArgs& args, int max_context) {
    RealEngineStack st;
    st.control = std::make_unique<RealEngineControl>(/*engine=*/nullptr, /*tok=*/nullptr,
                                                     max_context);
    st.control->set_context_mode(args.context_mode == "bounded"
                                     ? RealEngineControl::ContextMode::BoundedHistory
                                     : RealEngineControl::ContextMode::Stateless);
    st.control->set_history_budget_tokens(args.history_budget_tokens);
    return st;
}

// Fill in a stack that already has a control: read the checkpoint, build the
// engine, ADOPT it into the existing control, then run the post-engine
// configuration (isolation, audio head). This is the expensive half, and it is
// the half a lazy launch defers.
//
// MUST run on the ENGINE THREAD when the control is already live (adopt_engine
// verifies it); at startup it runs on main's thread before the engine thread is
// spawned, which is the same thread by adoption.
//
// INIT tier: throws on a bad checkpoint.
inline void load_engine_into(RealEngineStack& st, const TranslatorArgs& args,
                             whisper::WhisperDSP& dsp, int max_context, int device_id,
                             bool arm_streaming_plan, bool isolated_sessions,
                             bool load_audio_head) {
    select_cuda_device(device_id);
    {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, device_id) == cudaSuccess) {
            std::printf("[cuda] device %d: %s (sm_%d%d, %.1f GB)\n", device_id, prop.name,
                        prop.major, prop.minor,
                        static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));
        }
    }

    // Validate BEFORE any CUDA allocation: a projector/backbone width mismatch
    // must abort while it is still cheap, not after a 5.3 GB load.
    st.backbone = parse_backbone_config(args.model_dir);
    if (load_audio_head) {
        st.projector = resolve_projector_params(args.projector_path);
        validate_dimensions(st.projector, st.backbone, args.model_dir);
        std::printf("[validated] projector.output_dim == backbone.hidden_size (%d)\n",
                    st.backbone.hidden_size);
    } else {
        // Say it out loud. "The audio head is not loaded" is the single fact that
        // explains both the missing [audio] lines below and the ~2 GB the [vram]
        // lines do not show, and someone will otherwise read that gap as a bug.
        std::printf("[audio] CASCADE mode: the Ultravox audio head is NOT loaded "
                    "(no encoder, no projector, no dimension check)\n");
    }

    std::printf("[engine] loading tokenizer + backbone from %s ...\n", args.model_dir.c_str());
    std::fflush(stdout);
    st.tokenizer = blackwell::TokenizerFactory::create(args.model_dir);

    blackwell::InferenceConfig req;
    req.max_context_length = static_cast<size_t>(max_context);
    apply_streaming_flags(args, req);
    // The live products arm the streaming plan unconditionally so the UI's mode
    // toggle is functional and a commit can never silently fall back to the
    // multi-second whole-utterance re-prefill just because a launch flag was
    // missing. --stream-mode still picks the INITIAL mode.
    req.audio_streaming.enable = arm_streaming_plan;
    req.source_language = args.src_lang;
    req.target_language = args.tgt_lang;
    // Tier-2 intent, not a low-level poke: "I will fork sequences", which
    // build_and_validate_runtime resolves to KVCacheMode::Paged (CLAUDE.md
    // extension pattern #2). Rejected here, before any VRAM is touched, for a
    // model whose recurrent state cannot be snapshot.
    req.require_branching = isolated_sessions;

    // Exactly the two sequences this app runs (chat + transcription). The
    // default of 4 would inflate the paged host-mirror pool -- and the hybrid
    // state stores -- by 2x for branches we never fork.
    blackwell::RuntimeOverrides overrides;
    if (isolated_sessions) overrides.paged_branch_factor = 2;

    // The branching request is best-effort: a checkpoint the paged path cannot
    // serve must still boot the app, single-session, rather than take the binary
    // down over a context-isolation optimization.
    try {
        st.engine = std::make_unique<BlackwellEngine>(
            args.model_dir + "/model.safetensors.index.json", req, overrides);
    } catch (const std::exception& e) {
        if (!isolated_sessions) throw;
        std::fprintf(stderr,
                     "[session] WARN: paged (branching) KV unavailable for this model (%s)"
                     " -- falling back to a SINGLE shared context\n", e.what());
        req.require_branching = false;
        st.engine = std::make_unique<BlackwellEngine>(
            args.model_dir + "/model.safetensors.index.json", req,
            blackwell::RuntimeOverrides{});
    }
    std::printf("[engine] loaded (hidden=%d vocab=%d quant=%s)\n", st.backbone.hidden_size,
                st.backbone.vocab_size, st.backbone.quant_method.c_str());
    // Weights + KV pool together. The KV half is the part max_context controls:
    // the paged cache is bf16, so it costs
    //     num_layers * num_kv_heads * head_dim * 2 (K,V) * 2 B * max_context
    // per sequence -- 128 KB/token at 8B geometry, times paged_branch_factor.
    // 4096 tokens over 2 branches is ~1.05 GB; halving max_context halves it.
    report_vram("backbone + KV");

    // ADOPTED, not constructed. The control already exists -- either from
    // make_detached_real_control() on the lazy path, or from the line in
    // bring_up_real_engine() that creates one before calling this -- and its
    // identity is what every downstream borrower holds. adopt_engine() re-seeds
    // the language and streaming atomics off the resolved plan, which the
    // constructor would otherwise have done.
    if (!st.control->adopt_engine(st.engine.get(), st.tokenizer.get())) {
        throw std::runtime_error(
            "could not bind the loaded engine to the control (already bound, or called off "
            "the engine thread)");
    }

    // Fork the ephemeral transcription sequence off the (still empty) root, so
    // the two prompts never share a KV prefix. Capability-gated inside; reports
    // what it actually got, because "isolated" silently degrading to "shared" is
    // precisely the context poisoning this exists to prevent.
    if (isolated_sessions) {
        st.isolated_sessions = st.control->enable_isolated_sessions();
        std::printf("[session] context isolation: %s (kv=%s, branch capacity=%d)\n",
                    st.isolated_sessions ? "ON -- chat=seq 0, transcription=seq 1"
                                         : "OFF -- single shared context",
                    st.engine->get_capabilities().supports_cow_branching ? "paged" : "continuous",
                    st.engine->branch_capacity());
        std::fflush(stdout);
    }

    // Audio head: NON-FATAL by design (see the header preamble), and SKIPPED
    // ENTIRELY in cascade mode -- where its absence is the plan, not a
    // degradation, so it must not print the text-only warning below.
    if (!load_audio_head) {
        report_vram("cascade (no audio head)");
        return;
    }
    std::printf("[audio] loading audio head from %s ...\n", args.audio_head.c_str());
    std::fflush(stdout);
    try {
        st.control->load_audio_head(args.audio_head);
        st.control->set_dsp(&dsp);
        st.audio_head_ready = true;
        std::printf("[audio] encoder + projector loaded (%d soft-tokens/frame); "
                    "live audio->text ARMED\n", st.control->audio_out_frames());
        // The audio head is the LARGEST single reducible item in the budget and
        // the least obvious: src/audio/whisper_encoder.h carries its weights as
        // `const float*` throughout, so an fp16 Ultravox checkpoint is inflated
        // 2x on upload. Whatever this line reports is roughly twice what the
        // same weights would cost in fp16 -- see the note in main.cpp.
        report_vram("+ audio head");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[audio] WARN: audio head not loaded (%s) -- text-only mode\n",
                     e.what());
    }
}

// The eager path, unchanged for every existing caller: build the control and
// load the engine into it, in one call. Composed from the two halves above so
// there is exactly ONE definition of what a loaded stack looks like -- a lazy
// launch that later loads must arrive at the same configuration as an eager one,
// and two copies of this sequence is how the two would drift.
inline RealEngineStack bring_up_real_engine(const TranslatorArgs& args, whisper::WhisperDSP& dsp,
                                            int max_context, int device_id,
                                            bool arm_streaming_plan,
                                            bool isolated_sessions = false,
                                            bool load_audio_head = true) {
    RealEngineStack st = make_detached_real_control(args, max_context);
    load_engine_into(st, args, dsp, max_context, device_id, arm_streaming_plan,
                     isolated_sessions, load_audio_head);
    return st;
}

}  // namespace rt
