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
    ProjectorParams projector{};
    bool            audio_head_ready = false;

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

// Bring up the full GPU pipeline. `dsp` is borrowed and must outlive the stack
// (the control holds a pointer to it for the live commit path).
//
// max_context caps persistent KV growth across utterances; at 8B geometry the
// FP32 KV pool is ~256 KB/token, so 4096 tokens is ~1.05 GB on top of the
// ~5.3 GB AWQ weights.
inline RealEngineStack bring_up_real_engine(const TranslatorArgs& args, whisper::WhisperDSP& dsp,
                                            int max_context, int device_id,
                                            bool arm_streaming_plan) {
    RealEngineStack st;

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
    st.backbone  = parse_backbone_config(args.model_dir);
    st.projector = resolve_projector_params(args.projector_path);
    validate_dimensions(st.projector, st.backbone, args.model_dir);
    std::printf("[validated] projector.output_dim == backbone.hidden_size (%d)\n",
                st.backbone.hidden_size);

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

    st.engine = std::make_unique<BlackwellEngine>(
        args.model_dir + "/model.safetensors.index.json", req);
    std::printf("[engine] loaded (hidden=%d vocab=%d quant=%s)\n", st.backbone.hidden_size,
                st.backbone.vocab_size, st.backbone.quant_method.c_str());

    st.control = std::make_unique<RealEngineControl>(st.engine.get(), st.tokenizer.get(),
                                                     max_context);
    st.control->set_context_mode(args.context_mode == "bounded"
                                     ? RealEngineControl::ContextMode::BoundedHistory
                                     : RealEngineControl::ContextMode::Stateless);
    st.control->set_history_budget_tokens(args.history_budget_tokens);

    // Audio head: NON-FATAL by design (see the header preamble).
    std::printf("[audio] loading audio head from %s ...\n", args.audio_head.c_str());
    std::fflush(stdout);
    try {
        st.control->load_audio_head(args.audio_head);
        st.control->set_dsp(&dsp);
        st.audio_head_ready = true;
        std::printf("[audio] encoder + projector loaded (%d soft-tokens/frame); "
                    "live audio->text ARMED\n", st.control->audio_out_frames());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[audio] WARN: audio head not loaded (%s) -- text-only mode\n",
                     e.what());
    }
    return st;
}

}  // namespace rt
