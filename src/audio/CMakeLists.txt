# =============================================================================
# blackwell_audio — Ultravox multimodal frontend (preprocessor only).
# Shipping target: fixie-ai/ultravox-v0_5-llama-3_1-8b (projector text_hidden 4096).
# The v0_5-llama-3_2-1b variant (text_hidden 2048) is retained by the parity tests.
#
# ACTIVATED (Phase 0): the first real sources now exist, so this builds as a leaf
# static library. Nothing in the engine links it yet; the parity tests
# (tests/*ultravox*/*prompt_injector*) and, later, the Llama drop-in stage do.
#
# Follows the src/kernels/ convention: an EXPLICIT source list (never file(GLOB),
# which is evaluated only at configure time and silently drops new kernels), and
# NO exported include paths (consumers declare their own PRIVATE src/ includes;
# see skill cmake-hygiene). The projector pipeline is intentionally SELF-CONTAINED
# (its own FP32 RMSNorm / Linear / SwiGLU / stack kernels) to match the FP32
# PyTorch golden reference bit-for-bit, so it does NOT link blackwell_kernels.
# =============================================================================

# --- Audio-frontend CUDA sources (explicit; add each new .cu here) ------------
add_library(blackwell_audio STATIC
    ultravox_projector.cu               # StackAudioFrames + SwiGLU launchers
    ultravox_projector_pipeline.cu      # UltravoxProjector: 6-stage FP32 graph
    prompt_injector.cu)                 # inject_audio_embeddings splice kernel

# Device/engine deps are PRIVATE; no include/ public header exists yet, so
# consumers of the *.cuh declare PRIVATE ${CMAKE_SOURCE_DIR}/src/audio themselves.
target_include_directories(blackwell_audio PRIVATE
    ${CMAKE_SOURCE_DIR}/src             # common.h
    ${CMAKE_SOURCE_DIR}/src/core        # device_buffer.h
    ${CMAKE_CURRENT_SOURCE_DIR})        # *.cuh

# Mirror src/kernels/ CUDA flags (fast-math + O3 in Release, -G in Debug).
target_compile_options(blackwell_audio PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:
        --use_fast_math
        -Xptxas=-O3,-v
        -Xcompiler=/O2,/Ot,/Zc:preprocessor
        -DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
    >
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:
        -g -G
        -Xcompiler=/Zi,/Od,/Zc:preprocessor
        -DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
    >)
