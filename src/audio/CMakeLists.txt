# =============================================================================
# blackwell_audio — Ultravox multimodal frontend (preprocessor only).
# Target: fixie-ai/ultravox-v0_5-llama-3_2-1b.
#
# STATUS: SCAFFOLD. This file is intentionally NOT yet referenced by an
# add_subdirectory(audio) in src/CMakeLists.txt. Activate it in Phase 0 (see
# ULTRAVOX_AUDIO_PLAN.md) once the first real source file below exists — a target
# with zero sources fails to configure, so we keep the wiring out of the build
# until there is something to compile.
#
# Follows the src/kernels/ convention: an EXPLICIT source list (never file(GLOB),
# which is evaluated only at configure time and silently drops new kernels), and
# NO exported include paths (consumers declare their own PRIVATE src/ includes;
# see skill cmake-hygiene). Links blackwell_kernels to REUSE stateless text
# kernels (RMSNorm / SwiGLU / batched GEMM) — reuse by calling, never modifying.
# =============================================================================

# --- CUDA kernels (audio-only; independent of src/kernels/) -------------------
# Uncomment each entry as the matching .cu lands (ULTRAVOX_AUDIO_PLAN.md §4).
set(AUDIO_KERNEL_SOURCES
    # kernels/log_mel.cu            # Phase 1: STFT power -> mel filterbank -> log10  (FIRST KERNEL)
    # kernels/conv_subsample.cu     # Phase 2: conv1(k3,s1) + conv2(k3,s2) + GELU (time /2)
    # kernels/layer_norm.cu         # Phase 2: true LayerNorm (Whisper) -- NOT rmsnorm
    # kernels/audio_attention.cu    # Phase 3: bidirectional (non-causal) MHA, 20 heads
    # kernels/encoder_mlp.cu        # Phase 3: fc1 -> GELU -> fc2 (GELU epilogue over reused GEMM)
    # kernels/stack_audio_frames.cu # Phase 4: pad %8, reshape [T,1280]->[T',10240]
    # kernels/projector.cu          # Phase 4: swiglu-chunk (ln/linear reuse text kernels)
)

# --- Host orchestration -------------------------------------------------------
set(AUDIO_HOST_SOURCES
    # audio_weight_loader.cpp       # Phase 0/1: checkpoint -> geometry + device weights
    # whisper_encoder.cpp           # Phase 3: 32-layer bidirectional encoder
    # ultravox_projector.cpp        # Phase 4: per-text-model projector adapter
    # audio_frontend.cpp            # Phase 5: PIMPL over include/audio/audio_frontend.h
)

if(AUDIO_KERNEL_SOURCES OR AUDIO_HOST_SOURCES)
    add_library(blackwell_audio STATIC ${AUDIO_KERNEL_SOURCES} ${AUDIO_HOST_SOURCES})

    # Public contract header lives in include/audio; device/engine deps are PRIVATE.
    target_include_directories(blackwell_audio
        PUBLIC  ${CMAKE_SOURCE_DIR}/include
        PRIVATE ${CMAKE_SOURCE_DIR}/src        # common.h, device_buffer.h via core
                ${CMAKE_SOURCE_DIR}/src/core
                ${CMAKE_CURRENT_SOURCE_DIR})

    # Reuse stateless text kernels (RMSNorm / SwiGLU / batched GEMM) by CALLING
    # them from the projector + encoder MLP — never modifying src/kernels/.
    target_link_libraries(blackwell_audio PRIVATE blackwell_kernels)

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
else()
    message(STATUS "blackwell_audio: no sources yet (scaffold) — target not created")
endif()
