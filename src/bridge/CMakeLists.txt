# =============================================================================
# blackwell_bridge — Application-to-Inference bridge (SPSC audio ring, prompt
# injector splice kernel, C-ABI). Depends only on the engine's white-box surface
# (common.h, device_buffer.h, engine_status.h) + CUDA — NOT on the CUDA decode
# loop; the real engine plugs in behind IEngineControl (bridge_internal.hpp).
#
# Explicit source list (src/kernels/ convention; never file(GLOB)). No exported
# src/ include path — consumers declare their own PRIVATE includes.
# =============================================================================
add_library(blackwell_bridge STATIC
    audio_ring_buffer.cpp             # SPSC lock-free ring (pinned+mapped)
    prompt_injector.cu                # splice_audio_embeddings_kernel + launcher
    engine_api.cpp                    # C-ABI wrappers + exception/status translation
    speech_pipeline_controller.cpp)   # speculative speech state machine + C-ABI

# Public bridge headers in include/bridge; engine white-box deps are PRIVATE.
target_include_directories(blackwell_bridge
    PUBLIC  ${CMAKE_SOURCE_DIR}/include
    PRIVATE ${CMAKE_SOURCE_DIR}/src         # common.h
            ${CMAKE_SOURCE_DIR}/src/core    # device_buffer.h
            ${CMAKE_CURRENT_SOURCE_DIR})    # bridge_internal.hpp

# Mirror src/kernels/ + src/audio/ CUDA flags (fast-math + O3 in Release, -G Debug).
target_compile_options(blackwell_bridge PRIVATE
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

# Wired into the engine build via `add_subdirectory(bridge)` in src/CMakeLists.txt.
# It builds as a leaf static lib (nothing links it yet); the engine assembly will
# link it and provide the IEngineControl implementor (bridge_internal.hpp) that
# closes the seam.
