# =============================================================================
# blackwell_vad — Silero neural voice-activity detection (ONNXRuntime CPU).
#
# Deliberately CUDA-FREE and engine-free: it depends on nothing but ONNXRuntime
# and the STL, so linking it never drags the inference stack into a consumer.
# The pipeline consumes it through the bridge's VAD-scorer seam (a plain
# function pointer), NOT by linking it into blackwell_bridge — that keeps the
# C-ABI control plane free of a 15 MB runtime dependency that most of its
# consumers (poc_overlay, the engine apps, the bridge tests) never use.
#
# Explicit source list (src/kernels/ convention; never file(GLOB)).
#
# Gated on the ONNXRuntime target existing: with -DUSE_SILERO_VAD=OFF this
# directory produces no target at all, and every consumer's if(TARGET
# blackwell_vad) guard falls through to the built-in RMS-threshold VAD.
# =============================================================================
if(NOT TARGET blackwell_onnxruntime)
    return()
endif()

add_library(blackwell_vad STATIC silero_vad.cpp)

# silero_vad.hpp is the whole public surface; ONNXRuntime stays PRIVATE so the
# PIMPL actually buys something — a consumer of this target gets neither ORT's
# headers nor its include path.
target_include_directories(blackwell_vad PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(blackwell_vad PRIVATE blackwell::onnxruntime)
