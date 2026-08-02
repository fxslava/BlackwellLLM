cmake_minimum_required(VERSION 3.28)
project(BlackwellLLM LANGUAGES CXX CUDA)

# Group targets into Visual Studio Solution Explorer folders (see the FOLDER
# assignments at the bottom of this file). Also tucks the CMake-generated
# ALL_BUILD / ZERO_CHECK utility projects under "CMakePredefinedTargets".
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ============================================================================
# Static CRT (/MT, /MTd) -- enterprise-grade, dependency-free deployment.
# Set BEFORE any target or FetchContent dependency so googletest/nlohmann and
# every first-party target link the SAME runtime (a /MT<->/MD mismatch is a hard
# LNK2038). CMP0091 (NEW under cmake>=3.28) routes this through the abstract
# MSVC_RUNTIME_LIBRARY property instead of hand-patched CMAKE_*_FLAGS.
#
# Safe across the SHARED blackwell_core.dll boundary: the COM edge passes only
# HRESULTs + POD across the DLL seam (no CRT objects -- FILE*, malloc'd blocks,
# or C++ exceptions -- ever cross it), so the exe and the DLL each owning a
# private static CRT copy is by design (see the hybrid error doctrine).
# ============================================================================
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# ============================================================================
# CUDA AOT fatbinary matrix (deployment target coverage).
#   <arch>-real    -> native SASS baked into the fatbinary for that GPU.
#   <arch>-virtual -> PTX kept for forward JIT onto any newer, unlisted arch.
# Turing..Hopper get native SASS; 90-virtual PTX JITs forward onto future parts.
# 120-real is the project's PRIMARY target (Blackwell / RTX 50xx) -- keep native
# SASS so the latency-critical hot path never pays a first-launch PTX-JIT stall
# on the very hardware this engine exists for. (Drop 120-real ONLY for a build
# that will never run on Blackwell; the 90-virtual PTX would then cover it.)
#
# GUARDED so the CMakePresets cache var wins: the x64-release preset pins THIS
# matrix, while x64-debug stays single-arch (fast local iteration). This block is
# the fallback for a raw, preset-less `cmake -S . -B ...` and the canonical record
# of the shipping matrix. Keep it in lock-step with the release preset's
# CMAKE_CUDA_ARCHITECTURES.
#
# TEMPORARILY single-arch (120-real only): the current kernels use Blackwell/
# Hopper-specific instructions (sm_90+ intrinsics -- e.g. TMA / wgmma / cluster
# ops) that do NOT compile for older SASS, so building 75/86/89 fails outright.
# The broader matrix below is the INTENDED deployment target once those kernels
# grow pre-sm_90 fallbacks; re-enable arch-by-arch as each is ported and its
# build goes green. See the deployment matrix in deploy/README.md.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES
        # --- intended future deployment matrix (disabled: pre-sm_90 kernels do
        #     not yet compile -- Blackwell/Hopper-only intrinsics in use) ---
        # 75-real       # Turing   (RTX 20xx / GTX 16xx)
        # 86-real       # Ampere   (RTX 30xx, A-series)
        # 89-real       # Ada      (RTX 40xx)
        # 90-real       # Hopper   (H100 / H200)
        # 90-virtual    # PTX fallback: forward-compatible JIT for any arch > sm_90
        120-real      # Blackwell (RTX 50xx) -- primary target, native SASS
    )
endif()

# 2. Оптимизации и флаги
# Убираем -O3. В Visual Studio оптимизация включается выбором конфигурации "Release" в самой IDE.
# Добавляем быструю математику только для CUDA-файлов, чтобы не смущать MSVC:
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>")

# Включаем тестирование
enable_testing()

# 🎯 Опция для включения DirectStorage (по умолчанию ВЫКЛЮЧЕНА). Объявлена до
# включения модуля, который на неё смотрит.
option(USE_DIRECT_STORAGE "Enable DirectStorage for blazing fast weight loading on Windows" OFF)

# Silero neural VAD (ONNXRuntime CPU + the pinned silero_vad.onnx). ON by
# default: it is what the live speech pipeline's boundary detection runs on.
# OFF drops the ~75 MB ONNXRuntime fetch and builds the pipeline with only its
# built-in RMS-threshold VAD -- which stays the compiled-in default either way,
# so nothing downstream breaks. Declared before the module that reads it.
option(USE_SILERO_VAD "Enable the Silero neural VAD (fetches ONNXRuntime CPU + silero_vad.onnx)" ON)

# 3. Зависимости: pinned FetchContent + опциональный DirectStorage SDK
# (интерфейсный таргет blackwell::dstorage + blackwell_copy_dstorage_dlls())
# + ONNXRuntime CPU для нейросетевого VAD (blackwell::onnxruntime +
# blackwell_copy_onnxruntime_dlls() + blackwell_copy_vad_model())
# + общие хелперы (blackwell_copy_runtime_dlls).
include(cmake/BlackwellHelpers.cmake)
include(cmake/Dependencies.cmake)
include(cmake/DirectStorage.cmake)
include(cmake/OnnxRuntime.cmake)
# Chromium (Edge WebView2) embedding SDK -> blackwell::webview2. Declared here,
# once, so the two products that host a web UI (poc_overlay's settings window,
# voice_assistant's whole main view) cannot pin two different SDK versions
# depending on add_subdirectory order. Windows-only: gate on the target.
include(cmake/WebView2.cmake)

# ============================================================================
# Compiler hygiene: zero-warning policy on our first-party CXX code.
# Placed AFTER the dependency includes so already-defined FetchContent targets
# (googletest, ...) keep their own warning level -- this directory property
# only applies to targets created below (src/, tests/).
#   - CXX only: nvcc host-passthrough of /W4 /WX over CUDA/CCCL headers is not
#     viable, so CUDA (.cu) compilation is untouched.
#   - /external:anglebrackets + /external:W0: <>-includes (CUDA toolkit, STL,
#     Win32, deps) are "external" and drop to W0, so /W4 /WX fires ONLY on our
#     own "..."-included code.
# ============================================================================
if(MSVC)
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CXX>:/W4>"
        "$<$<COMPILE_LANGUAGE:CXX>:/WX>"
        "$<$<COMPILE_LANGUAGE:CXX>:/external:anglebrackets>"
        "$<$<COMPILE_LANGUAGE:CXX>:/external:W0>"
        # C4996 deprecates STANDARD C functions (getenv/fopen/...) in favour of
        # MSVC's non-portable *_s variants. This code is cross-platform-standard,
        # so silence that one non-warning (CXX only -- CUDA TUs are untouched, so
        # they never recompile for it) rather than un-portably rewriting it.
        "$<$<COMPILE_LANGUAGE:CXX>:/D_CRT_SECURE_NO_WARNINGS>"
    )
endif()

# ============================================================================
# Release optimization profile (enterprise build).
# Generator-expression gated on $<CONFIG:Release> so Debug/RelWithDebInfo are
# untouched. NDEBUG + the Release CRT flags come from CMake's own
# CMAKE_*_FLAGS_RELEASE; this block adds the AGGRESSIVE layer on top.
#   CXX  : /O2 max speed, /Ob3 aggressive inlining, /Oi intrinsics, /Ot favor
#          speed, /GL whole-program opt, /Gy COMDAT sections (feeds /OPT:ICF).
#   LINK : /LTCG cross-module codegen (pairs with /GL), /OPT:REF strips unused
#          code+data, /OPT:ICF folds identical COMDATs, /INCREMENTAL:NO (LTCG is
#          incompatible with incremental linking).
#   CUDA : /GL is deliberately NOT forwarded through nvcc (host-passthrough LTCG
#          is unsupported); device code is tuned via ptxas -O3 instead. Mixed
#          /GL (CXX) + non-/GL (CUDA) objects link cleanly -- LTCG applies to the
#          /GL subset only.
# ============================================================================
if(MSVC)
    add_compile_options(
        "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2>"
        "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/Ob3>"
        "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/Oi>"
        "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/Ot>"
        "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/GL>"
        "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/Gy>"
        # Device-side optimization for the CUDA translation units.
        "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>"
        "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-Xptxas=-O3>"
    )
    add_link_options(
        "$<$<CONFIG:Release>:/LTCG>"
        "$<$<CONFIG:Release>:/OPT:REF>"
        "$<$<CONFIG:Release>:/OPT:ICF>"
        "$<$<CONFIG:Release>:/INCREMENTAL:NO>"
    )
endif()

# Подключаем поддиректории
add_subdirectory(src)
add_subdirectory(tests)

# audio_sandbox — standalone Whisper log-mel front-end (parity tester + real-time
# visualiser). CUDA-free, no engine deps; its third-party libs (imgui, dr_libs,
# pocketfft, miniaudio) are the SHARED root targets from cmake/Dependencies.cmake.
# Added AFTER the /W4 /WX block so its first-party sources inherit the same
# zero-warning budget. Gated so an engine-only build can skip the GUI/audio libs.
option(BUILD_AUDIO_SANDBOX "Build the standalone Whisper log-mel audio_sandbox (parity + realtime GUI)" ON)
if(BUILD_AUDIO_SANDBOX)
    add_subdirectory(audio_sandbox)
endif()

# Streaming TTFT profiler wiring (tests/integration/test_streaming_ttft_profiler.cpp):
# it subclasses the translator's RealEngineControl, whose base lives in
# blackwell_bridge and whose virtuals reach WhisperDSP. Those targets are defined
# AFTER tests/, so the extra deps are attached HERE (same post-subdirectory
# pattern as the folder assignments below). Without the sandbox the test's TU
# compiles to nothing (it is gated on BLACKWELL_HAVE_SPEECH_STACK).
if(TARGET integration_tests AND TARGET blackwell_bridge AND TARGET whisper_dsp)
    target_link_libraries(integration_tests PRIVATE blackwell_bridge whisper_dsp)
    target_include_directories(integration_tests PRIVATE
        "${CMAKE_SOURCE_DIR}/src/bridge"              # engine_control_bridge.hpp, bridge_internal.hpp
        "${CMAKE_SOURCE_DIR}/audio_sandbox/include")  # whisper_dsp.h
    target_compile_definitions(integration_tests PRIVATE BLACKWELL_HAVE_SPEECH_STACK)
endif()

# ============================================================================
# Solution-folder assignments (USE_FOLDERS is ON above). Centralised here --
# after every add_subdirectory -- so all targets exist and the taxonomy lives in
# one readable place instead of scattered set_target_properties calls. Each is
# guarded by if(TARGET ...) because most subdirectories are option-gated.
# ============================================================================
function(blackwell_set_folder folder)
    foreach(tgt ${ARGN})
        if(TARGET ${tgt})
            set_target_properties(${tgt} PROPERTIES FOLDER "${folder}")
        endif()
    endforeach()
endfunction()

# --- Engine (CUDA inference stack) ---
blackwell_set_folder("Engine/Core"   blackwell_kernels blackwell_core_obj blackwell_core blackwell_audio blackwell_vad blackwell_tts)
blackwell_set_folder("Engine/Bridge" blackwell_bridge)
# --- Agent stack (CUDA-free) ---
blackwell_set_folder("Engine/Agent"  treesitter agent_core agent_env agent_orchestrator)
# --- Apps & tools ---
blackwell_set_folder("Apps"          blackwell_llm blackwell_bench agent_playground poc_overlay)
blackwell_set_folder("Experiments"   awq_benchmark)
# --- Audio sandbox ---
blackwell_set_folder("Audio"         whisper_dsp parity_check audio_realtime audio_translator)
# --- Tests ---
blackwell_set_folder("Tests"
    validation_tests benchmark_tests integration_tests
    bridge_tests kernel_unit_tests ultravox_model_integration_test vad_tests tts_tests
    agent_tests agent_env_tests agent_orchestrator_tests)
# --- Third-party (FetchContent) ---
blackwell_set_folder("ThirdParty"    imgui gtest gtest_main gmock gmock_main)