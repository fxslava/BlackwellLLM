# Test suite layout:
#   common/      - shared headers (CUDA fixtures, CPU references, problem
#                  generators, benchmark harness); no target of its own
#   validation/  - mathematical correctness vs CPU references on the exact
#                  Llama-FP8 / Qwen-AWQ target shapes  -> validation_tests
#   benchmarks/  - stress shapes (split-K extremes, ragged groups, non-aligned
#                  dims) and CUDA-event performance benchmarks -> benchmark_tests
#   integration/ - end-to-end engine vs PyTorch golden dumps (skips when the
#                  local model/dumps are absent)       -> integration_tests
#
# Typical invocations:
#   ctest -L validation     # fast correctness suite
#   ctest -L benchmark      # stress + performance suite
#   ctest -L integration    # needs the local checkpoint + golden dumps

include(GoogleTest)

# Defines one GTest executable from every .cpp under source_dir, wires up
# includes/links/runtime DLLs identically for each suite and registers the
# discovered tests in CTest under the given label.
function(blackwell_add_test_suite target source_dir label)
    # CONFIGURE_DEPENDS: newly added test files are picked up on the next
    # build without a manual CMake re-run.
    file(GLOB suite_sources CONFIGURE_DEPENDS "${source_dir}/*.cpp")

    add_executable(${target} ${suite_sources})

    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}    # common/...
        ${CMAKE_SOURCE_DIR}/src        # common.h, kernels/*.cuh
        ${CMAKE_SOURCE_DIR}/src/core   # engine_impl.h, memory_pool.h, paging/, ssm/
    )

    # White-box suites link the OBJECT library, NOT the DLL: they include
    # engine_impl.h / memory_pool.h and reach symbols the narrow COM boundary
    # of blackwell_core.dll deliberately does not export.
    target_link_libraries(${target} PRIVATE
        GTest::gtest
        GTest::gtest_main
        blackwell_kernels
        blackwell_core_obj
    )

    # No-op unless the core lib was built with DirectStorage (the helper is
    # defined unconditionally by cmake/DirectStorage.cmake).
    blackwell_copy_dstorage_dlls(${target})

    gtest_discover_tests(${target}
        PROPERTIES LABELS "${label}"
        DISCOVERY_TIMEOUT 120
    )
endfunction()

blackwell_add_test_suite(validation_tests  "${CMAKE_CURRENT_SOURCE_DIR}/validation"  validation)
blackwell_add_test_suite(benchmark_tests   "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks"  benchmark)
blackwell_add_test_suite(integration_tests "${CMAKE_CURRENT_SOURCE_DIR}/integration" integration)

# The Ultravox-8B full-pipeline test (integration/test_ultravox8b_full_pipeline.cpp)
# drives the audio-frontend kernels (projector + prompt injector) ON TOP of the
# engine, so the integration suite -- and ONLY it (validation/benchmark do not use
# them) -- links blackwell_audio and sees its *.cuh. No symbol clash: the audio
# launchers (launch_swiglu/launch_stack_audio_frames) are distinct from the kernels'
# launch_fused_swiglu_kernel/launch_*_embedding_kernel.
if(TARGET blackwell_audio)
    target_link_libraries(integration_tests PRIVATE blackwell_audio)
    target_include_directories(integration_tests PRIVATE
        ${CMAKE_SOURCE_DIR}/src/audio
        ${CMAKE_SOURCE_DIR}/audio_sandbox/translator)  # audio_head_loader.hpp (reconciliation field test)
endif()

# NOTE: the streaming TTFT profiler (integration/test_streaming_ttft_profiler.cpp)
# subclasses the translator's RealEngineControl and needs blackwell_bridge +
# whisper_dsp — targets defined AFTER tests/ (audio_sandbox is added later), so
# its wiring lives in the ROOT CMakeLists (the post-subdirectory if(TARGET ...)
# block), and the test compiles to nothing without BLACKWELL_HAVE_SPEECH_STACK.

# Ultravox audio-frontend parity tests. Unlike the gtest suites above these are
# STANDALONE mains (NOT gtest) that drive the blackwell_audio kernels against the
# FP32 PyTorch golden dumps (cosine > 0.999) -- kept standalone to preserve their
# self-documenting parity output. Wired here so they are real solution targets and
# run under CTest (`ctest -L audio`). They link blackwell_audio only (no engine /
# no gtest) and declare their own PRIVATE src/ includes (no target exports src/ --
# skill cmake-hygiene rule #2).
if(TARGET blackwell_audio)
    set(_ultravox_dumps "${CMAKE_SOURCE_DIR}/tests/integration/golden_dumps/ultravox")

    # Common wiring for a standalone audio parity executable: includes, link, the
    # CWD-independent CTest COMMAND (dumps dir as argv[1]) and the matching VS
    # debugger args (F5 bypasses the CTest COMMAND, so mirror them here).
    function(blackwell_wire_audio_test target)
        target_include_directories(${target} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}      # audio_test_utils.h
            ${CMAKE_SOURCE_DIR}/src          # common.h
            ${CMAKE_SOURCE_DIR}/src/core     # device_buffer.h
            ${CMAKE_SOURCE_DIR}/src/audio)   # ultravox_projector*.cuh, prompt_injector.cuh
        target_link_libraries(${target} PRIVATE blackwell_audio)
        add_test(NAME ${target} COMMAND ${target} "${_ultravox_dumps}")
        set_tests_properties(${target} PROPERTIES LABELS audio)
        set_target_properties(${target} PROPERTIES
            VS_DEBUGGER_COMMAND_ARGUMENTS "\"${_ultravox_dumps}\""
            VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
    endfunction()

    # Consolidated per-kernel unit suite: the former ultravox_projector_test and
    # prompt_injector_test micro-executables now compile into ONE binary (each
    # check keeps its own TU + entry point; kernel_unit_tests_main.cpp dispatches).
    # One link + one CTest process instead of two -> less linker/discovery overhead.
    add_executable(kernel_unit_tests
        kernel_unit_tests_main.cpp
        test_ultravox_projector.cpp
        test_prompt_injector.cpp)
    blackwell_wire_audio_test(kernel_unit_tests)

    # NOTE: the full end-to-end chain (projector -> injector -> Llama seam) is a
    # gtest now living in integration/test_ultravox_model_integration.cpp -- it is
    # picked up by the integration_tests suite via CONFIGURE_DEPENDS (which already
    # links blackwell_audio + its includes above), so it needs no target here.
endif()

# Bridge suite: CPU-only control-plane test for EngineControlBridge (SPSC command
# ring, barge-in epoch, wait/notify). It links blackwell_bridge for the bridge
# under test plus blackwell_core_obj to satisfy the engine symbols the bridge's
# execute hooks reference (BlackwellEngine::rewind / forward_status) — the test
# overrides those hooks, so they are never called, but must resolve at link. Runs
# under the fast `validation` label (no GPU needed).
if(TARGET blackwell_bridge)
    add_executable(bridge_tests
        "${CMAKE_CURRENT_SOURCE_DIR}/bridge/engine_control_bridge_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/bridge/mock_vad_streaming_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/bridge/speech_pipeline_manual_mode_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/bridge/speech_pipeline_vad_scorer_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/bridge/kv_ledger_test.cpp")
    target_include_directories(bridge_tests PRIVATE
        ${CMAKE_SOURCE_DIR}/include      # blackwell/*.h, bridge/*.h
        ${CMAKE_SOURCE_DIR}/src          # common.h
        ${CMAKE_SOURCE_DIR}/src/core     # device_buffer.h (transitive)
        ${CMAKE_SOURCE_DIR}/src/bridge   # engine_control_bridge.hpp, bridge_internal.hpp
        ${CMAKE_SOURCE_DIR}/src/audio)   # sliding_audio_window.h (center_slice_plan)
    target_link_libraries(bridge_tests PRIVATE
        GTest::gtest
        GTest::gtest_main
        blackwell_bridge
        blackwell_kernels
        blackwell_core_obj)
    blackwell_copy_dstorage_dlls(bridge_tests)
    gtest_discover_tests(bridge_tests
        PROPERTIES LABELS validation
        DISCOVERY_TIMEOUT 120
    )
endif()

# Agent-core suite: pure C++ (tree-sitter), no CUDA / no blackwell_kernels, so
# it gets its own minimal target rather than going through the macro above.
if(TARGET agent_core)
    file(GLOB agent_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/agent/*.cpp")
    add_executable(agent_tests ${agent_sources})
    target_link_libraries(agent_tests PRIVATE
        GTest::gtest
        GTest::gtest_main
        agent_core
    )
    gtest_discover_tests(agent_tests
        PROPERTIES LABELS agent
        DISCOVERY_TIMEOUT 120
    )
endif()

# Agent-env suite: pure C++ sandbox layer (filesystem / subprocess / git
# worktree), no CUDA / no blackwell_kernels, so it too gets its own minimal
# target rather than the CUDA-flavoured macro above.
if(TARGET agent_env)
    file(GLOB agent_env_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/agent_env/*.cpp")
    add_executable(agent_env_tests ${agent_env_sources})
    target_link_libraries(agent_env_tests PRIVATE
        GTest::gtest
        GTest::gtest_main
        agent_env
    )
    gtest_discover_tests(agent_env_tests
        PROPERTIES LABELS agent_env
        DISCOVERY_TIMEOUT 120
    )
endif()

# Agent-orchestrator suite: the ReAct state machine (parser / dispatcher / loop).
# Pure C++ and driven entirely by a MockLLM, so -- like the two subsystems it
# integrates -- it needs no CUDA / no blackwell_kernels and gets its own minimal
# target rather than the CUDA-flavoured macro above.
if(TARGET agent_orchestrator)
    file(GLOB agent_orch_sources CONFIGURE_DEPENDS
         "${CMAKE_CURRENT_SOURCE_DIR}/agent_orchestrator/*.cpp")
    add_executable(agent_orchestrator_tests ${agent_orch_sources})
    target_link_libraries(agent_orchestrator_tests PRIVATE
        GTest::gtest
        GTest::gtest_main
        agent_orchestrator
    )
    gtest_discover_tests(agent_orchestrator_tests
        PROPERTIES LABELS agent_orchestrator
        DISCOVERY_TIMEOUT 120
    )
endif()

# Silero neural-VAD suite: CPU-only (ONNXRuntime), no CUDA / no engine / no
# blackwell_kernels, so -- like the agent suites -- it gets its own minimal
# target instead of the CUDA-flavoured macro at the top. Runs under the fast
# `validation` label. Absent entirely with -DUSE_SILERO_VAD=OFF.
#
# The model path is baked in rather than passed as argv/CWD-relative, so the
# suite runs identically under CTest, under F5, and from any directory.
if(TARGET blackwell_vad)
    # speech_segmenter_test is CPU/STL-only and needs neither ORT nor the model,
    # but it rides in this target because SpeechSegmenter ships from src/vad as
    # the policy layer over whatever produced the probabilities.
    add_executable(vad_tests
        "${CMAKE_CURRENT_SOURCE_DIR}/vad/silero_vad_test.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/vad/speech_segmenter_test.cpp")
    # audio_test_wav.hpp: the reference speech clip's path + WAV reader + the
    # documented silence-valley geometry (header-only, STL-only, no CUDA).
    target_include_directories(vad_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/integration")
    target_link_libraries(vad_tests PRIVATE
        GTest::gtest
        GTest::gtest_main
        blackwell_vad
        psapi)  # GetProcessMemoryInfo (working-set smoke check)
    get_target_property(_vad_model blackwell_onnxruntime BLACKWELL_VAD_MODEL_PATH)
    target_compile_definitions(vad_tests PRIVATE BLACKWELL_VAD_MODEL_PATH="${_vad_model}")
    # onnxruntime.dll must sit next to the test exe (see the CRT note in
    # cmake/OnnxRuntime.cmake) -- without it the suite fails to launch at all.
    blackwell_copy_onnxruntime_dlls(vad_tests)
    gtest_discover_tests(vad_tests
        PROPERTIES LABELS validation
        DISCOVERY_TIMEOUT 120
    )
endif()
