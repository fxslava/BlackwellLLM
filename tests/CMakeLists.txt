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
    target_include_directories(integration_tests PRIVATE ${CMAKE_SOURCE_DIR}/src/audio)
endif()

# Ultravox audio-frontend parity tests. Unlike the suites above these are
# STANDALONE mains (each its own main(), NOT gtest) that drive the blackwell_audio
# kernels against the FP32 PyTorch golden dumps (cosine > 0.999) -- kept standalone
# to preserve their self-documenting nvcc build headers. Wired here so they are
# real solution targets and run under CTest (`ctest -L audio`). They link
# blackwell_audio only (no engine / no gtest) and declare their own PRIVATE
# src/ includes (no target exports src/ -- skill cmake-hygiene rule #2).
if(TARGET blackwell_audio)
    set(_ultravox_dumps "${CMAKE_SOURCE_DIR}/tests/integration/golden_dumps/ultravox")

    function(blackwell_add_audio_test target source)
        add_executable(${target} "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
        target_include_directories(${target} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}      # audio_test_utils.h
            ${CMAKE_SOURCE_DIR}/src          # common.h
            ${CMAKE_SOURCE_DIR}/src/core     # device_buffer.h
            ${CMAKE_SOURCE_DIR}/src/audio)   # ultravox_projector*.cuh, prompt_injector.cuh
        target_link_libraries(${target} PRIVATE blackwell_audio)
        # Pass the golden-dumps dir explicitly so the test is CWD-independent.
        add_test(NAME ${target} COMMAND ${target} "${_ultravox_dumps}")
        set_tests_properties(${target} PROPERTIES LABELS audio)
    endfunction()

    blackwell_add_audio_test(ultravox_projector_test         test_ultravox_projector.cpp)
    blackwell_add_audio_test(prompt_injector_test            test_prompt_injector.cpp)
    blackwell_add_audio_test(ultravox_model_integration_test test_ultravox_model_integration.cpp)
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
