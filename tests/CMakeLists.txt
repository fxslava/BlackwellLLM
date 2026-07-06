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
        ${CMAKE_CURRENT_SOURCE_DIR}   # common/...
        ${CMAKE_SOURCE_DIR}/src       # common.h, engine_impl.h, kernels/*.cuh
    )

    target_link_libraries(${target} PRIVATE
        GTest::gtest
        GTest::gtest_main
        blackwell_kernels
        blackwell_core
    )

    if(WIN32)
        # The DirectStorage SDK is only downloaded (into the build tree's
        # _deps) when USE_DIRECT_STORAGE=ON; without the option there is
        # nothing to copy.
        if(USE_DIRECT_STORAGE)
            set(ds_bin "${CMAKE_BINARY_DIR}/_deps/directstorage_sdk-src/native/bin/x64")
            add_custom_command(TARGET ${target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${ds_bin}/dstorage.dll"
                    "$<TARGET_FILE_DIR:${target}>"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${ds_bin}/dstoragecore.dll"
                    "$<TARGET_FILE_DIR:${target}>"
                COMMENT "Copying DirectStorage DLLs next to ${target}..."
            )
        endif()
    endif()

    gtest_discover_tests(${target}
        PROPERTIES LABELS "${label}"
        DISCOVERY_TIMEOUT 120
    )
endfunction()

blackwell_add_test_suite(validation_tests  "${CMAKE_CURRENT_SOURCE_DIR}/validation"  validation)
blackwell_add_test_suite(benchmark_tests   "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks"  benchmark)
blackwell_add_test_suite(integration_tests "${CMAKE_CURRENT_SOURCE_DIR}/integration" integration)

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
