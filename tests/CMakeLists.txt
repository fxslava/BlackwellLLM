# Test suite layout:
#   common/      - shared headers (CUDA fixtures, CPU references, problem
#                  generators, benchmark harness); no target of its own
#   validation/  - mathematical correctness vs CPU references on the exact
#                  Llama-FP8 / Qwen-AWQ target shapes  -> validation_tests
#   benchmarks/  - stress shapes (split-K extremes, ragged groups, non-aligned
#                  dims) and CUDA-event performance benchmarks -> benchmark_tests
#   integration/ - end-to-end engine vs PyTorch golden dumps (skips when the
#                  local model/dumps are absent)       -> integration_tests
#   reference/   - compatibility shim only: cpu_models.h is included by
#                  src/experiments/compress_experiment.cu via a relative path
#                  and exposed globally by the root CMakeLists.txt
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
        ${CMAKE_CURRENT_SOURCE_DIR}   # common/..., reference/...
        ${CMAKE_SOURCE_DIR}/src       # common.h, engine_impl.h, kernels/*.cuh
    )

    target_link_libraries(${target} PRIVATE
        GTest::gtest
        GTest::gtest_main
        blackwell_kernels
        blackwell_core
    )

    if(WIN32)
        # nvCOMP runtime DLL must sit next to the executable.
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_SOURCE_DIR}/external/nvcomp/bin/nvcomp64_5.dll"
                "$<TARGET_FILE_DIR:${target}>"
            COMMENT "Copying nvCOMP DLL next to ${target}..."
        )

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

# Standalone experiment binaries (awq_benchmark etc.). src/CMakeLists.txt does
# not include the experiments subdirectory itself, so it is hooked up here;
# long-term this line belongs in src/CMakeLists.txt as add_subdirectory(experiments).
add_subdirectory(${CMAKE_SOURCE_DIR}/src/experiments ${CMAKE_BINARY_DIR}/src/experiments)
