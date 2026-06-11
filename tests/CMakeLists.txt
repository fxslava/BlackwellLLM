# CONFIGURE_DEPENDS: newly added test files (e.g. test_awq_linear.cpp) are
# picked up on the next build without a manual CMake re-run.
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "*.cpp")

add_executable(unit_tests ${TEST_SOURCES})

target_include_directories(unit_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

# Линкуем GoogleTest, библиотеку ядер и CUDA Runtime
target_link_libraries(unit_tests PRIVATE
    GTest::gtest
    GTest::gtest_main
    blackwell_kernels
    blackwell_core
)

if(WIN32)
    # 1. Жесткий путь к nvCOMP
    set(LOCAL_NVCOMP_DLL "${CMAKE_SOURCE_DIR}/external/nvcomp/bin/nvcomp64_5.dll")
    
    add_custom_command(TARGET unit_tests POST_BUILD
        # Копируем nvCOMP
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${LOCAL_NVCOMP_DLL}"
        "$<TARGET_FILE_DIR:unit_tests>"
        COMMENT "Copying nvCOMP DLL to unit_tests directory..."
    )

    # 2. Жесткий путь к DirectStorage (он скачивается в папку сборки _deps).
    # SDK скачивается только при USE_DIRECT_STORAGE=ON — без этой опции
    # копировать нечего, и безусловная команда ломала сборку unit_tests.
    if(USE_DIRECT_STORAGE)
        set(LOCAL_DS_BIN "${CMAKE_BINARY_DIR}/_deps/directstorage_sdk-src/native/bin/x64")

        add_custom_command(TARGET unit_tests POST_BUILD
            # Копируем DirectStorage (основная DLL)
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${LOCAL_DS_BIN}/dstorage.dll"
            "$<TARGET_FILE_DIR:unit_tests>"

            # Копируем DirectStorage (Core DLL)
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${LOCAL_DS_BIN}/dstoragecore.dll"
            "$<TARGET_FILE_DIR:unit_tests>"

            COMMENT "Copying DirectStorage DLLs to unit_tests directory..."
        )
    endif()
endif()

# Регистрируем бинарник в CTest
include(GoogleTest)
gtest_discover_tests(unit_tests)

# Standalone experiment binaries (awq_benchmark etc.). src/CMakeLists.txt does
# not include the experiments subdirectory itself, so it is hooked up here;
# long-term this line belongs in src/CMakeLists.txt as add_subdirectory(experiments).
add_subdirectory(${CMAKE_SOURCE_DIR}/src/experiments ${CMAKE_BINARY_DIR}/src/experiments)