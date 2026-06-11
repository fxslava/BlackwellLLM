file(GLOB TEST_SOURCES "*.cpp")

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
    
    # 2. Жесткий путь к DirectStorage (он скачивается в папку сборки _deps)
    set(LOCAL_DS_BIN "${CMAKE_BINARY_DIR}/_deps/directstorage_sdk-src/native/bin/x64")

    add_custom_command(TARGET unit_tests POST_BUILD
        # Копируем nvCOMP
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${LOCAL_NVCOMP_DLL}"
        "$<TARGET_FILE_DIR:unit_tests>"
        
        # Копируем DirectStorage (основная DLL)
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${LOCAL_DS_BIN}/dstorage.dll"
        "$<TARGET_FILE_DIR:unit_tests>"
        
        # Копируем DirectStorage (Core DLL)
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${LOCAL_DS_BIN}/dstoragecore.dll"
        "$<TARGET_FILE_DIR:unit_tests>"
        
        COMMENT "Copying nvCOMP and DirectStorage DLLs to unit_tests directory..."
    )
endif()

# Регистрируем бинарник в CTest
include(GoogleTest)
gtest_discover_tests(unit_tests)