file(GLOB TEST_SOURCES "*.cpp")

add_executable(unit_tests ${TEST_SOURCES})

# Линкуем GoogleTest, библиотеку ядер и CUDA Runtime
target_link_libraries(unit_tests PRIVATE 
    GTest::gtest
    GTest::gtest_main
    blackwell_kernels
)

# Регистрируем бинарник в CTest
include(GoogleTest)
gtest_discover_tests(unit_tests)