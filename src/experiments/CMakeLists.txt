# Standalone experiment executables. One target per experiment, defined here
# (never inline in src/CMakeLists.txt). Explicit source lists, PRIVATE deps.

add_executable(awq_benchmark awq_benchmark.cu)

target_link_libraries(awq_benchmark PRIVATE blackwell_kernels)

# For "common.h" (CUDA_CHECK / CudaVector helpers)
target_include_directories(awq_benchmark PRIVATE ${CMAKE_SOURCE_DIR}/src)
