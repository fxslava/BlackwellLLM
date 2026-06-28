# Явный список вместо file(GLOB): GLOB вычисляется только на этапе configure,
# поэтому новые ядра (bias.cu, bf16_linear_bias.cu) молча выпадали бы из сборки
# до ручного переконфигурирования.
set(SOURCES
    attention.cu
    awq_linear.cu
    bf16_linear.cu
    bf16_linear_bias.cu
    bf16_linear_residual.cu
    bias.cu
    embedding.cu
    fp8_linear.cu
    full_attention.cu
    paged_flash_attention.cu
    rmsnorm.cu
    rope.cu
    sampling.cu
    ssm_kernels.cu
    sym_int4_linear.cu
    swiglu.cu
)

# Создаем статическую библиотеку с ядрами
add_library(blackwell_kernels STATIC ${SOURCES})

# --- Настройки для библиотеки ядер (blackwell_kernels) ---

# --- Настройки для библиотеки ядер ---

# Флаги для конфигурации RELEASE
target_compile_options(blackwell_kernels PRIVATE 
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:
        --use_fast_math
        -Xptxas=-O3,-v
        # Пробрасываем стандартный препроцессор и отключаем излишние проверки CCCL
        -Xcompiler=/O2,/Ot,/Zc:preprocessor
        -DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
    >
)

# Флаги для конфигурации DEBUG
target_compile_options(blackwell_kernels PRIVATE 
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:
        -g -G
        -Xcompiler=/Zi,/Od,/Zc:preprocessor
        -DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
    >
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set_target_properties(blackwell_kernels PROPERTIES CUDA_PTX_COMPILATION_FLAGS "-v")
endif()

# Разрешаем включать заголовочные файлы из этой директории
target_include_directories(blackwell_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)