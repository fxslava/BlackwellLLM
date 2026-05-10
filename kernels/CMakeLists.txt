file(GLOB SOURCES "*.cu")

# Создаем статическую библиотеку с ядрами
add_library(blackwell_kernels STATIC ${SOURCES})

# --- Настройки для библиотеки ядер (blackwell_kernels) ---

# 1. Флаги для конфигурации RELEASE (Ориентировано на бенчмарки)
target_compile_options(blackwell_kernels PRIVATE 
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:
        --use_fast_math          # Разрешает быстрые (но менее точные) математические функции
        -Xptxas=-O3,-v           # Оптимизация PTX-ассемблера + вывод использования регистров
        -Xcompiler=/O2,/Ot       # Флаги MSVC: Оптимизация по скорости и всему приложению
    >
)

# 2. Флаги для конфигурации DEBUG (Ориентировано на Nsight Debugger)
target_compile_options(blackwell_kernels PRIVATE 
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:
        -g -G                    # -g: символы хоста, -G: символы устройства (DEBUG ЯДЕР)
        -Xcompiler=/Zi,/Od,/RTC1 # Флаги MSVC: символы, отключение оптимизаций, проверки рантайма
    >
)

# Для Blackwell/Hopper полезно видеть статистику регистров в Release
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    # Выводит информацию о регистрах и Shared Memory при сборке
    set_target_properties(blackwell_kernels PROPERTIES CUDA_PTX_COMPILATION_FLAGS "-v")
endif()

# Разрешаем включать заголовочные файлы из этой директории
target_include_directories(blackwell_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})