cmake_minimum_required(VERSION 3.28)
project(BlackwellLLM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 1. ПРАВИЛЬНАЯ настройка архитектуры Blackwell для Visual Studio
# CMake сам корректно передаст нужные инструкции в NVCC
set(CMAKE_CUDA_ARCHITECTURES 120)

# 2. Оптимизации и флаги
# Убираем -O3. В Visual Studio оптимизация включается выбором конфигурации "Release" в самой IDE.
# Добавляем быструю математику только для CUDA-файлов, чтобы не смущать MSVC:
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>")

# Включаем тестирование
enable_testing()

# 🎯 Опция для включения DirectStorage (по умолчанию ВЫКЛЮЧЕНА). Объявлена до
# включения модуля, который на неё смотрит.
option(USE_DIRECT_STORAGE "Enable DirectStorage for blazing fast weight loading on Windows" OFF)

# 3. Зависимости: pinned FetchContent + опциональный DirectStorage SDK
# (интерфейсный таргет blackwell::dstorage + blackwell_copy_dstorage_dlls())
# + общие хелперы (blackwell_copy_runtime_dlls).
include(cmake/BlackwellHelpers.cmake)
include(cmake/Dependencies.cmake)
include(cmake/DirectStorage.cmake)

# ============================================================================
# Compiler hygiene: zero-warning policy on our first-party CXX code.
# Placed AFTER the dependency includes so already-defined FetchContent targets
# (googletest, ...) keep their own warning level -- this directory property
# only applies to targets created below (src/, tests/).
#   - CXX only: nvcc host-passthrough of /W4 /WX over CUDA/CCCL headers is not
#     viable, so CUDA (.cu) compilation is untouched.
#   - /external:anglebrackets + /external:W0: <>-includes (CUDA toolkit, STL,
#     Win32, deps) are "external" and drop to W0, so /W4 /WX fires ONLY on our
#     own "..."-included code.
# ============================================================================
if(MSVC)
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CXX>:/W4>"
        "$<$<COMPILE_LANGUAGE:CXX>:/WX>"
        "$<$<COMPILE_LANGUAGE:CXX>:/external:anglebrackets>"
        "$<$<COMPILE_LANGUAGE:CXX>:/external:W0>"
        # C4996 deprecates STANDARD C functions (getenv/fopen/...) in favour of
        # MSVC's non-portable *_s variants. This code is cross-platform-standard,
        # so silence that one non-warning (CXX only -- CUDA TUs are untouched, so
        # they never recompile for it) rather than un-portably rewriting it.
        "$<$<COMPILE_LANGUAGE:CXX>:/D_CRT_SECURE_NO_WARNINGS>"
    )
endif()

# Подключаем поддиректории
add_subdirectory(src)
add_subdirectory(tests)