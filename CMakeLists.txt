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

include(FetchContent)

# 3. Адаптация под Windows/MSVC
# Принудительно отключаем поиск pthreads, чтобы не мусорить в логах сборки
set(gtest_disable_pthreads ON CACHE BOOL "" FORCE)
# Заставляем GTest использовать ту же стандартную библиотеку (DLL/MD), что и VS2022
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Загрузка GoogleTest
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)
FetchContent_MakeAvailable(googletest)

# Загрузка nlohmann/json для парсинга Safetensors заголовков
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

# Общие заголовки
include_directories(${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/reference)

# Подключаем поддиректории
add_subdirectory(kernels)
add_subdirectory(src)
add_subdirectory(tests)