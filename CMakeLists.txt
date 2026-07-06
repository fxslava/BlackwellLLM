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
# (интерфейсный таргет blackwell::dstorage + blackwell_copy_dstorage_dlls()).
include(cmake/Dependencies.cmake)
include(cmake/DirectStorage.cmake)

# Подключаем поддиректории
add_subdirectory(src)
add_subdirectory(tests)