cmake_minimum_required(VERSION 3.20) 
project(BlackwellLLM CXX) 

# 🎯 Опция для включения DirectStorage (по умолчанию ВЫКЛЮЧЕНА)
option(USE_DIRECT_STORAGE "Enable DirectStorage for blazing fast weight loading on Windows" OFF)

# ==============================================================================
# 1. СТАТИЧЕСКАЯ БИБЛИОТЕКА ДВИЖКА (Core Library)
# ==============================================================================
set(CORE_SOURCES
    engine.cpp
    safetensors.cpp
    memory_pool.cpp
    tokenizer.cpp
    weight_loader.cpp
)

if(USE_DIRECT_STORAGE AND WIN32)
    list(APPEND CORE_SOURCES direct_storage_loader.cpp)
endif()

add_library(blackwell_core STATIC ${CORE_SOURCES}) 

# Линкуем базовые зависимости
target_link_libraries(blackwell_core PRIVATE 
    nlohmann_json::nlohmann_json 
    blackwell_kernels
) 

target_include_directories(blackwell_core PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/../include  
) 

# ==============================================================================
# 2. АВТОМАТИЧЕСКАЯ ЗАГРУЗКА DIRECTSTORAGE SDK С NUGET
# ==============================================================================
if(USE_DIRECT_STORAGE AND WIN32)
    target_compile_definitions(blackwell_core PUBLIC USE_DIRECT_STORAGE)

    include(FetchContent)
    
    # ✅ Правильный вывод в консоль CMake:
    message(STATUS "Fetching Microsoft DirectStorage SDK from NuGet...")
    
    FetchContent_Declare(
        directstorage_sdk
        # Используем v2 API, оно лучше переваривается wget/curl внутри CMake
        URL "https://www.nuget.org/api/v2/package/Microsoft.Direct3D.DirectStorage/1.3.0"
        # Принудительно сохраняем как обычный ZIP, чтобы архиватор не сошел с ума
        DOWNLOAD_NAME "directstorage.zip"
        # Отключаем проверку сертификатов на случай локальных сетевых экранов
        TLS_VERIFY OFF
    )
    FetchContent_MakeAvailable(directstorage_sdk)
    
    FetchContent_GetProperties(directstorage_sdk SOURCE_DIR DS_DIR)
    
    # Разделяем пути: include для заголовков, lib для линковки, bin для dll
    set(DS_INCLUDE_DIR "${DS_DIR}/native/include")
    set(DS_LIB_DIR     "${DS_DIR}/native/lib/x64")
    set(DS_BIN_DIR     "${DS_DIR}/native/bin/x64")
    
    # Подключаем заголовки
    target_include_directories(blackwell_core PUBLIC ${DS_INCLUDE_DIR})
    
    # 🎯 Указываем линковщику правильную папку с .lib файлами
    target_link_directories(blackwell_core PUBLIC ${DS_LIB_DIR})
    
    # Собственно сама линковка
    target_link_libraries(blackwell_core PRIVATE dstorage d3d12)
    
    # ✅ И здесь тоже правильный вывод:
    message(STATUS "DirectStorage SDK successfully integrated from: ${DS_DIR}")
endif()

# ==============================================================================
# 3. ИСПОЛНЯЕМЫЕ ФАЙЛЫ И ПОСТ-БИЛД АВТОМАТИЗАЦИЯ
# ==============================================================================
add_executable(blackwell_llm main.cpp) 
target_link_libraries(blackwell_llm PRIVATE blackwell_core) 

add_executable(blackwell_bench benchmark.cpp) 
target_link_libraries(blackwell_bench PRIVATE blackwell_core)

# 🎯 АВТОМАТИЧЕСКОЕ КОПИРОВАНИЕ РАНТАЙМ DLL
# Если DirectStorage включен, мы принудительно копируем dll-ки в папки сборки рядом с .exe
if(USE_DIRECT_STORAGE AND WIN32)
    # Копирование для основного чата
    add_custom_command(TARGET blackwell_llm POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${DS_BIN_DIR}/dstorage.dll"
        "${DS_BIN_DIR}/dstoragecore.dll"
        $<TARGET_FILE_DIR:blackwell_llm>
    )
    
    # Копирование для бенчмарка перплексии
    add_custom_command(TARGET blackwell_bench POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${DS_BIN_DIR}/dstorage.dll"
        "${DS_BIN_DIR}/dstoragecore.dll"
        $<TARGET_FILE_DIR:blackwell_bench>
    )
endif()