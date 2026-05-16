cmake_minimum_required(VERSION 3.14)
project(BlackwellLLM CXX)

# ==============================================================================
# 1. СТАТИЧЕСКАЯ БИБЛИОТЕКА ДВИЖКА (Core Library)
# ==============================================================================
add_library(blackwell_core STATIC
    engine.cpp
    safetensors.cpp
    memory_pool.cpp
    tokenizer.cpp
)

# Линкуем только JSON и твои CUDA-ядра
target_link_libraries(blackwell_core PRIVATE 
    nlohmann_json::nlohmann_json 
    blackwell_kernels
)

target_include_directories(blackwell_core PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/../include  
)

# ==============================================================================
# 2. ОСНОВНОЙ ИСПОЛНЯЕМЫЙ ФАЙЛ (Main Executable)
# ==============================================================================
add_executable(blackwell_llm main.cpp)
target_link_libraries(blackwell_llm PRIVATE blackwell_core)