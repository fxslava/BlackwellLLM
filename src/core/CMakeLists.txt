# ==============================================================================
# СТАТИЧЕСКАЯ БИБЛИОТЕКА ДВИЖКА (blackwell_core)
# ==============================================================================
# Явный список источников (не GLOB) -- см. правило в src/kernels/CMakeLists.txt.
set(CORE_SOURCES
    engine.cpp
    engine_prefill_coordinator.cpp
    runtime_config.cpp
    ops_dispatcher.cpp
    safetensors.cpp
    memory_pool.cpp
    kv_cache/continuous_kv_manager.cpp
    kv_cache/paged_kv_manager.cpp
    tokenizer/unicode.cpp
    tokenizer/bpe_tokenizer.cpp
    tokenizer/chat_template.cpp
    tokenizer/tokenizer_factory.cpp
    weight_loader.cpp
    config.cpp
)

# blackwell_dstorage существует только при USE_DIRECT_STORAGE=ON
# (см. cmake/DirectStorage.cmake); тот же гейт добавляет DS-загрузчик в сборку.
if(TARGET blackwell_dstorage)
    list(APPEND CORE_SOURCES direct_storage_loader.cpp)
endif()

add_library(blackwell_core STATIC ${CORE_SOURCES})

# PRIVATE достаточно для blackwell::dstorage -- дефайн USE_DIRECT_STORAGE
# потребляют только TU самой библиотеки (weight_loader.cpp /
# direct_storage_loader.cpp).
target_link_libraries(blackwell_core PRIVATE
    nlohmann_json::nlohmann_json
    blackwell_kernels
)
if(TARGET blackwell_dstorage)
    target_link_libraries(blackwell_core PRIVATE blackwell::dstorage)
endif()

target_include_directories(blackwell_core
    PUBLIC  ${CMAKE_SOURCE_DIR}/include       # blackwell/*.h -- публичный API
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}       # engine_impl.h, kv_cache/, paging/, ssm/, tokenizer/
    PRIVATE ${CMAKE_SOURCE_DIR}/src           # common.h, kernels/*.cuh
)
