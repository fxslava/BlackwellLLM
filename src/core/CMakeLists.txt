# ==============================================================================
# ДВИЖОК: blackwell_core_obj (внутренний слой) + blackwell_core (SHARED DLL)
# ==============================================================================
# Двухуровневая поверхность (Object Library Pattern):
#   blackwell_core_obj -- OBJECT-библиотека со ВСЕМИ исходниками движка. Это
#       полнодоверенный white-box-слой: его линкуют юнит-тесты (включают
#       engine_impl.h) и in-tree инструменты (poc_overlay / agent_playground),
#       которым нужен C++-API координатора префилла и paging-заголовки --
#       поверхность, которую COM-интерфейс пока не покрывает.
#   blackwell_core     -- SHARED (DLL) с УЗКОЙ бинарной границей: единственный
#       собственный TU engine_com.cpp экспортирует C-фабрики
#       CreateBlackwellEngine/CreateBlackwellTokenizer (COM-стиль, HRESULT).
#       Никакого WINDOWS_EXPORT_ALL_SYMBOLS: смысл границы -- узость.
#       Внешние потребители (src/apps/) линкуют только импорт-библиотеку и
#       включают только blackwell/iblackwell_engine.h.

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

add_library(blackwell_core_obj OBJECT ${CORE_SOURCES})

# kernels/dstorage -- PUBLIC: их объектники/импорт-библиотеки должны дойти до
# финальной линковки каждого потребителя OBJECT-библиотеки (тесты, тулзы, DLL).
# nlohmann_json -- PRIVATE: header-only, нужен только на компиляции наших TU.
target_link_libraries(blackwell_core_obj
    PUBLIC  blackwell_kernels
    PRIVATE nlohmann_json::nlohmann_json
)
if(TARGET blackwell_dstorage)
    target_link_libraries(blackwell_core_obj PUBLIC blackwell::dstorage)
endif()

target_include_directories(blackwell_core_obj
    PUBLIC  ${CMAKE_SOURCE_DIR}/include       # blackwell/*.h -- публичный API
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}       # engine_impl.h, kv_cache/, paging/, ssm/, tokenizer/
    PRIVATE ${CMAKE_SOURCE_DIR}/src           # common.h, kernels/*.cuh
)

# --- Бинарная граница ---------------------------------------------------------
add_library(blackwell_core SHARED engine_com.cpp)
target_link_libraries(blackwell_core PRIVATE blackwell_core_obj)
# BLACKWELL_CORE_EXPORTS переключает BLACKWELL_API на dllexport только здесь.
target_compile_definitions(blackwell_core PRIVATE BLACKWELL_CORE_EXPORTS)
target_include_directories(blackwell_core PUBLIC ${CMAKE_SOURCE_DIR}/include)
