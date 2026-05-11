# ==============================================================================
# 1. СТАТИЧЕСКАЯ БИБЛИОТЕКА ДВИЖКА (Core Library)
# ==============================================================================
add_library(blackwell_core STATIC
    engine.cpp
    safetensors.cpp
    memory_pool.cpp
)

# Линкуем ядра и JSON к нашей статической библиотеке
target_link_libraries(blackwell_core PRIVATE 
    nlohmann_json::nlohmann_json 
    blackwell_kernels
)

# Указываем, что все, кто линкует blackwell_core, автоматически получают пути к include/
target_include_directories(blackwell_core PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/../include  # Отрегулируйте путь к вашим заголовочным файлам
)

# ==============================================================================
# 2. ОСНОВНОЙ ИСПОЛНЯЕМЫЙ ФАЙЛ (Main Executable)
# ==============================================================================
add_executable(blackwell_llm main.cpp)

# Просто линкуем статичную библиотеку движка
target_link_libraries(blackwell_llm PRIVATE blackwell_core)

# ==============================================================================
# 3. ИНТЕГРАЦИОННЫЙ ТЕСТ (Integration Tests)
# ==============================================================================
# Если у вас тесты собираются здесь же, настройте линковку для них:
# add_executable(engine_integration_test tests/test_integration.cpp)
# target_link_libraries(engine_integration_test PRIVATE blackwell_core GTest::gtest_main)