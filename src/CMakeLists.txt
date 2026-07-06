add_subdirectory(kernels)

# 🤖 Autonomous code-agent core (tree-sitter CST + shadow graph). Isolated from
# the CUDA inference engine; see src/agent/CMakeLists.txt. Default ON.
option(BUILD_AGENT_CORE "Build the autonomous-agent code-analysis core (tree-sitter based, no CUDA)" ON)
if(BUILD_AGENT_CORE)
    add_subdirectory(agent)
endif()

# 🧰 Autonomous agent sandbox / "OS" layer (VFS + subprocess + git worktree).
# Also isolated from CUDA; see src/agent_env/CMakeLists.txt. Default ON.
option(BUILD_AGENT_ENV "Build the autonomous-agent sandbox OS layer (filesystem/exec/git, no CUDA)" ON)
if(BUILD_AGENT_ENV)
    add_subdirectory(agent_env)
endif()

# 🤝 Autonomous agent ReAct orchestrator (LLM <-> agent_core <-> agent_env).
# The integration seam: the first target that links BOTH isolated subsystems.
# Still CUDA-free (talks to the model via the ILLMGenerator text interface), so
# it requires both agent libs. Default ON when those are built.
option(BUILD_AGENT_ORCHESTRATOR "Build the autonomous-agent ReAct orchestrator (no CUDA)" ON)
if(BUILD_AGENT_ORCHESTRATOR AND BUILD_AGENT_CORE AND BUILD_AGENT_ENV)
    add_subdirectory(agent_orchestrator)
endif()

# ==============================================================================
# 1. СТАТИЧЕСКАЯ БИБЛИОТЕКА ДВИЖКА (Core Library)
# ==============================================================================
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

if(USE_DIRECT_STORAGE AND WIN32)
    list(APPEND CORE_SOURCES direct_storage_loader.cpp)
endif()

add_library(blackwell_core STATIC ${CORE_SOURCES}) 

# Линкуем базовые зависимости. blackwell::dstorage существует только при
# USE_DIRECT_STORAGE=ON (см. cmake/DirectStorage.cmake); PRIVATE достаточно --
# дефайн USE_DIRECT_STORAGE потребляют только TU самой библиотеки
# (weight_loader.cpp / direct_storage_loader.cpp).
target_link_libraries(blackwell_core PRIVATE
    nlohmann_json::nlohmann_json
    blackwell_kernels
)
if(TARGET blackwell_dstorage)
    target_link_libraries(blackwell_core PRIVATE blackwell::dstorage)
endif()

target_include_directories(blackwell_core
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/../include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

# ==============================================================================
# 2. ИСПОЛНЯЕМЫЕ ФАЙЛЫ (см. src/apps/CMakeLists.txt)
# ==============================================================================
add_subdirectory(apps)

# ==============================================================================
# 4. СТАНДАЛОН-ЭКСПЕРИМЕНТЫ (см. src/experiments/CMakeLists.txt)
# ==============================================================================
# The nvCOMP compression experiment (blackwell_compress) was removed together
# with the vendored external/nvcomp binaries; experiments now live in their own
# subdirectory with per-target scripts.
add_subdirectory(experiments)

# ==============================================================================
# 5. AGENT SIMULATOR / PROMPT PLAYGROUND (live LLM testing GUI)
# ==============================================================================
# A local-HTTP-server GUI for debugging the XML ReAct protocol AND running real
# inference. It links blackwell_core (CUDA engine + tokenizer) and the ReAct
# orchestrator, so it is gated on the orchestrator stack being built.
option(BUILD_AGENT_PLAYGROUND "Build the GUI Agent Simulator / Prompt Playground tool" ON)
if(BUILD_AGENT_PLAYGROUND AND BUILD_AGENT_ORCHESTRATOR AND BUILD_AGENT_CORE AND BUILD_AGENT_ENV)
    add_subdirectory(tools/playground)
endif()

# ==============================================================================
# 6. POC: input-hook / UIA-caret / D2D-overlay OS pipeline
# ==============================================================================
# Isolated feasibility PoC for the overlay-translator UI pipeline. No CUDA, no
# blackwell_core, no tokenizer -- just Win32 + COM + UI Automation + Direct2D.
# See src/tools/poc_overlay/CMakeLists.txt.
option(BUILD_POC_OVERLAY "Build the input-hook/UIA-caret/D2D-overlay OS pipeline PoC" ON)
if(BUILD_POC_OVERLAY AND WIN32)
    add_subdirectory(tools/poc_overlay)
endif()