# ==============================================================================
# agent_playground: a GUI "Agent Simulator / Prompt Playground" for the XML ReAct
# protocol, now upgraded into a LIVE LLM testing environment.
#
# ARCHITECTURE: a local HTTP server (cpp-httplib, header-only) serving an embedded
# single-page HTML/JS/CSS frontend. We chose a Web UI over Dear ImGui because live
# XML syntax highlighting, a click-to-inject snippet toolbar, and zero-friction
# build integration are all cheap in HTML/JS and expensive in an immediate-mode
# GUI.
#
# This target now links the REAL stack:
#   * blackwell_core      -- CUDA inference engine + tokenizer + (optional)
#                            DirectStorage weight loader. PRIVATE-propagates
#                            blackwell_kernels (CUDA) and the include/ dir, exactly
#                            as the blackwell_llm executable does.
#   * agent_orchestrator  -- ToolParser / ToolRegistry / AgentOrchestrator and the
#                            default agent_env+agent_core tools. (This also supplies
#                            tool_parser.cpp, so we no longer compile it here.)
# ==============================================================================

include(FetchContent)

# Header-only HTTP server/client. No OpenSSL/zlib by default -> nothing to find.
FetchContent_Declare(
    cpp_httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG        v0.15.3
)
FetchContent_MakeAvailable(cpp_httplib)

add_executable(agent_playground
    playground_server.cpp
    blackwell_llm_adapter.cpp
)

target_compile_features(agent_playground PRIVATE cxx_std_20)

# The adapter binds its tokenizer to the engine's prefill driver, which pulls
# the paging substrate headers (and through them <cuda_runtime.h>).
find_package(CUDAToolkit REQUIRED)

target_include_directories(agent_playground PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../agent_orchestrator  # orchestrator headers
    ${CMAKE_SOURCE_DIR}/src/core                          # engine_prefill_coordinator.h
    ${CMAKE_SOURCE_DIR}/src                               # paging/ headers reach kernels/*.cuh + common.h
    ${CUDAToolkit_INCLUDE_DIRS}
)

# blackwell_core PUBLIC-exposes the include/ dir (blackwell/*.h) and transitively
# links blackwell_kernels + CUDA; agent_orchestrator PUBLIC-exposes its own headers
# and links agent_core + agent_env. Linking these two is all the propagation we
# need -- no manual CUDA find_package, mirroring src/CMakeLists.txt's blackwell_llm.
target_link_libraries(agent_playground PRIVATE
    httplib::httplib
    nlohmann_json::nlohmann_json
    blackwell_core_obj   # white-box engine layer (adapter drives the prefill
                         # coordinator C++ API; see poc_overlay's CMakeLists)
    agent_orchestrator
)

# The frontend is no longer embedded in the C++ binary -- it lives in ui/ and is
# served as static files. Copy it next to the executable on every build so the
# server (which resolves <exe_dir>/ui) finds it. Tweaking CSS/JS now only needs a
# browser refresh; recompiling C++ is no longer required to iterate on the UI.
add_custom_command(TARGET agent_playground POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/ui"
        "$<TARGET_FILE_DIR:agent_playground>/ui"
    COMMENT "Copying agent_playground UI assets (ui/) next to the executable")

# When the core lib was built with DirectStorage, the engine needs the DS
# runtime DLLs next to the executable (no-op otherwise; the helper comes from
# cmake/DirectStorage.cmake).
blackwell_copy_dstorage_dlls(agent_playground)
