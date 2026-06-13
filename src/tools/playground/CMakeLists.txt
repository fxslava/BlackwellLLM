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

target_include_directories(agent_playground PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../agent_orchestrator  # orchestrator headers
)

# blackwell_core PUBLIC-exposes the include/ dir (blackwell/*.h) and transitively
# links blackwell_kernels + CUDA; agent_orchestrator PUBLIC-exposes its own headers
# and links agent_core + agent_env. Linking these two is all the propagation we
# need -- no manual CUDA find_package, mirroring src/CMakeLists.txt's blackwell_llm.
target_link_libraries(agent_playground PRIVATE
    httplib::httplib
    nlohmann_json::nlohmann_json
    blackwell_core
    agent_orchestrator
)

# When the core lib was built with DirectStorage, the engine needs the DS runtime
# DLLs next to the executable. DS_BIN_DIR is set in src/CMakeLists.txt and inherited
# here because this subdirectory is added after that block.
if(USE_DIRECT_STORAGE AND WIN32)
    add_custom_command(TARGET agent_playground POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${DS_BIN_DIR}/dstorage.dll"
        "${DS_BIN_DIR}/dstoragecore.dll"
        $<TARGET_FILE_DIR:agent_playground>
        COMMENT "Copying DirectStorage runtime DLLs next to agent_playground.exe")
endif()
