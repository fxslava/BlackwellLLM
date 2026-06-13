# ==============================================================================
# agent_playground: a GUI "Agent Simulator / Prompt Playground" for visually
# debugging the LLM's XML ReAct instruction-following.
#
# ARCHITECTURE: a local HTTP server (cpp-httplib, header-only) that serves an
# embedded single-page HTML/JS/CSS frontend. We chose a Web UI over Dear ImGui
# because the three things we care about -- live XML syntax highlighting, a
# click-to-inject snippet toolbar, and zero-friction build integration -- are all
# cheap in HTML/JS and expensive in an immediate-mode GUI (which would also drag
# in a windowing + GPU backend).
#
# GROUND TRUTH: instead of re-implementing the parser in JavaScript, the server
# compiles the REAL production parser (agent_orchestrator/tool_parser.cpp) and
# exposes it at /api/parse. tool_parser.cpp is dependency-free (only <map>,
# <string>, <string_view>, <optional>), so this target needs NEITHER agent_core,
# agent_env, tree-sitter, git, NOR CUDA. It stays tiny and fast to build.
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
    # The genuine ReAct parser -- compiled in directly so the debugger validates
    # against exactly what the orchestrator runs at inference time.
    ${CMAKE_CURRENT_SOURCE_DIR}/../../agent_orchestrator/tool_parser.cpp
)

target_compile_features(agent_playground PRIVATE cxx_std_20)

target_include_directories(agent_playground PRIVATE
    # for "tool_parser.h"
    ${CMAKE_CURRENT_SOURCE_DIR}/../../agent_orchestrator
)

# httplib::httplib is an INTERFACE target that also wires up ws2_32 / threads on
# Windows. nlohmann_json is already fetched by the top-level project, so reusing
# it adds no new download.
target_link_libraries(agent_playground PRIVATE
    httplib::httplib
    nlohmann_json::nlohmann_json
)
