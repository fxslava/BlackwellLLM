# =============================================================================
# blackwell_cloud — the remote legs of the Local Router: the Anthropic Messages
# API and any OpenAI-compatible /chat/completions endpoint.
#
# CUDA-free and engine-free. Its only coupling to the rest of the tree is
# src/bridge/intent_commit.hpp (header-only, no link dependency), so the
# dependency direction is cloud -> bridge and never the reverse: the engine has
# no idea a cloud exists, and cannot be made to wait on one.
#
# Explicit source list (src/kernels/ convention; never file(GLOB)).
# =============================================================================
add_library(blackwell_cloud STATIC
    # The ONLY two TUs that see libcurl + simdjson. Everything protocol-agnostic
    # (the poll loop, the watchdogs, SSE framing) is shared between them via
    # curl_stream_core.hpp / sse_framer.hpp rather than duplicated.
    claude_stream_client.cpp
    openai_stream_client.cpp)

target_include_directories(blackwell_cloud
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}      # *_stream_client.hpp, intent_dispatcher.hpp
    PRIVATE ${CMAKE_SOURCE_DIR}/src/bridge)  # intent_commit.hpp (header-only)

# intent_dispatcher.hpp is a PUBLIC header that includes intent_commit.hpp, so
# consumers need the bridge include path too. Exported as an INTERFACE path
# rather than PUBLIC above, to keep the rest of src/bridge out of this target's
# own compile line.
target_include_directories(blackwell_cloud INTERFACE ${CMAKE_SOURCE_DIR}/src/bridge)

target_link_libraries(blackwell_cloud PUBLIC blackwell::cloud_deps)

# /utf-8 so the R"(...)" JSON literals and the RU/EN comments survive MSVC's
# codepage. Zero-warning policy (roadmap #11) applies: first-party CXX only.
if(MSVC)
    target_compile_options(blackwell_cloud PRIVATE /utf-8)
endif()
