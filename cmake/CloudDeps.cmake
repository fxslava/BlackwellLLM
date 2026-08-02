# =============================================================================
# CloudDeps.cmake — libcurl + simdjson for the option-gated blackwell_cloud
# target (Anthropic Messages API client).
#
# find_package, NOT FetchContent, on purpose. Every other pinned dependency in
# this repo (gtest, nlohmann_json, DirectStorage, ONNXRuntime) is either tiny or
# a prebuilt binary drop. libcurl is neither: building it from source drags in
# zlib/nghttp2/TLS-backend configuration, and a from-source curl on Windows is a
# reliable way to end up with a client that silently lacks HTTP/2 or trusts no
# CA at all. Take the system/vcpkg build, and fail loudly if it is missing.
#
# On Windows the expected provider is vcpkg:
#     vcpkg install curl[core,ssl,http2]:x64-windows simdjson:x64-windows
# then configure with
#     -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake
#
# Defines: blackwell::cloud_deps (INTERFACE) — link this, nothing else.
# =============================================================================

function(blackwell_setup_cloud_deps)
    if(TARGET blackwell::cloud_deps)
        return()
    endif()

    find_package(CURL REQUIRED)
    find_package(simdjson CONFIG REQUIRED)

    add_library(blackwell_cloud_deps INTERFACE)
    target_link_libraries(blackwell_cloud_deps INTERFACE CURL::libcurl simdjson::simdjson)
    add_library(blackwell::cloud_deps ALIAS blackwell_cloud_deps)

    # HTTP/2 is requested via CURLOPT_HTTP_VERSION. A libcurl built without
    # nghttp2 accepts the option and silently falls back to HTTP/1.1, so the
    # only way to know is to check the build's feature set now. Not fatal --
    # keep-alive carries most of the benefit for a single sequential stream --
    # but it must not be a silent surprise later.
    if(DEFINED CURL_VERSION_STRING)
        message(STATUS "blackwell_cloud: libcurl ${CURL_VERSION_STRING}")
    endif()
    if(CURL_SUPPORTS_HTTP2)
        message(STATUS "blackwell_cloud: libcurl reports HTTP/2 support")
    else()
        message(STATUS
            "blackwell_cloud: could not confirm HTTP/2 in this libcurl build. "
            "CURLOPT_HTTP_VERSION will fall back to 1.1 silently if nghttp2 is absent; "
            "verify with curl_version_info() at runtime if HTTP/2 matters.")
    endif()
endfunction()
