# ==============================================================================
# WebView2.cmake — the Chromium (Edge WebView2) embedding SDK, fetched once and
# exported as ONE interface target: `blackwell::webview2`.
#
# WHY A MODULE. Two products now host a WebView2: poc_overlay's settings window
# and voice_assistant's whole main UI. FetchContent keys on the declaration NAME,
# so two `FetchContent_Declare(webview2_sdk ...)` calls in different directories
# silently resolve to whichever ran FIRST -- a version pin that depends on
# add_subdirectory ordering is a trap, not a build. One module, one pin.
#
# WHAT IT LINKS. The NuGet package is just a ZIP: headers plus a STATIC loader
# (`WebView2LoaderStatic.lib`), so no extra WebView2Loader.dll ships next to the
# exe. The static loader pulls `version.lib`, hence its inclusion here.
#
# WHAT IT DOES NOT DO. The WebView2 *runtime* is a machine-level dependency
# (it ships with modern Windows/Edge). A consumer must handle its absence at
# RUNTIME -- CreateCoreWebView2EnvironmentWithOptions fails and the app reports
# it -- because there is nothing a build system can assert about it.
#
# Windows/MSVC only; on any other platform the target is simply not defined and
# consumers must gate on `if(TARGET blackwell::webview2)`.
# ==============================================================================

if(TARGET blackwell::webview2)
    return()  # already provided (module included twice)
endif()

if(NOT WIN32)
    return()
endif()

include(FetchContent)

set(BLACKWELL_WEBVIEW2_VERSION "1.0.2792.45" CACHE STRING
    "Microsoft.Web.WebView2 NuGet package version")

message(STATUS "Fetching Microsoft.Web.WebView2 SDK ${BLACKWELL_WEBVIEW2_VERSION} from NuGet...")
FetchContent_Declare(
    webview2_sdk
    URL "https://www.nuget.org/api/v2/package/Microsoft.Web.WebView2/${BLACKWELL_WEBVIEW2_VERSION}"
    DOWNLOAD_NAME "webview2.zip"
    # TODO(debt #8): nuget.org serves a valid chain; this OFF is inherited from
    # the original poc_overlay fetch and is kept only so this refactor changes no
    # build behaviour. Flip it with the DirectStorage fetch, not in isolation.
    TLS_VERIFY OFF
)
FetchContent_MakeAvailable(webview2_sdk)
FetchContent_GetProperties(webview2_sdk SOURCE_DIR _WV2_DIR)

add_library(blackwell_webview2 INTERFACE)
# SYSTEM: the SDK headers are third-party and must not be measured against the
# first-party /W4 /WX budget (compiler-hygiene skill).
target_include_directories(blackwell_webview2 SYSTEM INTERFACE
    "${_WV2_DIR}/build/native/include")
target_link_libraries(blackwell_webview2 INTERFACE
    "${_WV2_DIR}/build/native/x64/WebView2LoaderStatic.lib"
    version)   # required by the static loader

add_library(blackwell::webview2 ALIAS blackwell_webview2)
