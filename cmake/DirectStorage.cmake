# DirectStorage SDK acquisition (NuGet) + the blackwell::dstorage INTERFACE
# target. Included from the root CMakeLists AFTER the USE_DIRECT_STORAGE option
# is defined. Everything below is a no-op when the option is OFF -- but
# blackwell_copy_dstorage_dlls() is ALWAYS defined, so consumers call it
# unconditionally instead of carrying their own if(USE_DIRECT_STORAGE) guards.

# Copies the DirectStorage runtime DLLs next to <target>'s executable
# (POST_BUILD, copy_if_different). No-op when DirectStorage is disabled.
function(blackwell_copy_dstorage_dlls target)
    if(NOT TARGET blackwell_dstorage)
        return()
    endif()
    get_target_property(ds_bin blackwell_dstorage BLACKWELL_DSTORAGE_BIN_DIR)
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ds_bin}/dstorage.dll"
            "${ds_bin}/dstoragecore.dll"
            "$<TARGET_FILE_DIR:${target}>"
        COMMENT "Copying DirectStorage runtime DLLs next to ${target}...")
endfunction()

if(NOT (USE_DIRECT_STORAGE AND WIN32))
    return()
endif()

include(FetchContent)
message(STATUS "Fetching Microsoft DirectStorage SDK from NuGet...")
FetchContent_Declare(
    directstorage_sdk
    # v2 API endpoint: digested reliably by CMake's downloader.
    URL "https://www.nuget.org/api/v2/package/Microsoft.Direct3D.DirectStorage/1.3.0"
    # Force a .zip name so the extractor recognizes the NuGet package format.
    DOWNLOAD_NAME "directstorage.zip"
)
FetchContent_MakeAvailable(directstorage_sdk)
FetchContent_GetProperties(directstorage_sdk SOURCE_DIR DS_DIR)

# Pure usage-requirements target. Linking it (even PRIVATE) gives a consumer
# the USE_DIRECT_STORAGE define, the SDK headers, and the import lib; the
# runtime DLL directory rides along as a property for the copy helper above,
# so no directory-scope variable (the old DS_BIN_DIR) leaks between scripts.
add_library(blackwell_dstorage INTERFACE)
add_library(blackwell::dstorage ALIAS blackwell_dstorage)
target_compile_definitions(blackwell_dstorage INTERFACE USE_DIRECT_STORAGE)
target_include_directories(blackwell_dstorage INTERFACE "${DS_DIR}/native/include")
target_link_directories(blackwell_dstorage INTERFACE "${DS_DIR}/native/lib/x64")
target_link_libraries(blackwell_dstorage INTERFACE dstorage d3d12)
set_target_properties(blackwell_dstorage PROPERTIES
    BLACKWELL_DSTORAGE_BIN_DIR "${DS_DIR}/native/bin/x64")
message(STATUS "DirectStorage SDK integrated from: ${DS_DIR}")
