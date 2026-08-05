# =============================================================================
# WebRTC AEC3 acquisition + the blackwell::webrtc_aec3 INTERFACE target.
#
# Included from the root CMakeLists AFTER USE_WEBRTC_AEC3 is defined. Everything
# below is a no-op when the option is OFF or the package is absent, and the
# audio stack falls back to BlackwellLLM's own BlockFdafEchoCanceller -- which is
# why nothing here is FATAL_ERROR. A missing AEC3 must degrade the canceller,
# not stop the build.
#
# =============================================================================
# WHY THIS IS find_package AND NOT FetchContent
# =============================================================================
# Every other third-party dependency in this repo is fetched and hash-pinned
# (Dependencies.cmake, OnnxRuntime.cmake, DirectStorage.cmake). WebRTC cannot
# follow that pattern and should not be made to: it does not build with CMake at
# all -- it is a GN/Ninja tree with a depot_tools-shaped checkout, a dozen of its
# own third-party dependencies, and a multi-gigabyte source drop. Reproducing
# that inside FetchContent would mean vendoring a build system.
#
# So it is consumed as a PREBUILT package, and the pinning moves to the package
# manager:
#
#     vcpkg install webrtc:x64-windows-static ^
#         --overlay-triplets=<repo>/cmake/vcpkg-triplets
#
# THE OVERLAY IS NOT OPTIONAL, even though the triplet name is the stock one.
# vcpkg selects the newest Visual Studio instance on the box; this repo pins
# "Visual Studio 17 2022". On a machine that also has VS 18, those disagree and
# WebRTC gets built by a toolset whose STL this repo cannot link against -- see
# the header of cmake/vcpkg-triplets/x64-windows-static.cmake for the failure and
# why the dependency is made to follow the project rather than the box.
#
# =============================================================================
# THE TRIPLET IS NOT OPTIONAL, AND THE FAILURE IS A LINK ERROR
# =============================================================================
# x64-windows-STATIC, not x64-windows. The root CMakeLists pins a static CRT
# (CMAKE_MSVC_RUNTIME_LIBRARY = MultiThreaded$<$<CONFIG:Debug>:Debug>) for every
# first-party target, and vcpkg's default x64-windows triplet builds against the
# DYNAMIC CRT. Mixing them is not a warning: it is LNK2038 on
# RuntimeLibrary=MT_StaticRelease vs MD_DynamicRelease, and it surfaces at the
# very end of a very long build. The check below turns that into a message at
# configure time instead.
# =============================================================================

set(BLACKWELL_HAVE_WEBRTC_AEC3 OFF CACHE INTERNAL "" FORCE)

if(NOT USE_WEBRTC_AEC3)
    message(STATUS "WebRTC AEC3: disabled (USE_WEBRTC_AEC3=OFF) -- using the built-in canceller")
    return()
endif()

# CONFIG mode only. The port ships unofficial-webrtcConfig.cmake, which pulls its
# own find_dependency() chain (abseil, aom, libvpx, libyuv, opus, openssl,
# libsrtp, jsoncpp, pffft) -- a hand-rolled find module would have to reproduce
# all of it and would get the link ORDER wrong, which on a static build is the
# difference between linking and several hundred unresolved symbols.
find_package(unofficial-webrtc CONFIG QUIET)

if(NOT TARGET unofficial::webrtc::webrtc)
    message(STATUS
        "WebRTC AEC3: package not found -- falling back to the built-in canceller.\n"
        "             To enable it:\n"
        "               vcpkg install webrtc:x64-windows-static "
        "--overlay-triplets=${CMAKE_SOURCE_DIR}/cmake/vcpkg-triplets\n"
        "               cmake --preset x64-release-aec3\n"
        "             (the -static triplet AND the overlay are both required;\n"
        "              see cmake/WebRtcAec3.cmake)")
    return()
endif()

# A pure usage-requirements target, matching blackwell::onnxruntime's shape: the
# consumer links ONE name and inherits the include path, the compile definitions
# WebRTC's headers require, and the whole transitive archive list.
add_library(blackwell_webrtc_aec3 INTERFACE)
add_library(blackwell::webrtc_aec3 ALIAS blackwell_webrtc_aec3)
target_link_libraries(blackwell_webrtc_aec3 INTERFACE unofficial::webrtc::webrtc)

# SYSTEM, unlike our own headers. WebRTC does not compile clean under /W4 and it
# is not ours to fix -- quarantining it here is the same treatment
# compiler-hygiene prescribes for every other third-party include tree, and it
# keeps the zero-warning policy pointed at first-party code.
get_target_property(_webrtc_inc unofficial::webrtc::webrtc INTERFACE_INCLUDE_DIRECTORIES)
if(_webrtc_inc)
    target_include_directories(blackwell_webrtc_aec3 SYSTEM INTERFACE ${_webrtc_inc})
endif()
unset(_webrtc_inc)

set(BLACKWELL_HAVE_WEBRTC_AEC3 ON CACHE INTERNAL "" FORCE)
message(STATUS "WebRTC AEC3: enabled (unofficial::webrtc::webrtc)")
