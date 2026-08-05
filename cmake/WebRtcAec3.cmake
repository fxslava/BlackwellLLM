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

# =============================================================================
# THE CRT-VINTAGE GATE, and why it is worth failing configure over.
#
# The package is present, but "present" is not "linkable". WebRTC is a PREBUILT
# static library, so its objects already contain calls to whatever STL entry
# points its own toolset inlined -- and MSVC's separately compiled STL grows
# those over time: __std_min_element_f, __std_max_element_f,
# __std_minmax_element_f, __std_find_first_of_trivial_1 and friends live in
# libcpmt.lib and simply do not exist in an older one.
#
# MSBuild picks the CRT from the v143 default file below, NOT from the newest
# toolset installed. When that file is older than the toolset vcpkg built WebRTC
# with, every one of those calls becomes LNK2001 on a symbol nobody in this repo
# wrote -- reported at the END of a link that follows a very long build, with
# nothing in the message pointing at the toolset.
#
# 14.40 is the floor: it is the first VS 2022 toolset carrying the vectorized
# algorithm entry points current WebRTC builds reference. Checked HERE rather
# than at the root because this is the only configuration that can hit it -- a
# tree without AEC3 links no prebuilt third-party archives and does not care.
# =============================================================================
# The EFFECTIVE toolset: the pin when one is set (it writes Directory.Build.props
# into the binary dir, which MSBuild obeys -- see the root CMakeLists for the two
# mechanisms that do NOT work), otherwise whatever the v143 default file names.
# Both are checked against the same floor, so silencing this by pinning is only
# possible with a pin that genuinely changes what gets linked.
option(BLACKWELL_ALLOW_OLD_TOOLSET
       "Skip the WebRTC AEC3 CRT-vintage gate. Escape hatch; the build will very likely fail at link." OFF)

if(MSVC AND CMAKE_GENERATOR MATCHES "Visual Studio")
    set(_bw_default_file
        "${CMAKE_GENERATOR_INSTANCE}/VC/Auxiliary/Build/Microsoft.VCToolsVersion.v143.default.txt")
    if(BLACKWELL_MSVC_TOOLSET_PIN)
        set(_bw_ts "${BLACKWELL_MSVC_TOOLSET_PIN}")
        set(_bw_ts_origin "BLACKWELL_MSVC_TOOLSET_PIN")
    else()
        set(_bw_ts "")
        set(_bw_ts_origin "${_bw_default_file}")
        if(EXISTS "${_bw_default_file}")
            file(READ "${_bw_default_file}" _bw_ts)
            string(STRIP "${_bw_ts}" _bw_ts)
        endif()
    endif()

    # A pin naming a toolset that is not installed would resolve to nothing at
    # build time, which is a worse failure than the one this gate exists for.
    if(BLACKWELL_MSVC_TOOLSET_PIN AND
       NOT IS_DIRECTORY "${CMAKE_GENERATOR_INSTANCE}/VC/Tools/MSVC/${BLACKWELL_MSVC_TOOLSET_PIN}")
        message(FATAL_ERROR
            "BLACKWELL_MSVC_TOOLSET_PIN=${BLACKWELL_MSVC_TOOLSET_PIN} is not installed under\n"
            "  ${CMAKE_GENERATOR_INSTANCE}/VC/Tools/MSVC/\n"
            "Pin one of the directories that exist there, or clear the pin.")
    endif()

    if(_bw_ts AND _bw_ts VERSION_LESS 14.40 AND NOT BLACKWELL_ALLOW_OLD_TOOLSET)
        message(FATAL_ERROR
            "WebRTC AEC3: MSBuild will build this tree with MSVC ${_bw_ts}, which cannot link\n"
            "             the prebuilt WebRTC in this vcpkg tree. That toolset's libcpmt.lib\n"
            "             does not export the STL entry points WebRTC's objects already call,\n"
            "             so the build would run to completion and then fail with LNK2001 on\n"
            "             __std_min_element_f, __std_max_element_f, __std_minmax_element_f and\n"
            "             __std_find_first_of_trivial_1. Minimum is 14.40.\n"
            "\n"
            "             Selected by: ${_bw_ts_origin}\n"
            "             Installing a newer compiler beside it does NOT change that file --\n"
            "             MSBuild reads it for PlatformToolset v143 regardless.\n"
            "\n"
            "             Pin this build tree to an installed toolset (writes\n"
            "             Directory.Build.props; works from the IDE too):\n"
            "               -D BLACKWELL_MSVC_TOOLSET_PIN=<x.y.z>\n"
            "             Or build without echo cancellation:  -D USE_WEBRTC_AEC3=OFF")
    endif()
    if(_bw_ts)
        message(STATUS "WebRTC AEC3: MSBuild toolset ${_bw_ts} (>= 14.40 required, via ${_bw_ts_origin})")
    endif()
    unset(_bw_ts)
    unset(_bw_ts_origin)
    unset(_bw_default_file)
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
