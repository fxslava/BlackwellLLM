# =============================================================================
# CopyRuntimeDeps.cmake — SCRIPT MODE (`cmake -P`), run as a POST_BUILD step.
#
# Deploys the runtime DLL closure of an executable next to it, by READING THE
# PE IMPORT TABLES rather than by listing imported CMake targets.
#
# WHY THIS EXISTS. `$<TARGET_RUNTIME_DLLS:tgt>` only knows what `tgt` itself
# links: it deployed libcurl and simdjson and stopped there. libcurl in turn
# imports nghttp2 and zlib, which are nobody's link dependency and therefore
# nobody's copy -- so the app linked, deployed, and then failed to START with a
# missing-DLL dialog naming a library that appears nowhere in the build files.
# That is the least diagnosable failure in the whole deployment.
#
# Under the vcpkg TOOLCHAIN this is invisible: VCPKG_APPLOCAL_DEPS does the same
# PE walk automatically. This repo finds curl/simdjson through
# CMAKE_PREFIX_PATH instead (no toolchain — see cmake/CloudDeps.cmake), so the
# walk has to be ours.
#
# Arguments (all -D on the cmake -P command line):
#   EXE   the built executable to scan
#   DIRS  ;-list of directories to resolve imports from (the dependency bin
#         dirs; $<TARGET_RUNTIME_DLL_DIRS:tgt> produces exactly this)
#   DEST  where to put what is found -- normally the exe's own directory
#
# NOT an error when something is unresolved: the exe's dependency set includes
# DLLs deployed by other steps (ONNXRuntime, cuDNN, DirectStorage) and delay-
# loaded CUDA libraries that are found on PATH. This step's job is the closure
# it CAN see, not a completeness audit.
# =============================================================================
cmake_minimum_required(VERSION 3.28)

# The POST_EXCLUDE_REGEXES below are written against forward slashes. Without
# this, dependencies come back with the separator the loader happened to use
# ("C:\Windows\system32/user32.dll") and matching is a coin flip. Guarded
# because the policy postdates this file's minimum.
if(POLICY CMP0207)
    cmake_policy(SET CMP0207 NEW)
endif()

if(NOT DEFINED EXE OR NOT DEFINED DEST)
    message(FATAL_ERROR "CopyRuntimeDeps.cmake: EXE and DEST are required")
endif()

file(GET_RUNTIME_DEPENDENCIES
    EXECUTABLES ${EXE}
    RESOLVED_DEPENDENCIES_VAR   _resolved
    UNRESOLVED_DEPENDENCIES_VAR _unresolved
    DIRECTORIES ${DIRS}
    # The API-set stubs are forwarders that never exist as files.
    PRE_EXCLUDE_REGEXES  "api-ms-.*" "ext-ms-.*"
    # Anything already living in a Windows system directory is the OS's to
    # provide -- copying it next to the exe is at best pointless and at worst a
    # version pin nobody asked for.
    # Spelled out per character because CMake's regex engine is case-SENSITIVE
    # and has no inline (?i) flag, while the loader reports this directory as
    # System32, system32 or SYSTEM32 depending on how it was reached.
    POST_EXCLUDE_REGEXES
        ".*/[Ss][Yy][Ss][Tt][Ee][Mm]32/.*"
        ".*/[Ss][Yy][Ss][Ww][Oo][Ww]64/.*"
)

get_filename_component(_dest_real "${DEST}" REALPATH)
foreach(_dll IN LISTS _resolved)
    get_filename_component(_dir "${_dll}" DIRECTORY)
    get_filename_component(_dir_real "${_dir}" REALPATH)
    # Already deployed by an earlier step (it resolved out of DEST itself).
    # Skipping is not just an optimisation: copying a file onto itself is what
    # would truncate it.
    if(_dir_real STREQUAL _dest_real)
        continue()
    endif()
    file(COPY "${_dll}" DESTINATION "${DEST}")
endforeach()
