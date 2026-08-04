# Shared build helpers.

# Copies each dependency target's runtime artifact (DLL) next to <target>'s
# executable (POST_BUILD, copy_if_different). Use for in-tree SHARED libraries
# the executable loads at startup -- e.g. blackwell_core.dll:
#   blackwell_copy_runtime_dlls(blackwell_llm blackwell_core)
function(blackwell_copy_runtime_dlls target)
    foreach(dep IN LISTS ARGN)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:${dep}>"
                "$<TARGET_FILE_DIR:${target}>"
            COMMENT "Copying $<TARGET_FILE_NAME:${dep}> next to ${target}...")
    endforeach()
endfunction()

# Deploy the FULL runtime DLL closure of <target> next to its executable.
#
# Two passes, because neither is sufficient alone:
#   1. $<TARGET_RUNTIME_DLLS> — every imported SHARED library <target> links.
#      Knows the first level only: it deployed libcurl and simdjson and stopped,
#      while libcurl in turn imports nghttp2 and zlib.
#   2. CopyRuntimeDeps.cmake — walks the PE import tables of the built exe for
#      everything pass 1 could not know about. See that file.
#
# Under the vcpkg TOOLCHAIN this is all automatic (VCPKG_APPLOCAL_DEPS); this
# tree finds its cloud deps through CMAKE_PREFIX_PATH instead, so the walk is
# ours. Symptom when it is missing: the target links and deploys fine, then
# fails to START with a missing-DLL dialog naming a library that appears nowhere
# in the build files.
#
# The $<IF:...> guard is the documented idiom for an empty list: with everything
# statically linked the genex expands to nothing, and `cmake -E copy_if_different`
# with no arguments is an error.
function(blackwell_copy_runtime_deps target)
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E
            $<IF:$<BOOL:$<TARGET_RUNTIME_DLLS:${target}>>,copy_if_different,true>
            $<TARGET_RUNTIME_DLLS:${target}> "$<TARGET_FILE_DIR:${target}>"
        COMMAND ${CMAKE_COMMAND}
            -DEXE=$<TARGET_FILE:${target}>
            "-DDIRS=$<JOIN:$<TARGET_RUNTIME_DLL_DIRS:${target}>,;>"
            -DDEST=$<TARGET_FILE_DIR:${target}>
            -P "${CMAKE_SOURCE_DIR}/cmake/CopyRuntimeDeps.cmake"
        COMMAND_EXPAND_LISTS
        COMMENT "Copying runtime DLL closure next to ${target}...")
endfunction()

# Deploy an asset DIRECTORY (data files, web UI, models) next to <target>'s
# executable as <exe dir>/<dst_name>.
#
# WHY: an app that opens "data/mel_filters.bin" resolves it against the CALLER's
# working directory, so it launches from a VS F5 (which CMake anchors via
# VS_DEBUGGER_WORKING_DIRECTORY) and dies from Explorer, from a shortcut, or from
# the output folder itself. Putting the assets next to the binary lets the app
# resolve them from its OWN module path and stop caring who launched it.
# copy_directory_if_different keeps incremental rebuilds from re-copying.
function(blackwell_copy_asset_dir target src_dir dst_name)
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
            "${src_dir}"
            "$<TARGET_FILE_DIR:${target}>/${dst_name}"
        COMMENT "Deploying ${dst_name}/ next to ${target}...")
endfunction()

# Put the CUDA toolkit's bin directory on the DEBUGGER's PATH for <target>, so
# F5 from Visual Studio resolves cublas64_*.dll / cublasLt64_*.dll even on a box
# where the toolkit is not on the system PATH.
#
# WHY NOT COPY THEM. cublasLt64_13.dll alone is 432 MB; copying the pair next to
# every executable would add ~half a gigabyte per target to the build tree to
# solve a problem the loader already solves from PATH. Deployment to a machine
# with no CUDA toolkit is a PACKAGING concern (the NVIDIA redistributable), not a
# dev-loop one -- see BLACKWELL_DEPLOY_CUDA_DLLS below for that case.
#
# No-op when the toolkit's bin dir is unknown or the generator is not VS.
function(blackwell_set_cuda_debugger_path target)
    if(NOT MSVC)
        return()
    endif()
    if(NOT DEFINED CUDAToolkit_BIN_DIR)
        find_package(CUDAToolkit QUIET)
    endif()
    if(NOT CUDAToolkit_BIN_DIR)
        return()
    endif()
    file(TO_NATIVE_PATH "${CUDAToolkit_BIN_DIR}" _cuda_bin)
    # x64/ holds the 64-bit math libraries on CUDA 13+; harmless when absent.
    file(TO_NATIVE_PATH "${CUDAToolkit_BIN_DIR}/x64" _cuda_bin_x64)
    set_property(TARGET ${target} PROPERTY
        VS_DEBUGGER_ENVIRONMENT "PATH=${_cuda_bin_x64};${_cuda_bin};%PATH%")
endfunction()

# Opt-in: copy the cuBLAS runtime next to <target> so the output directory is
# self-contained on a machine with no CUDA toolkit installed. OFF by default --
# it costs ~481 MB per target. Use for producing a shippable folder, not for
# day-to-day development.
option(BLACKWELL_DEPLOY_CUDA_DLLS
       "Copy the cuBLAS runtime DLLs (~481 MB) next to each executable" OFF)

function(blackwell_copy_cuda_dlls target)
    if(NOT BLACKWELL_DEPLOY_CUDA_DLLS OR NOT WIN32)
        return()
    endif()
    if(NOT DEFINED CUDAToolkit_BIN_DIR)
        find_package(CUDAToolkit QUIET)
    endif()
    if(NOT CUDAToolkit_BIN_DIR)
        message(WARNING "BLACKWELL_DEPLOY_CUDA_DLLS is ON but CUDAToolkit_BIN_DIR is unset")
        return()
    endif()
    # Glob at CONFIGURE time: the exact soname carries the CUDA major version
    # (cublas64_12 vs cublas64_13), so hard-coding it breaks on a toolkit bump.
    file(GLOB _cublas
        "${CUDAToolkit_BIN_DIR}/cublas64_*.dll" "${CUDAToolkit_BIN_DIR}/cublasLt64_*.dll"
        "${CUDAToolkit_BIN_DIR}/x64/cublas64_*.dll" "${CUDAToolkit_BIN_DIR}/x64/cublasLt64_*.dll")
    foreach(_dll IN LISTS _cublas)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_dll}" "$<TARGET_FILE_DIR:${target}>"
            COMMENT "Copying CUDA runtime next to ${target}...")
    endforeach()
endfunction()
