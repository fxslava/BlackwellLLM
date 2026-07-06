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
