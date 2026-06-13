# ==============================================================================
# agent_core: the autonomous code-agent's compiler-level core.
#
# Deliberately ISOLATED from the GPU inference engine: this target links nothing
# from blackwell_core / blackwell_kernels and pulls in no CUDA. It owns the
# syntax engine (tree-sitter CST) and the semantic engine (shadow graph).
# ==============================================================================

# tree-sitter ships as C; the root project only enables CXX/CUDA, so turn C on
# here (scoped to where we actually need it).
enable_language(C)

include(FetchContent)

# --- tree-sitter runtime + tree-sitter-cpp grammar (plain C sources) ----------
# A non-existent SOURCE_SUBDIR stops FetchContent from running the upstream
# CMake projects (tests, bindings, fuzzers); we only want the C sources, which
# we compile ourselves into one small static lib.
FetchContent_Declare(
    tree_sitter
    GIT_REPOSITORY https://github.com/tree-sitter/tree-sitter.git
    GIT_TAG        v0.22.6
    GIT_SHALLOW    TRUE
    SOURCE_SUBDIR  no-upstream-cmake
)
FetchContent_Declare(
    tree_sitter_cpp
    GIT_REPOSITORY https://github.com/tree-sitter/tree-sitter-cpp.git
    GIT_TAG        v0.22.0
    GIT_SHALLOW    TRUE
    SOURCE_SUBDIR  no-upstream-cmake
)
FetchContent_MakeAvailable(tree_sitter tree_sitter_cpp)

set(TS_SOURCES ${tree_sitter_SOURCE_DIR}/lib/src/lib.c
               ${tree_sitter_cpp_SOURCE_DIR}/src/parser.c)
# Some grammars ship an external scanner; tree-sitter-cpp's is C.
if(EXISTS ${tree_sitter_cpp_SOURCE_DIR}/src/scanner.c)
    list(APPEND TS_SOURCES ${tree_sitter_cpp_SOURCE_DIR}/src/scanner.c)
endif()

add_library(treesitter STATIC ${TS_SOURCES})
target_include_directories(treesitter
    PUBLIC  ${tree_sitter_SOURCE_DIR}/lib/include
    PRIVATE ${tree_sitter_SOURCE_DIR}/lib/src
            ${tree_sitter_cpp_SOURCE_DIR}/src)
set_target_properties(treesitter PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
if(MSVC)
    # Upstream C generates a lot of conversion/POSIX-name noise; silence it but
    # do not treat as errors.
    target_compile_definitions(treesitter PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(treesitter PRIVATE /wd4018 /wd4244 /wd4267 /wd4996 /wd4146)
endif()

# --- agent_core ---------------------------------------------------------------
add_library(agent_core STATIC
    cst_manager.cpp
    semantic_engine.cpp
)
target_compile_features(agent_core PUBLIC cxx_std_20)
target_include_directories(agent_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(agent_core PUBLIC treesitter)
