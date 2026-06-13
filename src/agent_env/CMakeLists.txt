# ==============================================================================
# agent_env: the autonomous agent's sandboxed "operating system" layer.
#
# Gives the LLM safe "hands" -- file I/O, subprocess execution and git/worktree
# management -- without letting a hallucinated `rm -rf` or a bad patch reach the
# host or the main repository. Deliberately ISOLATED from the GPU stack: it links
# nothing from blackwell_core / blackwell_kernels and pulls in no CUDA, mirroring
# the agent_core boundary.
#
# ------------------------------------------------------------------------------
# Architectural decisions
# ------------------------------------------------------------------------------
# Subprocess execution -> a hand-rolled, dependency-free wrapper over the native
#   process API (Win32 CreateProcess; fork/exec on POSIX), NOT reproc / Boost.
#   Process / a third-party lib. Rationale: zero FetchContent network dependency
#   (the rest of this tree already fights flaky downloads -- see DirectStorage's
#   TLS_VERIFY OFF), perfect separate stdout/stderr capture, exact exit codes and
#   an argv-only interface with no shell parsing, so the agent cannot smuggle
#   `&& rm -rf` through an argument. This is the single choke point for git,
#   ninja, cmake and the test binaries.
#
# Git handling -> drive the git *CLI* through that same subprocess wrapper, NOT
#   libgit2. Rationale: the workflow is worktree-centric (`git worktree add` /
#   `remove`), which libgit2 supports poorly; the CLI is the ground truth a human
#   would run, sidesteps libgit2's ABI/versioning headaches, and keeps CI/CD
#   green with zero extra binary dependencies. Pragmatic and zero-headache.
# ==============================================================================

add_library(agent_env STATIC
    sandbox_fs.cpp
    subprocess.cpp
    workspace.cpp
)

target_compile_features(agent_env PUBLIC cxx_std_20)
target_include_directories(agent_env PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# The pipe-draining reader threads need a threading runtime (no-op import on
# MSVC, -pthread on POSIX toolchains).
find_package(Threads REQUIRED)
target_link_libraries(agent_env PUBLIC Threads::Threads)

if(WIN32)
    target_compile_definitions(agent_env PRIVATE
        _CRT_SECURE_NO_WARNINGS WIN32_LEAN_AND_MEAN NOMINMAX)
endif()
