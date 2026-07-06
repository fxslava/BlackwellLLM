---
name: cmake-hygiene
description: Rules for editing BlackwellLLM's CMake build scripts — adding targets, sources, dependencies, or include paths, and the blueprint for breaking up the src/CMakeLists.txt monolith. Use whenever touching any CMakeLists.txt in this repo or adding a new library/executable/test target.
---

# CMake Hygiene — Target-Based Rules

The repo is mostly modern target-based CMake already; hold that line and fix the known
exceptions when you touch them (never extend them).

## Rules for any CMake edit

1. **Everything is target-scoped.** `target_include_directories` / `target_link_libraries`
   / `target_compile_definitions` with explicit `PUBLIC`/`PRIVATE`/`INTERFACE`. Never add
   global `include_directories()` / `link_directories()` / `add_definitions()` — the one
   existing global (`include_directories(tests/reference)` in the root) is Roadmap debt #4,
   not a precedent.
2. **PRIVATE by default.** `PUBLIC` only when the dependency's headers appear in *your*
   public headers. Narrow `PUBLIC` include dirs to what consumers actually need —
   `blackwell_kernels`'s `PUBLIC .../..` (all of `src/`) is debt #4, not a pattern.
3. **Explicit source lists for production targets** (kernels/core): GLOB is evaluated at
   configure time and silently drops new files — the kernels CMakeLists documents this.
   Test suites are the sanctioned exception: `file(GLOB ... CONFIGURE_DEPENDS)`.
4. **Options gate subdirectories** with documented `option()` + dependency conditions
   (pattern: `BUILD_AGENT_ORCHESTRATOR AND BUILD_AGENT_CORE AND BUILD_AGENT_ENV`). New
   optional components follow it. Keep the comment stating what the component is and what
   it deliberately does NOT link (the "no CUDA" annotations are load-bearing docs).
5. **Config-specific flags via generator expressions**
   (`$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:...>`), never
   `if(CMAKE_BUILD_TYPE ...)` — the latter is generator-dependent and already produced
   dead code in `src/kernels/CMakeLists.txt`.
6. **New test suites** go through `blackwell_add_test_suite()` in `tests/CMakeLists.txt`
   (CUDA-flavored: DLL copies, kernels+core links) or copy the minimal agent-suite pattern
   for CUDA-free targets. Always `gtest_discover_tests` with a `LABELS` property.
7. **Runtime DLLs** are delivered by `POST_BUILD copy_if_different` next to the exe.
   For DirectStorage, call `blackwell_copy_dstorage_dlls(<target>)` (defined
   unconditionally by `cmake/DirectStorage.cmake`; no-op when the SDK is off) — never
   hand-roll the copy or rely on PATH.
8. `project()` is declared **once**, at the root.

## Dependency management

- Header/source deps: `FetchContent` pinned to a release tag, declared in
  `cmake/Dependencies.cmake` (googletest v1.14.0, nlohmann_json v3.11.3 pattern). No
  unpinned branches, and never disable `TLS_VERIFY` (the last instance was removed
  2026-07). Optional binary SDKs follow the `cmake/DirectStorage.cmake` pattern: an
  INTERFACE target carrying usage requirements + a DLL-path target property + an
  unconditionally-defined copy helper.
- One binary SDK remains vendored in `external/` (renderdoc, currently unreferenced by
  any target); don't add new vendored binaries without recording the acquisition source
  in a comment. (The vendored nvcomp was deleted 2026-07 together with the obsolete
  `blackwell_compress` experiment.)

<evolution_protocol>
**Current workaround:** `src/CMakeLists.txt` is a ~200-line monolith holding: the
`blackwell_core` target, the DirectStorage FetchContent block, the two executables, the
nvcomp compression experiment, and the option-gating for tools — with ordering
dependencies ("Added LAST so DS_BIN_DIR is in scope") that make it fragile.
**Already landed (2026-07, Roadmap #3 done):** the monolith is split — `blackwell_core`
lives in `src/core/CMakeLists.txt` with its physically relocated sources, executables in
`src/apps/`, experiments per-target, deps in `cmake/Dependencies.cmake`, DirectStorage in
`cmake/DirectStorage.cmake` (`blackwell::dstorage` + `blackwell_copy_dstorage_dlls()`;
`DS_BIN_DIR` coupling and `TLS_VERIFY OFF` are gone); the obsolete `blackwell_compress`
experiment, vendored nvcomp, and the `tests/reference` shim were deleted;
`src/CMakeLists.txt` is orchestration-only (~60 lines).
**Include-dir convention for core-internal consumers** (tests, tools that reach past the
public API): add BOTH `${CMAKE_SOURCE_DIR}/src/core` (engine_impl.h, memory_pool.h,
paging/, ssm/, kv_cache/) and `${CMAKE_SOURCE_DIR}/src` (shared `common.h`,
`kernels/*.cuh` — paging headers include these) as PRIVATE include dirs. No `../../`
relative includes into the engine tree — that style was retired with the move.
**Target state (the last split item, Roadmap #4):** `blackwell_kernels` narrows its
`PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..` export (all of `src/` leaks to kernel linkers) to
a dedicated public header dir or an explicit `src/kernels` path.
**When executing the narrowing:** one commit; consumers that lose `src/` transitively
must declare their own PRIVATE include dirs (fix consumers, don't re-widen). Acceptance:
`.\build_target.bat blackwell_core && .\build_target.bat poc_overlay` +
`ctest -L validation`. Afterwards delete this protocol, fold the include-dir convention
into the rules above, and flip CLAUDE.md Roadmap #4 to done.
**Afterwards:** rewrite this skill — delete this protocol, replace the monolith references
with the final file map, and update CLAUDE.md's project map + Roadmap rows #3/#4.
</evolution_protocol>
