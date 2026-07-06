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
   public headers. **No target exports `src/`** — core-internal consumers (tests, tools
   reaching past the public API) declare BOTH `PRIVATE ${CMAKE_SOURCE_DIR}/src/core`
   (engine_impl.h, memory_pool.h, paging/, ssm/, kv_cache/) and
   `PRIVATE ${CMAKE_SOURCE_DIR}/src` (shared `common.h`, `kernels/*.cuh` — the paging
   headers include these). No `../../` relative includes into the engine tree.
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

<migration_context>
**Build-system stabilization is COMPLETE (2026-07, Roadmap #4–#8 done)** and the
**COM-style DLL boundary landed (Roadmap #1, 2026-07)**. The established two-tier core:

- `blackwell_core_obj` (OBJECT lib, `src/core/CMakeLists.txt`) — ALL engine sources; the
  full-trust white-box surface. Linked by: the test suites, `poc_overlay`,
  `agent_playground` (they drive the prefill-coordinator/paging C++ API the COM interface
  does not yet cover). kernels + dstorage are `PUBLIC` on it (their objects/import libs
  must reach every final link); nlohmann_json is `PRIVATE`.
- `blackwell_core` (SHARED DLL) — single own TU `engine_com.cpp`, sole exports = the C
  factories (`BLACKWELL_CORE_EXPORTS` → `__declspec(dllexport)`). No
  `WINDOWS_EXPORT_ALL_SYMBOLS`, ever — narrowness is the point. Linked by: `src/apps/`
  only, which get the DLL via `blackwell_copy_runtime_dlls(<exe> blackwell_core)`
  (`cmake/BlackwellHelpers.cmake`).

Rules when touching this: a target either links the DLL (COM consumer: boundary header
only) or the OBJECT lib (white-box: internal headers allowed) — NEVER both (two copies of
the engine code). New DLL exports go through `engine_com.cpp` + the boundary header, not
new dllexport sites. When the interface grows to cover the coordinator surface, migrate
poc_overlay/playground from `blackwell_core_obj` to the DLL and update this section.
</migration_context>
**Afterwards:** rewrite this skill — delete this protocol, replace the monolith references
with the final file map, and update CLAUDE.md's project map + Roadmap rows #3/#4.
</evolution_protocol>
