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
7. **Runtime DLLs** are delivered by `POST_BUILD copy_if_different` next to the exe —
   reuse the existing pattern; never rely on PATH.
8. `project()` is declared **once**, at the root. (The extra one in `src/CMakeLists.txt`
   is debt #3.)

## Dependency management

- Header/source deps: `FetchContent` pinned to a release tag (googletest v1.14.0,
  nlohmann_json v3.11.3 pattern). No unpinned branches, and no new `TLS_VERIFY OFF`
  (the DirectStorage fetch's one is debt #7).
- One binary SDK remains vendored in `external/` (renderdoc, currently unreferenced by
  any target); don't add new vendored binaries without recording the acquisition source
  in a comment. (The vendored nvcomp was deleted 2026-07 together with the obsolete
  `blackwell_compress` experiment.)

<evolution_protocol>
**Current workaround:** `src/CMakeLists.txt` is a ~200-line monolith holding: the
`blackwell_core` target, the DirectStorage FetchContent block, the two executables, the
nvcomp compression experiment, and the option-gating for tools — with ordering
dependencies ("Added LAST so DS_BIN_DIR is in scope") that make it fragile.
**Already landed (2026-07):** the obsolete `blackwell_compress` experiment, the vendored
`external/nvcomp` binaries, the global `tests/reference` include, and the
`tests/reference` shim were all **deleted**; `src/experiments/` is now wired via
`add_subdirectory` with per-target scripts.
**Target state (the remaining split, Roadmap #3/#4 in CLAUDE.md):**
- `src/CMakeLists.txt` → orchestration only: options + `add_subdirectory()` calls.
- `src/core/CMakeLists.txt` → `blackwell_core` target + its (physically relocated)
  sources; `src/apps/CMakeLists.txt` → the two executables.
- `cmake/DirectStorage.cmake` → the whole NuGet fetch, exposing an imported/interface
  target `blackwell::dstorage` (kills the DS_BIN_DIR scope coupling; consumers get DLL
  paths from target properties).
- `blackwell_kernels` narrows its PUBLIC include to a dedicated public header dir (or an
  explicit `src/kernels` path), removing the `src/`-wide leak.
**When executing the split:** migrate one block per commit in this order — DirectStorage
module, core/apps extraction, kernels include-narrowing — and after each commit run
`.\build_target.bat blackwell_core && .\build_target.bat poc_overlay` plus
`ctest -L validation` as the acceptance gate (include-narrowing will surface missing
`target_include_directories` in consumers; fix those consumers, don't re-widen).
**Afterwards:** rewrite this skill — delete this protocol, replace the monolith references
with the final file map, and update CLAUDE.md's project map + Roadmap rows #3/#4.
</evolution_protocol>
