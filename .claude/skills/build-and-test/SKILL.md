---
name: build-and-test
description: Build BlackwellLLM targets and run the labeled CTest suites. Use whenever compiling any target (poc_overlay, blackwell_core, blackwell_llm, tests), verifying a change builds, running validation/benchmark/integration/agent tests, or diagnosing MSVC/CUDA environment build failures on this repo.
---

# Build & Test — BlackwellLLM

Windows-only tree: MSVC (VS 2022 Community) + CUDA v13.2, Ninja. The canonical entry
points are the committed **`CMakePresets.json`** presets; `out/build/x64-Debug` and
`out/build/x64-Release` are the preset binary dirs.

## Building

From a VS Developer prompt (or any vcvars64 shell):

```bat
cmake --preset x64-debug                                  :: configure (once, or after CMakeLists edits)
cmake --build --preset x64-debug --target <target>
```

From any other shell, the transitional wrapper does the environment setup
(vcvars64 + the CUDA v13.2 INCLUDE injection nvcc needs) and forwards to the preset:

```bat
.\build_target.bat <target>
```

Presets pin `CMAKE_CUDA_ARCHITECTURES=120` and `USE_DIRECT_STORAGE=ON` (matching the
flagship overlay workflow). A bare `cmake --build` in a non-vcvars shell fails — cl.exe
is not on PATH; that is why the wrapper exists.

Useful target names:

| Target | Notes |
|---|---|
| `blackwell_kernels` | CUDA kernel static lib (`src/kernels/`) |
| `blackwell_core` | engine static lib (`src/core/`; pulls kernels) |
| `blackwell_llm` / `blackwell_bench` | chat CLI / perplexity benchmark (`src/apps/`) |
| `poc_overlay` | overlay translator; ~1–2 min, POST_BUILD deploys `web/` + DS DLLs next to the exe |
| `agent_playground` | HTTP playground GUI (links core + orchestrator) |
| `agent_core`, `agent_env`, `agent_orchestrator` | CUDA-free agent stack |
| `awq_benchmark` | standalone experiment (`src/experiments/`) |
| `blackwell_cloud` | remote legs (Anthropic + OpenAI-compatible); `BUILD_CLOUD_CLIENT` (ON by default, needs libcurl + simdjson) |
| `validation_tests`, `benchmark_tests`, `integration_tests`, `cloud_tests`, `agent_tests`, `agent_env_tests`, `agent_orchestrator_tests` | test executables |

Build only the target you need — a full build is slow (CUDA), and test suites glob their
sources with `CONFIGURE_DEPENDS`, so a *new* test file is picked up on the next build
without a manual CMake re-run.

## Running tests

Test presets run from the repo root and map 1:1 onto the ctest labels:

```bat
ctest --preset validation            :: fast kernel-correctness vs CPU references
ctest --preset benchmark             :: stress shapes + CUDA-event performance
ctest --preset integration           :: engine vs PyTorch golden dumps
ctest --preset cloud                 :: remote-leg SSE/HTTP (needs BUILD_CLOUD_CLIENT=ON)
ctest --preset agent                 :: tree-sitter agent core
ctest --preset agent-env             :: sandbox OS layer
ctest --preset agent-orchestrator    :: ReAct loop (MockLLM)
```

(`ctest -L <label>` from `out/build/x64-Debug` still works; the presets add exact-label
anchoring — plain `-L agent` regex-matches all three agent suites.)

Sharp edges:
- **Build the test target first** — ctest does not build, and `gtest_discover_tests`
  registers tests at build time.
- **`cloud` needs libcurl + simdjson**, and `BUILD_CLOUD_CLIENT` now defaults **ON**, so
  a machine without them fails at CONFIGURE time (`find_package(CURL REQUIRED)`) rather
  than merely lacking `cloud_tests`. Either install them with vcpkg or configure with
  `-DBUILD_CLOUD_CLIENT=OFF` — see `cmake/CloudDeps.cmake` for the install line and why
  `-DCMAKE_PREFIX_PATH=<vcpkg>/installed/x64-windows` is preferred over the toolchain
  file on an existing build tree. No GPU and no network egress: the suite targets the
  loopback discard port and never contacts a billed endpoint.
- **Integration tests skip silently** when the local checkpoint / golden dumps are absent
  — a green integration run on a machine without models proves nothing. Check the output
  for skip messages. Regenerating dumps: skill `golden-dumps`.
- GPU suites (`validation`, `benchmark`, `integration`) need the Blackwell GPU free;
  close the overlay/playground first (the engine holds multi-GB VRAM unless hibernated).
- Test executables get the DirectStorage DLLs copied next to themselves via
  `blackwell_copy_dstorage_dlls()` — if a test exe fails to start with a missing-DLL
  error, rebuild the target rather than copying DLLs by hand.
- Run a single test directly for fast iteration:
  `out\build\x64-Debug\tests\validation_tests.exe --gtest_filter=Foo.*`

<migration_context>
Roadmap #7 landed 2026-07: `CMakePresets.json` (configure `x64-debug`/`x64-release`,
matching build presets, per-label test presets) replaced the IDE-generated-cache-only
workflow; the presets were verified to adopt the pre-existing `out/build/x64-Debug` tree
without a cache wipe. `build_target.bat` remains ONLY as the environment wrapper for
non-vcvars shells — the vcvars64/INCLUDE injection is still required because nvcc/cl
discovery is environmental, not cache-persisted. If a future CMake/toolchain change makes
the injection unnecessary (verify: bare `cmake --build --preset x64-debug` from a plain
shell), delete `build_target.bat` and the wrapper mentions here and in CLAUDE.md's
Quickstart.
</migration_context>
