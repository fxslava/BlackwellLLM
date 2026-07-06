---
name: build-and-test
description: Build BlackwellLLM targets and run the labeled CTest suites. Use whenever compiling any target (poc_overlay, blackwell_core, blackwell_llm, tests), verifying a change builds, running validation/benchmark/integration/agent tests, or diagnosing MSVC/CUDA environment build failures on this repo.
---

# Build & Test — BlackwellLLM

Windows-only tree: MSVC (VS 2022 Community) + CUDA v13.2, cache generated from the VS IDE
into `out/build/x64-Debug` (Release lives in `out/build/x64-Release`, separate cache).

## Building a target

Always build through the wrapper — a bare `cmake --build` in a fresh shell fails because
nvcc needs vcvars64 plus the CUDA include path injected:

```bat
.\build_target.bat <target>
```

What it does (see `build_target.bat`): calls
`"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"`,
appends `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\include` to `INCLUDE`,
then runs `cmake --build out/build/x64-Debug --target <target>`.

Useful target names:

| Target | Notes |
|---|---|
| `blackwell_kernels` | CUDA kernel static lib |
| `blackwell_core` | engine static lib (pulls kernels) |
| `blackwell_llm` / `blackwell_bench` | chat CLI / perplexity benchmark |
| `poc_overlay` | overlay translator; ~1–2 min, POST_BUILD deploys `src/tools/poc_overlay/web/` next to the exe |
| `agent_core`, `agent_env`, `agent_orchestrator` | CUDA-free agent stack |
| `validation_tests`, `benchmark_tests`, `integration_tests`, `agent_tests`, `agent_env_tests`, `agent_orchestrator_tests` | test executables |

Build only the target you need — a full build is slow (CUDA), and test suites glob their
sources with `CONFIGURE_DEPENDS`, so a *new* test file is picked up on the next build
without re-running CMake.

## Running tests

From the build tree:

```bat
cd out\build\x64-Debug
ctest -L validation            :: fast kernel-correctness vs CPU references
ctest -L benchmark             :: stress shapes + CUDA-event performance
ctest -L integration           :: engine vs PyTorch golden dumps
ctest -L agent                 :: tree-sitter agent core
ctest -L agent_env             :: sandbox OS layer
ctest -L agent_orchestrator    :: ReAct loop (MockLLM)
```

Sharp edges:
- **Integration tests skip silently** when the local checkpoint / golden dumps are absent
  — a green integration run on a machine without models proves nothing. Check the test
  output for skip messages. Regenerating dumps: skill `golden-dumps`.
- GPU suites (`validation`, `benchmark`, `integration`) need the Blackwell GPU free;
  close the overlay/playground first (the engine holds multi-GB VRAM unless hibernated).
- Test executables copy `nvcomp64_5.dll` (and DirectStorage DLLs when enabled) next to
  themselves via POST_BUILD — if a test exe fails to start with a missing-DLL error,
  rebuild the target rather than copying DLLs by hand.
- Run a single test directly for fast iteration:
  `out\build\x64-Debug\tests\validation_tests.exe --gtest_filter=Foo.*`

<evolution_protocol>
**Current workaround:** `build_target.bat` exists because the CMake cache is IDE-generated
and the CLI environment (vcvars64 + CUDA INCLUDE) is not reproducible from a fresh shell.
**Target state:** a committed `CMakePresets.json` with configure presets (`x64-debug`,
`x64-release`), build presets, and test presets carrying the ctest labels — so the canonical
commands become `cmake --preset x64-debug`, `cmake --build --preset x64-debug --target <t>`,
`ctest --preset validation`.
**When CMakePresets.json lands (Roadmap #6 in CLAUDE.md):** rewrite this skill —
(1) replace the `build_target.bat` section with the preset commands and preset names taken
from the actual committed file; (2) verify whether the vcvars64/INCLUDE injection is still
needed (presets can set environment) and delete the environment section if not;
(3) keep the sharp-edges list, re-verifying each item; (4) update CLAUDE.md's Quickstart
to match; (5) only then propose deleting `build_target.bat`.
</evolution_protocol>
