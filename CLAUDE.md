# BlackwellLLM

A from-scratch CUDA LLM inference engine for Blackwell GPUs (`CMAKE_CUDA_ARCHITECTURES 120`),
C++20 / CUDA, Windows + MSVC (VS 2022). Tuned for **batch = 1, latency-critical** desktop
workloads: a background app that wakes on a hotkey, translates, and goes back to sleep.

> **Bridge document.** A major refactoring is in progress (see the Roadmap register at the
> bottom). This file describes both the *current* state and the *target* state. When a
> roadmap item lands, update the matching section here and in the affected skill under
> `.claude/skills/` — each file carries its own `<evolution_protocol>` telling you exactly
> what to rewrite.

## Project map

| Subsystem | Location | CUDA? | What it is |
|---|---|---|---|
| `blackwell_kernels` | `src/kernels/` | yes | Hand-written kernels: attention, paged-flash, AWQ/FP8/INT4/BF16 GEMV, RoPE, SSM, sampling. One `.cu` + `.cuh` pair per kernel, explicit source list in CMake. |
| `blackwell_core` | `src/` | yes | The engine: `engine.cpp` (facade + Impl), `memory_pool` (VRAMArena, offloading, hibernation), `kv_cache/` (Continuous vs Paged strategies), `paging/` (header-only tiered KV + prefix cache + radix tree), `ssm/`, `tokenizer/`, `safetensors`, `weight_loader`, `config`. |
| Public API | `include/blackwell/` | no | The only headers external consumers may include: `engine.h`, `config.h`, `runtime_config.h`, `tokenizer.h`, `chat_template.h`, `weight_loader.h`. |
| Agent stack | `src/agent/`, `src/agent_env/`, `src/agent_orchestrator/` | **no** | Tree-sitter CST analysis, sandbox OS layer (VFS/subprocess/git), ReAct orchestrator. Deliberately CUDA-free; talks to the model via the `ILLMGenerator` text interface. |
| Tools | `src/tools/playground/`, `src/tools/poc_overlay/` | links core | HTTP playground GUI; Win32/UIA/Direct2D overlay translator (the flagship consumer). |
| Tests | `tests/` | mixed | Labeled CTest suites — see Quickstart. `tests/common/` holds shared fixtures (CPU references, CUDA fixtures). |
| Design docs | `docs/` | — | Read these before touching the corresponding subsystem (list below). |
| Python scripts | `scripts/` | — | PyTorch golden-dump generators + CLI chat reference. Checkpoint roots resolve via `BLACKWELL_MODELS_DIR` (default `F:/AI`). |

## The single-threaded engine doctrine (the most important rule)

**The engine control plane is single-threaded. Exactly one thread may touch a
`BlackwellEngine` after load; every other thread marshals work onto it.**

- In `poc_overlay`, that thread is `LiveTranslationTracker`'s worker — the *sole* thread
  permitted to call `forward()`, `hibernate()`, `spill_kv_cache()`, prefill, everything.
  New engine-facing features must go through `LiveTranslationTracker::PostEngineTask()`.
- `TrackUpdate` / `TriggerGeneration` / `Cancel` are O(1) fire-and-forget enqueues, callable
  from any thread. Interruption is a monotone generation counter + latest-wins job slot.
- Callbacks fired from the worker thread must **only enqueue/marshal** — never touch a
  window, COM, or the engine directly.
- `VRAMArena::hibernate()/wakeup()` and the whole paging substrate share this contract
  ("Not thread-safe; call from the single engine-owning thread" — `src/memory_pool.h`).

During the refactoring, convert this doctrine from comment-folklore into a checked
invariant: capture `std::this_thread::get_id()` when the engine finishes loading and add
debug-only asserts to every public engine entry point. This is also our pragmatic TSan
substitute (MSVC has no TSan).

## Extension patterns — the law for new engine features

These three patterns are already in the code and are non-negotiable for extensions:

1. **Capability gating.** Never guess what a loaded model supports — query
   `ModelCapabilities` (`get_capabilities()`), and gate features the way
   `require_branching()` / `require_prefix_cache()` do in `src/engine.cpp`: throw a
   `std::runtime_error` whose message tells the caller *what to do instead*
   ("construct with KVCacheMode::Paged", "run linear ReAct only").

2. **Tiered configuration.** New knobs flow: caller intent → `blackwell::InferenceConfig`
   (tier-2 "what I want") → `build_and_validate_runtime()` → `blackwell::RuntimeConfig`
   (tier-3 validated execution plan). Low-level pokes go through `RuntimeOverrides`
   (the "`-x264-params` seam"), never as new loose constructor arguments. Members size
   themselves from the *resolved* plan, not from raw request fields.

3. **Declaration order = dependency order.** In `BlackwellEngine::Impl`
   (`src/engine_impl.h`), member declaration order *is* the construction order and the
   reverse destruction order, and the comments at each declaration site state why
   (e.g. `kv_mgr` after `arena` because the adapter holds a reference into the arena).
   When you add a member, place it deliberately and document the placement.

House comment style: comments state **contracts and invariants**, not narration — keep the
existing density. Mixed Russian/English is accepted; match whichever the surrounding file
uses. Prefer contract comments ("must precede any kernel that reads the layer's weights")
over what-comments.

## Quickstart: build & test

```bat
:: Build one target (sets up vcvars64 + CUDA 13.2 include, then cmake --build)
.\build_target.bat poc_overlay

:: Run a test suite (from the build tree)
cd out\build\x64-Debug
ctest -L validation      :: fast kernel-correctness suite
ctest -L benchmark       :: stress shapes + CUDA-event perf
ctest -L integration     :: engine vs PyTorch golden dumps (skips silently without local models)
ctest -L agent           :: tree-sitter agent core (no CUDA)
ctest -L agent_env       :: sandbox OS layer (no CUDA)
ctest -L agent_orchestrator :: ReAct loop, MockLLM-driven (no CUDA)
```

The build tree lives in `out/build/x64-Debug` (cache generated from the VS IDE). Details,
failure modes, and target names: skill `build-and-test`. Regenerating the PyTorch reference
dumps the integration suite compares against: skill `golden-dumps`.

## Design docs (read before touching the subsystem)

- `docs/INFERENCE_API.md` — the engine's public API philosophy, DirectStorage cold start.
- `docs/KV_PREFIX_CACHE.md` — Phase 1: radix tree, prefix sharing, `.bkv` serialization.
- `docs/TIERED_KV_AND_AOT.md` — Phase 2: VRAM⇄RAM⇄NVMe paging, AOT context compiler; has a component→file→status table.
- `docs/TRANSLATION_AGENT.md` — the translation agent product design.
- `docs/REACT_1STEP_ASSESSMENT.md` — strict 1-step ReAct assessment (dense checkpoints only).

## Technical Debt & Refactoring Roadmap

Ordered register. **Do not imitate these patterns in new code** — new code follows the
target state; existing code migrates opportunistically when you touch it.

| # | Debt (current state) | Target state | Status |
|---|---|---|---|
| 1 | `CUDA_CHECK` in `src/common.h` calls `exit(EXIT_FAILURE)` from library code | Throws a `blackwell::cuda_error : std::runtime_error` (file:line + `cudaGetErrorString`); consumers (overlay, playground) surface it | planned |
| 2 | ~105 raw `cudaMalloc`/`cudaFree` sites; `BlackwellEngine::Impl` holds ~30 raw `float*` freed by a hand-maintained list in `~Impl()` | `DeviceBuffer<T>` RAII wrapper (move-only, sized ctor, implicit `T*` like `CudaVector`); Impl's destructor becomes `= default` | planned |
| 3 | `src/CMakeLists.txt` is a ~170-line monolith (core lib + DirectStorage FetchContent + tool gating); re-declares `project()` | One `CMakeLists.txt` per target directory; root only orchestrates; deps in `cmake/` modules | planned |
| 4 | `blackwell_kernels` exports `PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..` (all of `src/` leaks) | Narrow per-target `target_include_directories` | in progress (global `tests/reference` include + shim removed with the compress experiment, 2026-07) |
| 5 | ~~Root pollution: 8 loose `.py` scripts, `logits_comparison.csv`, stray `CMakeCache.txt`~~ | Scripts live in `scripts/` (paths anchored to repo root, checkpoints via `BLACKWELL_MODELS_DIR`); stray artifacts deleted; the 126 MB of tracked golden dumps moved to Git LFS (`.gitattributes`); `backup/` stays local-only (gitignored) | **done** |
| 6 | `build_target.bat` + IDE-generated cache is the only CLI build path | Committed `CMakePresets.json` (configure + build + test presets) | planned |
| 7 | `TLS_VERIFY OFF` on the DirectStorage NuGet fetch | Verified fetch | in progress (vendored nvcomp binaries deleted with the compress experiment, 2026-07) |
| 8 | Thread-ownership doctrine enforced only by comments | Debug thread-ID asserts on engine entry points (see doctrine section) | planned |
| 9 | Sanitizers absent | ASan config for MSVC on the CUDA-free agent stack first; UBSan/TSan via clang-cl or Linux CI lane for `agent*` targets | idea |

When a row lands: flip its Status, update the affected section above, and follow the
`<evolution_protocol>` in the corresponding skill (`build-and-test` for #6,
`engine-extension` for #1/#2/#8, `cmake-hygiene` for #3/#4, `golden-dumps` for #5's
path migration).
