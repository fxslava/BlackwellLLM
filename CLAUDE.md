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
| `blackwell_core` | `src/core/` | yes | The engine: `engine.cpp` (facade + Impl), `memory_pool` (VRAMArena, offloading, hibernation), `kv_cache/` (Continuous vs Paged strategies), `paging/` (header-only tiered KV + prefix cache + radix tree), `ssm/`, `tokenizer/`, `safetensors`, `weight_loader`, `config`. Built two-tier: `blackwell_core_obj` (OBJECT lib, white-box surface for tests/tools) + `blackwell_core` (SHARED DLL whose only exports are the COM factories in `engine_com.cpp`). Shared `common.h` (CUDA_CHECK/CudaVector) stays one level up in `src/`. |
| Public API | `include/blackwell/` | no | `iblackwell_engine.h` — the COM-style DLL boundary (pure-virtual interfaces + C factories + HRESULT), the ONLY header for out-of-tree consumers. The C++ headers (`engine.h`, `config.h`, `runtime_config.h`, `tokenizer.h`, `chat_template.h`, `weight_loader.h`) serve the white-box tier. |
| `blackwell_vad` | `src/vad/` | **no** | Silero neural voice-activity detection (`silero_vad.hpp/.cpp`) over the ONNXRuntime **CPU** provider. PIMPL'd: this is the only TU in the repo that sees `<onnxruntime_cxx_api.h>`. Plugs into the pipeline through the bridge's `SpeechVadScoreFn` seam — deliberately NOT linked into `blackwell_bridge`, so the C-ABI control plane stays free of a 15 MB runtime its other consumers never use. Option-gated (`USE_SILERO_VAD`, default ON). |
| Agent stack | `src/agent/`, `src/agent_env/`, `src/agent_orchestrator/` | **no** | Tree-sitter CST analysis, sandbox OS layer (VFS/subprocess/git), ReAct orchestrator. Deliberately CUDA-free; talks to the model via the `ILLMGenerator` text interface. |
| Tools | `src/tools/playground/`, `src/tools/poc_overlay/` | links core | HTTP playground GUI; Win32/UIA/Direct2D overlay translator (the flagship consumer). |
| Tests | `tests/` | mixed | Labeled CTest suites — see Quickstart. `tests/common/` holds shared fixtures (CPU references, CUDA fixtures). |
| Build modules | `cmake/` | — | `Dependencies.cmake` (pinned FetchContent: gtest, nlohmann_json), `DirectStorage.cmake` (`blackwell::dstorage` INTERFACE target + `blackwell_copy_dstorage_dlls()` helper), `OnnxRuntime.cmake` (`blackwell::onnxruntime` + `blackwell_copy_onnxruntime_dlls()` / `blackwell_copy_vad_model()`; hash-pinned ORT CPU zip + `silero_vad.onnx` → gitignored `models/`). |
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
  ("Not thread-safe; call from the single engine-owning thread" — `src/core/memory_pool.h`).

This doctrine is a **checked invariant** at the DLL boundary (2026-07): `EngineCom`
(`src/core/engine_com.cpp`) captures `std::this_thread::get_id()` in the factory and every
operational method asserts it via `BLACKWELL_VERIFY_OWNING_THREAD()` — a debug-only
(`#ifndef NDEBUG`) `std::abort()` on violation, compiled to nothing in Release. `Release()`
is exempt (teardown-after-join). This is also our pragmatic TSan substitute (MSVC has no
TSan). White-box consumers that drive the engine below the COM edge (the coordinator, the
overlay tracker) still rely on the doctrine by convention.

## Extension patterns — the law for new engine features

These three patterns are already in the code and are non-negotiable for extensions:

1. **Capability gating.** Never guess what a loaded model supports — query
   `ModelCapabilities` (`get_capabilities()`), and gate features the way
   `require_branching()` / `require_prefix_cache()` do in `src/core/engine.cpp`: throw a
   `std::runtime_error` whose message tells the caller *what to do instead*
   ("construct with KVCacheMode::Paged", "run linear ReAct only").

2. **Tiered configuration.** New knobs flow: caller intent → `blackwell::InferenceConfig`
   (tier-2 "what I want") → `build_and_validate_runtime()` → `blackwell::RuntimeConfig`
   (tier-3 validated execution plan). Low-level pokes go through `RuntimeOverrides`
   (the "`-x264-params` seam"), never as new loose constructor arguments. Members size
   themselves from the *resolved* plan, not from raw request fields.

3. **Declaration order = dependency order.** In `BlackwellEngine::Impl`
   (`src/core/engine_impl.h`), member declaration order *is* the construction order and the
   reverse destruction order, and the comments at each declaration site state why
   (e.g. `kv_mgr` after `arena` because the adapter holds a reference into the arena).
   When you add a member, place it deliberately and document the placement.

4. **Hybrid error doctrine** (`include/blackwell/engine_status.h`). Two tiers, split
   by phase. **INIT tier** (ctors / factories / setup / AOT warmup): exceptions —
   `CUDA_CHECK_THROW` throws `blackwell::cuda_error`, RAII members unwind, the DLL
   boundary maps it to an HRESULT (`E_OUTOFMEMORY` for VRAM exhaustion). **RUNTIME
   tier** (the decode hot loop and everything a live stream touches): status codes,
   end to end, no exceptions anywhere — the ONLY inference endpoints are the noexcept
   `forward_status`/`forward_eval_status` (there is no throwing `forward()`; stray
   subsystem exceptions are caught inside and converted), `Impl::run_token`/`step_*`
   propagate via `CUDA_CHECK_RETURN` + `ENGINE_TRY`, and the prefill coordinator's
   COMPUTE phase reports through the `status` fields on
   `Result`/`UpdateStats`/`EngineSequence` (a faulted stream halts gracefully; the
   session self-heals on the next reconcile). Statuses translate exactly once:
   `hresult_from_status()` at the COM edge (`OutOfVram`→`E_OUTOFMEMORY`,
   `InvalidArgument`/`InvalidConfig`→`E_INVALIDARG`, `StateMismatch`→`E_NOT_VALID_STATE`,
   `CudaRuntimeError`→`BLACKWELL_E_CUDA_RUNTIME`); UI layers branch on the status.
   `blackwell::engine_error` exists only to carry a status across exception-tier
   (init/admin) surfaces. Never add legacy exit()-`CUDA_CHECK` call sites in engine code.

House comment style: comments state **contracts and invariants**, not narration — keep the
existing density. Mixed Russian/English is accepted; match whichever the surrounding file
uses. Prefer contract comments ("must precede any kernel that reads the layer's weights")
over what-comments.

## Quickstart: build & test

```bat
:: Canonical (CMakePresets.json; run from a VS Developer / vcvars64 shell):
cmake --preset x64-debug                                  :: configure (once, or after CMake edits)
cmake --build --preset x64-debug --target poc_overlay
ctest --preset validation                                 :: fast kernel-correctness suite
:: other test presets: benchmark / integration / agent / agent-env / agent-orchestrator

:: One-shot from ANY shell (transitional wrapper: vcvars64 + CUDA include, then the preset build):
.\build_target.bat poc_overlay
```

The build tree lives in `out/build/x64-Debug` (`x64-release` preset → `out/build/x64-Release`).
Details, failure modes, and target names: skill `build-and-test`. Regenerating the PyTorch
reference dumps the integration suite compares against: skill `golden-dumps`.

## Design docs (read before touching the subsystem)

- `docs/INFERENCE_API.md` — the engine's public API philosophy, DirectStorage cold start.
- `docs/KV_PREFIX_CACHE.md` — Phase 1: radix tree, prefix sharing, `.bkv` serialization.
- `docs/TIERED_KV_AND_AOT.md` — Phase 2: VRAM⇄RAM⇄NVMe paging, AOT context compiler; has a component→file→status table.
- `docs/TRANSLATION_AGENT.md` — the translation agent product design.
- `docs/REACT_1STEP_ASSESSMENT.md` — strict 1-step ReAct assessment (dense checkpoints only).
- `docs/CONTINUOUS_STREAMING.md` — Phase 8: SimulMT re-translation (draft-and-commit), the
  commit pointer, and RoPE-aware KV head eviction.
- `docs/LOCAL_ROUTER.md` — the strict local arbiter in front of the paid Cloud API: the
  EOS commit rule, `TerminationReason`, and why the network client has no cancellation.

**Architectural Atlas** (`docs/README.md` + numbered guides) — the cross-cutting
architecture the subsystem docs above assume: [`docs/README.md`](docs/README.md) (repo map),
`01_architecture_and_threading.md` (COM boundary + single-thread doctrine),
`02_hybrid_error_doctrine.md` (error tiers + `EngineStatus`→`HRESULT` table),
`03_memory_and_kv_modes.md` (`memory_pool`, Continuous vs Paged KV, `DeviceBuffer<T>`).

## Technical Debt & Refactoring Roadmap

Ordered register. **Do not imitate these patterns in new code** — new code follows the
target state; existing code migrates opportunistically when you touch it.

| # | Debt (current state) | Target state | Status |
|---|---|---|---|
| 1 | ~~No hard binary boundary~~ | DirectX/COM-style SHARED `blackwell_core.dll`: pure-virtual `IBlackwellEngine`/`IBlackwellTokenizer` + C factories (`include/blackwell/iblackwell_engine.h`), `HRESULT` mapping in `src/core/engine_com.cpp`; white-box tier `blackwell_core_obj` (OBJECT lib) for tests + tools. `src/apps/` are true COM consumers; poc_overlay/playground stay white-box until the interface covers the prefill-coordinator surface | **done** (2026-07) |
| 2 | ~~Engine error handling = `exit()` from library code~~ | Hybrid doctrine (extension pattern #4): INIT tier throws `blackwell::cuda_error`, RUNTIME tier returns `EngineStatus`, both mapped once at the DLL edge (`hresult_from_status` / `map_current_exception` in `engine_com.cpp`). No production code (`src/core` + `src/kernels`) calls `exit()`-`CUDA_CHECK`; the legacy macro survives only in test fixtures' `CudaVector` + `src/experiments` (test code — dying loudly is fine there) | **done** (2026-07) |
| 3 | ~~`BlackwellEngine::Impl` held ~30 raw `float*` freed by a hand-maintained list~~; raw `cudaMalloc`/`cudaFree` sites remain in `memory_pool` (RAII pending; error-handling migrated) / `paging/` / `ssm/` / kernels | `DeviceBuffer<T>` (`src/core/device_buffer.h`: move-only, `.allocate()`, implicit `T*` like `CudaVector`); Impl's dtor IS `= default`; remaining subsystems migrate opportunistically when touched | in progress (Impl migrated, 2026-07) |
| 4 | ~~`src/CMakeLists.txt` monolith~~ | One `CMakeLists.txt` per target directory (`src/core/`, `src/apps/`, `src/experiments/`, tools); `src/CMakeLists.txt` is orchestration-only; deps in `cmake/` modules | **done** (core physically relocated to `src/core/`, 2026-07) |
| 5 | ~~`blackwell_kernels` exports `PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..` (all of `src/` leaks)~~ | No target exports `src/`; core-internal consumers declare `PRIVATE src/` + `src/core` themselves | **done** (2026-07) |
| 6 | ~~Root pollution: 8 loose `.py` scripts, `logits_comparison.csv`, stray `CMakeCache.txt`~~ | Scripts live in `scripts/` (paths anchored to repo root, checkpoints via `BLACKWELL_MODELS_DIR`); stray artifacts deleted; the 126 MB of tracked golden dumps moved to Git LFS (`.gitattributes`); `backup/` stays local-only (gitignored) | **done** |
| 7 | ~~`build_target.bat` + IDE-generated cache is the only CLI build path~~ | Committed `CMakePresets.json` (configure + build + test presets); `build_target.bat` is a thin transitional wrapper over the presets | **done** (2026-07) |
| 8 | ~~`TLS_VERIFY OFF` on the DirectStorage NuGet fetch; nvcomp vendored as raw binaries~~ | All fetches TLS-verified (`cmake/DirectStorage.cmake`); vendored nvcomp deleted | **done** |
| 9 | ~~Thread-ownership doctrine enforced only by comments~~ | Debug thread-ID asserts (`BLACKWELL_VERIFY_OWNING_THREAD`) on every `EngineCom` operational method — `#ifndef NDEBUG`, `std::abort()` on violation, zero Release-build cost (see doctrine section) | **done** (2026-07) |
| 10 | Sanitizers absent | ASan config for MSVC on the CUDA-free agent stack first; UBSan/TSan via clang-cl or Linux CI lane for `agent*` targets | idea |
| 11 | ~~No warning enforcement~~ | Zero-warning policy: first-party CXX compiles under `/W4 /WX` with `/external:W0` quarantining `<>`-header noise (CUDA/Win32/STL/deps); `_CRT_SECURE_NO_WARNINGS` for the C4996 CRT nag. CUDA (`.cu`) untouched. Skill `compiler-hygiene` documents the fix conventions | **done** (2026-07) |

When a row lands: flip its Status, update the affected section above, and follow the
`<evolution_protocol>` in the corresponding skill (`engine-extension` for #1/#2/#3/#9,
`cmake-hygiene` for #1's SHARED-library build work, `golden-dumps` for dump-path
concerns, `build-and-test` for build-workflow changes, `compiler-hygiene` for #11's
zero-warning policy).
