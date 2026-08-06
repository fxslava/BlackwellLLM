# BlackwellLLM — Architectural Atlas

The definitive guide to how this engine is put together, for humans and AI agents.
Read [CLAUDE.md](../CLAUDE.md) first for the working rules; this Atlas is the *why* and
the *shape* behind them.

## The one-paragraph map

BlackwellLLM is a from-scratch CUDA LLM inference engine for Blackwell GPUs
(`CMAKE_CUDA_ARCHITECTURES 120`), C++20 / CUDA, Windows + MSVC, tuned for **batch = 1,
latency-critical** desktop workloads. The engine is a SHARED DLL with a DirectX/COM-style
binary boundary; internally it is single-threaded, error-tiered by phase, and RAII-owned.

## Repository layout — the three tiers of `src/`

| Directory | Target(s) | Language | Role |
|---|---|---|---|
| `src/kernels/` | `blackwell_kernels` (STATIC) | CUDA | The compute primitives — one `.cu`+`.cuh` per kernel (attention, paged-flash, AWQ/FP8/INT4/BF16 GEMV, RoPE, SSM, sampling). No engine state; pure launchers over raw device pointers. |
| `src/core/` | `blackwell_core_obj` (OBJECT) + `blackwell_core` (SHARED DLL) | C++ (+ CUDA headers) | The engine: `engine.cpp` (facade + `Impl`), `memory_pool` (VRAMArena + offloading + hibernation), `kv_cache/` (Continuous vs Paged strategies), `paging/` (header-only tiered KV + prefix cache + radix tree), `ssm/`, `tokenizer/`, `safetensors`, `weight_loader`, `config`, and `engine_com.cpp` (the COM boundary). |
| `src/apps/` | `blackwell_llm`, `blackwell_bench` | C++ | User-facing executables. The **first true COM consumers** — they include *only* `blackwell/iblackwell_engine.h`, link the import lib, and speak `HRESULT`. |

Plus: `src/agent*/` (CUDA-free tree-sitter + ReAct stack), `src/tools/` (poc_overlay,
playground — white-box consumers), `src/experiments/`, `include/blackwell/` (public API),
`cmake/` (build modules), `tests/` (labeled CTest suites).

## The two-tier core: OBJECT lib vs SHARED DLL

The most important structural fact. `blackwell_core` is built twice:

- **`blackwell_core_obj`** (OBJECT library) — every engine source, the *full white-box
  surface*. Linked by the test suites and the in-tree tools (poc_overlay, playground),
  which drive the `EnginePrefillCoordinator` / paging C++ API that the COM interface does
  not yet cover.
- **`blackwell_core`** (SHARED DLL) — one own TU, `engine_com.cpp`, whose *only* exports
  are the C factories. No `WINDOWS_EXPORT_ALL_SYMBOLS`; the narrow surface is the point.
  Linked by `src/apps/`.

**Rule:** a target links exactly one tier — the DLL (COM consumer, boundary header only)
or the OBJECT lib (white-box, internal headers allowed) — never both.

## The Atlas

| Doc | Covers |
|---|---|
| [01_architecture_and_threading.md](01_architecture_and_threading.md) | The `IBlackwellEngine` COM/ABI boundary; the single-threaded control-plane doctrine; `BLACKWELL_VERIFY_OWNING_THREAD()`. |
| [02_hybrid_error_doctrine.md](02_hybrid_error_doctrine.md) | The split-macro error system; `forward_status` noexcept guarantees; the full `EngineStatus`→`HRESULT` table. |
| [03_memory_and_kv_modes.md](03_memory_and_kv_modes.md) | `memory_pool` / VRAMArena; Continuous vs Paged KV; `DeviceBuffer<T>` RAII. |

## The other design docs (subsystem deep-dives)

These predate the Atlas and cover specific subsystems in depth — read before touching the
matching code:

- [INFERENCE_API.md](INFERENCE_API.md) — the engine's public API philosophy, DirectStorage cold start.
- [KV_PREFIX_CACHE.md](KV_PREFIX_CACHE.md) — Phase 1: radix tree, prefix sharing, `.bkv` serialization.
- [TIERED_KV_AND_AOT.md](TIERED_KV_AND_AOT.md) — Phase 2: VRAM⇄RAM⇄NVMe paging, AOT context compiler.
- [TRANSLATION_AGENT.md](TRANSLATION_AGENT.md) — the translation-agent product design.
- [REACT_1STEP_ASSESSMENT.md](REACT_1STEP_ASSESSMENT.md) — strict 1-step ReAct assessment.
- [TTS_INTEGRATION_AUDIT.md](TTS_INTEGRATION_AUDIT.md) — **proposal, not landed**: on-demand
  speech synthesis for `audio_translator` over the existing ONNXRuntime CPU provider.
- [VOICE_ASSISTANT.md](VOICE_ASSISTANT.md) — technical review of the `voice_assistant` app:
  threads, the two engine backends, the pipeline modes, full duplex, and a risk assessment.
  Read alongside [LOCAL_ROUTER.md](LOCAL_ROUTER.md), which owns the commit/routing semantics.
