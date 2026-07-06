---
name: engine-extension
description: Checklist and invariants for modifying BlackwellLLM engine internals — src/core/ (engine.cpp, engine_impl.h, memory_pool, kv_cache/, paging/, ssm/), or any code that calls BlackwellEngine. Use before adding engine features, new model-architecture paths, new device buffers, new config knobs, or engine-facing tool features (overlay/playground).
---

# Engine Extension — Invariants Checklist

Work through every item that applies. These invariants are load-bearing; the engine's own
comments state them at each site — preserve that comment discipline in your changes.

## 1. Threading (always applies)

- The engine control plane is **single-threaded**: one owning thread after load. In
  `poc_overlay` that is `LiveTranslationTracker`'s worker — route new engine-facing work
  through `LiveTranslationTracker::PostEngineTask()`, never call the engine from the UI /
  CaretTracker STA / timer threads.
- Callbacks you fire *from* the worker thread must only enqueue/marshal (PostMessage
  pattern) — never touch windows, COM, or the engine.
- Lifecycle contract: `hibernate()` requires an idle engine; `get_weight_ptr()` throws
  while hibernated (violations are loud by design). Activity wins over queued lifecycle
  ops.

## 2. Impl members & device memory

- **Declaration order = construction order = reverse destruction order** in
  `BlackwellEngine::Impl` (`src/core/engine_impl.h`). Config/runtime members precede `arena`;
  `kv_mgr` follows `arena` (holds a reference into it); the paging chain
  (`kv_vram_pool → kv_tier_backend → kv_pager → prefix_cache → prefill`) follows `kv_mgr`.
  Place new members deliberately and write the placement comment.
- New device buffers today: `CUDA_CHECK(cudaMalloc(...))` in the Impl ctor **and** a
  matching `cudaFree` in `~Impl()` — three places to keep in sync (member, alloc, free).
  This is Roadmap debt #3; see the evolution protocol below before adding many buffers.
- Buffers that are per-decode-step hot must be allocated once and reused (see the
  `d_next_token` comment in `step_embedding`: a per-step malloc/free pair is forbidden).
- Subsystem state that is conditional on model family stays behind `unique_ptr` + null
  checks (pattern: `ssm_state`, `prefix_cache`) — the ctor is the single composition root.

## 3. Capability gating & configuration (any new feature knob)

- Query `ModelCapabilities`; never infer support ad-hoc. Gate with a `require_*()` helper
  that throws `std::runtime_error` telling the caller **what to do instead** — copy the
  tone of `require_branching()` / `require_prefix_cache()` in `src/core/engine.cpp`.
- New knobs flow `InferenceConfig` (intent) → `build_and_validate_runtime()` (validation,
  in `runtime_config.cpp`) → `RuntimeConfig` (resolved plan) → members size themselves
  from the plan. Low-level escape hatches go on `RuntimeOverrides`. **Never** add loose
  constructor parameters — the legacy ctor maps onto this flow and stays frozen.
- Benign no-ops beat errors for lifecycle sweeps (`spill_kv_cache()` on a model without a
  substrate returns 0); hard errors are for caller mistakes.

## 4. Model-architecture paths

- New layer kinds dispatch inside `Impl::run_token`'s per-layer branch; keep the
  legacy uniform path untouched when `layer_types` is empty.
- Respect the stream contract: compute on the legacy stream, transfers on
  `VRAMArena`'s transfer stream, ordering via `cudaEvent` only — call
  `arena.ensure_layer_ready(layer)` before any kernel reading layer weights, and keep
  `prefetch_layer(i + 1, pos)` overlap intact.
- Recurrent (SSM) state has **no rewind** — anything that resets/branches sequences must
  handle it explicitly (see `reset_state()` and the CoW-branching veto in the ctor).

## 5. The DLL boundary (any facade / public API change)

- `blackwell_core.dll`'s surface is `include/blackwell/iblackwell_engine.h` +
  `src/core/engine_com.cpp` (Roadmap #1). **No exception escapes a boundary method**:
  every body runs under `boundary()`, which maps via `map_current_exception()`. Out-params
  are checked (`E_POINTER`) before work; arrays/strings use the two-call protocol
  documented in the header.
- A new `BlackwellEngine` facade method that boundary consumers need gets a matching
  `IBlackwellEngine` method (HRESULT + out-params, ABI-frozen types: `int32_t`, `BOOL`,
  POD structs) in the same change. White-box-only surface (coordinator, paging) does NOT
  go on the interface yet — tools link `blackwell_core_obj` for that.
- Consumers are one-tier-only: COM consumers include the boundary header exclusively;
  white-box consumers may use C++ headers but link the OBJECT lib, never the DLL.

## 6. Verification

- `ctest --preset validation` for kernel-level changes; `ctest --preset integration` for
  anything touching numerics (regenerate dumps first if geometry moved — skill
  `golden-dumps`).
- Public C++ API changes: keep `include/blackwell/engine.h` light (forward declarations,
  no engine/CUDA includes) and update `docs/INFERENCE_API.md`.

<evolution_protocol>
**Sequencing:** the Roadmap #1 COM boundary LANDED 2026-07 (see section 5), unblocking
the work below: `map_current_exception()` in `src/core/engine_com.cpp` carries a NOTE
marker at the exact spot where the `blackwell::cuda_error` catch clause goes.
**Current workaround:** raw `cudaMalloc`/`cudaFree` with hand-maintained destructor lists
(~105 sites), and `CUDA_CHECK` in `src/common.h` calling `exit(EXIT_FAILURE)` from library
code (Roadmap #2/#3 in CLAUDE.md). Also: the single-thread doctrine is comment-enforced
only (Roadmap #9 — the debug thread-ID asserts fit naturally on the `EngineCom` methods,
which now wrap every boundary entry).
**Target state:** a move-only `DeviceBuffer<T>` RAII wrapper (sized ctor, implicit `T*`
conversion like `CudaVector`, throwing allocation); `CUDA_CHECK` throws
`blackwell::cuda_error : std::runtime_error` with file:line, translated to `HRESULT` at
the #1 boundary; debug thread-ID asserts on every public engine entry point.
**When DeviceBuffer<T> / throwing CUDA_CHECK land:** rewrite section 2 of this skill —
(1) new buffers become `DeviceBuffer<float> d_foo{count};` members with **no** ctor/dtor
edits (declaration order still matters — keep that rule); (2) delete the three-places-in-
sync warning; (3) add the exception-safety rule: Impl construction may now throw, so any
remaining raw resources acquired before a throwing point must be owned by RAII members,
never by naked pointers; (4) when migrating existing buffers, convert one subsystem at a
time (SSM scratch, then full-attn cache, then core buffers), build + `ctest -L validation`
after each, and remove the freed entries from `~Impl()` as you go until it is `= default`;
(5) once thread asserts exist, update section 1 to reference the assert instead of only
the comments. Update CLAUDE.md Roadmap statuses in the same change.
</evolution_protocol>
