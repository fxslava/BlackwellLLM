# 01 — Architecture & Threading

## The binary boundary: `IBlackwellEngine`

`blackwell_core.dll` presents a DirectX/COM-style boundary. The **only** header an
out-of-tree consumer includes is [`include/blackwell/iblackwell_engine.h`](../include/blackwell/iblackwell_engine.h);
the **only** implementation TU inside the DLL is [`src/core/engine_com.cpp`](../src/core/engine_com.cpp).

### Design rules of the boundary

- **Pure-virtual interfaces + C factories.** `struct IBlackwellEngine` and
  `IBlackwellTokenizer` are pure-virtual; construction goes through exported
  `extern "C"` factories (`CreateBlackwellEngine`, `CreateBlackwellTokenizer`). Creation
  *is* initialization — the factory loads weights (the C++ ctor), so a failed load returns
  an `HRESULT` and `*ppOut` stays null. There is no separate `Initialize`.
- **ABI-frozen types only.** Methods take/return `int32_t`, `float`, `BOOL`, and POD
  descriptor structs (`BLACKWELL_ENGINE_DESC`, `BLACKWELL_CAPABILITIES`). No STL type ever
  crosses the edge. The header includes nothing but `<windows.h>` and `<cstdint>` — no
  CUDA, no engine internals, no third-party types — so any MSVC-built binary (different
  CRT, different exception model) can sit on the other side.
- **Every method returns `HRESULT`; payload comes back through out-params.** Arrays and
  strings use the classic two-call protocol (pass a null buffer to get the count, call
  again with a buffer of that size). See [02_hybrid_error_doctrine.md](02_hybrid_error_doctrine.md)
  for how internal errors become HRESULTs.
- **No exception escapes a boundary method.** Every body runs under `boundary()`, which
  catches and maps via `map_current_exception()`. That catch-all is the *panic net*, not
  the mechanism — the runtime path returns statuses that translate directly.
- **Single-owner lifetime.** `Release()` destroys the object. There is deliberately no
  `AddRef` / shared refcounting: a shareable handle would invite violations of the
  single-threaded doctrine below.

### The white-box tier

The in-tree tools (poc_overlay, playground) and all tests link the OBJECT library
(`blackwell_core_obj`) and use the richer C++ facade (`BlackwellEngine`,
`EnginePrefillCoordinator`, the paging headers) directly. The COM interface does not yet
cover the prefill-coordinator surface; those consumers stay white-box until it does.

## The single-threaded control-plane doctrine

> **The engine control plane is single-threaded. Exactly one thread may touch a
> `BlackwellEngine` after load; every other thread marshals work onto it.**

This is *the most important rule* in the codebase. It is what lets the engine, the whole
paging substrate, and `VRAMArena`'s hibernation lifecycle run with zero locks on the hot
path.

### How it plays out in `poc_overlay` (the flagship consumer)

- `LiveTranslationTracker` owns **one** dedicated worker thread — the *sole* thread
  permitted to call `forward_status()`, `hibernate()`, `spill_kv_cache()`, prefill,
  anything on the engine.
- `TrackUpdate` / `TriggerGeneration` / `Cancel` are O(1) fire-and-forget enqueues,
  callable from any thread. Interruption is a monotone generation counter plus a
  latest-wins single job slot.
- Callbacks fired *from* the worker thread must **only enqueue/marshal** (the PostMessage
  pattern) — they never touch a window, COM, or the engine directly.
- New engine-facing features go through `LiveTranslationTracker::PostEngineTask()`.

`VRAMArena::hibernate()/wakeup()` and the paging substrate state the same contract in
their own comments: *"Not thread-safe; call from the single engine-owning thread."*

## Enforcement: `BLACKWELL_VERIFY_OWNING_THREAD()`

The doctrine used to be comment-folklore. Since 2026-07 it is a **checked invariant** at
the DLL boundary (`engine_com.cpp`):

```cpp
// EngineCom captures the owner at construction (= the factory caller's thread):
explicit EngineCom(std::unique_ptr<BlackwellEngine> engine)
    : engine_(std::move(engine)), owner_thread_id_(std::this_thread::get_id()) {}

#ifndef NDEBUG
#define BLACKWELL_VERIFY_OWNING_THREAD()                       \
    do {                                                       \
        if (std::this_thread::get_id() != owner_thread_id_) {  \
            std::cerr << "... contract violated ...";          \
            std::abort();                                      \
        }                                                      \
    } while (0)
#else
#define BLACKWELL_VERIFY_OWNING_THREAD() ((void)0)
#endif
```

- **Every operational `IBlackwellEngine` method** opens with the macro: `GetCapabilities`,
  `Forward`, `ForwardEval`, `LastTokenProbability`, `Fork`, `Rewind`, `ResetState`,
  `SpillKvCache`, `Hibernate`, `Wakeup`, `IsHibernated`.
- On a cross-thread entry it logs both thread ids and `std::abort()`s — loud and
  immediate. This is our pragmatic **TSan substitute** (MSVC ships no TSan).
- **Debug-only.** `#ifndef NDEBUG` compiles it to `((void)0)` in Release: zero hot-loop
  overhead.
- **`Release()` is the sole exception** — teardown-after-join is an established pattern
  (the tracker joins its worker *before* releasing the engine, so no concurrent access is
  possible at that point).

### The gap it does not close

White-box consumers that drive the engine *below* the COM edge — the coordinator, the
overlay tracker calling `BlackwellEngine`/`EnginePrefillCoordinator` directly — are not
routed through `EngineCom` and so are **not** covered by the assert. They still rely on
the doctrine by convention; keep the marshaling discipline there.

## Extension checklist (when adding an engine feature)

1. **Capability-gate** with `ModelCapabilities` (`get_capabilities()`); throw a
   `std::runtime_error` telling the caller *what to do instead* (see `require_branching()`
   / `require_prefix_cache()` in `engine.cpp`).
2. **Tiered config:** `InferenceConfig` (intent) → `build_and_validate_runtime()` →
   `RuntimeConfig` (validated plan) → members size from the plan. Low-level pokes go on
   `RuntimeOverrides`, never as loose constructor args.
3. **Declaration order = dependency order** in `BlackwellEngine::Impl` — it is the
   construction and reverse-destruction order; document each placement.
4. **New COM method** ⇒ ABI-frozen signature + `BLACKWELL_VERIFY_OWNING_THREAD()` at the
   top + status translation (see doc 02).
