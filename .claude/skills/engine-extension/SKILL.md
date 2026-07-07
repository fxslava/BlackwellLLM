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
- The doctrine is CHECKED at the DLL edge: every `EngineCom` method (`engine_com.cpp`)
  runs `BLACKWELL_VERIFY_OWNING_THREAD()` — a debug-only `std::abort()` if a non-owning
  thread enters (owner = the `CreateBlackwellEngine` caller). A new operational
  `IBlackwellEngine` method MUST open with this macro (`Release` is the sole exception —
  teardown-after-join). White-box callers below the COM edge aren't covered — keep the
  marshaling discipline there.

## 2. Impl members & device memory

- **Declaration order = construction order = reverse destruction order** in
  `BlackwellEngine::Impl` (`src/core/engine_impl.h`). Config/runtime members precede `arena`;
  `kv_mgr` follows `arena` (holds a reference into it); the paging chain
  (`kv_vram_pool → kv_tier_backend → kv_pager → prefix_cache → prefill`) follows `kv_mgr`.
  Place new members deliberately and write the placement comment.
- New device buffers are `blackwell::DeviceBuffer<T>` members (`src/core/device_buffer.h`)
  — declare (placement still deliberate, with the comment), `.allocate(count)` in the
  ctor or the owning subsystem's composition branch, and NEVER touch `~Impl()` (it is
  `= default`; buffers free themselves in reverse declaration order). Model-conditional
  scratch declares empty and allocates inside its branch. The implicit `operator T*()`
  keeps kernel-launch sites and white-box test probes unchanged; no implicit zero-fill —
  call `.zero()` when a buffer is read before first write (the full-attn KV cache is the
  example).
- Exception-safety rule this layout exists for: once `CUDA_CHECK` throws (Roadmap #2),
  a mid-constructor failure must leak nothing — so any device resource acquired before a
  potentially-throwing point must be owned by a RAII member, never by a naked pointer.
  The only raw device pointers left in Impl are the NON-owning arena views
  (`d_X_accum` / `d_X_norm`); keep it that way.
- Buffers that are per-decode-step hot must be allocated once and reused (see the
  `d_next_token` comment in `step_embedding`: a per-step malloc/free pair is forbidden).
- Subsystem state that is conditional on model family stays behind `unique_ptr` + null
  checks (pattern: `ssm_state`, `prefix_cache`) — the ctor is the single composition root.

## 3. Error handling — the Hybrid doctrine (every new code path)

- Pick the tier by phase, not by taste (`blackwell/engine_status.h`):
  **INIT** (ctor / factory / setup) → `CUDA_CHECK_THROW` (throws
  `blackwell::cuda_error`; RAII members make it leak-free). **RUNTIME**
  (`run_token` / `step_*` / anything per-token) → return `blackwell::EngineStatus`,
  propagate with `ENGINE_TRY`, check CUDA calls with `CUDA_CHECK_RETURN` (both
  defined atop `src/core/engine.cpp`). Destructors use neither — raw `cudaFree`,
  errors swallowed (see `DeviceBuffer::reset()`).
- Runtime facade methods are `*_status` ONLY — noexcept, with a `catch(...)` →
  `status_from_current_exception()` net inside so stray subsystem throws become
  statuses (copy `forward_status`). There are NO throwing runtime wrappers (purged
  2026-07): every caller — COM boundary, coordinator, tracker, adapter, tests —
  branches on the returned `EngineStatus`. The coordinator's COMPUTE phase reports via
  the `status` fields on `Result`/`UpdateStats`/`EngineSequence` (halt gracefully,
  roll back, self-heal); its session-admin surfaces (acquire/budget/commit/finish) and
  the AOT-warmup facet stay exception-tier — `blackwell::engine_error` is the carrier
  there.
- Never add legacy exit()-`CUDA_CHECK` sites in engine code; unmigrated subsystems
  (`memory_pool`, `paging/`, `ssm/`, kernels) convert opportunistically when touched.

## 4. Capability gating & configuration (any new feature knob)

- Query `ModelCapabilities`; never infer support ad-hoc. Gate with a `require_*()` helper
  that throws `std::runtime_error` telling the caller **what to do instead** — copy the
  tone of `require_branching()` / `require_prefix_cache()` in `src/core/engine.cpp`.
- New knobs flow `InferenceConfig` (intent) → `build_and_validate_runtime()` (validation,
  in `runtime_config.cpp`) → `RuntimeConfig` (resolved plan) → members size themselves
  from the plan. Low-level escape hatches go on `RuntimeOverrides`. **Never** add loose
  constructor parameters — the legacy ctor maps onto this flow and stays frozen.
- Benign no-ops beat errors for lifecycle sweeps (`spill_kv_cache()` on a model without a
  substrate returns 0); hard errors are for caller mistakes.

## 5. Model-architecture paths

- New layer kinds dispatch inside `Impl::run_token`'s per-layer branch; keep the
  legacy uniform path untouched when `layer_types` is empty.
- Respect the stream contract: compute on the legacy stream, transfers on
  `VRAMArena`'s transfer stream, ordering via `cudaEvent` only — call
  `arena.ensure_layer_ready(layer)` before any kernel reading layer weights, and keep
  `prefetch_layer(i + 1, pos)` overlap intact.
- Recurrent (SSM) state has **no rewind** — anything that resets/branches sequences must
  handle it explicitly (see `reset_state()` and the CoW-branching veto in the ctor).

## 6. The DLL boundary (any facade / public API change)

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

## 7. Verification

- `ctest --preset validation` for kernel-level changes; `ctest --preset integration` for
  anything touching numerics (regenerate dumps first if geometry moved — skill
  `golden-dumps`).
- Public C++ API changes: keep `include/blackwell/engine.h` light (forward declarations,
  no engine/CUDA includes) and update `docs/INFERENCE_API.md`.

<evolution_protocol>
**Landed so far (2026-07):** the Roadmap #1 COM boundary (section 6), the Roadmap #3 Impl
migration to `DeviceBuffer<T>` (section 2, `~Impl()` is `= default`), the Roadmap #2
Hybrid error doctrine for the engine core (section 3: INIT throws `cuda_error`, RUNTIME
returns `EngineStatus`, both mapped at the DLL edge), and the Roadmap #9 thread-ownership
asserts (section 1: `BLACKWELL_VERIFY_OWNING_THREAD()` on every `EngineCom` method).
**Remaining debt this skill tracks:**
- Legacy exit()-`CUDA_CHECK` + raw `cudaMalloc`/`cudaFree` (~75 sites) in `memory_pool` /
  `paging/` / `ssm/` / kernels. When touching one of those subsystems: convert its
  allocations to `DeviceBuffer<T>` (or RAII equivalent for non-buffer resources), its
  init-path checks to `CUDA_CHECK_THROW`, and any per-token-path checks to the
  status tier — the priority audit is `memory_pool`'s ctor failure path
  (`release_pools()`), which predates RAII. Update CLAUDE.md rows #2/#3 as subsystems
  land; when the last legacy `CUDA_CHECK` site dies, delete the legacy macro from
  `src/common.h` (tests' `CudaVector` migrates to `CUDA_CHECK_THROW` then too).
- Sanitizers (Roadmap #10, idea): ASan for the CUDA-free agent stack first; UBSan/TSan
  via a clang-cl or Linux CI lane. (Note: the MSVC ASan runtime shipped with toolset
  14.34 crashes in its own init — `Symbolizer::PlatformInit`; use a 14.4x runtime.)
</evolution_protocol>
