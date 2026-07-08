# 02 — The Hybrid Error Doctrine

Error handling is split **by phase**, not by taste. The vocabulary lives in
[`include/blackwell/engine_status.h`](../include/blackwell/engine_status.h); the mapping
lives at the DLL edge in [`src/core/engine_com.cpp`](../src/core/engine_com.cpp).

## Why two tiers

Two failure phases have opposite constraints:

- **INIT** (constructors / factories / setup / AOT warmup): errors are *rare* (VRAM
  exhaustion at load) and *exceptional*. Exception safety matters — RAII members must
  unwind. Exceptions are the right tool.
- **RUNTIME** (the decode hot loop and everything a live stream touches): a `throw` in the
  hot path causes stack-unwinding latency spikes mid-generation. Zero-cost return codes
  are the right tool.

So: **INIT throws, RUNTIME returns status.** Both translate to an `HRESULT` exactly once,
at the COM boundary.

## The split-macro system

Three checks, one per phase, all defined where they are used:

| Macro | Where | Behavior | Defined in |
|---|---|---|---|
| `CUDA_CHECK_THROW(call)` | INIT paths (ctors, allocation, setup, lifecycle) | Throws `blackwell::cuda_error` (carries the raw `cudaError_t`) | `src/common.h` |
| `CUDA_CHECK_RETURN(call)` | RUNTIME paths (`Impl::run_token`, `step_*`) | Logs once, `return`s `EngineStatus::OutOfVram` (on `cudaErrorMemoryAllocation`) or `CudaRuntimeError` | top of `src/core/engine.cpp` |
| `ENGINE_TRY(expr)` | RUNTIME call chains | Propagates a non-`Success` status up by return value (one predictable branch; zero-cost on the happy path) | top of `src/core/engine.cpp` |

The **legacy `CUDA_CHECK(call)`** (which calls `exit(EXIT_FAILURE)`) survives in `common.h`
**only** for test fixtures' `CudaVector` and `src/experiments` — where dying loudly is
fine. **No production code (`src/core` + `src/kernels`) calls it.** Never add a new
`exit()`-`CUDA_CHECK` site in engine code.

### The migration pattern (how `memory_pool`/`paging`/`ssm` were converted)

Converting a whole subsystem was a mechanical `CUDA_CHECK → CUDA_CHECK_THROW` swap with
**no signature cascade**, because the doctrine's seams already catch each phase:

- INIT throws unwind via the object's ctor `try/catch → release_pools()`-style cleanup and
  map to `E_OUTOFMEMORY`.
- Lifecycle throws (hibernate/wakeup) surface as HRESULTs through the `EngineCom` boundary.
- A rare runtime-path throw (e.g. offload staging during decode) propagates up through
  `step_*` / `run_token` into `forward_status`'s noexcept catch, which converts it to a
  status.
- Teardown (`release_pools`, destructors) stays **raw** `cudaFree` — nothing may throw
  from a destructor.

## The runtime endpoints are `*_status` only, and `noexcept`

The two — and only — inference endpoints on the facade:

```cpp
blackwell::EngineStatus forward_status(int token_id, int pos, float temperature,
                                       float top_p, int seq_id, int* next_token) noexcept;
blackwell::EngineStatus forward_eval_status(int token_id, int pos, int target_token_id,
                                            int seq_id, float* log_prob) noexcept;
```

There is **no** throwing `forward()`/`forward_eval()` — they were purged (2026-07). The
`noexcept` is genuine: each wraps `run_token` in a `try/catch` whose
`status_from_current_exception()` converts *any* stray subsystem throw (an unmigrated
`cudaError`, a paged-manager `runtime_error` on a bad seq id, the arena's hibernation
guard) into an `EngineStatus`. The hot loop therefore **cannot** unwind across its own
boundary.

The prefill coordinator mirrors this: `run_delta` is `noexcept` and returns `EngineStatus`;
its COMPUTE phase reports failure through the `status` fields on
`Result` / `UpdateStats` / `EngineSequence` (a faulted stream halts gracefully and
self-heals on the next reconcile). Its **session-admin** surfaces
(acquire/budget/commit/finish) and the **AOT-warmup** facet stay exception-tier —
`blackwell::engine_error` is the carrier that ferries a status across those surfaces.

> Consumer contract: a faulted `prefill_prompt` **returns** a faulted `Result`
> (`engine_seq == -1`), it does **not** throw. Callers must check `r.status` (this is the
> class of bug fixed in the adapter, commit `b54a6f1`).

## Translation happens exactly once, at the edge

`engine_com.cpp` has two mapping functions:

- **`hresult_from_status(EngineStatus, op)`** — the runtime path. `EngineCom::Forward` /
  `ForwardEval` call the `*_status` variant and translate the result directly. No
  exception round-trip.
- **`map_current_exception(op)`** — the INIT/admin path and the last-resort panic net.
  Called only from a catch context.

### `EngineStatus` → `HRESULT`

| `EngineStatus` | `HRESULT` | Meaning |
|---|---|---|
| `Success` | `S_OK` | — |
| `OutOfVram` | `E_OUTOFMEMORY` | device allocation failure (`cudaErrorMemoryAllocation`) |
| `InvalidArgument` | `E_INVALIDARG` | caller error: `pos`/`seq_id` outside the resolved plan |
| `InvalidConfig` | `E_INVALIDARG` | checkpoint/plan integrity violation found mid-decode |
| `StateMismatch` | `E_NOT_VALID_STATE` | operation illegal in the current engine state |
| `CudaRuntimeError` | `BLACKWELL_E_CUDA_RUNTIME` | CUDA API/kernel fault during decode |

`BLACKWELL_E_CUDA_RUNTIME` is a dedicated code in the `FACILITY_ITF` custom range:
`MAKE_HRESULT(SEVERITY_ERROR, FACILITY_ITF, 0x0200)` — distinct from `E_OUTOFMEMORY` (VRAM
exhaustion) and `E_NOT_VALID_STATE` (illegal state). It is declared in the boundary header
so consumers can branch on it.

### Exception → `HRESULT` (the INIT/admin path, `map_current_exception`)

| Caught | `HRESULT` |
|---|---|
| `blackwell::cuda_error` with `cudaErrorMemoryAllocation` | `E_OUTOFMEMORY` |
| `blackwell::cuda_error` (other) | `BLACKWELL_E_CUDA_RUNTIME` |
| `blackwell::engine_error` | `hresult_from_status(e.status())` (unwrap + reuse the table above) |
| `std::bad_alloc` | `E_OUTOFMEMORY` |
| `std::invalid_argument` | `E_INVALIDARG` |
| `std::out_of_range` | `E_BOUNDS` |
| `std::exception` | `E_FAIL` |
| `...` (unknown) | `E_FAIL` |

## One-line mental model

> INIT throws `cuda_error` → RAII unwinds → mapped to an HRESULT. RUNTIME returns
> `EngineStatus` → mapped to an HRESULT. Nothing unwinds across the hot loop; everything
> is translated exactly once at the DLL edge; UI layers branch on the status.
