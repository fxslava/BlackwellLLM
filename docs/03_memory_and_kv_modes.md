# 03 — Memory & KV Modes

## `DeviceBuffer<T>` — the RAII device owner

[`src/core/device_buffer.h`](../src/core/device_buffer.h). A move-only owner of a
`cudaMalloc`'d buffer — the device-memory analogue of `std::unique_ptr`.

```cpp
blackwell::DeviceBuffer<float> d_logits;          // empty
d_logits.allocate(vocab_size);                     // cudaMalloc (CUDA_CHECK_THROW)
launch_gemv(..., d_logits, ...);                   // implicit operator T* -> raw ptr
// ~DeviceBuffer frees it; no manual cudaFree, no entry in any destructor list
```

Contract:

- **Move-only.** Copy is deleted (a copy would double-free on the GPU); move transfers the
  allocation and empties the source.
- **`allocate(count)` / `zero()` are INIT-tier** — they throw `blackwell::cuda_error` on
  failure (`CUDA_CHECK_THROW`). The RAII layout is what makes a throwing constructor
  leak-free: already-constructed members unwind.
- **No implicit zero-fill** (unlike the test-only `CudaVector`). Decode scratch is written
  before it is read every step; the one buffer read before first write (the full-attention
  KV cache, at masked positions) calls `.zero()` explicitly.
- **Implicit `operator T*()`** keeps kernel-launch sites and white-box test telemetry
  probes (`impl->d_logits`, `impl->d_ssm_core`, …) unchanged; `.get()` exists for contexts
  where the conversion is ambiguous.
- **`reset()` bypasses the throwing check** — it runs from destructors, where neither
  `exit()` nor a throw is acceptable; it swallows the `cudaFree` result.

### Where it is (and isn't) used

`BlackwellEngine::Impl` holds **27 owned device buffers as `DeviceBuffer<T>`** (all SSM
scratch, the core compute buffers, the gated full-attention cache); `~Impl()` is
`= default`. The only raw device pointers left in `Impl` are `d_X_accum` / `d_X_norm`,
which are **non-owning views** into `VRAMArena`'s activation pool.

`memory_pool` / `paging/` / `ssm/` still hold raw pointers freed by hand-maintained
destructor / `release_pools()` lists (their error-handling is migrated; the `DeviceBuffer`
RAII migration is Roadmap #3, pending).

## `memory_pool` — `VRAMArena` + `PinnedHostPool`

[`src/core/memory_pool.h`](../src/core/memory_pool.h) /
[`.cpp`](../src/core/memory_pool.cpp). The static memory orchestrator for the ~12 GB VRAM
budget. Not thread-safe — call from the single engine-owning thread.

### Residency & offloading

Layers `[0, num_gpu_layers)` keep weights **and** KV cache resident in VRAM. Layers
`[num_gpu_layers, num_layers)` are **offloaded**: their weights live in a pinned host
arena (`PinnedHostPool`, `cudaHostAlloc`) and stream through **two ping-pong device
staging slots** (`slot = layer % 2`) on a dedicated **non-blocking transfer stream**,
overlapping the PCIe traffic of layer *N+1* with the compute of layer *N*. Ordering
between the transfer stream and the legacy compute stream is expressed **only** through
`cudaEvent`s — the math kernels are untouched.

Runtime control plane (all no-ops for resident layers):
`prefetch_layer(N+1, pos)` enqueues the next layer's weights + KV prefix;
`ensure_layer_ready(layer)` makes the compute stream wait on the staged weights;
`prepare_layer_kv` / `commit_layer_kv` stage-in and spill KV columns.

### Soft hibernation (inactivity lifecycle)

`hibernate()` evacuates the multi-GB resident weights arena to a pinned host stash (D2H
DMA) and frees the device allocation; the arena object, the name→pointer registry, and
every pool stay alive. `wakeup()` re-allocates device memory, DMAs the stash back, and
**rebases the registry** to the new device base — callers keep using `get_weight_ptr()`
with no reload from disk. The stash is allocated once and kept for the process lifetime, so
repeat cycles never re-pay the pinning cost and a wakeup can never fail on host allocation.
Both are idempotent; `get_weight_ptr()` throws while hibernated to make violations loud.

### Error tiers here (see doc 02)

Every check is `CUDA_CHECK_THROW`: construction/allocation failures unwind via the ctor's
`try/catch → release_pools()` (→ `E_OUTOFMEMORY`); `hibernate`/`wakeup` throws map at the
`EngineCom` boundary; the rare offload-staging throw during decode is caught by
`forward_status`. Teardown uses raw `cudaFree`.

## Continuous KV vs Paged KV

The KV-cache strategy is chosen once at construction (`KVCacheMode`) and hides behind
`IKVCacheManager` ([`src/core/kv_cache/`](../src/core/kv_cache/)). `step_attention_math`
issues a single monomorphic virtual call per layer — free relative to the kernels it wraps.

| | **Continuous** (default) | **Paged** |
|---|---|---|
| Storage | Legacy FP32 contiguous cache + layer offloading | bf16 paged cache, page pool + block table |
| Kernel | Fused RoPE+append → decode attention | paged-flash attention over the block table |
| Branching | ✗ — `fork`/`rewind` throw | ✓ — Copy-on-Write `fork` / `rewind` |
| Prefix cache | ✗ | ✓ (dense uniform full-attention models only) |
| Sequences | seq_id 0 only | multiple sequences via `seq_id` |

**Paged mode** unlocks the whole speculative-tracking stack: the prefix cache (radix tree
over the page pool), the tiered VRAM⇄RAM⇄NVMe pager, and `EnginePrefillCoordinator`'s
Continuous Speculative Tracking (keystroke-driven micro-rewinds). It is restricted to
**dense uniform full-attention** models — hybrid SSM state and the gated head_dim-256
attention cache live *outside* the paged pools, so a radix "prefix hit" would silently skip
state those models need. The engine simply does not construct a coordinator for them
(`has_prefix_cache()` returns false; the overlay/adapter fall back to ordinary generate).

**Rewind — physical vs. virtual.** A dense paged sequence rolls back *physically*:
`rewind`/`truncate_sequence` drop trailing pages and the position-addressed cache
self-heals. A hybrid SSM sequence **cannot** — its recurrent state advances every
`forward()` with no positional inverse — so `rewind()` throws for it. Instead, hybrid
models rewind *virtually* via `HybridSnapshotRing` (`src/core/hybrid_snapshot_ring.*`):
`fork()` physically snapshots all three state stores (SSM recurrent/conv, gated
full-attention KV, paged CoW pool) into a ring of checkpoint slots every K tokens, and a
backtrack restores the nearest snapshot (`release_sequence` + `fork` back into the pinned
active slot 0, allocation-free) and replays the retained token tail. Slot budget is the
shared `branch_capacity()` (`paged_branch_factor`, default 4 → ~(B-1)·K token horizon);
`release_sequence` is the primitive that recycles a fork id (fork rejects a live id). The
overlay's `LiveTranslationTracker` selects this path for hybrids that fork; the
`Qwen35Hybrid.VirtualRewindMatchesFreshDecode` test pins the restored state to
bit-parity with a fresh decode.

### Why tests map to modes

Integration suites pick the mode that exercises what they assert:

- `PagedEngineIntegration.*` — construct with `KVCacheMode::Paged` and drive
  `forward_status` + `fork`/`rewind`/CoW to prove the paged path and branch API run with
  `Success` statuses (a decode fault is an error *status*, not a CUDA crash — the branch
  API stays exception-tier, so `fork(bad)` still `EXPECT_THROW`s).
- `LiveTrackerParity.*` — drive `EnginePrefillCoordinator::begin_sequence` /
  `update_sequence` (Paged) and assert the micro-rewind result matches a cold prefill
  (parity cosine).
- `QwenEngineIntegration` / `test_llama_engine` — Continuous mode, per-layer numeric parity
  against PyTorch golden dumps (driving `step_*` directly).
- `Qwen35Hybrid.*` — the hybrid SSM path under `KVCacheMode::Paged`: recurrent-state decode
  parity, `reset_state`, and *virtual rewind* (physical `fork` snapshot + `release_sequence`
  recycle + `HybridSnapshotRing` restore-to-bit-parity). `rewind()` stays rejected.
- `AsyncOffload.*` (validation) — `VRAMArena` offloading: ping-pong double-buffering and
  strided KV-column spill.

The one-line rule: **Continuous = numeric-parity & offloading tests; Paged = branching,
prefix-cache & speculative-tracking tests.**
