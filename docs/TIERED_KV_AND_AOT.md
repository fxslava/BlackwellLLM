# Tiered KV Paging & the AOT Context Compiler

Phase-2 of the prefix cache (see [KV_PREFIX_CACHE.md](KV_PREFIX_CACHE.md) for
Phase 1): active 3-tier residency (VRAM ⇄ pinned RAM ⇄ NVMe) with transparent
page faults, and the Text-to-Cache compiler that ships prompt libraries as
`.bkv` directories.

| Component | File | Status |
|---|---|---|
| `TieredMemoryPager` + `VramPool` / `TierBackend` seams | `src/paging/tiered_memory_pager.h` | implemented, unit-tested |
| `CudaTierBackend`, `SmVramPool` | `src/paging/cuda_tier_backend.h` | implemented |
| `PrefixCacheManager` (tier-routed rewrite) | `src/paging/prefix_cache_manager.h` | implemented |
| `RadixTreeIndex::touch()` | `src/paging/radix_tree.h` | implemented |
| Warmup DSL + parser | `src/paging/warmup_spec.h` | implemented, unit-tested |
| `AOTCacheWarmer` + `warm_start()` | `src/paging/aot_cache_warmer.h` | implemented |
| Pager state-machine tests | `tests/standalone/test_tiered_pager.cpp` | 64 checks passing |
| DSL parser tests | `tests/standalone/test_warmup_spec.cpp` | 19 checks passing |

## 1. The VPid indirection

One design move buys the whole tier system: the radix tree now stores
**virtual page ids** (`VPid`) instead of physical VRAM pages. The tree never
knew what its int32 ids meant (they flow through `retain`/`release`
callbacks), so it is byte-for-byte the same data structure; only the callback
bindings changed. Block tables — and therefore the CUDA kernels — still hold
physical `PageId`s: **the GPU contract is untouched for the second phase
running**.

```
 radix tree ──VPid──► TieredMemoryPager ──PageId──► block tables ──► kernels
                        │  VPid -> { VRAM phys │ RAM slot │ DISK slot }
                        │  per-tier intrusive LRU, pins, refs
                        ├── VramPool    (SequenceManager's PageAllocator)
                        └── TierBackend (bytes only: 4 moves + fence)
```

The pager is a **pure host state machine** — no CUDA includes, mirroring
`radix_tree.h`. Byte movement lives behind `TierBackend` (4 slot-addressed
moves + a fence), VRAM slots behind `VramPool`. Host tests inject fakes that
carry page *content* through the tiers, so the waterfall is validated for
data integrity, not just bookkeeping. The flat Phase-1 behaviour is the
degenerate `Config{0, 0}` — one code path, no legacy mode.

## 2. Residency rules (who may live where)

* `pins` — live block tables (+ in-flight serializer I/O). Pinned ⇒
  VRAM-locked, excluded from the LRU lists, never demoted. `acquire()` pins,
  `release()` unpins (a released-while-pinned "zombie" is reaped by the final
  unpin).
* `refs` — radix-tree adoptions. `refs == 0 && pins == 0` frees the backing
  at whatever tier it occupies.
* Pages known only to a live sequence (post-CoW, pre-commit) are invisible to
  the pager; the `PageAllocator` refcount covers them exactly as before.
* Demoting a page whose physical slot is still shared with a live sequence
  releases only the pager's reference — the sequence keeps decoding against
  its copy; the tree's RAM/disk snapshot stays valid for the tokens it
  indexes. CoW semantics make the snapshot immutable by construction.

## 3. The page fault

`fault_in(vp, n, compute_stream, pin)` restores pages **front to back** and
returns the *resident prefix count* `k ≤ n`:

* **Never fails a request.** If VRAM cannot be freed (everything pinned),
  `k` is just smaller; `acquire()` seeds the sequence with `k` pages and the
  engine prefills the rest. Worst case degrades to a cold prefill.
* **Block or yield.** Promotions are async H2D on the backend's transfer
  stream; one `fence(cs)` per fault batch either records an event the
  caller's *compute stream* waits on (GPU yields, host keeps going —
  decode overlaps the DMA) or host-blocks when `cs == nullptr`.
* **Waterfall demotion** feeds the faults: LRU VRAM page → pinned RAM slot;
  RAM full → LRU RAM page → disk slot; disk full → demotion fails and
  `ensure_free_pages()` escalates to radix-tree LRU *eviction* (the only
  destruction path — the pager never drops a page behind the index's back).

Memory-pressure ladder, in order of increasing pain:
`demote cold pages (contents survive)` → `evict tree branches (contents
die, re-prefillable)` → `report OOM to the caller`.

The spill file's page records are byte-identical to `.bkv` payload pages
(`[layer][K|V]` blobs), so spill restore, archive load and — later — the
DirectStorage request shape are one format.

## 4. AOT context compiler

`spec.json → parse_warmup_spec → AOTCacheWarmer::compile → out_dir/*.bkv +
manifest.json`. The DSL is a tree of prompt fragments; a node's full prompt
is the root-to-node concatenation:

```json
{ "name": "support-agent", "emit": "leaves",
  "nodes": [ { "id": "base", "text": "You are …", "children": [
      { "id": "base.en", "text": "Respond in English.", "children": [
          { "id": "base.en.tools", "text_file": "tools_prompt.md" } ] },
      { "id": "base.ru", "text": "Отвечай по-русски." } ] } ] }
```

The compiler walks the spec DFS: tokenize the node's segment, `acquire()`
(which **prefix-hits the parent committed one frame up the stack**),
prefill only the tail, `commit()`, optionally `dump()`. The parent's sequence
stays live while children compile, so its pins hold the shared path in VRAM.
Total prefill cost is O(unique tokens in the tree), not O(sum of path
lengths) — the compiler runs on its own prefix cache.

The engine is reached only through `IPrefillDriver` (tokenize + chunked
prefill), the same seam pattern as the orchestrator's `ILLMGenerator`: the
production driver binds the tokenizer and the `PagedKVManager` prefill sweep;
tests inject a mock and never link CUDA.

Deploy side: `AOTCacheWarmer::warm_start(pc, dir)` verifies the manifest's
`model_hash`, then loads every branch with `LoadPolicy::ColdRam` — the whole
prompt library sits in pinned RAM at startup (zero VRAM cost) and pages fault
into VRAM at DMA speed on first request. Overlapping branches dedup on load
through the radix tree exactly like overlapping commits.

## 5. Remaining integration work

1. **Engine wiring** — ✅ landed (Phase 3). `BlackwellEngine::Impl` is the
   composition root: `SmVramPool → CudaTierBackend → TieredMemoryPager →
   PrefixCacheManager → EnginePrefillCoordinator`, built over the
   `PagedKVManager`'s own `SequenceManager` for Paged-mode dense models, and
   exposed via `engine.prefix_cache()` / `engine.prefill_driver()`
   (`has_prefix_cache()` gates). Sizing is currently derived (RAM tier mirrors
   the device pool; disk tier off) — promoting it to `RuntimeConfig` knobs
   (`kv_ram_slots`, `kv_disk_slots`, `kv_spill_path`, `prefix_cache_dir`)
   remains open.
2. **Production `IPrefillDriver`** — ✅ landed:
   `EnginePrefillCoordinator` (`src/engine_prefill_coordinator.h`) implements
   the acquire → budget → bind → compute → commit prompt state machine over
   the per-token decode sweep (lm_head GEMV skipped for non-final prompt
   tokens), and serves `AOTCacheWarmer` via the `IPrefillDriver` facet
   (tokenizer injected with `set_tokenizer()`). Still open: the
   `blackwell_warmup` CLI target (`warmup.exe spec.json out_dir/`) and the
   batched Tensor-Core prefill kernels behind
   `prepare_prefill_step`/`attention_prefill` (the coordinator's `run_delta`
   is the single swap point).
3. **GPU round-trip test** — standalone: build a branch, force it down to
   the spill file, fault it back, byte-compare via `read_page`; then the
   parity gate: logits after a disk-faulted prefix hit must equal a cold
   prefill bit-for-bit.
4. **I/O upgrades (format-stable, behind `TierBackend`)** — double-buffered
   staging for `disk_to_vram`; unbuffered Win32 reads (records are 4 KiB
   multiples); DirectStorage batch reads straight to a device staging arena +
   scatter kernel, reusing the weight loader's queue and fallback chain.
5. **Batch-aware fencing** — today one fence per fault batch is enough
   (single-sequence decode); multi-sequence scheduling wants per-sequence
   events so one faulting branch never stalls another's compute.
