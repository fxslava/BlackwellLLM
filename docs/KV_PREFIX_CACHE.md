# KV Prefix Cache — Radix Tree + Disk Serialization

Design and implementation plan for the prefix-sharing / persistence upgrade of
the paged KV cache. **Phase 2 — tiered VRAM/RAM/NVMe residency with page
faults, and the AOT context compiler — is documented in
[TIERED_KV_AND_AOT.md](TIERED_KV_AND_AOT.md)** (note: `PrefixCacheManager` is
now constructed with a `TieredMemoryPager` and the radix tree stores virtual
page ids). Companion code:

| Component | File | Status |
|---|---|---|
| `SequenceManager` extensions (page IO, adoption, raw ownership) | `src/paging/paged_kv_cache.h` | implemented |
| `RadixTreeIndex` (page-aligned prefix tree) | `src/paging/radix_tree.h` | implemented, unit-tested |
| `KVBranchSerializer` (`.bkv` disk format) | `src/paging/kv_branch_serializer.h` | implemented |
| `PrefixCacheManager` (facade) | `src/paging/prefix_cache_manager.h` | implemented |
| Radix tree tests | `tests/standalone/test_radix_tree.cpp` | 58 checks passing |

## 1. Layering

```
                 engine decode loop (engine.cpp)
                          │  IKVCacheManager
                 ┌────────▼────────┐
                 │  PagedKVManager │  adapter: RoPE → append → paged-flash-attn
                 └────────┬────────┘
        acquire/commit/   │
        release/dump/load │
                 ┌────────▼──────────┐
                 │ PrefixCacheManager│  facade; owns lock table
                 └──┬──────┬──────┬──┘
        ┌───────────▼─┐  ┌─▼──────────────┐  ┌▼───────────────────┐
        │RadixTreeIndex│  │SequenceManager │  │ KVBranchSerializer │
        │ (host-only)  │  │ pools + CoW +  │  │  .bkv NVMe dump /  │
        │ prefix index │  │ block tables   │  │  zero-prefill load │
        └──────────────┘  └────────────────┘  └────────────────────┘
```

The GPU contract is unchanged: kernels see only the bf16 pools + a per-sequence
block table. **Prefix sharing and persistence are pure control-plane features**
— a reused or deserialized page is indistinguishable from a freshly prefilled
one at kernel level. No CUDA kernel changes are required for either objective.

## 2. Reference-count model (the one invariant)

`PageAllocator::ref_count` counts every holder of a page:

* each **block table** entry (sequences; as before),
* each **radix tree node** entry (`retain`/`release` callbacks),
* transient **serializer-load** ownership (`allocate_raw_page`, handed to the
  tree in `PrefixCacheManager::load`, then dropped).

A page is freed exactly when the count hits zero. CoW is untouched: prefix
pages handed to a new sequence are always *full*, so the first append past
them lands in a fresh page; a rewind into a shared page CoWs on the next
append via the existing `reserve_append_slot` path.

## 3. Radix tree semantics

* **Page granularity.** An edge label is a whole number of `PAGE_SIZE`(=16)
  token pages; a page is shared iff all 16 tokens match. Divergence inside a
  page costs at most 15 re-prefilled tokens.
* **O(1) descent per page.** Children are keyed by an FNV-1a hash of the
  child's first page of tokens; candidates verify token-exact (hash collisions
  chain — they can degrade sharing, never corrupt it).
* **Splits at page boundaries only**, preserving node identity: the *suffix*
  keeps the original node object, so `Match::node` pointers and lock counts
  held by live sequences stay valid across splits.
* **Locking**: a sequence pins only the deepest node of its path; eviction
  refuses childless-or-locked nodes, so ancestors are transitively protected.
  O(1) lock/unlock.
* **Eviction**: LRU over evictable leaves (min-heap on access tick), cascading
  to parents as they become leaves. Driven on demand by
  `ensure_free_pages(needed)` before a prefill/load, not by a background sweep.

## 4. `.bkv` disk format (zero-prefill restore)

```
[0,      4096)   BkvHeader     magic/version/dtype, geometry, model_hash,
                               token FNV hash, section offsets
[4096,   …   )   int32 tokens  zero-padded to a 4 KiB multiple
[payload,…   )   for page: for layer: { K blob, V blob }   (bf16, page-major)
```

* **Page-major payload** ⇒ a file *prefix* is itself a valid shorter branch
  (truncated loads restore the first N pages with all layers).
* **4 KiB alignment everywhere** and page blobs that are naturally 4 KiB
  multiples ⇒ the same format serves buffered `fstream` (v1), unbuffered
  `FILE_FLAG_NO_BUFFERING`, and DirectStorage file→VRAM without rewriting files.
* **Safety rails**: `model_hash` (checkpoint identity) and geometry fields are
  verified before any page is allocated; token-section hash catches
  truncation/corruption; load rolls back all allocated pages on any throw.
* Offload-aware: `read_page`/`write_page` route resident layers through
  device↔pinned-host DMA and offloaded layers straight to the pinned mirror.

Typical branch size: `tokens/16 pages × layers × 2(K,V) × kv_heads·16·head_dim·2B`
— e.g. a 2048-token system prompt on a 36-layer, 8-KV-head, 128-dim model is
128 pages × 36 × 64 KiB ≈ **288 MB**, a ~0.1 s sequential NVMe read at 3 GB/s
versus a full prefill pass.

## 5. Engine integration plan (remaining work)

**Phase 1 — wire `PagedKVManager` to the facade.**
`PagedKVManager` gains an optional `PrefixCacheManager` (constructed with the
`SequenceManager` it already owns and a `model_hash` derived from the
checkpoint path + KV geometry). Two `IKVCacheManager` additions:

```cpp
// returns tokens already cached; engine prefills only [cached, n)
virtual int  begin_sequence(SeqId seq, const int32_t* tokens, int n);
virtual void end_sequence(SeqId seq, const int32_t* tokens, int n, bool publish);
```

Continuous manager implements them as no-ops returning 0 (prefill everything),
so the engine code stays strategy-agnostic. Inside the paged adapter,
`begin_sequence` → `acquire` (replacing the bare `create_sequence` in the id
map), `end_sequence` → `commit` + `release`. The engine's prefill loop simply
starts at `start_pos = cached_tokens` — the existing
`prepare_prefill_step(seq, start_pos, num_tokens)` contract already supports a
nonzero start.

**Phase 2 — system-prompt persistence in the orchestrator.**
After the agent's system prompt is prefilled once: `commit` + `dump` to
`<model_dir>/prefix_cache/<fnv(tokens)>.bkv`. On engine start, scan that
directory and `load` each file (bounded by a config knob,
`RuntimeConfig::prefix_cache_dir` / `max_restored_pages`). Result: restart →
first token of the first request without any system-prompt prefill.

**Phase 3 — Tree-of-Thoughts unification.**
Today ToT branching uses explicit `fork()`. With the tree in place, branches
can instead `commit` the parent path and `acquire` per child — same physical
sharing, but branches become *discoverable* (a later, unrelated request that
happens to share the parent prefix also hits it) and survive the parent
sequence's destruction. `fork()` stays for the hot in-flight case (no token
re-hash); the tree covers the cross-request and cross-restart cases.

**Phase 4 — I/O performance (format-stable upgrades).**
1. Double-buffer the pinned staging (two `(K,V)` page buffers, overlap
   `fread`/`fwrite` with DMA on a dedicated transfer stream).
2. Unbuffered Win32 reads (`FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN`)
   — everything is already 4 KiB-aligned.
3. DirectStorage: reuse the existing `DirectStorageLoader` queue
   (`src/direct_storage_loader.cpp`) to read page blobs straight into a device
   staging arena, then a scatter kernel (trivial variant of
   `launch_cow_copy_page`) distributes them to allocated page slots. This is
   the true "instant reload" path; the fallback chain mirrors the weight
   loader's (DirectStorage → Win32).

## 6. CUDA / PagedAttention interaction summary

* `launch_paged_flash_attention_{decode,prefill}` already gather K/V through
  the block table — shared and restored pages need **no kernel changes**.
* Prefill-after-hit reuses the existing chunked path: Q rows are the tail
  tokens, `seq_len` counts cached + tail, and the kernel's causal mask handles
  the offset (`prepare_prefill_step` with `start_pos = cached_tokens`).
* CoW (`launch_cow_copy_page`) covers writes into tree-shared pages exactly as
  it covers fork-shared ones — same ref-count trigger in
  `reserve_append_slot`.
* The only new device work is Phase 4's optional scatter kernel for the
  DirectStorage restore path.

## 7. Testing strategy

* `tests/standalone/test_radix_tree.cpp` (passing) — match/insert/split/LRU/
  lock/refcount invariants, host-only.
* Next: `test_prefix_cache.cpp` (standalone, GPU) — acquire/commit/release
  refcount balance against a real `SequenceManager`; `.bkv` round-trip
  byte-compare via `read_page`; corrupt/foreign-model load rejection.
* Parity gate: extend `test_parity_paged_vs_fp32.cpp` with a
  prefill-hit-then-decode case — logits after a prefix hit must match logits
  after a cold full prefill bit-for-bit (same pages, same kernel path).
