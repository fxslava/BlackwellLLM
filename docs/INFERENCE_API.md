# BlackwellLLM Inference API

A C++ inference engine tuned for **batch = 1, latency-critical** workloads — the
profile of a background desktop app that wakes up on a hotkey, runs a short
translation, and goes back to sleep. Two things make that profile cheap:

- **DirectStorage weight loading** — model weights stream NVMe → VRAM with
  minimal host-side copying, so `BlackwellEngine` construction (the "cold
  start") is fast enough to pay on every hotkey press if you want to.
- **Layer offloading** — a fixed split of transformer layers between VRAM and
  pinned host RAM, so the engine can run under a hard VRAM budget instead of
  requiring the whole model resident.

This document covers the public headers under [`include/blackwell/`](../include/blackwell/):
`engine.h`, `runtime_config.h`, `config.h`, `tokenizer.h`, `chat_template.h`,
`weight_loader.h`.

---

## 1. High-Level Architecture & Lifecycle

### 1.1 The three-tier configuration pipeline

Like a video codec, settings flow through three layers of decreasing
abstraction ([`runtime_config.h`](../include/blackwell/runtime_config.h)):

```
ModelConfig (tier 1)        InferenceConfig (tier 2)       RuntimeConfig (tier 3)
immutable facts from the    what you ASK for: context   -> the resolved EXECUTION
checkpoint's config.json    length, batch width,           PLAN the engine actually
(head_dim, num_layers, …)   branching intent                consumes (KV mode, GPU
                                                              layer split, dispatch)
```

`build_and_validate_runtime()` reconciles tiers 1+2 (+ optional
`RuntimeOverrides`) into a `RuntimeConfig`, or throws with an actionable
message if the request is unsatisfiable (e.g. `require_branching` on a hybrid
SSM checkpoint, or a context length beyond the model's trained positional
range). You normally don't call this directly — the `BlackwellEngine(path,
InferenceConfig)` constructor does it for you.

### 1.2 Engine construction = the DirectStorage hot path

`BlackwellEngine`'s constructor is where all the expensive I/O happens:

1. Parse `config.json` next to the safetensors index → `ModelConfig`.
2. Resolve the `RuntimeConfig` (KV mode, VRAM/host layer split).
3. Build the `VRAMArena`: allocate device buffers, then stream every weight
   tensor from disk into VRAM via whichever `IWeightLoader` the build was
   compiled with.

If the binary was built with `-DUSE_DIRECT_STORAGE=ON` (Windows only), that
loader is a batched, asynchronous `DirectStorageLoader`
([`weight_loader.h`](../include/blackwell/weight_loader.h)):

- `load_to_vram()` never touches disk synchronously — it stages the tensor
  into a pinned arena and enqueues it. Requests batch up until an arena is
  full, then a single `IDStorageQueue::Submit()` covers the whole batch,
  overlapping NVMe reads for batch *N+1* with PCIe copies of batch *N*.
- `flush()` is the completion barrier: nothing is guaranteed resident in VRAM
  until it returns. `VRAMArena` calls this once at the end of construction.
- If DirectStorage fails to initialize, or a batch errors out mid-load, the
  loader transparently falls back to a synchronous `StandardLoader`
  (`fread` + `cudaMemcpy`) — correctness never depends on DS succeeding.

Without that build flag (or on non-Windows), `IWeightLoader::create()` always
returns the `StandardLoader`.

**Practical implication for a hotkey app:** engine construction is the one
call worth keeping off the UI thread, but it is not a one-time startup cost —
it's cheap enough to re-pay every time the user wakes the model up, which is
exactly what the TTL-offload pattern below relies on.

### 1.3 Layer offloading (VRAM budget, not a TTL)

Separately from load speed, `num_gpu_layers` (`RuntimeConfig::num_gpu_layers`)
controls how many *leading* transformer layers keep their weights and KV
cache resident in VRAM for the *entire lifetime* of the engine. The rest live
in a pinned host-RAM arena and are streamed into two ping-pong device staging
slots on a dedicated CUDA stream, prefetched one layer ahead of compute
(`VRAMArena::prefetch_layer` / `ensure_layer_ready`). This is a **static VRAM
budget**, decided once at construction (default: all layers resident, or
override via the `BLACKWELL_GPU_LAYERS` env var) — it is not an idle-timeout
or eviction mechanism, and there is currently no per-layer LRU/TTL policy in
the engine itself.

### 1.4 Where "TTL offload" actually lives: the application layer

> **Note:** the engine does not expose a built-in idle-timeout / auto-unload
> API. There is no `engine->unload()` or TTL manager class. What it *does*
> give you is (a) a destructor that fully and deterministically frees every
> VRAM/pinned-host allocation, and (b) a constructor fast enough (thanks to
> DirectStorage) to redo on demand. A hotkey app builds "zero-footprint TTL
> offloading" on top of those two primitives, in its own event loop:

```
 [idle]  --hotkey--> [construct BlackwellEngine]  --translate-->  [idle timer armed]
   ^                          (DirectStorage load)                       |
   |                                                                     |
   +-------- reset() the engine (frees VRAM + pinned RAM) <--- TTL expires
```

Concretely: hold the engine in a `std::unique_ptr<BlackwellEngine>`. On
hotkey, lazily construct it if null. After each response, (re)start an idle
timer. If the timer fires before the next hotkey, call `engine.reset()` —
the destructor tears down the `VRAMArena` (all `cudaFree`s) and the pinned
host pool, returning the process to a near-zero GPU/RAM footprint until the
next hotkey re-triggers construction. Section 3 shows this end to end.

### 1.5 Per-request lifecycle (once the engine exists)

```
tokenizer_->encode(...)              // text -> ids (chat template applied)
   |
   v
engine->forward(tok, pos)  x N       // prefill: one call per prompt token
   |                                    (position-addressed KV cache)
   v
engine->forward(tok, pos)  loop      // decode: feed the sampled id back in
   |                                    until eos / stop string / max tokens
   v
tokenizer_->decode(tok)               // id -> text, streamed incrementally
```

`forward()` is auto-regressive and **position-addressed**: `forward(token_id,
pos)` writes the KV cache at `pos` and returns the next sampled token id.
Re-decoding at a `pos` you've already written silently overwrites that slot —
this is how KV-cache reuse across turns works (see `reset_state` below for
the one case where it *isn't* safe).

---

## 2. Core API Reference

### 2.1 `BlackwellEngine` — [`engine.h`](../include/blackwell/engine.h)

The engine owns one loaded model: its weights, KV cache(s), and (for hybrid
checkpoints) recurrent SSM state.

```cpp
enum class KVCacheMode { Continuous, Paged };
```
- `Continuous` (default): legacy FP32 contiguous KV cache + layer offloading.
- `Paged`: bf16 paged KV cache with Copy-on-Write fork/rewind (agent
  tree-search). Requires a model whose attention state is snapshot-able —
  see `ModelCapabilities::supports_cow_branching`.

```cpp
BlackwellEngine(const std::string& index_path,
                size_t max_seq_len = 2048,
                size_t num_gpu_layers = SIZE_MAX,   // all-resident
                KVCacheMode kv_mode = KVCacheMode::Continuous);

BlackwellEngine(const std::string& index_path,
                 const blackwell::InferenceConfig& request);   // tier-2, preferred
```
`index_path` is the safetensors index file; `config.json` is read from its
parent directory. The tier-2 constructor is preferred for new integrations —
state *what you want* (context length, branching) and let
`build_and_validate_runtime()` pick the execution plan.

```cpp
int   forward(int token_id, int pos, float temperature = 0.6f,
              float top_p = 0.9f, int seq_id = 0);
float forward_eval(int token_id, int pos, int target_token_id, int seq_id = 0);
```
`forward` runs one auto-regressive step and returns the sampled next token
id. `forward_eval` scores a specific target token instead of sampling (useful
for perplexity / teacher-forcing checks). `seq_id` selects the sequence
(`Paged` mode only; `Continuous` supports only `seq_id = 0`).

```cpp
void fork(int parent_id, int child_id);   // Paged only; CoW-shares KV pages
void rewind(int seq_id, int pos);         // Paged only; rolls back to `pos`
```
Both throw `std::runtime_error` under `Continuous` mode or when the loaded
model's capabilities forbid branching.

```cpp
void reset_state(int seq_id = 0);
```
Zeroes the recurrent linear-attention (SSM) state for hybrid checkpoints. The
attention KV cache is position-addressed and self-heals on re-decode, but SSM
state accumulates across **every** `forward()` call with no rewind. **Call
this before re-prefilling a fresh or divergent prompt on a hybrid model**, or
the new sequence decodes on top of the previous conversation's recurrent
state (silent collapse/repetition). No-op for dense (non-SSM) checkpoints —
safe to call unconditionally.

```cpp
ModelCapabilities get_capabilities() const;
```
See §2.2 — check this once after construction to decide whether your app can
rely on branching/incremental KV reuse for the loaded checkpoint.

### 2.2 `ModelCapabilities` — [`engine.h`](../include/blackwell/engine.h)

```cpp
struct ModelCapabilities {
    bool supports_cow_branching   = false;  // fork()/rewind() permitted
    bool requires_ssm_subsystem   = false;  // >=1 AttnKind::Linear layer
    bool is_hybrid                = false;  // mixes Full + Linear layers
    int  num_full_attention_layers   = 0;
    int  num_linear_attention_layers = 0;
};
```
Derived once from the parsed `ModelConfig`. Hybrid SSM checkpoints (e.g.
Qwen3.5) reject branching outright — their recurrent state can't be
snapshotted — so tree-search-style apps should check
`supports_cow_branching` before attempting `fork()`.

### 2.3 Configuration pipeline — [`runtime_config.h`](../include/blackwell/runtime_config.h)

```cpp
struct InferenceConfig {                 // tier 2: what you ask for
    size_t max_context_length = 2048;
    size_t max_batch_size     = 1;
    bool   require_branching  = false;
    float  temperature = 0.6f;
    float  top_p       = 0.9f;
};

struct RuntimeConfig {                   // tier 3: resolved execution plan
    size_t max_seq_len   = 2048;
    size_t max_sequences = 1;
    BlackwellEngine::KVCacheMode kv_mode = BlackwellEngine::KVCacheMode::Continuous;
    int    paged_branch_factor = 4;
    size_t num_gpu_layers = RuntimeConfig::kAllLayersResident;
    bool   uses_dedicated_full_attention = false;
};

struct RuntimeOverrides {                // optional low-level knobs
    std::optional<BlackwellEngine::KVCacheMode> kv_mode;
    std::optional<size_t> num_gpu_layers;
    std::optional<int>    paged_branch_factor;
};

ModelCapabilities derive_capabilities(const ModelConfig& model);
RuntimeConfig build_and_validate_runtime(const ModelConfig& model,
                                         const ModelCapabilities& caps,
                                         const InferenceConfig& request,
                                         const RuntimeOverrides& overrides = {});
```
`num_gpu_layers` can also be forced globally via the `BLACKWELL_GPU_LAYERS`
environment variable when left at its `SIZE_MAX` ("all resident") default.

### 2.4 `ModelConfig` / `ConfigLoader` — [`config.h`](../include/blackwell/config.h)

`ConfigLoader::load_from_json(path)` parses a checkpoint's `config.json` into
the tier-1 `ModelConfig` struct: topology (`hidden_dim`, `num_layers`,
`head_dim`, ...), quantization strategy (`QuantStrategy::{NONE, ROWWISE_FP8,
WEIGHT_ONLY_PACKED, COMPRESSED_TENSORS_INT4}`), and hybrid-attention layout
(`layer_types`, `LinearAttnConfig`) for mixed SSM/full-attention checkpoints.
You rarely construct this yourself — `BlackwellEngine`'s constructor loads it
from `index_path`'s parent directory.

### 2.5 Tokenizer & chat template — [`tokenizer.h`](../include/blackwell/tokenizer.h), [`chat_template.h`](../include/blackwell/chat_template.h)

```cpp
class TokenizerFactory {
public:
    static std::unique_ptr<ITokenizer> create(const std::string& model_dir);
};
```
Reads `tokenizer.json` (required) plus `tokenizer_config.json` /
`generation_config.json` / `config.json` (optional, for special tokens and
the chat template).

```cpp
class ITokenizer {
public:
    std::vector<int> encode(const std::string& text, bool add_special_tokens) const;
    std::string      decode(int token_id, bool render_special = false) const;
    std::string      decode(const std::vector<int>& ids, bool render_special = false) const;

    std::vector<int> apply_chat_template(const std::vector<ChatMessage>& messages,
                                         bool add_generation_prompt) const;
    // Incremental variants for a streaming chat loop with a persistent KV cache:
    std::vector<int> encode_chat_prelude(const std::string& system_prompt) const;
    std::vector<int> encode_chat_message(const ChatMessage& msg) const;
    std::vector<int> encode_generation_prompt() const;

    bool is_stop(int token_id) const;               // checks special_tokens().stop_ids
    const SpecialTokens& special_tokens() const;
};
```

```cpp
struct SpecialTokens {
    int bos = -1, eos = -1, unk = -1, pad = -1;      // -1 == undefined for this checkpoint
    std::vector<int> stop_ids;
};
```

`IChatTemplate` renders a conversation into the model's native prompt DSL
(ChatML, Llama-3 headers, ...) — `ChatTemplateFactory::from_jinja_source()`
detects the family from the Jinja source string in `tokenizer_config.json`.
You typically don't touch this directly; `ITokenizer::apply_chat_template`
wraps it.

### 2.6 `IWeightLoader` — [`weight_loader.h`](../include/blackwell/weight_loader.h)

```cpp
class IWeightLoader {
public:
    virtual void load_to_vram(const std::string& filepath, size_t offset,
                              size_t size, void* d_ptr) = 0;
    virtual void flush() {}                    // completion barrier
    static std::unique_ptr<IWeightLoader> create();   // DirectStorage if built in, else Standard
};
std::unique_ptr<IWeightLoader> create_standard_loader();  // always available
```
Internal to `VRAMArena` — application code doesn't call this directly, but
it's the seam that explains why construction is fast: batched loaders may
*defer* the transfer, only guaranteeing completion after `flush()`.

**Build flag:** DirectStorage is opt-in and Windows-only:
```
cmake -DUSE_DIRECT_STORAGE=ON ...
```
Without it (or on failure at runtime), everything falls back to the
synchronous `StandardLoader` (`fread` + `cudaMemcpy`) — same API, no code
changes required.

---

## 3. Minimal Integration: Hotkey Translator

The flow below is the one a background translation tool actually needs:
lazy/idle-triggered engine construction, a token-by-token generation loop,
and a TTL timer that frees the engine's memory after a period of inactivity.

```cpp
#include "blackwell/engine.h"
#include "blackwell/tokenizer.h"
#include "blackwell/chat_template.h"
#include "blackwell/runtime_config.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <string>

using namespace std::chrono_literals;

class HotkeyTranslator {
public:
    explicit HotkeyTranslator(std::string model_dir)
        : model_dir_(std::move(model_dir)) {}

    // Called from the global-hotkey handler. Cheap to call repeatedly: it
    // only pays the DirectStorage load cost when the engine isn't resident.
    std::string translate(const std::string& source_text, const std::string& target_lang) {
        ensure_loaded();

        // --- 1. Build the prompt via the checkpoint's native chat template ---
        std::vector<blackwell::ChatMessage> messages = {
            {"system", "You are a terse translation engine. Output only the translation."},
            {"user",   "Translate to " + target_lang + ": " + source_text},
        };
        std::vector<int> prompt = tokenizer_->apply_chat_template(messages, /*add_generation_prompt=*/true);

        // --- 2. Prefill: one forward() per prompt token, position-addressed ---
        // Hybrid (SSM) checkpoints must zero recurrent state before a fresh
        // prefill; no-op for dense models, so this is always safe to call.
        engine_->reset_state();

        int pos = 0;
        int next = -1;
        for (int tok : prompt) {
            next = engine_->forward(tok, pos, 0.0f, 1.0f);  // temperature 0: deterministic
            ++pos;
        }

        // --- 3. Decode loop: feed each sampled token back in until stop ---
        std::string out;
        const int kMaxNewTokens = 512;
        for (int generated = 0; generated < kMaxNewTokens; ++generated) {
            if (tokenizer_->is_stop(next)) break;

            out += tokenizer_->decode(next);            // token-by-token text
            next = engine_->forward(next, pos, /*temperature=*/0.2f, /*top_p=*/0.9f);
            ++pos;
        }

        arm_ttl_timer();   // (re)start the idle countdown after a successful call
        return out;
    }

    // Idle-timer callback (wire to your event loop's timer facility).
    // Frees every VRAM/pinned-host allocation; next translate() reloads from
    // scratch via DirectStorage.
    void on_ttl_expired() {
        engine_.reset();
        tokenizer_.reset();
    }

private:
    void ensure_loaded() {
        if (engine_) return;   // still resident from a recent call

        tokenizer_ = blackwell::TokenizerFactory::create(model_dir_);

        blackwell::InferenceConfig request;
        request.max_context_length = 4096;
        request.max_batch_size     = 1;        // this app never needs branching
        request.require_branching  = false;

        // The slow, DirectStorage-backed call. Off the UI thread in a real app.
        engine_ = std::make_unique<BlackwellEngine>(model_dir_ + "/model.safetensors.index.json",
                                                     request);
    }

    void arm_ttl_timer() {
        // Reset your platform timer here (e.g. SetTimer / std::thread + condvar /
        // an event-loop deadline) to fire on_ttl_expired() after, say, 60s idle.
    }

    std::string model_dir_;
    std::unique_ptr<blackwell::ITokenizer> tokenizer_;
    std::unique_ptr<BlackwellEngine>       engine_;
};
```

**Notes on the snippet:**
- `translate()` is safe to call again immediately after `on_ttl_expired()` —
  it just re-triggers the DirectStorage load path from §1.2.
- This skips the incremental KV-reuse / stop-string-scanning optimizations
  used by the in-tree playground adapter
  (`src/tools/playground/blackwell_llm_adapter.cpp`) for brevity; look there
  for a production-hardened reference implementation (prefix-matching KV
  reuse across turns, streaming callback, multi-branch `seq_id` handling).
- `num_gpu_layers` was left at its default (all-resident) in `InferenceConfig`
  since it isn't exposed there — pass it via `RuntimeOverrides` to
  `build_and_validate_runtime`, or use the legacy
  `BlackwellEngine(path, max_seq_len, num_gpu_layers, kv_mode)` constructor,
  or set `BLACKWELL_GPU_LAYERS` if you need a fixed VRAM budget on smaller GPUs.
