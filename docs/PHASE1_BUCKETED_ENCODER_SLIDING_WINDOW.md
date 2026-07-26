# Phase 1 — Bucketed Whisper Encoder & Sliding-Window Context

Status: **implemented** (2026-07-26) — bucketed encoder + `SlidingAudioWindow` landed
and unit-tested (`tests/integration/test_whisper_encoder_buckets.cpp`, 4 tests green);
the live streaming wire-up (pipeline delta staging + Tier-2 GPU self-consistency test)
is the remaining step. Builds on Phase 0 (KV checkpoint/rollback + `VRAMArena::truncate_kv`).
Two complementary features that turn the batch-oriented 30 s encoder into a low-latency
streaming front-end.

> **Motivation.** Today every utterance is padded to 30 s (3000 mel frames → 1500
> encoder frames → 188 soft-tokens) and pushed through ONE graph. A 300 ms live chunk
> pays the full 30 s encode (O(1500²) bidirectional attention dominates). Phase 1 (a)
> sizes the encode to the audio via **graph buckets**, and (b) keeps a rolling
> **acoustic-context window** while appending only the **delta** soft-tokens to Llama.

---

## Feature 1 — Bucketed CUDA graphs in `WhisperEncoder`

### The constraint today
`WhisperEncoder` captures **one** `cudaGraphExec_t` whose shapes are fixed to
`cfg_.conv_frames = 3000` / `cfg_.max_source_positions = 1500`. `record()` reads those
members directly; `forward(d_mel)` copies a fixed `d_input_.size_bytes()` and launches
the single graph (`whisper_encoder.cu`).

### Design
Keep the workspace buffers sized for the **largest** bucket (3000/1500) — a smaller
bucket is a prefix of the same allocations, so no extra VRAM and no reallocation.
Capture **one graph per bucket**, lazily, and select the smallest bucket whose frame
count covers the input.

- **Buckets** (`conv_frames → encoder seq → soft-tokens @ stack 8`):

  | conv_frames | ≈ duration | encoder seq (`conv_out_frames`) | soft-tokens |
  |---|---|---|---|
  | 150  | 1.5 s | 75   | 10  |
  | 300  | 3 s   | 150  | 19  |
  | 500  | 5 s   | 250  | 32  |
  | 3000 | 30 s  | 1500 | 188 |

  `conv_out_frames(cf) = (cf - 1) / 2 + 1` (conv2 stride 2). The 3000 bucket stays the
  parity reference; the golden-dump integration test pins **only** that bucket.

- **Parameterise the recorder.** `record(cudaStream_t)` → `record(cudaStream_t s, int
  conv_frames)`: derive `cf`, `c2 = conv_out_frames(cf)`, `seq = c2` from the argument
  instead of `cfg_`. Every GEMM extent, the positional-embedding add (uses
  `embed_positions_[0:seq]`), and the O(seq²) attention shrink with the bucket. Nothing
  else in `record()` changes — it stays bit-identical between eager warm-up and capture.

- **Per-bucket graph table** replaces the single `graph_/graph_exec_/graph_ready_`:

  ```cpp
  struct GraphBucket {
      int             conv_frames = 0;      // input frames this graph is captured for
      int             seq         = 0;      // conv_out_frames(conv_frames)
      cudaGraph_t     graph       = nullptr;
      cudaGraphExec_t exec        = nullptr;
      bool            ready       = false;
  };
  std::vector<GraphBucket> buckets_;        // sorted ascending by conv_frames
  int last_seq_ = 0;                        // valid-output length of the last forward()
  ```

- **Selection + input packing.** New overload
  `const float* forward(const float* d_mel, int mel_frames)`:
  1. pick `b = ` smallest bucket with `conv_frames >= mel_frames` (clamp to largest);
  2. lazily `build_graph(b)` (eager warm-up at `b.conv_frames` + capture) if `!b.ready`;
  3. pack the input as `[num_mel_bins, b.conv_frames]`: async `cudaMemset` the used
     region to `mel_pad_value`, then `cudaMemcpy2DAsync` the `mel_frames` real columns
     (dpitch = `b.conv_frames`, height = `num_mel_bins`). Padding happens **outside**
     the graph, so it is legal per-call;
  4. `cudaGraphLaunch(b.exec)`, set `last_seq_ = b.seq`, return `output_.get()`.

  The existing `forward(d_mel)` becomes `return forward(d_mel, cfg_.conv_frames);` —
  100 % source-compatible with the current `--wav` path and tests.

- **Reported geometry.** `output_frames()` returns `last_seq_` (was the constant 1500),
  so the projector is driven with the bucket's real frame count. Add
  `int valid_output_frames(int mel_frames) const { return conv_out_frames(mel_frames); }`
  for the sliding-window slicer (the sub-bucket real length).

### Proposed header diff (`src/audio/whisper_encoder.h`)

```diff
 struct WhisperEncoderConfig {
     ...
     int conv_frames = 3000;           // mel time frames fed to conv1 (30 s)
     float ln_eps = 1e-5f;
+
+    // Graph buckets (conv_frame counts) the encoder may capture, ascending; the
+    // LARGEST must be >= conv_frames (it is the parity reference + workspace sizing).
+    // A forward() rounds its input UP to the smallest covering bucket.
+    std::vector<int> graph_buckets = {150, 300, 500, 3000};
+    // Log-mel value for the padded tail of a partially-filled bucket (see Feature 2).
+    float mel_pad_value = 0.0f;

     int head_dim() const { return d_model / num_heads; }
     int conv_out_frames() const { return (conv_frames + 2 * 1 - 3) / 2 + 1; }
+    static int conv_out_frames_of(int cf) { return (cf + 2 * 1 - 3) / 2 + 1; }
 };
```
```diff
     const float* forward(const float* d_mel);
+    // Bucketed forward: sizes the encode to `mel_frames` (rounds up to a captured
+    // bucket, pads the tail). Returns [output_frames(), d_model]; output_frames()
+    // reflects the SELECTED bucket after the call. Back-compat overload above feeds
+    // the full 3000-frame bucket.
+    const float* forward(const float* d_mel, int mel_frames);
     ...
-    int output_frames() const { return cfg_.max_source_positions; }
+    int output_frames() const { return last_seq_ ? last_seq_ : cfg_.max_source_positions; }
+    // Real (unpadded) encoder frames for `mel_frames` of input — the valid prefix
+    // the projector/delta-slicer should trust inside a padded bucket.
+    int valid_output_frames(int mel_frames) const {
+        return WhisperEncoderConfig::conv_out_frames_of(mel_frames);
+    }
```
```diff
-    void record(cudaStream_t s);
+    void record(cudaStream_t s, int conv_frames);
     ...
-    void build_graph(const float* d_mel);
+    void build_graph(GraphBucket& b);            // eager warm-up @ b.conv_frames + capture
+    GraphBucket& select_bucket(int mel_frames);  // smallest covering bucket (lazy table)
     ...
-    cudaGraph_t graph_ = nullptr;
-    cudaGraphExec_t graph_exec_ = nullptr;
-    bool graph_ready_ = false;
+    std::vector<GraphBucket> buckets_;           // sorted ascending; captured lazily
+    int last_seq_ = 0;                           // valid output length of last forward()
```

`whisper_encoder.cu` changes: `record()` takes `conv_frames`; `build_graph` operates on
a `GraphBucket&`; the dtor frees every bucket's `graph`/`exec`. No kernel or numerics
change — TF32/FP16 accumulation and the fused epilogues are untouched.

### Cost & caveats
- **One-time capture per bucket** (~tens of ms warm-up + a graph instantiation, lazy on
  first use). Four buckets ≈ four warm-ups over the session; the hot path is still a
  single `cudaGraphLaunch`.
- **Sub-30 s encodes are an approximation of the 30 s-padded reference.** Whisper's
  absolute positional embeddings + bidirectional attention were trained on 30 s, and the
  zero-/floor-padded tail leaks into valid frames through unmasked attention. This is
  acceptable (and is the entire latency point); parity-vs-PyTorch stays on the 3000
  bucket. A padded-position attention mask is a **future** refinement, not Phase 1.

---

## Feature 2 — Sliding window & delta soft-token extraction

New header-only component **`src/audio/sliding_audio_window.h`** (stub committed).

### The idea
Keep a rolling mel window `[ history (~2 s) ++ new chunk (~300 ms) ]`. Encode + project
the **whole** window (Feature-1 bucket sized to the window), then append **only** the
soft-tokens for the new chunk (the delta) to Llama's KV. History tokens are recomputed
each hop for acoustic context and discarded.

### The alignment trick (why it's exact)
One soft-token spans `conv_stride(2) * stack_factor(8) = 16` mel frames = **160 ms**.
Expressing window/hop in whole soft-tokens (multiples of 16 mel frames) makes the
history↔new boundary a clean soft-token edge, so the delta is an exact **tail slice** of
the projector output — no fractional-token resampling.

Default `SlidingWindowConfig`: `history_tokens = 12` (1.92 s), `hop_tokens = 2` (320 ms)
→ `window_tokens = 14` (2.24 s, 224 mel frames → bucket **300**). Steady-state: append
the **last 2** soft-tokens per hop.

### Delta rule
- **Warm (steady state):** `delta = projector_out[out_soft_tokens - hop_tokens :
  out_soft_tokens]` — the newest `hop_tokens` rows. Robust to the conv `+1`/ceil rounding
  because it counts from the tail, which is always freshest-audio.
- **Cold start** (history not yet filled): `delta =` every valid row (the whole
  utterance so far), so nothing is dropped before the ring fills. `warm()` gates this.

### Data flow (per hop, all on the audio stream — no host sync)
```
new PCM chunk ─▶ WhisperDSP log-mel ─▶ SlidingAudioWindow::push_chunk (ring update)
   window_mel() [128, 224] ─▶ WhisperEncoder::forward(mel, 224)  (bucket 300)
   ─▶ UltravoxProjector::forward(hidden, valid_output_frames(224)=112)  ─▶ [14, 4096]
   ─▶ extract_delta(out, 14) ─▶ {ptr → rows[12:14], count 2}
   ─▶ PingPong slot ─▶ RealEngineControl::prefill_audio_embeddings(count=2)  → KV += 2
```

### Wiring (no engine-core or public-API change)
- **Pipeline seam.** Either thread variable frames through `AudioEmbeddingPipeline`
  (`process_frame(d_mel, mel_frames, pp, frame)` → `encoder_.forward(d_mel, mel_frames)`
  → `projector_.forward(hidden, encoder_.valid_output_frames(mel_frames), stream)` →
  stage **only the delta rows**), or add a sibling `StreamingAudioPipeline` that owns a
  `SlidingAudioWindow`. The ping-pong slot is already sized to a full frame's
  `embed_elems()`, so a `hop_tokens`-row delta fits with room to spare.
- **Consumer.** `RealEngineControl::prefill_audio_embeddings(pp, frame, num_audio)`
  already loops `num_audio` rows and advances `pos_` per row — pass `delta.count`. No
  change to `step_*` or `forward_status`.
- **Speculation (ties in Phase 0).** A VAD-speculative hop:
  `cp = kv_cache_checkpoint();` → prefill delta → decode; if the chunk is later rejected,
  `kv_cache_rollback(cp)` drops exactly those delta tokens (and, via `truncate_kv`, any
  offload high-water marks). Accepted hops never roll back; KV stays monotonic.

### `SlidingAudioWindow` interface (see the stub for full doc comments)
```cpp
struct SlidingWindowConfig { int num_mel_bins, history_tokens, hop_tokens; float mel_pad_value;
                             int window_tokens(); int window_mel_frames(); };
struct DeltaTokens        { const float* data; int count; int text_hidden; };

class SlidingAudioWindow {
    SlidingAudioWindow(const SlidingWindowConfig&, int text_hidden);
    void        push_chunk(const float* d_chunk_mel, int chunk_frames, cudaStream_t);
    const float* window_mel() const;  int window_mel_frames() const;
    DeltaTokens extract_delta(const float* d_projector_out, int out_soft_tokens) const;
    int  hop_tokens() const;  bool warm() const;
};
```

---

## Build & test plan
- Feature 1 is behavior-preserving for the 3000 bucket: existing `--wav` + Ultravox
  parity tests must stay green (the back-compat `forward(d_mel)` overload).
- New unit tests (tests/integration, `blackwell_audio`): (a) bucket selection picks the
  smallest covering graph and pads correctly; (b) small-bucket encode matches a
  full-bucket encode on its **valid prefix** within the cosine bar for a synthetic
  band-limited tone; (c) `extract_delta` tail/cold-start indices; (d) end-to-end
  streaming hop grows Llama KV by exactly `hop_tokens` and a `kv_cache_rollback` undoes it.
- Add `sliding_audio_window.cu` (impl of the stub) to `src/audio/CMakeLists.txt`'s
  explicit source list when Phase 1 lands (cmake-hygiene: never glob).

## Files
- `src/audio/whisper_encoder.{h,cu}` — bucketed graphs (**landed**; the diff above is
  what shipped: `graph_buckets`, `record(s, conv_frames)`, `GraphBucket` table,
  `forward(d_mel, mel_frames)`, `select_bucket`, `output_frames()`/`valid_output_frames`).
- `src/audio/sliding_audio_window.h` — **landed** (header-only, double-buffered window +
  tail-delta `extract_delta`).
- `tests/integration/test_whisper_encoder_buckets.cpp` — **landed** (bucket geometry /
  finite output / back-compat / range-throw / window shift + tail delta; random weights,
  no checkpoint needed).
- `src/audio/audio_embedding_pipeline.h` — variable-frame `process_frame` + delta staging
  (**pending** — the live wire-up).
- `audio_sandbox/translator/real_engine_control.hpp` — streaming hop driver (consumer
  already delta-ready; checkpoint/rollback from Phase 0) (**pending**).
