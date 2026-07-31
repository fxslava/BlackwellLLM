# Continuous Streaming Translation (Phase 8)

Simultaneous speech translation with **no reset between utterances**: one KV cache that
lives for the whole session, one audio ring, and a translation that is continuously
re-drafted and then committed.

This replaces the ping-pong (dual-slot) experiment on `feat/ping-pong-audio-mvp`.
Double-buffering produced clean slot isolation but broke the LLM's semantic context at
every swap — each slot began the world anew. The architecture here keeps a single
sequence and instead makes the *tail* of that sequence cheap to throw away.

> Status: **end to end on the 8B backbone** (T1–T5). The loop, the eviction kernel and the
> live engine adapter are built and exercised together by an offline driver over the
> reference clip (§6). What remains is the live/mic wiring and the latency work — see §7.

---

## 1. The strategy: Re-translation (draft-and-commit)

This is the standard SimulMT approach (Seamless, USM): rather than teaching the model to
*revise* a translation in place, we re-run it. The model never sees a fragmented prompt
and is never asked to reason about its own earlier draft, so there is nothing to
hallucinate around.

Three zones in KV position order:

```
  [0, S)          FROZEN PREFIX   system prompt. Never rewound, never evicted.
  [S, C)          COMMITTED       finalized utterances + their translations.
  [C, tail)       DRAFT           the in-flight re-translation. Disposable.
```

`C` is the **commit pointer**: the end of the last fully finalized utterance.

**Drafting.** While the user speaks, the segmenter emits `Partial` events on a fixed
cadence, each covering the utterance *from its first sample* to now. For each one:

1. rewind the KV cache to exactly `C` — this drops the previous draft's audio **and** its
   generated text;
2. append the whole (grown) utterance audio;
3. decode a draft translation, display it as provisional.

**Committing.** When the release hangover expires the segmenter emits `Final`. Same three
steps against the complete utterance, then **advance `C` to the new tail**. The draft
becomes permanent and the next utterance drafts on top of it.

Two consequences worth being explicit about:

- **Draft text is never folded into committed context.** Only a `Final`'s output survives.
  A wrong guess made at 300 ms of audio cannot poison the session.
- **The prompt is always well-formed.** The model reads
  `[prefix][committed turns][one whole utterance]` every single time. There is no
  "here is a partial sentence, update your previous answer" prompting, and no
  mid-assistant-turn audio splice (which would be out of distribution for Ultravox).

The cost is recomputation: a rewind-and-refeed re-prefills the utterance's audio tokens
and re-decodes its text. That is bounded by one utterance, and it is why the rewind has to
be free (§3).

---

## 2. Front end: ring buffer + Silero as a segment gate

30 s FIFO of `float` @ 16 kHz = 1.92 MB, matching the encoder's ceiling.

Silero gates the ring, **but at segment granularity, not per 10 ms block.** Whisper's
encoder is bidirectional with a Conv1D receptive field; excising individual blocks splices
two acoustic contexts together at exactly the point a word has to be recovered. Empirically
this is not a theoretical worry in one direction and is in the other:

- Silero holds `p >= 0.9539` straight across the reference clip's 3.24–3.32 s inter-word
  pause (`tests/vad/silero_vad_test.cpp`), which the RMS detector reads as silence at
  −62 dB. So real pauses are *not* dropped by a per-block gate anyway.
- A keyboard clack scoring low between two words *would* punch a hole through a phoneme.

So `SpeechSegmenter` (§4) consumes probabilities and emits whole segments, with an onset
**pre-roll** and a release **hangover**. Room tone between segments never enters the ring.
The pre-roll is Phase 6's carry-over lesson restated: the encoder is weakest at a window's
left edge, so never hand it a segment that starts on the first phoneme.

Because silence is dropped, ring time is not wall-clock time. Keep a parallel
`(ring_offset -> source_sample)` index for timestamps and for the existing encoder
position-offset machinery.

---

## 3. Why the rewind is free

`ContinuousKVManager::rewind()` delegates to `VRAMArena::truncate_kv(target_pos)`, which is
pure bookkeeping — the FP32 K/V slabs are position-addressed and simply overwritten in
place on the next prefill. Nothing is moved, zeroed, or reallocated.

That is the whole reason re-translation is affordable here: the tail rewind that happens on
*every* `Partial` costs O(1), and only the recompute above `C` is real work.

---

## 4. Workers, and the single-threaded doctrine

This is **not** two threads driving the engine. Exactly one thread may touch
`BlackwellEngine` after load.

| Thread | Owns | Never touches |
|---|---|---|
| **A — audio/DSP** (exists) | mic callback, `SileroVAD`, `SpeechSegmenter`, `AbsoluteAudioRing`, posting to `SegmentJobQueue` | the engine |
| **B — engine** | the only `step_*` caller. Drains `SegmentJobQueue`, runs the redraft (encode → prefill → decode), commits, evicts | the ring's write end, ORT |

### The encoder worker collapsed — and that is a consequence of re-translation

The original plan called for a third thread producing soft tokens into an SPSC descriptor
queue, ping-pong style. **Re-translation removes it.** Every redraft re-encodes the
utterance *from its first sample* (that is the growing-window invariant), so there is no
incremental soft-token stream to produce, no partial frames to hand across, and nothing
for a descriptor queue to carry. `prefill_audio(begin, end)` encodes the whole window
inline, on the engine thread, on its own CUDA stream.

What remains of the original design is exactly one cross-thread handoff — `SegmentJobQueue`
— and it carries *segment boundaries*, not encoded tensors. That is a strictly smaller and
easier thing to get right: a dropped descriptor would have corrupted the audio timeline,
whereas a dropped `Partial` merely costs draft freshness (and a `Final`, which must never
drop, is protected explicitly).

The cost is recomputation, and it is the same cost the architecture already accepted for
the decode side. At a 500 ms cadence over a 15 s ceiling, the encoder re-runs on a window
that grows to ~146 ms of work — the bucketed CUDA graphs from Phase 1 are what keep that
affordable, and they carry over unchanged.

> **Measured (T5, Release): the encoder cost is not what matters.** The graphs work so well
> that the re-encode disappears into the noise, and the redraft turns out to be linear in
> TOTAL draft tokens at ~20.5 ms/token — i.e. bounded by the per-token engine step, not by
> the audio front end. See §7 for the numbers and what to do about them.

Rate: 16 mel frames per soft token × 160 samples = **160 ms of audio per soft token**
(6.25/s). A 500 ms `Partial` cadence is ~3 new soft tokens per redraft.

---

## 5. Head eviction: the sliding window (not yet built)

Eviction is what makes the session unbounded. It operates **strictly behind `C`** — the
draft zone is never involved — so it is pure background housekeeping and cannot interact
with an in-flight redraft.

**The constraint.** `rope_kv_append_kernel` computes `angle = pos * freq` and rotates K
*before* writing it into `float[kv_heads][max_seq_len][head_dim]`
(`src/kernels/rope.cu:64`). Cached keys carry **absolute** positional phase, so dropping
head tokens and sliding the rest down would leave every survivor at the wrong relative
distance. Leaving a positional gap instead is no better: absolute positions then grow
without bound and the sink→current distance degrades exactly as it does past the trained
context.

**The fix is exact.** `freq` is a pure function of `(k, head_dim, rope_theta, scaling)` —
position-independent — so RoPE rotations compose. A key cached at `p` carries `p·θⱼ` per
channel pair; applying a further `−Δ·θⱼ` yields precisely the key that would have been
cached at `p − Δ`. V is stored unrotated and only moves.

`evict_head(Δ)`, one fused kernel per layer:

1. move rows `[S+Δ, len)` → `[S, len−Δ)` for every kv_head (strided);
2. in the same pass, rotate the K rows by `−Δ·θⱼ`, reusing the existing `rotate_half`
   pairing (`k` with `k + head_dim/2`) and the same `RopeScaling` struct;
3. rows `< S` are never touched — the frozen prefix keeps absolute phase `0..S−1` and
   remains the attention sink anchored at true position 0;
4. shift `C` and `tail` down by Δ and reconcile the arena high-water marks.

**Batch it.** Trigger on a high-water mark and evict a large block. At 2048 context with 8B
geometry (32 layers × 8 kv_heads × 128 head_dim × fp32 × K+V ≈ **256 KB/token**, ~512 MB
resident), compacting two-thirds is ~700 MB of traffic — sub-millisecond, amortized over
~600 tokens ≈ **~1 µs/token**. Not a bottleneck.

**Cut semantically.** `KvLedger` (§6) only ever cuts on a turn boundary, so eviction can
never strand a dangling `<|start_header_id|>` or half an utterance.

### What eviction actually promises (measured, not assumed)

Eviction is **positionally exact and content-lossy**, and the difference matters when
reasoning about how the session degrades.

The tempting specification — "an evicted cache equals a from-scratch prefill of the
survivors" — is **false, and not because of a bug**. When `[P][A][B]` was prefilled, every
token of B attended to A, so B's K and V at layers 1..N-1 already encode A's content.
Sliding those rows down and re-phasing them fixes *where* they are; nothing can
un-condition *what* they are. This is the defining lossiness of every sliding-window
scheme, StreamingLLM included.

Measured on the 8B AWQ checkpoint (`tests/integration/test_kv_evict_logit_equivalence.cpp`):

| Observable | Result | Why |
|---|---|---|
| Layer 0 K vs fresh prefill | **cosine 1.00000000**, max abs 1.3e-5 | layer-0 K is a pure function of (token, position) — the one place the positional transform is visible in isolation |
| Layer 0 V vs fresh prefill | max abs 1e-7 | V is moved, never rotated |
| Layer 31 K vs fresh prefill | cosine 0.9924 | survivors retain the evicted context, as above |
| Answer after eviction | identical top-1 to a fresh prefill | the surviving context is still read correctly |

The practical consequence for Phase 8: evicting does not corrupt the session, and it does
not fully erase what was evicted either. Committed history fades rather than vanishing,
which is the desirable behaviour — but it means eviction must never be relied on as a
*privacy* or *reset* mechanism. A genuine reset is `KvLedger::reset()` plus a re-prefill.

Later, if a profile ever demands it: the paged manager (`KVCacheMode::Paged`) makes step 1 a
block-table edit. It does not remove step 2, so it buys the memmove and not the rotation.
Do not start there.

---

## 6. Components

| Component | File | Tier | Status |
|---|---|---|---|
| `SpeechSegmenter` | `src/vad/speech_segmenter.hpp` | CPU, header-only, STL-only | **done (T1)** |
| `KvLedger` | `src/bridge/kv_ledger.hpp` | CPU, header-only, STL-only | **done (T2)** |
| `RetranslationSession` | `src/bridge/retranslation_session.hpp` | CPU, header-only | **done (T4)** |
| `SegmentJobQueue` | `src/bridge/segment_job_queue.hpp` | CPU, header-only | **done (T4)** |
| `AbsoluteAudioRing` | `src/bridge/absolute_audio_ring.hpp` | CPU, header-only | **done (T4)** |
| `RetranslationEngine` (the live `IRetranslationEngine`) | `audio_sandbox/translator/retranslation_engine.hpp` | CUDA | **done (T5)** |
| `evict_head` kernel | `src/kernels/kv_evict.cu` | CUDA | **done (T3)** |
| `ContinuousStreamingConfig` | `src/bridge/continuous_streaming_config.hpp` | CPU, header-only | **done** |
| `offline_retranslate` driver | `audio_sandbox/translator/offline_retranslate.cpp` | CUDA | **done (T5)** |
| Live mic wiring (`SegmentJobQueue` in the path) | `audio_sandbox/translator/` | CUDA | remaining |

There is no "encoder worker" row: it collapsed into `prefill_audio` (see §4). The live
adapter deliberately does NOT subclass `RealEngineControl` — that class carries the
barge-in epoch, command ring, pause checkpoint and injected-soft-token history the
SUPERSEDED strategies need, and re-translation must have none of them. It reuses the
primitives (the `forward_status` prefill, the white-box `step_*` sweep, `kv_mgr->rewind`,
the penalized decode step) and nothing else.

`SpeechSegmenter` and `KvLedger` are deliberately free of CUDA, ORT and the engine: the two
places this design can be wrong in a way that is expensive to debug are *when* a segment is
cut and *where* the cache is cut, and both are now decidable at desk speed.

### Test ladder

- **T1** `tests/vad/speech_segmenter_test.cpp` — segmentation policy. Green.
- **T2** `tests/bridge/kv_ledger_test.cpp` — zone invariants, the draft cycle, eviction
  planning. Green.
- **T3** `tests/validation/test_kv_evict.cpp` (8) — the kernel in isolation: an evicted
  cache must equal one built from scratch with the dropped rows absent at shifted
  positions. The reference is produced by REPLAYING the append kernel rather than by a CPU
  reimplementation, so the test cannot agree with a shared misunderstanding of the
  frequency ladder. Covers the overlap split, llama3 scaling, 8B geometry, repeated
  evictions and the argument guards. **Verified to have teeth by mutation:** zeroing the
  angle fails 6/8, flipping its sign fails 6/8. Green.
- **T3b** `tests/integration/test_kv_evict_logit_equivalence.cpp` (3) — the same property
  inside the real engine at real geometry, plus the lossiness boundary above asserted in
  executable form. Green.
- **T4** `tests/bridge/{retranslation_session,segment_job_queue,absolute_audio_ring}_test.cpp`
  (33) — the redraft mechanic against a recording fake engine: call order, N partials as a
  fixed point on the cache, Final-only commit, atomicity under every failure mode, and
  eviction only-after-commit. **Verified to have teeth by mutation:** committing on every
  segment instead of only on a `Final` fails 5 of 13. Green.
- **T5** `audio_sandbox/translator/offline_retranslate.cpp` — the whole pipeline over the
  reference clip on the real 8B AWQ backbone: Silero → segmenter → ring → session →
  engine, single-threaded (a file has no real-time constraint, so `SegmentJobQueue` is
  deliberately out of the path — mixing it in would make every failure ambiguous between
  a wrong loop and a dropped handoff). Three invariants are checked on EVERY segment:
  `ledger.tail() == the engine's write cursor`, every `Partial` rewound to its utterance's
  commit pointer, and `S` never moved. **Green.** Measured (Debug, RTX 5070):

  | Run | Result |
  |---|---|
  | 1 pass, default watermarks | 10 partials + 1 Final. `begin_sample` pinned at 0.32 s while `end` grew 1.08 → 5.73 s; `C` pinned at 44 across all 10 rewinds; `draft == framing + audio + text` exactly every time; the draft refined "Г-н Кв." → "Мистер Квилтер - апостол среднего пути." → the full sentence. Commit moved C 44 → 138 and drained the draft to 0. |
  | 5 passes, watermarks 200/100 | 20 segments, 5 commits, **2 evictions, 378 tokens reclaimed**, 0 violations. Each eviction cut EXACTLY two utterance spans (188 = 94+94, then 190 = 94+96) — the ledger's turn-boundary rule, never a ragged cut. Cursor tracked the ledger through both (232→138, 234→134). |

  The watermarks were lowered from the shipping 3000/1000 to reach an eviction inside a
  short offline run (~96 committed tokens per utterance would otherwise need ~31 of them);
  the mechanism is watermark-independent, but the number is a test setting, not a default.

  **Transcript coherence across an eviction is printed, not asserted** — whether a
  translation still reads correctly is a human judgement, and a driver that claimed
  otherwise would be lying about what it measured. Observed: the first utterance drafted on
  an evicted cache came back fully coherent. But by the fifth, the model rendered
  "Mr. Quilter is the apostle..." as *"Он является апостолом..."* ("He is the apostle") —
  it pronominalized a subject the surviving committed context had already established.
  That is the shared-session design working as specified AND a translation-quality
  regression at the same time, and it is the strongest argument yet for §7's open question
  about what committed history should retain. See also §5's lossiness table: eviction fades
  history rather than erasing it, so the conditioning survives the cut.

---

## 7. Open questions

- **Partial cadence vs. GPU budget — MEASURED in Release (RTX 5070).** Same clip, same
  args, bit-identical output to the Debug run (greedy determinism across optimization
  levels):

  | Audio window | Draft tokens | Debug | Release |
  |---|---|---|---|
  | 0.76 s | 42 | 15 761 ms | **865 ms** |
  | 2.76 s | 65 | 24 739 ms | **1 254 ms** |
  | 5.26 s | 93 | 36 142 ms | **2 039 ms** |

  ~19x across the board. The cost model is linear in TOTAL draft tokens with a near-zero
  fixed term — every measured point sits in 19.3–21.9 ms/token:

  ```
  redraft_ms ~= 20.5 * (framing + audio_soft + decoded) tokens
  ```

  **The encoder is not the bottleneck.** The bucketed CUDA graphs make the Whisper encode
  effectively free; the redraft is dominated entirely by the per-token engine step. That
  reverses the assumption §4 was written under ("~146 ms of encoder work per redraft is
  what the bucketed graphs keep affordable") — the graphs did their job so well that the
  cost moved somewhere else.

  So `partial_cadence_ms = 500` is NOT achievable: the segmenter emits 2 Partials/s and the
  engine delivers 0.5–1.15/s. It does not back up (latest-wins drops stale Partials before
  they start), so the system degrades to its achievable rate rather than falling behind —
  but the config number lies about observed behaviour. Extrapolating the fit,
  `latency ~= 667 ms + 261 ms per second of audio`, a 15 s utterance's last redraft costs
  **~4.6 s**, which is not "simultaneous". Three levers, in order of leverage:
    1. **Cut `max_utterance_ms` to 6–8 s** — bounds the worst case to ~2.5 s. The forced cut
       is already contiguous (no pre-roll), so it costs nothing semantically.
    2. **Stop re-prefilling the turn prefix.** Those 29 framing tokens are byte-identical on
       every redraft of every utterance yet cost 29 x 20.5 ~= **600 ms each time** — 30% of
       the budget on a short draft. Prefill them once after each commit and rewind to
       `C + prefix_len` instead of `C`; it is a change to the rewind TARGET and nothing else.
    3. Set `partial_cadence_ms` to ~1000 so the knob states what the GPU can deliver.
- **What should committed history retain?** T5 observed the backbone pronominalizing a
  subject that earlier committed turns had established ("He is the apostle..." for audio
  saying "Mr. Quilter is the apostle..."). Each utterance is supposed to be translated on
  its own terms; the shared session makes that not quite true. Options: translate each
  utterance against the frozen prefix only, keep committed audio but drop committed
  translations, or instruct against it in the turn prefix. Not yet decided.
- **Eviction high-water and block size.** Needs the draft zone's worst case
  (`max_utterance_ms` of audio + `max_new_tokens`) as headroom above `C`.
- **Sink width.** The frozen prefix is currently the whole system prompt. StreamingLLM
  suggests as few as 4 sink tokens suffice; if prompt length becomes a problem, that is the
  knob.
