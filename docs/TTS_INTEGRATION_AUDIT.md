# TTS Integration Audit — on-demand speech synthesis for `audio_translator`

**Status:** research / design proposal. No code written. Nothing in this document has landed.
**Scope:** vocalizing finalized utterances in `audio_sandbox/translator` (`audio_translator`),
reusing the ONNXRuntime CPU provider already integrated for Silero VAD.

> Read alongside [`docs/CONTINUOUS_STREAMING.md`](CONTINUOUS_STREAMING.md) (the commit
> pointer, which decides what is safe to speak in Mode B) and
> [`docs/01_architecture_and_threading.md`](01_architecture_and_threading.md) (the
> single-engine-thread doctrine, which TTS must not perturb).

---

## 0. Executive summary

TTS is, structurally, the **easiest** major subsystem this repo has added: it touches no
CUDA, no KV cache, no engine state, and no COM boundary. It is a leaf. The entire
ONNXRuntime acquisition, PIMPL, static-CRT, hash-pinned-model and `if(TARGET ...)`-gating
machinery it needs already exists and is proven by `src/vad/`.

The hard part is not synthesis. It is that **the loudspeaker feeds the microphone**, and the
microphone drives a neural VAD whose verdict drives a barge-in epoch that cancels
generation. That single coupling is the only part of this feature that reaches back into
code governed by the threading doctrine. Everything else is additive.

| Question | Recommendation | Confidence |
|---|---|---|
| Model architecture | VITS/Piper-family ONNX, behind a swappable `ITtsBackend` | high |
| Text frontend | `IPhonemizer` seam; **the espeak-ng vs. grapheme-model choice is a licensing decision, not a technical one** — see §1.2 | medium (blocked on a product answer) |
| Execution provider | **CPU**, `intra_op_num_threads = 2`, sequential | high |
| Threading | One dedicated `TtsWorker` thread, outside the doctrine; monotone `speak_epoch` for cancel | high |
| Audio output | **miniaudio** — already in-tree; new playback `ma_device`, not duplex | high |
| Resampling | In the worker via `ma_resampler`; device pinned at 48 kHz | high |
| UI | Per-utterance `[▶]` in `TranscriptView` — **requires giving utterances stable IDs first** | high |
| Mode gating | On-demand in **both** modes; auto-speak in **Conversational only** | high |

---

## 1. Model selection & ONNX infrastructure

### 1.0 What we already have (audited, not assumed)

| Asset | Where | Reusable for TTS? |
|---|---|---|
| ONNXRuntime 1.28.0, CPU-only, SHA256-pinned zip | `cmake/OnnxRuntime.cmake` | **Yes, verbatim.** `blackwell::onnxruntime` is a pure usage-requirements INTERFACE target. |
| `blackwell_copy_onnxruntime_dlls(target)` | same | **Yes.** Already called on `audio_translator`. |
| Hash-pinned model fetch → gitignored `models/` | same (`file(DOWNLOAD ... EXPECTED_HASH)`) | **Yes**, with one caveat — §1.4. |
| PIMPL'd ORT wrapper, single ORT-visible TU | `src/vad/silero_vad.{hpp,cpp}` | **Yes** — this is the template to clone. |
| Static-CRT (`/MT`) vs. ORT's `/MD` reasoning | `cmake/OnnxRuntime.cmake` header block | **Yes.** The argument (import lib, no CRT state across the seam, `OrtStatus*` not exceptions) transfers unchanged. |
| Steady-state-allocation-free ORT session pattern | `SileroVAD::Impl` | Partially — TTS output is variable-length, so the output tensor must be ORT-allocated, not caller-owned. |

**Net new third-party dependency for the ONNX half: zero.** The runtime, the acquisition
module, the copy helpers and the wrapper idiom are all in place.

### 1.1 Which architecture

**Recommendation: a VITS-family end-to-end model, Piper's export format first.**

The comparison, scored on what this app actually needs (batch = 1, offline, low latency,
CPU, 11 target languages, must not grow the VRAM budget):

| Candidate | Params / size | Input | Latency (CPU, est.) | Languages | Notes |
|---|---|---|---|---|---|
| **Piper (VITS, `.onnx` + `.onnx.json`)** | ~15–30 M, 20–70 MB/voice | **phonemes** (eSpeak IPA → id) | ~10–25× realtime | ~40 incl. ru, uk, de, fr, es, it, pt, zh | Single-file voice + sidecar JSON config = an exact fit for the hash-pinned fetch pattern. Streaming-friendly per sentence. |
| **Silero TTS v4** | ~30–50 M, ~60 MB/lang | **graphemes** (built-in symbol table, internal stress) | ~10–20× realtime | ru, uk, en, de, es, fr, uz, tt | **No phonemizer needed at all** — the single biggest C++ effort saver. License is the blocker (§1.2). |
| **Kokoro-82M** | 82 M, ~310 MB (fp32) | phonemes (misaki/espeak) | ~3–8× realtime | en, ja, zh mainly | Best quality in class, but weakest coverage for our language table and heaviest per-utterance cost. |
| **MMS-TTS / VITS (Meta)** | ~36 M | phonemes (uroman) | ~10× realtime | 1000+ | CC-BY-NC. Coverage is unmatched; quality is uneven. |

Why VITS-family wins here specifically:

1. **One forward pass per sentence.** No autoregressive decode, no vocoder as a separate
   graph, no per-frame loop. This matters because it makes cancellation trivial (§2) and it
   makes the worst-case latency a *function of sentence length only*, not of a sampling loop
   that could stall.
2. **It is a single `.onnx` file plus a small JSON.** That maps 1:1 onto
   `file(DOWNLOAD ... EXPECTED_HASH)` + `blackwell_copy_*` — the exact pattern
   `silero_vad.onnx` already uses.
3. **Sizes that do not embarrass a 75 MB runtime.** A 60 MB voice next to a 5.3 GB AWQ
   backbone is noise.

**Design consequence:** do not hardcode Piper. The two leading candidates differ *only* in
whether the frontend emits phoneme ids or character ids, so the seam belongs exactly there:

```cpp
// src/tts/tts_backend.hpp — the seam that makes the §1.2 decision reversible.
struct TtsVoice {                 // parsed from the sidecar .json
    std::string  id;              // "ru_RU-irina-medium"
    int          sample_rate;     // 22050 | 24000 | 48000
    int          language_index;  // index into rt::kLanguages (append-only contract)
    // ... speaker id, symbol table, phoneme->id map
};

class ITtsBackend {
public:
    virtual ~ITtsBackend() = default;
    // RUNTIME tier: noexcept, returns a status. Appends mono f32 at voice.sample_rate.
    virtual TtsStatus synthesize(const std::vector<int64_t>& input_ids,
                                 std::vector<float>& out_pcm) noexcept = 0;
    virtual const TtsVoice& voice() const noexcept = 0;
};
```

### 1.2 The phonemizer — this is a product decision, and it blocks the model choice

Three options, honestly costed:

**(a) espeak-ng.** The only realistic way to cover all 11 entries in `rt::kLanguages`.
Battle-tested, and what every Piper voice was *trained against* — a different phonemizer
produces subtly wrong pronunciation, not just different pronunciation.
- *Cost:* it is **GPL-3.0**. `piper-phonemize` is MIT but vendors espeak-ng, so it does not
  change the answer. For a closed-source shipped desktop binary (`deploy/installer.iss`
  exists, so this ships) that is a genuine constraint. **I am not qualified to advise on
  whether that is acceptable for this product, and process- or DLL-isolation arguments
  around GPL are contested — this needs a real answer from you, not a workaround from me.**
- *Also:* MSVC + static-CRT build of espeak-ng, plus shipping its `espeak-ng-data` tree
  (~15 MB of dictionaries), is a day or two of build work on its own.

**(b) Custom dictionary + rules.** CMUdict for English is tractable. **Russian is not.**
Russian TTS quality is dominated by *stress placement*, which is lexically irregular and
genuinely ambiguous (за́мок / замо́к), and getting it wrong is immediately audible.
Building this per-language is a multi-week effort per language with a permanently worse
ceiling than (a).
- *Verdict:* do not do this. It is the option that looks cheap and is not.

**(c) Ship a grapheme-input model and have no phonemizer at all.** Silero TTS v4 accepts
text over a built-in character symbol table and does its own stress handling for Russian —
which happens to be the flagship target language of this translator.
- *Cost:* Silero's **TTS** models are, as far as I can determine, distributed under
  CC BY-NC-SA 4.0 — note that this is *different* from `silero_vad.onnx`, which is MIT and
  is why the VAD integration was uncomplicated. **Verify the current terms before
  committing**; a non-commercial license may be as blocking as GPL depending on the product.

**Recommendation.** Build `IPhonemizer` as a seam with a trivial `PassthroughPhonemizer`
(graphemes → symbol-table ids) as the first implementation, and ship V1 against a
grapheme-input voice. That gets a working, demoable feature with **zero** new third-party
build integration. Then add `EspeakPhonemizer` behind the same interface once the licensing
question has an answer — at which point the whole Piper voice catalogue unlocks without
touching the worker, the sink, or the UI.

```cpp
class IPhonemizer {
public:
    virtual ~IPhonemizer() = default;
    // Text (UTF-8) + target language -> model input ids. RUNTIME tier: noexcept.
    virtual TtsStatus to_ids(const std::string& utf8, int language_index,
                            const TtsVoice& voice,
                            std::vector<int64_t>& out_ids) noexcept = 0;
};
```

> **Non-negotiable regardless of choice:** the whole path is UTF-8. `audio_translator`
> already compiles with `/utf-8` and the target is Cyrillic; any phonemizer that assumes
> single-byte characters will silently mangle the primary use case. Grapheme iteration must
> be UTF-8-aware (codepoint-wise), not `char`-wise.

### 1.3 CPU or GPU — CPU, and not marginally

Run the ONNX TTS session on the **CPU execution provider**. Four independent reasons, any
one of which is sufficient:

1. **VRAM.** The budget is already committed: ~5.3 GB AWQ weights + ~1.05 GB FP32 KV pool at
   `kMaxContext = 4096` (translator/main.cpp documents the ~256 KB/token figure) + the
   Whisper encoder's bucketed CUDA graphs. There is no headroom being asked for here that
   has a good answer.
2. **SM contention is worse than it looks.** A GPU TTS forward does not run "alongside" the
   decode — at batch = 1 the decode loop is latency-bound, not throughput-bound, and a
   concurrent TTS graph steals SM time from exactly the thing the product measures.
   Putting TTS on the GPU makes token latency worse *in order to* produce audio for tokens
   already emitted. That is backwards.
3. **Second CUDA runtime in-process.** The ORT CUDA provider brings its own context and
   `onnxruntime_providers_shared.dll`, which `cmake/OnnxRuntime.cmake` explicitly and
   deliberately does not ship.
4. **Download size.** 75 MB (CPU) vs. 434 MB (CUDA 12) — the module already makes this
   argument for the VAD and it is the same argument.

The CPU is genuinely idle here. The DSP worker runs at a ~0.3% duty cycle (measured, per
`silero_vad.hpp`), the engine thread is blocked in `wait_and_pump` at 0% between utterances,
and a desktop Blackwell box has cores to spare. A VITS-small sentence at 22 kHz is ~10–25×
realtime on two cores.

**Session options — pin them explicitly, do not take ORT's defaults:**

```cpp
options.SetIntraOpNumThreads(2);                              // NOT the default (= core count)
options.SetExecutionMode(ORT_SEQUENTIAL);
options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
```

ORT's default intra-op pool is sized to the machine's core count. Left alone, a synthesis
burst spawns a thread per core and will preempt the DSP worker — the one thread in this
process with a genuine 10 ms real-time deadline. Two threads is enough for real-time-factor
headroom of 10×+ and leaves the audio path alone.

### 1.4 Voice acquisition — the one place the VAD pattern does **not** transfer cleanly

`silero_vad.onnx` is 2.3 MB, so fetching it unconditionally at configure time costs nothing.
Voices are **20–100 MB each**, and there are eleven languages in `rt::kLanguages`. Fetching
them all at configure time would add ~1 GB to a fresh `cmake --preset` — an unacceptable
regression to a workflow that is currently fast.

**Proposal:** new `cmake/TtsVoices.cmake`, same idioms (`EXPECTED_HASH`, `TLS_VERIFY ON`,
into gitignored `models/`, copy helper always defined), but:

```cmake
# ONE default voice at configure time; everything else opt-in.
set(BLACKWELL_TTS_VOICES "ru_RU-irina-medium" CACHE STRING
    "Semicolon-separated TTS voices to fetch. Each adds 20-100 MB to configure time.")
```

Voices not fetched simply do not appear in the UI's voice list — the same graceful-degradation
posture `blackwell_vad` and `load_audio_head` already take (missing asset → feature off +
a panel that says why → app still runs). Never fatal.

---

## 2. Threading & concurrency

### 2.1 The current thread inventory (audited from the source, not from the comments)

```
miniaudio callback  ──push──▶ SampleRing ──pop──▶ RealTimeDSP worker ──┬──▶ SpectrogramBuffer
(WASAPI, RT prio)              (mutex+deque)      (DSP + VAD producer) ├──▶ AudioRecorder
                                                                       └──▶ ISpeechMode::on_pcm_block
                                                                            └─▶ SPSC command ring
                                                                                      │
Engine thread ──── ISpeechMode::pump_engine() ◀────────────────────────────────────────┘
              (SOLE toucher of CUDA / KV / engine; parks at 0% CPU in wait_and_pump)
                    └──▶ SpeechTokenCallback ──▶ TranscriptView (short mutex)

UI thread ──── Direct2D + ImGui ──▶ TranscriptView::draw() / ControlPanel::draw()
               (atomics + lock-free speech_pipeline_* only; never the engine)
```

### 2.2 Where TTS runs

**A fourth thread: a dedicated `TtsWorker`, entirely outside the doctrine.**

This is the key architectural claim and it is worth stating precisely: **the doctrine is not
in play here.** "One thread touches the engine" constrains callers of `BlackwellEngine`,
`VRAMArena`, `kv_mgr` and CUDA. `TtsWorker` calls none of them — it is an ORT-CPU leaf, the
same class of citizen as `SileroVAD`. So it does not need to marshal onto the engine thread,
and it must not be given a reason to.

The invariant that replaces the doctrine for this subsystem:

> **`TtsWorker` must never call `IEngineControl`/`EngineControlBridge`, `speech_pipeline_*`,
> the CUDA runtime, ImGui, or a window.** Its only outputs are (a) PCM into a lock-free sink
> and (b) atomics the UI polls. If a future feature needs TTS to influence generation, it
> marshals through the existing `PostEngineTask`-style seam — it does not reach across.

Why not reuse an existing thread:

- **Not the engine thread.** Synthesis is 50–200 ms of CPU. Parked in `pump_engine()`, that
  thread's entire value is that it wakes instantly on a boundary event. Adding a
  hundred-millisecond CPU task to it directly violates the stated requirement ("must not
  block the LLM text generation loop") and would delay barge-in response.
- **Not the DSP worker.** It has a hard 10 ms budget and is the single producer for both
  rings. This is the thread most damaged by a latency spike, and it is the one where the
  requirement "must not interfere with the live audio capture thread" bites.
- **Not the UI thread.** A 200 ms synthesis is 12 dropped frames.
- **Not the miniaudio playback callback.** It must do nothing but memcpy from a ring.

### 2.3 The request queue and cancellation

```cpp
// src/tts/tts_worker.hpp
struct TtsRequest {
    std::uint64_t utterance_id;   // stable id from TranscriptView (see §4.1)
    std::uint64_t epoch;          // speak_epoch at enqueue time
    std::string   text;           // UTF-8, already stripped to the spoken half
    int           language_index; // rt::kLanguages index -> voice selection
};
```

- **Queue:** bounded `std::deque<TtsRequest>` + `std::mutex` + `std::condition_variable`.
  Deliberately **not** the lock-free SPSC ring used for engine commands: producers are
  plural (UI thread today, engine thread if auto-speak lands), the enqueue rate is
  single-digit per minute, and the latency budget is ~100 ms. A mutex is free here and a
  condvar gives the 0%-CPU park that matches how `pump_engine` already behaves.
- **Cancellation:** a monotone `std::atomic<uint64_t> speak_epoch_`, mirroring the barge-in
  epoch in `EngineControlBridge`. "Stop" bumps it; the worker drops any dequeued request
  whose `epoch` is stale, and the sink discards buffered PCM tagged with a stale epoch.
  Latest-wins, no cancellation tokens, no joins. **Deliberately mirroring the existing
  epoch idiom** so it reads as native to anyone who knows the bridge.
- **Granularity:** VITS is one forward per *sentence*, so cancel latency is bounded by one
  sentence's synthesis (~50–150 ms), not by the whole utterance. Good enough; no need for
  in-graph interruption.

### 2.4 Chunk on sentence boundaries — this is a latency feature, not a nicety

A VITS forward returns the whole waveform at once. If the worker synthesizes a
four-sentence translation as one graph call, **time-to-first-audio scales with the whole
utterance**. Splitting on sentence boundaries and pushing each sentence's PCM to the sink as
it completes makes time-to-first-audio a function of the *first sentence only* — typically
120–250 ms — while the rest synthesizes in the shadow of playback. Since real-time factor is
10×+, the sink never starves after the first chunk.

---

## 3. Audio playback pipeline

### 3.1 Current library: miniaudio — and yes, reuse it

**Audited answer: `miniaudio` (mackron), not PortAudio or SDL.**

- Declared in `cmake/Dependencies.cmake`, bundled into the `blackwell::sandbox_headers`
  INTERFACE target (SYSTEM includes, so it is outside the `/W4 /WX` budget).
- Compiled with `MINIAUDIO_IMPLEMENTATION` in **exactly one TU**:
  `audio_sandbox/src/audio_capture.cpp`.
- Capture is configured `ma_format_f32` / 1 channel / 16000 Hz, and **miniaudio does the
  device resampling and downmix internally** — `AudioCapture` never resamples by hand.

It supports playback natively (`ma_device_type_playback`) and ships a resampler
(`ma_resampler`) and format converter (`ma_data_converter`) in the same header. **Net new
third-party dependency for the audio-output half: zero.**

Two build facts that will bite if missed:

1. **Do not define `MINIAUDIO_IMPLEMENTATION` again.** The new `audio_playback.cpp` includes
   plain `"miniaudio.h"`; `audio_capture.cpp` stays the sole implementation TU. A second
   definition is a wall of duplicate-symbol link errors.
2. **Set `NOMINMAX` and `WIN32_LEAN_AND_MEAN` before the include**, as `audio_capture.cpp`
   does — miniaudio pulls in `<windows.h>` for WASAPI and the `min`/`max` macros will break
   `std::min`/`std::max` in the same TU.

**One pin issue, flagged in passing:** miniaudio is fetched at `GIT_TAG master`. Every other
fetch in this repo is pinned by tag + hash (`gtest`, `imgui v1.91.0`, ORT by SHA256, Silero
by tag + SHA256). Adding a second consumer of miniaudio is a reasonable moment to pin it to
a release tag — an upstream `master` break would now take out the output path as well as the
input path.

### 3.2 Playback device: separate, not duplex

Open a **second `ma_device` of type `ma_device_type_playback`**. Rejected alternative:
`ma_device_type_duplex`.

- Duplex would force a rewrite of `AudioCapture`, which is shared with `audio_realtime` and
  is currently untouched-by-design ("so `audio_realtime`'s own target stays byte-for-byte
  untouched" — `audio_sandbox/CMakeLists.txt`).
- Duplex is **incompatible with `ma_device_type_loopback`**, which is half of the app's
  capture story (`[L] System loopback`).
- The one real argument *for* duplex is that acoustic echo cancellation needs a playback
  reference sample-aligned to the capture clock (§3.4). Since V1 does not do AEC, that
  argument is not yet live — but note it here so a future AEC effort knows the separate-device
  choice is the thing it must revisit.

### 3.3 Sample-rate conversion: fixed 48 kHz device, resample in the worker

TTS models emit 22050 Hz or 24000 Hz. The capture path is 16000 Hz and irrelevant to output.

**Rejected:** configure the playback device at the model's rate and let miniaudio convert.
Simple, but it welds the device format to the voice — swapping a 22050 Hz voice for a
24000 Hz one requires `ma_device_uninit`/`init`, i.e. an audible gap and a teardown race
with the callback.

> **SUPERSEDED for the duplex design.** Once AEC entered scope, capture and playback became
> ONE device with one rate, and the choice was re-decided at **16 kHz** — see the phased plan:
> it keeps AEC at its native rate and leaves `WhisperDSP`'s golden-parity geometry untouched,
> at the cost of band-limiting the voice to 8 kHz. The reasoning below stands for the
> separate-device configuration that Phase 2 ships and that loopback mode keeps permanently.

**Recommended:** pin the playback device at **48000 Hz** and resample in `TtsWorker` with a
`ma_resampler`:

```
ORT out (mono f32 @ 22050)  ──ma_resampler──▶  mono f32 @ 48000  ──▶  PcmSink ──▶ ma callback
        [TtsWorker thread]                        [TtsWorker]         (lock-free)  [RT thread]
```

Why 48000: it is the WASAPI shared-mode mix rate on effectively every Windows machine, so
this also avoids a *second*, hidden resample inside the OS mixer. Why in the worker: the SRC
cost lands on the thread that already has a 10× latency budget, never on the callback; the
ring geometry is then fixed for all voices; and a voice swap reconfigures one `ma_resampler`
while the device keeps running.

Channels: TTS is mono. Configure the playback device with `channels = 1` and let miniaudio
upmix to the device layout — exactly the delegation `AudioCapture` already relies on for
downmix. Do not hand-write a mono→stereo splat.

### 3.4 `PcmSink` — and why *not* to copy `SampleRing`

`SampleRing::push()` does `q_.insert(...)` on a `std::deque` **from the audio callback
thread**, then `erase()`s from the front on overflow. On the *capture* side this is
defensible: an allocation-induced overrun costs 10 ms of mel that nobody sees.

On the *playback* callback the same stall is an **audible click**, and clicks are the entire
perceived quality of a TTS feature. The output path should get its own primitive:

- Fixed, preallocated ring (≥ 500 ms at 48 kHz ≈ 24000 floats).
- SPSC: `TtsWorker` writes, the miniaudio callback reads. `std::atomic` head/tail with
  acquire/release. **No allocation, no lock, no `erase` in the callback.**
- Underrun → write silence and bump an `underruns_` counter (surfaced in the panel, like
  `SileroVAD::inference_errors()`).
- Each buffered span carries its `epoch`; a stale span is skipped rather than played, which
  is how "Stop" becomes instant rather than draining.

**Landed as `blackwell::audio_rt::SpscRing<T>`** ([src/audio_rt/spsc_ring.hpp](../src/audio_rt/spsc_ring.hpp)).
One correction against the sketch above: fault accounting is **paired**, not automatic.
`write`/`read` move data and judge nothing; `write_or_drop`/`read_or_silence` are the
variants that declare a short transfer a fault and count it. The reason is that both
producers are real — a real-time capture callback cannot wait and must drop, while the TTS
worker runs ~10× realtime and must retry rather than discard audio mid-sentence. A counter
that fired on every retry would report hundreds of thousands of "drops" for a stream that
lost nothing, which is exactly what the concurrency test surfaced.

### 3.5 The feedback loop — the part that reaches back into existing code

Two concrete failure modes, both real today:

| Capture mode | What happens with TTS playing |
|---|---|
| `Loopback` | Playback **is** the captured signal. Guaranteed feedback: the bot hears itself, transcribes itself, translates itself, and speaks that. Unbounded. |
| `Microphone` + speakers | The mic hears the TTS. Silero scores it as speech — **because it is speech** — which fires `on_speech_start`, which bumps the barge-in epoch, which cancels the generation currently being spoken. |

Mitigations, cheapest first:

1. **Mic gating during playback (recommended for V1).** `TtsWorker` publishes
   `std::atomic<bool> speaking_` plus a release deadline; the DSP worker checks it and drops
   blocks (or feeds silence) while set, with a ~150 ms tail for room reverb. This lands on
   an existing seam — the pipeline already supports muting for push-to-talk, so this is a
   second reason to mute, not a new mechanism.
   *Honest cost:* **barge-in-while-speaking is disabled in V1.** The user cannot interrupt
   the bot's voice by talking over it; they use Stop, or push-to-talk. That is a real product
   regression against the Conversational mode's design intent and you should decide
   consciously whether to accept it.
2. **Headphones.** Documented workaround; removes the problem entirely and is what any live
   demo should use.
3. **Hard-block auto-speak in `Loopback` mode.** Not a mitigation, a safety interlock. In
   loopback there is no gating that works, because the gate would have to suppress the very
   signal the app exists to transcribe.
4. **Real AEC (out of scope).** WebRTC APM or Windows' processed capture endpoint. Needs the
   sample-aligned reference signal that §3.2 explicitly gave up. Revisit only if
   barge-in-while-speaking becomes a requirement.

---

## 4. Application state & UI integration

### 4.1 Blocking gap: `TranscriptView` utterances have no identity

`TranscriptView` stores `std::vector<std::string> history_`, and `flush_live_locked()` caps
it at 200 with `history_.erase(history_.begin())`. **Indices shift.** A `[▶]` button keyed
on vector index will, after the 201st utterance, speak a different line than the one the
user clicked — a bug that only appears in long sessions, i.e. exactly the ones being demoed.

This must be fixed before the button exists:

```cpp
struct Utterance {
    std::uint64_t id;    // monotone, never reused; the TTS request key
    std::uint64_t gen;   // the generation that produced it (barge-in correlation)
    std::string   text;  // "[Speech] ... | [Translation] ..."
    bool          interrupted;  // flushed by a gen bump rather than on_final
};
std::vector<Utterance> history_;
```

The change is contained: `flush_live_locked()` assigns `next_id_++`, `draw()` snapshots
`Utterance`s instead of strings, and `draw_utterance()` gains an `ImGui::PushID(id)` +
`SmallButton`. Nothing outside the class sees it.

### 4.2 Speak the translation, not the transcript

`draw_utterance()` already splits on `kDelimiter = " | "` into a source line and a
translation line. TTS should vocalize **the translation half only** — that is the product.
Consequences:

- **Voice language comes from the target**, `control_->target_language_index()`, not the
  source and not a separate TTS language setting. One source of truth.
- **`"Auto"` (index 0) has no voice.** The Speak button must be disabled with a tooltip
  explaining why, not silently no-op. Same for a target language with no installed voice
  (§1.4) — the panel says which voices are present.
- The index→voice map must honor the `kLanguages` **append-only** contract (the header is
  explicit: indices cross threads in atomics, never reorder).
- Utterances with no delimiter (transcribe-only task) fall back to speaking the whole line in
  the *source* language.

### 4.3 UI surface

**In `TranscriptView` (per-utterance, the "on-demand" requirement):**
- A `[▶]` `SmallButton` on each *finalized* history entry. **Not on the live line** — it is
  still growing, and in Mode B it is being redrafted.
- The currently-speaking entry gets a highlight and its button becomes `[■]` (stop).

**In `ControlPanel` (a new `TTS & Voice` collapsing header, matching the existing sections):**
- `Enable TTS` master toggle.
- Voice dropdown (installed voices only) + speed/length-scale slider + volume.
- `Auto-speak finalized utterances` — **default OFF**, which is what "on-demand" means. Gated
  per §4.4.
- Status readout: `Idle / Queued(n) / Synthesizing / Speaking`, plus `underruns` and
  `synthesis errors` counters — mirroring how the VAD section surfaces
  `inference_errors()`.
- Disabled-with-reason when TTS failed to initialize, exactly as the panel already does for
  a missing neural VAD.

**Threading compliance:** all of the above is UI-thread-only, touching a `TtsWorker` whose
entire cross-thread surface is atomics and one mutex-guarded enqueue — the same posture
`ControlPanel` already maintains ("never touches the engine, CUDA, or the KV cache").

### 4.4 Interaction with `ISpeechMode` — the sharp question

**Do not add `speak()` to `ISpeechMode`.** That interface is deliberately narrow: "a stream
of 10 ms PCM blocks in, text out, one engine thread." TTS is downstream of *text out* and
touches no engine state, so putting it on the mode interface would make every mode
implementation carry a concern none of them own — and would drag ORT into
`conversational_mode.hpp`, the exact mistake the header's own comments call out about
naming `SileroVAD` there.

**Do add a policy query,** because the answer genuinely differs per mode:

```cpp
// speech_mode.hpp
struct TtsPolicy {
    bool allow_on_demand = true;    // the user clicked [>] on a settled line
    bool allow_auto_speak = false;  // the app speaks every finalization unprompted
};
virtual TtsPolicy tts_policy() const noexcept = 0;
```

| Mode | `allow_on_demand` | `allow_auto_speak` | Why |
|---|---|---|---|
| **A — Conversational** | ✔ | ✔ | A finalization is genuinely final. This is the mode TTS was designed for: listen → silence → generate → *speak*. |
| **B — Simultaneous** | ✔ | ✘ | Drafts churn continuously by design ("its draft is disposable BY DESIGN"). Auto-speaking a redrafted line means speaking a sentence that is being retracted while it is still audible. |

So the answer to *"should it only be active in Conversational mode?"* is **no, not quite** —
and the distinction matters:

- **Auto-speak: Conversational only.** Correct as stated.
- **On-demand: both modes**, gated on the commit pointer. `docs/CONTINUOUS_STREAMING.md`'s
  commit pointer `C` is precisely the guarantee that text before it will never be redrafted.
  Text behind the commit pointer in Mode B is exactly as final as a Mode A finalization, so
  refusing to speak it would be throwing away a guarantee the design already provides.
  A `[▶]` on a committed Mode B segment is well-defined; on an uncommitted draft it is not,
  and the button simply is not drawn there.

Mode switches must bump `speak_epoch` and flush the sink — a mode flip is a session
boundary (per `speech_mode.hpp`, it re-prefills a different frozen prefix), so in-flight
audio from the previous session is stale by construction.

### 4.5 State machine

```
Idle ──enqueue──▶ Queued ──worker picks up──▶ Synthesizing ──first chunk──▶ Speaking
  ▲                  │                             │                            │
  └──────────────────┴──── stale epoch / Stop ─────┴──── sink drained ──────────┘
                                    │
                                 Failed ──▶ Idle   (counter++, panel shows it; never fatal)
```

Published as one `std::atomic<TtsState>` for the badge — the same treatment
`TranscriptView` gives `SpeechPipelineState`.

---

## 5. Proposed structure & build integration

### 5.1 New code

| Path | Target | Contents |
|---|---|---|
| `src/tts/tts_engine.{hpp,cpp}` | `blackwell_tts` (STATIC) | PIMPL'd ORT session; **the only TU that sees `<onnxruntime_cxx_api.h>`** besides the VAD's. Structural clone of `src/vad/`. |
| `src/tts/tts_backend.hpp` | " | `ITtsBackend`, `TtsVoice`, `TtsStatus`. |
| `src/tts/phonemizer.hpp` | " | `IPhonemizer` + `PassthroughPhonemizer`. |
| `src/tts/tts_worker.{hpp,cpp}` | " | The worker thread, queue, epoch, sentence splitter, `ma_resampler` use. |
| `src/tts/pcm_sink.hpp` | " | Lock-free SPSC f32 ring (§3.4). Header-only; no ORT, no miniaudio. |
| `audio_sandbox/src/audio_playback.{h,cpp}` | (compiled into `audio_translator`) | `ma_device_type_playback` wrapper. Plain `"miniaudio.h"` — no `MINIAUDIO_IMPLEMENTATION`. |
| `cmake/TtsVoices.cmake` | — | Hash-pinned voice fetch + `blackwell_copy_tts_voices()`. |
| `tests/tts/` | `tts_tests` | CPU-only, `LABELS validation`, registered exactly like `vad_tests`. |

### 5.2 CMake shape (per `cmake-hygiene`)

```cmake
# src/tts/CMakeLists.txt — mirrors src/vad/ exactly.
if(NOT TARGET blackwell_onnxruntime)
    return()                                  # -DUSE_SILERO_VAD=OFF -> no target at all
endif()
add_library(blackwell_tts STATIC tts_engine.cpp tts_worker.cpp)   # explicit list, never GLOB
target_include_directories(blackwell_tts PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(blackwell_tts PRIVATE blackwell::onnxruntime  # PIMPL: stays PRIVATE
                                            blackwell::sandbox_headers)  # ma_resampler
```

- `audio_translator` gains `if(TARGET blackwell_tts)` + `blackwell_copy_tts_voices(...)`.
  With TTS absent the app builds and runs with the panel saying so — the established posture.
- **Option name:** the ORT option is currently `USE_SILERO_VAD`, which will read wrong once a
  second ORT consumer exists. Suggest introducing `USE_ONNXRUNTIME` as the acquisition gate
  with `USE_SILERO_VAD` / `USE_ONNX_TTS` as feature gates on top. Cosmetic, but the current
  name becomes actively misleading the day this lands.
- `/W4 /WX` applies to all new first-party code. ORT and miniaudio headers are already SYSTEM
  (`/external:W0`); the `/utf-8` flag `audio_translator` already sets is **required** for the
  Cyrillic path and must be set on `blackwell_tts` too.

### 5.3 Error doctrine (extension pattern #4)

| Phase | Surface | Behavior |
|---|---|---|
| **INIT** | `TtsEngine` ctor, voice load, `ma_device_init`, worker start | Throws `std::runtime_error`. Caller catches → TTS disabled, panel explains, **app runs**. This is exactly the `neural_vad` and `load_audio_head` precedent in `translator/main.cpp`. |
| **RUNTIME** | `synthesize()`, `to_ids()`, resample, sink write, playback callback | `noexcept`, `TtsStatus` return, monotone error counters. An ORT hiccup mid-stream degrades audio; it never unwinds the audio thread. Mirrors `SileroVAD`'s split verbatim. |

### 5.4 Testing

CPU-only, no GPU, so it runs in the fast `validation` suite next to `vad_tests`:

1. `PcmSink` — wrap-around, overrun, underrun accounting, SPSC ordering. Pure unit test, no
   model needed; this is the piece where a bug is an audible artifact, so it deserves the
   most coverage.
2. Resampler — 22050 → 48000 length ratio within tolerance; no NaN/clip.
3. Sentence splitter — UTF-8/Cyrillic safety, abbreviations, no empty chunks.
4. Synthesis smoke (skipped when the voice is absent, as the VAD test does for its model) —
   fixed string → non-silent PCM, duration within plausible bounds, bit-identical across two
   runs (VITS is stochastic unless the noise scale is fixed, so **pin the seed / noise
   parameters** or assert on energy envelope rather than samples).
5. Epoch/cancel — enqueue N, bump epoch, assert zero stale PCM reaches the sink.

---

## 6. Phased plan

Revised once AEC replaced the hard interlock as the barge-in strategy. Each phase leaves a
shippable app; **the interlock stays as a safety net through Phase 4 and is removed in
Phase 5**, so no build ever ships that can self-trigger.

| Phase | Deliverable | Status |
|---|---|---|
| **1 — Foundation** | `IPhonemizer` seam + passthrough frontend; `SpscRing<T>`; `TranscriptModel` stable IDs; `blackwell_tts` + `tts_tests`. | **done** |
| **2 — Output path** | `PcmSink` + playback `ma_device` + test tone + xrun readout; pin miniaudio. | not started |
| **3 — Duplex refactor** | New `audio_device` (leave `audio_capture` alone for `audio_realtime`); 16 kHz duplex; fixed-block adapter; `SpscRing` replaces `SampleRing`; delay calibration. Loopback stays non-duplex, permanently interlocked. | not started |
| **4 — AEC** | `IEchoCanceller` seam; SpeexDSP first, AEC3 as the escalation; **dual tap** (aggressive → VAD, linear-only → Whisper); ERLE metering. | not started |
| **5 — `TtsWorker`** | Worker lifecycle, `speak_epoch`, sentence chunking, ORT `intra_op_num_threads = 2`; **interlock removed, barge-in enabled**. | not started |
| **6 — Mode policy & UI** | `TtsPolicy` on `ISpeechMode`; per-utterance `[▶]`; voice/auto-speak panel. | not started |

**Still blocking:** the licensing question in §1.2. Phase 1 was deliberately built so that it
does not depend on the answer — no GPL code, and no ORT dependency at all — but the model
family cannot be chosen until it is resolved.

### Phase 1 — what landed

| File | Role |
|---|---|
| [src/audio_rt/spsc_ring.hpp](../src/audio_rt/spsc_ring.hpp) | The lock-free primitive (§3.4). Own target `blackwell::audio_rt`, dependency-free, because its consumers are not all TTS — the capture path and the AEC reference tap use it too. |
| [src/tts/tts_status.hpp](../src/tts/tts_status.hpp) | RUNTIME-tier status codes; the INIT/RUNTIME split spelled out for this subsystem. |
| [src/tts/tts_voice.hpp](../src/tts/tts_voice.hpp) | Voice descriptor: symbol table, pad/BOS/EOS decoration, output geometry. |
| [src/tts/phonemizer.hpp](../src/tts/phonemizer.hpp) / [.cpp](../src/tts/phonemizer.cpp) | `IPhonemizer` + `PassthroughPhonemizer` (graphemes). The licensing firewall. |
| [audio_sandbox/translator/transcript_model.hpp](../audio_sandbox/translator/transcript_model.hpp) | Utterance storage with monotone IDs, split out of `transcript_view.hpp` (§4.1). |
| [tests/tts/](../tests/tts) | 47 tests, `LABELS validation`. CPU-only: no CUDA, no ORT, no model, no audio device, no GUI. |

---

## 7. The biggest technical hurdle

**The acoustic feedback loop between playback and the live VAD — §3.5.**

Everything else in this document is *additive*. A new leaf static lib, a second ORT session,
a second miniaudio device, a fourth thread that touches no engine state, a new ImGui section:
none of it perturbs a single existing invariant, and if any of it fails the app degrades to
exactly what it is today. That is why the ONNX half of this — the part the request framed as
the interesting question — is genuinely low-risk. The infrastructure is already built and
already proven by the VAD.

The feedback problem is different in kind, because it is the one part that **reaches
backwards into the code the doctrine governs**. The moment the app produces sound, the
microphone hears it, Silero correctly scores it as speech, the segmenter calls it an onset,
and the barge-in epoch cancels the generation whose output is currently being spoken. The
system's own correctness works against it: the VAD is not wrong, the barge-in is not wrong —
they are being fed a signal the architecture never anticipated. And the fix is not local. It
lives in the DSP worker's block loop, in the mode's PCM tap, and in the interlock between
capture mode and the auto-speak policy — three places, on two threads, in code whose whole
value is that it is currently simple and correct.

The V1 answer (gate the mic while speaking, plus a hard interlock in `Loopback`) is cheap and
lands on the existing push-to-talk muting seam, but it is not free: **it trades away
barge-in-while-speaking**, which is a headline behavior of Conversational mode. Buying that
back means real acoustic echo cancellation, which needs a playback reference sample-aligned
to the capture clock — which in turn means revisiting the separate-device decision in §3.2 in
favor of duplex, and duplex is incompatible with the loopback capture path. That chain is the
actual risk in this feature, and it is worth deciding deliberately now rather than
discovering it at T5.

**Runner-up: the phonemizer (§1.2).** Higher raw effort, and possibly weeks of it — but the
effort is *contained* behind one interface, and the decision is a licensing question with a
clear technical escape hatch (ship a grapheme-input model). It is expensive, not dangerous.
