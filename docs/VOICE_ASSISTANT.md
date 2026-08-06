# `voice_assistant` — technical review

**What it is.** A standalone Windows desktop voice assistant: a Chromium-rendered
messenger window on top of the Local Router pipeline. You speak, it hears you, a local
8B model drafts a reply, and a strict gate decides whether that reply is good enough to
stand on its own or whether the question should be escalated to a paid cloud model.
Optionally it speaks the answer back, full-duplex, with acoustic echo cancellation so you
can interrupt it mid-sentence.

**Scope of this document.** An architectural review of `audio_sandbox/voice_assistant/`
(11 files, ~6 600 lines C++ plus a ~3 100-line web UI) and how it composes subsystems
owned elsewhere. The routing/commit semantics are specified in
[`docs/LOCAL_ROUTER.md`](LOCAL_ROUTER.md) and are summarised, not restated, here. §10 is
an assessment with named risks, not a summary.

---

## 1. The loop

```
listen ──▶ pause ──▶ local generate ──┬─▶ [barge-in]  abort, dispatch nothing
                                      ├─▶ [TokenCap]  truncated, dispatch nothing
                                      └─▶ [EOS]       commit ──▶ transport ──▶ answer
```

The single load-bearing idea: **a local generation is dispatchable iff it terminated on
EOS** — the model emitted its own stop token. Everything else (a user interruption, a
truncation at the token ceiling, a decode fault) produces silence rather than a billed
request. The rule lives in exactly one function, `IntentCommitQueue::offer()`, whose first
parameter is the termination reason, so a caller cannot enqueue without stating why the
loop stopped.

The failure asymmetry is deliberate and worth restating because it is the app's whole
safety posture: **every "I got this wrong" path costs a missed answer, not money.** A
forgotten `publish_intent()` yields silence; `EngineControlBridge::is_eos()` returns
`false` in the base tier, so a bridge whose override was forgotten fails closed.

**Text and voice are the same path.** A typed message goes through
`submit_text` → the same SPSC command ring → the same decode loop → the same
`IntentCommitQueue` under the same EOS rule. There is no second route to a billed call —
which is the property that makes the rule auditable at all.

---

## 2. Build shape and the optional-dependency matrix

One target, `voice_assistant`, in `audio_sandbox/CMakeLists.txt` (gated on `WIN32` +
`blackwell_bridge` + `blackwell_core_obj`). It compiles `main.cpp`, `assistant_window.cpp`,
and recompiles three sandbox sources (`audio_capture`, `audio_recorder`, `realtime_dsp`)
into itself so the other sandbox targets stay untouched.

Notably it links **neither ImGui nor the D3D11/D2D stack** and does not compile
`window_d2d.cpp` — nothing here draws a spectrogram. That is a real departure from
`audio_translator`, and it is what let the main screen become an HTML messenger.

Almost every capability is additive and optional. Each is absent-by-construction rather
than disabled by a flag at runtime:

| Capability | Gate | Absent ⇒ |
|---|---|---|
| Chromium UI | `blackwell::webview2` | **hard `FATAL_ERROR`** — the only mandatory one |
| Neural VAD (Silero) | `USE_SILERO_VAD` → `blackwell_vad` | falls back to the built-in RMS threshold detector |
| Cascade ASR (Mode C) | `USE_WHISPER_CPP` → `blackwell::whisper_cpp` | `whisper_asr.cpp` not compiled; `pipeline_mode="whisper_cascade"` refused at startup with a message; legacy path runs |
| Speech output | `BLACKWELL_ORT_GPU` → `blackwell_tts_f5` | text-only; every use site compiles out |
| Echo cancellation | `USE_WEBRTC_AEC3` → `blackwell::audio_aec3` | PIMPL'd seam remains, AEC3 backend absent |
| Cloud leg | `BUILD_CLOUD_CLIENT` → `blackwell_cloud` | `OfflineTransport` stand-in only |

Two of these gates carry non-obvious reasoning that is worth preserving:

- **TTS is gated on the CUDA execution provider, not merely on ONNXRuntime.** The F5 DiT
  measures ~0.2× realtime on the CPU provider — audio produced slower than it plays, which
  is worse than no speech at all. The header also warns against gating on
  `BLACKWELL_HAVE_ORT_CUDA`, which belongs to `blackwell::onnxruntime` and is linked
  PRIVATE by `blackwell_tts_f5`, so it never reaches `main.cpp` and would silently
  vanish the whole integration.
- **AEC is gated with the TTS**, because the loudspeaker is what creates the problem AEC
  solves: no speech output ⇒ no echo, and no far-end reference to cancel one with.

---

## 3. Threads

The header block claims five; with everything armed the process actually runs **seven or
eight**. Both numbers are correct — five is the base pipeline.

| Thread | Owns | May never touch |
|---|---|---|
| **UI** | window message loop, WebView2 (STA) | anything but the window |
| **DSP worker** | mic → log-mel → spectrogram; the PCM tap into the active mode | engine, CUDA, window |
| **ENGINE** | the *only* caller into the control; decode loop; offers turns to the gate | the ring's write end |
| **DISPATCHER** | blocks on the gate, then blocks on the transport | the engine |
| **STATS** | 1 Hz poll of gate counters → UI queue | everything else |
| **ASR worker** *(Mode C)* | owns the `WhisperAsr`; one utterance at a time | engine (marshals instead) |
| **TTS worker** *(TTS builds)* | F5 synthesis → playback rings | — |
| WASAPI loopback | far-end reference capture for AEC | — |

The single-engine-thread doctrine holds trivially: exactly one thread calls into the
control, in every mode and on both backends.

**Two marshalling decisions are the interesting part.**

`WhisperCascadeMode` could publish its transcript straight from the ASR worker. It does
not, and the reasoning is exactly right: `IntentCommitQueue` is single-producer, and its
producer is *defined* to be the engine-owning thread. A second producer on a wait-free
SPSC ring "does not fail loudly, it corrupts quietly under contention that only shows up
when the user talks fast." So the publish is marshaled through `post_engine_task` — one
deque push.

`LocalEngineTransport::send()` runs on the *dispatcher* thread but generation is CUDA work
belonging to the engine thread, so it marshals and blocks — which is precisely what the
dispatcher thread exists for ("somewhere safe to block"). The stated consequence is
honest: `on_text` then fires on the **engine** thread rather than the dispatcher thread,
which is safe only because the dispatcher is parked inside `send()` for the whole
generation.

**Shutdown order is load-bearing and correctly documented:**

```
realtime.stop() → capture.stop() → running=false → active->stop()
  → engine_thread.join() → stats_thread.join() → dispatcher.stop()
```

The dispatcher stops *after* the engine joins, "so no commit can arrive post-join." The
mirror hazard is handled in `local_transport.hpp`: because the engine thread joins first, a
task queued at the wrong moment would never run and `send()` would block forever — so the
wait is bounded, re-checks an abandon flag, and the shared state is a `shared_ptr` that
outlives both threads.

---

## 4. Two backends, one pipeline

| Configuration | Backend |
|---|---|
| model directory set | `RealEngineControl` — CUDA decode, real checkpoint, audio head armed, genuine EOS from the tokenizer's stop set |
| no model dir / `--simulated` | `SimulatedEngineControl` — no GPU, no checkpoint, canned replies |

Both derive from `EngineControlBridge`, so everything above holds **one base pointer and
never branches**. The SPSC ring, barge-in epoch, commit gate and dispatcher are identical
on both paths.

This is the strongest single design decision in the app. The CMake comment states the
alternative that was rejected: a link-time split "would have meant two binaries and two
code paths to keep honest." Instead the GPU-free configuration is preserved *at runtime*,
which makes the offline path a genuine rehearsal of the GPU path rather than a parallel
implementation that drifts.

---

## 5. Pipeline modes

`ISpeechMode` (`translator/speech_mode.hpp`) defines three modes. **This app wires two.**

| Mode | Shape | Wired here |
|---|---|---|
| **A — Conversational** (`ultravox_legacy`) | listen → silence → generate → barge-in; audio soft-tokens spliced into the KV cache; the model writes the transcript itself | **yes** (default) |
| **B — Simultaneous** | continuous rolling re-translation, no bot voice | no — `audio_translator` only |
| **C — Whisper Cascade** (`whisper_cascade`) | VAD-bounded utterance → whisper.cpp (GGML) → UTF-8 → the backbone only ever sees text | **yes** |

Selected by the `pipeline_mode` setting (Restart tier), validated at load with a fallback
to `ultravox_legacy` on an unrecognised value.

**The A/C distinction in one sentence: in Mode C no audio ever reaches the backbone.** The
consequence is concrete — cascade mode does not load the Ultravox audio head at all
(`load_audio_head=false`), and that is where the VRAM for the ~1.6 GB GGML model comes
from. It is a genuine trade, not a strict upgrade: Mode C spends a separate ASR model and
a third CUDA context to buy a real text transcript and a backbone that never needs a
multimodal checkpoint.

---

## 6. Routing and the transport seam

`RoutedTransport` implements `IIntentTransport` over two legs (local, remote) and flips
between them with one atomic. Nothing above it branches; the dispatcher does not know a
swap happened. "Use the local model for responses" is therefore a *transport swap*, and —
the part that matters — the local answer is gated by exactly the same arithmetic the
billed one is.

`RoutedTransport::last_send_was_local()` deserves specific mention as a well-reasoned
piece of API design. It reports which leg *ran the finished intent*, not which leg the
toggle points at now. The header names the exact bug this prevents: persistence keys off
it, and asking the live `use_local()` in `on_complete` could write a local conversation to
disk — which the local leg promises never happens — or silently drop a paid cloud turn
from the log. Latched at the top of `send()`, read after it returns, ordered by
construction because the dispatcher holds exactly one intent in flight.

**Typed input reaches this seam too — as of 2026-08-06.** It previously did not: a typed
message went through `submit_text()`, which ran a *local decode of an answer* and offered
that answer to the gate as the intent. So with the cloud leg selected, the GPU generated a
reply nobody displayed and the remote model was asked to respond to it — the toggle was
ignored for anything typed. `submit_typed_turn` now offers the user's text as an intent,
exactly as the cascade offers a transcript, and everything above (the leg latch, the
session latch, persistence) applies unchanged because none of it was ever leg-specific.
The fix is a deletion, which is the shape a good seam produces.

**Interruption is now a first-class gesture rather than a hotkey.** `cancel_in_flight`
stops the local decode (epoch bump), the in-flight transfer (`RoutedTransport::abort()` →
a non-latching cancel epoch in `CurlStreamCore`) and the speaker, unconditionally on all
three legs. `abort()` is deliberately distinct from `shutdown()`: the latter latches, and
a Stop must leave the next turn sendable. See docs/LOCAL_ROUTER.md, "Stopping a
generation".

---

## 7. Speech output and full duplex

`TtsRuntime` owns the entire speech-output stack under one lifetime — F5 engine,
synthesizer adapter, tokenizer, two rings, duplex bridge, playback device, worker thread —
and `main.cpp` holds one `optional<TtsRuntime>` and calls four methods.

**One reply, two audiences (2026-08-06).** The model is asked for a short `<voice>` block
and a full `<ui>` block, and `ReplySplitter` divides the stream between the speaker and
the screen as it arrives. It fails *open* — an untagged reply is spoken in full at
end-of-stream — which is the deliberate opposite of the commit gate's fail-closed rule,
because a mistake there costs money and a mistake here costs the user their assistant.
Downstream, `NormalizeForSpeech` removes Markdown the model left in the spoken half and
resolves stress marks, whose failure mode on a character-level vocab is a *pause inside a
word* rather than a mispronunciation.

**Keeping the tags across a long thread (2026-08-06).** The contract is stated in the
persona, which is pinned about as hard as this codebase pins anything: its length is
published as the KV rewind floor, and `effective_keep_tokens()` (barge-in),
`kv_cache_rollback()` (end of turn) and `rebuild_history_kv()` (bounded history) all clamp
*up* to it. **It is never evicted — and it still drifts.** What it loses over a thread is
*recency*: a few turns in, the rule sits a thousand tokens behind the conversation and the
8B backbone starts answering in plain prose. Because `ReplySplitter` fails open the symptom
is not an error but a *silence* — the screen looks perfect while the spoken half arrives
late and whole, which is why this was worth a mechanism rather than a prompt tweak.

The fix is `rt::kFormatReminder` (`reply_split.hpp`, beside the contract and the parser it
must agree with), installed once via
`RealEngineControl::set_reply_format_reminder()` and appended by `generate_local_reply()` to
the end of every user block — the position immediately before the generation cue, which is
the same place `build_user_instruction()` puts the transcription contract for the same
reason. It is a *pointer* at the rule (~30 tokens), not a second copy of it.

Its lifetime is **one turn**: it is prefilled into the KV with the user block but is absent
from the retained history (`encode_history_turn` records the user's words verbatim), so a
context rebuild does not stack a reminder onto every past turn. It is deliberately *not*
applied in `commit_text_decode`, whose output is an intent offered to the commit gate rather
than a reply — tagging that would dispatch the markup as the user's words.

**The self-trigger problem and its resolution is the most instructive part of this app.**
The loudspeaker feeds the microphone; Silero scores the assistant's own voice as speech —
*correctly, because it is speech* — which fires speech-onset, which is barge-in, which
cancels the generation currently being spoken. The system's own correctness works against
it.

The first answer was to **gate the microphone** while the speaker was live. It worked, and
it cost the whole feature: during exactly the window in which a user would interrupt, the
assistant was deaf, so barge-in-while-speaking could not happen at all. The gate is now
gone — config, state and interlock — and nothing mutes, zeroes or pauses capture for any
reason. It was replaced by acoustic echo cancellation: the mic runs continuously and the
assistant's voice is *subtracted* rather than the mic being switched off.

The far-end reference has its own correction worth preserving: it was once tapped at
`PullForPlayback`, "correct about alignment and wrong about content, because the gain is
applied downstream of that tap." It is now a **WASAPI loopback capture of the render
endpoint** — post-mix, post-volume, and including audio this process never produced.

---

## 8. UI and settings

**A pure state bridge.** `AssistantWindow` is a Win32 frame whose entire client area is a
WebView2 view, with a virtual host name mapped onto the on-disk `web/` folder so
`https://<host>/index.html` serves the UI like a real site. No markup, styling or layout
logic in C++.

`AssistantView` owns no widgets and draws nothing: each producer edge turns one pipeline
event into one JSON message. This exists because events arrive from **three threads**
(engine, audio/VAD, dispatcher) and none may touch a window, COM, or WebView2 (STA:
creating thread only). Every producer is a pure function of its arguments into a JSON
string; `post_event` appends to a mutex-guarded queue and `PostMessage()`s a wake-up.

`AssistantWindow` deliberately does **not** self-delete on `WM_NCDESTROY`, unlike the
`poc_overlay` dialog it was ported from — it is owned by `main()` and outlives the HWND,
because the engine and dispatcher threads keep calling `post_event()` until they are
joined, which happens after `run_message_loop()` returns.

**The telemetry demotion is a product decision with a technical consequence.** The main
screen carries no TTFT, no KV watermarks, no gate counters; they moved to a collapsed
Diagnostics section inside Settings. One survived the move for a stated reason: a
`TokenCap` truncation produces no cloud call *and no error*, so `local.final` carries the
termination reason and the bubble marks the turn incomplete. "That is a state a person can
act on ('say it again'), not a metric."

**`settings_store.hpp` — one field list, five consumers.** `visit_fields()` is the single
declaration of what a setting is; loading, saving, page-seed JSON, save-back JSON and the
restart question are all loops over it. The header is explicit that this is not tidiness:
the previous shape spelled each field out in four places plus a `data-restart` attribute
in the markup, and the failure mode was silent — "a setting that persists but never loads,
or one the restart banner does not know about."

The **tier split** answers exactly one question — can this be applied to the running
engine?

- **LIVE** — sampling, reply cap, VAD, cadence, context policy, hotkeys → atomics + the
  `speech_pipeline_*` setters, read at the next VAD block or turn boundary.
- **LIVE\*** — `system_prompt`, the one live setting that costs engine work: its KV is the
  frozen prefix every turn is built on, so a change is a cache *rebuild* marshaled onto
  the engine thread, not an atomic store.
- **RESTART** — `model_dir`, `audio_head`, `device_id`, `simulated`, `max_context`,
  `pipeline_mode`, …

Precedence is `persisted settings < CLI flags`, and a typed flag is **not written back** —
a one-off `--simulated` must not silently become permanent. This needs the `TypedFlags`
struct because `parse_cli` *resolves* values (from a CWD `config.json`, from built-in
defaults) whether or not anything was typed, so a non-empty field says nothing about user
intent.

---

## 9. The CUDA context budget

With everything armed the process holds **three CUDA contexts**: the engine, the
ONNXRuntime CUDA EP (F5 TTS), and ggml (whisper.cpp). This is acknowledged rather than
hidden, and the mitigation is honest about its own limits: "nothing can make two contexts
not share SMs, so the contention is managed by **scheduling** instead" — at most one
utterance in flight, the encode between turns rather than during one, and a deliberately
shallow queue so a backlog is refused rather than silently deepening the latency it exists
to reduce.

The device is selected per-thread in each owner's constructor, because CUDA's current
device is per-thread — a detail that is easy to get wrong and is called out at three
separate sites.

---

## 10. Assessment

### What is genuinely strong

1. **The commit gate's failure asymmetry.** One rule, one function, one parameter that
   cannot be omitted, and every mistake costs a missed answer instead of money. This is
   the right shape for a feature that spends the user's money, and it is enforced
   structurally rather than by discipline.
2. **One base pointer across two backends.** The simulated path is a rehearsal, not a
   parallel implementation. This is the difference between an offline mode that works and
   one that rots.
3. **Transport-shaped local inference.** Making "answer locally" an `IIntentTransport`
   rather than a second pipeline means the local answer cannot be gated by different
   arithmetic than the billed one. A second route from speech to an answer would have been
   a second place for the commit rule to be subtly wrong.
4. **The header comments state contracts and, unusually, record superseded designs and why
   they lost** — the mic gate that cost barge-in, the `PullForPlayback` tap that was right
   about alignment and wrong about content, the four-places-per-field settings shape. That
   is institutional memory a reader cannot reconstruct from the code.
5. **Optional-by-construction dependencies.** Six capabilities, each absent from the binary
   rather than disabled at runtime, each with a documented degradation.

### Risks and gaps

1. **`main.cpp` is 2 518 lines — the app's one monolith.** Every module around it is
   tightly scoped (the largest is `settings_store.hpp` at 703), which makes the imbalance
   conspicuous rather than uniform. It holds CLI layering, two backend bring-ups, mode
   construction, TTS/AEC wiring, five thread bodies and the shutdown sequence. Nothing is
   *wrong* in it, but it is the file where a future contributor is most likely to break an
   ordering invariant that is only documented in a comment several hundred lines away.
   The `ISpeechMode` extraction pattern already used for Modes A/C is the obvious template.
2. **VRAM is the binding constraint and no single place states the total.** Components:
   ~5.3 GB AWQ weights + ~0.5 GB FP32 KV at the default `max_context = 2048` (~256 KB/token
   at 8B geometry) + ~1.6 GB GGML (Mode C) + the F5 DiT + the ORT CUDA EP's own arena. On a
   12 GB card that is tight enough that the *combination* matters, but each subsystem
   documents only its own footprint — and `max_context` is a user-settable Restart knob
   clamped to 131072, so a user can raise the KV term by ~32 GB with no warning that the
   other four terms exist. A configuration matrix ("Mode C + TTS at context N needs X")
   would prevent a class of startup OOM that currently surfaces as a failure inside
   whichever component happens to allocate last.
3. **The web UI loads `marked` and `highlight.js` from a CDN.** There is a documented
   self-contained fallback in `app.js`, and the comment correctly notes that no network is
   "a very ordinary state for a local voice assistant" — so this is handled, not broken.
   But a local-first assistant reaching for jsdelivr on every launch is still a posture
   mismatch, and vendoring both would cost little.
4. **Mode B is named in the `ISpeechMode` contract but unreachable here.** Harmless today;
   worth a line in the enum comment saying which app wires which mode, so the interface
   does not read as three-of-three when it is two-of-three.
5. **The single-engine-thread doctrine is enforced by convention in this app.** `EngineCom`
   has `BLACKWELL_VERIFY_OWNING_THREAD()` debug asserts at the COM boundary, but
   `voice_assistant` drives the white-box tier below that edge, so the marshalling
   discipline (`post_engine_task` from the ASR worker, from `send()`) rests on review. A
   debug-only owning-thread assert on `EngineControlBridge`'s execute hooks would close
   this cheaply and would have caught the exact bug the cascade's marshalling comment
   describes as "corrupts quietly."

### Verdict

This is the most complete application in the repository and the one where the
architectural doctrine pays off most visibly: the commit gate, the two-backend symmetry
and the transport seam are all cases where a structural choice made a whole class of bug
unrepresentable rather than merely unlikely. The dominant technical debt is concentration
in `main.cpp` and an unstated aggregate VRAM budget — both additive to fix, neither
requiring a design change.
