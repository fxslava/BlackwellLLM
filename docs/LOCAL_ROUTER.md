# Local Router — the strict arbiter in front of the Cloud API

**Status:** landed 2026-08-02. Supersedes the eager barge-in cloud design.
Extended 2026-08-06: typed input routes through the same gate and the same
transport seam, a generation can be stopped mid-flight, and every reply is split
into a spoken half and a written one.

## The problem this replaces

The previous design streamed to the Anthropic API eagerly and aborted the request
mid-stream when the user barged in. That is backwards, financially: **input and
prefill tokens are billed at acceptance.** By the time there is an in-flight
stream to abort, the expensive part of the request has already been paid for. A
user who restarts a sentence three times paid three prefills for one answer.

Aborting also can't be made cheap by tuning. The only way to not pay for a
fragment is to never send it.

## The pattern

The local LLM (Llama 1B/8B via Ultravox) is the **sole gatekeeper**. It decides
what a finished thought is; the network layer has no opinion and no veto.

```
                    audio / VAD thread          engine thread              dispatcher thread
                    ──────────────────          ─────────────              ─────────────────
  user speaks  ──▶  push_pcm()
  user pauses  ──▶  on_silence_timeout()  ──▶  commit_and_decode()
                                                run_decode_loop()
                                                  │ per token:
                                                  │   cancelled(gen)? ──▶ BargeIn 
                                                  │   decode fault?   ──▶ Fault
                                                  │   is_eos(next)?   ──▶ Eos     ◀── the only
                                                  │   cap reached?    ──▶ TokenCap    accepting
                                                  ▼                                   branch
                                                publish_intent(reason, …)
                                                  │
                                                  ▼
                                          IntentCommitQueue::offer()
                                            Eos      → commit ═══════════▶ wait_pop()
                                            anything → drop + count            │
                                            else                                ▼
                                                                    build_intent_request()
                                                                          │
                                                                          ▼
                                                                  ClaudeStreamClient::send()
                                                                     ← money is spent HERE
  user barges in ──▶ on_speech_start()  ──▶  epoch bump  ──▶  loop exits BargeIn, nothing sent
```

Continuous re-translation streaming is bypassed on this pathway; it runs in
**wait-for-pause** mode, because the commit rule needs a definite end-of-thought
and a re-translating stream never has one.

## The commit rule

> A local generation is dispatchable **iff** it terminated on EOS — the model
> emitted `<|eot_id|>` / `<|end_of_text|>` of its own accord.

| `TerminationReason` | Meaning | Dispatch |
|---|---|---|
| `Eos` | Model stopped on its own. A complete thought. | **yes** |
| `BargeIn` | The user interrupted; the epoch was bumped mid-generation. | no |
| `TokenCap` | Hit `max_new_tokens`. Truncated, not finished. | no |
| `Fault` | `decode_one` returned a non-Success status. | no |
| `None` | Loop never ran. | no |

The rule is enforced in **exactly one function**: `IntentCommitQueue::offer()`,
whose first parameter is the reason. There is no other path into the queue, so a
caller cannot enqueue without stating why the loop stopped.

**The failure asymmetry is deliberate.** A caller who forgets to call
`publish_intent()` at all gets *silence* — nothing dispatched — never a spurious
paid request. Likewise, `EngineControlBridge::is_eos()` returns `false` in the
base tier (no tokenizer there), so a bridge whose override was forgotten fails
closed. Every "I got this wrong" path costs a missed answer, not money.

## What made this possible

Before this change, `run_decode_loop` had **no EOS check at all**. It ran to
`max_new_tokens` and exited `is_final=1, BRIDGE_OK` for barge-in, token cap, and
completion alike. No consumer could tell a finished thought from a severed one,
so there was nothing to gate on. Adding the reason to the loop was the bulk of
the work; the network client was the easy half.

Note the deliberate split: `run_decode_loop` still returns
`EngineStatus::Success` for both `BargeIn` and `TokenCap`, because neither is
an *engine* fault. **Dispatchability is carried by the reason, never re-derived
from the status.**

## What the network client stopped doing

With requests guaranteed final before they are sent, `ClaudeStreamClient` lost
its entire concurrency layer: the worker thread, the SPSC job queue, the
monotone cancel epoch, `curl_multi_wakeup()`, and latest-wins supersede. It is
now **synchronous** — one blocking `send()` on the dispatcher thread.

`curl_multi` is still used internally rather than `curl_easy_perform`, for one
reason: `easy_perform` cannot express separate TTFT and inter-byte-stall
budgets, and `CURLOPT_TIMEOUT` is a wall-clock cap on the whole transfer, which
is exactly wrong for a long generation.

`shutdown()` remains, but it is teardown only — it exists so the process can
exit promptly, **not** as a barge-in mechanism.

**One honest residual — since collected (2026-08-06).** Aborting mid-stream still
saves *output* tokens, which bill as they stream. That is a real but much smaller
number than the prefill, and it was judged not worth reintroducing cross-thread
cancellation for. The Stop button changed the arithmetic: the same abort also
stops the GPU and the speaker, and a user pressing it wants all three. So a
narrow, non-latching `abort()` exists (see "Stopping a generation" below) — but
note what did **not** come back with it: no worker thread, no job queue, no
supersede, no latest-wins. `send()` is still one blocking call, and the abort is
one atomic the poll loop reads.

## Two operational risks this design creates

**1. `TokenCap` is a silent black hole.** "Missing EOS = no dispatch" means a
local truncation produces no answer *and no error*. If `max_new_tokens` is
mis-sized for the intent-extraction prompt, the entire cloud pathway is quietly
dead and nothing in the system says so. `IntentCommitQueue::dropped_token_cap()`
exists for exactly this — **surface it in the UI or telemetry**, not just in a
debugger. A rising `dropped_token_cap` is the signature.

**2. Prompt-cache invalidation is the other silent, expensive failure.** Caching
is a byte-exact prefix match rendered `tools → system → messages`. A timestamp
or session UUID in the "frozen" instructions, a rebuilt system prompt on a mode
toggle, or non-deterministic glossary serialisation each drive cache hits to
zero with no error raised. The layout in `intent_request.hpp` puts stable
content first and the volatile intent after the last breakpoint; the check is
`Usage::cache_read_input_tokens`, and a persistent zero across requests means
one of the above.

Retrying is safe here in a way it was not in the streaming design: the committed
payload is immutable, so a retry is byte-identical and hits the warm cache
(~0.1× input cost).

## Typed input is an intent, not a local generation

**Landed 2026-08-06.** A typed message now enters the pipeline at exactly the
point a transcript does: offered to `IntentCommitQueue` as `TerminationReason::Eos`,
drained by the dispatcher, answered by whichever leg `RoutedTransport` picks.

It did not used to. `EngineControlBridge::submit_text()` ran a full **local
decode of an answer** and then offered *that answer* to the gate as the intent —
so with the cloud leg selected the GPU generated a reply nobody displayed and the
remote model was asked to respond to it. The "use the local model" toggle was, in
effect, ignored for anything typed, and the turn was paid for twice.

The fix is a deletion: the routing already existed and this was the one path
bypassing it (`submit_typed_turn` in `main.cpp`). Two properties it inherits by
construction rather than by re-implementation — the **leg latch**
(`last_send_was_local()`, so a typed turn is persisted iff the cloud answered it)
and the **session latch**.

Two details are load-bearing and both are copied from the cascade's `publish()`:

- **Offered on the engine thread**, via `post_engine_task`. `IntentCommitQueue`
  is single-producer and its producer is defined to be the engine thread.
- **The user's bubble is painted before the offer.** `offer()` wakes the
  dispatcher, whose first act paints the assistant's bubble; both land in the
  same FIFO UI queue, so offering first puts the answer above the question.

## Stopping a generation

The commit gate's argument is about what may be **sent**; this is about what is
already in flight. The Stop button (`interrupt_generation`, sharing
`on_cancel` with the cancel hotkey) reaches `cancel_in_flight`, which stops all
three legs unconditionally — asking which one is live would cancel the idle leg
when the user has just flipped the toggle:

| Leg | Mechanism |
|---|---|
| local decode | epoch bump (`cancel_generation`) — the loop checks it per token and ends as `BargeIn`, so nothing is re-dispatched |
| network | `IIntentTransport::abort()` → a monotone **cancel epoch** in `CurlStreamCore`, checked in the poll loop and in the write callback |
| speaker | `TtsRuntime::BargeIn()` |

`abort()` is deliberately **not** `shutdown()`: shutdown latches (the client is
finished for the process), while a cancel must leave the very next turn sendable.
The epoch is compared against a value latched at the top of `run()`, so a Stop
pressed between two transfers lands ahead of the next one instead of killing it.
An aborted send reports `Status::Cancelled`, which `is_retryable()` refuses —
retrying an intent the user stopped would bill them twice.

**What this does and does not save.** Input and prefill were paid at acceptance
and no cancellation recovers them — that is why the gate exists and why this
button is not a substitute for it. What it saves is **output** tokens, which bill
as they stream, plus the GPU time and the speech nobody wanted. That is the
"honest residual" the section above records, now collected.

## One answer, two audiences

Every reply — from either leg — is asked for in two tagged blocks:

```xml
<voice>Short spoken summary, plain prose.</voice>
<ui>The full answer, Markdown and all.</ui>
```

`reply_split.hpp` holds **both halves of that agreement**: the instruction
(`kOutputContract`, appended to the persona by `compose_system_prompt` — one
call site per place a persona reaches a model, so the legs cannot drift) and the
parser (`ReplySplitter`). They live in one file because a tag renamed in one and
not the other fails silently, and the failure is the assistant going mute while
the screen looks fine.

The splitter sits on `IntentDispatcher::set_on_text`, above `RoutedTransport`'s
choice, so a local reply and a billed one are divided by the same code. The `<ui>`
half goes to the screen, to `ChatHistory` and thus to `sessions.json`; the
`<voice>` half goes to the speaker and nowhere else.

**It fails OPEN, and that is the opposite of the commit rule on purpose.** A
model that emits no `<voice>` block has its written answer spoken in full at
end-of-stream. The cost is time-to-first-audio on non-compliant turns; the
alternative is speech that silently stops whenever the model forgets a tag,
which is indistinguishable from broken audio output. The gate fails closed
because a mistake there costs money; this fails open because a mistake here
costs the user their assistant.

`saw_voice_block()` is the telemetry for it — a run of `false` means the persona
edit dropped the contract or the backbone is too small to follow it.

### The last thing that touches the text

`NormalizeForSpeech` (`src/tts/speech_text.hpp`), applied per chunk in
`TTSDuplexBridge` — **per chunk and not per token**, because a token stream
splits `**` down the middle and a normaliser fed fragments cannot see the
constructs it is removing.

The prompt asks; this guarantees. It covers the three residuals a prompt cannot:
a model that puts Markdown in `<voice>` anyway, the fallback path where the
Markdown answer *is* what gets spoken, and stress marks — which no instruction
can regularise because the two notations look identical on screen.

> Russian TTS text marks lexical stress either as `U+0301` **after** the vowel
> (`хорошо́`) or as `+` **before** it (`хорош+о`). F5 is character-level, so only
> the convention its checkpoint was fine-tuned with means anything — and an
> unknown character maps to id 0, which **is the space character**. A stress mark
> the voice does not know therefore inserts a pause *inside a word*. The symptom
> is a stutter, not a mispronunciation, and nothing raises an error.

Hence `StressPolicy` rather than a hardcoded rule: the caller states which
convention its checkpoint speaks and the normaliser converts between them.
`Strip` is the default because getting it wrong that way is merely flat, while
getting it wrong the other way is unintelligible.

## Conversation memory on the remote leg

The local leg remembers by construction — its memory *is* the KV cache above the
system-prefix floor. A remote `/chat/completions` call has no such thing: it is
stateless, so an intent sent on its own makes the model forget the user's name
the moment the turn ends.

`src/cloud/chat_history.hpp` supplies the missing half as a **bounded sliding
window** of finished turns, replayed ahead of the live utterance by
`openai_request.hpp`. Bounded is the operative word: replaying the whole session
grows the input bill quadratically (turn *N* re-sends turns 1…*N-1*), so the
window is capped twice, by two different owners:

| Knob | Bounds | Default |
|---|---|---|
| `ChatHistory::Config::max_turns` | pairs the store **keeps** (memory) | 4 |
| `ChatHistory::Config::max_chars_per_message` | one runaway reply's footprint | 4000 |
| `OpenAiRequestOptions::max_history_pairs` | pairs that go **on the wire** (cost) | 3 |

Truncation is always from the front — the recent turn is the one the current
utterance refers back to. Three rules make the payload safe rather than merely
short:

- **Pairs only, never a half turn.** Strict `user`/`assistant` alternation is a
  protocol requirement (two consecutive `user` messages are a 400 on the stricter
  gateways), so a turn enters the window only when both halves exist.
- **Only completed exchanges become memory.** A failure, a refusal or a barge-in
  drops the turn whole: replaying a sentence the user talked over would teach the
  model that its interrupted half-answer was accepted.
- **Clamps land on codepoint boundaries.** Every cap cuts at a byte offset, and
  this app transcribes Russian — a cut through a multi-byte character is invalid
  UTF-8, which the gateway answers with a 400 that reaches the user as "the
  assistant did not answer".

**The Anthropic renderer deliberately ignores the window.** `intent_request.hpp`
is built around two cache breakpoints whose value depends on the prefix staying
byte-identical between turns; splicing a growing message list in ahead of the
last breakpoint would invalidate the cache on *every* turn — see risk 2 above.
The OpenAI-compatible renderer has no breakpoints to protect, so it replays.

## Persistence: the log vs. the window

`src/cloud/session_store.hpp` gives the remote leg a memory that survives the
process. It is a **different object from the window**, and the split is the thing
to understand before touching either:

| | `ChatHistory` (the **window**) | `SessionStore` (the **log**) |
|---|---|---|
| Question it answers | what may a request carry? | what does the app remember? |
| Bound | 4 pairs | 200 turns × 32 sessions |
| Lifetime | the process | the install |
| Touches disk | never | `%LOCALAPPDATA%\BlackwellVoiceAssistant\sessions.json` |

Fusing them would make one number answer both questions, and the honest answers
differ by an order of magnitude — the log is a **retention** decision, the window
is a **cost** one. So the log keeps everything, the window takes its *tail* at
startup (`ChatHistory::restore`), and the wire is bounded a third time by the
renderer. The acceptance shape is exactly that: the disk holds the full history,
the payload does not grow with it.

### The rule, and its two legs

> **The disk holds what the cloud leg was told and answered. Nothing else.**

The local leg is ephemeral **by construction**: its memory is the KV cache above
the system-prefix floor — VRAM this process frees on the way out — so a local
conversation has no durable form and is deliberately not given one. That is two
omissions in `main.cpp`, not one:

- **Nothing is loaded** at startup when `local_inference` is set. The blank slate
  is a decision, not an absence of data: the file may well be populated by an
  earlier cloud session.
- **No locally-answered turn is appended**, ever — keyed off
  `RoutedTransport::last_send_was_local()`, the leg *latched at the top of
  `send()`*, not the toggle's position afterwards. `local_inference` is a **live**
  setting, so a user can flip it mid-answer; reading the switch in `on_complete`
  would occasionally write a local conversation to disk or drop a paid turn.

The startup seed is **not** re-taken when the toggle flips mid-session. Splicing a
conversation from three days ago into one already on screen is worse than starting
the cloud leg cold, and nobody asked for it. A mixed session therefore leaves the
window (and the screen) as one conversation while the *log* skips the local turns.

### Multi-session, and the drawer that exposes it

Every operation is keyed by session id and the file format is a *list* of
sessions, evicted least-recently-appended. `default_session` is a constant only
because the app has to boot into *something* it can name without having
remembered it first; every conversation created afterwards is named by
`make_session_id()` (epoch seconds + a per-process counter — sortable as text,
and it matches the `updated_at` sitting next to it in the file).

The UI is a slide-over drawer behind a hamburger in the top-left, deliberately
**not** a layout column: the shell stays `grid-template-rows: auto 1fr auto` and
the panel is `position: fixed`, so a closed drawer costs the chat no reflow, no
gutter and no width. Only `transform` animates.

**The page never mutates the transcript itself.** A click sends `session.select`
and waits for the app to answer with `session.restore`, exactly as the composer
paints no bubble and waits for `user.text`. One source of truth for what is on
screen, on every path.

| Page → app | App → page |
|---|---|
| `session.list_request` (on every drawer open) | `session.list` — rows, `active`, `ephemeral` |
| `session.select` / `session.new` / `session.delete` | `session.restore` — the whole transcript, repainted |

`session.restore` carries the **entire** transcript in one event rather than
replaying `user.text`/`remote.delta` per turn. Those producers advance the phase,
freeze live bubbles and drive the typing indicator; replaying them would animate a
conversation that already happened and leave the status line describing a turn
that is not running. It is also the only shape that can render an *empty* session
(a new chat) without a special case.

The turns it carries come from the **window**, not the store — so what is drawn is
exactly what the model will be told. If the tail was clamped to four pairs, four
pairs is what the user sees; painting the full log beside a model that only knows
its tail would be a more elaborate lie than painting nothing.

**Switching a session is five steps, and step 4 is the one with no symptom until
it bites:**

1. cancel the turn in flight — it belongs to the conversation being left
2. swap the active id — so the next dispatch latches the new one
3. reload the window from disk (`ChatHistory::restore`, tail-first)
4. **rewind the local KV** (`rebuild_system_prompt`) — the local leg's memory is
   its KV above the system-prefix floor and it survives a switch by default, so
   without this the on-device model answers the newly-opened conversation using
   the context of the one just closed
5. repaint the transcript, then the drawer

A new conversation is **not written to disk until its first turn is answered**, so
opening five and talking in none leaves nothing behind. The drawer still shows it,
as a provisional row — a sidebar that could not show where you *are* would be
lying by omission.

### The session latch

A turn belongs to the conversation it was **asked in**. The active id is read once
at `on_dispatch_start` and that latched value is what `on_complete` files the
answer under — the same argument as `RoutedTransport`'s leg latch one section up.
Without it, opening another conversation while an answer streams drops that answer
into a conversation it has nothing to do with.

### Other decisions worth knowing

- **Writes are atomic** (temp file + rename). The writer is a per-turn flush on
  the dispatcher thread; a plain truncate-and-write loses the whole log to a crash
  in the millisecond it is empty.
- **Flushed per turn**, not at shutdown. This process holds ~8 GB of GPU state and
  is killed rather than closed often enough that a shutdown-only flush is a
  history that mostly does not survive.
- **The file is untrusted input.** It sits next to `settings.json` in a directory
  the user can open, so corrupt JSON degrades to an empty store, one malformed
  session costs only that session, half turns are dropped on the way in, and every
  message is re-clamped on a codepoint boundary.
- **A persona change wipes the log too** — but only the *active* session.
  `on_system_prompt_apply` clears both halves of it: a transcript produced by a
  different assistant, restored into a session running the new one, reproduces
  exactly the incoherence the live `clear()` prevents, just delayed by one launch.
  The other conversations are left alone; they were produced by the old persona
  too, but they are not on screen, and a settings edit deleting data the user was
  not looking at is a different and much worse thing.
- **The drawer lists sessions in local mode as well, and says so.** Reading the
  file is not the same act as seeding a conversation from it, so the list loads
  unconditionally while the startup *seed* stays gated. A footer in the drawer
  states that nothing is being saved, pushed from `RoutedTransport::use_local()`
  rather than re-derived from a settings copy — a "not being saved" notice must
  not be one stale field away from lying.

## File map

| File | Role |
|---|---|
| `src/bridge/intent_commit.hpp` | `TerminationReason`, `IntentRecord`, `IntentCommitQueue`. **The gate.** Header-only, CUDA-free. |
| `src/bridge/engine_control_bridge.{hpp,cpp}` | `run_decode_loop` reports the reason; `is_eos()` hook; `publish_intent()`. |
| `src/cloud/intent_request.hpp` | `RequestContext` + `ChatTurn`; the Anthropic body builder and its cache-breakpoint layout. Dependency-free. |
| `src/cloud/openai_request.hpp` | The `/chat/completions` body: system → replayed history → live turn. Dependency-free. |
| `src/cloud/chat_history.hpp` | `ChatHistory` — the bounded window that gives the remote leg a memory. Header-only, thread-safe. |
| `src/cloud/session_store.hpp` | `SessionStore` — the durable, multi-session log behind that window, plus `SessionSummary`/`make_session_id()` for the picker. Atomic writes to `sessions.json`; nlohmann, no curl. |
| `audio_sandbox/voice_assistant/assistant_view.hpp` | `set_sessions` / `on_session_restored` — the two producers that carry the drawer's data and the transcript repaint to the page. |
| `audio_sandbox/voice_assistant/web/` | The drawer's markup, styling and logic (`#drawer`, `restoreTranscript`, `renderSessions`). |
| `src/cloud/intent_dispatcher.hpp` | Owns the one thread allowed to make a paid request. Waits on `wait_pop()`. |
| `src/cloud/cloud_types.hpp` | `Status`/`Usage`/`Result`/`Callbacks` + inline `to_string`/`is_retryable`. Transport-agnostic, no libcurl. |
| `src/cloud/intent_transport.hpp` | `IIntentTransport` — the seam, interface only. Carries `abort()` (stop what is in flight, stay armed) next to `shutdown()` (latching teardown). |
| `src/cloud/curl_stream_core.hpp` | The shared HTTP half; owns the **cancel epoch** both live clients abort through. |
| `audio_sandbox/voice_assistant/reply_split.hpp` | The `<voice>`/`<ui>` contract: `kOutputContract` + `compose_system_prompt` (the instruction) and `ReplySplitter` (the parser). Header-only, CUDA-free. |
| `src/tts/speech_text.{hpp,cpp}` | `NormalizeForSpeech` — Markdown removal, `StressPolicy`, symbol filtering. Pure function. |
| `tests/bridge/reply_split_test.cpp` | Tags split across deltas, and the fail-open fallback. CPU-only. |
| `tests/tts/speech_text_test.cpp` | Both stress notations in all three directions, and what must NOT be rewritten (`2 + 2`, `max_new_tokens`). CPU-only. |
| `src/cloud/offline_transport.hpp` | `OfflineTransport` — the simulated remote. Header-only; builds with `BUILD_CLOUD_CLIENT=OFF`. |
| `src/cloud/claude_transport.hpp` | `ClaudeTransport`: the live transport over `ClaudeStreamClient`. |
| `audio_sandbox/voice_assistant/` | Standalone app: the voice loop + the two-card GUI + the gate telemetry bar. |
| `src/cloud/claude_stream_client.{hpp,cpp}` | Synchronous SSE client. PIMPL'd; the `.cpp` is the only TU that sees libcurl/simdjson. |
| `src/cloud/claude_sse.hpp` | Incremental SSE framing + simdjson extraction. Templated on the sink, so it is testable without libcurl. |
| `tests/bridge/intent_commit_test.cpp` | The commit rule, mechanically checked. CPU-only. |
| `tests/bridge/chat_history_test.cpp` | The window's bounds + the dispatcher wiring, asserted on the bytes a gateway would receive. CPU-only. |
| `tests/bridge/session_store_test.cpp` | The log's bounds, the untrusted-file cases, the sidebar's summaries, and the restart behaviour — including that a local turn never reaches a disk. CPU-only. |

## Build

The gate and the decode-loop changes are always built — they are header-only /
engine-side and add no dependencies. The **cloud target is on by default**, and
it is the one component whose dependencies are neither fetched nor vendored, so
they must be present before the first configure:

```bat
vcpkg install curl[core,ssl,http2]:x64-windows simdjson:x64-windows
cmake --preset x64-debug -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake
```

`BUILD_CLOUD_CLIENT` defaults **ON** (root `CMakeLists.txt`). `CloudDeps.cmake`
uses `find_package(... REQUIRED)`, so a machine without libcurl + simdjson now
**fails to configure** rather than silently producing an offline-only app. On
such a machine, opt out explicitly:

```bat
cmake --preset x64-debug -DBUILD_CLOUD_CLIENT=OFF
```

`OFF` remains a fully supported configuration — the local pipeline is testable
with no network client at all.

## Wiring it up

```cpp
blackwell::bridge::IntentCommitQueue commit_queue(8);
bridge.set_commit_queue(&commit_queue);        // engine thread, during setup

blackwell::cloud::ClaudeStreamClient client({.api_key = read_env("ANTHROPIC_API_KEY"),
                                             .beta = "server-side-fallback-2026-07-01"});
blackwell::cloud::IntentDispatcher dispatcher(commit_queue, client, make_context);
dispatcher.set_on_text([&](std::string_view t) { ui_queue.push(std::string(t)); });
dispatcher.start();
```

Leaving `set_commit_queue()` unbound is a supported configuration: the local
pipeline runs exactly as before and nothing is ever dispatched. Cloud routing is
opt-in at runtime as well as at build time.

---

## The transport seam

`IntentDispatcher` holds an `IIntentTransport&`, not a concrete client. Two
implementations:

| Transport | Needs libcurl | Use |
|---|---|---|
| `OfflineTransport` | no | Offline development. Streams a canned reply in pieces after a simulated round trip, so the GUI's incremental-render path is exercised without a network. |
| `ClaudeTransport` | yes | Live. Thin adapter over `ClaudeStreamClient`. |

The seam is at the transport, not the dispatcher, because everything *around*
the send — draining the gate, retry with backoff, honouring `retry-after`,
counting outcomes, staying interruptible on shutdown — is identical for both. A
mock that reimplemented that loop would exercise different control flow than
production, which defeats the purpose of having one.

The decisive constraint: `OfflineTransport` must work with
`BUILD_CLOUD_CLIENT=OFF`. That is why `to_string()` and `is_retryable()` are
`inline` in `cloud_types.hpp` rather than living in the libcurl TU — otherwise
merely *classifying* a `Result` would drag in libcurl and the offline
configuration would not exist.

## `voice_assistant`

A standalone binary that runs the whole gated pipeline with **no checkpoint, no
GPU, and no network**: the real control plane (SPSC command ring, barge-in
epoch, wait/pump), the real VAD state machine, the real commit gate, and a real
dispatcher thread — with `SimulatedEngineControl` and `OfflineTransport` at the two
ends. Every acceptance path is reachable on a laptop.

Four threads, and the single-engine-thread doctrine holds trivially:

| Thread | Touches |
|---|---|
| UI | `AssistantView` + ImGui only |
| DSP worker | mic → log-mel → spectrogram; PCM tap into the mode |
| Engine | the **only** thread that calls the control; offers turns to the gate |
| Dispatcher | blocks on the gate, then on the transport; never touches the engine |

`RealEngineControl` and `SimulatedEngineControl` both derive from
`EngineControlBridge`, and `ConversationalMode` borrows the control as a base
pointer, so the GPU path is a single construction site away — plus
`audio_translator`'s checkpoint/audio-head bootstrap, which is deliberately not
duplicated here.

The **Gate test** panel caps the mock at N pieces, forcing the `TokenCap` path
so the truncation telemetry can be seen without sitting through a real
256-token decode.

## Running `voice_assistant`

```bat
:: Simulated backend — no GPU, no checkpoint, no network.
voice_assistant

:: Real GPU pipeline (8B AWQ + Ultravox audio head).
voice_assistant --model-dir F:/AI/llama-3.1-8B-Instruct-AWQ-INT4 [--device 0]

:: Force the simulated backend even with a checkpoint on disk.
voice_assistant --model-dir <path> --simulated
```

Run from `audio_sandbox/` (the default positional data dir is `data`, and
`mel_filters.bin` lives at `audio_sandbox/data`).

**The real path is opt-in by a TYPED flag, not by checkpoint availability.**
`rt::parse_cli` resolves `kDefaultModelDir` from disk when no `--model-dir` is
given, so `have_model_dir` is true on any machine that happens to have the
default checkout. Keying off it would boot a 5.3 GB GPU pipeline on a bare
`voice_assistant` — the opposite of the documented fallback. `voice_assistant`
therefore scans argv itself for `--model-dir` / `--real` and requires *both* the
flag and a resolved checkpoint.

`--device <id>` is applied on **two** threads: the bootstrap thread that
allocates, and the engine thread that launches kernels. CUDA's current device is
per-thread, so setting it only in `main()` would run kernels on device 0 against
device-N memory — a bug that only reproduces on a multi-GPU box.

Shared bootstrap lives in `audio_sandbox/translator/engine_bootstrap.hpp`
(`bring_up_real_engine`), so `voice_assistant` and `audio_translator` cannot
drift on the validate-before-allocate ordering or the engine-outlives-control
destruction order.
