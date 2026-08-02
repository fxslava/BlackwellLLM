# Local Router — the strict arbiter in front of the Cloud API

**Status:** landed 2026-08-02. Supersedes the eager barge-in cloud design.

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

**One honest residual:** aborting mid-stream would still have saved *output*
tokens, which bill as they stream. That is a real but much smaller number than
the prefill, and it is not worth reintroducing cross-thread cancellation for. If
a barged-in answer should be discarded, discard it at the consumer — the bytes
are paid for either way.

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

## File map

| File | Role |
|---|---|
| `src/bridge/intent_commit.hpp` | `TerminationReason`, `IntentRecord`, `IntentCommitQueue`. **The gate.** Header-only, CUDA-free. |
| `src/bridge/engine_control_bridge.{hpp,cpp}` | `run_decode_loop` reports the reason; `is_eos()` hook; `publish_intent()`. |
| `src/cloud/intent_request.hpp` | Request-body builder + cache-breakpoint layout. Dependency-free. |
| `src/cloud/intent_dispatcher.hpp` | Owns the one thread allowed to make a paid request. Waits on `wait_pop()`. |
| `src/cloud/cloud_types.hpp` | `Status`/`Usage`/`Result`/`Callbacks` + inline `to_string`/`is_retryable`. Transport-agnostic, no libcurl. |
| `src/cloud/intent_transport.hpp` | `IIntentTransport` — the seam, interface only. |
| `src/cloud/offline_transport.hpp` | `OfflineTransport` — the simulated remote. Header-only; builds with `BUILD_CLOUD_CLIENT=OFF`. |
| `src/cloud/claude_transport.hpp` | `ClaudeTransport`: the live transport over `ClaudeStreamClient`. |
| `audio_sandbox/voice_assistant/` | Standalone app: the voice loop + the two-card GUI + the gate telemetry bar. |
| `src/cloud/claude_stream_client.{hpp,cpp}` | Synchronous SSE client. PIMPL'd; the `.cpp` is the only TU that sees libcurl/simdjson. |
| `src/cloud/claude_sse.hpp` | Incremental SSE framing + simdjson extraction. Templated on the sink, so it is testable without libcurl. |
| `tests/bridge/intent_commit_test.cpp` | The commit rule, mechanically checked. CPU-only. |

## Build

The gate and the decode-loop changes are always built — they are header-only /
engine-side and add no dependencies. The **cloud target is opt-in**:

```bat
vcpkg install curl[core,ssl,http2]:x64-windows simdjson:x64-windows
cmake --preset x64-debug -DBUILD_CLOUD_CLIENT=ON ^
      -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake
```

`BUILD_CLOUD_CLIENT` defaults **OFF** so the default preset keeps working on a
machine with neither library. The local pipeline is fully testable with it off.

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
