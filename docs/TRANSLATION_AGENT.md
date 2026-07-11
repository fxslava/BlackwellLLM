# TranslationAgent — Architectural Proposal (R&D)

Status: **partially implemented** — the live-preview path has landed in
poc_overlay (`live_translation_tracker`, `spotlight_composer`,
`translation_service`, `caret_tracker`), riding the engine's Continuous
Speculative Tracking primitive; preview debounce and divergence-point
reprefill are done (see §6, §9). The commit-tier ReAct loop and per-app
profiles remain proposal-stage.
Scope: the integration layer between the poc_overlay UI pipeline
(`HookManager` → `CaretTracker` → `OverlayWindow` / `TextInjector`) and the
agent stack (`AgentOrchestrator` + `ToolRegistry` + `BlackwellLLMAdapter`).

---

## 1. Where it sits

A new module `src/translation_agent/`, linked by the overlay app. It is the
*fourth thread* of the process and the single owner of the CUDA engine —
a direct reuse of the playground's proven `ModelService` pattern
(worker thread + mutex-guarded state machine), repackaged as a library
instead of an HTTP server.

```
[UI thread]        WH_KEYBOARD_LL hook, OverlayWindow, tray
     |  onTrigger / onCommit (O(1) enqueue)
     v
[STA thread]       CaretTracker: UIA source of truth, TextInjector (apartment-bound)
     |  SegmentContext snapshot (plain data, NO COM pointers)
     v
[Agent worker]     TranslationAgent: BlackwellLLMAdapter + AgentOrchestrator
     |  callbacks (run on agent worker; carry a generation id)
     v
marshal back:  preview  -> PostMessage to UI thread   (overlay text)
               commit   -> CaretTracker queue          (re-resolve + inject on STA)
```

Prerequisite refactor: `BlackwellLLMAdapter` moves out of
`src/tools/playground/` into a shared library (e.g. `src/llm_adapter/`) so
both the playground and the overlay link it. No code changes, just a move —
it already has zero playground dependencies.

## 2. Two request classes, two execution tiers

The translator is not one workload but two, with very different
latency/quality budgets:

| | **Preview** (word boundary) | **Commit** (Ctrl+Enter) |
|---|---|---|
| Trigger | `CaretUpdate::wordBoundary` (the gate already built for this) | `HookManager::onCommit` |
| Budget | ~sub-second, streamed token-by-token into the overlay | seconds are acceptable |
| Execution | **single `generate()` + `ToolParser::parse` for a `<finish>`** — no tool loop | **full `AgentOrchestrator::run()`** with tools, `max_iterations` ≈ 4 |
| History effect | ephemeral — never appended to the durable transcript | appended as a durable (source, translation) turn |
| Cancellation | superseded by every newer preview/commit (latest-wins coalescing) | cancels any in-flight preview; not itself coalesced |

Both tiers share the same system prompt, registry and adapter; the preview
tier simply skips the loop. This keeps the common case (one-shot translation)
at exactly one model call while still letting the commit path consult tools
(glossary, history search) when the model decides it needs them.

## 3. Interfaces (sketch, not final code)

```cpp
namespace translation {

// A snapshot taken on the STA thread. Plain data only: UIA COM interfaces are
// apartment-bound and borrowed, so they must NEVER ride into the async world.
struct SegmentContext {
    std::wstring visual_buffer;     // authoritative text before the caret (UIA)
    std::wstring app_hint;          // focused-app exe name, optional domain signal
    uint64_t     generation;        // monotone snapshot id (staleness filter)
};

struct CommitResult {
    std::wstring source;            // what we translated (for inject-time verify)
    std::wstring replacement;       // the translation to inject
    uint64_t     generation;
};

class TranslationAgent {
public:
    enum class State { Unloaded, Loading, Ready, Busy, Error };

    struct Callbacks {                       // ALL run on the agent worker thread;
        // streaming preview text (incremental), for the overlay
        std::function<void(uint64_t gen, const std::wstring& delta)> on_preview_delta;
        std::function<void(uint64_t gen, const std::wstring& full)>  on_preview_done;
        // finished commit — receiver marshals to the STA thread and injects
        std::function<void(CommitResult)>                            on_commit_done;
        std::function<void(State, const std::string& detail)>        on_state;
    };

    TranslationAgent(std::string model_dir, AgentProfile profile, Callbacks cbs);

    // All non-blocking, callable from any thread; O(1) enqueue + condvar notify.
    void request_preview(SegmentContext ctx);   // coalesces: latest wins
    void request_commit(SegmentContext ctx);    // preempts previews
    void notify_committed(std::wstring source, std::wstring translation);
                                                // append to durable history
    void cancel_all();                          // e.g. HookManager::onReset
    void set_idle_ttl(std::chrono::seconds ttl);// engine teardown on idle
};

// Everything needed to build the system prompt + registry for one session.
struct AgentProfile {
    std::string  target_lang;        // "English", ...
    std::string  style;              // "concise, keep code identifiers verbatim"
    agent::orch::ToolRegistry tools; // pre-registered translation tools
};

}  // namespace translation
```

### Worker-side state (owned exclusively by the agent thread)

```cpp
BlackwellLLMAdapter        adapter_;      // engine + tokenizer + KV reuse
agent::orch::ToolRegistry  tools_;
std::vector<Message>       durable_;     // system + committed (src, tr) turns
std::atomic<uint64_t>      current_gen_; // bumped by every enqueue
```

## 4. System prompt construction

Built once per session (per `AgentProfile`), then **byte-stable** — this is a
hard requirement, not a style preference (see §6 on KV reuse). Three parts:

1. **Role + protocol preamble** — "you are a translation agent"; target
   language and style from the profile; the exact `<tool_call>` / `<arg>` /
   `<finish>` grammar (same wording the playground uses, so the parser's
   expectations are met); the rule that the final translation travels in
   `<finish>...</finish>` *and contains nothing else*.
2. **Tool manifest** — rendered from `ToolRegistry::manifest()`, the API that
   was built for exactly this. Proposed initial tools (all pure-C++ callbacks,
   total, error-as-text per the ReAct contract):
   - `glossary_lookup(term)` / `glossary_pin(term, translation)` — enforce
     terminology consistency across the session (stored in the agent, not the
     model's context).
   - `search_history(query)` — full-text search over committed segments, so
     old context stays reachable *without* inlining it into the prompt
     (see §6 for why the inline window must stay small).
3. **One few-shot exchange** — a USER buffer → ASSISTANT `<finish>` pair, so
   the preview tier reliably one-shots without touring the tools.

Notably absent: timestamps, request counters, per-request app hints — anything
volatile would invalidate the KV prefix on every call. The `app_hint` rides in
the *user* turn instead.

## 5. Data flow

### Preview (word boundary)
1. Hook fires `onTrigger` → `CaretTracker::RequestUpdate` (existing path).
2. STA thread resolves the authoritative UIA text; the update callback (in
   overlay `main.cpp`) now *also* calls `request_preview({text, app, ++gen})`
   when `wordBoundary` is set — replacing today's `LogMock(L"OnWordComplete")`.
3. Agent worker: render `durable_` transcript + one ephemeral USER turn
   (visual buffer), call `adapter_.generate(...)` with a `StreamCallback` that
   (a) checks `gen == current_gen_` and returns `false` to cooperatively abort
   if stale, (b) forwards deltas to `on_preview_delta`.
4. Callback marshals to the UI thread via the `OverlayWindow::PostUpdate`
   pattern (PostMessage); the overlay renders source + live translation.
5. `ToolParser::parse` on the full completion extracts the `<finish>` text;
   fall back to the raw completion if the model skipped the tag (preview is
   best-effort; never nudge-loop here).

### Commit (Ctrl+Enter)
1. `onCommit` → `CaretTracker::RequestCommit` — **changed contract**: instead
   of running the synchronous `TransformCallback`, PerformCommit now only
   *snapshots* the authoritative source text and calls
   `request_commit({source, app, ++gen})`. The STA thread is freed
   immediately (a blocked STA thread = frozen UIA updates for seconds).
2. Agent worker runs the real ReAct loop: a **scratch** `AgentOrchestrator`
   seeded with `durable_` history + the commit USER turn. Tool calls dispatch
   through the registry; `<finish>` yields the translation.
3. `on_commit_done(CommitResult)` marshals back to the **STA thread** (new
   `CaretTracker::RequestInject(source, replacement)` queue entry — injection
   is apartment-bound, and the borrowed COM pointers from step 1 are long
   dead, so we *re-resolve* UIA fresh at inject time).
4. Inject-time reconciliation on the STA thread:
   - current UIA text == `source` → 3-tier `TextInjector::Replace` (wrapped in
     the existing `SetInjecting` guard);
   - current text *extends* `source` (user kept typing) → replace only the
     matching prefix range and leave the tail;
   - diverged → drop the result and (optionally) auto-rerun.
5. On successful injection: `notify_committed(source, translation)` appends
   the turn to `durable_` — it becomes context for every later request.

### Staleness & cancellation
One monotone `generation` counter, bumped by every enqueue. In-flight decodes
notice via the stream callback (return `false` = the adapter's built-in
cooperative stop) — no thread interruption, no engine-level cancel API needed.
Deliveries carrying `gen < current_gen_` are dropped at the callback sink.

## 6. Context history & the KV-reuse contract

This is the load-bearing performance decision. `BlackwellLLMAdapter` already
implements prefix-matching KV reuse: *a transcript that strictly extends the
previous one prefills only the tail; any divergence causes a full reprefill
from position 0.*

Therefore the transcript is structured as **stable prefix + volatile tail**:

```
[SYSTEM]  role + tools + few-shot          <- byte-stable for the whole session
[USER/ASSISTANT] committed turn 1..N       <- append-only (commits only)
--------------------------------------------- KV-reusable prefix ends here
[USER]    current visual buffer            <- differs every request
[ASSISTANT] (generation)
```

Consequences, in decreasing order of comfort:

- **Commits are cheap by construction.** Each commit extends the previous
  transcript, so prefill is O(new tokens).
- **Consecutive previews diverge at the tail.** The plain adapter path
  reprefills from position 0 each time, but the live preview path no longer
  uses it — mitigation 1 has **landed** as the engine's Continuous Speculative
  Tracking primitive:
  1. *Adapter enhancement (dense models) — DONE:* reprefill from the divergence
     point, not position 0. `EnginePrefillCoordinator::update_sequence`
     re-tokenizes the buffer, diffs it against the sequence's token mirror
     (LCP), rewinds the KV cache to the divergence point (CoW page rewind), and
     recomputes only the new suffix — the attention KV cache is
     position-addressed and self-heals on overwrite, so this is legal for
     pure-attention checkpoints. `LiveTranslationTracker` drives it per
     keystroke (`live_translation_tracker.{h,cpp}`).
  2. *Paged-mode fork (dense models):* `fork_sequence(0, 1)` and decode
     previews on a throwaway seq 1 — the durable prefix on seq 0 is never
     disturbed (CoW pages). Still available as an alternative.
- **Hybrid SSM checkpoints (Qwen3.5) get neither**: recurrent state cannot be
  rewound or forked (`supports_cow_branching == false`), so every preview
  costs `reset_state()` + a full prefill. Mitigation is *policy*: keep the
  inline history window small (system prompt + last K committed turns, K ≈ 4–8)
  and expose older context through the `search_history` tool instead of the
  prompt. This caps preview prefill cost at a constant regardless of session
  length — and it is why the tool exists at all.

Preview turns are never appended to `durable_` — they would poison the
append-only prefix and force a reprefill on the *next* commit.

## 7. Yielding results: callbacks over futures/coroutines

**Recommendation: plain event callbacks + explicit marshaling, with the
generation id as the staleness token.** Rationale:

- The process already has three event loops (Win32 message pump, CaretTracker
  condvar queue, agent worker queue) and two established marshaling idioms
  (`OverlayWindow::PostUpdate` PostMessage; CaretTracker's pending-slot
  condvar). Callbacks compose with both; nothing new to build.
- `std::future`/`promise` is shaped wrong twice over: no streaming (previews
  are token-by-token) and no cancellation (previews are cancelled more often
  than they complete). `.get()` blocks, which is the one forbidden act.
- C++20 coroutines would read nicely (`co_await agent.translate(ctx)`) but
  require an executor/awaiter layer over the Win32 pump and the STA queue that
  the codebase doesn't have; that's a project in itself and buys ergonomics,
  not capability. If wanted later, it layers cleanly *on top of* the callback
  API without touching the agent.

Threading contract, stated once and enforced everywhere: **callbacks run on
the agent worker thread and must only enqueue** — PostMessage to the UI
thread for overlay updates, `RequestInject` to the STA thread for injection.
No callback ever touches a window, COM interface, or the engine.

## 8. Engine lifetime (TTL)

Per `docs/INFERENCE_API.md` §1.4, TTL-offload is application-layer. It lands
naturally in the agent worker: after each completed job, arm a deadline; if
the queue stays empty past `idle_ttl`, destroy the adapter
(`unique_ptr::reset()` frees all VRAM/pinned RAM). The next request finds
`State::Unloaded`, flips to `Loading` (DirectStorage reload, seconds), and the
overlay shows a "warming up" state via `on_state`. The durable *text* history
survives teardown — only the KV cache is lost, and the first request after a
reload pays one full prefill.

## 9. Open questions / future work

### Landed

- **Preview debounce** — implemented. A decode fires only after the user
  pauses: `SpotlightComposer` re-arms a `kDebounceMs = 130` timer on every edit
  (`spotlight_composer.{h,cpp}`), and the caret-driven path gates generation on
  the configurable typing-idle `idleTimerMs` (default 700, exposed in
  settings — `config.h`, `caret_tracker.cpp`). Restarting the timer on each
  edit coalesces a burst into a single generation, on top of the latest-wins
  coalescing.
- **Divergence-point reprefill** — implemented as the engine's "Continuous
  Speculative Tracking" primitive (see §6). `LiveTranslationTracker` drives
  `EnginePrefillCoordinator::update_sequence`, which diffs the fresh
  tokenization against the sequence's token mirror (LCP), rewinds the KV cache
  to the divergence point, and recomputes only the new suffix — the preview
  latency win on dense checkpoints, now live.

### Still open

- **Retro-editing committed text** (`revise_segment(id, new_text)` as an agent
  tool) — blocked on `TextInjector`, which can only replace text before the
  caret today. Park it.
- **Per-app profiles** (chat vs. code editor vs. email) — `app_hint` is in the
  schema; deciding whether it selects a different `AgentProfile` (new session,
  new KV prefix) or just rides in the user turn needs usage data.
