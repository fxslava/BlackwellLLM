# R&D Assessment — Strict 1-Step ReAct Loop for the TranslationAgent Prototype

Status: assessment for the first prototype (dense checkpoints only, no hybrid
SSM). Companion to `TRANSLATION_AGENT.md`; supersedes nothing.

Sources reviewed: `src/agent_orchestrator/{orchestrator,tool_parser,tool_registry}.{h,cpp}`,
`tests/agent_orchestrator/test_orchestrator.cpp`,
`src/tools/playground/blackwell_llm_adapter.h`.

---

## 1. Orchestrator readiness

### 1.1 The verdict on loop safety: solid

The "can a misbehaving model wedge us?" question has a clean answer: **no
input can make the loop spin or crash.** The three containment layers are all
implemented and all tested:

- `ToolParser::parse` is total — every `find()` is `npos`-checked before any
  `substr` (`tool_parser.cpp`), garbage degrades to `ActionKind::None`, and
  the `CrashProofOnGarbage` test sweeps 13 malformed inputs.
- `ToolRegistry::dispatch` is total — hallucinated names and throwing
  callbacks both become `"ERROR: ..."` observations
  (`UnknownToolDoesNotCrashAndModelRecovers`, `ThrowingToolIsContained`).
- The loop is bounded by `max_iterations` no matter what the model does, and
  `max_iterations <= 0` is clamped to 1 (`orchestrator.cpp:50`). A model that
  calls tools forever exits via the cap (`MaxIterationsSafeguard`); pure prose
  gets one nudge per iteration and still hits the cap.

`<think>` chain-of-thought handling is also production-grade: rehearsed tags
inside thoughts are inert, an unterminated `<think>` swallows the tail, and
all five scenarios are covered by tests.

### 1.2 Findings that DO affect a strict 1-step design

**F-1 — "tool_call then finish in the same completion" — now supported
(act-then-finish landed).** `ToolParser::parse` still returns only the
*earliest* valid action, but the orchestrator no longer stops there: after
dispatching a tool call it re-parses the remainder of the SAME assistant turn
(past `</tool_call>`) and, if a `<finish>` is the remainder's earliest action,
finishes cleanly in that one generation (`orchestrator.cpp:83-98`). So the
prompt shape "invoke the tool AND output `<finish>`" now completes in a single
iteration. A second `<tool_call>` ahead of the finish is treated as a protocol
violation — not dispatched, and the finish behind it is not trusted; the model
gets the observation and must produce a fresh turn. Options remain: (a)
**finish-only protocol** — no tool at all, translation rides in `<finish>`
(recommended, see §2); (b) **act-then-finish** — implemented (F-1b landed),
"one tool call, then finish" costs one generation; (c) `max_iterations = 2`
(tool → observation → finish), two generations, still available for
multi-turn tool use.

**F-2 — the cap-exit `answer` is a footgun at `max_iterations = 1`.**
When the loop exits on the cap, `RunResult::answer` is the *raw last
assistant turn* (`orchestrator.cpp:84-90`). If a preview-mode model
misbehaves and emits a tool call, that raw text — XML markup included —
becomes `answer`. The caller MUST gate on `r.finished` and apply a fallback
policy (§3) rather than injecting/overlaying `answer` blindly. `RunResult`
also doesn't record whether a tool was dispatched; if needed, scrape
`history()` for an `Observation` entry or capture via the tool callback.

**F-3 — history-seeding API — now supported (`seed_history` landed).**
`AgentOrchestrator::seed_history(std::vector<Message>)` injects the durable
committed-context history as prior USER/ASSISTANT turns before `run()`
(`orchestrator.cpp:39-46`). A seed that carries its own leading System turn
wins over the config prompt; otherwise `run()` inserts the config system
prompt at the FRONT so seeded turns always sit behind it. The prototype
workaround (inline the last K committed pairs into the user prompt) is no
longer necessary — seed the transcript directly.

**F-4 — no streaming or cancellation at the orchestrator seam.**
`ILLMGenerator::generate` is whole-string. This is already solved one layer
down: `BlackwellLLMAdapter::set_stream_callback` feeds the single-argument
`generate()` override, and returning `false` from the callback is a
cooperative stop. The service (§3) installs the callback before `run()` and
uses it for three things at once: staleness cancel (generation counter),
live preview streaming, and an **early hard stop when the accumulated text
contains `</finish>`** — the adapter's stop-string list is checkpoint-fixed
with no public setter, so the callback is the supported way to stop decode at
the protocol boundary and save every token after the close tag.

**F-5 — divergence-point reprefill — now landed (Continuous Speculative
Tracking).** The plain `blackwell_llm_adapter.h` path still reprefills from
position 0 on any divergence (KV reuse requires a strict extension), but the
live preview path no longer uses it: the engine grew a
`EnginePrefillCoordinator::begin_sequence / update_sequence / commit_sequence`
primitive that re-tokenizes the buffer, diffs it against the sequence's own
token mirror (LCP), rewinds the KV cache to the divergence point via
Copy-on-Write page rewind, and recomputes only the new suffix. In poc_overlay
`LiveTranslationTracker` drives it, so consecutive previews of "t" → "th" →
"the" cost only the new-suffix prefill instead of a from-scratch reprefill.
The predicted enhancement (legal on dense checkpoints — attention KV is
position-addressed and self-heals) is exactly what shipped.

**F-6 — pick a non-thinking dense checkpoint for preview.** The parser
*tolerates* `<think>` blocks, but a reasoning model will happily burn the
entire sub-second budget thinking before the first `<finish>` token. Model
choice / chat-template guard, not a code change.

### 1.3 Readiness summary

| Concern | Status |
|---|---|
| Infinite loop on misbehaving model | Safe — cap always honored, tested |
| Parser crash on garbage/truncation | Safe — total, tested |
| Hallucinated / throwing tools | Safe — error observations, tested |
| Tool call + finish in one completion | **Supported** — act-then-finish landed (F-1b) |
| Clean result at `max_iterations = 1` | **Caller must gate on `finished` (F-2)** |
| Context history seeding | **Supported** — `seed_history()` landed (F-3) |
| Streaming / cancel / early stop | Use adapter `StreamCallback` (F-4) |

**Conclusion: no orchestrator changes are required to ship the prototype
today** — provided the protocol is finish-only (§2). The two enhancements
once queued right after — the act-then-finish patch (F-1b) and `seed_history`
(F-3) — have both since landed (`orchestrator.cpp`), so the act-then-finish
and durable-history-seeding paths are now available as well.

---

## 2. The 1-step system prompt

### 2.1 Design decision: finish-only, no `translate_text` tool

A `translate_text` tool is information-free in a 1-step loop: the model *is*
the translator, so the tool call would only be a delivery envelope — and
`<finish>` already is the delivery envelope, parsed by the same
crash-proof parser, terminating the loop cleanly (`finished = true`) instead
of via the cap. The tool variant would also trip F-1. So: **the translation
rides in `<finish>` for both Preview and Commit** in the prototype.

What still validates the ReAct machinery: parse-and-terminate, the nudge
path, cap behavior, history threading — everything except a real tool
round-trip. To validate dispatch too, Commit mode gets an *optional* glossary
tool at `max_iterations = 2` (§2.3) as the controlled next notch, not today's
blocker.

### 2.2 Preview prompt (exact text)

Byte-stable per session; `{TARGET_LANG}` resolved once at service start.
~90 tokens — small enough that the per-preview full reprefill (F-5) stays
well inside the sub-second budget.

```text
You translate text into {TARGET_LANG}.
Reply with exactly one tag and nothing else:
<finish>TRANSLATION</finish>

Rules:
- Translate the user's message into {TARGET_LANG}.
- Keep numbers, names, code identifiers, URLs and emoji unchanged.
- Match the source's punctuation and casing style.
- If the text is already in {TARGET_LANG}, return it unchanged.

Example:
User: привет, как дела?
Assistant: <finish>hi, how are you?</finish>
```

Notes:
- The one-line example is load-bearing: it anchors both the tag protocol and
  the "no commentary" rule far more reliably than prose rules alone.
- No tool manifest at all — nothing for the model to be tempted by.
- Belt-and-braces (cheap insurance against F-2): register a `translate_text`
  tool anyway, whose callback returns
  `"Output the translation as <finish>TRANSLATION</finish> now."` — then a
  model that hallucinates the tool self-corrects if you ever raise the cap;
  at cap 1 the request is simply discarded by the `finished` gate.

### 2.3 Commit prompt (exact text, `max_iterations = 2`)

Same skeleton plus one optional tool round-trip — the first controlled
validation of a *real* dispatch in production:

```text
You are a translation agent. Translate the user's message into {TARGET_LANG}.

You may check ONE domain term first:
<tool_call name="glossary_lookup"><arg name="term">TERM</arg></tool_call>
The result arrives as an OBSERVATION message.

Then reply with exactly one tag and nothing else:
<finish>TRANSLATION</finish>

Rules:
- At most one tool call, and only if a term's established translation matters.
- If no lookup is needed, output <finish>TRANSLATION</finish> immediately.
- Keep numbers, names, code identifiers, URLs and emoji unchanged.

Example:
User: привет, как дела?
Assistant: <finish>hi, how are you?</finish>
```

Worst case: 2 generations (lookup + finish). Best case: identical to preview.
A model that calls the tool twice exits via the cap and the fallback policy
applies — still bounded, still safe.

---

## 3. Execution path — minimal wiring, buildable today

New pair of files in `src/tools/poc_overlay/`: `translation_service.{h,cpp}`.
It is a slimmed `ModelService` (playground pattern) fused with CaretTracker's
pending-slot coalescing. No changes to the orchestrator, parser, registry, or
adapter. One prerequisite: `blackwell_llm_adapter.{h,cpp}` must be linkable
from poc_overlay (move to a shared lib target, or add the source to the
poc_overlay target for the prototype).

### 3.1 The service

```cpp
// translation_service.h — owns the engine worker thread; all entry points O(1).
class TranslationService {
public:
    using PreviewCallback = std::function<void(uint64_t gen, std::wstring translation)>;

    TranslationService(std::string model_dir, std::string system_prompt_preview,
                       PreviewCallback on_preview);   // spawns worker; load is lazy

    // Latest-wins pending slot (CaretTracker pattern). Bumps the generation
    // counter, which in-flight decodes observe via the stream callback.
    void RequestPreview(std::wstring text_before_caret);

    // Commit path, called ON the CaretTracker STA thread from the existing
    // synchronous TransformCallback. Blocks up to `timeout` on a condvar;
    // nullopt on timeout / not-finished (caller falls back to the source).
    std::optional<std::wstring> TranslateBlocking(const std::wstring& source,
                                                  std::chrono::milliseconds timeout);

private:
    void ThreadMain();   // lazy-load adapter, then serve the pending slot
    // members: worker thread, mutex+condvar, pending slot,
    // std::atomic<uint64_t> current_gen_, unique_ptr<BlackwellLLMAdapter>,
    // ToolRegistry (translate_text decoy + glossary_lookup), prompt strings.
};
```

### 3.2 Worker body per request (the actual 1-step run)

```cpp
// UTF-16 -> UTF-8 (WideCharToMultiByte helpers; the whole agent stack is UTF-8)
std::string user = utf8(text);

// Streaming: staleness cancel + early stop at the protocol boundary.
std::string acc;
adapter_->set_stream_callback([&](const std::string& delta) {
    if (gen != current_gen_.load(std::memory_order_relaxed)) return false; // superseded
    acc += delta;
    return acc.find("</finish>") == std::string::npos;   // hard-stop after close tag
});

agent::orch::OrchestratorConfig cfg;
cfg.max_iterations = 1;                    // 2 for the commit prompt
cfg.system_prompt  = preview_prompt_;      // byte-stable per session
agent::orch::AgentOrchestrator orch(*adapter_, tools_, cfg);  // fresh per request (F-3)

agent::orch::RunResult r = orch.run(user);

// F-2 gate: never surface a cap-exit answer that contains protocol markup.
std::wstring out;
if (r.finished && !r.answer.empty())            out = utf16(r.answer);
else if (!r.finished && looks_like_prose(r.answer)) out = utf16(r.answer); // nudged prose: best effort
else                                            return;                    // discard (stale/garbled)

if (gen == current_gen_.load()) on_preview_(gen, std::move(out));
```

(`looks_like_prose` = `answer.find('<tool_call') == npos` — the minimal F-2
policy.)

### 3.3 Wiring in `main.cpp` (replaces the two mocks)

```cpp
TranslationService svc(modelDir, kPreviewPrompt,
    /*on_preview=*/[&overlay](uint64_t, std::wstring tr) {
        overlay.PostTranslation(std::move(tr));   // PostMessage-marshaled, like PostUpdate
    });

CaretTracker caretTracker(
    [&](const CaretUpdate& u) {
        if (u.caretFound) overlay.PostUpdate(u.text, u.caretScreenPos);
        if (u.wordBoundary && !u.text.empty())
            svc.RequestPreview(u.text);           // replaces LogMock(L"OnWordComplete")
    },
    /*commitTransform*/ [&svc](const std::wstring& src) -> std::wstring {
        auto tr = svc.TranslateBlocking(src, std::chrono::seconds(3));
        return tr.value_or(src);                  // timeout/garble -> inject source unchanged
    },
    injectionGuard);
```

Deliberate prototype shortcuts, acknowledged:
- **`TranslateBlocking` blocks the STA thread** (up to the timeout) so the
  `TransformCallback` signature and the whole TextInjector flow stay
  untouched. UIA polling freezes for the duration; acceptable for a prototype
  because the overlay is hidden during commit anyway. The two-phase async
  commit from TRANSLATION_AGENT.md §5 replaces this next.
- The overlay needs one small addition (`PostTranslation` or reusing
  `PostUpdate` with the translated string) to show preview text; the next
  trigger update overwrites it, which is fine.
- Context history is not yet fed in (F-3): every request is standalone.

### 3.4 Build checklist for today

1. Move/compile `blackwell_llm_adapter.{h,cpp}` into a target poc_overlay can
   link (`agent_orchestrator` + engine libs come with it).
2. Add `translation_service.{h,cpp}`; register the `translate_text` decoy and
   `glossary_lookup` (stub returning "no glossary entry" is enough to
   exercise dispatch).
3. Replace the two `LogMock` call sites in `main.cpp` as above; add the model
   directory to settings/args.
4. Smoke sequence: (a) unit-test the service against a `MockLLM`-style
   scripted `ILLMGenerator` (finish-only, prose-nudge, hallucinated-tool,
   stale-generation); (b) live run with a small dense checkpoint; (c) confirm
   the `</finish>` early-stop fires (decode ends within ~2 tokens of the tag).

---

## 4. Follow-ups

### Landed

1. ✅ **Act-then-finish orchestrator patch** (F-1b) — enables true
   "one generation, one tool, clean finish" (`orchestrator.cpp:83-98`).
2. ✅ **`seed_history()`** (F-3) — durable committed-context turns instead of
   inlining into the user prompt (`orchestrator.cpp:39-46`).
3. ✅ **Divergence-point reprefill** (F-5) — landed engine-side as the
   "Continuous Speculative Tracking" primitive
   (`EnginePrefillCoordinator::update_sequence`, driven by
   `LiveTranslationTracker` in poc_overlay): re-tokenizes the buffer, diffs it
   against the sequence's token mirror (LCP), rewinds the KV cache to the
   divergence point, and recomputes only the new suffix — the preview latency
   win on dense checkpoints.

### Still queued (explicitly NOT today)

4. Two-phase async commit (kill the STA-blocking shortcut).
5. `RunResult` enrichment: `tools_dispatched` count, so callers stop scraping
   `history()`.
