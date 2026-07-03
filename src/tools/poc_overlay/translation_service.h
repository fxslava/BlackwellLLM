// TranslationService: the single owner of the CUDA inference stack inside the
// overlay app -- the library-shaped descendant of the playground's ModelService
// pattern (one worker thread owns the non-thread-safe engine; every public
// entry point is an O(1) enqueue).
//
// Threads, and what each is allowed to do:
//
//   [UI thread]    hooks + overlay. Never calls into this class directly.
//   [STA thread]   CaretTracker. Calls RequestPreview() (fire-and-forget) and
//                  TranslateBlocking() (bounded condvar wait -- the ONLY place
//                  the STA thread ever blocks, capped by the caller's timeout).
//   [agent worker] owned here. Constructs the BlackwellLLMAdapter (slow,
//                  DirectStorage), runs every AgentOrchestrator loop, and is
//                  the only thread that ever touches the engine.
//
// Preview requests coalesce latest-wins (CaretTracker's own pending-slot
// idiom); a monotone generation counter makes stale work self-cancel through
// the adapter's cooperative StreamCallback. Preview text streams back through
// PreviewSink ON THE WORKER THREAD -- the sink must marshal (the main.cpp
// wiring forwards to OverlayWindow::PostTranslation, which is PostMessage-
// based and therefore safe from any thread).
//
// ReAct budgets (docs/REACT_1STEP_ASSESSMENT.md): Preview runs the strict
// 1-iteration finish-only protocol; Commit allows 2 iterations so the model
// may take one glossary_lookup round-trip -- or settle in a single generation
// via the orchestrator's act-then-finish path. Committed context is injected
// with AgentOrchestrator::seed_history(), never inlined into the system
// prompt, so the prompt stays byte-stable for KV prefix reuse.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include "tool_registry.h"  // agent::orch::ToolRegistry (value member)

namespace playground {
class BlackwellLLMAdapter;
}

class TranslationService {
public:
    struct Settings {
        std::wstring modelDir;        // HuggingFace-layout checkpoint directory
        std::wstring promptCacheDir;  // AOT-compiled .bkv directory (manifest.json +
                                      // system_prompt.bkv); empty = skip warm start
        std::string targetLang = "English";
        size_t maxSeqLen = 4096;
        float temperature = 0.2f;  // low: translation wants determinism
        float topP = 0.9f;
        int previewMaxNewTokens = 160;  // preview is one short segment
        int commitMaxNewTokens = 512;
    };

    enum class State { Loading, Ready, Generating, Error };

    // Runs ON THE WORKER THREAD for every preview delta and once with
    // done=true for the final text. `gen` identifies the request so a sink
    // that buffers can drop stale deliveries itself (the service already
    // suppresses most of them). Must only enqueue/marshal -- never touch a
    // window or COM directly.
    using PreviewSink =
        std::function<void(std::uint64_t gen, const std::wstring& text, bool done)>;

    TranslationService(Settings settings, PreviewSink previewSink);
    ~TranslationService();

    TranslationService(const TranslationService&) = delete;
    TranslationService& operator=(const TranslationService&) = delete;

    // Word-boundary preview: coalescing latest-wins, O(1), any thread. Bumps
    // the generation counter, which cooperatively aborts any in-flight decode.
    // `inferenceContext` is CaretTracker's accumulated ORIGINAL (pre-
    // translation) committed text; it is seeded into the orchestrator as a
    // durable context turn.
    void RequestPreview(std::wstring segment, std::wstring inferenceContext);

    // Ctrl+Enter commit: translate `segment` with the 2-iteration budget and
    // block (bounded) for the result. Called on the CaretTracker STA thread
    // from inside the commit TransformCallback; the overlay is already hidden
    // there, which is what makes the (deliberate, prototype-scoped) STA wait
    // acceptable. Returns nullopt on timeout / engine error / a run that did
    // not end in a clean <finish> -- the caller then injects the source
    // unchanged rather than garbling the user's text.
    std::optional<std::wstring> TranslateBlocking(const std::wstring& segment,
                                                  const std::wstring& inferenceContext,
                                                  std::chrono::milliseconds timeout);

    // Drop any queued preview and cancel in-flight decode (e.g. the hook's
    // onReset: focus moved, overlay hidden -- the result could never be shown).
    void CancelPending();

    State state() const { return state_.load(std::memory_order_relaxed); }

private:
    enum class JobKind { Preview, Commit };
    struct Job {
        JobKind kind = JobKind::Preview;
        std::wstring segment;
        std::wstring context;
        std::uint64_t gen = 0;
        // Commit only: fulfilled exactly once with the translation (or nullopt).
        std::shared_ptr<std::promise<std::optional<std::wstring>>> reply;
    };

    void ThreadMain();
    void LoadEngine();            // worker: adapter construction (the slow call)
    void WarmStartPromptCache();  // worker: .bkv cold-load, see .cpp for gating
    // Worker: one full orchestrator run for `job`. Returns the translation on a
    // clean finish (or salvageable preview prose), nullopt otherwise.
    std::optional<std::wstring> RunTranslation(const Job& job);
    // Worker (inside the stream callback): extract the text between <finish>
    // and the (possibly still streaming) </finish> from `acc` and push it to
    // the sink if it changed.
    void StreamPreviewPartial(const std::string& acc, std::uint64_t gen,
                              std::wstring& lastPosted);

    Settings settings_;
    PreviewSink previewSink_;

    // Byte-stable per session (KV prefix reuse); built once in the constructor.
    std::string previewPrompt_;
    std::string commitPrompt_;

    // Worker-owned after construction: the engine stack and the tool registry
    // (glossary_lookup + the translate_text decoy).
    std::unique_ptr<playground::BlackwellLLMAdapter> adapter_;
    agent::orch::ToolRegistry tools_;

    std::atomic<State> state_{State::Loading};
    std::atomic<std::uint64_t> currentGen_{0};
    std::atomic<bool> stop_{false};

    std::mutex mutex_;
    std::condition_variable cv_;
    std::optional<Job> pendingPreview_;  // latest-wins slot
    std::optional<Job> pendingCommit_;   // at most one; served before previews
    std::thread worker_;
};
