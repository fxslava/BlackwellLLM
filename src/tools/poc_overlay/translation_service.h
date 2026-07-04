// TranslationService: the single owner of the CUDA inference stack inside the
// overlay app -- the library-shaped descendant of the playground's ModelService
// pattern (one worker thread owns the non-thread-safe engine; every public
// entry point is an O(1) enqueue).
//
// Since the stateful-UX redesign this service is PREVIEW-ONLY: the debounced
// idle timer in CaretTracker is the sole inference trigger, and Ctrl+Enter is
// a pure host-side TextInjector replacement of the already-produced preview --
// no ReAct loop, no engine call, no blocking on the commit path at all.
//
// Threads, and what each is allowed to do:
//
//   [UI thread]    hooks + overlay. Never calls into this class directly.
//   [STA thread]   CaretTracker. Calls RequestPreview() / CancelPending()
//                  (both fire-and-forget).
//   [agent worker] owned here. Constructs the BlackwellLLMAdapter (slow,
//                  DirectStorage), runs every AgentOrchestrator loop, and is
//                  the only thread that ever touches the engine.
//
// Preview requests coalesce latest-wins; a monotone generation counter makes
// stale work self-cancel through the adapter's cooperative StreamCallback.
// Results stream back through the PreviewSink ON THE WORKER THREAD -- the sink
// must marshal (main.cpp forwards to CaretTracker::OnPreviewResult, which is a
// lock+notify enqueue).
//
// ReAct budget (docs/REACT_1STEP_ASSESSMENT.md): the strict 1-iteration
// finish-only protocol. Committed context is injected with
// AgentOrchestrator::seed_history(), never inlined into the system prompt, so
// the prompt stays byte-stable for KV prefix reuse.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
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
        std::wstring modelDir;  // HuggingFace-layout checkpoint directory
        // Root of the per-model JIT prompt caches. The service derives
        // <root>\<model-hash-hex>\ from the loaded checkpoint, compiles the
        // .bkv prompt library there on first run, and warm-starts from it on
        // every run. Empty = %LOCALAPPDATA%\Blackwell\Cache.
        std::wstring promptCacheRoot;
        std::string targetLang = "English";
        size_t maxSeqLen = 4096;
        float temperature = 0.2f;  // low: translation wants determinism
        float topP = 0.9f;
        int previewMaxNewTokens = 192;  // a capture area is one clause/sentence

        // Tiered KV prefix-cache budget (engine RuntimeConfig knobs; one block =
        // one 16-token KV page across all layers). Mirrors Config's fields.
        int vramCacheBlocks = 1024;
        int ramTierBlocks = 2048;
        bool diskSpillEnabled = true;
        int diskSpillBlocks = 8192;
        std::wstring spillFilePath;  // required when diskSpillEnabled
    };

    enum class State { Loading, Ready, Generating, Error };

    // Runs ON THE WORKER THREAD for every streamed delta (done=false, growing
    // text) and exactly once per request with done=true -- final translation,
    // or EMPTY text when the run failed / was superseded, so the consumer's
    // state machine can leave its loading state either way. Must only
    // enqueue/marshal -- never touch a window or COM directly.
    using PreviewSink =
        std::function<void(std::uint64_t gen, const std::wstring& text, bool done)>;

    explicit TranslationService(Settings settings);
    ~TranslationService();  // calls Shutdown()

    TranslationService(const TranslationService&) = delete;
    TranslationService& operator=(const TranslationService&) = delete;

    // Late-bound because the sink's target (CaretTracker) is constructed after
    // this service (the tracker's callbacks point back here). Thread-safe.
    void SetPreviewSink(PreviewSink sink);

    // Debounce expired: translate `segment`. Coalescing latest-wins, O(1), any
    // thread. Bumps the generation counter, which cooperatively aborts any
    // in-flight decode. `inferenceContext` is CaretTracker's accumulated
    // ORIGINAL (pre-translation) committed text, seeded into the orchestrator
    // as a durable context turn.
    void RequestPreview(std::wstring segment, std::wstring inferenceContext);

    // Keystroke/reset interrupt: drop any queued preview and cooperatively
    // cancel the in-flight decode.
    void CancelPending();

    // Join the worker. Called explicitly from wWinMain BEFORE stack unwinding
    // so the sink can never fire into an already-destroyed CaretTracker (the
    // tracker is declared after -- and thus destroyed before -- this service).
    // Idempotent; the destructor calls it too.
    void Shutdown();

    State state() const { return state_.load(std::memory_order_relaxed); }

private:
    struct Job {
        std::wstring segment;
        std::wstring context;
        std::uint64_t gen = 0;
    };

    void ThreadMain();
    void LoadEngine();  // worker: adapter construction (the slow call)

    // --- JIT prompt cache (worker thread, inside LoadEngine) ------------------
    // Because the user picks the checkpoint at runtime, .bkv files cannot ship
    // with the app: KV page geometry and token ids are model-specific. Instead
    // the service self-bootstraps: resolve <promptCacheRoot>\<model-hash>\,
    // compile the WarmupSpec there on first run (CompilePromptCache), then
    // cold-load it (warm_start, LoadPolicy::ColdRam). All three are no-ops /
    // logged fallbacks when the loaded model has no prefix-cache substrate.
    void WarmStartPromptCache();
    // Compile the in-C++ WarmupSpec (the rendered system-prompt prefix) into
    // <dir>. Returns false when there was nothing cacheable (e.g. the stable
    // prefix is shorter than one KV page). Throws on engine/serializer errors.
    bool CompilePromptCache(const std::wstring& dir);
    // The byte-exact, BPE-seam-safe prefix of every serving prompt: the
    // chat-template-rendered system block, cut right after the end-of-turn
    // marker. Empty when no safe prefix could be derived.
    std::string StableServingPrefix() const;
    // Worker: one strict 1-step orchestrator run. Returns the translation on a
    // clean finish (or salvageable nudged prose), nullopt otherwise.
    std::optional<std::wstring> RunTranslation(const Job& job);
    // Worker (inside the stream callback): extract the text between <finish>
    // and the (possibly still streaming) </finish> from `acc` and push it to
    // the sink if it changed.
    void StreamPreviewPartial(const std::string& acc, std::uint64_t gen,
                              std::wstring& lastPosted);
    void DeliverToSink(std::uint64_t gen, const std::wstring& text, bool done);

    Settings settings_;

    // Byte-stable per session (KV prefix reuse); built once in the constructor.
    std::string previewPrompt_;

    // Worker-owned after construction.
    std::unique_ptr<playground::BlackwellLLMAdapter> adapter_;
    agent::orch::ToolRegistry tools_;

    std::atomic<State> state_{State::Loading};
    std::atomic<std::uint64_t> currentGen_{0};
    std::atomic<bool> stop_{false};

    std::mutex mutex_;
    std::condition_variable cv_;
    PreviewSink previewSink_;            // guarded by mutex_ (late-bound)
    std::optional<Job> pendingPreview_;  // latest-wins slot
    std::thread worker_;
};
