// TranslationService: the single owner of the CUDA inference stack inside the
// overlay app -- the library-shaped descendant of the playground's ModelService
// pattern (one worker thread owns the non-thread-safe engine; every public
// entry point is an O(1) enqueue).
//
// Since the LiveTranslationTracker pivot, this service does two things and no
// more: (1) the slow, one-time model load + JIT prompt-cache bootstrap, and
// (2) a thin, thread-safe forwarding shell around a LiveTranslationTracker,
// which owns ALL further engine interaction on its own dedicated thread. There
// is no AgentOrchestrator, no ToolRegistry, no ReAct loop anywhere in this
// class anymore -- see live_translation_tracker.h for why and how.
//
// Threads, and what each is allowed to do:
//
//   [UI thread]      hooks + overlay. Never calls into this class directly.
//   [STA thread]     CaretTracker. Calls TrackUpdate() / TriggerGeneration() /
//                    Cancel() (all fire-and-forget).
//   [load thread]    owned here (worker_). Constructs the BlackwellLLMAdapter
//                    (slow, DirectStorage) and JIT-bootstraps the prompt
//                    cache, THEN constructs the LiveTranslationTracker and
//                    exits -- its one job is the slow load, done off any
//                    latency-sensitive thread.
//   [tracker thread] owned by tracker_ (see live_translation_tracker.h). The
//                    ONLY thread that touches the adapter/engine once the
//                    tracker exists -- worker_ never does so again after
//                    handing off.
//
// TrackUpdate/TriggerGeneration/Cancel forward straight to tracker_ under
// mutex_ (which also guards previewSink_ and is never held across engine
// work, so it stays effectively uncontended). Results stream back through the
// PreviewSink ON THE TRACKER'S THREAD -- the sink must marshal (main.cpp
// forwards to CaretTracker::OnPreviewResult, which is a lock+notify enqueue).
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

class LiveTranslationTracker;

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
        // Target language NAME per translation direction (e.g. {"Russian",
        // "English"}); the model auto-detects the source. Each gets its own
        // system prompt + .bkv branch; the radix tree dedups the shared prefix.
        std::vector<std::string> targetLanguages{"English"};
        int activeLanguage = 0;  // initial index into targetLanguages
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

    enum class State { Loading, Ready, Error };

    // Runs ON THE TRACKER'S THREAD for every streamed delta (done=false,
    // growing text) and exactly once per TriggerGeneration with done=true --
    // final translation, or EMPTY text when the run failed / was superseded,
    // so the consumer's state machine can leave its loading state either way.
    // Must only enqueue/marshal -- never touch a window or COM directly.
    using PreviewSink =
        std::function<void(std::uint64_t gen, const std::wstring& text, bool done)>;

    explicit TranslationService(Settings settings);
    ~TranslationService();  // calls Shutdown()

    TranslationService(const TranslationService&) = delete;
    TranslationService& operator=(const TranslationService&) = delete;

    // Late-bound because the sink's target (CaretTracker) is constructed after
    // this service (the tracker's callbacks point back here). Thread-safe.
    void SetPreviewSink(PreviewSink sink);

    // Every keystroke: fire-and-forget speculative background prefill that
    // warms the engine's radix tree for `segment` (+ durable `context`), so a
    // subsequent TriggerGeneration() for the same text has (ideally) nothing
    // left to prefill. A no-op until the model has finished loading.
    void TrackUpdate(std::wstring segment, std::wstring inferenceContext);

    // Debounce fired: generate now. Streams through the bound PreviewSink.
    void TriggerGeneration(std::wstring segment, std::wstring inferenceContext);

    // Keystroke/reset interrupt: drop any pending work and cooperatively
    // abort an in-flight decode.
    void Cancel();

    // Switch the active translation direction (index into Settings::
    // targetLanguages). Thread-safe; remembered and applied to the tracker
    // even if it is called before the model finishes loading.
    void SetActiveLanguage(int index);

    // Join the worker (and, once constructed, the tracker's own thread).
    // Called explicitly from wWinMain BEFORE stack unwinding so the sink can
    // never fire into an already-destroyed CaretTracker (the tracker is
    // declared after -- and thus destroyed before -- this service).
    // Idempotent; the destructor calls it too.
    void Shutdown();

    State state() const { return state_.load(std::memory_order_relaxed); }

private:
    void ThreadMain();
    void LoadEngine();  // load thread: adapter construction (the slow call)

    // --- JIT prompt cache (load thread, inside LoadEngine) --------------------
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
    // The byte-exact, BPE-seam-safe prefix of `previewPrompt`'s serving prompt:
    // the chat-template-rendered system block, cut right after the end-of-turn
    // marker. Empty when no safe prefix could be derived. LiveTranslationTracker
    // reconstructs the SAME transcript shape at request time (see BuildTokens);
    // this is only the STATIC prefix used to seed one .bkv branch.
    std::string StableServingPrefix(const std::string& previewPrompt) const;
    void DeliverToSink(std::uint64_t gen, const std::wstring& text, bool done);

    Settings settings_;

    // One byte-stable system prompt per translation direction (KV prefix reuse);
    // built once in the constructor, shared by CompilePromptCache (JIT compile,
    // one .bkv branch each) and the tracker (live requests) so both address the
    // same radix-tree pages. Never empty (at least one entry).
    std::vector<std::string> previewPrompts_;
    std::atomic<int> activeLanguage_{0};  // remembered across the tracker handoff

    // Owned by the load thread until handed off; tracker_ then owns all
    // further engine access on its own thread. Both guarded by mutex_ so
    // TrackUpdate/TriggerGeneration/Cancel (called from CaretTracker's STA
    // thread) never race the handoff.
    std::unique_ptr<playground::BlackwellLLMAdapter> adapter_;
    std::unique_ptr<LiveTranslationTracker> tracker_;

    std::atomic<State> state_{State::Loading};
    std::atomic<std::uint64_t> requestSeq_{0};  // debug/log id surfaced to the sink
    std::atomic<bool> stop_{false};

    std::mutex mutex_;
    PreviewSink previewSink_;  // guarded by mutex_ (late-bound)
    std::thread worker_;       // the one-shot load thread
};
