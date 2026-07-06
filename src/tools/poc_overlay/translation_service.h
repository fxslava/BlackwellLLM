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

#include "live_translation_tracker.h"  // LiveTranslationTracker::LifecycleEvent

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

        // Deferred launch: construct the service (instant) but do NOT load the
        // model weights into VRAM until EnsureLoaded() -- the "start inactive"
        // startup mode. false keeps the historical load-immediately behavior.
        bool deferLoad = false;
    };

    // Idle exists only for deferLoad: the C++ backend is alive but no load has
    // been requested yet (no VRAM touched). EnsureLoaded() moves Idle->Loading.
    enum class State { Idle, Loading, Ready, Error };

    // Re-exported from the tracker so main.cpp binds one HUD sink without
    // reaching into the tracker type itself.
    using LifecycleEvent = LiveTranslationTracker::LifecycleEvent;
    using LifecycleSink = std::function<void(LifecycleEvent event)>;

    // Runs ON THE TRACKER'S THREAD for every streamed delta (done=false,
    // growing text) and exactly once per TriggerGeneration with done=true --
    // final translation, or EMPTY text when the run failed / was superseded,
    // so the consumer's state machine can leave its loading state either way.
    // `tokens` is the per-token confidence heatmap for `text`, non-empty only
    // on the final delivery in Developer Mode (see LiveTranslationTracker's
    // StreamCallback). Must only enqueue/marshal -- never touch a window or COM.
    using PreviewSink = std::function<void(std::uint64_t gen, const std::wstring& text,
                                           const TokenHeatmap& tokens, bool done)>;

    explicit TranslationService(Settings settings);
    ~TranslationService();  // calls Shutdown()

    TranslationService(const TranslationService&) = delete;
    TranslationService& operator=(const TranslationService&) = delete;

    // Late-bound because the sink's target (CaretTracker) is constructed after
    // this service (the tracker's callbacks point back here). Thread-safe.
    void SetPreviewSink(PreviewSink sink);

    // Late-bound lifecycle observer (HUD banners: hibernated / "Waking up...").
    // Fires on the tracker's worker thread -- enqueue/marshal only. Thread-safe.
    void SetLifecycleSink(LifecycleSink sink);

    // Kick the deferred engine load (no-op once a load has ever started, so
    // it is safe to call on every activation toggle). Instant: the slow work
    // runs on the service's own load thread exactly as the eager path does.
    void EnsureLoaded();

    // The dual-stage inactivity state machine, driven by a coarse UI-thread
    // timer. Compares the time since the last typing/selection activity
    // (TrackUpdate / TriggerGeneration) against the two thresholds (SECONDS)
    // and asks the tracker's worker to run the due stage: kvSpillTimeoutSec ->
    // spill the KV prefix cache to disk; hibernateTimeoutSec -> soft-hibernate
    // (weights VRAM -> host RAM; the engine object survives). Each stage
    // fires at most once per idle period; any new activity re-arms both.
    // Cheap and non-blocking -- safe to call every few seconds.
    void LifecycleTick(int kvSpillTimeoutSec, int hibernateTimeoutSec);

    // Hot-reload the translation directions (Settings saved a new pair set --
    // NO app restart). Rebuilds the per-direction system prompts and swaps
    // them into the live tracker: the very next typing/selection request
    // addresses the new set. New directions simply prefill cold on first use
    // and get committed to the radix tree like any other prompt; the explicit
    // JIT pre-cache below is optional. UI thread; cheap.
    void UpdateLanguagePairs(std::vector<std::string> targetLanguages, int activeLanguage);

    // On-demand JIT pre-cache: compile the .bkv prompt-cache branch for ONE
    // direction (looked up by its target-language name in the CURRENT pair
    // set) on the tracker's engine-owning worker thread. `done(ok, tokens)`
    // fires there -- marshal, don't touch UI. Fails fast (done(false, 0),
    // called synchronously) when the engine isn't Ready or the target isn't in
    // the current set (e.g. the user hasn't saved the pair yet).
    void PrecacheLanguage(const std::string& targetLanguage,
                          std::function<void(bool ok, int prefilledTokens)> done);

    // Weight-load progress, 0..100, while state() == Loading (0 before the
    // loader reaches the weights; 100 once they are resident). The readiness
    // poll renders it into the "Initializing... (N%)" HUD.
    int LoadProgressPercent() const {
        return loadProgressPct_.load(std::memory_order_relaxed);
    }

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

    // Developer Mode: enable per-token probability collection for the heatmap.
    // Thread-safe; remembered across the tracker handoff and applied live.
    void SetDeveloperMode(bool enabled);

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
    // Per-branch variant for the on-demand pre-cache: compile ONLY the branch
    // at `index` into the shared cache dir and MERGE the manifest (the AOT
    // warmer rewrites it with just the compiled entries, which would strand
    // the other branches on the next warm start). Runs on the tracker's
    // worker thread; returns false when nothing was cacheable.
    bool CompilePromptCacheBranch(int index, const std::wstring& dir, int* prefilledTokens);
    // <promptCacheRoot>\<model-hash-hex> -- requires a live engine WITH a
    // prefix-cache substrate (empty otherwise). Engine-owning thread only.
    std::wstring ResolvePromptCacheDir() const;
    // The byte-exact, BPE-seam-safe prefix of `previewPrompt`'s serving prompt:
    // the chat-template-rendered system block, cut right after the end-of-turn
    // marker. Empty when no safe prefix could be derived. LiveTranslationTracker
    // reconstructs the SAME transcript shape at request time (see BuildTokens);
    // this is only the STATIC prefix used to seed one .bkv branch.
    std::string StableServingPrefix(const std::string& previewPrompt) const;
    void DeliverToSink(std::uint64_t gen, const std::wstring& text,
                       const TokenHeatmap& tokens, bool done);
    void DeliverLifecycle(LifecycleEvent event);  // tracker worker -> bound sink
    void MarkActivity();  // refresh lastActivityMs_ and re-arm the stage ladder

    Settings settings_;  // targetLanguages hot-reloads under mutex_; rest immutable

    // One byte-stable system prompt per translation direction (KV prefix reuse);
    // shared by the JIT compile paths (one .bkv branch each) and the tracker
    // (live requests) so both address the same radix-tree pages. Never empty.
    // Hot-reloaded by UpdateLanguagePairs -- guarded by mutex_ (readers on the
    // load/worker threads snapshot it under the lock).
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
    std::atomic<bool> developerMode_{false};  // remembered across the tracker handoff

    // Inactivity lifecycle bookkeeping. lastActivityMs_ is a steady-clock
    // millisecond stamp refreshed by TrackUpdate/TriggerGeneration; stage_
    // records how far down the ladder this idle period has gone (0 = engaged,
    // 1 = KV spilled, 2 = hibernated) so LifecycleTick fires each stage once.
    std::atomic<std::int64_t> lastActivityMs_{0};
    std::atomic<int> lifecycleStage_{0};
    std::atomic<int> loadProgressPct_{0};  // weight-load progress (engine callback)
    bool loadStarted_ = false;  // ctor (eager) or EnsureLoaded; UI thread only

    std::mutex mutex_;
    PreviewSink previewSink_;      // guarded by mutex_ (late-bound)
    LifecycleSink lifecycleSink_;  // guarded by mutex_ (late-bound)
    std::thread worker_;           // the one-shot load thread
};
