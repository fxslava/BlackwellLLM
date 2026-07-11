// LiveTranslationTracker: the zero-overhead "spinal cord" for live preview
// translation. It talks DIRECTLY to BlackwellLLMAdapter / BlackwellEngine --
// NO AgentOrchestrator, NO ToolRegistry, NO ReAct loop, NO tool parsing.
//
// It rides the engine's "Continuous Speculative Tracking" primitive
// (EnginePrefillCoordinator::begin_sequence / update_sequence /
// commit_sequence -- see engine_prefill_coordinator.h) rather than issuing a
// fresh prefill_prompt() per keystroke: ONE EngineSequence persists for this
// object's entire lifetime. BPE tokenizers re-segment as characters arrive
// ("t" -> "th" -> "the" are three different token streams), so a live tracker
// cannot just append -- update_sequence diffs the fresh tokenization against
// the sequence's own token mirror, truncates (rewinds) the KV cache to the
// divergence point, and recomputes only the new suffix. Critically,
// update_sequence does NOT publish to the radix tree -- keystroke states are
// transient, and committing every partial word would fill the tree with pages
// superseded milliseconds later. commit_sequence() is called ONLY at the one
// genuine stable boundary this class has: a debounce fire (TriggerGeneration).
//
// Two entry points, mapped 1:1 onto the two things a keystroke-driven UI does:
//
//   TrackUpdate(text, context)            every keystroke -- fire-and-forget,
//                                          reconciles the tracking session,
//                                          never commits.
//   TriggerGeneration(text, context, cb)  debounce fired -- reconciles once
//                                          more, COMMITS, then decodes,
//                                          streaming `cb`.
//
// Both build the IDENTICAL token sequence for the same (text, context) pair
// (see BuildTokens): the exact chat-template-rendered, add_special_tokens =
// false transcript the JIT-compiled .bkv system prompt was ALSO compiled
// against (see translation_service.cpp's StableServingPrefix). That triple
// alignment -- JIT cache, TrackUpdate, TriggerGeneration -- is what makes the
// radix tree a real cache instead of three parties talking past each other:
// begin_sequence's initial commit hits the warm-started .bkv system-prompt
// pages on the very first keystroke of a session (a free, deduped commit),
// and the user-text tail gets progressively reconciled -- never re-computed
// from scratch -- as the user keeps typing.
//
// Threading: owns ONE dedicated worker thread -- the sole thread permitted to
// touch the adapter/engine for the lifetime of this object (the engine's
// control plane is single-threaded, like the whole paging substrate). The
// persistent EngineSequence is worker-thread-only state; it is released in
// the destructor, AFTER the worker thread has been joined (so no concurrent
// engine access is possible at that point, even though the destructor itself
// runs on a different, arbitrary calling thread). TrackUpdate() /
// TriggerGeneration() / Cancel() are all fire-and-forget, callable from any
// thread (the CaretTracker STA thread in practice): O(1) enqueue + condvar
// notify, never blocking on engine work.
//
// Interrupt model: one monotone generation counter. Every TrackUpdate and
// TriggerGeneration call bumps it and replaces the single pending-job slot
// (latest wins -- Track and Generate share the slot, since a debounce firing
// really does supersede a not-yet-started speculative reconcile, and a fresh
// keystroke really does supersede a firing that hasn't started yet). The
// worker checks the counter at every decode-loop iteration and abandons stale
// work immediately: a superseded TrackUpdate never reaches its reconcile call
// at all; a superseded TriggerGeneration's decode loop hard-stops on the next
// token boundary and delivers no further callback data (the interrupting call
// owns the next `done` delivery). The tracking session itself is left exactly
// as the interrupted call left it -- always internally consistent (the
// coordinator's own contract) -- and self-heals on the next reconcile
// regardless of how large the resulting diff turns out to be.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "token_info.h"  // TokenInfo / TokenHeatmap

namespace playground {
class BlackwellLLMAdapter;
}
namespace blackwell {
class HybridSnapshotRing;
}

class LiveTranslationTracker {
public:
    // Runs on THIS CLASS'S WORKER THREAD. `done=false` carries a growing
    // partial translation as it streams from the model; `done=true` marks the
    // end (text empty on failure -- no <finish> ever closed, or the model's
    // real eos fired before it did). `tokens` is the per-token confidence
    // heatmap for `text`, populated ONLY on the final delivery and ONLY when
    // probability collection is enabled (Developer Mode); it is empty for
    // partials and in the normal path. When non-empty, concatenating
    // tokens[].text reproduces `text` exactly, so the overlay can align the
    // heatmap. The receiver must only enqueue/marshal -- never touch a window
    // or COM directly.
    using StreamCallback =
        std::function<void(const std::wstring& text, const TokenHeatmap& tokens, bool done)>;

    // The dual-stage inactivity lifecycle, as observed from the worker thread:
    //   KvSpilled  -- stage 1 done: the KV prefix cache was demoted to disk.
    //   Hibernated -- stage 2 done: model weights left VRAM for host RAM.
    //   WakingUp   -- a job arrived while hibernated; the PCIe DMA restore is
    //                 about to run (show the "Waking up..." HUD now).
    //   Awake      -- the restore finished; the job is about to be served.
    // Fired on THIS CLASS'S WORKER THREAD -- enqueue/marshal only, like the
    // StreamCallback.
    enum class LifecycleEvent { KvSpilled, Hibernated, WakingUp, Awake };
    using LifecycleSink = std::function<void(LifecycleEvent event)>;

    // `adapter` must outlive this object and must already own a live, Ready
    // engine (this class never constructs/tears down the engine itself --
    // that stays TranslationService's job). `system_prompts` holds one exact,
    // already-rendered finish-only protocol text PER translation direction (see
    // translation_service.cpp BuildPreviewPrompt) -- each byte-identical to the
    // matching .bkv branch the JIT cache compiled, so they share radix-tree
    // pages. `active_language` indexes it; SetActiveLanguage() switches live.
    // `temperature`/`top_p` govern the DECODE steps only; the prompt phase
    // (reconcile + first-token sample) is always deterministic.
    LiveTranslationTracker(playground::BlackwellLLMAdapter& adapter,
                           std::vector<std::string> system_prompts, int active_language,
                           int max_new_tokens, float temperature, float top_p);
    ~LiveTranslationTracker();

    LiveTranslationTracker(const LiveTranslationTracker&) = delete;
    LiveTranslationTracker& operator=(const LiveTranslationTracker&) = delete;

    // Keystroke event. Cancels any in-flight generation (via the generation-
    // counter interrupt) and enqueues a SPECULATIVE reconcile of the tracking
    // session against `current_text` (+ durable `context`) -- diff, truncate,
    // delta-compute, NO tree commit, no decode, no callback: purely a radix-
    // tree-adjacent KV warm-up. Coalesces latest-wins: a burst of keystrokes
    // costs at most one reconcile (the latest). A no-op (logged once) on
    // models with no prefix-cache substrate -- there is no tracking session
    // to speak of.
    void TrackUpdate(std::wstring current_text, std::wstring context);

    // Debounce fired. Reconciles the session once more (cheap if TrackUpdate
    // already did the work), COMMITS it to the radix tree -- the one genuine
    // stable boundary this class has -- then decodes, streaming through
    // `callback` on the worker thread until <finish> closes, the token budget
    // is hit, or a newer TrackUpdate/TriggerGeneration interrupts it. Falls
    // back to the adapter's own generate() on models with no prefix-cache
    // substrate (still correct, just without the speculative-tracking win).
    void TriggerGeneration(std::wstring current_text, std::wstring context,
                           StreamCallback callback);

    // Cancel any pending/in-flight work without starting new work (e.g. focus
    // lost, field cleared). Cheap; safe to call redundantly. Does NOT tear
    // down the persistent tracking session -- the next TrackUpdate/
    // TriggerGeneration reconciles it to whatever text follows regardless of
    // how different it is from what came before.
    void Cancel();

    // Virtual rewind (hybrid SSM models only): ask an IN-FLIGHT hybrid generation
    // to roll its decode head back `tokens_back` generated tokens and resume from
    // the nearest snapshot, streaming the corrected continuation on the SAME job.
    // Fire-and-forget, callable from any thread (a critic/UI action). Rides a
    // dedicated latest-wins signal, NOT the pending_ slot: a backtrack MUTATES the
    // running generation rather than superseding it, so it must not displace the
    // job the way a fresh TrackUpdate/TriggerGeneration does. A no-op when no
    // hybrid generation is decoding, when the model is dense (dense uses the
    // coordinator's physical micro-rewind, not this), or when the target predates
    // the snapshot horizon (the decode simply keeps streaming). tokens_back <= 0
    // is ignored.
    void RequestVirtualRewind(int tokens_back);

    // Switch the active translation direction (index into the CURRENT prompt
    // set). Thread-safe (atomic); out-of-range indices are clamped. The next
    // request builds tokens from the new prompt -- update_sequence simply
    // reconciles the (now larger) diff, so no explicit session reset is needed.
    void SetActiveLanguage(int index);

    // Developer Mode toggle: when true, the decode loop records each sampled
    // token's probability (a full-vocab logits read per token) and the final
    // delivery carries the per-token heatmap. Off by default -- purely a debug
    // cost. Thread-safe (atomic); the next generation observes the new value.
    void SetCollectProbabilities(bool enabled) {
        collect_probs_.store(enabled, std::memory_order_relaxed);
    }

    // Hot-reload the prompt set (language pairs edited in Settings -- NO app
    // restart). Thread-safe: the prompts live behind a shared_ptr snapshot
    // that each job copies once at its start, so an in-flight decode keeps the
    // set it started with and the very next job sees the new one. The
    // persistent tracking session is deliberately left alone -- its next
    // reconcile diffs against the new prompt tokens and self-heals (the shared
    // "You translate text into " prefix keeps even that diff small). An empty
    // vector is ignored (there is always at least one direction).
    void UpdatePrompts(std::vector<std::string> prompts, int active);

    // Run an arbitrary engine-touching task on THIS CLASS'S WORKER THREAD --
    // the only thread allowed to drive the engine (used for the on-demand JIT
    // prompt pre-cache). Tasks queue FIFO behind any pending translation job,
    // are NEVER displaced by activity (unlike lifecycle requests), and run
    // with the engine awake (a hibernated engine is restored first). The task
    // must not throw.
    void PostEngineTask(std::function<void()> task);

    // Late-bound observer for the inactivity lifecycle (TranslationService
    // wires it to the HUD). Thread-safe; pass nullptr to detach.
    void SetLifecycleSink(LifecycleSink sink);

    // Inactivity lifecycle, requested from any thread (the UI-thread timer in
    // practice) and EXECUTED on the worker thread -- the only thread allowed
    // to touch the engine. Both first release the persistent tracking session
    // (its pinned pages would otherwise be excluded from the spill; it
    // self-heals on the next reconcile) and spill the KV prefix cache to
    // disk; RequestHibernate additionally offloads the model weights from GPU
    // VRAM to pinned host RAM (BlackwellEngine::hibernate -- the engine object
    // survives). A queued translation job always wins: new activity clears a
    // not-yet-executed lifecycle request, and a hibernated engine is woken
    // (with WakingUp/Awake events bracketing the PCIe DMA restore) before any
    // job runs.
    void RequestKvSpill();
    void RequestHibernate();

private:
    enum class JobKind { Track, Generate };
    enum class LifecycleOp { None, Spill, Hibernate };
    struct Job {
        JobKind kind = JobKind::Track;
        std::wstring text;
        std::wstring context;
        std::uint64_t gen = 0;
        StreamCallback callback;  // Generate only
    };

    // Opaque holder for blackwell::EnginePrefillCoordinator::EngineSequence --
    // defined in the .cpp so this header stays free of engine/CUDA includes
    // (the coordinator's own convention; see BlackwellEngine::Impl's pImpl).
    struct TrackedSession;

    void ThreadMain();
    void RunTrack(const Job& job);
    void RunGenerate(const Job& job);
    // Hybrid-SSM decode with snapshot-driven virtual rewind. Selected instead of
    // the coordinator path (which needs a prefix-cache substrate hybrids lack)
    // and instead of the plain fallback when the model forks and has branch
    // headroom (HybridSnapshotRing::supported). Reprefills the whole prompt on
    // slot 0 (recurrent state has no rewind), then decodes while checkpointing
    // every K tokens, honoring RequestVirtualRewind() in-loop.
    void RunGenerateHybridSnapshot(const Job& job);
    // Worker thread: execute a Spill/Hibernate request (release session ->
    // spill -> optionally offload weights) and fire the lifecycle sink.
    void RunLifecycle(LifecycleOp op);
    // Worker thread: if the engine is hibernated, restore it (WakingUp/Awake
    // events bracket the DMA) before the caller touches forward()/prefill.
    void WakeIfHibernated();
    void EmitLifecycle(LifecycleEvent event);
    // Fallback decode for models without a prefix-cache substrate: routes
    // through the adapter's own generate(), forfeiting the speculative-
    // tracking advantage but keeping correctness for every model the user can
    // load.
    void RunGenerateFallback(const Job& job);
    // Build the exact role-tagged transcript -> chat-template -> tokens that
    // BOTH RunTrack and RunGenerate use, so a speculative reconcile and the
    // generation that follows it always address the identical token sequence
    // (and the same sequence the JIT cache compiler produced for the system
    // prompt prefix).
    std::vector<int> BuildTokens(const std::wstring& text, const std::wstring& context) const;
    // The system prompt for the currently-active direction: one shared_ptr
    // snapshot + clamped index, returned BY VALUE so a concurrent
    // UpdatePrompts() can never invalidate the caller's string (the prompt is
    // a few hundred bytes; the copy is noise next to a forward pass).
    std::string ActivePrompt() const;
    // Stream the text between <finish> and the (possibly still-streaming)
    // </finish> to `callback` if it changed since `last_posted`.
    void StreamPartial(const std::string& acc, const StreamCallback& callback,
                       std::wstring& last_posted) const;
    // Build the final translation heatmap: intersect the decoded token pieces
    // (raw UTF-8 offsets in `acc`, parallel to `probs`) with the <finish> body
    // window, emitting one TokenInfo per piece that lands inside it. The
    // concatenation of the result's text equals ExtractFinishBody(acc) so the
    // overlay's per-token measuring stays aligned. Empty if collection was off.
    TokenHeatmap BuildHeatmap(const std::string& acc,
                              const std::vector<std::pair<size_t, float>>& pieceEnds) const;

    playground::BlackwellLLMAdapter& adapter_;
    // One system prompt per translation direction, behind an immutable
    // snapshot: readers copy the shared_ptr under mutex_ and keep using their
    // copy lock-free; UpdatePrompts() swaps in a whole new vector. Never null,
    // never empty (ctor and UpdatePrompts both guarantee it).
    std::shared_ptr<const std::vector<std::string>> system_prompts_;  // guarded by mutex_
    std::atomic<int> active_language_{0};          // index into the current prompt set
    int max_new_tokens_;
    float temperature_;
    float top_p_;
    std::atomic<bool> logged_fallback_{false};  // one-time note, not per-call spam
    std::atomic<bool> collect_probs_{false};    // Developer Mode heatmap collection

    // Worker-thread-only: the persistent tracking session (Continuous
    // Speculative Tracking). Lazily created on the first reconcile; released
    // in the destructor once the worker thread is confirmed joined. Never
    // touched from any thread but the worker (and, after join(), the
    // destructor's calling thread -- see the destructor).
    std::unique_ptr<TrackedSession> session_;

    // Worker-thread-only: the hybrid-SSM virtual-rewind snapshot ring (fork-based
    // checkpoints on slot 0). Lazily created on the first hybrid generation and
    // reused; released in the destructor after the worker join, alongside
    // session_. Null for dense models (they use session_'s physical micro-rewind)
    // and for hybrids without branch headroom (plain fallback).
    std::unique_ptr<blackwell::HybridSnapshotRing> rewind_ring_;

    std::atomic<std::uint64_t> current_gen_{0};
    std::atomic<bool> stop_{false};
    // Latest-wins virtual-rewind request (generated tokens to roll back); 0 =
    // none. Set by RequestVirtualRewind() from any thread, consumed at the
    // hybrid decode loop's per-token boundary. Separate from pending_ on purpose
    // (see RequestVirtualRewind): it edits the running job, never supersedes it.
    std::atomic<int> pendingRewind_{0};

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::optional<Job> pending_;  // latest-wins single slot (Track OR Generate)
    // FIFO engine tasks (JIT pre-cache): behind jobs, ahead of lifecycle ops,
    // never displaced by activity -- the user explicitly asked for them.
    std::deque<std::function<void()>> engineTasks_;     // guarded by mutex_
    // Latest-wins lifecycle slot, separate from pending_ so a spill/hibernate
    // request never displaces a translation job (jobs run first; enqueuing a
    // job clears a not-yet-executed lifecycle request -- activity wins).
    LifecycleOp pendingLifecycle_ = LifecycleOp::None;  // guarded by mutex_
    LifecycleSink lifecycleSink_;                       // guarded by mutex_ (late-bound)
    std::thread worker_;
};
