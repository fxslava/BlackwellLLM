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
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace playground {
class BlackwellLLMAdapter;
}

class LiveTranslationTracker {
public:
    // Runs on THIS CLASS'S WORKER THREAD. `done=false` carries a growing
    // partial translation as it streams from the model; `done=true` marks the
    // end (text empty on failure -- no <finish> ever closed, or the model's
    // real eos fired before it did). The receiver must only enqueue/marshal --
    // never touch a window or COM directly.
    using StreamCallback = std::function<void(const std::wstring& text, bool done)>;

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

    // Switch the active translation direction (index into `system_prompts`).
    // Thread-safe (atomic); out-of-range indices are clamped. The next request
    // builds tokens from the new prompt -- update_sequence simply reconciles the
    // (now larger) diff, so no explicit session reset is needed.
    void SetActiveLanguage(int index);

private:
    enum class JobKind { Track, Generate };
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
    // The system prompt for the currently-active direction (clamped read).
    const std::string& ActivePrompt() const;
    // Stream the text between <finish> and the (possibly still-streaming)
    // </finish> to `callback` if it changed since `last_posted`.
    void StreamPartial(const std::string& acc, const StreamCallback& callback,
                       std::wstring& last_posted) const;

    playground::BlackwellLLMAdapter& adapter_;
    std::vector<std::string> system_prompts_;      // one per translation direction
    std::atomic<int> active_language_{0};          // index into system_prompts_
    int max_new_tokens_;
    float temperature_;
    float top_p_;
    std::atomic<bool> logged_fallback_{false};  // one-time note, not per-call spam

    // Worker-thread-only: the persistent tracking session (Continuous
    // Speculative Tracking). Lazily created on the first reconcile; released
    // in the destructor once the worker thread is confirmed joined. Never
    // touched from any thread but the worker (and, after join(), the
    // destructor's calling thread -- see the destructor).
    std::unique_ptr<TrackedSession> session_;

    std::atomic<std::uint64_t> current_gen_{0};
    std::atomic<bool> stop_{false};

    std::mutex mutex_;
    std::condition_variable cv_;
    std::optional<Job> pending_;  // latest-wins single slot (Track OR Generate)
    std::thread worker_;
};
