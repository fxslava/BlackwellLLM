#include "live_translation_tracker.h"

// windows.h's min/max macros would shred the std::min/std::max calls inside
// the paging substrate headers included below.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>  // OutputDebugStringW

#include <algorithm>
#include <string_view>
#include <utility>

#include "blackwell_llm_adapter.h"              // playground::BlackwellLLMAdapter
#include "config.h"                              // ToUtf8 / FromUtf8
#include "../../engine_prefill_coordinator.h"    // EnginePrefillCoordinator, EngineSequence

// The opaque holder promised by the header: keeps engine/CUDA includes out of
// live_translation_tracker.h, mirroring BlackwellEngine::Impl's own pImpl.
struct LiveTranslationTracker::TrackedSession {
    blackwell::EnginePrefillCoordinator::EngineSequence seq;
};

namespace {

void Log(const std::wstring& message) {
    OutputDebugStringW((L"[live_tracker] " + message + L"\n").c_str());
}

// Length of the longest prefix of `s` that ends on a UTF-8 codepoint boundary
// (i.e. contains only whole sequences). Used to keep a token whose bytes end
// mid-codepoint from being converted until the completing bytes arrive.
size_t CompleteUtf8Prefix(const std::string& s) {
    size_t i = 0;
    while (i < s.size()) {
        const auto c = static_cast<unsigned char>(s[i]);
        const size_t len = c < 0x80 ? 1 : (c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1);
        if (i + len > s.size()) break;  // sequence runs past the end -> incomplete
        i += len;
    }
    return i;
}

// Chop a trailing INCOMPLETE UTF-8 sequence so a mid-codepoint streaming
// boundary never flashes U+FFFD in the overlay. (The dropped bytes reappear
// with the next delta.)
void TrimIncompleteUtf8(std::string& s) {
    size_t i = s.size();
    size_t continuation = 0;
    while (i > 0 && (static_cast<unsigned char>(s[i - 1]) & 0xC0) == 0x80 &&
           continuation < 3) {
        --i;
        ++continuation;
    }
    if (i == 0) return;
    const auto lead = static_cast<unsigned char>(s[i - 1]);
    size_t expected = 1;
    if (lead >= 0xF0) expected = 4;
    else if (lead >= 0xE0) expected = 3;
    else if (lead >= 0xC0) expected = 2;
    if (expected > 1 && continuation + 1 < expected) s.resize(i - 1);
}

// Extract the text between <finish> and </finish>. `require_close`:
//   true  -- a clean, fully-streamed body (used for the FINAL delivery); an
//            unclosed tag returns empty rather than a body we can't trust.
//   false -- the best-effort PARTIAL body while still streaming: strips a
//            partially-streamed "</finis" tail and any incomplete UTF-8 tail
//            so nothing ever flashes garbled on screen.
std::wstring ExtractFinishBody(const std::string& acc, bool require_close) {
    static constexpr std::string_view kOpen = "<finish>";
    static constexpr std::string_view kClose = "</finish>";

    const size_t open = acc.find(kOpen);
    if (open == std::string::npos) return {};

    std::string body = acc.substr(open + kOpen.size());
    if (const size_t close = body.find(kClose); close != std::string::npos) {
        body.erase(close);
    } else if (require_close) {
        return {};
    } else {
        if (const size_t lt = body.rfind('<'); lt != std::string::npos) {
            const std::string_view tail = std::string_view(body).substr(lt);
            if (tail.size() < kClose.size() && kClose.substr(0, tail.size()) == tail) {
                body.erase(lt);
            }
        }
        TrimIncompleteUtf8(body);
    }
    return FromUtf8(body);
}

}  // namespace

LiveTranslationTracker::LiveTranslationTracker(playground::BlackwellLLMAdapter& adapter,
                                               std::vector<std::string> system_prompts,
                                               int active_language, int max_new_tokens,
                                               float temperature, float top_p)
    : adapter_(adapter),
      max_new_tokens_(max_new_tokens),
      temperature_(temperature),
      top_p_(top_p) {
    if (system_prompts.empty()) {
        system_prompts.emplace_back();  // never index into an empty vector
    }
    if (active_language < 0 || active_language >= static_cast<int>(system_prompts.size())) {
        active_language = 0;
    }
    system_prompts_ =
        std::make_shared<const std::vector<std::string>>(std::move(system_prompts));
    active_language_.store(active_language, std::memory_order_relaxed);
    worker_ = std::thread(&LiveTranslationTracker::ThreadMain, this);
}

void LiveTranslationTracker::SetActiveLanguage(int index) {
    std::shared_ptr<const std::vector<std::string>> prompts;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        prompts = system_prompts_;
    }
    if (index < 0 || index >= static_cast<int>(prompts->size())) {
        return;  // out of range for the CURRENT set: ignore
    }
    active_language_.store(index, std::memory_order_relaxed);
}

void LiveTranslationTracker::UpdatePrompts(std::vector<std::string> prompts, int active) {
    if (prompts.empty()) {
        return;  // there is always at least one direction; ignore a bogus update
    }
    if (active < 0 || active >= static_cast<int>(prompts.size())) {
        active = 0;
    }
    auto snapshot = std::make_shared<const std::vector<std::string>>(std::move(prompts));
    {
        std::lock_guard<std::mutex> lock(mutex_);
        system_prompts_ = std::move(snapshot);
    }
    // Ordered after the swap so a concurrent ActivePrompt() can at worst pair
    // the OLD index with the NEW set -- and both stores clamp, so any pairing
    // is in range. The tracking session self-heals on the next reconcile.
    active_language_.store(active, std::memory_order_relaxed);
}

std::string LiveTranslationTracker::ActivePrompt() const {
    std::shared_ptr<const std::vector<std::string>> prompts;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        prompts = system_prompts_;
    }
    int index = active_language_.load(std::memory_order_relaxed);
    if (index < 0 || index >= static_cast<int>(prompts->size())) {
        index = 0;  // set shrank under the index: fall back to the first pair
    }
    return (*prompts)[index];
}

LiveTranslationTracker::~LiveTranslationTracker() {
    stop_.store(true, std::memory_order_relaxed);
    current_gen_.fetch_add(1, std::memory_order_relaxed);  // abort in-flight decode
    {
        std::lock_guard<std::mutex> lock(mutex_);
    }  // pairs the flag with the cv (no wakeup may fall between check and wait)
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();

    // The worker thread has fully exited -- join() is a happens-before fence,
    // so no concurrent engine access remains possible from here on, even
    // though THIS destructor runs on whatever arbitrary thread is tearing
    // this object down. Release the persistent tracking session so its
    // pin/lock don't leak (mirrors ~EnginePrefillCoordinator's own "abandoned
    // session" cleanup, just one call earlier).
    if (session_ && session_->seq.valid()) {
        try {
            adapter_.engine().prefill_driver().finish(session_->seq);
        } catch (...) {
        }
    }
}

void LiveTranslationTracker::TrackUpdate(std::wstring current_text, std::wstring context) {
    if (stop_.load(std::memory_order_relaxed) || current_text.empty()) return;
    Job job;
    job.kind = JobKind::Track;
    job.text = std::move(current_text);
    job.context = std::move(context);
    job.gen = current_gen_.fetch_add(1, std::memory_order_relaxed) + 1;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_ = std::move(job);  // latest wins, whether it displaces a Track
                                    // or a not-yet-started Generate
        pendingLifecycle_ = LifecycleOp::None;  // activity cancels a queued spill/hibernate
    }
    cv_.notify_all();
}

void LiveTranslationTracker::TriggerGeneration(std::wstring current_text, std::wstring context,
                                               StreamCallback callback) {
    if (stop_.load(std::memory_order_relaxed) || current_text.empty()) {
        if (callback) callback(L"", {}, true);
        return;
    }
    Job job;
    job.kind = JobKind::Generate;
    job.text = std::move(current_text);
    job.context = std::move(context);
    job.gen = current_gen_.fetch_add(1, std::memory_order_relaxed) + 1;
    job.callback = std::move(callback);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_ = std::move(job);
        pendingLifecycle_ = LifecycleOp::None;  // activity cancels a queued spill/hibernate
    }
    cv_.notify_all();
}

void LiveTranslationTracker::Cancel() {
    current_gen_.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(mutex_);
    pending_.reset();
}

void LiveTranslationTracker::SetLifecycleSink(LifecycleSink sink) {
    std::lock_guard<std::mutex> lock(mutex_);
    lifecycleSink_ = std::move(sink);
}

void LiveTranslationTracker::PostEngineTask(std::function<void()> task) {
    if (!task || stop_.load(std::memory_order_relaxed)) return;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        engineTasks_.push_back(std::move(task));
    }
    cv_.notify_all();
}

void LiveTranslationTracker::RequestKvSpill() {
    if (stop_.load(std::memory_order_relaxed)) return;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        // Never escalate past a queued Hibernate, and never race a queued job
        // (the job clears lifecycle requests when it lands anyway).
        if (pendingLifecycle_ == LifecycleOp::None) {
            pendingLifecycle_ = LifecycleOp::Spill;
        }
    }
    cv_.notify_all();
}

void LiveTranslationTracker::RequestHibernate() {
    if (stop_.load(std::memory_order_relaxed)) return;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pendingLifecycle_ = LifecycleOp::Hibernate;  // supersedes a queued Spill
    }
    cv_.notify_all();
}

void LiveTranslationTracker::EmitLifecycle(LifecycleEvent event) {
    LifecycleSink sink;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        sink = lifecycleSink_;  // copy under the lock; call outside it
    }
    if (sink) sink(event);
}

void LiveTranslationTracker::WakeIfHibernated() {
    if (!adapter_.engine().hibernated()) return;
    EmitLifecycle(LifecycleEvent::WakingUp);
    try {
        adapter_.engine().wakeup();  // one pinned-RAM -> VRAM PCIe DMA burst
        EmitLifecycle(LifecycleEvent::Awake);
    } catch (const std::exception& e) {
        // VRAM reclaim failed (another process grabbed it while we slept).
        // Leave the engine hibernated; the job that triggered us will fail
        // loudly on its engine call and the next activity retries the wakeup.
        Log(L"wakeup FAILED (still hibernated): " + FromUtf8(e.what()));
    }
}

void LiveTranslationTracker::RunLifecycle(LifecycleOp op) {
    try {
        // Release the persistent tracking session first: its pinned prefix
        // pages are excluded from demotion, and holding a live sequence across
        // a weight offload has no meaning anyway. The session self-heals -- the
        // next reconcile simply begin_sequence()s from the (spilled, faulted-
        // back-on-demand) radix tree.
        if (session_ && session_->seq.valid()) {
            adapter_.engine().prefill_driver().finish(session_->seq);
        }
        session_.reset();

        // Stage 1 either way: hibernation without the KV spill would strand
        // the radix tree's VRAM pages while the weights leave.
        const int moved = adapter_.engine().spill_kv_cache();
        Log(L"lifecycle: KV spill demoted " + std::to_wstring(moved) + L" page(s)");

        if (op == LifecycleOp::Hibernate) {
            if (!adapter_.engine().hibernated()) {
                adapter_.engine().hibernate();
            }
            EmitLifecycle(LifecycleEvent::Hibernated);
        } else {
            EmitLifecycle(LifecycleEvent::KvSpilled);
        }
    } catch (const std::exception& e) {
        // Best-effort by design: a failed lifecycle step must never take the
        // translator down -- the engine stays in whatever consistent state the
        // failing call guaranteed, and the next tick may retry.
        Log(L"lifecycle step failed: " + FromUtf8(e.what()));
    }
}

void LiveTranslationTracker::ThreadMain() {
    while (true) {
        Job job;
        bool haveJob = false;
        std::function<void()> task;
        LifecycleOp lifecycle = LifecycleOp::None;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] {
                return stop_.load(std::memory_order_relaxed) || pending_.has_value() ||
                       !engineTasks_.empty() || pendingLifecycle_ != LifecycleOp::None;
            });
            if (stop_.load(std::memory_order_relaxed)) break;
            // Priority per wake-up: translation job, then one queued engine
            // task (pre-cache), then a lifecycle op -- so a user-visible
            // translation is never stuck behind a compile, and a spill/
            // hibernate never preempts explicitly requested work.
            if (pending_.has_value()) {
                job = std::move(*pending_);
                pending_.reset();
                haveJob = true;
            } else if (!engineTasks_.empty()) {
                task = std::move(engineTasks_.front());
                engineTasks_.pop_front();
            } else {
                lifecycle = pendingLifecycle_;
                pendingLifecycle_ = LifecycleOp::None;
            }
        }

        if (!haveJob) {
            if (task) {
                WakeIfHibernated();  // engine work needs the weights resident
                task();
            } else {
                RunLifecycle(lifecycle);
            }
            continue;
        }

        // Superseded while queued: dead on arrival, skip the engine call
        // entirely instead of letting the decode loop cancel it mid-flight.
        if (job.gen != current_gen_.load(std::memory_order_relaxed)) continue;

        // First activity after a hibernation restores the weights before any
        // engine work; the KV pages it spilled fault back in on demand during
        // the reconcile itself.
        WakeIfHibernated();

        if (job.kind == JobKind::Track) {
            RunTrack(job);
        } else {
            RunGenerate(job);
        }
    }
}

std::vector<int> LiveTranslationTracker::BuildTokens(const std::wstring& text,
                                                     const std::wstring& context) const {
    using Adapter = playground::BlackwellLLMAdapter;

    std::string user_turn = ToUtf8(text);
    if (!context.empty()) {
        // Same durable-context framing the (now-removed) AgentOrchestrator
        // preview path used: an extra grounding note ahead of the live
        // segment. TrackUpdate and TriggerGeneration are always called with
        // the SAME context for a given typing session (it only changes at
        // commit time), so update_sequence's diff still lands where you'd
        // expect -- and a context change (post-commit) is simply a bigger,
        // still-correct diff, not a special case.
        user_turn = "Text committed earlier in this field (already translated; "
                    "keep the new translation consistent with it):\n" +
                    ToUtf8(context) + "\n\n" + user_turn;
    }
    // Exactly AgentOrchestrator::render_transcript()'s shape (and exactly what
    // translation_service.cpp's StableServingPrefix rendered for the JIT cache
    // compile): exact match matters here, not resemblance. ActivePrompt() picks
    // the current direction's system block -> the matching cached radix branch.
    const std::string role_transcript =
        "[SYSTEM]\n" + ActivePrompt() + "\n\n[USER]\n" + user_turn + "\n\n[ASSISTANT]\n";
    const std::string prompt_text =
        Adapter::apply_chat_template(role_transcript, adapter_.chat_template());

    // tokenize() is the SAME tokenizer generate() and the JIT compiler use --
    // the adapter bound it to the coordinator at construction -- so this
    // never needs its own tokenizer handle. add_special=false mirrors the
    // serving encode(prompt_text, /*add_special_tokens=*/false) exactly.
    return adapter_.engine().prefill_driver().tokenize(prompt_text, /*add_special=*/false);
}

void LiveTranslationTracker::RunTrack(const Job& job) {
    if (!adapter_.engine().has_prefix_cache()) {
        return;  // no tracking session on this model -- nothing to reconcile
    }
    try {
        const std::vector<int> tokens = BuildTokens(job.text, job.context);
        if (tokens.empty() || tokens.size() >= adapter_.max_seq_len()) return;

        auto& coord = adapter_.engine().prefill_driver();
        if (!session_ || !session_->seq.valid()) {
            // First reconcile of this tracker's lifetime: begin_sequence's
            // initial commit publishes (mostly the already-JIT-cached system
            // prompt +) a short user-text tail -- cheap, deduped, and it is
            // the stable anchor every later update_sequence() diffs against.
            session_ = std::make_unique<TrackedSession>();
            session_->seq = coord.begin_sequence(tokens);
        } else {
            // Cheap: diff against the mirror, truncate/rewind to the
            // divergence point, recompute only the new suffix. Deliberately
            // NOT committed to the radix tree -- see the class doc comment.
            coord.update_sequence(session_->seq, tokens);
        }
    } catch (const std::exception& e) {
        // Never fatal: update_sequence's own failure path already rolls the
        // session back to a consistent common-prefix state (still valid,
        // still usable) before rethrowing, so there is nothing to repair
        // here -- just note that this particular keystroke's warm-up was
        // lost. A failed FIRST begin_sequence() throws before session_ is
        // ever assigned, so it simply stays null and retries next time.
        Log(L"speculative reconcile skipped: " + FromUtf8(e.what()));
    }
}

void LiveTranslationTracker::RunGenerate(const Job& job) {
    if (!adapter_.engine().has_prefix_cache()) {
        if (!logged_fallback_.exchange(true, std::memory_order_relaxed)) {
            Log(L"this model has no prefix-cache substrate (hybrid SSM / gated "
                L"attention / Continuous mode); live tracker falls back to the "
                L"adapter's ordinary generate() -- correct, just without "
                L"speculative tracking");
        }
        RunGenerateFallback(job);
        return;
    }

    std::vector<int> tokens;
    try {
        tokens = BuildTokens(job.text, job.context);
    } catch (const std::exception& e) {
        Log(L"generation aborted (token build): " + FromUtf8(e.what()));
        if (job.callback) job.callback(L"", {}, true);
        return;
    }
    if (tokens.empty() || tokens.size() >= adapter_.max_seq_len()) {
        if (job.callback) job.callback(L"", {}, true);
        return;
    }

    auto& coord = adapter_.engine().prefill_driver();
    try {
        if (!session_ || !session_->seq.valid()) {
            session_ = std::make_unique<TrackedSession>();
            session_->seq = coord.begin_sequence(tokens);
        } else {
            coord.update_sequence(session_->seq, tokens);
        }
        // The debounce firing IS the stable boundary the coordinator's docs
        // call for ("Call commit_sequence() at ... debounce timeout"):
        // publish the prefix we are about to decode from for real, unlike
        // TrackUpdate's transient, uncommitted per-keystroke reconciles.
        coord.commit_sequence(session_->seq);
    } catch (const std::exception& e) {
        Log(L"generation aborted (prefill): " + FromUtf8(e.what()));
        if (job.callback) job.callback(L"", {}, true);
        return;
    }

    const blackwell::SeqId seq = session_->seq.engine_seq;
    int pos = static_cast<int>(session_->seq.tokens.size());
    int next;
    try {
        // Deterministic, like the adapter's own prefill convention: the first
        // token comes straight from the logits the reconcile left behind, no
        // extra forward() needed.
        next = coord.sample_last_logits(0.0f, 1.0f);
    } catch (const std::exception& e) {
        Log(L"generation aborted (sampling): " + FromUtf8(e.what()));
        if (job.callback) job.callback(L"", {}, true);
        return;
    }

    const bool collectProbs = collect_probs_.load(std::memory_order_relaxed);
    const size_t cap = adapter_.max_seq_len();
    std::string acc;
    std::wstring last_posted;
    // Parallel to `acc`: the byte length of `acc` after each token AND that
    // token's probability, so BuildHeatmap can slice pieces against the
    // <finish> body window without re-tokenizing. Empty unless collecting.
    std::vector<std::pair<size_t, float>> pieceEnds;
    int generated = 0;
    bool finished = false;
    bool interrupted = false;

    while (static_cast<size_t>(pos) < cap && generated < max_new_tokens_) {
        if (job.gen != current_gen_.load(std::memory_order_relaxed)) {
            interrupted = true;  // a newer TrackUpdate/TriggerGeneration superseded us
            break;
        }
        if (adapter_.tokenizer().is_stop(next)) break;  // real eos, no <finish> ever closed

        // Developer Mode: read the sampled token's probability from the CURRENT
        // logits (the ones it was drawn from) BEFORE the next forward() pass
        // overwrites them. A full-vocab read -- gated on the debug flag.
        float prob = 1.0f;
        if (collectProbs) {
            try {
                prob = adapter_.engine().last_token_probability(next);
            } catch (const std::exception&) {
                prob = 1.0f;  // never let a debug read break the decode
            }
        }

        acc += adapter_.tokenizer().decode(next);
        if (collectProbs) pieceEnds.emplace_back(acc.size(), prob);
        if (acc.find("</finish>") != std::string::npos) finished = true;

        StreamPartial(acc, job.callback, last_posted);
        if (finished) break;

        next = adapter_.engine().forward(next, pos, temperature_, top_p_, seq);
        ++pos;
        ++generated;
    }

    // The tracking session stays OPEN and UNTOUCHED regardless of how the
    // loop ended (natural stop, <finish> close, or interrupt): its token
    // mirror still reflects only the committed prompt, so the NEXT
    // update_sequence() call correctly truncates away whatever this decode
    // wrote past it, exactly as if nothing had been generated at all. No
    // finish()/cleanup needed here -- that is the entire point of the
    // persistent-session design over a per-call prefill_prompt()+finish().

    if (interrupted || !job.callback) {
        return;  // superseded: the interrupting call owns the next delivery
    }
    const std::wstring final_text =
        finished ? ExtractFinishBody(acc, /*require_close=*/true) : std::wstring();
    const TokenHeatmap heatmap =
        (collectProbs && finished) ? BuildHeatmap(acc, pieceEnds) : TokenHeatmap{};
    job.callback(final_text, heatmap, /*done=*/true);
}

void LiveTranslationTracker::RunGenerateFallback(const Job& job) {
    std::string user_turn = ToUtf8(job.text);
    if (!job.context.empty()) {
        user_turn = "Text committed earlier in this field (already translated; "
                    "keep the new translation consistent with it):\n" +
                    ToUtf8(job.context) + "\n\n" + user_turn;
    }
    const std::string transcript =
        "[SYSTEM]\n" + ActivePrompt() + "\n\n[USER]\n" + user_turn + "\n\n[ASSISTANT]\n";

    playground::BlackwellLLMAdapter::Params params;
    params.temperature = temperature_;
    params.top_p = top_p_;
    params.max_new_tokens = max_new_tokens_;

    std::string acc;
    std::wstring last_posted;
    bool interrupted = false;
    const std::string raw = adapter_.generate(
        transcript, params, /*prompt_out=*/nullptr,
        [&](const std::string& delta) -> bool {
            if (job.gen != current_gen_.load(std::memory_order_relaxed)) {
                interrupted = true;
                return false;
            }
            acc += delta;
            StreamPartial(acc, job.callback, last_posted);
            return acc.find("</finish>") == std::string::npos;
        },
        /*seq_id=*/0);

    if (interrupted || !job.callback) return;

    // The adapter's own decode loop already stripped any matched checkpoint-
    // level stop string (e.g. <|im_end|>); extract our <finish> body on top of
    // that, falling back to the raw text if the model skipped the tag
    // entirely (best-effort, same spirit as the old orchestrator's "nudged
    // prose" salvage).
    std::wstring final_text = ExtractFinishBody(raw, /*require_close=*/true);
    if (final_text.empty() && !raw.empty() && raw.find('<') == std::string::npos) {
        final_text = FromUtf8(raw);
    }
    // The fallback path (adapter generate()) exposes no per-token logits, so it
    // carries no heatmap -- correctness over instrumentation on legacy models.
    job.callback(final_text, {}, /*done=*/true);
}

void LiveTranslationTracker::StreamPartial(const std::string& acc,
                                           const StreamCallback& callback,
                                           std::wstring& last_posted) const {
    if (!callback) return;
    std::wstring text = ExtractFinishBody(acc, /*require_close=*/false);
    if (text.empty() || text == last_posted) return;
    last_posted = text;
    callback(last_posted, {}, /*done=*/false);  // partials never carry the heatmap
}

TokenHeatmap LiveTranslationTracker::BuildHeatmap(
    const std::string& acc, const std::vector<std::pair<size_t, float>>& pieceEnds) const {
    // Body window in RAW (UTF-8) byte offsets: the same slice ExtractFinishBody
    // returns, but tracked in `acc`'s coordinates so it lines up with pieceEnds.
    static constexpr std::string_view kOpen = "<finish>";
    static constexpr std::string_view kClose = "</finish>";
    const size_t open = acc.find(kOpen);
    if (open == std::string::npos) return {};
    const size_t bodyStart = open + kOpen.size();
    size_t bodyEnd = acc.find(kClose, bodyStart);
    if (bodyEnd == std::string::npos) bodyEnd = acc.size();

    // Walk the tokens in order, clipping each to the body window. A token can
    // end mid-codepoint (byte-level BPE), so carry the incomplete tail forward
    // to the next token before converting -- this guarantees the concatenation
    // of the emitted pieces equals FromUtf8(whole body) == the delivered
    // translation, which is exactly what the overlay measures the heatmap
    // against. The rare split codepoint takes the later token's probability.
    TokenHeatmap heatmap;
    size_t pieceStart = 0;
    std::string carry;
    float carryProb = 1.0f;
    for (const auto& [pieceEnd, prob] : pieceEnds) {
        const size_t lo = std::max(pieceStart, bodyStart);
        const size_t hi = std::min(pieceEnd, bodyEnd);
        pieceStart = pieceEnd;
        if (lo >= hi) continue;  // token lies entirely in the <finish>/</finish> framing

        std::string bytes = carry + acc.substr(lo, hi - lo);
        const size_t good = CompleteUtf8Prefix(bytes);
        std::wstring piece = FromUtf8(bytes.substr(0, good));
        carry = bytes.substr(good);  // incomplete tail -> next token
        carryProb = prob;
        if (!piece.empty()) {
            heatmap.push_back({std::move(piece), prob});
        }
    }
    if (!carry.empty()) {  // valid UTF-8 never leaves a tail, but stay safe
        std::wstring piece = FromUtf8(carry);
        if (!piece.empty()) heatmap.push_back({std::move(piece), carryProb});
    }
    return heatmap;
}
