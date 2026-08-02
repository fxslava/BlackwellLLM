#pragma once
// -----------------------------------------------------------------------------
// transcript_model.hpp — the thread-safe STORAGE behind the live transcript:
// completed utterances, the in-progress line, and the stable identity each
// finished utterance carries.
//
// WHY THIS IS SPLIT OUT OF transcript_view.hpp. Two reasons, and the second is
// the one that motivated it:
//   1. STL-ONLY. No ImGui, no bridge headers, no engine headers. That makes the
//      identity and capping rules directly testable in a CPU-only suite instead
//      of by eye.
//   2. It is the seam a "speak this line" action attaches to. A request to
//      synthesise an utterance has to name one, and it has to still name the
//      SAME one by the time a worker thread picks it up.
//
// THE BUG THIS EXISTS TO PREVENT. history_ is capped, and the cap drops from the
// FRONT. Under a positional key (an index into the vector) every entry silently
// renames itself the moment the cap starts biting — so a button wired to index 7
// speaks utterance 7 for the first 200 utterances of a session and a different
// one forever after. That is a bug that cannot reproduce in a short test and
// cannot be missed in a long demo. Hence: a monotone id, assigned once when a
// line is committed, never reused, never reordered, and stable against any
// amount of trimming.
//
// THREADING
//   * on_token()/on_final() are called FROM the engine thread (the speech
//     pipeline's token callback). They append under a short mutex.
//   * snapshot() runs on the UI thread, once per frame; it copies under the lock
//     and the caller renders outside it.
//   No ImGui call ever happens under the lock — enforced structurally here, by
//   this class having no way to make one.
// -----------------------------------------------------------------------------
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace rt {

// One committed line of transcript. Committed means "will never change again":
// the text is frozen at the moment it lands in history, which is exactly the
// property that makes it safe to hand to another thread by id.
struct Utterance {
    // Monotone, assigned at commit, NEVER reused within a session. The key any
    // cross-thread reference to this line must use. Zero is the "no utterance"
    // sentinel and is never assigned.
    std::uint64_t id = 0;

    // The generation that produced the line. A barge-in bumps the generation, so
    // this is what correlates a line with the turn it belongs to.
    std::uint64_t gen = 0;

    // "[Speech] ... | [Translation] ..." — the raw accumulated text, delimiter
    // and all. Splitting it is a presentation concern and stays in the view.
    std::string text;

    // TRUE when this line was flushed because a NEW generation started, i.e. the
    // speaker interrupted it, rather than because it finished. The text is a
    // partial translation: honest to display, but NOT a finished thought. Any
    // consumer that speaks, exports, or feeds this text onward must treat it as
    // incomplete.
    bool interrupted = false;
};

// Never-assigned id, for "no selection" / "not found" state in consumers.
inline constexpr std::uint64_t kInvalidUtteranceId = 0;

class TranscriptModel {
public:
    // What the UI thread reads per frame. A struct rather than three out-params
    // so the history/live pair is always internally consistent — snapshotting
    // them in two separate locked calls could catch a flush in between and show
    // a line twice, or not at all.
    struct Snapshot {
        std::vector<Utterance> history;
        std::string   live;      // the in-progress line; empty when idle
        std::uint64_t live_gen = 0;
    };

    // ---- engine thread ------------------------------------------------------
    // A streamed piece of the current utterance. A piece from a DIFFERENT
    // generation means the previous line was interrupted: it is flushed to
    // history (marked interrupted) and a fresh live line starts.
    void on_token(const char* text, std::uint64_t gen) {
        std::lock_guard<std::mutex> lk(m_);
        if (gen != live_gen_) {
            flush_live_locked(/*interrupted=*/true);
            live_gen_ = gen;
        }
        if (text != nullptr) live_ += text;
    }

    // The utterance finished cleanly (EOS / token cap): commit it. Ignored if the
    // generation has already moved on — that line was flushed as interrupted and
    // committing it twice would duplicate it.
    void on_final(std::uint64_t gen) {
        std::lock_guard<std::mutex> lk(m_);
        if (gen == live_gen_) flush_live_locked(/*interrupted=*/false);
    }

    // ---- UI thread ----------------------------------------------------------
    // Copies the current state into `out`, REUSING its buffers (the caller keeps
    // one snapshot across frames, so the steady state performs no allocation).
    void snapshot(Snapshot& out) const {
        std::lock_guard<std::mutex> lk(m_);
        out.history.assign(history_.begin(), history_.end());
        out.live.assign(live_);
        out.live_gen = live_gen_;
    }

    // ---- any thread ---------------------------------------------------------
    // Look up a committed line by id. Returns false if it has been trimmed by the
    // cap — which is not an error: a consumer holding an id across time MUST
    // handle the line having aged out, and this is where that is decided rather
    // than at an index that would silently resolve to the wrong text.
    bool find(std::uint64_t id, Utterance& out) const {
        if (id == kInvalidUtteranceId) return false;
        std::lock_guard<std::mutex> lk(m_);
        for (const Utterance& u : history_) {
            if (u.id == id) { out = u; return true; }
        }
        return false;
    }

    // Id that WILL be assigned to the next committed line. Test/diagnostic seam.
    std::uint64_t next_id() const {
        std::lock_guard<std::mutex> lk(m_);
        return next_id_;
    }

    std::size_t history_size() const {
        std::lock_guard<std::mutex> lk(m_);
        return history_.size();
    }

    // Drops all history and the in-progress line. Ids continue from where they
    // were — a cleared transcript must not start handing out ids that an
    // in-flight request already refers to.
    void clear() {
        std::lock_guard<std::mutex> lk(m_);
        history_.clear();
        live_.clear();
        live_gen_ = kNoLiveGen;
    }

    // Scrollback cap. The oldest entries are dropped, which is precisely why
    // positional keys are unusable and ids exist.
    static constexpr std::size_t kMaxHistory = 200;

private:
    // Moves the in-progress line into history under a fresh id. An interrupted
    // partial is PRESERVED rather than dropped, so a barge-in stays visible in
    // the transcript. Caller holds m_.
    void flush_live_locked(bool interrupted) {
        if (live_.empty()) return;
        Utterance u;
        u.id = next_id_++;
        u.gen = live_gen_;
        u.text = std::move(live_);
        u.interrupted = interrupted;
        history_.push_back(std::move(u));
        if (history_.size() > kMaxHistory) history_.erase(history_.begin());
        live_.clear();   // moved-from: restore a defined empty state
    }

    // Sentinel for "no live line yet". UINT64_MAX rather than 0 because 0 is a
    // legitimate generation id, and the first on_token must be seen as a
    // generation CHANGE so the live line adopts it.
    static constexpr std::uint64_t kNoLiveGen = UINT64_MAX;

    mutable std::mutex m_;
    std::vector<Utterance> history_;
    std::string live_;
    std::uint64_t live_gen_ = kNoLiveGen;
    std::uint64_t next_id_ = 1;   // 0 is kInvalidUtteranceId
};

}  // namespace rt
