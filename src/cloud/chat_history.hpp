#pragma once
// =============================================================================
// cloud/chat_history.hpp — the BOUNDED sliding window of finished turns that
// gives the remote leg a memory. Header-only and dependency-free (no curl, no
// simdjson, no CUDA), exactly like intent_request.hpp, so the app and the tests
// share one implementation.
//
// WHY THIS EXISTS
//   A remote /chat/completions call is stateless: the endpoint remembers
//   nothing between requests, so without a replayed transcript the model forgets
//   the user's name the moment the turn ends. The LOCAL leg has no such problem
//   -- its memory IS the KV cache, which survives across turns above the system
//   prefix floor -- which is precisely why this class is needed only on the
//   remote side and why nothing here touches the engine.
//
// WHY IT IS BOUNDED, TWICE
//   Cloud endpoints bill by the input token, and a transcript replayed in full
//   grows the bill QUADRATICALLY in a session (turn N re-sends turns 1..N-1).
//   So the window is capped in two independent places, each with a different
//   owner:
//
//     Config::max_turns            how many pairs this store KEEPS (memory)
//     OpenAiRequestOptions::
//       max_history_pairs          how many pairs actually go ON THE WIRE (cost)
//
//   Keeping the two separate is deliberate: the store is what the UI could one
//   day show, the wire cap is a spend decision. The wire cap is the smaller of
//   the two by default and truncation is always FROM THE FRONT -- the oldest
//   turn is the one that gets dropped.
//
//   A THIRD bound lives one level out, in session_store.hpp: that is the durable
//   LOG, which keeps far more than either cap above and survives the process.
//   restore() seeds this window from its tail at startup, and the app appends
//   each sealed turn back to it -- but only for turns the REMOTE leg answered.
//   A local answer never reaches a disk; see the preamble of session_store.hpp.
//
// PAIRS ONLY, NEVER A HALF TURN. OpenAI- and Anthropic-compatible endpoints
// both require strictly alternating user/assistant messages; a dangling `user`
// with no reply after it is an HTTP 400 on the strict ones. A turn therefore
// enters the ring only when BOTH halves exist (commit_turn), and a turn that
// failed, was barged in on, or produced no text is dropped whole
// (abandon_turn).
//
// THREADING. begin_turn/commit_turn/abandon_turn run on the DISPATCHER thread;
// append_reply runs on the dispatcher thread for the remote leg and on the
// ENGINE thread for the local one (local_transport.hpp); snapshot() runs on the
// dispatcher thread inside the context provider. One mutex covers all of it --
// the contention is one short lock per token on a path that already takes one
// inside the TTS chunker, and the alternative (lock-free) would buy nothing on a
// path bounded by a network round trip.
// =============================================================================
#include <cstddef>
#include <deque>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "intent_request.hpp"  // ChatTurn

namespace blackwell::cloud {

namespace detail {

// Drops any trailing INCOMPLETE UTF-8 sequence.
//
// Not a nicety: every clamp below cuts at a byte offset, and a cut through a
// multi-byte codepoint produces a string that append_json_string will happily
// serialise and the gateway will reject with a 400 ("invalid utf-8") -- which
// reaches the user as "the assistant did not answer", with nothing in the log
// pointing here. This app transcribes Russian, so multi-byte is the COMMON
// case, not an edge one.
inline void trim_utf8_tail(std::string& s) {
    // Continuation bytes are 10xxxxxx; a lead byte is 110/1110/11110xxxx and
    // announces how many continuations must follow it. Walk back over at most 3
    // continuations to find the lead of the final sequence, then keep that
    // sequence only if all of it arrived.
    std::size_t cont = 0;
    while (cont < s.size() && cont < 3 &&
           (static_cast<unsigned char>(s[s.size() - 1 - cont]) & 0xC0) == 0x80) {
        ++cont;
    }
    if (cont >= s.size()) {  // continuations with no lead at all: not UTF-8
        s.clear();
        return;
    }
    const auto lead = static_cast<unsigned char>(s[s.size() - 1 - cont]);
    if (lead < 0x80) return;  // plain ASCII tail; nothing is pending
    std::size_t need = 0;
    if ((lead & 0xE0) == 0xC0)      need = 1;
    else if ((lead & 0xF0) == 0xE0) need = 2;
    else if ((lead & 0xF8) == 0xF0) need = 3;
    else return;  // not a lead byte either -- malformed input, and not ours to fix
    if (need > cont) s.resize(s.size() - cont - 1);  // short: drop the lead too
}

// Appends as much of `chunk` as `cap` still allows, then repairs the tail.
// Silently stops once full: a reply long enough to hit the cap is already past
// the point where more of it helps the next turn.
inline void append_clamped(std::string& dst, std::string_view chunk, std::size_t cap) {
    if (dst.size() >= cap) return;
    const std::size_t room = cap - dst.size();
    dst.append(chunk.substr(0, room));
    if (chunk.size() > room) trim_utf8_tail(dst);
}

}  // namespace detail

class ChatHistory {
public:
    struct Config {
        // Pairs KEPT. 4 rather than 3 so the wire cap (3 by default) can be
        // raised at the endpoint level without also having to re-reason about
        // the store -- and so a single dropped turn does not empty the window.
        std::size_t max_turns = 4;
        // Hard per-message ceiling. The per-turn token cap already bounds a
        // well-behaved reply; this bounds a MISBEHAVING one (a model that loops
        // until max_tokens) so a single runaway turn cannot poison the next
        // several requests with thousands of replayed tokens.
        std::size_t max_chars_per_message = 4000;
    };

    explicit ChatHistory(Config cfg = {}) : cfg_(cfg) {}

    // Opens a turn. Called when the intent is DISPATCHED, not when it is
    // committed by the gate: an intent the gate accepted but the dispatcher
    // never sent is not part of the conversation.
    void begin_turn(std::string_view user_text) {
        const std::lock_guard<std::mutex> lk(mu_);
        pending_ = ChatTurn{};
        detail::append_clamped(pending_.user, user_text, cfg_.max_chars_per_message);
        pending_open_ = true;
    }

    // One streamed reply fragment. Ignored when no turn is open, which is the
    // right answer for a stray delta arriving after abandon_turn().
    void append_reply(std::string_view chunk) {
        const std::lock_guard<std::mutex> lk(mu_);
        if (!pending_open_) return;
        detail::append_clamped(pending_.assistant, chunk, cfg_.max_chars_per_message);
    }

    // Seeds the window from a persisted transcript, replacing whatever is in it.
    // The TAIL is what survives: `in` is the full log (session_store.hpp keeps far
    // more than any request replays), and the recent turns are the ones the next
    // utterance refers back to.
    //
    // Applies the same clamps and the same pair rule as commit_turn, because the
    // file it comes from is untrusted input -- a hand-edited half turn must not
    // become two consecutive `user` messages on the first request after a restart.
    void restore(const std::vector<ChatTurn>& in) {
        const std::lock_guard<std::mutex> lk(mu_);
        turns_.clear();
        pending_ = ChatTurn{};
        pending_open_ = false;
        for (const ChatTurn& src : in) {
            ChatTurn t;
            detail::append_clamped(t.user, src.user, cfg_.max_chars_per_message);
            detail::append_clamped(t.assistant, src.assistant, cfg_.max_chars_per_message);
            detail::trim_utf8_tail(t.user);
            detail::trim_utf8_tail(t.assistant);
            if (t.user.empty() || t.assistant.empty()) continue;
            turns_.push_back(std::move(t));
            if (turns_.size() > cfg_.max_turns) turns_.pop_front();
        }
    }

    // Seals the pending turn into the ring. Returns false (and drops it) when
    // either half is empty -- see the PAIRS ONLY note in the preamble.
    //
    // `out`, when non-null and the return is true, receives a copy of the turn as
    // it was SEALED -- clamped and UTF-8-repaired. The persistence layer wants
    // exactly those bytes and must not re-derive them from the caller's originals:
    // the log and the window would then disagree about what was said, and only
    // after a restart.
    bool commit_turn(ChatTurn* out = nullptr) {
        const std::lock_guard<std::mutex> lk(mu_);
        pending_open_ = false;
        // Final repair, because a STREAM can also end mid-codepoint: deltas are
        // split at token boundaries, not character ones, so the last fragment of
        // a truncated reply may be half a Cyrillic letter that no later chunk
        // will complete. append_clamped only repairs what IT cut.
        detail::trim_utf8_tail(pending_.user);
        detail::trim_utf8_tail(pending_.assistant);
        if (pending_.user.empty() || pending_.assistant.empty()) {
            pending_ = ChatTurn{};
            return false;
        }
        turns_.push_back(std::move(pending_));
        pending_ = ChatTurn{};
        if (out != nullptr) *out = turns_.back();
        while (turns_.size() > cfg_.max_turns) turns_.pop_front();  // oldest first
        return true;
    }

    // Discards the pending turn: a failed request, a refusal, or a barge-in.
    // Deliberately NOT recorded -- replaying a turn the user talked over would
    // teach the model that its interrupted half-sentence was an accepted answer.
    void abandon_turn() {
        const std::lock_guard<std::mutex> lk(mu_);
        pending_open_ = false;
        pending_ = ChatTurn{};
    }

    // Forgets everything, pending turn included. Called when the system prompt
    // is rebuilt: that rewinds the LOCAL KV to the prefix floor, and leaving the
    // remote leg replaying turns the local leg has forgotten would make the two
    // answer as different assistants.
    void clear() {
        const std::lock_guard<std::mutex> lk(mu_);
        turns_.clear();
        pending_ = ChatTurn{};
        pending_open_ = false;
    }

    // Refreshes `out` with the kept turns, OLDEST FIRST, excluding the pending
    // one.
    //
    // Fills a caller-owned buffer instead of returning a vector because
    // RequestContext::history is a SPAN: the storage it points at must outlive
    // the body build, and a returned temporary would dangle exactly the way
    // intent_request.hpp warns about. Give it a buffer that lives as long as the
    // dispatcher (see main.cpp).
    void snapshot(std::vector<ChatTurn>& out) const {
        const std::lock_guard<std::mutex> lk(mu_);
        out.assign(turns_.begin(), turns_.end());
    }

    [[nodiscard]] std::size_t size() const {
        const std::lock_guard<std::mutex> lk(mu_);
        return turns_.size();
    }

private:
    mutable std::mutex   mu_;
    Config               cfg_;
    std::deque<ChatTurn> turns_;      // oldest at the front
    ChatTurn             pending_;
    bool                 pending_open_ = false;
};

}  // namespace blackwell::cloud
