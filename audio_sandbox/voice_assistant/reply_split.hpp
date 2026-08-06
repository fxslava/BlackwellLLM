#pragma once
// -----------------------------------------------------------------------------
// reply_split.hpp — ONE answer, TWO audiences.
//
// The assistant is read and heard at the same time, and the two want opposite
// things. The screen wants the full answer with its lists, its code and its
// links. The speaker wants two sentences of prose: a chunk of Markdown handed to
// a character-level TTS is read out as "asterisk asterisk", and a four-paragraph
// answer takes a minute to speak that nobody sits through.
//
// So the model is asked for both, tagged:
//
//     <voice>Short spoken summary.</voice>
//     <ui>The full answer, Markdown and all.</ui>
//
// and this file is the two halves of that contract: the INSTRUCTION that asks
// for it (kOutputContract / compose_system_prompt) and the PARSER that reads it
// back (ReplySplitter). They live together because they are one agreement -- a
// tag renamed in one and not the other is a silent failure, and the failure it
// produces is the assistant going mute while the screen looks fine.
//
// STREAMING, NOT POST-HOC. The tags have to be resolved as the deltas arrive:
// the whole point of the <voice> block is that speech starts before the written
// answer has finished generating. That is what makes this a state machine rather
// than a regex over a finished string, and it is why a tag arriving split across
// two chunks ("<vo" then "ice>") is the case the implementation is organised
// around rather than an edge case bolted on.
//
// FAILS OPEN, WHICH IS THE WHOLE SAFETY ARGUMENT. A model that ignores the
// contract -- an 8B local backbone does, sometimes -- produces an untagged
// reply. That reply is shown as it streams and SPOKEN AT THE END, in full. The
// cost is time-to-first-audio on non-compliant turns; the alternative is an
// assistant that silently stops talking whenever the model forgets a tag, which
// is indistinguishable from broken speech output. Compare EngineControlBridge's
// is_eos(), which fails CLOSED for the opposite reason: there, the failure costs
// money.
//
// THREADING. Not thread-safe, deliberately: `on_text` fires on the ENGINE thread
// for the local leg and the DISPATCHER thread for the remote one, and the caller
// already owns a mutex for its own reply bookkeeping. A lock in here would be a
// second, differently-scoped one over the same data.
// -----------------------------------------------------------------------------
#include <algorithm>
#include <cstddef>
#include <functional>
#include <string>
#include <string_view>
#include <utility>

namespace rt {

// Appended to the persona so BOTH legs answer in the same shape -- the local
// model reads it in its frozen KV prefix, the remote one in `instructions`.
//
// WORDED AS A HARD FORMAT RULE, not a preference, because the failure it guards
// is invisible from inside the model: a reply with no <voice> block still looks
// perfectly good on screen. The examples are deliberately absent -- they would
// have to be repeated in the prefix of every request and this string sits inside
// the prompt-cache prefix, where every byte is paid for on cache misses.
inline constexpr const char* kOutputContract =
    "\n\n"
    "OUTPUT FORMAT (strict). Every reply consists of exactly two XML blocks, in "
    "this order and with nothing outside them:\n"
    "<voice>A short, natural spoken summary of the answer: one or two sentences, "
    "plain prose only. No Markdown, no lists, no code, no URLs, no emoji.</voice>\n"
    "<ui>The full answer for the screen. Markdown is welcome here: lists, code "
    "blocks, emphasis, links.</ui>\n"
    "The <voice> block is read aloud and the <ui> block is displayed, so the "
    "first must stand on its own as speech and never mention the second.";

// The persona the user wrote, plus the contract above. ONE function, called
// wherever a persona reaches a model, so the two legs cannot drift into
// answering in different shapes.
//
// The user's text goes FIRST and the contract last: the persona is what the
// model should be, the contract is how it must reply, and an instruction placed
// last is the one that survives a long prefix.
[[nodiscard]] inline std::string compose_system_prompt(std::string_view persona) {
    std::string out(persona);
    out.append(kOutputContract);
    return out;
}

// -----------------------------------------------------------------------------
// ReplySplitter — the reader half of the contract.
//
// Feed it the reply's deltas; it feeds `on_ui` what belongs on screen and
// `on_voice` what belongs in the speaker. Both sinks receive UTF-8 fragments,
// never a whole reply, so the caller's existing streaming path is unchanged.
// -----------------------------------------------------------------------------
class ReplySplitter {
public:
    using Sink = std::function<void(std::string_view)>;

    // Sinks are copied and may be empty (a build with no speech output binds no
    // voice sink, and the parsing still has to happen -- otherwise the tags
    // themselves would be printed on screen).
    ReplySplitter(Sink ui, Sink voice) : ui_(std::move(ui)), voice_(std::move(voice)) {}

    // Start a new reply. Every scrap of the previous one is dropped: a fragment
    // surviving into the next turn would splice half a tag onto the front of it.
    void reset() noexcept {
        held_.clear();
        fallback_.clear();
        state_ = State::Outside;
        saw_voice_ = false;
    }

    // One delta from the transport. May be a partial tag, a partial code point,
    // or both; neither is emitted until it is whole.
    void push(std::string_view chunk) {
        held_.append(chunk);
        scan(/*at_end=*/false);
    }

    // End of the reply. Surrenders whatever is still held -- a tail that looked
    // like the start of a tag and turned out to be the end of the answer -- and
    // then applies the fallback: if the model never opened a <voice> block, the
    // written answer is what gets spoken.
    void finish() {
        scan(/*at_end=*/true);
        if (!held_.empty()) {
            emit(held_);
            held_.clear();
        }
        if (!saw_voice_ && !fallback_.empty() && voice_) voice_(fallback_);
        fallback_.clear();
        state_ = State::Outside;
    }

    // Did this reply honour the contract? For telemetry: a run of `false` means
    // the persona edit dropped the contract, or the model is too small to follow
    // it -- and the symptom (speech arriving all at once, at the end) is
    // otherwise hard to attribute.
    [[nodiscard]] bool saw_voice_block() const noexcept { return saw_voice_; }

private:
    enum class State { Outside, Voice, Ui };

    static constexpr std::string_view kVoiceOpen  = "<voice>";
    static constexpr std::string_view kVoiceClose = "</voice>";
    static constexpr std::string_view kUiOpen     = "<ui>";
    static constexpr std::string_view kUiClose    = "</ui>";

    // Could `s` still become `tag` if more bytes arrived? True for a strict
    // prefix, which is exactly the "wait, do not emit" condition.
    static bool could_become(std::string_view s, std::string_view tag) noexcept {
        return s.size() < tag.size() && tag.compare(0, s.size(), s) == 0;
    }

    void emit(std::string_view text) {
        if (text.empty()) return;
        switch (state_) {
            case State::Voice:
                if (voice_) voice_(text);
                return;
            case State::Ui:
            case State::Outside:
                // OUTSIDE TEXT GOES TO THE SCREEN, not nowhere. Before the first
                // tag it is a preamble the model wrote instead of complying;
                // after the last one it is a trailing thought. Dropping either
                // would mean the user's answer silently loses words whenever the
                // model is imperfect -- and the words are the product.
                if (ui_) ui_(text);
                // The spoken fallback is everything that reached the screen, so a
                // reply with no <voice> block is still heard in full. Bounded:
                // past a few thousand characters this is a wall of text nobody
                // wants read aloud anyway, and an unbounded buffer on a stream
                // the model controls is not a thing to leave lying around.
                if (!saw_voice_ && fallback_.size() < kFallbackCap) {
                    fallback_.append(text, 0,
                                     std::min(text.size(), kFallbackCap - fallback_.size()));
                }
                return;
        }
    }

    // Consume as much of `held_` as can be resolved. What survives is either a
    // partial tag or (at_end) nothing.
    void scan(bool at_end) {
        std::size_t pos = 0;
        for (;;) {
            const std::string_view rest(held_.data() + pos, held_.size() - pos);
            if (rest.empty()) break;

            const std::size_t lt = rest.find('<');
            if (lt == std::string_view::npos) {
                emit(rest);
                pos = held_.size();
                break;
            }
            emit(rest.substr(0, lt));
            pos += lt;

            const std::string_view tail(held_.data() + pos, held_.size() - pos);
            if (take_tag(tail, pos)) continue;

            // Not a tag -- but it may be one that has not fully arrived yet.
            // Holding here is the ONLY reason a tag split across two deltas
            // works, and it costs at most a few bytes of latency on a literal
            // '<' in the answer.
            if (!at_end && (could_become(tail, kVoiceOpen) || could_become(tail, kVoiceClose) ||
                            could_become(tail, kUiOpen) || could_become(tail, kUiClose))) {
                break;
            }
            // A genuine '<' in the text (a comparison, a generic, an HTML
            // example). Emit it and carry on past it.
            emit(tail.substr(0, 1));
            pos += 1;
        }
        held_.erase(0, pos);
    }

    // If `tail` opens with one of the four tags, apply it and advance `pos`.
    bool take_tag(std::string_view tail, std::size_t& pos) {
        if (tail.compare(0, kVoiceOpen.size(), kVoiceOpen) == 0) {
            state_ = State::Voice;
            saw_voice_ = true;
            // The fallback is dead the moment a real voice block starts: the
            // model IS complying, and speaking the screen text as well would say
            // everything twice.
            fallback_.clear();
            pos += kVoiceOpen.size();
            return true;
        }
        if (tail.compare(0, kUiOpen.size(), kUiOpen) == 0) {
            state_ = State::Ui;
            pos += kUiOpen.size();
            return true;
        }
        if (tail.compare(0, kVoiceClose.size(), kVoiceClose) == 0) {
            state_ = State::Outside;
            pos += kVoiceClose.size();
            return true;
        }
        if (tail.compare(0, kUiClose.size(), kUiClose) == 0) {
            state_ = State::Outside;
            pos += kUiClose.size();
            return true;
        }
        return false;
    }

    static constexpr std::size_t kFallbackCap = 4000;

    Sink        ui_;
    Sink        voice_;
    std::string held_;       // bytes that may still be part of a tag
    std::string fallback_;   // what to speak if no <voice> block ever arrives
    State       state_ = State::Outside;
    bool        saw_voice_ = false;
};

}  // namespace rt
