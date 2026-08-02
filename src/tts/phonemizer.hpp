#pragma once
// -----------------------------------------------------------------------------
// IPhonemizer — the text-frontend seam: UTF-8 text in, model input ids out.
//
// WHY THIS IS AN INTERFACE AND NOT A FUNCTION. The two viable TTS model families
// differ in EXACTLY this one place and nowhere else:
//   * a GRAPHEME model (Silero-family) is fed characters and does its own
//     letter-to-sound and stress placement inside the graph;
//   * a PHONEME model (Piper/VITS-family) is fed IPA-ish tokens and expects an
//     external frontend — in practice espeak-ng — to have produced them.
// Everything downstream (the ONNX session, the resampler, the sink, the worker,
// the UI) is byte-for-byte identical between the two. Putting the seam here is
// what makes the model-family choice reversible instead of a rewrite.
//
// AND IT IS A LICENSING FIREWALL. espeak-ng is GPL-3.0. Keeping the frontend
// behind a pure-virtual boundary means the GPL implementation, if it is ever
// built, is ONE leaf .cpp in ONE optional target — not a dependency threaded
// through the synthesis stack. Nothing in this header, or in any core TTS
// header, may include or name an espeak type.
//
// ERROR TIER: RUNTIME (CLAUDE.md extension pattern #4). to_ids() is noexcept and
// returns a TtsStatus. It runs on the TTS worker thread, which by Phase 5 is
// feeding a playback sink with a live deadline behind it; an exception escaping
// here would take down synthesis mid-utterance instead of dropping one line.
//
// THREADING: ONE thread calls to_ids() (the TTS worker). Implementations may
// therefore hold scratch state without synchronisation. Any counters they
// publish must be atomic — that is the UI-readout seam, and the only cross-
// thread surface, exactly as SileroVAD does it.
// -----------------------------------------------------------------------------
#include <atomic>
#include <cstdint>
#include <string_view>
#include <vector>

#include "tts_status.hpp"
#include "tts_voice.hpp"

namespace blackwell::tts {

class IPhonemizer {
public:
    virtual ~IPhonemizer() = default;

    IPhonemizer(const IPhonemizer&) = delete;
    IPhonemizer& operator=(const IPhonemizer&) = delete;

    // For logs and the settings panel ("Frontend: passthrough (graphemes)").
    virtual const char* name() const noexcept = 0;

    // Converts `utf8_text` into the id sequence `voice`'s graph expects,
    // INCLUDING whatever pad/BOS/EOS decoration the voice declares.
    //
    // `out_ids` is CLEARED first and is the caller's reusable buffer — the worker
    // keeps one across utterances so the steady state performs no allocation.
    // On any non-Success return it is left EMPTY, so a caller that ignores the
    // status still cannot feed a half-built sequence to the model.
    //
    // `language_index` indexes the app's language table and is checked against
    // the voice (see TtsVoice::accepts_language). It is passed separately rather
    // than read off the voice because the CALLER's intent and the voice's
    // capability are different facts, and a mismatch is a real, reportable
    // condition (UnsupportedLanguage) rather than something to silently ignore.
    virtual TtsStatus to_ids(std::string_view utf8_text, int language_index,
                             const TtsVoice& voice,
                             std::vector<std::int64_t>& out_ids) noexcept = 0;

protected:
    IPhonemizer() = default;
};

// -----------------------------------------------------------------------------
// PassthroughPhonemizer — the grapheme frontend: one UTF-8 codepoint = one
// symbol, looked up directly in the voice's table. No letter-to-sound rules, no
// dictionary, no stress model, no third-party dependency of any kind.
//
// This is not a stub. For a grapheme-input voice it is the CORRECT and complete
// frontend: such models are trained on exactly this mapping and do their own
// letter-to-sound work internally. It is a placeholder only if the voice is a
// phoneme model, in which case it will produce a stream of unknown symbols —
// loudly, via unknown_symbols(), rather than silently mispronouncing.
//
// WHAT IT DELIBERATELY DOES NOT DO:
//   * No case folding. ASCII tolower() would corrupt Cyrillic — the primary
//     target language of this translator — and a correct Unicode fold needs ICU.
//     The voice's table is expected to carry the casing the model was exported
//     with; a mismatch shows up as unknown symbols, which is visible, instead of
//     as wrong audio, which is not.
//   * No normalisation, no number/abbreviation expansion ("42" -> "forty-two").
//     That is a separate, per-language text-normalisation concern which belongs
//     in front of any phonemizer, not inside one of them.
//   * No whitespace collapsing. Space is a symbol like any other; if the voice's
//     table names it, it is pronounced as the pause it was trained to be.
// -----------------------------------------------------------------------------
class PassthroughPhonemizer final : public IPhonemizer {
public:
    PassthroughPhonemizer() = default;

    const char* name() const noexcept override { return "passthrough(graphemes)"; }

    TtsStatus to_ids(std::string_view utf8_text, int language_index, const TtsVoice& voice,
                     std::vector<std::int64_t>& out_ids) noexcept override;

    // ---- cross-thread counters (the panel readout) --------------------------
    // All monotone. Nonzero unknown_symbols with plausible audio usually means a
    // casing or normalisation gap; unknown_symbols climbing at roughly one per
    // character means the voice is a PHONEME model and this frontend is the wrong
    // one for it — the single most likely misconfiguration, so it is worth
    // surfacing rather than inferring from bad audio.
    std::uint64_t unknown_symbols() const noexcept {
        return unknown_symbols_.load(std::memory_order_relaxed);
    }
    // Malformed UTF-8 byte positions skipped. Should be 0 for text that came from
    // the tokenizer; nonzero means something upstream mangled an encoding.
    std::uint64_t invalid_sequences() const noexcept {
        return invalid_sequences_.load(std::memory_order_relaxed);
    }
    // Requests clamped by TtsVoice::max_input_ids. Nonzero means the worker's
    // sentence splitter let something oversized through.
    std::uint64_t truncations() const noexcept {
        return truncations_.load(std::memory_order_relaxed);
    }

private:
    std::atomic<std::uint64_t> unknown_symbols_{0};
    std::atomic<std::uint64_t> invalid_sequences_{0};
    std::atomic<std::uint64_t> truncations_{0};
};

// Length in bytes of the UTF-8 sequence starting at `text[pos]`, or 0 if the
// bytes there are not a well-formed sequence (bad lead byte, missing or
// malformed continuation, or a sequence running off the end of the string).
//
// Exposed because it is the one piece of this file worth testing directly:
// getting it wrong on a 2-byte Cyrillic codepoint means every Russian utterance
// degrades into unknown symbols, and that is the project's primary use case.
// Overlong encodings and surrogate halves are NOT rejected — a symbol-table
// lookup is the real validity check here, and it rejects them by simply not
// naming them.
std::size_t utf8_sequence_length(std::string_view text, std::size_t pos) noexcept;

}  // namespace blackwell::tts
