// -----------------------------------------------------------------------------
// phonemizer.cpp — PassthroughPhonemizer + the UTF-8 scanner it walks with.
// See phonemizer.hpp for the seam's rationale and the threading contract.
//
// THE ID LAYOUT PRODUCED HERE (Piper/VITS convention, all parts optional and all
// driven by the voice's sidecar JSON — never assumed):
//
//     bos, pad, s0, pad, s1, pad, ... , sN, pad, eos
//      ^    ^        ^                        ^   ^
//      |    |        |                        |   +-- voice.eos_id, if != kNoId
//      |    |        +--- one pad BETWEEN and AFTER every symbol, and one
//      |    |             LEADING pad right after bos, when interleave_pad
//      |    +------------ (that leading pad)
//      +----------------- voice.bos_id, if != kNoId
//
// The leading pad is easy to miss and is not cosmetic: exports trained with
// interspersed pads expect the very first symbol to be preceded by one, and
// omitting it clips the utterance onset. Whether a voice wants any of this is
// data, which is why none of it is inferred from the symbol table's contents.
// -----------------------------------------------------------------------------
#include "phonemizer.hpp"

#include <cstddef>

namespace blackwell::tts {

std::size_t utf8_sequence_length(std::string_view text, std::size_t pos) noexcept {
    if (pos >= text.size()) return 0;

    const auto lead = static_cast<unsigned char>(text[pos]);
    std::size_t len = 0;
    if (lead < 0x80u) {
        len = 1;                            // ASCII
    } else if ((lead & 0xE0u) == 0xC0u) {
        len = 2;                            // U+0080..U+07FF -- Cyrillic lives here
    } else if ((lead & 0xF0u) == 0xE0u) {
        len = 3;                            // U+0800..U+FFFF -- CJK, most punctuation
    } else if ((lead & 0xF8u) == 0xF0u) {
        len = 4;                            // U+10000..U+10FFFF
    } else {
        return 0;   // a bare continuation byte (10xxxxxx) or an invalid lead (0xF8+)
    }

    // Truncated at the end of the string: report invalid rather than reading past
    // it. string_view has no terminator to save us here.
    if (text.size() - pos < len) return 0;

    for (std::size_t i = 1; i < len; ++i) {
        if ((static_cast<unsigned char>(text[pos + i]) & 0xC0u) != 0x80u) return 0;
    }
    return len;
}

TtsStatus PassthroughPhonemizer::to_ids(std::string_view utf8_text, int language_index,
                                        const TtsVoice& voice,
                                        std::vector<std::int64_t>& out_ids) noexcept {
    // Cleared FIRST and on every failure path below: the header promises that a
    // non-Success return leaves an empty buffer, so a caller that ignores the
    // status still cannot hand a half-built sequence to the graph.
    out_ids.clear();

    if (!voice.is_usable()) return TtsStatus::InvalidArgument;
    if (!voice.accepts_language(language_index)) return TtsStatus::UnsupportedLanguage;

    const bool use_pad = voice.interleave_pad && voice.pad_id != kNoId;
    const bool use_bos = voice.bos_id != kNoId;
    const bool use_eos = voice.eos_id != kNoId;

    // Reserve room for the trailing eos up front, so a sequence clamped by
    // max_input_ids is still terminated the way the graph expects. A truncated
    // utterance should sound cut short, not malformed.
    const std::size_t tail = use_eos ? 1u : 0u;
    const std::size_t budget = voice.max_input_ids > tail ? voice.max_input_ids - tail : 0u;

    // The whole body is guarded because to_ids() is noexcept and the only thing
    // in it that can throw is vector growth. RUNTIME tier: a bad_alloc here drops
    // one utterance and is counted by the caller, it does not unwind the worker.
    try {
        // Byte count is an upper bound on codepoint count, so this over-reserves
        // for multi-byte text and never under-reserves. Clamped so a pathological
        // input cannot turn a reservation into the stall the cap exists to
        // prevent. The worker reuses one buffer, so this is a startup cost.
        const std::size_t per_symbol = use_pad ? 2u : 1u;
        std::size_t want = utf8_text.size() * per_symbol + 3u;
        if (want > voice.max_input_ids) want = voice.max_input_ids;
        out_ids.reserve(want);

        if (use_bos && out_ids.size() < budget) out_ids.push_back(voice.bos_id);
        if (use_pad && out_ids.size() < budget) out_ids.push_back(voice.pad_id);

        // Accumulated locally and published once: an atomic RMW per character
        // would dominate the cost of this loop for no benefit, since nothing
        // observes the counters mid-call.
        std::uint64_t unknown = 0;
        std::uint64_t invalid = 0;
        std::size_t symbols = 0;
        bool truncated = false;

        std::size_t pos = 0;
        while (pos < utf8_text.size()) {
            const std::size_t len = utf8_sequence_length(utf8_text, pos);
            if (len == 0) {
                // Resynchronise by one byte. Skipping the whole (unknown) sequence
                // is not possible precisely because it is malformed, and advancing
                // by one is what guarantees this loop terminates.
                ++invalid;
                ++pos;
                continue;
            }

            const std::string_view symbol = utf8_text.substr(pos, len);
            pos += len;

            // Transparent comparator (std::less<>) -- looks up the view directly,
            // with no temporary std::string per character.
            const auto it = voice.symbol_to_id.find(symbol);
            if (it == voice.symbol_to_id.end()) {
                ++unknown;
                continue;
            }

            if (out_ids.size() + per_symbol > budget) {
                truncated = true;
                break;
            }
            out_ids.push_back(it->second);
            if (use_pad) out_ids.push_back(voice.pad_id);
            ++symbols;
        }

        if (unknown != 0) unknown_symbols_.fetch_add(unknown, std::memory_order_relaxed);
        if (invalid != 0) invalid_sequences_.fetch_add(invalid, std::memory_order_relaxed);
        if (truncated) truncations_.fetch_add(1, std::memory_order_relaxed);

        // Nothing pronounceable survived. Returning a bare [bos, eos] here would
        // ask the graph to synthesise "nothing" -- which costs a full forward pass
        // and yields a click. EmptyResult tells the worker to skip the utterance.
        if (symbols == 0) {
            out_ids.clear();
            return TtsStatus::EmptyResult;
        }

        if (use_eos) out_ids.push_back(voice.eos_id);
        return TtsStatus::Success;
    } catch (...) {
        out_ids.clear();
        return TtsStatus::RuntimeFailure;
    }
}

}  // namespace blackwell::tts
