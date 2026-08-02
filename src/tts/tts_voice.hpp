#pragma once
// -----------------------------------------------------------------------------
// TtsVoice — everything about a synthesis voice that is NOT its ONNX graph:
// the symbol table, the id decorations the model was exported with, and the
// output geometry. Parsed from the voice's sidecar JSON at load time (INIT tier),
// then immutable and shared by reference.
//
// WHY THE SYMBOL TABLE LIVES ON THE VOICE AND NOT IN THE PHONEMIZER. This is the
// hinge the whole IPhonemizer seam turns on. A VITS-family export is trained
// against ONE specific symbol inventory and ONE specific id assignment; a
// phonemizer that carried its own table would be asserting a mapping the model
// never agreed to, and the failure mode is not a crash but subtly wrong
// pronunciation. So the voice owns the table, and a phonemizer's whole job is
// "turn text into symbols THIS table can name". That is exactly what lets a
// grapheme model and a phoneme model sit behind the same interface.
//
// GRAPHEME vs PHONEME is therefore not a flag on the voice — it is which
// IPhonemizer you pair with it. A grapheme voice's table is keyed by characters
// ("п", "a", " "), a phoneme voice's by IPA-ish tokens ("p", "ɪ", "ˈ"). Both are
// just UTF-8 keys here.
// -----------------------------------------------------------------------------
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <string>

namespace blackwell::tts {

// Sentinel for the three optional decoration ids below. Negative because every
// real id in a VITS symbol table is a non-negative index into an embedding.
inline constexpr std::int64_t kNoId = -1;

// The id sequence a VITS export expects is rarely just "one id per symbol".
// Piper-family exports interleave a pad id BETWEEN every symbol and wrap the
// whole sequence in BOS/EOS; omitting either produces audio that is recognisable
// but wrong — clipped onsets, run-together phonemes — which is the kind of bug
// that gets misdiagnosed as a bad voice. All three are described by the sidecar
// JSON, so they are data here, never assumptions in code.
struct TtsVoice {
    // Stable identifier, e.g. "ru_RU-irina-medium". Used as the UI label, the
    // config key, and the on-disk stem for the .onnx / .json pair.
    std::string id;

    // Index into the APP's language table (rt::kLanguages in the translator).
    // Deliberately a bare int and NOT an include of that header: this library
    // must not depend on an app's UI table. The contract that matters is the one
    // language_table.hpp already states — the table is APPEND-ONLY and indices
    // cross threads in atomics, so a voice's index is only meaningful against the
    // build that produced it. kAnyLanguage disables the check entirely (test
    // fixtures, and multilingual voices that accept anything their table names).
    int language_index = kAnyLanguage;

    // Rate of the PCM the graph emits, e.g. 22050 or 24000. The worker resamples
    // from this to the device rate; nothing downstream may assume 16 kHz.
    int sample_rate = 22050;

    // Multi-speaker voices select a timbre by index; single-speaker exports
    // ignore it. 0 is the correct default for both.
    std::int64_t speaker_id = 0;

    // Symbol (UTF-8, one map key per pronounceable unit) -> embedding id.
    // std::less<> makes this transparently comparable, so a lookup by
    // std::string_view over a slice of the input costs no temporary allocation —
    // which matters because the phonemizer does one lookup per codepoint.
    std::map<std::string, std::int64_t, std::less<>> symbol_to_id;

    // Optional decorations, all kNoId when the export does not use them.
    std::int64_t pad_id = kNoId;
    std::int64_t bos_id = kNoId;
    std::int64_t eos_id = kNoId;

    // Piper-family behaviour: emit pad_id between every pair of symbols (and
    // around the ends, inside BOS/EOS). Requires pad_id != kNoId; ignored
    // otherwise. Read from the sidecar JSON, never inferred.
    bool interleave_pad = false;

    // Hard cap on the id sequence handed to the graph. A guard, not a tuning
    // knob: VITS attention is quadratic in input length, so a pathological input
    // (a pasted document, a decode that ran away) would otherwise turn one
    // "speak" click into a multi-second stall on the TTS thread. The worker
    // splits on sentence boundaries well below this; anything that still hits the
    // cap is truncated and counted.
    std::size_t max_input_ids = 2048;

    // language_index value meaning "do not check". See the field above.
    static constexpr int kAnyLanguage = -1;

    // A voice with no symbol table can name nothing, so nothing can be
    // pronounced with it — checked at load (INIT tier) rather than discovered as
    // a stream of EmptyResult at runtime.
    bool is_usable() const noexcept { return !symbol_to_id.empty(); }

    // Whether a request for `requested` should be served by this voice.
    bool accepts_language(int requested) const noexcept {
        return language_index == kAnyLanguage || language_index == requested;
    }
};

}  // namespace blackwell::tts
