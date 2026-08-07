#pragma once
// -----------------------------------------------------------------------------
// stress_marker.hpp — the accentor: where the '+' marks come from.
//
// WHY A WHOLE STAGE FOR ONE PUNCTUATION MARK. Russian stress is not predictable
// from spelling ("за+мок" a castle, "замо+к" a lock, same letters) and it is
// not marked in ordinary text. A character-level TTS therefore has to guess,
// and it guesses from whatever the fine-tune saw -- which for the Russian F5
// checkpoints is text where stress WAS marked. Hand such a model unmarked
// prose and the failure is not a wrong syllable here and there: whole words
// come out with the emphasis on the wrong end, which is the single most
// "robotic" artefact a Russian listener will name.
//
// So the pipeline marks the text itself, and the mark is a literal '+' next to
// the stressed vowel: "золота+я ры+бка".
//
// =============================================================================
// WHICH SIDE OF THE VOWEL, AND WHY IT IS A KNOB
// =============================================================================
// Two conventions are in circulation and they are NOT interchangeable, because
// the model reads the '+' as a character in a sequence:
//     AfterVowel   "ры+бка"   -- the mark follows the vowel it stresses
//     BeforeVowel  "р+ыбка"   -- the ruaccent convention, mark precedes it
// A checkpoint understands exactly the one it was trained with; fed the other,
// every marked word acquires a '+' in a position the model has never seen, and
// since an out-of-vocabulary character maps to id 0 -- which in F5 IS the space
// character (f5_tokenizer.hpp) -- the audible result is a PAUSE INSIDE A WORD.
// A stutter, not a mispronunciation.
//
// AfterVowel is the default because that is the notation this project's voice
// packs use. Changing it is one field, and getting it wrong is loud rather than
// subtle, which is the failure mode to want.
//
// =============================================================================
// WHAT THE DEFAULT ACCENTOR KNOWS, AND WHAT IT REFUSES TO GUESS
// =============================================================================
// DictionaryAccentor marks a word when it can be RIGHT about it, and leaves it
// alone otherwise. In order:
//
//   1. an explicit dictionary entry (the built-in table, plus anything loaded
//      from a file) -- checked first, because compounds like "трёхсо+тый" are
//      exactly where the ё rule below is wrong;
//   2. 'ё', which carries the stress in every simple Russian word;
//   3. a single-vowel word, which is unambiguous and therefore needs no mark
//      at all (marking it is allowed but off by default -- it only adds
//      characters for the model to read);
//   4. otherwise: NOTHING. An unmarked word falls back on the model's own
//      prior, which is a coin flip; a WRONGLY marked word overrides that prior
//      with a confident error. Silence is the better failure.
//
// The built-in table is a seed, not a lexicon. It covers what this pipeline
// GENERATES rather than what a user might say -- every numeral, scale word and
// unit that text_normalizer.hpp can emit, so "сто два+дцать три проце+нта"
// comes out right end to end -- plus the highest-frequency function and
// assistant vocabulary. Real coverage comes from LoadDictionary().
//
// IAccentor IS A SEAM, for the same reason ISynthesizer is one: an ONNX
// accentor (RUAccent and friends are small BERT-ish taggers) is the correct
// long-term answer, and it must be possible to drop one in without touching
// the bridge, the chunker or the tokenizer. Anything that turns UTF-8 text into
// the same text with '+' marks satisfies this interface.
//
// THREADING: Accentuate() is const and re-entrant once construction/loading is
// done; any number of threads may share one instance. LoadDictionary() mutates
// and must not race a call to Accentuate().
// -----------------------------------------------------------------------------
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>

namespace blackwell::tts {

enum class PlusPlacement : std::int32_t {
    AfterVowel = 0,   // "ры+бка"  -- this project's voice packs
    BeforeVowel = 1,  // "р+ыбка"  -- the ruaccent convention
};

struct AccentOptions {
    PlusPlacement placement = PlusPlacement::AfterVowel;

    // Mark words with exactly one vowel. Off: a monosyllable has nowhere else
    // for the stress to go, so the mark carries no information and only costs
    // the model a character.
    bool mark_monosyllables = false;

    // Treat 'ё' as stressed when the word is not in the dictionary. True is
    // the standard orthographic rule; the exceptions are compounds, which is
    // why the dictionary is consulted first.
    bool trust_yo = true;
};

// The seam. UTF-8 in, the same UTF-8 with stress marks out.
class IAccentor {
public:
    virtual ~IAccentor() = default;
    IAccentor(const IAccentor&) = delete;
    IAccentor& operator=(const IAccentor&) = delete;

    // MUST be a pure rewrite: no words added, none removed, nothing reordered.
    // The only permitted difference is the inserted marks -- the tokenizer and
    // the duration model downstream both count characters.
    [[nodiscard]] virtual std::string Accentuate(std::string_view utf8) const = 0;

protected:
    IAccentor() = default;
};

class DictionaryAccentor final : public IAccentor {
public:
    explicit DictionaryAccentor(AccentOptions opts = {});

    [[nodiscard]] std::string Accentuate(std::string_view utf8) const override;

    // Merges entries from a UTF-8 text file, one word per line, in either
    //     золота+я        (marked form -- also what this class emits)
    //     золотая 4       (word, then the 1-based index of the stressed vowel)
    // Blank lines and lines starting with '#' are skipped. Later entries win,
    // so a loaded file overrides the built-in table.
    //
    // Returns the number of entries accepted. INIT tier: throws
    // std::runtime_error only when the file cannot be opened -- a malformed
    // LINE is skipped rather than fatal, because a 100k-word dump with three
    // bad rows should still load.
    std::size_t LoadDictionary(const std::string& path);

    // One entry. `word` is lowercased internally; `vowel_index` is 1-based over
    // the word's vowels. An index outside the word's vowel count is ignored.
    void Add(std::string_view word, int vowel_index);

    std::size_t size() const noexcept { return dict_.size(); }
    const AccentOptions& options() const noexcept { return opts_; }

    // Marks ONE word (no surrounding punctuation). Exposed for tests and for
    // callers that already tokenise; Accentuate() is this plus a word scan.
    [[nodiscard]] std::string AccentuateWord(std::string_view word) const;

    // Parses a marked form ("золота+я") into its 1-based vowel index, or 0 when
    // there is no usable mark. Accepts BOTH conventions: a '+' whose neighbours
    // disambiguate it (vowel after but not before -> BeforeVowel notation) is
    // read the way it was written, so a ruaccent dump loads unchanged.
    [[nodiscard]] static int VowelIndexOfMarked(std::string_view marked);

private:
    AccentOptions opts_;
    // Lowercased word -> 1-based index of the stressed vowel. Keyed by word
    // rather than by lemma: this is a pronunciation table, and inflected forms
    // move the stress ("рубль" / "рубля+"), so they are separate entries.
    std::unordered_map<std::string, int> dict_;
};

}  // namespace blackwell::tts
