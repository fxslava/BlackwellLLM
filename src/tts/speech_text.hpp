#pragma once
// -----------------------------------------------------------------------------
// speech_text.hpp — NormalizeForSpeech: the last thing that touches a chunk
// before it becomes character ids.
//
// WHY IT IS NOT THE MODEL'S JOB. The reply is split upstream into a spoken half
// and a written half (`<voice>` / `<ui>`), and the spoken half is asked to be
// plain prose. That handles the COMMON case and none of the residual ones, all
// of which are real:
//
//   * a model that ignores the contract and emits Markdown into <voice> anyway
//     -- an 8B backbone does this often enough that "the TTS reads asterisks
//     out loud" is a user-visible bug, not a hypothetical;
//   * the fallback path, where no <voice> block was produced at all and the
//     Markdown answer IS what gets spoken (reply_split.hpp);
//   * stress marks, which no prompt can regularise because the two notations
//     look identical on screen and the model has no idea which one this
//     checkpoint was trained on.
//
// So the prompt asks, and this guarantees. A normalisation the pipeline can
// PROVE is worth more than an instruction the model may follow.
//
// STRESS MARKS, AND WHY THEY NEED A POLICY RATHER THAN A STRIP
//   Russian TTS text carries lexical stress in one of two conventions:
//       "хорошо́"   U+0301 COMBINING ACUTE ACCENT, AFTER the stressed vowel
//       "хорош+о"  a '+' PRECEDING the stressed vowel (the ruaccent convention)
//   F5 is character-level, so whichever one the checkpoint was fine-tuned with is
//   the only one it understands -- and an unknown character does not fail loudly,
//   it maps to id 0, which IS the space character (f5_tokenizer.hpp). A stress
//   mark the voice does not know therefore inserts a PAUSE INSIDE A WORD. That
//   is the actual symptom: not a mispronunciation, a stutter.
//
//   Hence a policy rather than a hardcoded rule. The caller states which
//   convention its checkpoint speaks; this converts between them and removes
//   what neither wants. Strip is the default because the base vocabulary is the
//   common case and an unstressed word is merely flat, whereas a wrongly-placed
//   pause is unintelligible.
//
// NOT A MARKDOWN PARSER, and deliberately so. It removes the marks that are
// AUDIBLE as noise (emphasis runs, code fences, heading hashes, bullets, link
// syntax) and leaves everything else alone. A real parser would have to decide
// what a table or a nested blockquote SOUNDS like, which is a product question
// nobody has answered; dropping the punctuation and reading the words is the
// honest approximation.
//
// PURE FUNCTION: no state, no allocation beyond the returned string, callable
// from any thread. It runs per chunk on the TTS worker, where it is dwarfed by
// a ~200 ms diffusion step, so it is written for clarity rather than speed.
// -----------------------------------------------------------------------------
#include <string>
#include <string_view>

namespace blackwell::tts {

// What this voice's vocabulary understands. Get it wrong in the direction of
// Strip and speech is flat; get it wrong in the other direction and every
// marked word acquires a pause in the middle.
enum class StressPolicy {
    // Remove both notations. THE DEFAULT: correct for every checkpoint that was
    // not trained on stress, and merely suboptimal for the ones that were.
    Strip,
    // Keep stress as U+0301 following the vowel; converts '+' notation into it.
    Combining,
    // Keep stress as '+' preceding the vowel; converts U+0301 into it.
    Plus,
};

struct SpeechTextOptions {
    StressPolicy stress = StressPolicy::Strip;

    // Strip Markdown emphasis/code/heading/bullet/link syntax. Off only makes
    // sense for a caller feeding text it produced itself.
    bool strip_markdown = true;

    // Collapse runs of whitespace to one space and trim the ends. The chunker
    // already trims chunk edges; this also flattens the newlines a list leaves
    // behind, which otherwise reach the model as pauses of arbitrary length.
    bool collapse_whitespace = true;

    // Drop characters outside letters/digits/punctuation the voice can read --
    // emoji, box drawing, private-use. They tokenise to id 0 (a space), so
    // leaving them in inserts pauses; there is nothing to gain by keeping them.
    bool drop_unspeakable_symbols = true;
};

// UTF-8 in, UTF-8 out. Malformed input is passed through byte-for-byte rather
// than repaired: this is not the layer that owns code-point integrity (the
// chunker is), and a normaliser that silently ate bytes would hide that.
[[nodiscard]] std::string NormalizeForSpeech(std::string_view utf8,
                                             const SpeechTextOptions& opts = {});

}  // namespace blackwell::tts
