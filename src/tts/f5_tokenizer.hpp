#pragma once
// -----------------------------------------------------------------------------
// F5Tokenizer — UTF-8 text -> F5-TTS character ids, against the vocab.txt that
// SHIPS WITH THE CHECKPOINT.
//
// F5 is a character-level model: there is no BPE, no merges file, no subword
// anything. `vocab.txt` is one character per line and the id IS the line index.
// That makes this class small, and makes the two ways it can go wrong entirely
// silent, so both are handled explicitly here.
//
// =============================================================================
// WHAT THIS REPRODUCES, AND WHAT IT DELIBERATELY DOES NOT
// =============================================================================
// F5's inference path is
//     convert_char_to_pinyin([ref_text + gen_text])  ->  list_str_to_idx(...)
// and convert_char_to_pinyin runs UNCONDITIONALLY, even for a custom
// (non-pinyin) vocabulary. So "just look each character up" is not automatically
// right, and I measured what it actually does to Cyrillic rather than assuming:
//
//   * Cyrillic is 2 bytes/codepoint in UTF-8, so a Russian segment matches
//     neither the pure-ASCII branch (1 byte/cp) nor the CJK branch (3 bytes/cp)
//     and falls through to the mixed branch, where every non-Chinese character
//     is appended UNCHANGED. Verified on real sentences: output == input.
//   * The ONLY transformation that survives for Russian is the `custom_trans`
//     table, which is reproduced exactly in kCustomTrans below. Measured: the
//     sole difference on a test sentence was ';' -> ','.
//   * Tokenising ref and gen SEPARATELY and concatenating is equivalent to
//     tokenising the joined string — verified:
//         convert(ref) + [' '] + convert(gen) == convert(ref + ' ' + gen)
//     which is what licenses F5TtsEngine's `ref_ids ++ gen_ids` layout.
//
// NOT REPRODUCED, and this is the one honest gap: for a PURE-ASCII segment of
// more than one character, convert_char_to_pinyin inserts a space before it when
// the previous character is not one of " :'\"". Predicting that requires
// rjieba's segmenter, which is a Chinese word-segmentation model this project
// will not be linking. In practice it is a no-op for our text, because such
// segments are almost always already preceded by a space (which is in the
// exclusion set). It can only bite on Latin/digits jammed against Cyrillic with
// no separator, e.g. "модельF5". Insert a space yourself in that case.
//
// =============================================================================
// THE TWO SILENT FAILURES
// =============================================================================
// (1) LINE ENDINGS. The shipped vocab.txt is CRLF. Python's text-mode reader
//     folds \r\n to \n before get_tokenizer's `char[:-1]` strips it, so its keys
//     come out clean. A C++ reader that strips only '\n' keys every entry as
//     "<char>\r", matches nothing, and maps EVERY character to the unknown id.
//     The model then receives one long run of id 0 and synthesises fluent
//     nonsense. Both terminators are stripped below, and the ctor cross-checks
//     that ' ' landed on id 0 — which no mis-parsed vocab can satisfy.
//
// (2) VOCAB/CHECKPOINT MISMATCH. Loading a fine-tune against the base vocab
//     maps every character to the wrong embedding row. There is nothing in the
//     text path that can detect this, so the check lives where the evidence is:
//     the exporter compares the vocab's line count against the checkpoint's
//     embedding rows. Pass the vocab.txt that shipped WITH the checkpoint.
//
// UNKNOWN CHARACTERS map to id 0, matching list_str_to_idx's
// `vocab_char_map.get(c, 0)`. Note 0 is not a dedicated <unk> slot — it is the
// SPACE character, so an unknown reads to the model as a pause rather than as a
// special token. That is F5's behaviour, not a choice made here, and it is why
// unknowns are also COUNTED: a caller that silently substitutes pauses for a
// third of its text should be told.
//
// THREADING: immutable after construction. Tokenize() is const and re-entrant;
// any number of threads may share one instance.
// -----------------------------------------------------------------------------
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace blackwell::tts {

class F5Tokenizer {
public:
    // The id an out-of-vocabulary character maps to. F5's own default, and it is
    // the space character — see the header block.
    static constexpr std::int32_t kUnknownId = 0;

    // The tail padding value for the model's text tensor. NOT produced by
    // Tokenize(); the engine fills the unused frame axis with it, and F5's
    // TextEmbedding adds 1 so this becomes the filler token 0. Mirrors
    // kF5TextPadId in f5_tts_engine.hpp.
    static constexpr std::int32_t kPadId = -1;

    // INIT tier: throws std::runtime_error on a missing/unreadable file, on a
    // vocab whose first entry is not ' ' (the mis-parse guard), or on an empty
    // vocab.
    explicit F5Tokenizer(const std::string& vocab_path);

    // UTF-8 text -> ids. Applies the custom_trans normalisation, then one lookup
    // per codepoint. Out-of-vocabulary codepoints become kUnknownId.
    //
    // `out_unknown`, when non-null, receives the number of substituted
    // characters — the signal that the text and the voice disagree about
    // alphabet (Cyrillic text against an English vocab lands here).
    std::vector<std::int32_t> Tokenize(std::string_view utf8_text,
                                       std::size_t* out_unknown = nullptr) const;

    // Same, into a caller-owned buffer (cleared first). For a caller tokenising
    // repeatedly.
    void TokenizeInto(std::string_view utf8_text, std::vector<std::int32_t>& out,
                      std::size_t* out_unknown = nullptr) const;

    // The out-of-vocabulary codepoints seen in `utf8_text`, as UTF-8 strings, in
    // first-appearance order. For diagnostics: "these characters are not in this
    // voice's alphabet" is the actionable form of a bare unknown count.
    std::vector<std::string> UnknownCharacters(std::string_view utf8_text) const;

    // Number of vocabulary entries == the checkpoint's text_num_embeds. The
    // exporter checks this against the checkpoint's embedding rows.
    std::size_t vocab_size() const { return size_; }

    // Lookup for one codepoint; kUnknownId when absent. Exposed for tests.
    std::int32_t IdForCodepoint(std::uint32_t cp) const;

private:
    // Codepoint -> id. Keyed by codepoint rather than by string because every
    // usable vocab entry is exactly one character and lookup happens per
    // character; a multi-codepoint line could never be matched by F5's own
    // per-character iteration either.
    std::unordered_map<std::uint32_t, std::int32_t> map_;
    std::size_t size_ = 0;
};

}  // namespace blackwell::tts
