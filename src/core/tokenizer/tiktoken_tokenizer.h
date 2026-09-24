#pragma once
// Raw-byte BPE tokenizer of the OpenAI tiktoken family, as shipped by GLM-4 /
// ChatGLM ("tokenizer.model": base64(token_bytes) + " " + rank, one per line).
//
// Why this exists beside ByteLevelBpeTokenizer -- the two are NOT the same BPE:
//
//   ByteLevelBpeTokenizer (HuggingFace tokenizer.json, Llama-3 / Qwen2)
//     symbols are bytes remapped into the printable GPT-2 alphabet, and the
//     merge order comes from an explicit `merges` list keyed "left right".
//
//   TiktokenTokenizer (this file, GLM-4)
//     symbols are RAW bytes, and there is no merges list at all: the merge
//     order is implied by the rank of the *concatenated* token in the vocab.
//     Lower rank == merged earlier. That single difference is why the existing
//     tokenizer cannot read this checkpoint.
//
// Both split text on the same pre-tokenizer regex (unicode::match_pretoken).
#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "blackwell/tokenizer.h"

namespace blackwell {

// Decodes standard base64 (RFC 4648 §4, with or without '=' padding). Throws
// std::runtime_error on any character outside the alphabet or on a truncated
// group -- a corrupt tokenizer.model must fail loudly at load (INIT tier),
// never silently produce a wrong vocabulary.
std::string base64_decode(std::string_view in);

struct TiktokenModelData {
    // Raw token bytes -> rank. Ranks are dense over [0, ranks.size()).
    std::unordered_map<std::string, int> encoder;
    // rank -> raw token bytes; index IS the id for every non-special token.
    std::vector<std::string> decoder;

    // Special tokens live ABOVE the base vocabulary and never participate in
    // BPE: literal content -> id, matched greedily in raw text before merging.
    std::unordered_map<std::string, int> specials;
    std::unordered_map<int, std::string> special_by_id;

    // Prepended by encode(..., add_special_tokens = true). For GLM-4 this is
    // get_prefix_tokens(): [gMASK], <sop>.
    std::vector<int> prefix_ids;

    // Longest digit run the pre-tokenizer keeps together (GLM-4: \p{N}{1,3}).
    int max_digit_run = 3;

    SpecialTokens special_tokens;
};

class TiktokenTokenizer final : public ITokenizer {
public:
    TiktokenTokenizer(TiktokenModelData data, std::unique_ptr<IChatTemplate> tmpl);

    // Parses a "tokenizer.model" rank file. Throws std::runtime_error on a
    // malformed line, a bad base64 payload, a non-integer or duplicate rank.
    static TiktokenModelData load_rank_file(const std::string& path);

    // Cheap format probe for TokenizerFactory: true when the file's first
    // non-empty lines all look like "<base64> <integer>". Never throws.
    static bool looks_like_rank_file(const std::string& path);

    std::vector<int> encode(const std::string& text, bool add_special_tokens) const override;
    std::string decode(int token_id, bool render_special) const override;
    std::string decode(const std::vector<int>& token_ids, bool render_special) const override;
    int token_to_id(const std::string& token) const override;
    size_t vocab_size() const override { return data_.decoder.size() + data_.specials.size(); }
    const SpecialTokens& special_tokens() const override { return data_.special_tokens; }
    bool is_special(int token_id) const override {
        return data_.special_by_id.count(token_id) != 0;
    }
    const IChatTemplate* chat_template() const override { return chat_template_.get(); }

private:
    // Rank-merge BPE over the raw bytes of one pre-token; appends ids to `out`.
    void bpe_encode_piece(std::string_view piece, std::vector<int>& out) const;
    // Encodes a span known to contain no special-token literal.
    void encode_plain_segment(std::string_view segment, std::vector<int>& out) const;
    // Rank of `piece`, or -1 when it is not a vocabulary entry.
    int rank_of(std::string_view piece) const;

    TiktokenModelData data_;
    std::unique_ptr<IChatTemplate> chat_template_;

    // Greedy special-token scan accelerators, both derived in the ctor. The
    // scan probes only at bytes that actually start some special literal
    // ("<" and "[" for GLM-4), and never probes longer than the longest one --
    // so text with no special tokens costs one array lookup per byte.
    std::array<bool, 256> special_first_byte_{};
    size_t max_special_len_ = 0;
};

} // namespace blackwell
