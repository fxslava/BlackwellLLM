#pragma once
// Byte-level BPE tokenizer (tiktoken/HuggingFace "ByteLevel BPE" family) used
// by both Llama-3 and Qwen2. Everything model-specific lives in BpeModelData,
// which TokenizerFactory fills from the checkpoint's JSON sidecar files.
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "blackwell/tokenizer.h"

namespace blackwell {

struct BpeModelData {
    // model.vocab: byte-alphabet string -> id.
    std::unordered_map<std::string, int> vocab;
    // model.merges: "left right" -> rank (position in the merges list). Token
    // strings never contain a raw space (it is Ġ in the byte alphabet), so a
    // space-joined key is unambiguous.
    std::unordered_map<std::string, int> merge_ranks;
    // added_tokens: literal content -> id, matched in raw text before BPE.
    std::unordered_map<std::string, int> added_tokens;
    // ids of added tokens flagged "special": true (suppressed by decode()).
    std::unordered_set<int> special_ids;
    // id -> string for vocab and added tokens.
    std::unordered_map<int, std::string> id_to_token;

    // Pre-tokenizer split regex parameter: longest digit run kept together
    // (\p{N}{1,3} for Llama-3 -> 3, bare \p{N} for Qwen2 -> 1).
    int max_digit_run = 3;
    // model.ignore_merges: pre-tokens found verbatim in the vocab skip BPE.
    bool ignore_merges = false;
    // Ids the post-processor template inserts around a single sequence
    // (Llama-3: prefix = [<|begin_of_text|>]; Qwen2: both empty).
    std::vector<int> postproc_prefix_ids;
    std::vector<int> postproc_suffix_ids;

    SpecialTokens special_tokens;
};

class ByteLevelBpeTokenizer final : public ITokenizer {
public:
    ByteLevelBpeTokenizer(BpeModelData data, std::unique_ptr<IChatTemplate> tmpl);

    std::vector<int> encode(const std::string& text, bool add_special_tokens) const override;
    std::string decode(int token_id, bool render_special) const override;
    std::string decode(const std::vector<int>& token_ids, bool render_special) const override;
    int token_to_id(const std::string& token) const override;
    size_t vocab_size() const override { return data_.id_to_token.size(); }
    const SpecialTokens& special_tokens() const override { return data_.special_tokens; }
    bool is_special(int token_id) const override { return data_.special_ids.count(token_id) != 0; }
    const IChatTemplate* chat_template() const override { return chat_template_.get(); }

private:
    // Splits raw text on the tiktoken-family regex; returns byte slices.
    std::vector<std::string> pre_tokenize(const std::string& text) const;
    // Runs rank-based BPE over one pre-token and appends the resulting ids.
    void bpe_encode_piece(const std::string& piece, std::vector<int>& out) const;
    // Appends the encoding of a text segment known to contain no added tokens.
    void encode_plain_segment(const std::string& segment, std::vector<int>& out) const;

    BpeModelData data_;
    std::unique_ptr<IChatTemplate> chat_template_;
};

} // namespace blackwell
