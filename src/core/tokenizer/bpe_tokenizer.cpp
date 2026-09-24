#include "tokenizer/bpe_tokenizer.h"

#include <limits>
#include <stdexcept>

#include "tokenizer/unicode.h"

namespace blackwell {

ByteLevelBpeTokenizer::ByteLevelBpeTokenizer(BpeModelData data,
                                             std::unique_ptr<IChatTemplate> tmpl)
    : data_(std::move(data)), chat_template_(std::move(tmpl)) {
    if (data_.vocab.empty())
        throw std::runtime_error("ByteLevelBpeTokenizer: empty vocabulary");
}

std::vector<std::string> ByteLevelBpeTokenizer::pre_tokenize(const std::string& text) const {
    std::vector<size_t> off;
    const std::vector<uint32_t> cps = unicode::decode_utf8(text, &off);
    off.push_back(text.size());

    std::vector<std::string> pieces;
    size_t i = 0;
    while (i < cps.size()) {
        const size_t end = unicode::match_pretoken(cps, i, data_.max_digit_run);
        pieces.push_back(text.substr(off[i], off[end] - off[i]));
        i = end;
    }
    return pieces;
}

void ByteLevelBpeTokenizer::bpe_encode_piece(const std::string& piece,
                                             std::vector<int>& out) const {
    const uint32_t* byte_map = unicode::byte_to_unicode_table();

    // Byte-level start state: one symbol per raw input byte, remapped into the
    // printable BPE alphabet.
    std::vector<std::string> symbols;
    symbols.reserve(piece.size());
    std::string mapped_piece;
    for (const unsigned char b : piece) {
        std::string s;
        unicode::append_utf8(s, byte_map[b]);
        mapped_piece += s;
        symbols.push_back(std::move(s));
    }

    if (data_.ignore_merges) {
        const auto it = data_.vocab.find(mapped_piece);
        if (it != data_.vocab.end()) {
            out.push_back(it->second);
            return;
        }
    }

    // Standard BPE: repeatedly merge the adjacent pair with the lowest rank in
    // the checkpoint's merges list. Vocabulary ids are NOT a valid merge order
    // (Qwen2's ids do not even correlate with it), so only merge_ranks decides.
    while (symbols.size() > 1) {
        int best_rank = std::numeric_limits<int>::max();
        size_t best_idx = 0;
        for (size_t k = 0; k + 1 < symbols.size(); ++k) {
            const auto it = data_.merge_ranks.find(symbols[k] + " " + symbols[k + 1]);
            if (it != data_.merge_ranks.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = k;
            }
        }
        if (best_rank == std::numeric_limits<int>::max()) break;

        symbols[best_idx] += symbols[best_idx + 1];
        symbols.erase(symbols.begin() + best_idx + 1);
    }

    for (const auto& s : symbols) {
        const auto it = data_.vocab.find(s);
        if (it != data_.vocab.end()) {
            out.push_back(it->second);
        } else if (data_.special_tokens.unk >= 0) {
            out.push_back(data_.special_tokens.unk);
        } else {
            throw std::runtime_error(
                "ByteLevelBpeTokenizer: symbol not in vocabulary and no UNK token "
                "defined (corrupt vocab/merges?): \"" + s + "\"");
        }
    }
}

void ByteLevelBpeTokenizer::encode_plain_segment(const std::string& segment,
                                                 std::vector<int>& out) const {
    for (const auto& piece : pre_tokenize(segment))
        bpe_encode_piece(piece, out);
}

std::vector<int> ByteLevelBpeTokenizer::encode(const std::string& text,
                                               bool add_special_tokens) const {
    std::vector<int> out;
    if (add_special_tokens)
        out.insert(out.end(), data_.postproc_prefix_ids.begin(),
                   data_.postproc_prefix_ids.end());

    // Added tokens (e.g. "<|im_start|>") are matched literally in the raw text
    // before any pre-tokenization, exactly like HuggingFace's AddedVocabulary.
    // Leftmost match wins; ties on position go to the longest token.
    size_t pos = 0;
    while (pos < text.size()) {
        size_t best_start = std::string::npos;
        size_t best_len = 0;
        int best_id = -1;
        for (const auto& [content, id] : data_.added_tokens) {
            const size_t p = text.find(content, pos);
            if (p == std::string::npos) continue;
            if (p < best_start || (p == best_start && content.size() > best_len)) {
                best_start = p;
                best_len = content.size();
                best_id = id;
            }
        }
        if (best_start == std::string::npos) break;

        if (best_start > pos)
            encode_plain_segment(text.substr(pos, best_start - pos), out);
        out.push_back(best_id);
        pos = best_start + best_len;
    }
    if (pos < text.size())
        encode_plain_segment(text.substr(pos), out);

    if (add_special_tokens)
        out.insert(out.end(), data_.postproc_suffix_ids.begin(),
                   data_.postproc_suffix_ids.end());
    return out;
}

std::string ByteLevelBpeTokenizer::decode(int token_id, bool render_special) const {
    const auto it = data_.id_to_token.find(token_id);
    if (it == data_.id_to_token.end()) return "";

    if (is_special(token_id)) return render_special ? it->second : "";

    // Added-but-not-special tokens store literal content, not byte-alphabet
    // strings, so they must bypass the reverse byte mapping.
    for (const auto& [content, id] : data_.added_tokens)
        if (id == token_id) return content;

    std::string out;
    for (const uint32_t cp : unicode::decode_utf8(it->second)) {
        const int b = unicode::unicode_to_byte(cp);
        if (b >= 0) out.push_back(static_cast<char>(b));
        else unicode::append_utf8(out, cp); // not in the alphabet: keep as-is
    }
    return out;
}

std::string ByteLevelBpeTokenizer::decode(const std::vector<int>& token_ids,
                                          bool render_special) const {
    std::string out;
    for (const int id : token_ids) out += decode(id, render_special);
    return out;
}

int ByteLevelBpeTokenizer::token_to_id(const std::string& token) const {
    if (const auto it = data_.added_tokens.find(token); it != data_.added_tokens.end())
        return it->second;
    if (const auto it = data_.vocab.find(token); it != data_.vocab.end())
        return it->second;
    return -1;
}

// ---- ITokenizer convenience layer ------------------------------------------

const IChatTemplate& ITokenizer::require_chat_template() const {
    const IChatTemplate* tmpl = chat_template();
    if (!tmpl)
        throw std::runtime_error(
            "ITokenizer: checkpoint ships no chat_template in tokenizer_config.json; "
            "chat-style encoding is unavailable for this model");
    return *tmpl;
}

std::vector<int> ITokenizer::apply_chat_template(const std::vector<ChatMessage>& messages,
                                                 bool add_generation_prompt) const {
    // The rendered text embeds special tokens (BOS included where the template
    // family demands it), so the post-processor must not add another one.
    return encode(require_chat_template().render_conversation(messages, add_generation_prompt),
                  /*add_special_tokens=*/false);
}

std::vector<int> ITokenizer::encode_chat_prelude(const std::string& system_prompt) const {
    const IChatTemplate& tmpl = require_chat_template();
    const std::string& sys =
        system_prompt.empty() ? tmpl.default_system_prompt() : system_prompt;
    return encode(tmpl.render_prelude(sys), /*add_special_tokens=*/false);
}

std::vector<int> ITokenizer::encode_chat_message(const ChatMessage& msg) const {
    return encode(require_chat_template().render_message(msg), /*add_special_tokens=*/false);
}

std::vector<int> ITokenizer::encode_generation_prompt() const {
    return encode(require_chat_template().render_generation_prompt(),
                  /*add_special_tokens=*/false);
}

} // namespace blackwell
