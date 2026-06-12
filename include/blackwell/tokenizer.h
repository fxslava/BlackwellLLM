#pragma once
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "blackwell/chat_template.h"

namespace blackwell {

// Special-token ids resolved at load time from tokenizer_config.json /
// generation_config.json / config.json. -1 means "the checkpoint does not
// define this token" (e.g. Qwen2 has no BOS, neither model has UNK).
struct SpecialTokens {
    int bos = -1;
    int eos = -1;
    int unk = -1;
    int pad = -1;
    std::vector<int> stop_ids; // generation stop set (eos_token_id list)
};

// Model-agnostic tokenizer interface. Implementations configure themselves
// entirely from the checkpoint's JSON sidecar files; no token id, vocabulary
// size or template marker is ever hardcoded in engine/host code.
class ITokenizer {
public:
    virtual ~ITokenizer() = default;

    // Text -> ids. Special-token *strings* occurring in `text` (e.g.
    // "<|im_start|>") are recognized and mapped to their ids, matching the
    // HuggingFace behaviour. `add_special_tokens` applies the checkpoint's
    // post-processor template (Llama-3 prepends BOS, Qwen2 adds nothing).
    virtual std::vector<int> encode(const std::string& text,
                                    bool add_special_tokens) const = 0;

    // Id -> text. Special tokens render as "" unless `render_special` is set.
    virtual std::string decode(int token_id, bool render_special = false) const = 0;
    virtual std::string decode(const std::vector<int>& token_ids,
                               bool render_special = false) const = 0;

    // -1 when the string is neither a vocabulary entry nor an added token.
    virtual int token_to_id(const std::string& token) const = 0;

    virtual size_t vocab_size() const = 0;
    virtual const SpecialTokens& special_tokens() const = 0;
    virtual bool is_special(int token_id) const = 0;

    // nullptr when the checkpoint ships no chat template (base models).
    virtual const IChatTemplate* chat_template() const = 0;

    // Full-conversation encoding via the checkpoint's chat template; matches
    // HuggingFace apply_chat_template(). Throws if there is no template.
    std::vector<int> apply_chat_template(const std::vector<ChatMessage>& messages,
                                         bool add_generation_prompt) const;

    // Incremental variants for a streaming chat loop with a persistent KV
    // cache: prelude once, then one call per turn + the generation prompt.
    std::vector<int> encode_chat_prelude(const std::string& system_prompt) const;
    std::vector<int> encode_chat_message(const ChatMessage& msg) const;
    std::vector<int> encode_generation_prompt() const;

    bool is_stop(int token_id) const {
        const auto& s = special_tokens().stop_ids;
        return std::find(s.begin(), s.end(), token_id) != s.end();
    }

protected:
    const IChatTemplate& require_chat_template() const;
};

class TokenizerFactory {
public:
    // `model_dir` is a checkpoint directory in HuggingFace layout. Reads:
    //   tokenizer.json         (required: vocab, merges, added tokens,
    //                           pre-tokenizer regex, post-processor)
    //   tokenizer_config.json  (optional: special-token names, chat template)
    //   generation_config.json (optional: generation stop ids)
    //   config.json            (optional: stop-id fallback)
    static std::unique_ptr<ITokenizer> create(const std::string& model_dir);
};

} // namespace blackwell
