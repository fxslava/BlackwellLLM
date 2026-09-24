#pragma once
#include <memory>
#include <string>
#include <vector>

namespace blackwell {

// One turn of a conversation, mirroring the HuggingFace chat message schema.
struct ChatMessage {
    std::string role;    // "system" | "user" | "assistant" | ...
    std::string content;
};

// Renders a conversation into the model's prompt DSL (ChatML, Llama-3 headers,
// ...). Templates produce *text*; the tokenizer turns that text into ids, so a
// template never needs to know a single token id. Concrete implementations are
// selected by ChatTemplateFactory from the Jinja source string shipped in
// tokenizer_config.json (the structural families are implemented natively; a
// full Jinja interpreter is deliberately out of scope).
class IChatTemplate {
public:
    virtual ~IChatTemplate() = default;

    virtual std::string name() const = 0;

    // Conversation prefix: textual BOS (if the template embeds one) plus the
    // system block for `system_prompt`. An empty prompt falls back to the
    // template's default; templates without a default may render nothing.
    virtual std::string render_prelude(const std::string& system_prompt) const = 0;

    // One non-system message, fully framed (role header + terminator).
    virtual std::string render_message(const ChatMessage& msg) const = 0;

    // The cue that makes the model start speaking as the assistant.
    virtual std::string render_generation_prompt() const = 0;

    // System prompt injected when the conversation does not begin with one.
    virtual const std::string& default_system_prompt() const = 0;

    // prelude + messages [+ generation prompt], composed from the parts above.
    std::string render_conversation(const std::vector<ChatMessage>& messages,
                                    bool add_generation_prompt) const;
};

class ChatTemplateFactory {
public:
    // Detects the template family from the Jinja source in tokenizer_config.json.
    // `bos_token` is the textual BOS some templates (Llama-3) emit themselves.
    // Returns nullptr when `jinja_source` is empty; throws on an unknown family.
    static std::unique_ptr<IChatTemplate> from_jinja_source(const std::string& jinja_source,
                                                            const std::string& bos_token);

    // Fallback for checkpoints that ship no chat_template but whose config.json
    // `model_type` names a family with a fixed, non-negotiable prompt DSL
    // (GLM-4). Returns nullptr for any other model_type -- a base model without
    // a template is legitimate, so this never throws.
    static std::unique_ptr<IChatTemplate> from_model_type(const std::string& model_type);
};

} // namespace blackwell
