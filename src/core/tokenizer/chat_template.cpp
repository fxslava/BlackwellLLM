// Native implementations of the chat-template families the engine supports.
// The checkpoint's tokenizer_config.json carries a Jinja template; embedding a
// Jinja interpreter is out of scope, so ChatTemplateFactory instead detects
// the structural family (ChatML / Llama-3 headers) from the Jinja source and
// extracts the family's variable parts (default system prompt, knowledge
// cutoff, date default) from the template's string literals. Token ids never
// appear here: templates render text, the tokenizer resolves ids.
#include "blackwell/chat_template.h"

#include <stdexcept>

namespace blackwell {

namespace {

std::string trim_copy(const std::string& s) {
    const char* ws = " \t\r\n\f\v";
    const size_t b = s.find_first_not_of(ws);
    if (b == std::string::npos) return "";
    const size_t e = s.find_last_not_of(ws);
    return s.substr(b, e - b + 1);
}

// Captures the literal between `prefix` and `suffix` in the Jinja source.
// Returns "" when absent.
std::string capture_between(const std::string& src, const std::string& prefix,
                            const std::string& suffix, size_t from = 0) {
    const size_t p = src.find(prefix, from);
    if (p == std::string::npos) return "";
    const size_t start = p + prefix.size();
    const size_t e = src.find(suffix, start);
    if (e == std::string::npos) return "";
    return src.substr(start, e - start);
}

// ---- ChatML (Qwen2 / Qwen2.5 and friends) ----------------------------------
//   <|im_start|>{role}\n{content}<|im_end|>\n
class ChatMLTemplate final : public IChatTemplate {
public:
    explicit ChatMLTemplate(std::string default_system)
        : default_system_(std::move(default_system)) {}

    std::string name() const override { return "chatml"; }

    std::string render_prelude(const std::string& system_prompt) const override {
        if (system_prompt.empty()) return "";
        return "<|im_start|>system\n" + system_prompt + "<|im_end|>\n";
    }

    std::string render_message(const ChatMessage& msg) const override {
        return "<|im_start|>" + msg.role + "\n" + msg.content + "<|im_end|>\n";
    }

    std::string render_generation_prompt() const override {
        return "<|im_start|>assistant\n";
    }

    const std::string& default_system_prompt() const override { return default_system_; }

private:
    std::string default_system_;
};

// ---- Llama-3 instruct header format -----------------------------------------
//   {bos}<|start_header_id|>{role}<|end_header_id|>\n\n{content|trim}<|eot_id|>
// The system block is always emitted (even when the prompt is empty) and, for
// Llama-3.1-style templates, carries a knowledge-cutoff/date preamble.
class Llama3Template final : public IChatTemplate {
public:
    Llama3Template(std::string bos_token, std::string system_preamble)
        : bos_token_(std::move(bos_token)),
          system_preamble_(std::move(system_preamble)) {}

    std::string name() const override { return "llama3"; }

    std::string render_prelude(const std::string& system_prompt) const override {
        return bos_token_ + "<|start_header_id|>system<|end_header_id|>\n\n" +
               system_preamble_ + trim_copy(system_prompt) + "<|eot_id|>";
    }

    std::string render_message(const ChatMessage& msg) const override {
        return "<|start_header_id|>" + msg.role + "<|end_header_id|>\n\n" +
               trim_copy(msg.content) + "<|eot_id|>";
    }

    std::string render_generation_prompt() const override {
        return "<|start_header_id|>assistant<|end_header_id|>\n\n";
    }

    const std::string& default_system_prompt() const override { return default_system_; }

private:
    std::string bos_token_;
    std::string system_preamble_;
    std::string default_system_; // Llama-3 has none; the system block itself is unconditional.
};

// ---- GLM-4 / ChatGLM --------------------------------------------------------
//   [gMASK]<sop>  then, per turn,  <|{role}|>{metadata}\n{content}
// Three things make this family unlike the two above:
//   * the prelude is textual and UNCONDITIONAL -- "[gMASK]<sop>" opens every
//     conversation, system block or not (the tokenizer maps both literals to
//     ids, so apply_chat_template must not also prepend a numeric prefix);
//   * turns have NO terminator: the next "<|role|>" marker is what ends them;
//   * the generation prompt is a bare "<|assistant|>" with NO trailing newline,
//     unlike the "\n" that follows the marker in a *rendered* turn. The model
//     emits that newline itself -- adding one here shifts every generation by a
//     token and is the classic way to break GLM-4 output.
// `metadata` (the tool-call slot) is always empty for plain chat, so the
// rendered marker line is exactly "<|role|>\n".
class GlmChatTemplate final : public IChatTemplate {
public:
    std::string name() const override { return "glm4"; }

    std::string render_prelude(const std::string& system_prompt) const override {
        std::string out = "[gMASK]<sop>";
        if (!system_prompt.empty()) out += "<|system|>\n" + system_prompt;
        return out;
    }

    std::string render_message(const ChatMessage& msg) const override {
        // {% if item['content'] %} -- an empty turn renders nothing at all.
        if (msg.content.empty()) return "";
        return "<|" + msg.role + "|>\n" + msg.content;
    }

    std::string render_generation_prompt() const override { return "<|assistant|>"; }

    const std::string& default_system_prompt() const override { return default_system_; }

private:
    std::string default_system_; // GLM-4 ships none; the tool-call preamble is not one.
};

// Default system prompt of a ChatML template, e.g. Qwen2.5:
//   {{- '<|im_start|>system\nYou are Qwen, ...<|im_end|>\n' }}
// In the Jinja source "\n" is the two-character escape, which makes the
// literal easy to lift out: take what follows "<|im_start|>system\n" up to
// "<|im_end|>" and reject candidates that still contain Jinja syntax.
std::string extract_chatml_default_system(const std::string& src) {
    size_t from = 0;
    while (true) {
        const size_t p = src.find("<|im_start|>system", from);
        if (p == std::string::npos) return "";
        from = p + 1;

        size_t start = p + std::string("<|im_start|>system").size();
        // Jinja sources normally escape the newline ("\\n" -> 2 chars); accept
        // a raw newline as well for hand-written templates.
        if (src.compare(start, 2, "\\n") == 0) start += 2;
        else if (start < src.size() && src[start] == '\n') start += 1;

        const size_t e = src.find("<|im_end|>", start);
        if (e == std::string::npos) continue;

        const std::string candidate = src.substr(start, e - start);
        if (candidate.empty() ||
            candidate.find_first_of("'\"{}") != std::string::npos)
            continue; // template code, not a literal default prompt
        return candidate;
    }
}

// Llama-3.1-style date preamble, lifted from the template's literals:
//   {{- "Cutting Knowledge Date: December 2023\n" }}
//   {{- "Today Date: " + date_string + "\n\n" }}     (default "26 Jul 2024")
std::string extract_llama3_system_preamble(const std::string& src) {
    const std::string cutoff =
        capture_between(src, "Cutting Knowledge Date: ", "\\n");
    if (cutoff.empty()) return ""; // plain Llama-3.0 template: no preamble

    std::string date = capture_between(src, "date_string = \"", "\"");
    if (date.empty()) date = "26 Jul 2024"; // canonical Llama-3.1 default
    return "Cutting Knowledge Date: " + cutoff + "\nToday Date: " + date + "\n\n";
}

} // namespace

std::string IChatTemplate::render_conversation(const std::vector<ChatMessage>& messages,
                                               bool add_generation_prompt) const {
    size_t first = 0;
    std::string system_prompt = default_system_prompt();
    if (!messages.empty() && messages.front().role == "system") {
        system_prompt = messages.front().content;
        first = 1;
    }

    std::string out = render_prelude(system_prompt);
    for (size_t i = first; i < messages.size(); ++i)
        out += render_message(messages[i]);
    if (add_generation_prompt)
        out += render_generation_prompt();
    return out;
}

std::unique_ptr<IChatTemplate> ChatTemplateFactory::from_jinja_source(
    const std::string& jinja_source, const std::string& bos_token) {
    if (jinja_source.empty()) return nullptr;

    if (jinja_source.find("<|im_start|>") != std::string::npos)
        return std::make_unique<ChatMLTemplate>(
            extract_chatml_default_system(jinja_source));

    if (jinja_source.find("<|start_header_id|>") != std::string::npos)
        return std::make_unique<Llama3Template>(
            bos_token, extract_llama3_system_preamble(jinja_source));

    // GLM-4: "[gMASK]<sop>" opens the template source itself. Probe for it
    // rather than for "<|assistant|>", which ChatML-adjacent families also use.
    if (jinja_source.find("[gMASK]") != std::string::npos &&
        jinja_source.find("<sop>") != std::string::npos)
        return std::make_unique<GlmChatTemplate>();

    throw std::runtime_error(
        "ChatTemplateFactory: tokenizer_config.json carries a chat_template of an "
        "unrecognized family (none of ChatML \"<|im_start|>\", Llama-3 "
        "\"<|start_header_id|>\" or GLM-4 \"[gMASK]<sop>\" markers found). Add a "
        "native implementation in src/core/tokenizer/chat_template.cpp.");
}

std::unique_ptr<IChatTemplate> ChatTemplateFactory::from_model_type(const std::string& model_type) {
    if (model_type == "glm" || model_type == "glm4" || model_type == "chatglm")
        return std::make_unique<GlmChatTemplate>();
    return nullptr;
}

} // namespace blackwell
