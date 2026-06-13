#include "blackwell_llm_adapter.h"

#include <cctype>
#include <string>
#include <vector>

namespace playground {
namespace {

std::string trim(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

// Map an AgentOrchestrator role header to its ChatML role. Observations (tool
// results) go back to the model as "user" content, matching how ChatML chat
// loops surface environment replies.
const char* chatml_role(const std::string& header) {
    if (header == "SYSTEM")      return "system";
    if (header == "USER")        return "user";
    if (header == "ASSISTANT")   return "assistant";
    if (header == "OBSERVATION") return "user";
    return nullptr;  // not a recognised header line
}

struct Block {
    std::string role;     // canonical ChatML role
    std::string content;  // raw body, trailing blank lines trimmed
};

// Split a role-tagged transcript into blocks. A header is a line that is exactly
// one of "[SYSTEM]" / "[USER]" / "[ASSISTANT]" / "[OBSERVATION]"; everything up
// to the next header is that block's content.
std::vector<Block> parse_blocks(const std::string& t) {
    std::vector<Block> blocks;
    size_t i = 0;
    const size_t n = t.size();
    Block* cur = nullptr;
    while (i < n) {
        size_t eol = t.find('\n', i);
        if (eol == std::string::npos) eol = n;
        std::string line = t.substr(i, eol - i);
        std::string stripped = trim(line);

        const char* role = nullptr;
        if (stripped.size() >= 2 && stripped.front() == '[' && stripped.back() == ']') {
            role = chatml_role(stripped.substr(1, stripped.size() - 2));
        }

        if (role) {
            blocks.push_back(Block{role, ""});
            cur = &blocks.back();
        } else if (cur) {
            if (!cur->content.empty()) cur->content += '\n';
            cur->content += line;
        }
        i = eol + 1;
    }
    for (auto& b : blocks) {
        // Strip only trailing whitespace so intentional internal blank lines and
        // leading indentation in tool output are preserved.
        size_t e = b.content.size();
        while (e > 0 && std::isspace(static_cast<unsigned char>(b.content[e - 1]))) --e;
        b.content.resize(e);
    }
    return blocks;
}

}  // namespace

std::string BlackwellLLMAdapter::to_chatml(const std::string& role_transcript) {
    const std::vector<Block> blocks = parse_blocks(role_transcript);
    std::string cm;
    for (size_t i = 0; i < blocks.size(); ++i) {
        const Block& b = blocks[i];
        // The orchestrator always appends a trailing empty [ASSISTANT] cue; that
        // is the generation prompt, not a real turn -- drop it here and emit the
        // open assistant header once at the end instead.
        const bool trailing_cue =
            (i + 1 == blocks.size()) && b.role == std::string("assistant") && b.content.empty();
        if (trailing_cue) continue;
        cm += "<|im_start|>";
        cm += b.role;
        cm += '\n';
        cm += b.content;
        cm += "<|im_end|>\n";
    }
    cm += "<|im_start|>assistant\n";  // cue: model starts speaking as assistant
    return cm;
}

BlackwellLLMAdapter::BlackwellLLMAdapter(const std::string& model_dir, size_t max_seq_len)
    : model_dir_(model_dir), max_seq_len_(max_seq_len) {
    // Tokenizer + chat-template config come from the checkpoint's JSON sidecars.
    tokenizer_ = blackwell::TokenizerFactory::create(model_dir);
    // The index file fully describes the weight shards; the engine streams them.
    const std::string index_path = model_dir + "/model.safetensors.index.json";
    engine_ = std::make_unique<BlackwellEngine>(index_path, max_seq_len_);
}

BlackwellLLMAdapter::~BlackwellLLMAdapter() = default;

std::string BlackwellLLMAdapter::generate(const std::string& transcript) {
    return generate(transcript, params_, nullptr);
}

std::string BlackwellLLMAdapter::generate(const std::string& transcript,
                                          const Params& params, std::string* chatml_out) {
    const std::string chatml = to_chatml(transcript);
    if (chatml_out) *chatml_out = chatml;

    // The tokenizer recognises the literal special-token strings in the ChatML
    // and emits their ids; we already applied the template, so no extra specials.
    const std::vector<int> prompt = tokenizer_->encode(chatml, /*add_special_tokens=*/false);
    if (prompt.empty()) return "";

    const size_t cap = max_seq_len_;
    int pos = 0;
    int next = -1;

    // Prefill the conversation deterministically (temperature 0): only the final
    // forward of the prefill yields the first generated token.
    for (int id : prompt) {
        if (static_cast<size_t>(pos) >= cap) return "";  // prompt alone overflows
        next = engine_->forward(id, pos, 0.0f, 1.0f);
        ++pos;
    }

    std::string out;
    int generated = 0;
    while (static_cast<size_t>(pos) < cap && generated < params.max_new_tokens) {
        if (tokenizer_->is_stop(next)) {
            // Feed the stop token so the cache reflects a closed turn, then halt.
            engine_->forward(next, pos, 0.0f, 1.0f);
            ++pos;
            break;
        }
        out += tokenizer_->decode(next);
        next = engine_->forward(next, pos, params.temperature, params.top_p);
        ++pos;
        ++generated;
    }
    return out;
}

}  // namespace playground
