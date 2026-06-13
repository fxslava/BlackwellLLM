#include "blackwell_llm_adapter.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace playground {
namespace {

std::string trim(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

// Map an AgentOrchestrator role header to its canonical chat role. Observations
// (tool results) go back to the model as "user" content, matching how chat loops
// surface environment replies. The same role names work verbatim in both ChatML
// and Llama-3 headers, so the parsed blocks are template-independent.
const char* canonical_role(const std::string& header) {
    if (header == "SYSTEM")      return "system";
    if (header == "USER")        return "user";
    if (header == "ASSISTANT")   return "assistant";
    if (header == "OBSERVATION") return "user";
    return nullptr;  // not a recognised header line
}

struct Block {
    std::string role;     // canonical chat role: system / user / assistant
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
            role = canonical_role(stripped.substr(1, stripped.size() - 2));
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

// The orchestrator always appends a trailing empty [ASSISTANT] cue; that is the
// generation prompt, not a real turn -- both renderers drop it and emit the open
// assistant header once at the end instead.
bool is_trailing_cue(const std::vector<Block>& blocks, size_t i) {
    return (i + 1 == blocks.size()) && blocks[i].role == "assistant" && blocks[i].content.empty();
}

// Qwen / ChatML: <|im_start|>role\n...content...<|im_end|>\n , then an open
// <|im_start|>assistant\n cue.
std::string render_chatml(const std::vector<Block>& blocks) {
    std::string s;
    for (size_t i = 0; i < blocks.size(); ++i) {
        if (is_trailing_cue(blocks, i)) continue;
        s += "<|im_start|>";
        s += blocks[i].role;
        s += '\n';
        s += blocks[i].content;
        s += "<|im_end|>\n";
    }
    s += "<|im_start|>assistant\n";
    return s;
}

// Llama-3: a single <|begin_of_text|>, then per turn
// <|start_header_id|>role<|end_header_id|>\n\n...content...<|eot_id|> , then an
// open assistant header cue.
std::string render_llama3(const std::vector<Block>& blocks) {
    std::string s = "<|begin_of_text|>";
    for (size_t i = 0; i < blocks.size(); ++i) {
        if (is_trailing_cue(blocks, i)) continue;
        s += "<|start_header_id|>";
        s += blocks[i].role;
        s += "<|end_header_id|>\n\n";
        s += blocks[i].content;
        s += "<|eot_id|>";
    }
    s += "<|start_header_id|>assistant<|end_header_id|>\n\n";
    return s;
}

}  // namespace

std::string BlackwellLLMAdapter::apply_chat_template(const std::string& role_transcript,
                                                     ChatTemplate tmpl) {
    const std::vector<Block> blocks = parse_blocks(role_transcript);
    return tmpl == ChatTemplate::Llama3 ? render_llama3(blocks) : render_chatml(blocks);
}

void BlackwellLLMAdapter::load_chat_config() {
    // Defaults hold even if the checkpoint ships no tokenizer_config.json (or it
    // lacks a chat_template): ChatML + a foolproof fallback list of end markers
    // covering both families. Cross-family markers are safe to keep -- a model
    // only emits its own end token in normal output; the others appear only as
    // hallucinations, which we *want* to stop on.
    template_kind_ = ChatTemplate::ChatML;
    std::vector<std::string> stops = {"<|im_end|>", "<|eot_id|>", "<|eom_id|>", "<|end_of_text|>"};

    std::string chat_template;
    try {
        std::ifstream f(model_dir_ + "/tokenizer_config.json");
        if (f) {
            nlohmann::json j;
            f >> j;

            // chat_template: usually a string; some newer exports use an array of
            // {name, template} objects -- concatenate them so the heuristic still
            // sees the marker tokens.
            if (j.contains("chat_template")) {
                const auto& ct = j["chat_template"];
                if (ct.is_string()) {
                    chat_template = ct.get<std::string>();
                } else if (ct.is_array()) {
                    for (const auto& e : ct)
                        if (e.is_object() && e.value("template", std::string{}).size())
                            chat_template += e["template"].get<std::string>();
                }
            }

            // eos_token may be a bare string or an AddedToken object {"content": ..}.
            // This is the only stop we pull from JSON: broader lists (e.g. Qwen's
            // additional_special_tokens) include <tool_call>, which the agent must
            // be allowed to emit -- treating it as a stop would break ReAct.
            auto token_text = [](const nlohmann::json& v) -> std::string {
                if (v.is_string()) return v.get<std::string>();
                if (v.is_object()) return v.value("content", std::string{});
                return "";
            };
            if (j.contains("eos_token")) {
                std::string t = token_text(j["eos_token"]);
                if (!t.empty()) stops.push_back(std::move(t));
            }
        }
    } catch (...) {
        // Malformed JSON -> keep defaults. A bad sidecar must not break loading.
    }

    // Heuristic router: pick the formatting routine by the marker tokens the
    // template references. Llama-3's header tokens are unambiguous, so they win.
    if (chat_template.find("<|start_header_id|>") != std::string::npos)
        template_kind_ = ChatTemplate::Llama3;
    else if (chat_template.find("<|im_start|>") != std::string::npos)
        template_kind_ = ChatTemplate::ChatML;
    // else: keep the ChatML default.

    // De-dup + drop empties; cache the longest length for the bounded tail scan.
    stop_strings_.clear();
    max_stop_len_ = 0;
    for (auto& s : stops) {
        if (s.empty()) continue;
        if (std::find(stop_strings_.begin(), stop_strings_.end(), s) != stop_strings_.end()) continue;
        max_stop_len_ = std::max(max_stop_len_, s.size());
        stop_strings_.push_back(std::move(s));
    }
}

BlackwellLLMAdapter::BlackwellLLMAdapter(const std::string& model_dir, size_t max_seq_len)
    : model_dir_(model_dir), max_seq_len_(max_seq_len) {
    // Tokenizer + chat-template config come from the checkpoint's JSON sidecars.
    tokenizer_ = blackwell::TokenizerFactory::create(model_dir);
    load_chat_config();  // sets template_kind_, stop_strings_, max_stop_len_
    // The index file fully describes the weight shards; the engine streams them.
    const std::string index_path = model_dir + "/model.safetensors.index.json";
    engine_ = std::make_unique<BlackwellEngine>(index_path, max_seq_len_);
}

BlackwellLLMAdapter::~BlackwellLLMAdapter() = default;

std::string BlackwellLLMAdapter::generate(const std::string& transcript) {
    // Route through the streaming-capable overload so ReAct turns (which only
    // reach us via this entry point) feed any installed stream callback.
    return generate(transcript, params_, nullptr, stream_cb_);
}

std::string BlackwellLLMAdapter::generate(const std::string& transcript,
                                          const Params& params, std::string* prompt_out,
                                          StreamCallback stream_cb) {
    const std::string prompt_text = apply_chat_template(transcript, template_kind_);
    if (prompt_out) *prompt_out = prompt_text;

    // The tokenizer recognises the literal special-token strings in the prompt and
    // emits their ids; we already applied the template, so no extra specials.
    const std::vector<int> prompt = tokenizer_->encode(prompt_text, /*add_special_tokens=*/false);
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

    // String-based stop detection (see below) is the safety net for the failure
    // mode where a model under the wrong prompt format hallucinates an end marker
    // as ordinary text -- those map to normal token ids, so is_stop() never fires.
    // We scan only a bounded tail window of `out`, so the cost per token is O(Lmax
    // * #stops) -- constant, independent of how long the output grows.
    const size_t Lmax = max_stop_len_;

    std::string out;        // full raw decode (stop string, if any, stripped at the end)
    size_t emitted = 0;     // bytes of `out` already handed to stream_cb
    int generated = 0;
    bool aborted = false;   // cooperative stop via the callback

    while (static_cast<size_t>(pos) < cap && generated < params.max_new_tokens) {
        if (tokenizer_->is_stop(next)) {
            // Real eos token: feed it so the cache reflects a closed turn, then halt.
            engine_->forward(next, pos, 0.0f, 1.0f);
            ++pos;
            break;
        }

        const std::string piece = tokenizer_->decode(next);
        const size_t prev_len = out.size();
        out += piece;

        // (1) Did a stop string just *complete* in the tail? A stop that finished
        //     this step starts no earlier than prev_len - Lmax, so we only search
        //     from there. Cut at the earliest hit and drop it (plus any trailing).
        if (Lmax) {
            const size_t from = prev_len > Lmax ? prev_len - Lmax : 0;
            size_t cut = std::string::npos;
            for (const auto& s : stop_strings_) {
                const size_t hit = out.find(s, from);
                if (hit < cut) cut = hit;  // npos compares high, so this picks the min
            }
            if (cut != std::string::npos) {
                out.resize(cut);  // strip the stop string -- it is never emitted
                break;
            }
        }

        // (2) Stream the confirmed-safe prefix, holding back any tail that is a
        //     prefix of a stop string (it might still complete into one next step,
        //     and we must never emit even a fragment of a stop string to the UI).
        if (stream_cb) {
            size_t hold = 0;
            const size_t maxL = std::min(Lmax, out.size());
            for (size_t L = maxL; L >= 1; --L) {
                for (const auto& s : stop_strings_) {
                    if (s.size() >= L && out.compare(out.size() - L, L, s, 0, L) == 0) { hold = L; break; }
                }
                if (hold) break;
            }
            const size_t safe_len = out.size() - hold;
            if (safe_len > emitted) {
                const bool keep_going = stream_cb(out.substr(emitted, safe_len - emitted));
                emitted = safe_len;
                if (!keep_going) { aborted = true; break; }
            }
        }

        next = engine_->forward(next, pos, params.temperature, params.top_p);
        ++pos;
        ++generated;
    }

    // Flush the held-back-but-confirmed tail. On a string-stop we already resized
    // `out` past it; on natural/token/length termination the held suffix is just
    // ordinary text that happened to look like a stop prefix. Skip on cooperative
    // abort -- the caller asked us to stop emitting.
    if (stream_cb && !aborted && emitted < out.size()) {
        stream_cb(out.substr(emitted));
    }
    return out;
}

}  // namespace playground
