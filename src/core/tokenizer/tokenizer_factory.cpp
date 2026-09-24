// TokenizerFactory: builds a tokenizer purely from a checkpoint directory in
// HuggingFace layout. Every id (BOS/EOS/stop set, post-processor inserts,
// tiktoken prefix) is resolved from the JSON sidecar files at load time --
// nothing in the engine hardcodes vocabulary knowledge.
//
// Two families, chosen by which vocabulary file the checkpoint ships:
//   tokenizer.json  -> ByteLevelBpeTokenizer (Llama-3, Qwen2): byte-alphabet
//                      symbols, explicit merges list.
//   tokenizer.model -> TiktokenTokenizer (GLM-4): raw-byte symbols, merge order
//                      implied by vocabulary rank. Only probed when there is no
//                      tokenizer.json, so no existing checkpoint changes path.
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "blackwell/tokenizer.h"
#include "tokenizer/bpe_tokenizer.h"
#include "tokenizer/tiktoken_tokenizer.h"

namespace blackwell {

namespace {

using nlohmann::json;

std::optional<json> load_json_if_present(const std::filesystem::path& path) {
    std::ifstream f(path);
    if (!f.is_open()) return std::nullopt;
    json j;
    try {
        f >> j;
    } catch (const json::parse_error& e) {
        throw std::runtime_error("TokenizerFactory: failed to parse " + path.string() +
                                 ": " + e.what());
    }
    return j;
}

// tokenizer_config.json stores token names either as a plain string or as an
// AddedToken object {"content": "...", ...}; "None" -> absent.
std::string token_name(const json& cfg, const char* key) {
    const auto it = cfg.find(key);
    if (it == cfg.end() || it->is_null()) return "";
    if (it->is_string()) return it->get<std::string>();
    if (it->is_object() && it->contains("content")) return (*it)["content"].get<std::string>();
    return "";
}

// The digit alternative of the pre-tokenizer regex is the only part that
// varies inside the tiktoken family: \p{N}{1,3} (Llama-3) vs bare \p{N}
// (Qwen2). Parse the bound instead of assuming a model.
int parse_max_digit_run(const json& tokenizer_json) {
    int run = 3; // tiktoken-family default when no Split pattern is found
    const auto pt = tokenizer_json.find("pre_tokenizer");
    if (pt == tokenizer_json.end() || pt->is_null()) return run;

    std::vector<json> stack{*pt};
    while (!stack.empty()) {
        const json node = stack.back();
        stack.pop_back();
        if (!node.is_object()) continue;
        if (node.contains("pretokenizers"))
            for (const auto& sub : node["pretokenizers"]) stack.push_back(sub);
        if (node.value("type", "") != "Split" || !node.contains("pattern")) continue;

        const std::string regex = node["pattern"].value("Regex", "");
        // Look for the standalone digit *alternative* "|\p{N}..." -- a bare
        // find would hit the \p{N} inside the [^\r\n\p{L}\p{N}] class first.
        const size_t p = regex.find("|\\p{N}");
        if (p == std::string::npos) continue;
        const size_t brace = p + 6;
        if (brace < regex.size() && regex[brace] == '{') {
            const size_t comma = regex.find(',', brace);
            const size_t close = regex.find('}', brace);
            if (comma != std::string::npos && close != std::string::npos && comma < close)
                run = std::stoi(regex.substr(comma + 1, close - comma - 1));
        } else {
            run = 1; // bare \p{N}
        }
        return run;
    }
    return run;
}

// Resolves the ids a TemplateProcessing post-processor inserts around a single
// encoded sequence (e.g. Llama-3's BOS prefix). Handles the processor being
// wrapped in a Sequence node.
void parse_postprocessor(const json& tokenizer_json, const BpeModelData& data,
                         std::vector<int>& prefix, std::vector<int>& suffix) {
    const auto pp = tokenizer_json.find("post_processor");
    if (pp == tokenizer_json.end() || pp->is_null()) return;

    std::vector<json> stack{*pp};
    while (!stack.empty()) {
        const json node = stack.back();
        stack.pop_back();
        if (!node.is_object()) continue;
        if (node.contains("processors"))
            for (const auto& sub : node["processors"]) stack.push_back(sub);
        if (node.value("type", "") != "TemplateProcessing" || !node.contains("single"))
            continue;

        bool seen_sequence = false;
        for (const auto& entry : node["single"]) {
            if (entry.contains("Sequence")) {
                seen_sequence = true;
                continue;
            }
            if (!entry.contains("SpecialToken")) continue;
            const std::string name = entry["SpecialToken"].value("id", "");
            const auto it = data.added_tokens.find(name);
            if (it == data.added_tokens.end())
                throw std::runtime_error(
                    "TokenizerFactory: post_processor references unknown special token \"" +
                    name + "\"");
            (seen_sequence ? suffix : prefix).push_back(it->second);
        }
        return;
    }
}

// generation_config.json / config.json carry eos_token_id as int or int array.
std::vector<int> parse_stop_ids(const json& cfg) {
    const auto it = cfg.find("eos_token_id");
    if (it == cfg.end() || it->is_null()) return {};
    if (it->is_number_integer()) return {it->get<int>()};
    if (it->is_array()) {
        std::vector<int> ids;
        for (const auto& v : *it)
            if (v.is_number_integer()) ids.push_back(v.get<int>());
        return ids;
    }
    return {};
}

// ---- tiktoken branch ("tokenizer.model" rank files: GLM-4 / ChatGLM) --------

// Special tokens live in tokenizer_config.json's added_tokens_decoder, keyed by
// id-as-string: {"151331": {"content": "[gMASK]", "special": true, ...}}.
void parse_added_tokens_decoder(const json& cfg, TiktokenModelData& data) {
    const auto it = cfg.find("added_tokens_decoder");
    if (it == cfg.end() || !it->is_object()) return;

    for (const auto& [id_str, entry] : it->items()) {
        if (!entry.is_object() || !entry.contains("content")) continue;
        if (!entry.value("special", false)) continue; // non-special added tokens go through BPE

        int id = -1;
        try {
            id = std::stoi(id_str);
        } catch (const std::exception&) {
            throw std::runtime_error("TokenizerFactory: added_tokens_decoder key \"" + id_str +
                                     "\" is not an integer token id");
        }
        data.specials.emplace(entry["content"].get<std::string>(), id);
    }
}

// The ids encode(..., add_special_tokens=true) prepends. There is no JSON field
// for these, but the chat template's *leading literal* is exactly them (GLM-4's
// source opens "[gMASK]<sop>{% for ... %}", which is ChatGLM4Tokenizer's
// get_prefix_tokens()). Deriving them from the template keeps the rule that no
// vocabulary knowledge is hardcoded here; an absent template means no prefix.
std::vector<int> derive_prefix_ids(const std::string& jinja_source,
                                   const TiktokenModelData& data) {
    size_t longest = 0;
    for (const auto& kv : data.specials) longest = std::max(longest, kv.first.size());

    std::vector<int> prefix;
    size_t pos = 0;
    while (pos < jinja_source.size()) {
        bool matched = false;
        // Longest-first so "[gMASK]" is preferred over any shorter literal
        // sharing its prefix.
        const size_t probe = std::min(longest, jinja_source.size() - pos);
        for (size_t len = probe; len >= 1 && !matched; --len) {
            const auto it = data.specials.find(jinja_source.substr(pos, len));
            if (it == data.specials.end()) continue;
            prefix.push_back(it->second);
            pos += len;
            matched = true;
        }
        if (!matched) break; // first non-special character ends the prelude
    }
    return prefix;
}

std::unique_ptr<ITokenizer> create_tiktoken(const std::filesystem::path& dir,
                                            const std::filesystem::path& rank_file) {
    TiktokenModelData data = TiktokenTokenizer::load_rank_file(rank_file.string());

    // GLM-4's pre-tokenizer regex lives in tokenization_chatglm.py, not in any
    // JSON sidecar, so there is nothing to parse: \p{N}{1,3} is both GLM-4's
    // value and the tiktoken-family default that TiktokenModelData starts with.

    std::string chat_template_src;
    std::string model_type;
    if (const auto cfg = load_json_if_present(dir / "tokenizer_config.json")) {
        parse_added_tokens_decoder(*cfg, data);

        auto resolve = [&](const char* key) {
            const std::string name = token_name(*cfg, key);
            if (name.empty()) return -1;
            if (const auto it = data.specials.find(name); it != data.specials.end())
                return it->second;
            const auto vit = data.encoder.find(name);
            return vit != data.encoder.end() ? vit->second : -1;
        };
        data.special_tokens.bos = resolve("bos_token");
        data.special_tokens.eos = resolve("eos_token");
        data.special_tokens.unk = resolve("unk_token");
        data.special_tokens.pad = resolve("pad_token");

        if (cfg->contains("chat_template") && (*cfg)["chat_template"].is_string())
            chat_template_src = (*cfg)["chat_template"].get<std::string>();
    }

    data.prefix_ids = derive_prefix_ids(chat_template_src, data);

    // Stop set: generation_config wins (GLM-4 lists <|endoftext|>, <|user|> and
    // <|observation|> -- a turn ends when the model hands the floor back, not
    // only at EOS), then config.json, then the tokenizer's own EOS.
    if (const auto gen = load_json_if_present(dir / "generation_config.json"))
        data.special_tokens.stop_ids = parse_stop_ids(*gen);
    if (const auto cfg = load_json_if_present(dir / "config.json")) {
        if (data.special_tokens.stop_ids.empty())
            data.special_tokens.stop_ids = parse_stop_ids(*cfg);
        model_type = cfg->value("model_type", "");
    }
    if (data.special_tokens.stop_ids.empty() && data.special_tokens.eos >= 0)
        data.special_tokens.stop_ids = {data.special_tokens.eos};

    auto tmpl = ChatTemplateFactory::from_jinja_source(chat_template_src, /*bos_token=*/"");
    if (!tmpl) tmpl = ChatTemplateFactory::from_model_type(model_type);

    std::cout << "[Tokenizer] tiktoken: " << data.decoder.size() << " ranks + "
              << data.specials.size() << " special, digit-run<=" << data.max_digit_run
              << ", prefix=" << data.prefix_ids.size()
              << ", template=" << (tmpl ? tmpl->name() : "none") << "\n";

    return std::make_unique<TiktokenTokenizer>(std::move(data), std::move(tmpl));
}

} // namespace

std::unique_ptr<ITokenizer> TokenizerFactory::create(const std::string& model_dir) {
    namespace fs = std::filesystem;
    const fs::path dir(model_dir);

    // Probe order: tokenizer.json (HuggingFace byte-level BPE) first, so every
    // checkpoint that has one keeps its existing, validated path. Only when it
    // is absent do we consider a tiktoken rank file.
    const auto tokenizer_json = load_json_if_present(dir / "tokenizer.json");
    if (!tokenizer_json) {
        const fs::path rank_file = dir / "tokenizer.model";
        std::error_code ec;
        if (fs::exists(rank_file, ec)) {
            if (!TiktokenTokenizer::looks_like_rank_file(rank_file.string()))
                throw std::runtime_error(
                    "TokenizerFactory: " + rank_file.string() + " exists but is not a tiktoken "
                    "rank file (\"<base64> <rank>\" per line). SentencePiece .model files are "
                    "not supported; convert the checkpoint or supply tokenizer.json");
            return create_tiktoken(dir, rank_file);
        }
        throw std::runtime_error("TokenizerFactory: neither " +
                                 (dir / "tokenizer.json").string() + " nor " +
                                 rank_file.string() + " found; cannot construct a tokenizer");
    }

    const json& tj = *tokenizer_json;
    if (!tj.contains("model") || !tj["model"].contains("vocab"))
        throw std::runtime_error("TokenizerFactory: tokenizer.json has no model.vocab");
    const json& model = tj["model"];
    if (model.value("type", "BPE") != "BPE")
        throw std::runtime_error("TokenizerFactory: unsupported tokenizer model type \"" +
                                 model.value("type", "?") + "\" (only byte-level BPE)");

    BpeModelData data;

    for (const auto& [tok, id] : model["vocab"].items()) {
        data.vocab.emplace(tok, id.get<int>());
        data.id_to_token.emplace(id.get<int>(), tok);
    }

    // merges: ["a b", ...] in older dumps, [["a","b"], ...] in newer ones.
    int rank = 0;
    for (const auto& m : model.value("merges", json::array())) {
        std::string key;
        if (m.is_string()) {
            key = m.get<std::string>();
        } else if (m.is_array() && m.size() == 2) {
            key = m[0].get<std::string>() + " " + m[1].get<std::string>();
        } else {
            throw std::runtime_error("TokenizerFactory: malformed merges entry");
        }
        data.merge_ranks.emplace(std::move(key), rank++);
    }

    data.ignore_merges = model.value("ignore_merges", false);
    data.max_digit_run = parse_max_digit_run(tj);

    for (const auto& at : tj.value("added_tokens", json::array())) {
        const int id = at.at("id").get<int>();
        const std::string content = at.at("content").get<std::string>();
        data.added_tokens.emplace(content, id);
        data.id_to_token[id] = content;
        if (at.value("special", false)) data.special_ids.insert(id);
    }

    parse_postprocessor(tj, data, data.postproc_prefix_ids, data.postproc_suffix_ids);

    // ---- tokenizer_config.json: special-token names + chat template --------
    std::string chat_template_src;
    std::string bos_name, eos_name;
    if (const auto cfg = load_json_if_present(dir / "tokenizer_config.json")) {
        bos_name = token_name(*cfg, "bos_token");
        eos_name = token_name(*cfg, "eos_token");

        auto resolve = [&](const char* key) {
            const std::string name = token_name(*cfg, key);
            if (name.empty()) return -1;
            const auto it = data.added_tokens.find(name);
            if (it != data.added_tokens.end()) return it->second;
            const auto vit = data.vocab.find(name);
            return vit != data.vocab.end() ? vit->second : -1;
        };
        data.special_tokens.bos = resolve("bos_token");
        data.special_tokens.eos = resolve("eos_token");
        data.special_tokens.unk = resolve("unk_token");
        data.special_tokens.pad = resolve("pad_token");

        if (cfg->contains("chat_template") && (*cfg)["chat_template"].is_string())
            chat_template_src = (*cfg)["chat_template"].get<std::string>();
    }

    // ---- generation stop set: generation_config preferred, then config.json,
    // then the tokenizer's own EOS. ------------------------------------------
    if (const auto gen = load_json_if_present(dir / "generation_config.json"))
        data.special_tokens.stop_ids = parse_stop_ids(*gen);
    if (data.special_tokens.stop_ids.empty())
        if (const auto cfg = load_json_if_present(dir / "config.json"))
            data.special_tokens.stop_ids = parse_stop_ids(*cfg);
    if (data.special_tokens.stop_ids.empty() && data.special_tokens.eos >= 0)
        data.special_tokens.stop_ids = {data.special_tokens.eos};

    // BOS/EOS fallback for checkpoints whose names did not resolve: config.json ids.
    if (const auto cfg = load_json_if_present(dir / "config.json")) {
        if (data.special_tokens.bos < 0 && cfg->contains("bos_token_id") &&
            (*cfg)["bos_token_id"].is_number_integer())
            data.special_tokens.bos = (*cfg)["bos_token_id"].get<int>();
        if (data.special_tokens.eos < 0 && !data.special_tokens.stop_ids.empty())
            data.special_tokens.eos = data.special_tokens.stop_ids.front();
    }

    auto tmpl = ChatTemplateFactory::from_jinja_source(chat_template_src, bos_name);

    std::cout << "[Tokenizer] " << data.vocab.size() << " BPE tokens + "
              << data.added_tokens.size() << " added, " << data.merge_ranks.size()
              << " merges, digit-run<=" << data.max_digit_run
              << (data.ignore_merges ? ", ignore_merges" : "")
              << ", template=" << (tmpl ? tmpl->name() : "none") << "\n";

    return std::make_unique<ByteLevelBpeTokenizer>(std::move(data), std::move(tmpl));
}

} // namespace blackwell
