#include "blackwell_llm_adapter.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

// Full definition of EnginePrefillCoordinator (engine.h only forward-declares
// it) for prefill_driver().set_tokenizer(). Pulls the paging substrate headers,
// which is why this target needs the CUDA include directories.
#include "../../engine_prefill_coordinator.h"

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

BlackwellLLMAdapter::BlackwellLLMAdapter(const std::string& model_dir, size_t max_seq_len,
                                         bool use_paged_attention, size_t num_gpu_layers,
                                         const blackwell::RuntimeOverrides& overrides)
    : model_dir_(model_dir), max_seq_len_(max_seq_len),
      use_paged_attention_(use_paged_attention), num_gpu_layers_(num_gpu_layers) {
    // Tokenizer + chat-template config come from the checkpoint's JSON sidecars.
    tokenizer_ = blackwell::TokenizerFactory::create(model_dir);
    load_chat_config();  // sets template_kind_, stop_strings_, max_stop_len_
    // The index file fully describes the weight shards; the engine streams them.
    const std::string index_path = model_dir + "/model.safetensors.index.json";

    // Merge the caller's low-level overrides (tiered KV sizing etc.) with the
    // adapter's own explicit knobs -- which win, to keep one source of truth.
    blackwell::InferenceConfig request;
    request.max_context_length = max_seq_len_;
    blackwell::RuntimeOverrides merged = overrides;
    merged.kv_mode = use_paged_attention_ ? BlackwellEngine::KVCacheMode::Paged
                                          : BlackwellEngine::KVCacheMode::Continuous;
    // num_gpu_layers_ drives the CPU/GPU weight split inside the engine (SIZE_MAX =
    // everything resident in VRAM / env override). The engine offloads the
    // remainder to pinned host RAM regardless of the quant backend.
    merged.num_gpu_layers = num_gpu_layers_;
    engine_ = std::make_unique<BlackwellEngine>(index_path, request, merged);

    // Bind the tokenizer to the prefill driver (Paged dense models only), so
    // the AOT warm-start / warmup-CLI facet can tokenize without owning a
    // tokenizer of its own. TokenId is int32 == the tokenizer's int.
    if (engine_->has_prefix_cache()) {
        engine_->prefill_driver().set_tokenizer(
            [this](const std::string& text, bool add_special) {
                const std::vector<int> ids = tokenizer_->encode(text, add_special);
                return std::vector<blackwell::EnginePrefillCoordinator::TokenId>(
                    ids.begin(), ids.end());
            });
    }
}

BlackwellLLMAdapter::~BlackwellLLMAdapter() = default;

std::string BlackwellLLMAdapter::generate(const std::string& transcript) {
    // Route through the streaming-capable overload so ReAct turns (which only
    // reach us via this entry point) feed any installed stream callback.
    return generate(transcript, params_, nullptr, stream_cb_);
}

std::string BlackwellLLMAdapter::generate(const std::string& transcript,
                                          const Params& params, std::string* prompt_out,
                                          StreamCallback stream_cb, int seq_id) {
    const std::string prompt_text = apply_chat_template(transcript, template_kind_);
    if (prompt_out) *prompt_out = prompt_text;

    // The tokenizer recognises the literal special-token strings in the prompt and
    // emits their ids; we already applied the template, so no extra specials.
    const std::vector<int> prompt = tokenizer_->encode(prompt_text, /*add_special_tokens=*/false);
    if (prompt.empty()) return "";

    const size_t cap = max_seq_len_;

    // Per-branch KV-reuse history (seq 0 = main). Every forward below targets this
    // sequence, so parallel branches never clobber each other's KV cache.
    std::vector<int>& cached = cached_prompt_by_seq_[seq_id];

    // ---- Append-or-Reset prefix match (incremental KV reuse) -----------------
    // `cached` mirrors the tokens currently resident in this sequence's KV cache
    // (last turn's prompt + everything we then generated). If the whole of it is a
    // prefix of the new prompt, the conversation only grew at the end, so we KEEP
    // that KV and prefill ONLY the new tail -- O(new tokens) instead of O(N). Any
    // divergence (an edited/deleted past turn, or a shorter prompt) means the
    // cached KV is stale, so we RESET: clear it and reprefill from pos 0. The
    // engine treats forward(token, pos) as position-addressed, so reprefilling
    // from 0 truncates+rewrites this sequence; slots past the new length are
    // never attended to.
    // Incremental KV reuse (APPEND only the new tail) is safe only when the engine
    // can rewind to an arbitrary prefix. Hybrid linear-attention (SSM) models cannot:
    // their recurrent state advances with every forward() with no rewind, and the
    // stop-token handling below even forwards an eos that is never committed to
    // `cached`, so the resident SSM state and `cached` drift apart. For those models
    // we DISABLE the optimization -- always zero the recurrent state and reprefill
    // the whole prompt from pos 0 -- which is the only way to keep the SSM state in
    // lockstep with the prompt (otherwise the model collapses into repetition).
    // Dense models keep the fast incremental path; reset_state() is a no-op for them.
    const bool incremental_safe = !engine_->get_capabilities().requires_ssm_subsystem;
    const bool append = incremental_safe &&
                        cached.size() <= prompt.size() &&
                        std::equal(cached.begin(), cached.end(), prompt.begin());
    if (!append) {
        cached.clear();                 // RESET (reprefill the whole prompt from pos 0)
        engine_->reset_state(seq_id);   // zero recurrent SSM state (no-op if dense)
    }

    int pos = static_cast<int>(cached.size());  // APPEND: start past the reused prefix
    int next = -1;

    // Prefill only the tokens not already resident, deterministically (temperature
    // 0): only the final forward of the prefill yields the first generated token.
    for (size_t i = cached.size(); i < prompt.size(); ++i) {
        if (static_cast<size_t>(pos) >= cap) {  // prompt overflows the window
            cached.clear();                     // KV is now partial -> force a reset next call
            return "";
        }
        next = engine_->forward(prompt[i], pos, 0.0f, 1.0f, seq_id);
        ++pos;
    }
    cached = prompt;  // KV now holds exactly `prompt` at positions 0..N-1

    // Regenerating an identical prompt forwarded nothing, leaving `next` unset;
    // re-run the final token (same id, same slot -> idempotent) to recover the
    // first-token logits.
    if (next < 0) {
        pos = static_cast<int>(prompt.size()) - 1;
        next = engine_->forward(prompt.back(), pos, 0.0f, 1.0f, seq_id);
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
            // It is a stop, so it is NOT committed to `cached`; next turn the
            // template re-emits the closing marker over this same slot.
            engine_->forward(next, pos, 0.0f, 1.0f, seq_id);
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

        // Commit the confirmed token to the session history (mirrors the KV write
        // below) so the next turn can append straight onto it, then advance.
        cached.push_back(next);
        next = engine_->forward(next, pos, params.temperature, params.top_p, seq_id);
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

void BlackwellLLMAdapter::fork_sequence(int parent_id, int child_id) {
    // Engine-level CoW branch: the child shares the parent's KV pages until it
    // writes (throws under the continuous cache). Seed the child's reuse history
    // from the parent so continuing the child from the fork point appends (reuse)
    // rather than reprefilling.
    engine_->fork(parent_id, child_id);
    cached_prompt_by_seq_[child_id] = cached_prompt_by_seq_[parent_id];
}

}  // namespace playground
