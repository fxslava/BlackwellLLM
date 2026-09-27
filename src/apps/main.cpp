// Interactive chat CLI -- the first true COM-side consumer of
// blackwell_core.dll: includes ONLY the public boundary header, creates the
// engine/tokenizer through the C factories, and speaks HRESULT. No C++
// exception ever crosses the DLL edge; the try/catch below handles only
// this file's own failures (check() throwing on a FAILED hr).
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <fcntl.h>
#include <io.h>
#include <windows.h>

#include "blackwell/iblackwell_engine.h"

namespace {

// ---- command line -----------------------------------------------------------

struct Options {
    // Defaults preserve the historical behaviour of a bare `blackwell_llm.exe`.
    std::string model_dir      = "F:/AI/llama-3.1-8B-Instruct-AWQ-INT4";
    std::string system_prompt  = "You are a helpful, smart, and concise AI assistant.";
    uint32_t    max_context    = 16384;
    uint32_t    num_gpu_layers = BLACKWELL_ALL_LAYERS_RESIDENT;
    bool        serve          = false;   // --serve: batch protocol instead of chat
    bool        paged_kv       = false;   // --kv-mode paged
    // --prefill loop restores the per-token Forward() sweep. Kept because it is
    // the reference the chunked path is checked against: same prompt, same greedy
    // token, or the fast path is wrong.
    bool        prefill_loop   = false;
};

void print_usage(const char* exe) {
    std::cout <<
        "Usage: " << exe << " [--model <dir>] [options]\n"
        "\n"
        "  -m, --model <dir>      checkpoint directory (weights + config + tokenizer).\n"
        "                         Also accepted as the first positional argument.\n"
        "      --system <text>    system prompt for the conversation prelude\n"
        "      --ctx <n>          max context length in tokens (default 16384)\n"
        "      --gpu-layers <n>   layers kept resident in VRAM; the rest stream from\n"
        "                         pinned host RAM. Default: all resident. A model whose\n"
        "                         weights exceed VRAM needs this -- GLM-4-9B bf16 is\n"
        "                         17.7 GiB, so on a 12 GB card use 17.\n"
        "      --chat             interactive chat (the only mode; accepted for clarity)\n"
        "  -h, --help             this message\n"
        "\n"
        "Any supported checkpoint runs through the same path: the directory fully\n"
        "describes the model, so Llama-3, Qwen2.5/ChatML and GLM-4 differ only by what\n"
        "is in it.\n";
}

[[noreturn]] void bad_usage(const std::string& msg) {
    throw std::runtime_error(msg + "\n       (run with --help for usage)");
}

// Reads the value of a flag written either as "--flag value" or "--flag=value".
// `eq` is the position of '=' within argv[i], or npos when the value is the
// next argv token. Advances `i` past the value it consumed.
std::string take_value(int argc, char** argv, int& i, size_t eq, const char* flag) {
    const std::string_view arg(argv[i]);
    if (eq != std::string_view::npos) {
        const std::string_view v = arg.substr(eq + 1);
        if (v.empty()) bad_usage(std::string(flag) + " was given an empty value");
        return std::string(v);
    }
    if (i + 1 >= argc)
        bad_usage(std::string(flag) + " requires a value (e.g. " + flag + " F:/AI/my-model)");
    return argv[++i];
}

uint32_t parse_u32(const std::string& text, const char* flag) {
    char* end = nullptr;
    const unsigned long long v = std::strtoull(text.c_str(), &end, 10);
    if (end == text.c_str() || *end != '\0' || v == 0 || v > 0xFFFFFFFEu)
        bad_usage(std::string(flag) + " expects a positive integer, got \"" + text + "\"");
    return static_cast<uint32_t>(v);
}

// Returns false when the caller should exit successfully without running (--help).
bool parse_args(int argc, char** argv, Options& opt) {
    bool model_set = false;

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        // Split "--flag=value" at the FIRST '=', so the flag match below sees
        // only the name and a value may itself contain '=' (a path like
        // "--model=F:/ckpt=v2" must not be mistaken for an unknown option).
        const size_t eq = arg.find('=');
        const std::string_view name =
            (arg.size() > 1 && arg[0] == '-' && eq != std::string_view::npos)
                ? arg.substr(0, eq) : arg;

        if (name == "-h" || name == "--help") {
            print_usage(argv[0]);
            return false;
        }
        if (name == "--chat") {
            continue;   // the default mode; accepted so the documented invocation works
        }
        if (name == "--serve") {
            opt.serve = true;
            continue;
        }
        if (name == "--kv-mode") {
            const std::string mode = take_value(argc, argv, i, eq, "--kv-mode");
            if (mode == "paged")           opt.paged_kv = true;
            else if (mode == "continuous") opt.paged_kv = false;
            else bad_usage("--kv-mode expects \"continuous\" or \"paged\", got \"" + mode + "\"");
            continue;
        }
        if (name == "--prefill") {
            const std::string mode = take_value(argc, argv, i, eq, "--prefill");
            if (mode == "loop")      opt.prefill_loop = true;
            else if (mode == "fast") opt.prefill_loop = false;
            else bad_usage("--prefill expects \"fast\" or \"loop\", got \"" + mode + "\"");
            continue;
        }
        if (name == "-m" || name == "--model") {
            opt.model_dir = take_value(argc, argv, i, eq, "--model");
            model_set = true;
            continue;
        }
        if (name == "--system") {
            opt.system_prompt = take_value(argc, argv, i, eq, "--system");
            continue;
        }
        if (name == "--ctx") {
            opt.max_context = parse_u32(take_value(argc, argv, i, eq, "--ctx"), "--ctx");
            continue;
        }
        if (name == "--gpu-layers") {
            opt.num_gpu_layers =
                parse_u32(take_value(argc, argv, i, eq, "--gpu-layers"), "--gpu-layers");
            continue;
        }

        // An unknown dash-led token is a typo, not a path: reporting it beats
        // silently using "--modle" as the checkpoint directory, which is exactly
        // how a missing parser used to fail (it reached the tokenizer factory as
        // "--model/tokenizer.json not found").
        if (!arg.empty() && arg[0] == '-')
            bad_usage("unknown option \"" + std::string(arg) + "\"");

        if (model_set)
            bad_usage("unexpected extra argument \"" + std::string(arg) + "\"");
        opt.model_dir = std::string(arg);   // bare positional path, as before
        model_set = true;
    }
    return true;
}

void check(HRESULT hr, const char* what) {
    if (FAILED(hr)) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "%s failed (hr=0x%08lX)", what,
                      static_cast<unsigned long>(hr));
        throw std::runtime_error(buf);
    }
}

// Single-owner boundary handles: Release() on scope exit.
struct ComRelease {
    template <typename T>
    void operator()(T* p) const {
        if (p) p->Release();
    }
};
template <typename T>
using com_ptr = std::unique_ptr<T, ComRelease>;

// Two-call protocol wrapper: size query, then fill.
template <typename Call>
std::vector<int32_t> query_ids(Call&& call, const char* what) {
    uint32_t count = 0;
    check(call(nullptr, 0u, &count), what);
    std::vector<int32_t> ids(count);
    if (count) check(call(ids.data(), count, &count), what);
    return ids;
}

// Reads one user turn as UTF-8. Returns false at end of input, which is the
// loop's exit condition.
//
// The two branches are not interchangeable. ReadConsoleW is what makes typed
// non-ASCII work (it hands back UTF-16 regardless of the console code page),
// but it FAILS on a redirected handle -- so `echo hi | blackwell_llm.exe` used
// to read nothing, and, since an empty line just continues, spin forever
// printing the prompt. When stdin is a pipe or a file, read it as bytes.
bool read_user_line(std::string& out) {
    out.clear();
    const HANDLE hStdin = GetStdHandle(STD_INPUT_HANDLE);

    DWORD console_mode = 0;
    if (GetConsoleMode(hStdin, &console_mode) == 0) {
        // Redirected: plain byte-oriented stdin, already UTF-8 by convention.
        std::string line;
        if (!std::getline(std::cin, line)) return false;
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        // PowerShell (and several editors) prefix redirected UTF-8 with a BOM.
        // Left in place it becomes a stray U+FEFF token at the head of the very
        // first user turn.
        if (line.compare(0, 3, "\xEF\xBB\xBF") == 0) line.erase(0, 3);
        out = std::move(line);
        std::cout << out << "\n";   // echo it, so a piped transcript reads correctly
        return true;
    }

    wchar_t wbuf[4096];
    DWORD read_chars = 0;
    FlushConsoleInputBuffer(hStdin);
    if (!ReadConsoleW(hStdin, wbuf, 4096, &read_chars, nullptr)) return false;
    if (read_chars == 0) return false;   // Ctrl+Z
    while (read_chars > 0 && (wbuf[read_chars - 1] == L'\n' || wbuf[read_chars - 1] == L'\r'))
        --read_chars;
    if (read_chars == 0) return true;    // bare Enter: an empty turn, not EOF

    const int size = WideCharToMultiByte(CP_UTF8, 0, wbuf, static_cast<int>(read_chars),
                                         nullptr, 0, nullptr, nullptr);
    if (size <= 0) return true;
    out.assign(static_cast<size_t>(size), '\0');
    WideCharToMultiByte(CP_UTF8, 0, wbuf, static_cast<int>(read_chars), out.data(), size,
                        nullptr, nullptr);
    return true;
}

std::string decode_token(IBlackwellTokenizer* tok, int32_t token_id) {
    char buf[512];
    uint32_t len = 0;
    HRESULT hr = tok->DecodeToken(token_id, FALSE, buf, sizeof(buf), &len);
    if (hr == HRESULT_FROM_WIN32(ERROR_INSUFFICIENT_BUFFER)) {
        std::string big(len, '\0');
        hr = tok->DecodeToken(token_id, FALSE, big.data(), len, &len);
        check(hr, "DecodeToken");
        return big;
    }
    check(hr, "DecodeToken");
    return std::string(buf, len);
}

// ---- batch ("serve") mode ---------------------------------------------------
//
// A persistent, non-interactive request/response loop. It exists because an
// evaluation harness cannot drive the interactive path at all: read_user_line()
// is line-oriented, so a LongBench prompt -- a multi-thousand-token document
// full of newlines -- would arrive as hundreds of separate user turns; and the
// chat loop has no generation cap, so one degenerate repetition runs to the
// context limit instead of stopping at the task's max_gen.
//
// Framing is length-prefixed and never delimiter-scanned, because the engine DLL
// writes its own progress lines ("[VRAM Arena] ...") to this same stdout: the
// harness must be free to skip any line it does not recognise. Request:
//
//     GEN <byte_len> <max_new> <max_input_tokens> <temp> <top_p> <chat>\n
//     <byte_len bytes of UTF-8 prompt>
//
// Response -- the generated text is escaped onto ONE line, so an interleaved DLL
// log line can never be mistaken for part of the payload:
//
//     <<<RESULT prompt_tokens=N gen_tokens=N prefill_ms=F decode_ms=F stop=S trunc=B>>>
//     TEXT <escaped>
//     <<<END>>>
//
// A per-request failure reports <<<ERROR ...>>> and the loop continues: one
// sample that faults must not cost the harness the 7.2 GiB reload. "QUIT" or EOF
// ends the loop.

struct ServeRequest {
    size_t   byte_len         = 0;
    uint32_t max_new          = 64;
    uint32_t max_input_tokens = 0;      // 0 => clamp to the context budget only
    float    temperature      = 0.0f;   // 0 => greedy argmax (sample_top_p's fast path)
    float    top_p            = 1.0f;
    bool     chat             = true;   // false => raw completion, no chat template
};

std::string escape_line(const std::string& text) {
    std::string out;
    out.reserve(text.size() + 16);
    for (const char c : text) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

// LongBench's own truncation rule: when a prompt exceeds the budget, keep its
// head and its tail and drop the middle -- the instruction block at the front
// and the question at the back are precisely what must survive. Returns true
// when it cut anything.
bool middle_truncate(std::vector<int32_t>& ids, size_t budget) {
    if (budget == 0 || ids.size() <= budget) return false;
    const size_t head = budget / 2;
    const size_t tail = budget - head;
    std::vector<int32_t> kept;
    kept.reserve(budget);
    kept.insert(kept.end(), ids.begin(), ids.begin() + static_cast<ptrdiff_t>(head));
    kept.insert(kept.end(), ids.end() - static_cast<ptrdiff_t>(tail), ids.end());
    ids.swap(kept);
    return true;
}

std::string decode_ids(IBlackwellTokenizer* tok, const std::vector<int32_t>& ids) {
    std::string text;
    for (const int32_t id : ids) text += decode_token(tok, id);
    return text;
}

// Turns one request's prompt text into the exact token sequence to prefill.
//
// Truncation happens on the DOCUMENT, before the chat template is applied, which
// is what the official script does -- so the template's framing tokens and the
// assistant cue always survive. Because the tokenizer wraps text and not ids, a
// truncated document is decoded back to text before wrapping (the same
// round-trip the official script performs); the caller's `hard_cap` retry then
// catches the case where the template's own overhead pushes the total back over
// the budget.
std::vector<int32_t> build_prompt_ids(IBlackwellTokenizer* tok, const Options& opt,
                                      const std::string& prompt, bool chat,
                                      size_t doc_budget, bool& truncated) {
    std::vector<int32_t> doc = query_ids(
        [&](int32_t* ids, uint32_t cap, uint32_t* n) {
            return tok->Encode(prompt.c_str(), FALSE, ids, cap, n);
        },
        "Encode");

    truncated = middle_truncate(doc, doc_budget);
    const std::string text = truncated ? decode_ids(tok, doc) : prompt;

    if (!chat) {
        return query_ids(
            [&](int32_t* ids, uint32_t cap, uint32_t* n) {
                return tok->Encode(text.c_str(), TRUE, ids, cap, n);
            },
            "Encode");
    }

    std::vector<int32_t> out = query_ids(
        [&](int32_t* ids, uint32_t cap, uint32_t* n) {
            return tok->EncodeChatPrelude(opt.system_prompt.c_str(), ids, cap, n);
        },
        "EncodeChatPrelude");
    const std::vector<int32_t> turn = query_ids(
        [&](int32_t* ids, uint32_t cap, uint32_t* n) {
            return tok->EncodeChatMessage("user", text.c_str(), ids, cap, n);
        },
        "EncodeChatMessage");
    const std::vector<int32_t> cue = query_ids(
        [&](int32_t* ids, uint32_t cap, uint32_t* n) {
            return tok->EncodeGenerationPrompt(ids, cap, n);
        },
        "EncodeGenerationPrompt");
    out.insert(out.end(), turn.begin(), turn.end());
    out.insert(out.end(), cue.begin(), cue.end());
    return out;
}

void run_one_request(IBlackwellEngine* engine, IBlackwellTokenizer* tok, const Options& opt,
                     const ServeRequest& req, const std::string& prompt) {
    using clock = std::chrono::steady_clock;
    const auto elapsed_ms = [](clock::time_point a, clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    // Room for the answer plus a small margin for the template's framing tokens.
    const size_t reserved = static_cast<size_t>(req.max_new) + 8u;
    const size_t hard_cap = opt.max_context > reserved
                                ? static_cast<size_t>(opt.max_context) - reserved : 1u;
    size_t doc_budget = (req.max_input_tokens == 0 || req.max_input_tokens > hard_cap)
                            ? hard_cap : static_cast<size_t>(req.max_input_tokens);

    std::vector<int32_t> ids;
    bool truncated = false;
    for (int attempt = 0; attempt < 3; ++attempt) {
        ids = build_prompt_ids(tok, opt, prompt, req.chat, doc_budget, truncated);
        if (ids.size() <= hard_cap) break;
        const size_t excess = ids.size() - hard_cap;
        doc_budget = doc_budget > excess ? doc_budget - excess : doc_budget / 2;
        truncated = true;
    }
    // Last resort: the template alone overflows the context. Cut the ids -- a
    // mangled prompt still beats writing past the end of the KV cache.
    if (ids.size() > hard_cap) {
        middle_truncate(ids, hard_cap);
        truncated = true;
    }

    // Each request is independent: prefill restarts at position 0 (the Continuous
    // KV cache is only ever read over [0, pos], so the previous request's tail is
    // overwritten, never attended to) and any recurrent state is zeroed.
    check(engine->ResetState(0), "ResetState");

    const auto t_prefill_begin = clock::now();
    int32_t next_token = -1;
    int32_t pos = 0;
    if (opt.prefill_loop) {
        // Reference sweep: one Forward() per prompt token, each paying the lm_head
        // for logits that are then thrown away.
        for (const int32_t id : ids) {
            check(engine->Forward(id, pos, 0.0f, 1.0f, 0, &next_token), "Forward");
            ++pos;
        }
    } else {
        check(engine->PrefillTokens(ids.data(), static_cast<uint32_t>(ids.size()), 0,
                                    0.0f, 1.0f, 0, &next_token),
              "PrefillTokens");
        pos = static_cast<int32_t>(ids.size());
    }
    const auto t_decode_begin = clock::now();

    std::string text;
    uint32_t gen_tokens = 0;
    const char* stop = "limit";          // reached max_new
    while (gen_tokens < req.max_new) {
        BOOL is_stop = FALSE;
        check(tok->IsStop(next_token, &is_stop), "IsStop");
        if (is_stop) { stop = "eos"; break; }
        if (pos >= static_cast<int32_t>(opt.max_context)) { stop = "ctx"; break; }

        text += decode_token(tok, next_token);
        ++gen_tokens;
        check(engine->Forward(next_token, pos, req.temperature, req.top_p, 0, &next_token),
              "Forward");
        ++pos;
    }
    const auto t_end = clock::now();

    std::printf("<<<RESULT prompt_tokens=%zu gen_tokens=%u prefill_ms=%.1f decode_ms=%.1f "
                "stop=%s trunc=%d prefill=%s>>>\n",
                ids.size(), gen_tokens, elapsed_ms(t_prefill_begin, t_decode_begin),
                elapsed_ms(t_decode_begin, t_end), stop, truncated ? 1 : 0,
                opt.prefill_loop ? "loop" : "fast");
    std::printf("TEXT %s\n<<<END>>>\n", escape_line(text).c_str());
    std::fflush(stdout);
}

int run_serve(IBlackwellEngine* engine, IBlackwellTokenizer* tok, const Options& opt) {
    // Byte-exact framing needs binary stdin: in text mode the CRT would eat the
    // \r of any CRLF and every byte_len would come up short.
    _setmode(_fileno(stdin), _O_BINARY);

    std::printf("<<<READY ctx=%u kv=%s prefill=%s>>>\n", opt.max_context,
                opt.paged_kv ? "paged" : "continuous",
                opt.prefill_loop ? "loop" : "fast");
    std::fflush(stdout);

    std::string header;
    while (std::getline(std::cin, header)) {
        if (!header.empty() && header.back() == '\r') header.pop_back();
        if (header.empty()) continue;

        std::istringstream in(header);
        std::string verb;
        in >> verb;
        if (verb == "QUIT") break;
        if (verb != "GEN") {
            std::printf("<<<ERROR what=unknown verb %s>>>\n<<<END>>>\n", verb.c_str());
            std::fflush(stdout);
            continue;
        }

        ServeRequest req;
        int chat_flag = 1;
        in >> req.byte_len >> req.max_new >> req.max_input_tokens >> req.temperature
           >> req.top_p >> chat_flag;
        if (!in) {
            std::printf("<<<ERROR what=malformed GEN header>>>\n<<<END>>>\n");
            std::fflush(stdout);
            continue;
        }
        req.chat = (chat_flag != 0);

        std::string prompt(req.byte_len, '\0');
        if (req.byte_len != 0) {
            std::cin.read(prompt.data(), static_cast<std::streamsize>(req.byte_len));
            if (std::cin.gcount() != static_cast<std::streamsize>(req.byte_len)) break;  // EOF
        }

        try {
            run_one_request(engine, tok, opt, req, prompt);
        } catch (const std::exception& e) {
            // Per-request isolation: report and keep the loaded weights alive.
            std::printf("<<<ERROR what=%s>>>\n<<<END>>>\n", escape_line(e.what()).c_str());
            std::fflush(stdout);
        }
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    Options opt;
    try {
        if (!parse_args(argc, argv, opt)) return 0;
    } catch (const std::exception& e) {
        std::cerr << "[usage] " << e.what() << "\n";
        return 2;
    }

    if (!opt.serve) {
        std::cout << "==================================================\n";
        std::cout << " Blackwell LLM: Interactive Chat Mode\n";
        std::cout << " (Type 'exit' or 'quit' to close)\n";
        std::cout << "==================================================\n\n";
    }

    try {
        // The checkpoint directory fully describes the model: weights, config
        // and tokenizer all come from it, so any supported model (Llama-3,
        // Qwen2.5/ChatML, GLM-4/tiktoken, ...) runs through the same code path.
        const std::string& model_dir = opt.model_dir;
        const std::string index_path = model_dir + "/model.safetensors.index.json";
        const uint32_t max_context = opt.max_context;

        if (!opt.serve) std::cout << "[System] Model: " << model_dir << "\n";

        IBlackwellTokenizer* tok_raw = nullptr;
        check(CreateBlackwellTokenizer(model_dir.c_str(), &tok_raw), "CreateBlackwellTokenizer");
        com_ptr<IBlackwellTokenizer> tokenizer(tok_raw);

        BLACKWELL_ENGINE_DESC desc{};
        desc.index_path         = index_path.c_str();
        desc.max_context_length = max_context;
        desc.num_gpu_layers     = opt.num_gpu_layers;
        desc.kv_mode            = opt.paged_kv ? BLACKWELL_KV_MODE_PAGED
                                               : BLACKWELL_KV_MODE_CONTINUOUS;

        IBlackwellEngine* eng_raw = nullptr;
        check(CreateBlackwellEngine(&desc, &eng_raw), "CreateBlackwellEngine");
        com_ptr<IBlackwellEngine> engine(eng_raw);

        if (opt.serve) return run_serve(engine.get(), tokenizer.get(), opt);

        int32_t current_pos = 0;

        // Conversation prefix (BOS + system block) rendered by the model's own
        // chat template; no token id appears anywhere in this file.
        const std::vector<int32_t> prelude = query_ids(
            [&](int32_t* ids, uint32_t cap, uint32_t* n) {
                return tokenizer->EncodeChatPrelude(opt.system_prompt.c_str(), ids, cap, n);
            },
            "EncodeChatPrelude");

        std::cout << "[System] Initializing context...\n";
        int32_t next_token = -1;
        for (const int32_t token : prelude) {
            check(engine->Forward(token, current_pos, 0.0f, 1.0f, 0, &next_token), "Forward");
            current_pos++;
        }

        while (true) {
            std::cout << "\n\nUser > " << std::flush;
            std::string user_prompt;
            if (!read_user_line(user_prompt)) break;   // end of input

            if (user_prompt == "exit" || user_prompt == "quit") break;
            if (user_prompt.empty()) continue;

            // User turn + assistant cue, framed by the chat template.
            std::vector<int32_t> turn_tokens = query_ids(
                [&](int32_t* ids, uint32_t cap, uint32_t* n) {
                    return tokenizer->EncodeChatMessage("user", user_prompt.c_str(), ids,
                                                        cap, n);
                },
                "EncodeChatMessage");
            const std::vector<int32_t> gen_prompt = query_ids(
                [&](int32_t* ids, uint32_t cap, uint32_t* n) {
                    return tokenizer->EncodeGenerationPrompt(ids, cap, n);
                },
                "EncodeGenerationPrompt");
            turn_tokens.insert(turn_tokens.end(), gen_prompt.begin(), gen_prompt.end());

            if (current_pos + turn_tokens.size() >= max_context) {
                std::cout << "\n[System Warning] Context limit reached!\n";
                break;
            }

            for (const int32_t token : turn_tokens) {
                check(engine->Forward(token, current_pos, 0.0f, 1.0f, 0, &next_token),
                      "Forward");
                current_pos++;
            }

            std::cout << "Assistant > ";

            while (current_pos < static_cast<int32_t>(max_context)) {
                BOOL is_stop = FALSE;
                check(tokenizer->IsStop(next_token, &is_stop), "IsStop");
                if (is_stop) {
                    // Feed the stop token into the cache so the model knows the
                    // assistant turn is closed.
                    int32_t ignored = -1;
                    check(engine->Forward(next_token, current_pos, 0.0f, 1.0f, 0, &ignored),
                          "Forward");
                    current_pos++;
                    break;
                }

                std::cout << decode_token(tokenizer.get(), next_token) << std::flush;

                check(engine->Forward(next_token, current_pos, 0.1f, 0.9f, 0, &next_token),
                      "Forward");
                current_pos++;
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "\n[CRITICAL ERROR]: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
