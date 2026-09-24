// Interactive chat CLI -- the first true COM-side consumer of
// blackwell_core.dll: includes ONLY the public boundary header, creates the
// engine/tokenizer through the C factories, and speaks HRESULT. No C++
// exception ever crosses the DLL edge; the try/catch below handles only
// this file's own failures (check() throwing on a FAILED hr).
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

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
            continue;   // the only mode; accepted so the documented invocation works
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

    std::cout << "==================================================\n";
    std::cout << " Blackwell LLM: Interactive Chat Mode\n";
    std::cout << " (Type 'exit' or 'quit' to close)\n";
    std::cout << "==================================================\n\n";

    try {
        // The checkpoint directory fully describes the model: weights, config
        // and tokenizer all come from it, so any supported model (Llama-3,
        // Qwen2.5/ChatML, GLM-4/tiktoken, ...) runs through the same code path.
        const std::string& model_dir = opt.model_dir;
        const std::string index_path = model_dir + "/model.safetensors.index.json";
        const uint32_t max_context = opt.max_context;

        std::cout << "[System] Model: " << model_dir << "\n";

        IBlackwellTokenizer* tok_raw = nullptr;
        check(CreateBlackwellTokenizer(model_dir.c_str(), &tok_raw), "CreateBlackwellTokenizer");
        com_ptr<IBlackwellTokenizer> tokenizer(tok_raw);

        BLACKWELL_ENGINE_DESC desc{};
        desc.index_path         = index_path.c_str();
        desc.max_context_length = max_context;
        desc.num_gpu_layers     = opt.num_gpu_layers;
        desc.kv_mode            = BLACKWELL_KV_MODE_CONTINUOUS;

        IBlackwellEngine* eng_raw = nullptr;
        check(CreateBlackwellEngine(&desc, &eng_raw), "CreateBlackwellEngine");
        com_ptr<IBlackwellEngine> engine(eng_raw);

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
