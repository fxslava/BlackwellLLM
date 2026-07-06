// Interactive chat CLI -- the first true COM-side consumer of
// blackwell_core.dll: includes ONLY the public boundary header, creates the
// engine/tokenizer through the C factories, and speaks HRESULT. No C++
// exception ever crosses the DLL edge; the try/catch below handles only
// this file's own failures (check() throwing on a FAILED hr).
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <windows.h>

#include "blackwell/iblackwell_engine.h"

namespace {

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

    std::cout << "==================================================\n";
    std::cout << " Blackwell LLM: Interactive Chat Mode\n";
    std::cout << " (Type 'exit' or 'quit' to close)\n";
    std::cout << "==================================================\n\n";

    try {
        // The checkpoint directory fully describes the model: weights, config
        // and tokenizer all come from it, so any supported model (Llama-3,
        // Qwen2.5/ChatML, ...) runs through the same code path.
        const std::string model_dir = argc > 1 ? argv[1] : "F:/AI/llama3-8b-fp8";
        const std::string index_path = model_dir + "/model.safetensors.index.json";
        const uint32_t max_context = 16384;

        IBlackwellTokenizer* tok_raw = nullptr;
        check(CreateBlackwellTokenizer(model_dir.c_str(), &tok_raw), "CreateBlackwellTokenizer");
        com_ptr<IBlackwellTokenizer> tokenizer(tok_raw);

        BLACKWELL_ENGINE_DESC desc{};
        desc.index_path         = index_path.c_str();
        desc.max_context_length = max_context;
        desc.num_gpu_layers     = BLACKWELL_ALL_LAYERS_RESIDENT;
        desc.kv_mode            = BLACKWELL_KV_MODE_CONTINUOUS;

        IBlackwellEngine* eng_raw = nullptr;
        check(CreateBlackwellEngine(&desc, &eng_raw), "CreateBlackwellEngine");
        com_ptr<IBlackwellEngine> engine(eng_raw);

        int32_t current_pos = 0;

        // Conversation prefix (BOS + system block) rendered by the model's own
        // chat template; no token id appears anywhere in this file.
        const std::vector<int32_t> prelude = query_ids(
            [&](int32_t* ids, uint32_t cap, uint32_t* n) {
                return tokenizer->EncodeChatPrelude(
                    "You are a helpful, smart, and concise AI assistant.", ids, cap, n);
            },
            "EncodeChatPrelude");

        std::cout << "[System] Initializing context...\n";
        int32_t next_token = -1;
        for (const int32_t token : prelude) {
            check(engine->Forward(token, current_pos, 0.0f, 1.0f, 0, &next_token), "Forward");
            current_pos++;
        }

        while (true) {
            std::cout << "\n\nUser > ";
            std::string user_prompt;

            wchar_t wbuf[4096];
            DWORD read_chars = 0;
            HANDLE hStdin = GetStdHandle(STD_INPUT_HANDLE);
            FlushConsoleInputBuffer(hStdin);
            if (ReadConsoleW(hStdin, wbuf, 4096, &read_chars, NULL)) {
                while (read_chars > 0 && (wbuf[read_chars - 1] == L'\n' || wbuf[read_chars - 1] == L'\r')) read_chars--;
                if (read_chars > 0) {
                    int size = WideCharToMultiByte(CP_UTF8, 0, wbuf, read_chars, NULL, 0, NULL, NULL);
                    user_prompt.assign(size, 0);
                    WideCharToMultiByte(CP_UTF8, 0, wbuf, read_chars, &user_prompt[0], size, NULL, NULL);
                }
            }

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
