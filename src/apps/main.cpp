#include <iostream>
#include <string>
#include <vector>
#include <exception>

#ifdef _WIN32
#include <windows.h>
#endif

#include "blackwell/engine.h"
#include "blackwell/tokenizer.h"

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

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
        const size_t max_context = 16384;

        auto tokenizer = blackwell::TokenizerFactory::create(model_dir);
        BlackwellEngine engine(index_path, max_context);

        int current_pos = 0;

        // Conversation prefix (BOS + system block) rendered by the model's own
        // chat template; no token id appears anywhere in this file.
        const std::vector<int> prelude = tokenizer->encode_chat_prelude(
            "You are a helpful, smart, and concise AI assistant.");

        std::cout << "[System] Initializing context...\n";
        int next_token = -1;
        for (const int token : prelude) {
            next_token = engine.forward(token, current_pos, 0.0f, 1.0f);
            current_pos++;
        }

        while (true) {
            std::cout << "\n\nUser > ";
            std::string user_prompt;

#ifdef _WIN32
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
#else
            std::getline(std::cin, user_prompt);
#endif

            if (user_prompt == "exit" || user_prompt == "quit") break;
            if (user_prompt.empty()) continue;

            // User turn + assistant cue, framed by the chat template.
            std::vector<int> turn_tokens =
                tokenizer->encode_chat_message({"user", user_prompt});
            const std::vector<int> gen_prompt = tokenizer->encode_generation_prompt();
            turn_tokens.insert(turn_tokens.end(), gen_prompt.begin(), gen_prompt.end());

            if (current_pos + turn_tokens.size() >= max_context) {
                std::cout << "\n[System Warning] Context limit reached!\n";
                break;
            }

            for (const int token : turn_tokens) {
                next_token = engine.forward(token, current_pos, 0.0f, 1.0f);
                current_pos++;
            }

            std::cout << "Assistant > ";

            while (current_pos < max_context) {
                if (tokenizer->is_stop(next_token)) {
                    // Feed the stop token into the cache so the model knows the
                    // assistant turn is closed.
                    engine.forward(next_token, current_pos, 0.0f, 1.0f);
                    current_pos++;
                    break;
                }

                std::cout << tokenizer->decode(next_token) << std::flush;

                next_token = engine.forward(next_token, current_pos, 0.1f, 0.9f);
                current_pos++;
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "\n[CRITICAL ERROR]: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
