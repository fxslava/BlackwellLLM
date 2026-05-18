#include <iostream>
#include <string>
#include <vector>
#include <exception>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#endif

#include "engine.h"
#include "tokenizer.h"

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    std::cout << "==================================================\n";
    std::cout << " Blackwell LLM: Interactive Chat Mode\n";
    std::cout << " (Type 'exit' or 'quit' to close)\n";
    std::cout << "==================================================\n\n";

    try {
        std::string index_path = "F:/AI/llama3-8b-fp8/model.safetensors.index.json";
        std::string vocab_path = "F:/AI/llama3-8b-fp8/tokenizer.json"; 
        size_t max_context = 16384;
        
        LlamaTokenizer tokenizer(vocab_path);
        BlackwellEngine engine(index_path, max_context);

        int current_pos = 0;
        
        // 🎯 ФИКС 1: Собираем системный промпт через правильные ID
        std::vector<int> history_tokens;
        history_tokens.push_back(128000); // <|begin_of_text|>
        history_tokens.push_back(128006); // <|start_header_id|>
        history_tokens.push_back(9125);   // "system"
        history_tokens.push_back(128007); // <|end_header_id|>
        history_tokens.push_back(271);    // "\n\n"
        
        std::vector<int> sys_text = tokenizer.encode("You are a helpful, smart, and concise AI assistant.", false);
        history_tokens.insert(history_tokens.end(), sys_text.begin(), sys_text.end());
        history_tokens.push_back(128009); // <|eot_id|>

        std::cout << "[System] Initializing context...\n";
        int next_token = -1;
        for (int token : history_tokens) {
            next_token = engine.forward(token, current_pos, 0.0f, 1.0f);
            current_pos++;
        }

        std::vector<int> eos_tokens = {128001, 128008, 128009};

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

            // 🎯 ФИКС 2: Собираем реплику юзера и вызов ассистента через правильные ID
            std::vector<int> user_tokens;
            user_tokens.push_back(128006); // <|start_header_id|>
            user_tokens.push_back(882);    // "user"
            user_tokens.push_back(128007); // <|end_header_id|>
            user_tokens.push_back(271);    // "\n\n"
            
            std::vector<int> text_ids = tokenizer.encode(user_prompt, false);
            user_tokens.insert(user_tokens.end(), text_ids.begin(), text_ids.end());
            
            user_tokens.push_back(128009); // <|eot_id|> (Остановка юзера)
            user_tokens.push_back(128006); // <|start_header_id|>
            user_tokens.push_back(78191);  // "assistant"
            user_tokens.push_back(128007); // <|end_header_id|>
            user_tokens.push_back(271);    // "\n\n"

            if (current_pos + user_tokens.size() >= max_context) {
                std::cout << "\n[System Warning] Context limit reached!\n";
                break;
            }

            for (int token : user_tokens) {
                next_token = engine.forward(token, current_pos, 0.0f, 1.0f);
                current_pos++;
            }

            std::cout << "Llama > ";

            while (current_pos < max_context) {
                if (std::find(eos_tokens.begin(), eos_tokens.end(), next_token) != eos_tokens.end()) {
                    // Загоняем токен остановки в кэш, чтобы модель поняла, что она закончила!
                    engine.forward(next_token, current_pos, 0.0f, 1.0f);
                    current_pos++;
                    break;
                }

                std::cout << tokenizer.decode(next_token) << std::flush;

                next_token = engine.forward(next_token, current_pos, 0.6f, 0.9f);
                current_pos++;
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "\n[CRITICAL ERROR]: " << e.what() << "\n";
        return 1;
    }

    return 0;
}