#include <iostream>
#include <string>
#include <vector>
#include <exception>
#include <chrono>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#endif

#include "engine.h"
#include "tokenizer.h"

int main() {
    // === ХАК ДЛЯ КИРИЛЛИЦЫ В WINDOWS ===
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8); // Заставляем консоль выводить UTF-8
    SetConsoleCP(CP_UTF8);       // Заставляем консоль читать UTF-8 с клавиатуры
#endif
    // ===================================

    std::cout << "==================================================\n";
    std::cout << " Blackwell LLM: Full Text-to-Text End-to-End Engine\n";
    std::cout << "==================================================\n\n";

    try {
        std::string index_path = "llama3-8b-fp8/model.safetensors.index.json";
        std::string vocab_path = "llama3-8b-fp8/tokenizer.json"; 
        size_t max_context = 2048;
        
        std::cout << "[System] Loading Llama 3 BPE Mappings...\n";
        LlamaTokenizer tokenizer(vocab_path);

        std::cout << "[System] Initializing BlackwellEngine and static VRAM Arena...\n";
        BlackwellEngine engine(index_path, max_context);

        // ЖИВОЙ ВВОД ПРОМПТА ИЗ КОНСОЛИ
        std::cout << "\n==================================================\n";
        std::cout << "Enter your prompt: ";
        std::string user_prompt;
#ifdef _WIN32
        // Читаем сырой UTF-16 прямо из консоли Windows
        wchar_t wbuf[4096];
        DWORD read_chars = 0;
        HANDLE hStdin = GetStdHandle(STD_INPUT_HANDLE);
        
        // Очищаем буфер от мусора
        FlushConsoleInputBuffer(hStdin); 
        
        if (ReadConsoleW(hStdin, wbuf, 4096, &read_chars, NULL)) {
            // Отрезаем символы переноса строки (\r\n)
            while (read_chars > 0 && (wbuf[read_chars - 1] == L'\n' || wbuf[read_chars - 1] == L'\r')) {
                read_chars--;
            }
            // Конвертируем UTF-16 в правильный UTF-8 для токенизатора
            if (read_chars > 0) {
                int size_needed = WideCharToMultiByte(CP_UTF8, 0, wbuf, read_chars, NULL, 0, NULL, NULL);
                user_prompt.assign(size_needed, 0);
                WideCharToMultiByte(CP_UTF8, 0, wbuf, read_chars, &user_prompt[0], size_needed, NULL, NULL);
            }
        }
#else
        // На Linux/Mac std::cin работает с UTF-8 идеально из коробки
        std::getline(std::cin, user_prompt);
#endif
        
        if (user_prompt.empty()) {
            user_prompt = "Hello llama!"; // Дефолтный фолбэк
        }

        // --- 1. ТОКЕНИЗАЦИЯ ВХОДНОГО ТЕКСТА ---
        std::vector<int> prompt_tokens;
        
        // 1. Заголовок пользователя
        prompt_tokens.push_back(128000); // <|begin_of_text|>
        prompt_tokens.push_back(128006); // <|start_header_id|>
        prompt_tokens.push_back(882);    // "user"
        prompt_tokens.push_back(128007); // <|end_header_id|>
        prompt_tokens.push_back(271);    // "\n\n"
        
        // 2. Текст пользователя (без BOS, так как мы его уже добавили)
        std::vector<int> user_text_ids = tokenizer.encode(user_prompt, false);
        prompt_tokens.insert(prompt_tokens.end(), user_text_ids.begin(), user_text_ids.end());
        
        // 3. Заголовок ассистента (призыв к ответу)
        prompt_tokens.push_back(128009); // <|eot_id|> (Конец реплики юзера)
        prompt_tokens.push_back(128006); // <|start_header_id|>
        prompt_tokens.push_back(78191);  // "assistant"
        prompt_tokens.push_back(128007); // <|end_header_id|>
        prompt_tokens.push_back(271);    // "\n\n"
        
        std::cout << "\n[Tokenizer] Encoded Prompt IDs: ";
        for (int id : prompt_tokens) {
            std::cout << id << " ";
        }
        std::cout << "\n";
        
        // EOS токены остановки Llama 3
        std::vector<int> eos_tokens = {128001, 128008, 128009};
        
        int current_pos = 0;
        int next_token = -1;

        std::cout << "\n[Engine] Phase 1: Prefill (Processing prompt)... \n";
        
        // --- 2. PREFILL PHASE ---
        for (size_t i = 0; i < prompt_tokens.size(); ++i) {
            next_token = engine.forward(prompt_tokens[i], current_pos);
            current_pos++;
        }

        std::cout << "\n[Engine] Phase 2: Decoding (Autoregressive generation)...";
        std::cout << "\n--------------------------------------------------\n";
        std::cout << "[Blackwell LLM]: "; 
        
        std::vector<int> generated_tokens;
        auto start_time = std::chrono::high_resolution_clock::now();

        // --- 3. DECODING PHASE (Streaming) ---
        while (current_pos < max_context) {
            // Проверяем токен конца генерации
            if (std::find(eos_tokens.begin(), eos_tokens.end(), next_token) != eos_tokens.end()) {
                tokenizer.decode(next_token); // Наш декодер красиво напечатает [EOS]
                break;
            }

            generated_tokens.push_back(next_token);
            
            // Декодируем текущий токен и мгновенно выводим в консоль
            std::string text_piece = tokenizer.decode(next_token);
            std::cout << text_piece << std::flush;

            // Передаём токен обратно на следующий шаг инференса
            next_token = engine.forward(next_token, current_pos);
            current_pos++;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;

        std::cout << "\n--------------------------------------------------\n\n";

        if (current_pos >= max_context) {
            std::cout << "[System] Reached maximum context length (" << max_context << ").\n";
        }

        // Выводим статистику скорости
        double tokens_per_sec = generated_tokens.size() / duration.count();
        std::cout << "[PERFORMANCE STATS]\n";
        std::cout << "  Prompt tokens:    " << prompt_tokens.size() << "\n";
        std::cout << "  Generated tokens: " << generated_tokens.size() << "\n";
        std::cout << "  Generation speed: " << tokens_per_sec << " tokens/sec\n";
        std::cout << "--------------------------------------------------\n\n";

    } catch (const std::exception& e) {
        std::cerr << "\n[CRITICAL ENGINE ERROR]: " << e.what() << "\n";
        return 1;
    }

    return 0;
}