#include <iostream>
#include <string>
#include <vector>
#include <exception>
#include <chrono>
#include <algorithm>
#include "engine.h"
#include "tokenizer.h" // 🎯 Подключаем наш кастомный токенизатор

int main() {
    std::cout << "==================================================\n";
    std::cout << " Blackwell LLM: Autoregressive Generation Engine\n";
    std::cout << "==================================================\n\n";

    try {
        std::string index_path = "llama3-8b-fp8/model.safetensors.index.json";
        std::string vocab_path = "llama3-8b-fp8/tokenizer.json"; // 🎯 Путь к JSON конфигурации BPE
        size_t max_context = 2048;
        
        std::cout << "[System] Loading Llama 3 BPE Vocabulary mappings...\n";
        LlamaTokenizer tokenizer(vocab_path);

        std::cout << "[System] Initializing BlackwellEngine and static VRAM Arena...\n";
        BlackwellEngine engine(index_path, max_context);

        // Хардкодим промпт. 128000 - это <|begin_of_text|>
        std::vector<int> prompt_tokens = {128000, 9906, 94776, 0}; 
        
        // EOS токены Llama 3.1
        std::vector<int> eos_tokens = {128001, 128008, 128009};
        
        int current_pos = 0;
        int next_token = -1;

        std::cout << "\n[Engine] Phase 1: Prefill (Processing prompt)...\n";
        std::cout << "Prompt IDs: ";
        
        // --- 1. PREFILL PHASE ---
        for (size_t i = 0; i < prompt_tokens.size(); ++i) {
            std::cout << prompt_tokens[i] << " " << std::flush;
            next_token = engine.forward(prompt_tokens[i], current_pos);
            current_pos++;
        }

        std::cout << "\n\n[Engine] Phase 2: Decoding (Autoregressive generation)...";
        std::cout << "\n--------------------------------------------------\n";
        std::cout << "[Blackwell LLM Output]: "; // 🎯 Подготавливаем красивый вывод текста
        
        std::vector<int> generated_tokens;
        auto start_time = std::chrono::high_resolution_clock::now();

        // --- 2. DECODING PHASE ---
        while (current_pos < max_context) {
            // Проверяем, не сгенерировала ли сеть токен конца текста
            if (std::find(eos_tokens.begin(), eos_tokens.end(), next_token) != eos_tokens.end()) {
                // 🎯 Наш decode внутри tokenizer.cpp сам выведет красивый тег [EOS], так что здесь просто выходим
                tokenizer.decode(next_token); 
                break;
            }

            // Сохраняем токен
            generated_tokens.push_back(next_token);
            
            // 🎯 ДЕКОДИРУЕМ И СТРИМИМ: Превращаем ID в сырые байты, чистим UTF-8 маски и сразу кидаем в stdout
            std::string text_piece = tokenizer.decode(next_token);
            std::cout << text_piece << std::flush;

            // Кормим сгенерированный токен обратно в сеть
            next_token = engine.forward(next_token, current_pos);
            current_pos++;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;

        std::cout << "\n--------------------------------------------------\n\n";

        if (current_pos >= max_context) {
            std::cout << "[System] Reached maximum context length (" << max_context << ").\n";
        }

        // Выводим статистику производительности
        double tokens_per_sec = generated_tokens.size() / duration.count();
        std::cout << "[STATISTICS]\n";
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