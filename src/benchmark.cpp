#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <iomanip>

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
    std::cout << " Blackwell LLM: Perplexity (PPL) Benchmark\n";
    std::cout << "==================================================\n\n";

    try {
        std::string index_path = "F:/AI/llama3-8b-fp8/model.safetensors.index.json";
        std::string vocab_path = "F:/AI/llama3-8b-fp8/tokenizer.json"; 
        
        LlamaTokenizer tokenizer(vocab_path);
        BlackwellEngine engine(index_path, 2048);

        // Текст для проверки. Здесь много запятых, фактов и логики.
        std::string eval_text = "The GPU was popularized by Nvidia in 1999, who marketed the GeForce 256 as the world's first GPU. GPUs have evolved from fixed-function 3D graphics pipelines to highly programmable parallel processing engines. Modern AI models, including Large Language Models like GPT-4 and Llama 3, rely heavily on the tensor cores inside modern GPUs to perform billions of floating-point operations per second.";
        
        std::vector<int> tokens = tokenizer.encode(eval_text, true);
        
        std::cout << "\n[Evaluator] Running benchmark on " << tokens.size() << " tokens...\n";
        std::cout << "--------------------------------------------------\n";
        
        float total_log_prob = 0.0f;
        int valid_tokens = 0;

        for (size_t i = 0; i < tokens.size() - 1; ++i) {
            int current_token = tokens[i];
            int target_token = tokens[i+1];
            
            float log_prob = engine.forward_eval(current_token, i, target_token);
            total_log_prob += log_prob;
            valid_tokens++;

            std::string word = tokenizer.decode(target_token);
            // Убираем переносы строк для красивого вывода
            word.erase(std::remove(word.begin(), word.end(), '\n'), word.end());
            
            std::cout << "Step " << std::setw(3) << i 
                      << " | Target: '" << word << "' "
                      << "| LogProb: " << std::fixed << std::setprecision(4) << log_prob << "\n";
        }

        // Формула Перплексии: exp(-1/N * sum(log(P)))
        float avg_log_prob = total_log_prob / valid_tokens;
        float perplexity = std::exp(-avg_log_prob);

        std::cout << "\n==================================================\n";
        std::cout << "[RESULTS]\n";
        std::cout << "  Average Log-Likelihood : " << avg_log_prob << "\n";
        std::cout << "  Final Perplexity (PPL) : " << perplexity << "\n";
        std::cout << "==================================================\n";

    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR]: " << e.what() << "\n";
        return 1;
    }
    return 0;
}