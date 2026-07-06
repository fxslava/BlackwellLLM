// Perplexity benchmark -- COM-side consumer of blackwell_core.dll (see
// main.cpp for the boundary conventions shared by both apps).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
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

struct ComRelease {
    template <typename T>
    void operator()(T* p) const {
        if (p) p->Release();
    }
};
template <typename T>
using com_ptr = std::unique_ptr<T, ComRelease>;

std::string decode_token(IBlackwellTokenizer* tok, int32_t token_id) {
    char buf[512];
    uint32_t len = 0;
    check(tok->DecodeToken(token_id, FALSE, buf, sizeof(buf), &len), "DecodeToken");
    return std::string(buf, len);
}

} // namespace

int main() {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    std::cout << "==================================================\n";
    std::cout << " Blackwell LLM: Perplexity (PPL) Benchmark\n";
    std::cout << "==================================================\n\n";

    try {
        const std::string model_dir = "F:/AI/llama3-8b-fp8";
        const std::string index_path = model_dir + "/model.safetensors.index.json";

        IBlackwellTokenizer* tok_raw = nullptr;
        check(CreateBlackwellTokenizer(model_dir.c_str(), &tok_raw), "CreateBlackwellTokenizer");
        com_ptr<IBlackwellTokenizer> tokenizer(tok_raw);

        BLACKWELL_ENGINE_DESC desc{};
        desc.index_path         = index_path.c_str();
        desc.max_context_length = 2048;
        desc.num_gpu_layers     = BLACKWELL_ALL_LAYERS_RESIDENT;
        desc.kv_mode            = BLACKWELL_KV_MODE_CONTINUOUS;

        IBlackwellEngine* eng_raw = nullptr;
        check(CreateBlackwellEngine(&desc, &eng_raw), "CreateBlackwellEngine");
        com_ptr<IBlackwellEngine> engine(eng_raw);

        // Текст для проверки. Здесь много запятых, фактов и логики.
        const std::string eval_text = "The GPU was popularized by Nvidia in 1999, who marketed the GeForce 256 as the world's first GPU. GPUs have evolved from fixed-function 3D graphics pipelines to highly programmable parallel processing engines. Modern AI models, including Large Language Models like GPT-4 and Llama 3, rely heavily on the tensor cores inside modern GPUs to perform billions of floating-point operations per second.";

        // Two-call encode: size query, then fill.
        uint32_t count = 0;
        check(tokenizer->Encode(eval_text.c_str(), TRUE, nullptr, 0, &count), "Encode");
        std::vector<int32_t> tokens(count);
        check(tokenizer->Encode(eval_text.c_str(), TRUE, tokens.data(), count, &count),
              "Encode");

        std::cout << "\n[Evaluator] Running benchmark on " << tokens.size() << " tokens...\n";
        std::cout << "--------------------------------------------------\n";

        float total_log_prob = 0.0f;
        int valid_tokens = 0;

        for (size_t i = 0; i + 1 < tokens.size(); ++i) {
            const int32_t current_token = tokens[i];
            const int32_t target_token = tokens[i + 1];

            float log_prob = 0.0f;
            check(engine->ForwardEval(current_token, static_cast<int32_t>(i), target_token,
                                      0, &log_prob),
                  "ForwardEval");
            total_log_prob += log_prob;
            valid_tokens++;

            std::string word = decode_token(tokenizer.get(), target_token);
            // Убираем переносы строк для красивого вывода
            word.erase(std::remove(word.begin(), word.end(), '\n'), word.end());

            std::cout << "Step " << std::setw(3) << i
                      << " | Target: '" << word << "' "
                      << "| LogProb: " << std::fixed << std::setprecision(4) << log_prob << "\n";
        }

        if (valid_tokens == 0)
            throw std::runtime_error("Benchmark: eval text produced fewer than 2 tokens, nothing to score");

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
