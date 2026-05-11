#include <iostream>
#include <string>
#include <exception>
#include "engine.h"

int main() {
    std::cout << "==================================================\n";
    std::cout << " Blackwell LLM: Full Engine Inference (1 Token) \n";
    std::cout << "==================================================\n\n";

    try {
        // Путь к конфигурации распределения весов
        std::string index_path = "llama3-8b-fp8/model.safetensors.index.json";
        
        std::cout << "[System] Initializing BlackwellEngine and static VRAM Arena...\n";
        BlackwellEngine engine(index_path, 2048);

        // Тестовые входные параметры (как в нашем Python-эталоне)
        int input_token = 128000;
        int current_pos = 0;

        std::cout << "[Engine] Executing accelerated forward pass...\n";
        
        // Запуск инференса одной строчкой
        int next_token = engine.forward(input_token, current_pos);

        std::cout << "\n--------------------------------------------------\n";
        std::cout << "[SUCCESS] Engine execution successfully completed!\n";
        std::cout << "          Input Token ID:  " << input_token << "\n";
        std::cout << "          Output Token ID: " << next_token << "\n";
        std::cout << "--------------------------------------------------\n\n";

    } catch (const std::exception& e) {
        std::cerr << "\n[CRITICAL ENGINE ERROR]: " << e.what() << "\n";
        return 1;
    }

    return 0;
}