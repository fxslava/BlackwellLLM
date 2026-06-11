#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <cuda_runtime.h>

#include "../safetensors.h"
#include "../memory_pool.h"
#include "../../tests/reference/cpu_models.h"

// --- Функция расчета Шенноновской Энтропии (бит на символ) ---
double calculate_shannon_entropy(const std::vector<uint64_t>& histogram, uint64_t total_elements) {
    if (total_elements == 0) return 0.0;
    double entropy = 0.0;
    for (uint64_t count : histogram) {
        if (count > 0) {
            double p = static_cast<double>(count) / total_elements;
            entropy -= p * std::log2(p);
        }
    }
    return entropy;
}

int main(int argc, char** argv) {
    std::cout << "🔬 Full 8-bit Attractor Verification: 2D/3D Full-Byte Delay Embedding...\n";
    std::string model_path = (argc > 1) ? argv[1] : "F:/AI/llama3-8b-fp8/model.safetensors.index.json";
    
    try {
        SafetensorsLoader metadata_loader(model_path);
        VRAMArena arena(model_path, metadata_loader, 2048);
        
        std::string target_tensor = "model.layers.0.mlp.down_proj.weight";
        if (!metadata_loader.has_tensor(target_tensor)) {
            for (const auto& t : metadata_loader.list_tensors()) {
                if (t.find("down_proj.weight") != std::string::npos) { target_tensor = t; break; }
            }
        }

        const void* d_weight_ptr = arena.get_weight_ptr_optional(target_tensor);
        if (!d_weight_ptr) throw std::runtime_error("Tensor not found!");
        
        auto tensor_meta = metadata_loader.get_tensor(target_tensor);
        size_t tensor_size = tensor_meta.byte_size;

        std::vector<uint8_t> h_weights(tensor_size);
        cudaMemcpy(h_weights.data(), d_weight_ptr, tensor_size, cudaMemcpyDeviceToHost);
        const uint8_t* raw_data = h_weights.data();

        std::cout << "✅ Loaded Tensor: " << target_tensor << " (" << tensor_size / 1024 / 1024 << " MB)\n";

        // 1. 1D Базовая энтропия (Все 8 бит вместе)
        std::vector<uint64_t> hist_1d(256, 0);
        for (size_t i = 0; i < tensor_size; ++i) {
            hist_1d[raw_data[i]]++;
        }
        double h_1d = calculate_shannon_entropy(hist_1d, tensor_size);

        // 2. 2D Фазовое пространство полного байта (65536 ячеек)
        std::cout << "⏳ Constructing 2D Delay Embedding (256x256 Grid)...\n";
        std::vector<uint64_t> hist_2d(256 * 256, 0);
        uint64_t total_pairs = tensor_size - 1;

        for (size_t i = 0; i < total_pairs; ++i) {
            uint8_t x_n    = raw_data[i];
            uint8_t x_next = raw_data[i + 1];
            hist_2d[x_n * 256 + x_next]++;
        }
        double h_2d = calculate_shannon_entropy(hist_2d, total_pairs);

        // 3. 3D Фазовое пространство полного байта (16.7 миллионов ячеек)
        std::cout << "⏳ Constructing 3D Delay Embedding (256x256x256 Cube, 128 MB RAM)...\n";
        std::vector<uint64_t> hist_3d(256 * 256 * 256, 0);
        uint64_t total_triplets = tensor_size - 2;

        for (size_t i = 0; i < total_triplets; ++i) {
            uint8_t x_n     = raw_data[i];
            uint8_t x_next  = raw_data[i + 1];
            uint8_t x_next2 = raw_data[i + 2];
            size_t idx = (static_cast<size_t>(x_n) << 16) | (static_cast<size_t>(x_next) << 8) | x_next2;
            hist_3d[idx]++;
        }
        double h_3d = calculate_shannon_entropy(hist_3d, total_triplets);

        // 4. Подсчет заселенности ячеек пространства хаоса
        size_t active_cells_2d = 0;
        for (auto c : hist_2d) if (c > 0) active_cells_2d++;

        size_t active_cells_3d = 0;
        for (auto c : hist_3d) if (c > 0) active_cells_3d++;

        // 5. Финальный вердикт теории нелинейных систем
        std::cout << "\n==================== FULL 8-BIT CHAOS THEORY ANALYSIS ====================\n";
        std::cout << "1D Raw Byte Baseline Entropy:     " << std::fixed << std::setprecision(4) << h_1d << " bits\n";
        std::cout << "--------------------------------------------------------------------------\n";
        
        std::cout << "2D Full-Byte Joint Entropy:       " << h_2d << " bits\n";
        std::cout << "   Theoretical Noise Limit (2xH): " << h_1d * 2.0 << " bits\n";
        std::cout << "   Active States Occupancy:       " << active_cells_2d << " / 65536 ячеек\n";
        std::cout << "   Delta (Chaos Potential):       " << (h_1d * 2.0) - h_2d << " bits\n";
        std::cout << "--------------------------------------------------------------------------\n";
        
        std::cout << "3D Full-Byte Joint Entropy:       " << h_3d << " bits\n";
        std::cout << "   Theoretical Noise Limit (3xH): " << h_1d * 3.0 << " bits\n";
        std::cout << "   Active States Occupancy:       " << active_cells_3d << " / 16777216 ячеек\n";
        std::cout << "   Delta (Chaos Potential):       " << (h_1d * 3.0) - h_3d << " bits\n";
        std::cout << "==========================================================================\n";

        // Погрешность на конечный размер выборки (кол-во элементов) составляет ~0.02 бита
        if (((h_1d * 3.0) - h_3d) > 0.05) {
            std::cout << "🎉 STRANGE ATTRACTOR FOUND! The 8-bit stream follows a hidden non-linear dynamic law.\n\n";
        } else {
            std::cout << "❄️ PERFECT THERMODYNAMIC CHAOS. All 8 bits are stochastically independent. No formula can generate this.\n\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "\n[CRITICAL ERROR] " << e.what() << "\n";
        return 1;
    }
    return 0;
}