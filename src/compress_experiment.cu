#include <iostream>
#include <vector>
#include <string>
#include <set>
#include <iomanip>
#include <algorithm>
#include <cuda_runtime.h>

#include "safetensors.h"
#include "memory_pool.h"
#include "cpu_models.h"
#include "nvcomp_baseline.h"

struct PaletteResult {
    std::string tensor_name;
    size_t original_bytes;
    float avg_unique_values;
    float theoretical_palette_ratio;
};

int main(int argc, char** argv) {
    std::cout << "🔬 Launching 16x16 Tile Palette Cardinality Analysis...\n";
    std::string model_path = (argc > 1) ? argv[1] : "F:/AI/llama3-8b-fp8/model.safetensors.index.json";
    
    try {
        SafetensorsLoader metadata_loader(model_path);
        VRAMArena arena(model_path, metadata_loader, 2048);
        auto all_tensors = metadata_loader.list_tensors();
        std::vector<PaletteResult> sweep_results;

        size_t total_orig_bytes = 0;
        size_t total_palette_compressed_bytes = 0;

        for (const auto& tensor_name : all_tensors) {
            const void* d_weight_ptr = arena.get_weight_ptr_optional(tensor_name);
            if (!d_weight_ptr) continue;
            
            auto tensor_meta = metadata_loader.get_tensor(tensor_name);
            size_t tensor_size = tensor_meta.byte_size;
            auto shape = tensor_meta.shape;
            if (shape.size() != 2 || shape[0] % 16 != 0 || shape[1] % 16 != 0) continue;

            std::vector<uint8_t> h_weights(tensor_size);
            cudaMemcpy(h_weights.data(), d_weight_ptr, tensor_size, cudaMemcpyDeviceToHost);
            
            size_t rows = shape[0], cols = shape[1];
            size_t num_tiles = (rows / 16) * (cols / 16);
            
            uint64_t total_unique_across_tensor = 0;
            size_t tensor_compressed_bytes = 0;

            for (size_t r = 0; r < rows; r += 16) {
                for (size_t c = 0; c < cols; c += 16) {
                    
                    // Собираем уникальные значения внутри тайла через bit-set (быстрее чем std::set)
                    bool unique_flags[256] = {false};
                    int unique_count = 0;

                    for (size_t tr = 0; tr < 16; ++tr) {
                        for (size_t tc = 0; tc < 16; ++tc) {
                            uint8_t val = h_weights[(r + tr) * cols + (c + tc)];
                            if (!unique_flags[val]) {
                                unique_flags[val] = true;
                                unique_count++;
                            }
                        }
                    }

                    total_unique_across_tensor += unique_count;

                    // Вычисляем, сколько бит нужно на один индекс для этого тайла
                    int bits_per_index = 0;
                    if (unique_count <= 2) bits_per_index = 1;
                    else if (unique_count <= 4) bits_per_index = 2;
                    else if (unique_count <= 8) bits_per_index = 3;
                    else if (unique_count <= 16) bits_per_index = 4;
                    else if (unique_count <= 32) bits_per_index = 5;
                    else if (unique_count <= 64) bits_per_index = 6;
                    else if (unique_count <= 128) bits_per_index = 7;
                    else bits_per_index = 8;

                    // Стоимость тайла = (размер палитры в байтах) + (256 индексов * bits_per_index) / 8
                    size_t tile_indices_bytes = (256 * bits_per_index) / 8;
                    size_t tile_palette_bytes = unique_count; // 1 байт на каждый элемент палитры

                    tensor_compressed_bytes += tile_palette_bytes + tile_indices_bytes;
                }
            }

            float avg_unique = static_cast<float>(total_unique_across_tensor) / num_tiles;
            float ratio = static_cast<float>(tensor_size) / tensor_compressed_bytes;

            sweep_results.push_back({tensor_name, tensor_size, avg_unique, ratio});

            total_orig_bytes += tensor_size;
            total_palette_compressed_bytes += tensor_compressed_bytes;
        }

        std::cout << "\n=================================== 16x16 TILE PALETTE POTENTIAL REPORT ===================================\n";
        std::cout << std::left << std::setw(45) << "Tensor Name (2D Matrices)" << std::setw(15) << "Orig (MB)" << std::setw(22) << "Avg Unique Vals/Tile" << std::setw(15) << "Palette Ratio" << "\n";
        std::cout << "------------------------------------------------------------------------------------------------------------\n";
        for (const auto& r : sweep_results) {
            std::cout << std::left << std::setw(45) << (r.tensor_name.length() > 42 ? r.tensor_name.substr(0, 39) + "..." : r.tensor_name)
                      << std::setw(15) << std::fixed << std::setprecision(1) << (r.original_bytes / 1024.0 / 1024.0)
                      << std::setw(22) << std::fixed << std::setprecision(2) << r.avg_unique_values
                      << std::setw(15) << (std::to_string(r.theoretical_palette_ratio).substr(0,4) + "x") << "\n";
        }
        std::cout << "------------------------------------------------------------------------------------------------------------\n";
        float final_ratio = static_cast<float>(total_orig_bytes) / total_palette_compressed_bytes;
        std::cout << std::left << std::setw(45) << "TOTAL MODEL SUMMARY" 
                  << std::setw(15) << std::fixed << std::setprecision(1) << (total_orig_bytes / 1024.0 / 1024.0)
                  << std::setw(22) << "---"
                  << std::setw(15) << (std::to_string(final_ratio).substr(0,4) + "x") << "\n";
        std::cout << "============================================================================================================\n\n";

    } catch (const std::exception& e) { std::cerr << "\n[CRITICAL ERROR] " << e.what() << "\n"; return 1; }
    return 0;
}