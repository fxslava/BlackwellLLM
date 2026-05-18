#include <iostream>
#include <vector>
#include <string>
#include <queue>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>

#include "safetensors.h"
#include "memory_pool.h"
#include "cpu_models.h"
#include "nvcomp_baseline.h"

struct SweepResult {
    std::string tensor_name;
    size_t original_bytes;
    float r_raster;
    float r_custom_swizzle;
    float r_cluster8;
    float r_ideal;
};

// 🎯 Твой кастомный swizzle бит адресации (мапинг 2D координат в 1D индекс)
// Группирует биты как Y3 Y2 X3 X2 Y1 Y0 X1 X0 (Локальные макро/микро тайлы 4х4)
inline static unsigned int custom_swizzle_encode(unsigned int x, unsigned int y) {
    unsigned int y3 = (y & 8) >> 3;
    unsigned int y2 = (y & 4) >> 2;
    unsigned int y1 = (y & 2) >> 1;
    unsigned int y0 = (y & 1);
    
    unsigned int x3 = (x & 8) >> 3;
    unsigned int x2 = (x & 4) >> 2;
    unsigned int x1 = (x & 2) >> 1;
    unsigned int x0 = (x & 1);
    
    return (y3 << 7) | (y2 << 6) | (x3 << 5) | (x2 << 4) | 
           (y1 << 3) | (y0 << 2) | (x1 << 1) | x0;
}

struct HuffmanNode {
    uint8_t byte_value;
    uint64_t freq;
    HuffmanNode* left;
    HuffmanNode* right;
    HuffmanNode(uint8_t val, uint64_t f) : byte_value(val), freq(f), left(nullptr), right(nullptr) {}
};

struct CompareNodes {
    bool operator()(HuffmanNode* const& n1, HuffmanNode* const& n2) { return n1->freq > n2->freq; }
};

void free_huffman_tree(HuffmanNode* root) {
    if (!root) return;
    free_huffman_tree(root->left); free_huffman_tree(root->right); delete root;
}

void calculate_bit_lengths(HuffmanNode* root, int current_depth, std::vector<int>& bit_lengths) {
    if (!root) return;
    if (!root->left && !root->right) { bit_lengths[root->byte_value] = (current_depth == 0) ? 1 : current_depth; return; }
    calculate_bit_lengths(root->left, current_depth + 1, bit_lengths);
    calculate_bit_lengths(root->right, current_depth + 1, bit_lengths);
}

std::vector<int> get_huffman_lengths(const std::vector<uint64_t>& histogram) {
    std::priority_queue<HuffmanNode*, std::vector<HuffmanNode*>, CompareNodes> min_heap;
    for (int i = 0; i < 256; ++i) { if (histogram[i] > 0) min_heap.push(new HuffmanNode(static_cast<uint8_t>(i), histogram[i])); }
    std::vector<int> bit_lengths(256, 0);
    if (min_heap.empty()) return bit_lengths;
    while (min_heap.size() > 1) {
        HuffmanNode* left = min_heap.top(); min_heap.pop();
        HuffmanNode* right = min_heap.top(); min_heap.pop();
        HuffmanNode* parent = new HuffmanNode(0, left->freq + right->freq);
        parent->left = left; parent->right = right; min_heap.push(parent);
    }
    HuffmanNode* root = min_heap.top(); calculate_bit_lengths(root, 0, bit_lengths); free_huffman_tree(root);
    return bit_lengths;
}

size_t evaluate_delta_stream_bytes(const std::vector<uint64_t>& delta_hist, size_t num_tiles) {
    std::vector<int> lengths = get_huffman_lengths(delta_hist);
    size_t bits = 0;
    for (int i = 0; i < 256; ++i) bits += delta_hist[i] * lengths[i];
    return ((bits + 7) / 8) + (num_tiles * 1);
}

int main(int argc, char** argv) {
    std::cout << "🔬 Launching Cluster-8 Factoradic Sort & Custom Address Swizzle Experiment...\n";
    std::string model_path = (argc > 1) ? argv[1] : "F:/AI/llama3-8b-fp8/model.safetensors.index.json";
    
    try {
        SafetensorsLoader metadata_loader(model_path);
        VRAMArena arena(model_path, metadata_loader, 2048);
        auto all_tensors = metadata_loader.list_tensors();
        std::vector<SweepResult> sweep_results;

        // Построение LUT-таблицы для твоего кастомного swizzle обхода внутри тайла 16х16
        std::vector<int> custom_order(256);
        for(int r=0; r<16; ++r) {
            for(int c=0; c<16; ++c) {
                unsigned int code = custom_swizzle_encode(c, r);
                custom_order[code] = r * 16 + c;
            }
        }

        size_t total_orig = 0, total_raster = 0, total_custom = 0, total_c8 = 0, total_ideal = 0;

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

            std::vector<uint64_t> hist_raster(256, 0), hist_custom(256, 0), hist_cluster8(256, 0), hist_ideal(256, 0);

            for (size_t r = 0; r < rows; r += 16) {
                for (size_t c = 0; c < cols; c += 16) {
                    std::vector<uint8_t> tile(256);
                    for (size_t tr = 0; tr < 16; ++tr) {
                        for (size_t tc = 0; tc < 16; ++tc) {
                            tile[tr * 16 + tc] = h_weights[(r + tr) * cols + (c + tc)];
                        }
                    }

                    // 1. Обычный Raster
                    for(size_t i=0; i<255; ++i) hist_raster[static_cast<uint8_t>(tile[i+1] - tile[i])]++;

                    // 2. Твой кастомный Блочный Swizzle XXXXYYYY
                    for(size_t i=0; i<255; ++i) {
                        uint8_t v1 = tile[custom_order[i]];
                        uint8_t v2 = tile[custom_order[i+1]];
                        hist_custom[static_cast<uint8_t>(v2 - v1)]++;
                    }

                    // 3. Твоя Факториальная Сортировка по N=8 элементам
                    for (size_t block = 0; block < 32; ++block) {
                        std::vector<uint8_t> sub_block(8);
                        for(int i=0; i<8; ++i) sub_block[i] = tile[block * 8 + i];
                        
                        // Сортируем мини-блок из 8 элементов
                        std::sort(sub_block.begin(), sub_block.end());
                        
                        // Считаем 7 дельт внутри отсортированного мини-блока
                        for(size_t i=0; i<7; ++i) {
                            hist_cluster8[static_cast<uint8_t>(sub_block[i+1] - sub_block[i])]++;
                        }
                    }

                    // 4. Идеал (Сортировка всего тайла 256 элементов)
                    std::sort(tile.begin(), tile.end());
                    for(size_t i=0; i<255; ++i) {
                        hist_ideal[static_cast<uint8_t>(tile[i+1] - tile[i])]++;
                    }
                }
            }

            size_t b_raster = evaluate_delta_stream_bytes(hist_raster, num_tiles);
            size_t b_custom = evaluate_delta_stream_bytes(hist_custom, num_tiles);
            size_t b_ideal  = evaluate_delta_stream_bytes(hist_ideal, num_tiles);

            // Чистый подсчет для Cluster-8 с учетом жесткого налога на метаданные (96 байт на тайл)
            std::vector<int> c8_lengths = get_huffman_lengths(hist_cluster8);
            size_t c8_bits = 0;
            for(int i=0; i<256; ++i) c8_bits += hist_cluster8[i] * c8_lengths[i];
            size_t b_cluster8 = ((c8_bits + 7) / 8) + (num_tiles * 96);

            sweep_results.push_back({
                tensor_name, tensor_size, 
                static_cast<float>(tensor_size)/b_raster, static_cast<float>(tensor_size)/b_custom,
                static_cast<float>(tensor_size)/b_cluster8, static_cast<float>(tensor_size)/b_ideal
            });

            total_orig += tensor_size; total_raster += b_raster; total_custom += b_custom; total_c8 += b_cluster8; total_ideal += b_ideal;
        }

        std::cout << "\n=================================== ADVANCED TRAVERSAL & CLUSTER REPORT ===================================\n";
        std::cout << std::left << std::setw(38) << "Tensor Name" << std::setw(12) << "Orig(MB)" << std::setw(15) << "Raster Delta" << std::setw(15) << "Custom Swizzle" << std::setw(16) << "Cluster-8 Sort" << std::setw(14) << "Ideal Sorted" << "\n";
        std::cout << "----------------------------------------------------------------------------------------------------------------------------\n";
        for (const auto& r : sweep_results) {
            std::cout << std::left << std::setw(38) << (r.tensor_name.length() > 35 ? r.tensor_name.substr(0, 32) + "..." : r.tensor_name)
                      << std::setw(12) << std::fixed << std::setprecision(1) << (r.original_bytes / 1024.0 / 1024.0)
                      << std::setw(15) << (std::to_string(r.r_raster).substr(0,4) + "x")
                      << std::setw(15) << (std::to_string(r.r_custom_swizzle).substr(0,4) + "x")
                      << std::setw(16) << (std::to_string(r.r_cluster8).substr(0,4) + "x")
                      << std::setw(14) << (std::to_string(r.r_ideal).substr(0,4) + "x") << "\n";
        }
        std::cout << "----------------------------------------------------------------------------------------------------------------------------\n";
        std::cout << std::left << std::setw(38) << "TOTAL MODEL SUMMARY" << std::setw(12) << std::fixed << std::setprecision(1) << (total_orig / 1024.0 / 1024.0)
                  << std::setw(15) << (std::to_string(static_cast<float>(total_orig)/total_raster).substr(0,4) + "x")
                  << std::setw(15) << (std::to_string(static_cast<float>(total_orig)/total_custom).substr(0,4) + "x")
                  << std::setw(16) << (std::to_string(static_cast<float>(total_orig)/total_c8).substr(0,4) + "x")
                  << std::setw(14) << (std::to_string(static_cast<float>(total_orig)/total_ideal).substr(0,4) + "x") << "\n";
        std::cout << "============================================================================================================================\n\n";

    } catch (const std::exception& e) { std::cerr << "\n[CRITICAL ERROR] " << e.what() << "\n"; return 1; }
    return 0;
}