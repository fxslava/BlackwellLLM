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
    float r_morton;
    float r_hilbert;
    float r_ideal;
};

// --- ФРАКТАЛЬНЫЕ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

// Генератор Z-кривой (Morton Code) для 2D координат
inline static unsigned int morton_encode(unsigned int x, unsigned int y) {
    unsigned int answer = 0;
    for (unsigned int i = 0; i < 4; ++i) { // Для сетки 16х16 достаточно 4 бит
        answer |= ((x & (1 << i)) << i) | ((y & (1 << i)) << (i + 1));
    }
    return answer;
}

// Генератор кривой Гильберта (Hilbert Curve) для 2D координат
static void hilbert_d2xy(int n, int d, int *x, int *y) {
    int rx, ry, s, t = d;
    *x = *y = 0;
    for (s = 1; s < n; s *= 2) {
        rx = 1 & (t / 2);
        ry = 1 & (t ^ rx);
        // Поворот системы координат
        if (ry == 0) {
            if (rx == 1) {
                *x = s - 1 - *x;
                *y = s - 1 - *y;
            }
            int tmp = *x; *x = *y; *y = tmp;
        }
        *x += s * rx;
        *y += s * ry;
        t /= 4;
    }
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

// Вспомогательный метод оценки размера по гистограмме дельт + оверхед базовых байт
size_t evaluate_delta_stream_bytes(const std::vector<uint64_t>& delta_hist, size_t num_tiles) {
    std::vector<int> lengths = get_huffman_lengths(delta_hist);
    size_t bits = 0;
    for (int i = 0; i < 256; ++i) bits += delta_hist[i] * lengths[i];
    return ((bits + 7) / 8) + (num_tiles * 1); // Дельты + 1 базовый байт на тайл
}

int main(int argc, char** argv) {
    std::cout << "🔬 Launching Space-Filling Curves vs Ideal Sort 16x16 Tile Experiment...\n";
    std::string model_path = (argc > 1) ? argv[1] : "F:/AI/llama3-8b-fp8/model.safetensors.index.json";
    
    try {
        SafetensorsLoader metadata_loader(model_path);
        VRAMArena arena(model_path, metadata_loader, 2048);
        auto all_tensors = metadata_loader.list_tensors();
        std::vector<SweepResult> sweep_results;

        // Построение карт индексов для Мортона и Гильберта внутри тайла 16х16 один раз
        std::vector<int> morton_order(256);
        std::vector<std::pair<int, int>> morton_coords(256);
        for(int r=0; r<16; ++r) {
            for(int c=0; c<16; ++c) {
                unsigned int code = morton_encode(c, r);
                morton_order[code] = r * 16 + c;
            }
        }

        std::vector<int> hilbert_order(256);
        for(int d=0; d<256; ++d) {
            int x, y;
            hilbert_d2xy(16, d, &x, &y);
            hilbert_order[d] = y * 16 + x;
        }

        size_t total_orig = 0, total_raster = 0, total_morton = 0, total_hilbert = 0, total_ideal = 0;

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

            std::vector<uint64_t> hist_raster(256, 0), hist_morton(256, 0), hist_hilbert(256, 0), hist_ideal(256, 0);

            for (size_t r = 0; r < rows; r += 16) {
                for (size_t c = 0; c < cols; c += 16) {
                    std::vector<uint8_t> tile(256);
                    for (size_t tr = 0; tr < 16; ++tr) {
                        for (size_t tc = 0; tc < 16; ++tc) {
                            tile[tr * 16 + tc] = h_weights[(r + tr) * cols + (c + tc)];
                        }
                    }

                    // 1. Линейный Raster дельта-обход
                    for(size_t i=0; i<255; ++i) hist_raster[static_cast<uint8_t>(tile[i+1] - tile[i])]++;

                    // 2. Мортон (Z-кривая) дельта-обход
                    for(size_t i=0; i<255; ++i) {
                        uint8_t v1 = tile[morton_order[i]];
                        uint8_t v2 = tile[morton_order[i+1]];
                        hist_morton[static_cast<uint8_t>(v2 - v1)]++;
                    }

                    // 3. Гильберт дельта-обход
                    for(size_t i=0; i<255; ++i) {
                        uint8_t v1 = tile[hilbert_order[i]];
                        uint8_t v2 = tile[hilbert_order[i+1]];
                        hist_hilbert[static_cast<uint8_t>(v2 - v1)]++;
                    }

                    // 4. Идеальный отсортированный дельта-обход
                    std::sort(tile.begin(), tile.end());
                    for(size_t i=0; i<255; ++i) hist_ideal[static_cast<uint8_t>(tile[i+1] - tile[i])]++;
                }
            }

            size_t b_raster  = evaluate_delta_stream_bytes(hist_raster, num_tiles);
            size_t b_morton  = evaluate_delta_stream_bytes(hist_morton, num_tiles);
            size_t b_hilbert = evaluate_delta_stream_bytes(hist_hilbert, num_tiles);
            size_t b_ideal   = evaluate_delta_stream_bytes(hist_ideal, num_tiles);

            sweep_results.push_back({
                tensor_name, tensor_size, 
                static_cast<float>(tensor_size)/b_raster, static_cast<float>(tensor_size)/b_morton,
                static_cast<float>(tensor_size)/b_hilbert, static_cast<float>(tensor_size)/b_ideal
            });

            total_orig += tensor_size; total_raster += b_raster; total_morton += b_morton; total_hilbert += b_hilbert; total_ideal += b_ideal;
        }

        std::cout << "\n=================================== 2D GEOMETRY TRAVERSAL COMPARISON REPORT ===================================\n";
        std::cout << std::left << std::setw(38) << "Tensor Name" << std::setw(12) << "Orig(MB)" << std::setw(14) << "Raster Delta" << std::setw(14) << "Morton Delta" << std::setw(15) << "Hilbert Delta" << std::setw(14) << "Ideal Sorted" << "\n";
        std::cout << "----------------------------------------------------------------------------------------------------------------\n";
        for (const auto& r : sweep_results) {
            std::cout << std::left << std::setw(38) << (r.tensor_name.length() > 35 ? r.tensor_name.substr(0, 32) + "..." : r.tensor_name)
                      << std::setw(12) << std::fixed << std::setprecision(1) << (r.original_bytes / 1024.0 / 1024.0)
                      << std::setw(14) << (std::to_string(r.r_raster).substr(0,4) + "x")
                      << std::setw(14) << (std::to_string(r.r_morton).substr(0,4) + "x")
                      << std::setw(15) << (std::to_string(r.r_hilbert).substr(0,4) + "x")
                      << std::setw(14) << (std::to_string(r.r_ideal).substr(0,4) + "x") << "\n";
        }
        std::cout << "----------------------------------------------------------------------------------------------------------------\n";
        std::cout << std::left << std::setw(38) << "TOTAL MODEL SUMMARY" << std::setw(12) << std::fixed << std::setprecision(1) << (total_orig / 1024.0 / 1024.0)
                  << std::setw(14) << (std::to_string(static_cast<float>(total_orig)/total_raster).substr(0,4) + "x")
                  << std::setw(14) << (std::to_string(static_cast<float>(total_orig)/total_morton).substr(0,4) + "x")
                  << std::setw(15) << (std::to_string(static_cast<float>(total_orig)/total_hilbert).substr(0,4) + "x")
                  << std::setw(14) << (std::to_string(static_cast<float>(total_orig)/total_ideal).substr(0,4) + "x") << "\n";
        std::cout << "================================================================================================================\n\n";

    } catch (const std::exception& e) { std::cerr << "\n[CRITICAL ERROR] " << e.what() << "\n"; return 1; }
    return 0;
}