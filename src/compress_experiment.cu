#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <cuda_runtime.h>

#include "safetensors.h"
#include "memory_pool.h"
#include "cpu_models.h"

void export_matrix_to_excel_grid(const void* d_weight_ptr, size_t rows, size_t cols, const std::string& csv_filename) {
    std::cout << "📂 Exporting 2D weight matrix to Excel grid format...\n";
    std::cout << "📐 Matrix dimensions: " << rows << " rows x " << cols << " columns\n";

    if (cols > 16384) {
        std::cout << "⚠️ WARNING: Column count (" << cols << ") exceeds Excel's maximum limit of 16,384!\n";
        std::cout << "   Excel will truncate columns beyond this limit when opening the file.\n";
    }

    // 1. Скачиваем веса из VRAM на Хост
    size_t total_elements = rows * cols;
    std::vector<uint8_t> h_weights(total_elements);
    cudaError_t err = cudaMemcpy(h_weights.data(), d_weight_ptr, total_elements, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA Memcpy failed: " + std::string(cudaGetErrorString(err)));
    }

    // 2. Открываем файл на запись
    std::ofstream csv_file(csv_filename);
    if (!csv_file.is_open()) {
        throw std::runtime_error("Failed to create file: " + csv_filename);
    }

    // Выставляем точность для красивого вывода чисел с плавающей точкой
    csv_file << std::fixed << std::setprecision(5);

    // 3. Записываем сетку данных
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            // Извлекаем сырой байт и распаковываем его в честный float
            uint8_t raw_byte = h_weights[r * cols + c];
            float fp_value = cpu_unpack_fp8_e4m3(raw_byte);

            csv_file << fp_value;
            
            // Ставим разделитель-запятую везде, кроме последнего элемента в строке
            if (c < cols - 1) {
                csv_file << ",";
            }
        }
        // Перенос строки для Excel — переход к следующему ряду матрицы
        csv_file << "\n";
    }

    csv_file.close();
    std::cout << "✅ Matrix successfully saved as 2D grid to: " << csv_filename << "\n";
}

int main(int argc, char** argv) {
    std::cout << "🔬 Launching Blackwell Matrix Visualizer...\n";
    std::string model_path = (argc > 1) ? argv[1] : "F:/AI/llama3-8b-fp8/model.safetensors.index.json";
    
    try {
        SafetensorsLoader metadata_loader(model_path);
        VRAMArena arena(model_path, metadata_loader, 2048);
        
        // Выберем для визуализации down_proj слой, он идеально демонстрирует структуру калибровки
        std::string target_tensor = "model.layers.0.mlp.down_proj.weight";
        
        if (!metadata_loader.has_tensor(target_tensor)) {
            for (const auto& t : metadata_loader.list_tensors()) {
                if (t.find("down_proj.weight") != std::string::npos) { target_tensor = t; break; }
            }
        }
        
        const void* d_weight_ptr = arena.get_weight_ptr_optional(target_tensor);
        if (!d_weight_ptr) throw std::runtime_error("Tensor not found in VRAMArena!");
        
        auto meta = metadata_loader.get_tensor(target_tensor);
        
        // Передаем указатель, количество строк (shape[0]) и столбцов (shape[1])
        export_matrix_to_excel_grid(d_weight_ptr, meta.shape[0], meta.shape[1], "matrix_surface.csv");
        
    } catch (const std::exception& e) {
        std::cerr << "\n[CRITICAL ERROR] " << e.what() << "\n";
        return 1;
    }
    return 0;
}