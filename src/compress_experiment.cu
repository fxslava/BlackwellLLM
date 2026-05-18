#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <numeric>
#include <cuda_runtime.h>
#include <nvcomp.hpp>
#include <nvcomp/lz4.hpp>
#include <nvcomp/gdeflate.hpp>

// Подключаем наш движок
#include "safetensors.h"
#include "memory_pool.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

struct TestResult {
    std::string algorithm;
    size_t chunk_size;
    size_t original_size;
    size_t compressed_size;
    float compression_ratio;
    float time_ms;
};

// Универсальная функция прогона компрессора (Только High-Level C++ API)
template <typename ManagerType>
TestResult run_nvcomp_test(const std::string& algo_name, size_t chunk_size, const uint8_t* d_uncompressed_data, size_t uncompressed_size) {
    // 1. Создаем менеджер компрессии (он сам внутри управляет делением на чанки)
    ManagerType manager{chunk_size};
    
    // 2. Получаем требования к памяти
    nvcomp::CompressionConfig config = manager.configure_compression(uncompressed_size);
    
    // 3. В High-Level API нам нужен только буфер для результата и указатель под размер
    uint8_t* d_comp_buffer;
    size_t* d_comp_size;
    
    CUDA_CHECK(cudaMalloc(&d_comp_buffer, config.max_compressed_buffer_size));
    CUDA_CHECK(cudaMalloc(&d_comp_size, sizeof(size_t)));
    
    // 4. Настраиваем замеры времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 5. Запускаем компрессию (используем дефолтный CUDA поток - 0)
    cudaEventRecord(start, 0);
    
    // 🎯 Метод compress вместо compress_async. 
    // Менеджер сам разберется с temp storage под капотом.
    manager.compress(
        d_uncompressed_data,
        d_comp_buffer,
        config,
        d_comp_size // Видеокарта сама запишет сюда финальный размер в байтах
    );
    
    cudaEventRecord(stop, 0);
    CUDA_CHECK(cudaStreamSynchronize(0));
    
    float time_ms = 0;
    cudaEventElapsedTime(&time_ms, start, stop);
    
    // 6. Читаем итоговый размер с видеокарты
    size_t total_compressed_size = 0;
    CUDA_CHECK(cudaMemcpy(&total_compressed_size, d_comp_size, sizeof(size_t), cudaMemcpyDeviceToHost));
    
    // 7. Уборка
    cudaFree(d_comp_buffer);
    cudaFree(d_comp_size);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return {
        algo_name,
        chunk_size,
        uncompressed_size,
        total_compressed_size,
        (float)uncompressed_size / total_compressed_size,
        time_ms
    };
}

void print_table(const std::vector<TestResult>& results) {
    std::cout << "\n=========================================================================================\n";
    std::cout << std::left 
              << std::setw(15) << "Algorithm" 
              << std::setw(15) << "Chunk Size" 
              << std::setw(15) << "Original(MB)" 
              << std::setw(15) << "Compr.(MB)" 
              << std::setw(15) << "Ratio" 
              << std::setw(15) << "Time(ms)" 
              << "\n";
    std::cout << "-----------------------------------------------------------------------------------------\n";
    
    for (const auto& r : results) {
        std::cout << std::left 
                  << std::setw(15) << r.algorithm
                  << std::setw(15) << (std::to_string(r.chunk_size / 1024) + " KB")
                  << std::setw(15) << std::fixed << std::setprecision(2) << (r.original_size / 1024.0 / 1024.0)
                  << std::setw(15) << (r.compressed_size / 1024.0 / 1024.0)
                  << std::setw(14) << (std::to_string(r.compression_ratio).substr(0,4) + "x")
                  << std::setw(15) << r.time_ms
                  << "\n";
    }
    std::cout << "=========================================================================================\n";
}

int main(int argc, char** argv) {
    std::cout << "🚀 Запуск стенда сжатия весов FP8 (nvCOMP)...\n";
    
    // Путь к модели Llama 3 (можно передать аргументом или захардкодить)
    std::string model_path = (argc > 1) ? argv[1] : "F:/AI/llama3-8b-fp8/model.safetensors.index.json";
    
    try {
        // 1. Инициализируем наш загрузчик и пул памяти
        std::cout << "📂 Чтение метаданных: " << model_path << "\n";
        SafetensorsLoader metadata_loader(model_path);
        
        std::cout << "🧠 Аллокация VRAMArena и загрузка весов (DirectStorage)...\n";
        VRAMArena arena(model_path, metadata_loader, 2048);
        
        // 2. Ищем самый "жирный" тензор, обычно это down_proj в одном из слоев
        std::string target_tensor = "model.layers.0.mlp.down_proj.weight";
        
        if (!metadata_loader.has_tensor(target_tensor)) {
            // Фолбэк, если слой называется иначе
            auto all_tensors = metadata_loader.list_tensors();
            for (const auto& t : all_tensors) {
                if (t.find("down_proj.weight") != std::string::npos) {
                    target_tensor = t;
                    break;
                }
            }
        }
        
        const void* d_weight_ptr = arena.get_weight_ptr_optional(target_tensor);
        if (!d_weight_ptr) {
            throw std::runtime_error("Тензор не найден в VRAMArena!");
        }
        
        size_t tensor_size = metadata_loader.get_tensor(target_tensor).byte_size;
        std::cout << "✅ Тензор для эксперимента: " << target_tensor << " (" << tensor_size / 1024 / 1024 << " MB)\n";

        // 3. Подготовка CUDA потока
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        // 4. Запускаем эксперименты
        std::vector<TestResult> results;
        std::vector<size_t> chunk_sizes = {16384, 65536, 262144, 1048576, 4194304}; // 16KB, 64KB, 256KB, 1MB, 4MB

        for (size_t chunk : chunk_sizes) {
            // Тест LZ4
            results.push_back(run_nvcomp_test<nvcomp::LZ4Manager>(
                "LZ4", chunk, static_cast<const uint8_t*>(d_weight_ptr), tensor_size
            ));
            
            // Тест Gdeflate
            results.push_back(run_nvcomp_test<nvcomp::GdeflateManager>(
                "Gdeflate", chunk, static_cast<const uint8_t*>(d_weight_ptr), tensor_size
            ));
        }

        // 5. Вывод красивой таблицы
        print_table(results);

        CUDA_CHECK(cudaStreamDestroy(stream));
        
    } catch (const std::exception& e) {
        std::cerr << "\n[CRITICAL ERROR] " << e.what() << "\n";
        return 1;
    }

    return 0;
}