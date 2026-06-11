#include "nvcomp_baseline.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <numeric>
#include <cuda_runtime.h>
#include <nvcomp.hpp>
#include <nvcomp/gdeflate.hpp>
#include <nvcomp/snappy.hpp>
#include <nvcomp/cascaded.hpp>
#include <nvcomp/bitcomp.hpp>

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

template <typename ManagerType>
TestResult run_single_nvcomp_test(const std::string& algo_name, ManagerType& manager, size_t chunk_size, const uint8_t* d_uncompressed_data, size_t uncompressed_size) {
    nvcomp::CompressionConfig config = manager.configure_compression(uncompressed_size);
    
    uint8_t* d_comp_buffer;
    size_t* d_comp_size;
    
    CUDA_CHECK(cudaMalloc(&d_comp_buffer, config.max_compressed_buffer_size));
    CUDA_CHECK(cudaMalloc(&d_comp_size, sizeof(size_t)));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, 0);
    manager.compress(d_uncompressed_data, d_comp_buffer, config, d_comp_size);
    cudaEventRecord(stop, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float time_ms = 0;
    cudaEventElapsedTime(&time_ms, start, stop);
    
    size_t total_compressed_size = 0;
    CUDA_CHECK(cudaMemcpy(&total_compressed_size, d_comp_size, sizeof(size_t), cudaMemcpyDeviceToHost));
    
    cudaFree(d_comp_buffer);
    cudaFree(d_comp_size);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return { algo_name, chunk_size, uncompressed_size, total_compressed_size, (float)uncompressed_size / total_compressed_size, time_ms };
}

void run_nvcomp_baseline(const void* d_weight_ptr, size_t tensor_size) {
    std::vector<TestResult> results;
    std::vector<size_t> chunk_sizes = {65536, 1048576}; 

    for (size_t chunk : chunk_sizes) {
        nvcomp::GdeflateManager gdef(chunk);
        results.push_back(run_single_nvcomp_test("Gdeflate", gdef, chunk, static_cast<const uint8_t*>(d_weight_ptr), tensor_size));
        
        nvcomp::SnappyManager snappy(chunk);
        results.push_back(run_single_nvcomp_test("Snappy", snappy, chunk, static_cast<const uint8_t*>(d_weight_ptr), tensor_size));

        nvcompBatchedCascadedCompressOpts_t cascaded_opts = {0};
        cascaded_opts.type = NVCOMP_TYPE_UCHAR;
        cascaded_opts.num_RLEs = 0;
        cascaded_opts.num_deltas = 1;
        cascaded_opts.use_bp = 1;
        nvcomp::CascadedManager cascaded(chunk, cascaded_opts);
        results.push_back(run_single_nvcomp_test("Cascaded", cascaded, chunk, static_cast<const uint8_t*>(d_weight_ptr), tensor_size));

        nvcompBatchedBitcompCompressOpts_t bitcomp_opts = {0};
        bitcomp_opts.data_type = NVCOMP_TYPE_UCHAR;
        bitcomp_opts.algorithm = 0; 
        nvcomp::BitcompManager bitcomp(chunk, bitcomp_opts);
        results.push_back(run_single_nvcomp_test("Bitcomp", bitcomp, chunk, static_cast<const uint8_t*>(d_weight_ptr), tensor_size));
    }

    std::cout << "\n==================== NVCOMP BASELINE BENCHMARKS =========================================\n";
    std::cout << std::left << std::setw(15) << "Algorithm" << std::setw(15) << "Chunk Size" << std::setw(15) << "Original(MB)" << std::setw(15) << "Compr.(MB)" << std::setw(15) << "Ratio" << std::setw(15) << "Time(ms)" << "\n";
    std::cout << "-----------------------------------------------------------------------------------------\n";
    for (const auto& r : results) {
        std::cout << std::left << std::setw(15) << r.algorithm << std::setw(15) << (std::to_string(r.chunk_size / 1024) + " KB") << std::setw(15) << std::fixed << std::setprecision(2) << (r.original_size / 1024.0 / 1024.0) << std::setw(15) << (r.compressed_size / 1024.0 / 1024.0) << std::setw(14) << (std::to_string(r.compression_ratio).substr(0,4) + "x") << std::setw(15) << r.time_ms << "\n";
    }
    std::cout << "=========================================================================================\n";
}