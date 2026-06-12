#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <cstddef>
#include <stdexcept>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// RAII Обёртка для автоматического управления памятью GPU в тестах
template <typename T>
class CudaVector {
public:
    T* d_ptr = nullptr;
    size_t num_elements = 0;

    explicit CudaVector(size_t size) : num_elements(size) {
        CUDA_CHECK(cudaMalloc(&d_ptr, num_elements * sizeof(T)));
        CUDA_CHECK(cudaMemset(d_ptr, 0, num_elements * sizeof(T)));
    }

    ~CudaVector() {
        if (d_ptr) {
            cudaFree(d_ptr);
            d_ptr = nullptr;
        }
    }

    // Запрещаем копирование, чтобы не получить двойной cudaFree
    CudaVector(const CudaVector&) = delete;
    CudaVector& operator=(const CudaVector&) = delete;

    // Быстрая загрузка из std::vector (Host -> Device)
    void upload(const std::vector<T>& h_vec) {
        if (h_vec.size() != num_elements) throw std::runtime_error("Size mismatch on upload");
        CUDA_CHECK(cudaMemcpy(d_ptr, h_vec.data(), num_elements * sizeof(T), cudaMemcpyHostToDevice));
    }

    // Быстрая выгрузка в std::vector (Device -> Host)
    void download(std::vector<T>& h_vec) const {
        if (h_vec.size() != num_elements) h_vec.resize(num_elements);
        CUDA_CHECK(cudaMemcpy(h_vec.data(), d_ptr, num_elements * sizeof(T), cudaMemcpyDeviceToHost));
    }

    // Неявное приведение к сырому указателю для передачи в CUDA-ядра
    operator T*() { return d_ptr; }
    operator const T*() const { return d_ptr; }
};