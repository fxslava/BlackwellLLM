#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace blackwell {
// CUDA failure in an INIT-tier path (Hybrid error doctrine, Roadmap #2).
// Carries the raw cudaError_t so the DLL boundary can distinguish VRAM
// exhaustion (-> E_OUTOFMEMORY) from other runtime faults.
class cuda_error : public std::runtime_error {
public:
    cuda_error(cudaError_t code, const std::string& what_arg)
        : std::runtime_error(what_arg), code_(code) {}
    cudaError_t code() const noexcept { return code_; }

private:
    cudaError_t code_;
};
} // namespace blackwell

// INIT-tier check: throws blackwell::cuda_error. For constructors / factories /
// setup paths ONLY -- RAII members make a throwing ctor leak-free, and the DLL
// boundary maps the exception to an HRESULT. NEVER use in destructors (nothing
// may throw there) or in the decode hot loop (use CUDA_CHECK_RETURN, defined
// next to the runtime methods in src/core/engine.cpp).
#define CUDA_CHECK_THROW(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            throw blackwell::cuda_error(err, std::string("CUDA error: ") + \
                cudaGetErrorString(err) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        } \
    } while (0)

// LEGACY check: kills the process. Remaining users (memory_pool, paging/, ssm/,
// kernels, tests' CudaVector) migrate to _THROW (init) or _RETURN (runtime)
// opportunistically when touched -- do not add new call sites in engine code
// (Roadmap #2/#3 in CLAUDE.md).
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