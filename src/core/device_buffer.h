#pragma once
// Move-only RAII owner of a cudaMalloc'd device buffer (Roadmap #3).
//
// Contracts:
//   - Ownership is exclusive: copying is deleted (a copy would double-free on
//     the GPU); moving transfers the allocation and leaves the source empty.
//   - A default-constructed buffer is empty (nullptr) -- the pattern for
//     model-conditional scratch (SSM / gated full-attention): declare empty,
//     allocate() inside the subsystem's composition branch.
//   - allocate()/zero() are INIT-tier (Hybrid error doctrine): they THROW
//     blackwell::cuda_error on failure. The RAII layout is what makes the
//     throwing Impl ctor leak-free -- already-constructed members unwind.
//   - NO implicit zero-fill (unlike CudaVector): decode scratch is written
//     before it is read every step, and the one consumer that needs zeroed
//     memory (the full-attention KV cache) calls zero() explicitly.
//   - Implicit conversion to T* (like CudaVector) keeps kernel-launch sites
//     and the white-box tests' telemetry probes untouched; get() exists for
//     contexts where the conversion is ambiguous or explicitness reads better.
#include <cstddef>

#include <cuda_runtime.h>

#include "common.h"  // CUDA_CHECK

namespace blackwell {

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) { allocate(count); }

    ~DeviceBuffer() { reset(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    // (Re)allocates `count` elements, releasing any previous allocation first.
    // count == 0 leaves the buffer empty.
    void allocate(size_t count) {
        reset();
        if (count == 0) return;
        CUDA_CHECK_THROW(cudaMalloc(&ptr_, count * sizeof(T)));
        count_ = count;
    }

    // Releases the allocation (idempotent). Deliberately ignores the cudaFree
    // result: reset() runs from destructors, where CUDA_CHECK's failure path
    // (today exit(), tomorrow throw) is unacceptable either way.
    void reset() noexcept {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
        count_ = 0;
    }

    void zero() {
        if (ptr_) CUDA_CHECK_THROW(cudaMemset(ptr_, 0, count_ * sizeof(T)));
    }

    T* get() const noexcept { return ptr_; }
    size_t count() const noexcept { return count_; }
    size_t size_bytes() const noexcept { return count_ * sizeof(T); }
    explicit operator bool() const noexcept { return ptr_ != nullptr; }

    // Неявное приведение к сырому указателю для передачи в CUDA-ядра
    // (та же эргономика, что у CudaVector в src/common.h).
    operator T*() const noexcept { return ptr_; }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

} // namespace blackwell
