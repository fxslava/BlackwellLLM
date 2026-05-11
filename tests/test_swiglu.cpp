#include <gtest/gtest.h>
#include <vector>
#include <cuda_runtime.h>
#include "common.h"
#include "swiglu.cuh"
#include "cpu_models.h"

class SwigluTests : public ::testing::Test {
protected:
    void TearDown() override {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "\n[ОШИБКА CUDA]: " << cudaGetErrorString(err) << "\n";
        }
    }
};

TEST_F(SwigluTests, ActivationCorrectness) {
    const size_t num_elements = 14336; 

    std::vector<float> h_gate(num_elements);
    std::vector<float> h_up(num_elements);
    std::vector<float> h_cpu_output(num_elements, 0.0f);

    for (size_t i = 0; i < num_elements; ++i) {
        h_gate[i] = static_cast<float>(i % 60) * 0.1f - 3.0f;
        h_up[i]   = static_cast<float>(i % 40) * 0.1f - 2.0f;
    }

    cpu_fused_swiglu(h_gate.data(), h_up.data(), h_cpu_output.data(), num_elements);

    CudaVector<float> d_gate(num_elements);   d_gate.upload(h_gate);
    CudaVector<float> d_up(num_elements);     d_up.upload(h_up);
    CudaVector<float> d_output(num_elements);

    launch_fused_swiglu_kernel(d_gate, d_up, d_output, num_elements);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_gpu_output(num_elements);
    d_output.download(h_gpu_output);

    for (size_t i = 0; i < num_elements; ++i) {
        ASSERT_NEAR(h_cpu_output[i], h_gpu_output[i], 1e-4f) 
            << "Расхождение в SwiGLU по абсолютному индексу " << i;
    }
}