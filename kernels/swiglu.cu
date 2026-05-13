#include "swiglu.cuh"
#include <cuda_runtime.h>

#define SWIGLU_BLOCK_SIZE 256

// 🎯 Принудительное усечение мантиссы (Truncation) для паритета с тензорами PyTorch Bfloat16
__inline__ __device__ float cast_to_bf16_and_back(float val) {
    unsigned int bits = __float_as_uint(val);
    bits &= 0xFFFF0000;
    return __uint_as_float(bits);
}

__global__ void fused_swiglu_kernel(const float* __restrict__ gate,
                                    const float* __restrict__ up,
                                    float* __restrict__ output,
                                    size_t num_elements) 
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (size_t i = idx; i < num_elements; i += gridDim.x * blockDim.x) {
        // 🎯 Входные буферы gate и up в эталоне уже усечены до BF16
        float g = cast_to_bf16_and_back(gate[i]);
        float u = cast_to_bf16_and_back(up[i]);

        // Аппаратный расчет SiLU
        float sigmoid = __fdividef(1.0f, 1.0f + expf(-g));
        float swish = g * sigmoid;

        // 🎯 В PyTorch результат F.silu(gate) приводится к BF16 перед умножением на up
        float swish_bf16 = cast_to_bf16_and_back(swish);

        // 🎯 Итоговый выход слоя активации также обязан быть в сетке Bfloat16
        output[i] = cast_to_bf16_and_back(swish_bf16 * u);
    }
}

void launch_fused_swiglu_kernel(const float* d_gate, 
                                const float* d_up, 
                                float* d_output, 
                                size_t num_elements) 
{
    unsigned int blocks = (num_elements + SWIGLU_BLOCK_SIZE - 1) / SWIGLU_BLOCK_SIZE;
    if (blocks > 1024) blocks = 1024; 

    dim3 grid(blocks);
    dim3 threads(SWIGLU_BLOCK_SIZE);

    fused_swiglu_kernel<<<grid, threads>>>(d_gate, d_up, d_output, num_elements);
}