#include "swiglu.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define SWIGLU_BLOCK_SIZE 256

// 🎯 Принудительное усечение мантиссы (Truncation) для паритета с тензорами PyTorch Bfloat16
__inline__ __device__ float cast_to_bf16_and_back(float val) {
    return __bfloat162float(__float2bfloat16(val));
    /*unsigned int bits = __float_as_uint(val);
    bits &= 0xFFFF0000;
    return __uint_as_float(bits);*/
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

// Fused gate|up rows: element i of token t reads gate at [t, i] and up at
// [t, intermediate_dim + i] of a [num_tokens, 2*intermediate_dim] buffer. The
// arithmetic (and every bf16 latch) is the same as fused_swiglu_kernel -- only the
// addressing differs.
__global__ void fused_swiglu_gate_up_kernel(const float* __restrict__ gate_up,
                                            float* __restrict__ output,
                                            size_t num_tokens,
                                            size_t inter_dim)
{
    const size_t total = num_tokens * inter_dim;
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = idx; i < total; i += gridDim.x * blockDim.x) {
        const size_t token = i / inter_dim;
        const size_t col   = i - token * inter_dim;
        const float* row   = gate_up + token * 2 * inter_dim;

        float g = cast_to_bf16_and_back(row[col]);
        float u = cast_to_bf16_and_back(row[inter_dim + col]);

        float sigmoid = __fdividef(1.0f, 1.0f + expf(-g));
        float swish = g * sigmoid;
        float swish_bf16 = cast_to_bf16_and_back(swish);

        output[i] = cast_to_bf16_and_back(swish_bf16 * u);
    }
}

void launch_fused_swiglu_gate_up_kernel(const float* d_gate_up,
                                        float* d_output,
                                        size_t num_tokens,
                                        size_t intermediate_dim)
{
    const size_t total = num_tokens * intermediate_dim;
    unsigned int blocks = static_cast<unsigned int>((total + SWIGLU_BLOCK_SIZE - 1) / SWIGLU_BLOCK_SIZE);
    if (blocks > 1024) blocks = 1024;
    if (blocks == 0) return;

    fused_swiglu_gate_up_kernel<<<dim3(blocks), dim3(SWIGLU_BLOCK_SIZE)>>>(
        d_gate_up, d_output, num_tokens, intermediate_dim);
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