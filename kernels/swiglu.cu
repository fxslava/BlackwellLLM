#include "swiglu.cuh"
#include <cuda_runtime.h>

#define SWIGLU_BLOCK_SIZE 256

__global__ void fused_swiglu_kernel(const float* __restrict__ gate,
                                    const float* __restrict__ up,
                                    float* __restrict__ output,
                                    size_t num_elements) 
{
    // Глобальный индекс потока
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Grid-stride цикл для надежной обработки любых объемов данных
    for (size_t i = idx; i < num_elements; i += gridDim.x * blockDim.x) {
        float g = gate[i];
        float u = up[i];

        // Быстрое вычисление: g / (1.0f + expf(-g))
        // expf - аппаратная инструкция SFU
        // __fdividef - быстрое аппаратное деление
        float sigmoid = __fdividef(1.0f, 1.0f + expf(-g));
        float swish = g * sigmoid;

        output[i] = swish * u;
    }
}

void launch_fused_swiglu_kernel(const float* d_gate, 
                                const float* d_up, 
                                float* d_output, 
                                size_t num_elements) 
{
    // Расчет необходимого количества блоков для покрытия всех элементов
    unsigned int blocks = (num_elements + SWIGLU_BLOCK_SIZE - 1) / SWIGLU_BLOCK_SIZE;
    
    // Ограничиваем максимальное количество блоков разумным пределом для сатурации SM
    if (blocks > 1024) blocks = 1024; 

    dim3 grid(blocks);
    dim3 threads(SWIGLU_BLOCK_SIZE);

    fused_swiglu_kernel<<<grid, threads>>>(d_gate, d_up, d_output, num_elements);
}