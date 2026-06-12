#include "embedding.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// Перегрузки распаковки половинных типов в FP32 для шаблонных ядер
__device__ __forceinline__ float weight_to_fp32(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float weight_to_fp32(__half v)        { return __half2float(v); }

__global__ void embedding_lookup_kernel(const int* tokens, 
                                        const float* embed_table, 
                                        float* output, 
                                        size_t seq_len, 
                                        size_t hidden_dim) {
    // blockIdx.x отвечает за индекс токена в запросе (seq_len)
    size_t seq_idx = blockIdx.x;
    if (seq_idx >= seq_len) return;

    int token_id = tokens[seq_idx];
    const float* src_row = embed_table + (size_t)token_id * hidden_dim;
    float* dst_row = output + seq_idx * hidden_dim;

    // threadIdx.x итерируется по скрытой размерности (hidden_dim)
    // Grid-stride цикл на случай, если hidden_dim больше количества потоков в блоке
    for (size_t h = threadIdx.x; h < hidden_dim; h += blockDim.x) {
        dst_row[h] = src_row[h];
    }
}

void launch_embedding_kernel(const int* d_tokens, 
                             const float* d_embed_table, 
                             float* d_output, 
                             size_t seq_len, 
                             size_t hidden_dim) {
    // Запускаем сетку: количество блоков = количеству токенов
    dim3 blocks(seq_len);
    
    // 512 потоков на блок — оптимально для сатурации кэша L1/L2 на Blackwell
    unsigned int num_threads = (hidden_dim < 512) ? static_cast<unsigned int>(hidden_dim) : 512;
    dim3 threads(num_threads);

    embedding_lookup_kernel<<<blocks, threads>>>(d_tokens, d_embed_table, d_output, seq_len, hidden_dim);
}

template <typename WT>
__global__ void half_typed_embedding_kernel(
    const int* __restrict__ tokens,
    const WT* __restrict__ embed_table,
    float* __restrict__ output,
    size_t seq_len,
    size_t hidden_dim)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = seq_len * hidden_dim;

    if (idx >= total_elements) return;

    // Определяем, какой токен и какую размерность обрабатывает поток
    size_t token_idx = idx / hidden_dim;
    size_t dim_idx   = idx % hidden_dim;

    int token_id = tokens[token_idx];

    // Коалесцированное чтение 2-байтового BF16/FP16
    WT half_val = embed_table[token_id * hidden_dim + dim_idx];

    // Аппаратная распаковка в 4-байтовый FP32 и запись в буфер активаций
    output[idx] = weight_to_fp32(half_val);
}

template <typename WT>
static void launch_half_typed_embedding_kernel(
    const int* d_tokens,
    const void* d_embed_table,
    float* d_output,
    size_t seq_len,
    size_t hidden_dim)
{
    size_t total_elements = seq_len * hidden_dim;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    half_typed_embedding_kernel<WT><<<blocks, threads>>>(
        d_tokens,
        reinterpret_cast<const WT*>(d_embed_table),
        d_output,
        seq_len,
        hidden_dim
    );
}

void launch_bf16_embedding_kernel(
    const int* d_tokens,
    const void* d_embed_table,
    float* d_output,
    size_t seq_len,
    size_t hidden_dim)
{
    launch_half_typed_embedding_kernel<__nv_bfloat16>(
        d_tokens, d_embed_table, d_output, seq_len, hidden_dim);
}

// FP16 вариант: AWQ/GPTQ чекпойнты хранят embed_tokens в half
void launch_fp16_embedding_kernel(
    const int* d_tokens,
    const void* d_embed_table,
    float* d_output,
    size_t seq_len,
    size_t hidden_dim)
{
    launch_half_typed_embedding_kernel<__half>(
        d_tokens, d_embed_table, d_output, seq_len, hidden_dim);
}