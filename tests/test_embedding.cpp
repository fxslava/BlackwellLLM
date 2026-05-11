#include <gtest/gtest.h>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "common.h"
#include "embedding.cuh"
#include "cpu_models.h"

#ifdef _WIN32
#include <windows.h>
static struct ConsoleInitializer {
    ConsoleInitializer() {
        SetConsoleOutputCP(CP_UTF8);
        SetConsoleCP(CP_UTF8);
    }
} console_init;
#endif

class KernelTests : public ::testing::Test {
protected:
    void TearDown() override {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "\n[КРИТИЧЕСКАЯ ОШИБКА CUDA в TearDown]: " << cudaGetErrorString(err) << "\n";
        }
    }
};

TEST_F(KernelTests, EmbeddingLookupCorrectness) {
    const size_t vocab_size = 32000; 
    const size_t hidden_dim = 4096;  
    const size_t seq_len = 8;        

    std::vector<float> h_embed_table(vocab_size * hidden_dim);
    for (size_t i = 0; i < h_embed_table.size(); ++i) {
        h_embed_table[i] = static_cast<float>(i % 100) * 0.01f;
    }

    std::vector<int> h_tokens = {12, 450, 0, 31999, 5, 88, 42, 10};
    std::vector<float> h_cpu_output(seq_len * hidden_dim, 0.0f);

    cpu_embedding_lookup(h_tokens.data(), h_embed_table.data(), h_cpu_output.data(), seq_len, hidden_dim);

    CudaVector<float> d_embed_table(h_embed_table.size()); d_embed_table.upload(h_embed_table);
    CudaVector<int>   d_tokens(h_tokens.size());           d_tokens.upload(h_tokens);
    CudaVector<float> d_gpu_output(h_cpu_output.size());

    launch_embedding_kernel(d_tokens, d_embed_table, d_gpu_output, seq_len, hidden_dim);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_gpu_output(h_cpu_output.size());
    d_gpu_output.download(h_gpu_output);

    for (size_t i = 0; i < h_gpu_output.size(); ++i) {
        ASSERT_NEAR(h_cpu_output[i], h_gpu_output[i], 1e-5f) << "Ошибка по абсолютному индексу " << i;
    }
}

TEST_F(KernelTests, Bf16EmbeddingLookupCorrectness) {
    const size_t vocab_size = 32000; 
    const size_t hidden_dim = 4096;  
    const size_t seq_len = 8;        

    std::vector<__nv_bfloat16> h_bf16_embed_table(vocab_size * hidden_dim);
    std::vector<float> h_float_embed_table(vocab_size * hidden_dim);

    for (size_t i = 0; i < h_bf16_embed_table.size(); ++i) {
        float orig_val = static_cast<float>(i % 100) * 0.01f;
        h_bf16_embed_table[i] = __float2bfloat16(orig_val);
        h_float_embed_table[i] = __bfloat162float(h_bf16_embed_table[i]);
    }

    std::vector<int> h_tokens = {12, 450, 0, 31999, 5, 88, 42, 10};
    std::vector<float> h_cpu_output(seq_len * hidden_dim, 0.0f);

    cpu_embedding_lookup(h_tokens.data(), h_float_embed_table.data(), h_cpu_output.data(), seq_len, hidden_dim);

    CudaVector<__nv_bfloat16> d_bf16_embed_table(h_bf16_embed_table.size()); d_bf16_embed_table.upload(h_bf16_embed_table);
    CudaVector<int>           d_tokens(h_tokens.size());                     d_tokens.upload(h_tokens);
    CudaVector<float>         d_gpu_output(h_cpu_output.size());

    launch_bf16_embedding_kernel(d_tokens, d_bf16_embed_table, d_gpu_output, seq_len, hidden_dim);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_gpu_output(h_cpu_output.size());
    d_gpu_output.download(h_gpu_output);

    for (size_t i = 0; i < h_gpu_output.size(); ++i) {
        ASSERT_NEAR(h_cpu_output[i], h_gpu_output[i], 1e-5f) << "Ошибка в BF16 распаковке по индексу " << i;
    }
}

TEST_F(KernelTests, Bf16EmbeddingRealisticStressTest) {
    const size_t vocab_size = 128256; 
    const size_t hidden_dim = 4096;  
    const size_t seq_len = 8;        

    std::cout << "[Стресс-тест] Подготовка паттернов 1.05 ГБ на Host...\n";
    std::vector<__nv_bfloat16> h_bf16_embed_table(vocab_size * hidden_dim, __float2bfloat16(0.0f));
    std::vector<int> h_tokens = {0, 42, 1024, 50000, 84042, 100500, 128000, 128255};
    std::vector<float> h_cpu_output(seq_len * hidden_dim, 0.0f);

    for (size_t s = 0; s < seq_len; ++s) {
        int token_id = h_tokens[s];
        size_t row_base = static_cast<size_t>(token_id) * hidden_dim;
        for (size_t h = 0; h < hidden_dim; ++h) {
            float orig_val = static_cast<float>(token_id % 100) + (static_cast<float>(h % 50) * 0.01f);
            __nv_bfloat16 bf16_val = __float2bfloat16(orig_val);
            h_bf16_embed_table[row_base + h] = bf16_val;
            h_cpu_output[s * hidden_dim + h] = __bfloat162float(bf16_val);
        }
    }

    std::cout << "[Стресс-тест] Аллокация и трансфер 1.05 ГБ в VRAM...\n";
    CudaVector<__nv_bfloat16> d_bf16_embed_table(h_bf16_embed_table.size()); d_bf16_embed_table.upload(h_bf16_embed_table);
    CudaVector<int>           d_tokens(h_tokens.size());                     d_tokens.upload(h_tokens);
    CudaVector<float>         d_gpu_output(h_cpu_output.size());

    launch_bf16_embedding_kernel(d_tokens, d_bf16_embed_table, d_gpu_output, seq_len, hidden_dim);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_gpu_output(h_cpu_output.size());
    d_gpu_output.download(h_gpu_output);

    std::cout << "[Стресс-тест] Валидация 64-битных смещений...\n";
    for (size_t s = 0; s < seq_len; ++s) {
        int token_id = h_tokens[s];
        for (size_t h = 0; h < hidden_dim; ++h) {
            size_t idx = s * hidden_dim + h;
            ASSERT_NEAR(h_cpu_output[idx], h_gpu_output[idx], 1e-5f) 
                << "Сбой адресации на токене " << token_id << " (индекс " << h << ")";
        }
    }
    std::cout << "[Стресс-тест] Успешно!\n";
}