#include "sampling.cuh"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <stdexcept>
#include <string>

#define ARGMAX_BLOCK_SIZE 512

// ============================================================================
// 1. ОРИГИНАЛЬНЫЙ GPU ARGMAX (Для Temperature = 0.0)
// ============================================================================
__inline__ __device__ void warp_reduce_argmax(float& max_val, int& max_idx) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_val = __shfl_down_sync(0xFFFFFFFF, max_val, offset);
        int other_idx   = __shfl_down_sync(0xFFFFFFFF, max_idx, offset);
        if (other_val > max_val) {
            max_val = other_val;
            max_idx = other_idx;
        }
    }
}

__global__ void argmax_kernel(const float* __restrict__ logits, 
                              int* __restrict__ out_token_id, 
                              size_t vocab_size) 
{
    int tid = threadIdx.x;
    float thread_max_val = -CUDART_INF_F;
    int thread_max_idx   = 0;

    for (size_t i = tid; i < vocab_size; i += blockDim.x) {
        float val = logits[i];
        if (val > thread_max_val) {
            thread_max_val = val;
            thread_max_idx = static_cast<int>(i);
        }
    }

    __shared__ float s_max_vals[ARGMAX_BLOCK_SIZE / 32];
    __shared__ int   s_max_idxs[ARGMAX_BLOCK_SIZE / 32];

    int warp_id = tid >> 5;
    int lane_id = tid & 31;

    warp_reduce_argmax(thread_max_val, thread_max_idx);

    if (lane_id == 0) {
        s_max_vals[warp_id] = thread_max_val;
        s_max_idxs[warp_id] = thread_max_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        float warp_max_val = (lane_id < (ARGMAX_BLOCK_SIZE / 32)) ? s_max_vals[lane_id] : -CUDART_INF_F;
        int   warp_max_idx = (lane_id < (ARGMAX_BLOCK_SIZE / 32)) ? s_max_idxs[lane_id] : 0;

        warp_reduce_argmax(warp_max_val, warp_max_idx);

        if (tid == 0) {
            *out_token_id = warp_max_idx;
        }
    }
}

void launch_argmax_kernel(const float* d_logits, int* d_out_token_id, size_t vocab_size) {
    argmax_kernel<<<1, ARGMAX_BLOCK_SIZE>>>(d_logits, d_out_token_id, vocab_size);
}

// ============================================================================
// 1b. REPETITION PENALTY (in-place logit shaping, applied BEFORE sampling)
// ============================================================================
// Contract + rationale: sampling.cuh. One thread per entry of `ids`; only the
// FIRST occurrence of a given id acts, which makes the pass exactly-once per
// DISTINCT id and removes any read-modify-write race on a shared logit slot
// (two threads holding the same id would otherwise compound the penalty
// non-deterministically). `ids` is a small rolling window (tens of entries), so
// the O(n^2) first-occurrence scan is a few thousand comparisons -- far cheaper
// than a device-side dedup pass or an atomic protocol.
__global__ void repetition_penalty_kernel(float* __restrict__ logits,
                                          size_t vocab_size,
                                          const int* __restrict__ ids,
                                          int num_tokens,
                                          float penalty)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_tokens) return;

    const int id = ids[i];
    if (id < 0 || static_cast<size_t>(id) >= vocab_size) return;
    for (int j = 0; j < i; ++j) {
        if (ids[j] == id) return;   // an earlier thread owns this id
    }

    const float v = logits[id];
    // Sign-aware: dividing a NEGATIVE logit would make it larger (less penalized),
    // so the negative branch multiplies instead. This is the CTRL formulation.
    logits[id] = (v > 0.0f) ? (v / penalty) : (v * penalty);
}

void launch_repetition_penalty_kernel(float* d_logits, size_t vocab_size,
                                      const int* d_token_ids, int num_tokens,
                                      float penalty)
{
    if (d_logits == nullptr || d_token_ids == nullptr || num_tokens <= 0 ||
        penalty <= 1.0f) {
        return;   // documented no-op (penalty disabled / empty history)
    }
    constexpr int kBlock = 128;
    const int blocks = (num_tokens + kBlock - 1) / kBlock;
    repetition_penalty_kernel<<<blocks, kBlock>>>(d_logits, vocab_size, d_token_ids,
                                                  num_tokens, penalty);
}

// ============================================================================
// 2. НОВОЕ CPU СЭМПЛИРОВАНИЕ (Temperature + Top-P)
// ============================================================================
struct ProbIndex {
    float prob;
    int index;
};

int sample_top_p(const float* d_logits, size_t vocab_size, float temperature, float top_p) {
    // Если температура 0 (или близка к ней), используем жадный поиск на GPU
    if (temperature < 1e-5f) {
        int h_out = 0;
        int* d_out = nullptr;
        // An unchecked failure here would leave d_out dangling: the kernel launch,
        // the copy-back and the cudaFree would all operate on a garbage pointer.
        if (cudaMalloc(&d_out, sizeof(int)) != cudaSuccess) {
            throw std::runtime_error("sample_top_p: cudaMalloc failed for argmax output slot");
        }
        launch_argmax_kernel(d_logits, d_out, vocab_size);
        cudaError_t copy_err = cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_out);
        if (copy_err != cudaSuccess) {
            throw std::runtime_error("sample_top_p: failed to copy argmax result to host");
        }
        return h_out;
    }

    // 1. Копируем логиты с видеокарты в RAM
    std::vector<float> h_logits(vocab_size);
    cudaMemcpy(h_logits.data(), d_logits, vocab_size * sizeof(float), cudaMemcpyDeviceToHost);

    // 2. Применяем Температуру и находим максимум (для стабильности Softmax)
    float max_val = -INFINITY;
    for (size_t i = 0; i < vocab_size; i++) {
        h_logits[i] /= temperature;
        if (h_logits[i] > max_val) max_val = h_logits[i];
    }

    // 3. Вычисляем вероятности (Softmax)
    float sum = 0.0f;
    std::vector<ProbIndex> probs(vocab_size);
    for (size_t i = 0; i < vocab_size; i++) {
        // Вычитаем max_val, чтобы экспонента не улетела в бесконечность
        float p = std::exp(h_logits[i] - max_val); 
        probs[i] = {p, static_cast<int>(i)};
        sum += p;
    }

    // Нормализуем, чтобы сумма всех вероятностей равнялась 1.0
    for (size_t i = 0; i < vocab_size; i++) {
        probs[i].prob /= sum;
    }

    // 4. Сортируем токены по убыванию вероятности (для Top-P)
    std::sort(probs.begin(), probs.end(), [](const ProbIndex& a, const ProbIndex& b) {
        return a.prob > b.prob;
    });

    // 5. Отрезаем "длинный хвост" бредовых токенов (Top-P Nucleus)
    float cumsum = 0.0f;
    int last_idx = 0;
    for (size_t i = 0; i < vocab_size; i++) {
        cumsum += probs[i].prob;
        last_idx = static_cast<int>(i);
        if (cumsum >= top_p) {
            break;
        }
    }

    // 6. Бросаем случайную кость (сэмплируем) в рамках оставшегося "ядра"
    static std::random_device rd;
    static std::mt19937 gen(rd()); 
    std::uniform_real_distribution<float> dis(0.0f, cumsum);
    float r = dis(gen);

    float current_sum = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        current_sum += probs[i].prob;
        if (r <= current_sum) {
            return probs[i].index; // Возвращаем выбранный токен!
        }
    }
    
    // Fallback
    return probs[last_idx].index;
}

// Вычисляет логарифм вероятности конкретного токена после Softmax
float compute_log_prob(const float* d_logits, size_t vocab_size, int target_token_id) {
    // Guard the host-side h_logits[target_token_id] read below: an out-of-range
    // token id would index past the end of the vector (undefined behavior).
    if (target_token_id < 0 || static_cast<size_t>(target_token_id) >= vocab_size) {
        throw std::out_of_range("compute_log_prob: target_token_id " +
                                std::to_string(target_token_id) +
                                " outside vocab_size " + std::to_string(vocab_size));
    }

    // 1. Копируем логиты в оперативную память
    std::vector<float> h_logits(vocab_size);
    cudaMemcpy(h_logits.data(), d_logits, vocab_size * sizeof(float), cudaMemcpyDeviceToHost);

    // 2. Ищем максимум (нужно для математической стабильности экспоненты)
    float max_val = -INFINITY;
    for (size_t i = 0; i < vocab_size; i++) {
        if (h_logits[i] > max_val) max_val = h_logits[i];
    }

    // 3. Считаем сумму экспонент (знаменатель Softmax)
    float sum_exp = 0.0f;
    for (size_t i = 0; i < vocab_size; i++) {
        sum_exp += std::exp(h_logits[i] - max_val);
    }

    // 4. log( P(x) ) = log( exp(logit - max) / sum_exp ) = (logit - max) - log(sum_exp)
    float log_prob = (h_logits[target_token_id] - max_val) - std::log(sum_exp);
    
    return log_prob;
}