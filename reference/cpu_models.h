#pragma once
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>

// Таблица эмбеддингов — это матрица [vocab_size, hidden_dim].
// Мы берем ID токена и копируем соответствующую строку в выходной буфер.
inline void cpu_embedding_lookup(const int* tokens, 
                                 const float* embed_table, 
                                 float* output, 
                                 size_t seq_len, 
                                 size_t hidden_dim) {
    for (size_t s = 0; s < seq_len; ++s) {
        int token_id = tokens[s];
        
        // Находим начало строки для текущего токена
        const float* src_row = embed_table + (size_t)token_id * hidden_dim;
        float* dst_row = output + s * hidden_dim;

        // Копируем всю размерность hidden_dim
        for (size_t h = 0; h < hidden_dim; ++h) {
            dst_row[h] = src_row[h];
        }
    }
}

// Функция выполняет сложение с остаточным потоком и RMS нормализацию.
// x_out       - входной вектор X (перезаписывается нормализованным выходом Y)
// residual    - накопительный буфер (перезаписывается суммой residual + X)
// weight      - веса слоя нормализации (g)
inline void cpu_rmsnorm_residual(float* x_out, 
                                 float* residual, 
                                 const float* weight, 
                                 size_t seq_len, 
                                 size_t hidden_dim, 
                                 float eps = 1e-5f) {
    for (size_t s = 0; s < seq_len; ++s) {
        float* cur_x = x_out + s * hidden_dim;
        float* cur_res = residual + s * hidden_dim;

        // 1. Сложение остатка и расчет суммы квадратов
        float sum_sq = 0.0f;
        for (size_t h = 0; h < hidden_dim; ++h) {
            cur_res[h] += cur_x[h];          // Residual Add
            sum_sq += cur_res[h] * cur_res[h]; // Аккумуляция квадратов
        }

        // 2. Расчет обратного квадратного корня (1 / RMS)
        float rsqrt = 1.0f / std::sqrt((sum_sq / hidden_dim) + eps);

        // 3. Применение весов нормализации
        for (size_t h = 0; h < hidden_dim; ++h) {
            cur_x[h] = cur_res[h] * rsqrt * weight[h];
        }
    }
}

// Точная распаковка 1 байта FP8 (E4M3) в стандартный float
inline float cpu_unpack_fp8_e4m3(uint8_t byte_val) {
    // Проверка на чистый ноль
    if ((byte_val & 0x7F) == 0) {
        return (byte_val & 0x80) ? -0.0f : 0.0f;
    }

    int sign = (byte_val & 0x80) ? -1 : 1;
    int exp  = (byte_val & 0x78) >> 3;
    int mant = byte_val & 0x07;

    if (exp == 0) {
        // Субнормальные числа (экспонента фиксирована как 1 - bias = -6)
        return sign * std::ldexp(static_cast<float>(mant) / 8.0f, -6);
    }
    
    // Нормальные числа (bias = 7)
    return sign * std::ldexp(1.0f + static_cast<float>(mant) / 8.0f, exp - 7);
}

// Эталонное умножение квантованной матрицы FP8 на вектор X
// W_fp8  - матрица [M, K] в байтах
// X      - вектор активаций [K]
// scales - вектор коэффициентов [M]
// Y_out  - выходной вектор [M]
inline void cpu_fp8_gemv(const uint8_t* W_fp8,
                         const float* X,
                         const float* scales,
                         float* Y_out,
                         size_t M,
                         size_t K) {
    for (size_t i = 0; i < M; ++i) {
        double row_sum = 0.0;
        const uint8_t* cur_W_row = W_fp8 + i * K;

        for (size_t j = 0; j < K; ++j) {
            float w_float = cpu_unpack_fp8_e4m3(cur_W_row[j]);
            row_sum += w_float * X[j];
        }

        Y_out[i] = static_cast<float>(row_sum) * scales[i];
    }
}

// Применение RoPE к Q и K с одновременной записью K и V в кэш.
// Работает для одного токена на заданной позиции pos.
inline void cpu_fused_rope_kv_append(
    float* Q,               // Входной/выходной вектор Q [q_heads * head_dim]
    float* K,               // Входной вектор K [kv_heads * head_dim]
    const float* V,         // Входной вектор V [kv_heads * head_dim]
    float* K_cache,         // Глобальный кэш ключей
    float* V_cache,         // Глобальный кэш значений
    int pos,                // Текущая позиция токена в тексте
    size_t q_heads,         // Количество голов Q
    size_t kv_heads,        // Количество голов K/V
    size_t head_dim,        // Размерность одной головы (обычно 128)
    size_t max_seq_len,     // Максимальная емкость кэша
    float rope_theta = 500000.0f) 
{
    // 1. Обработка Query (Q)
    for (size_t h = 0; h < q_heads; ++h) {
        float* cur_q = Q + h * head_dim;
        
        // RoPE вращает пары элементов (i, i+1)
        for (size_t i = 0; i < head_dim; i += 2) {
            // Расчет частоты для текущей пары
            float freq = 1.0f / std::pow(rope_theta, static_cast<float>(i) / head_dim);
            float angle = pos * freq;
            float cos_val = std::cos(angle);
            float sin_val = std::sin(angle);

            float q0 = cur_q[i];
            float q1 = cur_q[i + 1];

            cur_q[i]     = q0 * cos_val - q1 * sin_val;
            cur_q[i + 1] = q0 * sin_val + q1 * cos_val;
        }
    }

    // 2. Обработка Key (K) и запись K/V в кэш
    // Для простоты MWV кэш имеет линейную структуру [kv_heads, max_seq_len, head_dim]
    for (size_t h = 0; h < kv_heads; ++h) {
        float* cur_k = K + h * head_dim;
        const float* cur_v = V + h * head_dim;
        
        // Указатели на слоты кэша для текущей позиции pos
        float* k_slot = K_cache + (h * max_seq_len + pos) * head_dim;
        float* v_slot = V_cache + (h * max_seq_len + pos) * head_dim;

        for (size_t i = 0; i < head_dim; i += 2) {
            float freq = 1.0f / std::pow(rope_theta, static_cast<float>(i) / head_dim);
            float angle = pos * freq;
            float cos_val = std::cos(angle);
            float sin_val = std::sin(angle);

            float k0 = cur_k[i];
            float k1 = cur_k[i + 1];

            // Поворот ключа
            float k0_rot = k0 * cos_val - k1 * sin_val;
            float k1_rot = k0 * sin_val + k1 * cos_val;

            // Перезаписываем локальный K (опционально) и пишем в кэш
            cur_k[i]     = k0_rot;
            cur_k[i + 1] = k1_rot;

            k_slot[i]     = k0_rot;
            k_slot[i + 1] = k1_rot;

            // Значение V просто копируется в свой кэш
            v_slot[i]     = cur_v[i];
            v_slot[i + 1] = cur_v[i + 1];
        }
    }
}

// Эталонное вычисление внимания для фазы Decoding (генерация 1 токена)
// Q       - вектор запроса текущего токена [q_heads, head_dim]
// K_cache - глобальный кэш ключей [kv_heads, max_seq_len, head_dim]
// V_cache - глобальный кэш значений [kv_heads, max_seq_len, head_dim]
// O_out   - выходной вектор [q_heads, head_dim]
inline void cpu_attention_decoding(
    const float* Q,
    const float* K_cache,
    const float* V_cache,
    float* O_out,
    int pos,                // Текущая позиция (сколько токенов в кэше уже есть)
    size_t q_heads,
    size_t kv_heads,
    size_t head_dim,
    size_t max_seq_len) 
{
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    size_t gqa_ratio = q_heads / kv_heads;

    for (size_t qh = 0; qh < q_heads; ++qh) {
        size_t kvh = qh / gqa_ratio; // Соответствующая голова K/V для текущей Q
        
        const float* cur_q = Q + qh * head_dim;
        std::vector<float> scores(pos + 1, 0.0f);
        float max_score = -INFINITY;

        // 1. Считаем Q * K^T для всех токенов от 0 до pos
        for (int t = 0; t <= pos; ++t) {
            const float* cur_k = K_cache + (kvh * max_seq_len + t) * head_dim;
            float dot = 0.0f;
            for (size_t i = 0; i < head_dim; ++i) {
                dot += cur_q[i] * cur_k[i];
            }
            float score = dot * scale;
            scores[t] = score;
            if (score > max_score) {
                max_score = score;
            }
        }

        // 2. Считаем Softmax (с вычитанием максимума для стабильности)
        float sum_exp = 0.0f;
        for (int t = 0; t <= pos; ++t) {
            scores[t] = std::exp(scores[t] - max_score);
            sum_exp += scores[t];
        }

        // 3. Умножаем на V и аккумулируем в выходной вектор
        float* cur_o = O_out + qh * head_dim;
        for (size_t i = 0; i < head_dim; ++i) {
            cur_o[i] = 0.0f; // Инициализируем нулем
        }

        for (int t = 0; t <= pos; ++t) {
            float prob = scores[t] / sum_exp;
            const float* cur_v = V_cache + (kvh * max_seq_len + t) * head_dim;
            for (size_t i = 0; i < head_dim; ++i) {
                cur_o[i] += prob * cur_v[i];
            }
        }
    }
}

// Эталонное вычисление Fused SwiGLU
// num_elements - общее количество элементов (для Decoding это просто intermediate_dim)
inline void cpu_fused_swiglu(const float* gate, 
                             const float* up, 
                             float* output, 
                             size_t num_elements) 
{
    for (size_t i = 0; i < num_elements; ++i) {
        float g = gate[i];
        float u = up[i];
        
        // SiLU / Swish активация
        float sigmoid = 1.0f / (1.0f + std::exp(-g));
        float swish = g * sigmoid;
        
        output[i] = swish * u;
    }
}

// Эталонный поиск индекса максимального элемента (Argmax)
inline void cpu_argmax(const float* logits, int* out_token_id, size_t vocab_size) {
    float max_val = logits[0];
    int max_idx = 0;

    for (size_t i = 1; i < vocab_size; ++i) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = static_cast<int>(i);
        }
    }

    *out_token_id = max_idx;
}