#pragma once
#include <cstddef>

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