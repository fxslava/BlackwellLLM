#pragma once
#include <cstddef>

// Оставляем оригинальный Argmax
void launch_argmax_kernel(const float* d_logits, int* d_out_token_id, size_t vocab_size);

// Новая гибридная функция сэмплирования (GPU -> CPU)
int sample_top_p(const float* d_logits, size_t vocab_size, float temperature, float top_p);

float compute_log_prob(const float* d_logits, size_t vocab_size, int target_token_id);