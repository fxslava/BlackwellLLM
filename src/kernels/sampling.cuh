#pragma once
#include <cstddef>

// Оставляем оригинальный Argmax
void launch_argmax_kernel(const float* d_logits, int* d_out_token_id, size_t vocab_size);

// Новая гибридная функция сэмплирования (GPU -> CPU)
int sample_top_p(const float* d_logits, size_t vocab_size, float temperature, float top_p);

// CTRL-style repetition penalty (Keskar et al. 2019) applied IN PLACE to the raw
// logits, BEFORE any temperature/top-p/argmax step: every id present in
// d_token_ids has its logit divided by `penalty` when positive and multiplied by
// it when negative, so a token the model has already emitted becomes strictly
// less likely. This is the mathematical brake on degenerate phrase loops (the
// model repeating one phrase until the context cap), and it runs on the DEVICE so
// the greedy temperature-0 path keeps its GPU-only fast lane -- no vocab-size D2H
// copy is introduced.
//
//   d_token_ids : device array of recently generated ids; duplicates are allowed
//                 (each DISTINCT id is penalized exactly once, so the result is
//                 deterministic and independent of thread scheduling).
//   num_tokens  : its length; <= 0 is a no-op.
//   penalty     : > 1 penalizes. <= 1 is a no-op (the caller's "disabled" value).
//
// Out-of-range ids are skipped. Launches on the default stream, like the argmax
// kernel above, so it orders with the decode pass that produced the logits.
void launch_repetition_penalty_kernel(float* d_logits, size_t vocab_size,
                                      const int* d_token_ids, int num_tokens,
                                      float penalty);

float compute_log_prob(const float* d_logits, size_t vocab_size, int target_token_id);