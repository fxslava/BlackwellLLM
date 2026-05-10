#pragma once
#include <cstddef>

void launch_argmax_kernel(const float* d_logits, int* d_out_token_id, size_t vocab_size);