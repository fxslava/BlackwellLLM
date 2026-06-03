#pragma once
#include <cstddef>
#include <string>

// Рутина для запуска базовых тестов nvCOMP (Gdeflate, Snappy, Cascaded, Bitcomp)
void run_nvcomp_baseline(const void* d_weight_ptr, size_t tensor_size);