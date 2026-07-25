#pragma once
// Bridges ModelConfig's flat rope_scaling fields (blackwell/config.h, CUDA-free)
// to the kernels' RopeScaling POD (kernels/rope.cuh). Kept out of config.h so the
// public header carries no kernel dependency; every RoPE call site in src/core
// includes this and passes the result to the launchers.
#include "blackwell/config.h"
#include "kernels/rope.cuh"

// ModelConfig and RopeScaling are both at global scope, so this bridge is too --
// callers span global-scope (engine.cpp) and namespace blackwell (kv managers).
inline RopeScaling rope_scaling_from(const ModelConfig& c) {
    RopeScaling s;
    s.enabled          = c.rope_scaling_type;   // 0 == vanilla, 1 == llama3
    s.factor           = c.rope_scaling_factor;
    s.low_freq_factor  = c.rope_low_freq_factor;
    s.high_freq_factor = c.rope_high_freq_factor;
    s.orig_ctx         = c.rope_orig_max_pos;
    return s;
}
