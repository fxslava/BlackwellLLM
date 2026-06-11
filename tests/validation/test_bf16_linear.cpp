// BF16 GEMV correctness (the Llama logits / lm_head projection path) against
// an FP64-accumulated CPU reference.
#include <gtest/gtest.h>

#include "common/bf16_reference.h"

using test_utils::Bf16Problem;
using test_utils::Bf16Device;
using test_utils::make_bf16_problem;
using test_utils::max_rel_error;

class Bf16Validation : public test_utils::CudaTest {};

TEST_F(Bf16Validation, GemvMatchesReference) {
    const Bf16Problem p = make_bf16_problem(256, 4096, 61);
    Bf16Device dev(p);
    const float max_rel = max_rel_error(dev.run(p), p.ref, 1e-2f);
    EXPECT_LE(max_rel, 1e-3f) << "BF16 GEMV mismatch, max_rel=" << max_rel;
}

// K=14336 mirrors a hidden-state-sized reduction over the MLP intermediate
// width; longer accumulation chains expose FP32 ordering drift.
TEST_F(Bf16Validation, DeepReductionK14336) {
    const Bf16Problem p = make_bf16_problem(64, 14336, 62);
    Bf16Device dev(p);
    const float max_rel = max_rel_error(dev.run(p), p.ref, 1e-2f);
    EXPECT_LE(max_rel, 1e-3f) << "BF16 GEMV mismatch, max_rel=" << max_rel;
}
