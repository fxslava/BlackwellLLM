// BF16 GEMV residual accumulation (the unquantized o_proj / down_proj path)
// against an FP64-accumulated CPU reference. The contract under test is
// Y_accum[row] += W @ x: the pre-existing residual stream must survive.
#include <gtest/gtest.h>

#include "common/bf16_reference.h"

using test_utils::Bf16ResidualProblem;
using test_utils::Bf16ResidualDevice;
using test_utils::make_bf16_residual_problem;
using test_utils::max_rel_error;

class Bf16ResidualValidation : public test_utils::CudaTest {};

TEST_F(Bf16ResidualValidation, AccumMatchesReference) {
    const Bf16ResidualProblem p = make_bf16_residual_problem(256, 4096, 71);
    Bf16ResidualDevice dev(p);
    const float max_rel = max_rel_error(dev.run(p), p.ref, 1e-2f);
    EXPECT_LE(max_rel, 1e-3f) << "BF16 residual GEMV mismatch, max_rel=" << max_rel;
}

// K=18944 mirrors the Qwen2.5-Coder-7B down_proj reduction over the MLP
// intermediate width; longer accumulation chains expose FP32 ordering drift.
TEST_F(Bf16ResidualValidation, DeepReductionK18944) {
    const Bf16ResidualProblem p = make_bf16_residual_problem(64, 18944, 72);
    Bf16ResidualDevice dev(p);
    const float max_rel = max_rel_error(dev.run(p), p.ref, 1e-2f);
    EXPECT_LE(max_rel, 1e-3f) << "BF16 residual GEMV mismatch, max_rel=" << max_rel;
}

// Launching twice must add the dot product twice (accum_init + 2 * W @ x):
// a kernel that overwrites instead of accumulating passes a single-launch
// comparison whenever the residual happens to be small, so this pins the +=.
TEST_F(Bf16ResidualValidation, RepeatedLaunchAccumulates) {
    const Bf16ResidualProblem p = make_bf16_residual_problem(256, 4096, 73);
    Bf16ResidualDevice dev(p);
    dev.launch(p);
    const std::vector<float> gpu = dev.run(p);

    std::vector<float> ref2(p.base.M);
    for (size_t i = 0; i < p.base.M; ++i)
        ref2[i] = p.accum_init[i] + 2.0f * p.base.ref[i];

    const float max_rel = max_rel_error(gpu, ref2, 1e-2f);
    EXPECT_LE(max_rel, 1e-3f) << "BF16 residual GEMV double-launch mismatch, max_rel=" << max_rel;
}
