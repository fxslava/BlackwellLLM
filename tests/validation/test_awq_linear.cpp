// AWQ int4 GEMV mathematical correctness on the exact Qwen2.5-Coder-7B decode
// shapes (hidden=3584, intermediate=18944, GQA KV dim = 4 heads * 128 = 512,
// group_size=128), asserted against an FP64 CPU reference within FP16-scale
// tolerance. Edge shapes and malformed parameters live in
// benchmarks/stress_awq_linear.cpp.
#include <gtest/gtest.h>

#include "common/awq_reference.h"

using test_utils::expect_awq_matches_reference;

class AwqValidation : public test_utils::CudaTest {};

TEST_F(AwqValidation, QwenAttentionQProj) {
    expect_awq_matches_reference(3584, 3584, 128, 11);
}

// OC=512 -> only 64 packed columns: the launcher distributes the reduction
// dimension across blocks (split-K), so GQA K/V projections take the
// atomic-accumulation path.
TEST_F(AwqValidation, QwenGqaKvProjSplitK) {
    expect_awq_matches_reference(3584, 512, 128, 12);
}

TEST_F(AwqValidation, QwenMlpGateUpProj) {
    expect_awq_matches_reference(3584, 18944, 128, 13);
}

TEST_F(AwqValidation, QwenMlpDownProj) {
    expect_awq_matches_reference(18944, 3584, 128, 14);
}

// Vocab-sized projection (Qwen lm_head): exercises grid sizing and 64-bit
// indexing at the largest shape the engine launches. Constant-pattern
// weights keep the CPU reference O(IC) instead of O(IC*OC).
TEST_F(AwqValidation, QwenVocabSizedLmHead) {
    const test_utils::AwqProblem p =
        test_utils::make_awq_constant_problem(3584, 152064, 128, 15);
    test_utils::AwqDevice dev(p);
    const float max_rel = test_utils::max_rel_error(dev.run(p), p.ref, 1e-2f);
    EXPECT_LE(max_rel, test_utils::kAwqRelTolerance)
        << "lm_head-sized GEMV mismatch, max_rel=" << max_rel;
}
