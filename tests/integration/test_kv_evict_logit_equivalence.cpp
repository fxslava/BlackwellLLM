// =============================================================================
// KV HEAD EVICTION — IN-ENGINE POSITIONAL EQUIVALENCE (Tier 2: GPU + checkpoint).
//
// tests/validation/test_kv_evict.cpp proves the kernel's math in isolation. This
// proves it inside the real engine, at real 8B geometry, through the real
// attention path — and pins the boundary of what eviction can actually promise.
//
// WHAT EVICTION DOES AND DOES NOT PROMISE (the thing to understand before
// reading the assertions, and the reason this file is not what it first was):
//
// The obvious specification — "an evicted cache equals a from-scratch prefill of
// the surviving tokens" — is FALSE, and not because of any bug. When [P][A][B]
// was prefilled, every token of B attended to A, so B's K and V at layers 1..N-1
// already encode A's content. Sliding those rows down and re-phasing them fixes
// WHERE they are; nothing can un-condition WHAT they are. Evicting from a KV
// cache never yields "as if those tokens were never there" — it yields "those
// positions can no longer be attended to". That is equally true of StreamingLLM
// and of every sliding-window scheme; it is the defining lossiness of the
// technique, not a defect in this kernel.
//
// Measured here, for the record: full logits diverge (cosine ~0.83), while
//
//   *** LAYER 0's K CACHE MATCHES A FRESH PREFILL AT COSINE 1.00000000 ***
//
// and that is the exact, decisive statement. Layer 0's K is a pure function of
// (token embedding, position) — it has no context term at all — so it is the one
// place in the network where the positional transform is observable in isolation.
// If the composed rotation were wrong by any amount, in any channel, under this
// checkpoint's theta and scaling, layer 0 would not match. Layers 1+ carrying
// forward the evicted context is expected, is asserted as expected, and is
// exactly the property that makes eviction lossy-but-coherent.
//
// The model-level claim is therefore the honest one: after eviction the model
// still reads the SURVIVING context correctly and answers from it identically.
// =============================================================================
// Needs only the engine, the eviction kernel, a checkpoint and its tokenizer —
// no speech stack, no audio head, no golden dumps. SKIPS (never fails) when the
// checkpoint is absent, like every other Tier-2 suite here.
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "blackwell/engine.h"
#include "blackwell/tokenizer.h"
#include "audio_test_wav.hpp"      // awq_index_path(), model_dir_of() — shipping 8B
#include "engine_impl.h"           // white-box: run_token, d_logits, kv_mgr, arena
#include "kernels/kv_evict.cuh"
#include "rope_config.h"           // rope_scaling_from(ModelConfig)

namespace {

using blackwell::EngineStatus;
using blackwell_test_audio::awq_index_path;
using blackwell_test_audio::file_exists;
using blackwell_test_audio::model_dir_of;

// The prompt. The FACT LIVES IN THE SURVIVING BLOCK: that is what makes the
// model-level assertion answerable at all (a fact in the evicted block would be
// partly remembered through B's conditioned K/V — see the header — and would
// make the test measure the lossiness rather than the transform).
constexpr const char* kPrefixText =
    "You are a precise assistant. Answer with a single word.";
constexpr const char* kEvictedText =
    " The weather today is mild and the meeting moved to the larger room upstairs.";
constexpr const char* kSurvivorText =
    " The password is ZEBRA. Question: what is the password? Answer:";
// Same shape, different answer — the control that proves the surviving block is
// actually being read rather than the reply coming from the prefix alone.
constexpr const char* kSurvivorOther =
    " The password is FALCON. Question: what is the password? Answer:";

// The measurement is the forward pass for the prompt's OWN LAST token, so the
// logits predict the answer itself. Appending a junk probe token instead would
// measure "what follows that junk token", which is the same bland continuation
// whatever the password was — the control then has nothing to detect.


double cosine(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        na += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        nb += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
}

int argmax(const std::vector<float>& v) {
    int best = 0;
    for (size_t i = 1; i < v.size(); ++i) {
        if (v[i] > v[static_cast<size_t>(best)]) best = static_cast<int>(i);
    }
    return best;
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    }
    return m;
}

class KvEvictLogitEquivalence : public ::testing::Test {
protected:
    // BlackwellEngine::Impl is only forward-declared in the private section of
    // engine.h, so the type is not nameable out here — reach it through `auto`,
    // exactly as the other white-box suites do. Defined before its first use: an
    // auto-returning member has no type until the compiler has seen its body.
    auto* core() const { return engine_->get_impl(); }

    void SetUp() override {
        const std::string index = awq_index_path();
        if (!file_exists(index))
            GTEST_SKIP() << "checkpoint absent: " << index << " (set BLACKWELL_AWQ_INDEX)";

        auto tok = blackwell::TokenizerFactory::create(model_dir_of(index));
        ASSERT_NE(tok, nullptr);
        const std::vector<int> p = tok->encode(kPrefixText, /*add_special=*/true);
        const std::vector<int> a = tok->encode(kEvictedText, /*add_special=*/false);
        const std::vector<int> b = tok->encode(kSurvivorText, /*add_special=*/false);
        const std::vector<int> o = tok->encode(kSurvivorOther, /*add_special=*/false);
        ASSERT_FALSE(p.empty());
        ASSERT_FALSE(a.empty());
        ASSERT_FALSE(b.empty());

        prefix_ = static_cast<int>(p.size());
        evicted_ = static_cast<int>(a.size());
        survivor_ = static_cast<int>(b.size());
        full_ = p;
        full_.insert(full_.end(), a.begin(), a.end());
        full_.insert(full_.end(), b.begin(), b.end());
        truncated_ = p;
        truncated_.insert(truncated_.end(), b.begin(), b.end());
        other_ = p;
        other_.insert(other_.end(), o.begin(), o.end());

        std::printf("[kv-evict] prefix=%d evicted=%d survivor=%d total=%d\n", prefix_,
                    evicted_, survivor_, static_cast<int>(full_.size()));

        const size_t cap = std::max(full_.size(), other_.size()) + 8;
        engine_ = std::make_unique<BlackwellEngine>(index, cap);
        vocab_ = static_cast<int>(core()->m_config.vocab_size);
    }

    // Prefills every token of `src` EXCEPT the last, which is held back to be the
    // measured forward (see the note above kSurvivorText).
    void prefill_body(const std::vector<int>& src) {
        reset_cache();
        for (size_t pos = 0; pos + 1 < src.size(); ++pos) {
            ASSERT_EQ(core()->run_token(src[pos], static_cast<int>(pos), /*seq_id=*/0,
                                        /*want_logits=*/false),
                      EngineStatus::Success);
        }
    }

    // The measurement: one forward for `token` at `pos`, reading the whole cache
    // beneath it, returning the logits it predicts.
    std::vector<float> probe(int token, int pos) {
        EXPECT_EQ(core()->run_token(token, pos, /*seq_id=*/0, /*want_logits=*/true),
                  EngineStatus::Success);
        std::vector<float> out(static_cast<size_t>(vocab_));
        CUDA_CHECK(cudaMemcpy(out.data(), core()->d_logits,
                              static_cast<size_t>(vocab_) * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
        return out;
    }

    std::vector<float> prefill_and_probe(const std::vector<int>& src) {
        prefill_body(src);
        return probe(src.back(), static_cast<int>(src.size()) - 1);
    }

    // The live region of one layer's K or V cache: [kv_heads][rows][head_dim].
    std::vector<float> snapshot(int layer, int rows, bool key) {
        const auto& c = core()->m_config;
        const float* base = static_cast<const float*>(
            key ? core()->kv_mgr->get_layer_k_ptr(layer) : core()->kv_mgr->get_layer_v_ptr(layer));
        const size_t msl = core()->arena.get_max_seq_len();
        std::vector<float> out(c.num_key_value_heads * static_cast<size_t>(rows) * c.head_dim);
        for (size_t h = 0; h < c.num_key_value_heads; ++h) {
            CUDA_CHECK(cudaMemcpy(out.data() + h * static_cast<size_t>(rows) * c.head_dim,
                                  base + h * msl * c.head_dim,
                                  static_cast<size_t>(rows) * c.head_dim * sizeof(float),
                                  cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        return out;
    }

    // Drops [prefix_, prefix_ + evicted_) from every layer, re-phasing survivors.
    // Uses the engine's OWN geometry and RoPE parameters — passing a different
    // theta or scaling than the append path used is exactly the mistake this test
    // is positioned to catch.
    void evict_head_all_layers(int cache_len) {
        const auto& c = core()->m_config;
        for (int l = 0; l < static_cast<int>(c.num_layers); ++l) {
            launch_kv_evict_head(static_cast<float*>(core()->kv_mgr->get_layer_k_ptr(l)),
                                 static_cast<float*>(core()->kv_mgr->get_layer_v_ptr(l)),
                                 prefix_, evicted_, cache_len, c.num_key_value_heads,
                                 c.head_dim, core()->arena.get_max_seq_len(), c.rope_theta,
                                 rope_scaling_from(c));
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // The contiguous FP32 cache is position-addressed and overwritten in place,
    // so a rewind to 0 plus a fresh prefill leaves nothing of the previous run
    // readable: attention only ever reads [0, pos].
    void reset_cache() { core()->kv_mgr->rewind(0, 0); }

    std::unique_ptr<BlackwellEngine> engine_;
    std::vector<int> full_, truncated_, other_;
    int prefix_ = 0, evicted_ = 0, survivor_ = 0, vocab_ = 0;
};

// THE POSITIONAL CONTRACT, inside the engine. Layer 0's K is a pure function of
// (token embedding, position) — no context term — so it is the one tensor in the
// network where the re-phase is observable on its own. Exact agreement here means
// the composed rotation is right for every channel under this checkpoint's actual
// theta, scaling, head_dim and kv_head count.
TEST_F(KvEvictLogitEquivalence, Layer0CacheMatchesAFreshPrefillExactly) {
    // The last token is held back as the probe, so the prefilled body is one
    // shorter than the full sequence.
    const int live = prefix_ + survivor_ - 1;

    prefill_body(full_);
    evict_head_all_layers(static_cast<int>(full_.size()) - 1);
    const std::vector<float> k_evicted = snapshot(0, live, /*key=*/true);
    const std::vector<float> v_evicted = snapshot(0, live, /*key=*/false);

    prefill_body(truncated_);
    const std::vector<float> k_ref = snapshot(0, live, /*key=*/true);
    const std::vector<float> v_ref = snapshot(0, live, /*key=*/false);

    const auto& c = core()->m_config;
    std::printf("[kv-evict] geometry: layers=%zu kv_heads=%zu head_dim=%zu rotary=%zu "
                "msl=%zu theta=%.1f scaling=%d\n",
                c.num_layers, c.num_key_value_heads, c.head_dim, c.rotary_dim,
                core()->arena.get_max_seq_len(), static_cast<double>(c.rope_theta),
                c.rope_scaling_type);
    std::printf("[kv-evict] L0 K: cosine %.8f  max|d| %.7f\n", cosine(k_evicted, k_ref),
                max_abs_diff(k_evicted, k_ref));
    std::printf("[kv-evict] L0 V: cosine %.8f  max|d| %.7f\n", cosine(v_evicted, v_ref),
                max_abs_diff(v_evicted, v_ref));

    EXPECT_GT(cosine(k_evicted, k_ref), 0.9999999)
        << "re-phased layer-0 keys do not match a fresh prefill: the composed rotation is "
           "wrong under this checkpoint's rope parameters";
    EXPECT_LT(max_abs_diff(k_evicted, k_ref), 1e-3);
    // V is moved, never rotated, so eviction contributes NOTHING to its error.
    // Not asserted bit-exact only because the engine itself is not bit-reproducible
    // (the determinism floor measured in the model-level test below is ~1e-5 on
    // logits, from reduction order in the AWQ GEMV) — so this bar is the engine's
    // own noise, an order of magnitude under the re-phased keys'.
    EXPECT_LT(max_abs_diff(v_evicted, v_ref), 1e-4)
        << "V was altered by eviction; it carries no phase and must move untouched";
}

// The counterpart, asserted so the limitation is documented in executable form
// rather than in a comment somebody can drift away from: deeper layers DO retain
// the evicted context, because B's rows were computed while attending to A.
TEST_F(KvEvictLogitEquivalence, DeeperLayersRetainTheEvictedContextAsExpected) {
    // The last token is held back as the probe, so the prefilled body is one
    // shorter than the full sequence.
    const int live = prefix_ + survivor_ - 1;
    const int last = static_cast<int>(core()->m_config.num_layers) - 1;

    prefill_body(full_);
    evict_head_all_layers(static_cast<int>(full_.size()) - 1);
    const std::vector<float> deep_evicted = snapshot(last, live, /*key=*/true);

    prefill_body(truncated_);
    const std::vector<float> deep_ref = snapshot(last, live, /*key=*/true);

    const double cos_deep = cosine(deep_evicted, deep_ref);
    std::printf("[kv-evict] L%d K (expected to differ): cosine %.8f  max|d| %.5f\n", last,
                cos_deep, max_abs_diff(deep_evicted, deep_ref));

    EXPECT_LT(cos_deep, 0.9999999)
        << "the last layer's keys matched a fresh prefill exactly. That would mean the "
           "surviving rows never encoded the evicted block at all — i.e. attention is not "
           "reading the cache — and would invalidate the premise of every other assertion "
           "here";
}

// The model-level claim: after eviction the surviving context is still read
// correctly and produces the same answer.
TEST_F(KvEvictLogitEquivalence, ModelAnswersFromTheSurvivingContextAfterEviction) {
    prefill_body(full_);
    evict_head_all_layers(static_cast<int>(full_.size()) - 1);
    const std::vector<float> logits_evicted = probe(full_.back(), prefix_ + survivor_ - 1);

    const std::vector<float> logits_scratch = prefill_and_probe(truncated_);
    const std::vector<float> logits_scratch2 = prefill_and_probe(truncated_);
    const std::vector<float> logits_other = prefill_and_probe(other_);

    const double cos_floor = cosine(logits_scratch, logits_scratch2);
    const double cos_eq    = cosine(logits_evicted, logits_scratch);
    const double cos_ctrl  = cosine(logits_other, logits_scratch);

    std::printf("[kv-evict] determinism floor    : cosine %.8f\n", cos_floor);
    std::printf("[kv-evict] evicted  vs scratch  : cosine %.8f  top1 %d vs %d\n", cos_eq,
                argmax(logits_evicted), argmax(logits_scratch));
    std::printf("[kv-evict] control  other vs scr: cosine %.8f  top1 %d vs %d\n", cos_ctrl,
                argmax(logits_other), argmax(logits_scratch));

    // Control first: if a different password does not change the answer, the model
    // is not reading the surviving block and the agreement below proves nothing.
    ASSERT_NE(argmax(logits_other), argmax(logits_scratch))
        << "changing the password did not change the answer — the surviving block is not "
           "being read, so this test has no teeth";

    EXPECT_EQ(argmax(logits_evicted), argmax(logits_scratch))
        << "after eviction the model gave a different answer than a fresh prefill of the "
           "same surviving context";
    // Eviction must land far closer to the fresh prefill than a genuine content
    // change does. It cannot land AT it — see the header.
    EXPECT_LT(1.0 - cos_eq, 1.0 - cos_ctrl)
        << "an evicted cache is no closer to the fresh prefill than a different prompt is";
}

}  // namespace
