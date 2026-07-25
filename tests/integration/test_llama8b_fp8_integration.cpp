// =============================================================================
// 8B FP8 backbone integration test (INJECTED-EMBEDDING prefill).
//
// Verifies the engine's Llama-3 8B FP8 prefill INDEPENDENTLY of the audio
// projector: a [seq_len, 4096] embedding matrix (dumped from the HF FP8 model via
// scripts/dump_llama8b_fp8.py) is injected directly into the residual stream, the
// full 32-layer FP8 transformer is run position-by-position over the causal KV
// cache, and the per-position logits are compared against the PyTorch reference.
//
// WHY THE GRANULAR step_* LOOP (not run_chunk): run_chunk is token-id driven (it
// does its OWN embedding lookup) and emits logits for the last token only. The
// granular decode steps are the exact decomposition of run_token, so replacing
// step_embedding with a row-copy into d_X_accum and looping over positions
// reproduces a causal prefill FROM INJECTED EMBEDDINGS with per-position logits,
// reusing every real FP8 kernel (qkv/o/gate/up/down GEMV) and the KV cache. The
// forward loop allocates NO device memory (rule: zero-allocation forward pass).
//
// External requirements (SKIPs if either is missing):
//   - checkpoint index  F:/AI/llama3-8b-fp8/model.safetensors.index.json
//                       (override: env BLACKWELL_LLAMA_INDEX)
//   - dumps dir <repo>/dumps with 08b_input_embeds.bin + 08b_final_logits.bin
//                       (override: env BLACKWELL_DUMPS_DIR)  -- run the generator.
// =============================================================================
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#include "common.h"                 // CUDA_CHECK
#include "device_buffer.h"          // blackwell::DeviceBuffer
#include "blackwell/engine.h"
#include "engine_impl.h"

using blackwell::DeviceBuffer;

namespace {

std::string env_or(const char* name, const std::string& fallback) {
    const char* v = std::getenv(name);
    return (v && *v) ? std::string(v) : fallback;
}
std::string model_index_path() {
    return env_or("BLACKWELL_LLAMA_INDEX", "F:/AI/llama3-8b-fp8/model.safetensors.index.json");
}
std::string dumps_dir() {
    return env_or("BLACKWELL_DUMPS_DIR", "D:/Projects/BlackwellLLM/dumps");
}
bool file_exists(const std::string& p) { return std::ifstream(p).good(); }

std::vector<float> load_dump(const std::string& filename, size_t num_elements) {
    const std::string full = dumps_dir() + "/" + filename;
    std::ifstream f(full, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("Dump not found: " + full + " (run scripts/dump_llama8b_fp8.py).");
    std::vector<float> buf(num_elements);
    f.read(reinterpret_cast<char*>(buf.data()), num_elements * sizeof(float));
    if (!f) throw std::runtime_error("Dump size mismatch: " + full);
    return buf;
}

// Cosine similarity over one logits row (double accumulation).
double cosine_row(const float* a, const float* b, size_t n) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; ++i) {
        const double x = a[i], y = b[i];
        dot += x * y; na += x * x; nb += y * y;
    }
    const double den = std::sqrt(na) * std::sqrt(nb);
    return den > 1e-12 ? dot / den : 0.0;
}
int argmax_row(const float* a, size_t n) {
    int best = 0; float bv = a[0];
    for (size_t i = 1; i < n; ++i) if (a[i] > bv) { bv = a[i]; best = static_cast<int>(i); }
    return best;
}

constexpr double kParityThreshold = 0.999;

}  // namespace

// -----------------------------------------------------------------------------
// Injected-embedding prefill over the 8B FP8 backbone; per-position logits parity.
// -----------------------------------------------------------------------------
TEST(Llama8bFp8Integration, InjectedEmbeddingPrefillLogitsParity) {
    if (!file_exists(model_index_path()))
        GTEST_SKIP() << "checkpoint absent: " << model_index_path()
                     << " (set BLACKWELL_LLAMA_INDEX)";
    if (!file_exists(dumps_dir() + "/08b_input_embeds.bin"))
        GTEST_SKIP() << "dumps absent in " << dumps_dir()
                     << " (run scripts/dump_llama8b_fp8.py, or set BLACKWELL_DUMPS_DIR)";

    // Geometry sidecar (falls back to the known 8B shape if absent).
    int seq_len = 194, hidden = 4096, vocab = 128256;
    {
        std::ifstream m(dumps_dir() + "/08b_meta.txt");
        if (m) m >> seq_len >> hidden >> vocab;
    }
    ASSERT_GT(seq_len, 0);
    std::cout << "[8bfp8] seq_len=" << seq_len << " hidden=" << hidden
              << " vocab=" << vocab << "\n";

    // --- Load golden tensors -------------------------------------------------
    const size_t vocab_sz = static_cast<size_t>(vocab);
    const size_t n_embed  = static_cast<size_t>(seq_len) * hidden;
    const size_t n_logits = static_cast<size_t>(seq_len) * vocab_sz;
    std::vector<float> h_embeds = load_dump("08b_input_embeds.bin", n_embed);
    std::vector<float> golden   = load_dump("08b_final_logits.bin", n_logits);

    // --- Engine + white-box surface -----------------------------------------
    std::cout << "[8bfp8] constructing engine (max_seq_len=" << (seq_len + 8) << ")...\n";
    BlackwellEngine engine(model_index_path(), static_cast<size_t>(seq_len) + 8);
    auto* core = engine.get_impl();
    ASSERT_EQ(static_cast<int>(core->m_config.hidden_dim), hidden);
    ASSERT_EQ(static_cast<int>(core->m_config.vocab_size), vocab);
    const int num_layers = static_cast<int>(core->m_config.num_layers);

    // Upload injected embeddings once; pre-size the host logits sink. The forward
    // loop below performs NO device allocation (rule: zero-alloc forward pass).
    DeviceBuffer<float> d_embeds(n_embed);
    CUDA_CHECK(cudaMemcpy(d_embeds.get(), h_embeds.data(), n_embed * sizeof(float),
                          cudaMemcpyHostToDevice));
    std::vector<float> h_logits(n_logits);

    // --- Injected-embedding causal prefill, position by position -------------
    std::cout << "[8bfp8] running " << seq_len << "-position FP8 prefill...\n";
    for (int pos = 0; pos < seq_len; ++pos) {
        // Inject embedding row -> residual stream (replaces step_embedding's
        // memset + embed_tokens lookup). d_X_accum is the [hidden] token-0 slot.
        CUDA_CHECK(cudaMemcpy(core->d_X_accum, d_embeds.get() + static_cast<size_t>(pos) * hidden,
                              hidden * sizeof(float), cudaMemcpyDeviceToDevice));

        // Latch the sequence for this position (mirrors run_token exactly).
        core->kv_mgr->prepare_decode_step(0, pos);

        for (int l = 0; l < num_layers; ++l) {
            ASSERT_EQ(core->step_attention_norm(l),            blackwell::EngineStatus::Success);
            ASSERT_EQ(core->step_attention_qkv_projections(l), blackwell::EngineStatus::Success);
            ASSERT_EQ(core->step_attention_math(l, pos),       blackwell::EngineStatus::Success);
            ASSERT_EQ(core->step_attention_out(l),             blackwell::EngineStatus::Success);
            ASSERT_EQ(core->step_mlp_norm(l),                  blackwell::EngineStatus::Success);
            ASSERT_EQ(core->step_mlp_projections(l),           blackwell::EngineStatus::Success);
            ASSERT_EQ(core->step_mlp_out(l),                   blackwell::EngineStatus::Success);
        }
        ASSERT_EQ(core->step_final_ops(), blackwell::EngineStatus::Success);

        CUDA_CHECK(cudaMemcpy(h_logits.data() + static_cast<size_t>(pos) * vocab,
                              core->d_logits, vocab * sizeof(float), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Parity: aggregate + per-position cosine, top-1 agreement ------------
    const double agg_cos = cosine_row(h_logits.data(), golden.data(), n_logits);

    double min_cos = 1.0; int min_pos = 0, top1_match = 0;
    for (int pos = 0; pos < seq_len; ++pos) {
        const float* a = h_logits.data() + static_cast<size_t>(pos) * vocab_sz;
        const float* g = golden.data()   + static_cast<size_t>(pos) * vocab_sz;
        const double c = cosine_row(a, g, vocab_sz);
        if (c < min_cos) { min_cos = c; min_pos = pos; }
        if (argmax_row(a, vocab_sz) == argmax_row(g, vocab_sz)) ++top1_match;
    }
    const int last = seq_len - 1;
    const int eng_top1 = argmax_row(h_logits.data() + static_cast<size_t>(last) * vocab_sz, vocab_sz);
    const int ref_top1 = argmax_row(golden.data()   + static_cast<size_t>(last) * vocab_sz, vocab_sz);

    std::cout << std::fixed << std::setprecision(8)
              << "[8bfp8] aggregate cosine      = " << agg_cos << "\n"
              << "[8bfp8] worst per-pos cosine  = " << min_cos << " (pos " << min_pos << ")\n"
              << "[8bfp8] top-1 agreement       = " << top1_match << "/" << seq_len
              << " (" << std::setprecision(1) << (100.0 * top1_match / seq_len) << "%)\n"
              << "[8bfp8] last-position top-1   : engine=" << eng_top1
              << " reference=" << ref_top1 << "\n";

    EXPECT_EQ(eng_top1, ref_top1)
        << "last-position next-token prediction diverged (FP8 engine vs bf16 reference)";
    EXPECT_GT(agg_cos, kParityThreshold)
        << "aggregate logits cosine below the " << kParityThreshold << " parity bar";
}
