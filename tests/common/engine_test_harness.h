#pragma once
// ============================================================================
// Shared scaffolding for the end-to-end engine integration tests.
// ============================================================================
// Every "engine vs PyTorch golden dump" suite (Llama-FP8, Qwen-AWQ, GLM-4) does
// the same five things, and used to do them with its own copy of the code:
//
//   1. load a flat fp32 .bin dump and complain usefully when it is absent or the
//      wrong size,
//   2. download a device tensor and score it against that dump (RMSE / max
//      error / cosine similarity),
//   3. measure the entropy of a probability row (how sharp the attention or the
//      output distribution is),
//   4. print a layer-by-layer telemetry table that a human can scan,
//   5. render the final-logits verdict: cosine floor OR golden top-1 inside the
//      engine's top-k.
//
// The one behavioural upgrade over the copies this replaces: `probe()` RETURNS
// the metrics instead of only printing them, so a suite can assert a per-stage
// cosine floor rather than eyeballing the table.
//
// Dumps are flat little-endian fp32, written by the scripts/generate_*_dumps.py
// generators. Device pointers are raw `const float*` into engine-owned buffers
// (the suites reach through BlackwellEngine::Impl, which is why they link
// blackwell_core_obj rather than the DLL).
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common.h"  // CUDA_CHECK

namespace engine_test {

// ---------------------------------------------------------------------------
// Environment / path helpers (every suite overrides its checkpoint and dump
// directory through env vars so CI and a workstation can disagree).
// ---------------------------------------------------------------------------
inline std::string env_or(const char* name, const std::string& fallback) {
    const char* v = std::getenv(name);
    return (v && *v) ? std::string(v) : fallback;
}

inline bool file_exists(const std::string& path) {
    return std::ifstream(path).good();
}

// ---------------------------------------------------------------------------
// 1. Golden dump loading
// ---------------------------------------------------------------------------
// Reads exactly `num_elements` fp32 values. Throws std::runtime_error with the
// full path on a missing file, a short read, or a file whose size is not the
// requested length -- a dump that is the wrong SHAPE is the most common failure
// after a geometry change, and silently reading a prefix of it would turn that
// into a mysterious cosine of 0.3 instead of a clear error.
inline std::vector<float> load_golden_bin(const std::string& path, size_t num_elements) {
    if (num_elements == 0)
        throw std::runtime_error("load_golden_bin: refusing to load 0 elements from " + path);

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        throw std::runtime_error("Dump file not found: " + path +
                                 " (run the matching scripts/generate_*_dumps.py first)");

    const std::streamsize file_bytes = file.tellg();
    const std::streamsize want_bytes =
        static_cast<std::streamsize>(num_elements * sizeof(float));
    if (file_bytes != want_bytes)
        throw std::runtime_error(
            "Dump size mismatch for " + path + ": file holds " +
            std::to_string(static_cast<long long>(file_bytes) / 4) +
            " floats, caller asked for " + std::to_string(num_elements) +
            " (regenerate the dumps -- the model geometry moved)");

    file.seekg(0, std::ios::beg);
    std::vector<float> buffer(num_elements);
    file.read(reinterpret_cast<char*>(buffer.data()), want_bytes);
    if (!file)
        throw std::runtime_error("Short read from dump file: " + path);
    return buffer;
}

// Convenience overload for the <dir>/<name> pattern every suite uses.
inline std::vector<float> load_golden_bin(const std::string& dir, const std::string& name,
                                          size_t num_elements) {
    return load_golden_bin(dir + "/" + name, num_elements);
}

// int32 sidecar (token id lists): same robustness, different element type. Length
// is taken FROM the file, because the caller generally does not know the prompt
// length until it reads it.
inline std::vector<int> load_golden_i32(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        throw std::runtime_error("Dump file not found: " + path);
    const std::streamsize bytes = file.tellg();
    if (bytes <= 0 || bytes % static_cast<std::streamsize>(sizeof(int32_t)) != 0)
        throw std::runtime_error("Not an int32 dump (" + std::to_string(bytes) +
                                 " bytes): " + path);
    file.seekg(0, std::ios::beg);
    std::vector<int> out(static_cast<size_t>(bytes) / sizeof(int32_t));
    file.read(reinterpret_cast<char*>(out.data()), bytes);
    if (!file)
        throw std::runtime_error("Short read from dump file: " + path);
    return out;
}

// ---------------------------------------------------------------------------
// 2. Divergence metrics
// ---------------------------------------------------------------------------
struct Telemetry {
    double rmse = 0.0;
    double max_err = 0.0;
    double cosine_sim = 0.0;
    size_t max_err_idx = 0;    // where the worst element sits: the first clue on a divergence
    size_t size = 0;
};

// Accumulates in double: the vectors are fp32 and can be 150k wide (logits), so
// a float accumulator loses the tail of the sum of squares.
inline Telemetry compute_telemetry(const std::vector<float>& golden,
                                   const std::vector<float>& actual) {
    Telemetry t;
    t.size = golden.size();
    if (golden.size() != actual.size())
        throw std::runtime_error("compute_telemetry: size mismatch (" +
                                 std::to_string(golden.size()) + " vs " +
                                 std::to_string(actual.size()) + ")");
    if (t.size == 0) return t;

    double sum_sq = 0.0, dot = 0.0, norm_g = 0.0, norm_a = 0.0;
    for (size_t i = 0; i < t.size; ++i) {
        const double g = golden[i];
        const double a = actual[i];
        const double err = std::abs(g - a);
        sum_sq += err * err;
        if (err > t.max_err) {
            t.max_err = err;
            t.max_err_idx = i;
        }
        dot += g * a;
        norm_g += g * g;
        norm_a += a * a;
    }
    t.rmse = std::sqrt(sum_sq / static_cast<double>(t.size));
    const double denom = std::sqrt(norm_g) * std::sqrt(norm_a);
    t.cosine_sim = (denom > 1e-9) ? (dot / denom) : 0.0;
    return t;
}

// Device-pointer form: synchronizes, downloads `size` floats from `d_actual` and
// scores them. This is the call the suites actually make.
inline Telemetry compute_telemetry(const std::vector<float>& golden,
                                   const float* d_actual, size_t size) {
    std::vector<float> host(size);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host.data(), d_actual, size * sizeof(float),
                          cudaMemcpyDeviceToHost));
    return compute_telemetry(golden, host);
}

inline std::vector<float> download(const float* d_tensor, size_t size) {
    std::vector<float> host(size);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host.data(), d_tensor, size * sizeof(float),
                          cudaMemcpyDeviceToHost));
    return host;
}

// ---------------------------------------------------------------------------
// 3. Entropy of a probability row: -sum p ln p, in nats.
// ---------------------------------------------------------------------------
// `length <= 1` returns 0.0 -- a single-element distribution carries no
// uncertainty, and for attention that is exactly the pos=0 decode case (one key,
// p == 1), which is why a single-token probe cannot measure attention entropy at
// all. Non-positive entries are skipped rather than producing -inf: a softmax row
// can underflow to 0 and the limit of p ln p there is 0.
inline double compute_entropy(const float* probs, size_t length) {
    if (length <= 1) return 0.0;
    double h = 0.0;
    for (size_t i = 0; i < length; ++i) {
        const double p = static_cast<double>(probs[i]);
        if (p > 0.0) h -= p * std::log(p);
    }
    return h;
}

inline double compute_entropy(const std::vector<float>& probs) {
    return compute_entropy(probs.data(), probs.size());
}

// Softmax of a score row, in double, max-shifted. Used to turn dumped attention
// scores (or logits) into the distribution compute_entropy consumes.
inline std::vector<float> softmax(const std::vector<float>& scores) {
    std::vector<float> p(scores.size());
    if (scores.empty()) return p;
    const double m = *std::max_element(scores.begin(), scores.end());
    double sum = 0.0;
    for (size_t i = 0; i < scores.size(); ++i) {
        const double e = std::exp(static_cast<double>(scores[i]) - m);
        p[i] = static_cast<float>(e);
        sum += e;
    }
    if (sum > 0.0)
        for (auto& v : p) v = static_cast<float>(v / sum);
    return p;
}

// ---------------------------------------------------------------------------
// 4. Layer-by-layer telemetry table
// ---------------------------------------------------------------------------
class TelemetryTablePrinter {
public:
    explicit TelemetryTablePrinter(std::ostream& os = std::cout, double diverge_below = 0.99)
        : m_os(os), m_diverge_below(diverge_below) {}

    void header(const char* title = nullptr) {
        if (title) m_os << "\n[Integration] " << title << "\n";
        m_os << std::string(kWidth, '-') << "\n"
             << std::left
             << std::setw(6)  << "Layer"
             << std::setw(16) << "Stage"
             << std::setw(13) << "RMSE"
             << std::setw(13) << "MaxErr"
             << std::setw(10) << "@idx"
             << std::setw(14) << "CosineSim"
             << "Status\n"
             << std::string(kWidth, '-') << "\n";
    }

    void row(int layer, const std::string& stage, const Telemetry& t) {
        m_os << std::left
             << std::setw(6)  << layer
             << std::setw(16) << stage
             << std::fixed << std::setprecision(6)
             << std::setw(13) << t.rmse
             << std::setw(13) << t.max_err
             << std::setw(10) << t.max_err_idx
             << std::setprecision(8)
             << std::setw(14) << t.cosine_sim
             << (t.cosine_sim < m_diverge_below ? "DIVERGING" : "STABLE") << "\n";
    }

    void skipped(int layer, const std::string& stage,
                 const char* why = "[dump unavailable for telemetry]") {
        m_os << std::left
             << std::setw(6)  << layer
             << std::setw(16) << stage
             << std::setw(50) << why
             << "SKIPPED\n";
    }

    void note(const std::string& text) { m_os << "  " << text << "\n"; }

    void footer() { m_os << std::string(kWidth, '-') << "\n"; }

    // The DRY workhorse: load the dump, score the device tensor against it, print
    // the row, and hand the metrics back so the caller can ASSERT on them. A
    // missing dump prints SKIPPED and returns nullopt -- absent telemetry is not a
    // failure (the suites are written to run against a partial dump set), but a
    // caller that requires a stage can treat nullopt as one.
    std::optional<Telemetry> probe(const std::string& dumps_dir, const std::string& dump_file,
                                   const float* d_tensor, size_t num_elements,
                                   int layer, const std::string& stage) {
        std::vector<float> golden;
        try {
            golden = load_golden_bin(dumps_dir, dump_file, num_elements);
        } catch (const std::exception& e) {
            m_last_error = e.what();
            skipped(layer, stage);
            return std::nullopt;
        }
        const Telemetry t = compute_telemetry(golden, d_tensor, num_elements);
        row(layer, stage, t);
        return t;
    }

    const std::string& last_error() const { return m_last_error; }

private:
    static constexpr int kWidth = 88;
    std::ostream& m_os;
    double m_diverge_below;
    std::string m_last_error;
};

// ---------------------------------------------------------------------------
// 5. Final-logits convergence verdict
// ---------------------------------------------------------------------------
inline std::vector<int> topk_indices(const std::vector<float>& v, int k) {
    k = std::max(1, std::min<int>(k, static_cast<int>(v.size())));
    std::vector<int> idx(v.size());
    for (size_t i = 0; i < v.size(); ++i) idx[i] = static_cast<int>(i);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return v[a] > v[b]; });
    idx.resize(static_cast<size_t>(k));
    return idx;
}

struct LogitsVerdict {
    double cosine = 0.0;
    int    golden_top1 = -1;
    int    engine_top1 = -1;
    bool   top1_exact = false;          // engine top-1 == golden top-1
    bool   top1_in_engine_topk = false; // golden top-1 anywhere in the engine's top-k
    std::vector<int>   golden_topk, engine_topk;
    std::vector<float> engine_logits;
};

// Prints the metric plus both top-k rows, then returns an AssertionResult that is
// false only when BOTH acceptance routes fail: the cosine floor and the top-k
// containment. Two routes because a quantized stack drifts in magnitude while
// keeping the ranking -- which is what actually matters for greedy decode.
//
// Use as:  ASSERT_TRUE(engine_test::assert_logits_convergence(...));
inline ::testing::AssertionResult assert_logits_convergence(
    const float* d_logits, const std::vector<float>& golden_logits,
    size_t vocab_size, double min_cosine = 0.95, int top_k = 5,
    LogitsVerdict* out = nullptr)
{
    if (golden_logits.size() != vocab_size)
        return ::testing::AssertionFailure()
               << "golden logits hold " << golden_logits.size()
               << " values but vocab_size is " << vocab_size;

    LogitsVerdict v;
    v.engine_logits = download(d_logits, vocab_size);
    const Telemetry t = compute_telemetry(golden_logits, v.engine_logits);
    v.cosine = t.cosine_sim;
    v.golden_topk = topk_indices(golden_logits, top_k);
    v.engine_topk = topk_indices(v.engine_logits, top_k);
    v.golden_top1 = v.golden_topk.front();
    v.engine_top1 = v.engine_topk.front();
    v.top1_exact = (v.golden_top1 == v.engine_top1);
    v.top1_in_engine_topk =
        std::find(v.engine_topk.begin(), v.engine_topk.end(), v.golden_top1) != v.engine_topk.end();

    std::cout << std::fixed << std::setprecision(8)
              << "  [METRIC] logits cosine similarity: " << v.cosine
              << "  (rmse " << std::setprecision(6) << t.rmse << ")\n";
    auto print_topk = [&](const char* title, const std::vector<int>& idx) {
        std::cout << "  [METRIC] " << title << " top-" << top_k << ":";
        for (int tok : idx)
            std::cout << " " << tok << " (g=" << std::setprecision(3) << golden_logits[tok]
                      << "/e=" << v.engine_logits[tok] << ")";
        std::cout << "\n";
    };
    print_topk("PyTorch golden", v.golden_topk);
    print_topk("engine        ", v.engine_topk);

    const bool passed = (v.cosine >= min_cosine) || v.top1_in_engine_topk;
    if (out) *out = v;

    if (passed)
        return ::testing::AssertionSuccess()
               << "logits cosine " << v.cosine << " (top-1 exact: "
               << (v.top1_exact ? "yes" : "no") << ")";
    return ::testing::AssertionFailure()
           << "convergence FAILED: cosine " << std::setprecision(8) << v.cosine
           << " < " << min_cosine << " and golden top-1 token " << v.golden_top1
           << " is absent from the engine's top-" << top_k;
}

}  // namespace engine_test
