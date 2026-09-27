// Split-K (Flash-Decoding) decode attention: latency vs. the single-block kernel
// across the context lengths the runtime ladder brackets, at GLM-4-9B geometry.
//
// WHAT THIS MEASURES AND WHY. Decode attention is launched once per layer per
// token, with grid == q_heads. At GLM-4-9B geometry that is 32 blocks; this part
// reports its SM count in the header, and the gap between the two is the whole
// motivation. Split-K widens the grid to [q_heads, S] over the context dimension,
// so the tables below answer two questions: how much latency the widening buys at
// each context length, and where adding slices stops paying (the reduction pass
// and the per-block Q load are fixed costs the slices do not amortize).
//
// HOW IT MEASURES. Every number is the minimum of several ROUND-ROBIN passes over
// all candidates -- see measure_interleaved(). This part idles at a few percent of
// peak bandwidth on these kernels, so it never leaves a low power state and its
// clocks wander on a timescale longer than one timed loop. Measuring candidates in
// separate back-to-back passes hands whichever one ran during a boost window a 2-3x
// advantage it does not really have (observed before the interleaving went in).
//
// Run this from the RELEASE tree. The Debug build compiles CUDA with -G -Od, so a
// microsecond figure taken from it measures the build flags:
//   ctest --preset benchmark-release
#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

#include "common/benchmark_utils.h"
#include "kernels/attention.cuh"

namespace {

class AttentionSplitKBenchmark : public ::testing::Test {};

constexpr size_t kHeadDim = 128;
constexpr size_t kQHeads  = 32;    // GLM-4-9B
constexpr size_t kMaxSeq  = 4096;

// The context lengths the task brackets. Each is a full cache, i.e. pos = N - 1.
constexpr int kContextLengths[] = {256, 512, 1024, 2048, 4096};

// Bytes the kernels must touch for one decode at context length N: Q in, K and V
// rows [0, N) in, O out, plus the scratchpad round-trip (written by phase 1, read
// by phase 2 -- it is sized to stay in L2, so counting it is an upper bound on DRAM
// traffic, not a claim about it).
double traffic_bytes(int n, size_t kv_heads, int splits) {
    const double q_bytes  = (double)kQHeads * kHeadDim * sizeof(float);
    const double kv_bytes = 2.0 * (double)kv_heads * n * kHeadDim * sizeof(float);
    const double scratch  = splits > 1
        ? 2.0 * (double)blackwell::attn::split_k_scratch_floats(kQHeads, kHeadDim, splits)
              * sizeof(float)
        : 0.0;
    return q_bytes * 2 + kv_bytes + scratch;
}

// One measured configuration.
struct Candidate {
    std::string           label;
    int                   splits;     // 1 == the single-block kernel
    std::function<void()> launch;
    std::vector<double>   passes;     // one timing per round-robin pass

    double median_us() const {
        std::vector<double> v = passes;
        std::sort(v.begin(), v.end());
        return v.empty() ? 0.0 : v[v.size() / 2];
    }
    double min_us() const {
        return passes.empty() ? 0.0 : *std::min_element(passes.begin(), passes.end());
    }
    // Spread of the passes, as max/min. A value near 1 means the configuration
    // timed reproducibly; well above it means the number below is soft.
    double spread() const {
        if (passes.empty()) return 0.0;
        const auto mm = std::minmax_element(passes.begin(), passes.end());
        return *mm.second / *mm.first;
    }
};

// Times every candidate round-robin, `repeats` passes over, recording each pass.
//
// The INTERLEAVING is the point (see the file header). The MEDIAN is the headline
// statistic rather than the minimum: on this part these kernels time reproducibly
// to within a few percent most passes but occasionally land 2-3x fast, and a
// minimum-of-N would promote exactly those flukes into the table. Clocks were
// measured pinned at ~2.87 GHz throughout a full sweep, so the flukes are not the
// part boosting -- they are the Windows WDDM submission path, which dominates
// these launches outright (a 256-token split-16 decode is ~1 us of kernel work
// inside a ~24 us measured launch pair). The min column is kept beside the median
// so the gap is visible instead of silently chosen.
void measure_interleaved(std::vector<Candidate>& candidates, int repeats = 9) {
    std::vector<int> iters(candidates.size());
    for (size_t i = 0; i < candidates.size(); ++i) {
        const double probe_us =
            bench_utils::time_kernel_us(candidates[i].launch, /*warmup=*/5, /*iters=*/20);
        iters[i] = std::clamp(static_cast<int>(20000.0 / std::max(probe_us, 1.0)), 20, 1000);
        candidates[i].passes.reserve(repeats);
    }
    for (int r = 0; r < repeats; ++r)
        for (size_t i = 0; i < candidates.size(); ++i)
            candidates[i].passes.push_back(
                bench_utils::time_kernel_us(candidates[i].launch, /*warmup=*/5, iters[i]));
}

struct Buffers {
    CudaVector<float> q, k_cache, v_cache, o, scratch;

    explicit Buffers(size_t kv_heads)
        : q(kQHeads * kHeadDim),
          k_cache(kv_heads * kMaxSeq * kHeadDim),
          v_cache(kv_heads * kMaxSeq * kHeadDim),
          o(kQHeads * kHeadDim),
          scratch(blackwell::attn::split_k_scratch_floats(kQHeads, kHeadDim)) {
        // Any finite bit pattern will do -- the kernels' cost is data-independent.
        CUDA_CHECK(cudaMemset(q.d_ptr, 0x3D, q.num_elements * sizeof(float)));
        CUDA_CHECK(cudaMemset(k_cache.d_ptr, 0x3C, k_cache.num_elements * sizeof(float)));
        CUDA_CHECK(cudaMemset(v_cache.d_ptr, 0x3C, v_cache.num_elements * sizeof(float)));
    }
};

// Builds the candidate list for one context length: the single-block baseline
// first, then each split factor.
std::vector<Candidate> candidates_for(Buffers& buf, size_t kv_heads, int n,
                                      const std::vector<int>& split_factors) {
    const int pos = n - 1;
    std::vector<Candidate> out;

    out.push_back({"single-block", 1, [&buf, kv_heads, pos] {
        launch_attention_decoding_kernel(buf.q, buf.k_cache, buf.v_cache, buf.o, pos,
                                         kQHeads, kv_heads, kHeadDim, kMaxSeq);
    }});
    for (int splits : split_factors) {
        out.push_back({"split-K S=" + std::to_string(splits), splits,
                       [&buf, kv_heads, pos, splits] {
            launch_attention_decoding_split_k(buf.q, buf.k_cache, buf.v_cache, buf.o,
                                              buf.scratch, splits, pos, kQHeads,
                                              kv_heads, kHeadDim, kMaxSeq);
        }});
    }
    return out;
}

void print_device_header(size_t kv_heads) {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("\nSplit-K decode attention -- %zu query heads / %zu KV heads, head_dim %zu\n",
           kQHeads, kv_heads, kHeadDim);
    printf("Device: %s, %d SMs, %.0f GB/s peak. Single-block grid = %zu blocks"
           " (%.0f%% of the SMs); split-K grid = %zu x S.\n\n",
           prop.name, prop.multiProcessorCount, bench_utils::device_peak_bandwidth_gbs(),
           kQHeads, 100.0 * (double)kQHeads / prop.multiProcessorCount, kQHeads);
}

}  // namespace

// The headline table: for each context length, the single-block kernel against
// every split factor, so the crossover is visible rather than asserted.
TEST_F(AttentionSplitKBenchmark, SplitFactorSweepVsSingleBlock) {
    const std::vector<int> split_factors = {2, 4, 8, 16};

    for (size_t kv_heads : {size_t{2}, size_t{4}}) {
        print_device_header(kv_heads);
        Buffers buf(kv_heads);

        printf("| %6s | %-14s | %10s | %8s | %6s | %6s | %8s | %7s |\n",
               "ctx N", "kernel", "median(us)", "min(us)", "max/min", "GB/s", "speedup", "grid");
        printf("|--------|----------------|------------|----------|---------|--------|----------|---------|\n");

        const double peak = bench_utils::device_peak_bandwidth_gbs();
        for (int n : kContextLengths) {
            auto cands = candidates_for(buf, kv_heads, n, split_factors);
            measure_interleaved(cands);

            const int ladder = blackwell::attn::select_split_k(n);
            const double base_us = cands.front().median_us();
            bool first = true;
            for (const Candidate& c : cands) {
                const double med = c.median_us();
                const double gbs = traffic_bytes(n, kv_heads, c.splits) / (med * 1e-6) / 1e9;
                const std::string label = c.label + (c.splits == ladder ? " *" : "");
                char grid[16];
                if (c.splits == 1) snprintf(grid, sizeof grid, "%zu", kQHeads);
                else               snprintf(grid, sizeof grid, "%zux%d", kQHeads, c.splits);
                printf("| %6s | %-14s | %10.2f | %8.2f | %6.2fx | %6.1f | %7.2fx | %7s |\n",
                       first ? std::to_string(n).c_str() : "", label.c_str(), med,
                       c.min_us(), c.spread(), gbs, base_us / med, grid);
                first = false;

                const std::string key = "kv" + std::to_string(kv_heads) + "_n" +
                                        std::to_string(n) + "_S" + std::to_string(c.splits);
                ::testing::Test::RecordProperty(key + "_median_us", std::to_string(med));
                ::testing::Test::RecordProperty(key + "_min_us", std::to_string(c.min_us()));
            }
        }
        printf("\n(* = the factor attn::select_split_k picks. GB/s is from the median and"
               " counts the L2-resident scratchpad round-trip, so it is an upper bound;"
               " device peak is %.0f GB/s.)\n\n", peak);
    }
}

// What the engine actually gets: the ladder's choice at each context length, and
// the per-token saving once the per-layer launch is multiplied out. Attention runs
// once per layer per token, so a per-launch microsecond is worth num_layers of it.
TEST_F(AttentionSplitKBenchmark, LadderChoicePerTokenSaving) {
    constexpr int kGlm4Layers = 40;   // GLM-4-9B decoder depth
    constexpr size_t kv_heads = 4;

    print_device_header(kv_heads);
    Buffers buf(kv_heads);

    printf("| %6s | %2s | %11s | %11s | %8s | %13s | %13s |\n",
           "ctx N", "S", "single(us)", "split-K(us)", "speedup", "attn/token", "saved/token");
    printf("|--------|----|-------------|-------------|----------|---------------|---------------|\n");

    for (int n : kContextLengths) {
        const int splits = blackwell::attn::select_split_k(n);
        auto cands = candidates_for(buf, kv_heads, n, {splits});
        measure_interleaved(cands);

        const double base_us  = cands[0].median_us();
        const double split_us = cands[1].median_us();
        printf("| %6d | %2d | %11.2f | %11.2f | %7.2fx | %10.3f ms | %10.3f ms |\n", n, splits,
               base_us, split_us, base_us / split_us,
               split_us * kGlm4Layers / 1000.0,
               (base_us - split_us) * kGlm4Layers / 1000.0);
        ::testing::Test::RecordProperty("ladder_n" + std::to_string(n) + "_speedup",
                                        std::to_string(base_us / split_us));
    }
    printf("\n(attn/token and saved/token = the per-launch figures x %d layers:"
           " what one decode step spends on attention, and what it gets back.)\n\n",
           kGlm4Layers);
}

// Occupancy pressure, as the driver reports it for this device. A register spill
// (spill > 0) in phase 1 would quietly undo the latency win, so this is reported
// from the driver rather than inferred from the source.
TEST_F(AttentionSplitKBenchmark, RegisterAndSharedMemoryPressure) {
    const blackwell::attn::SplitKKernelStats s =
        blackwell::attn::query_split_k_kernel_stats();

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    int regs_per_block = 0, shared_per_block = 0, max_threads_per_sm = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&regs_per_block, cudaDevAttrMaxRegistersPerBlock, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&shared_per_block, cudaDevAttrMaxSharedMemoryPerBlock, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&max_threads_per_sm,
                                      cudaDevAttrMaxThreadsPerMultiProcessor, dev));

    printf("\nDecode-attention kernel resource footprint (block = %d threads)\n\n",
           blackwell::attn::kSplitKBlockSize);
    printf("| %-26s | %5s | %10s | %8s | %12s |\n",
           "kernel", "regs", "shared(B)", "spill(B)", "blocks/SM*");
    printf("|----------------------------|-------|------------|----------|--------------|\n");

    const auto row = [&](const char* name, int regs, size_t shared, size_t local) {
        // Register-limited blocks per SM, the only limit these kernels can hit:
        // shared use is a handful of bytes and there is no dynamic allocation.
        const int threads  = blackwell::attn::kSplitKBlockSize;
        const int by_regs  = regs > 0 ? regs_per_block / (regs * threads) : 0;
        const int by_slots = max_threads_per_sm / threads;
        printf("| %-26s | %5d | %10zu | %8zu | %12d |\n", name, regs, shared, local,
               by_regs > 0 ? std::min(by_regs, by_slots) : by_slots);
    };
    row("split_k_partial (phase 1)", s.partial_num_regs, s.partial_shared_bytes,
        s.partial_local_bytes);
    row("split_k_reduce  (phase 2)", s.reduce_num_regs, s.reduce_shared_bytes,
        s.reduce_local_bytes);
    row("single-block (baseline)", s.legacy_num_regs, s.legacy_shared_bytes,
        s.legacy_local_bytes);
    printf("\n(* register-limited occupancy; device caps: %d regs/block, %d B shared/block,"
           " %d threads/SM.)\n\n", regs_per_block, shared_per_block, max_threads_per_sm);

    // Spills are the one outcome that would invalidate the whole approach.
    EXPECT_EQ(s.partial_local_bytes, 0u) << "phase 1 spills registers to local memory";
    EXPECT_EQ(s.reduce_local_bytes, 0u) << "phase 2 spills registers to local memory";
    // Shared use must stay negligible -- it is what keeps occupancy register-bound.
    EXPECT_LT(s.partial_shared_bytes, 1024u);
    EXPECT_LT(s.reduce_shared_bytes, 1024u);
}
