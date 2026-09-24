// GLM-4-9B-Chat-1M multi-turn chat regression, driven by token ids.
//
// Tokenization here comes from HuggingFace, through
// scripts/glm4_chat_regression.py: this runner speaks only ids -- it reads
// jobs.tsv, generates greedily, and writes outputs.tsv for the script to decode.
// Everything between the two files -- prefill, sampling, the KV cache, the
// offload pipeline -- is the engine under test.
//
// The engine can now tokenize GLM-4 itself (TiktokenTokenizer + GlmChatTemplate;
// see Glm4ChatRegression.TokenizerFactoryServesGlm4Checkpoint below, and the
// Python-free end-to-end run in test_glm4_native_chat.cpp). This file is kept
// deliberately: driving the same four prompts from HuggingFace's own tokenizer
// is what makes the native one's agreement evidence rather than self-report.
//
// THE POINT OF THE MULTI-TURN CASE. A turn with reset=0 does NOT re-prefill: its
// delta is appended at the position where the previous turn's generation stopped, so
// the model must answer from KV state alone, with RoPE phase continuing monotonically
// across the turn boundary. A re-prefilling harness would pass even with the cache
// completely broken.
//
// Requires (SKIPs otherwise):
//   - BLACKWELL_GLM4_CHAT_DIR (or out/glm4_chat_regression) holding jobs.tsv,
//     written by `python scripts/glm4_chat_regression.py --emit`
//   - the HF-native checkpoint (BLACKWELL_GLM4_INDEX); see test_glm4_engine.cpp
#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common.h"
#include "common/engine_test_harness.h"
#include "blackwell/engine.h"
#include "blackwell/tokenizer.h"
#include "engine_impl.h"

namespace {

std::string model_index_path() {
    return engine_test::env_or(
        "BLACKWELL_GLM4_INDEX",
        "F:/AI/models/GLM-4-9B-Chat-1M-hf/model.safetensors.index.json");
}

std::string model_dir() {
    const std::string idx = model_index_path();
    const size_t slash = idx.find_last_of("/\\");
    return (slash == std::string::npos) ? std::string(".") : idx.substr(0, slash);
}

std::string chat_dir() {
    return engine_test::env_or("BLACKWELL_GLM4_CHAT_DIR",
                               "D:/Projects/BlackwellLLM/out/glm4_chat_regression");
}

size_t gpu_layers() {
    const std::string v = engine_test::env_or("BLACKWELL_GLM4_GPU_LAYERS", "16");
    return static_cast<size_t>(std::strtoul(v.c_str(), nullptr, 10));
}

constexpr size_t kMaxSeqLen = 1024;   // longest case: 2 turns + both generations

struct Job {
    std::string case_name;
    int  turn = 0;
    bool reset = false;          // true: fresh sequence from position 0
    int  max_new = 0;
    std::vector<int> prompt_ids;
};

std::vector<int> parse_csv_ids(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ','))
        if (!tok.empty()) out.push_back(std::stoi(tok));
    return out;
}

// jobs.tsv:  case \t turn \t reset \t max_new \t id,id,...
// plus a leading "# stop_ids\t<csv>" comment line.
bool load_jobs(const std::string& path, std::vector<Job>& jobs, std::vector<int>& stop_ids) {
    std::ifstream f(path);
    if (!f.is_open()) return false;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (line[0] == '#') {
            const size_t tab = line.find('\t');
            if (tab != std::string::npos && line.find("stop_ids") != std::string::npos)
                stop_ids = parse_csv_ids(line.substr(tab + 1));
            continue;
        }
        std::stringstream ss(line);
        std::string case_name, turn, reset, max_new, ids;
        if (!std::getline(ss, case_name, '\t') || !std::getline(ss, turn, '\t') ||
            !std::getline(ss, reset, '\t')     || !std::getline(ss, max_new, '\t') ||
            !std::getline(ss, ids))
            continue;
        Job j;
        j.case_name = case_name;
        j.turn = std::stoi(turn);
        j.reset = (std::stoi(reset) != 0);
        j.max_new = std::stoi(max_new);
        j.prompt_ids = parse_csv_ids(ids);
        jobs.push_back(std::move(j));
    }
    return true;
}

bool is_stop(const std::vector<int>& stop_ids, int token) {
    for (int s : stop_ids) if (s == token) return true;
    return false;
}

size_t vram_used_bytes() {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return 0;
    return total_b - free_b;
}

using clock_t_ = std::chrono::steady_clock;
double ms_since(const clock_t_::time_point& t0) {
    return std::chrono::duration<double, std::milli>(clock_t_::now() - t0).count();
}

}  // namespace

// TokenizerFactory now serves this checkpoint natively (TiktokenTokenizer +
// GlmChatTemplate). This test pins that: it is the precondition that lets
// test_glm4_native_chat.cpp run the whole chat in C++, and the reason the
// Python bridge below is a cross-check rather than the only way in.
TEST(Glm4ChatRegression, TokenizerFactoryServesGlm4Checkpoint) {
    if (!engine_test::file_exists(model_index_path()))
        GTEST_SKIP() << "GLM-4 checkpoint not found at " << model_index_path();

    const std::string dir = model_dir();
    ASSERT_TRUE(engine_test::file_exists(dir + "/tokenizer.model"))
        << "GLM-4 is expected to ship a tiktoken-format tokenizer.model";

    auto tok = blackwell::TokenizerFactory::create(dir);
    ASSERT_NE(tok, nullptr);

    // The tiktoken branch, not the byte-level-BPE one: 151329 ranks + 14
    // special tokens from added_tokens_decoder.
    EXPECT_EQ(tok->vocab_size(), 151329u + 14u);
    ASSERT_NE(tok->chat_template(), nullptr);
    EXPECT_EQ(tok->chat_template()->name(), "glm4");

    // The stop set the Python bridge writes into jobs.tsv, resolved natively.
    EXPECT_EQ(tok->special_tokens().stop_ids, (std::vector<int>{151329, 151336, 151338}));
}

TEST(Glm4ChatRegression, FourPromptGreedyChat) {
    using namespace engine_test;

    if (!file_exists(model_index_path()))
        GTEST_SKIP() << "GLM-4 checkpoint not found at " << model_index_path()
                     << " (convert with scripts/convert_glm4_thudm_to_hf.py)";

    const std::string cd = chat_dir();
    std::vector<Job> jobs;
    std::vector<int> stop_ids;
    if (!load_jobs(cd + "/jobs.tsv", jobs, stop_ids))
        GTEST_SKIP() << "jobs.tsv not found in " << cd
                     << " (run: python scripts/glm4_chat_regression.py --emit)";
    ASSERT_FALSE(jobs.empty());
    ASSERT_FALSE(stop_ids.empty()) << "jobs.tsv carried no stop_ids header";

    std::cout << "\n[Chat] " << jobs.size() << " turn(s), stop ids:";
    for (int s : stop_ids) std::cout << " " << s;
    std::cout << "\n[Chat] engine: " << gpu_layers()
              << " of 40 layers VRAM-resident, greedy (temperature 0)\n";

    const size_t vram_before = vram_used_bytes();
    const auto load_t0 = clock_t_::now();
    BlackwellEngine engine(model_index_path(), kMaxSeqLen, gpu_layers(),
                           BlackwellEngine::KVCacheMode::Continuous);
    const double load_ms = ms_since(load_t0);
    const size_t vram_after_load = vram_used_bytes();

    std::ofstream out(cd + "/outputs.tsv", std::ios::trunc);
    ASSERT_TRUE(out.is_open()) << "cannot write " << cd << "/outputs.tsv";

    size_t peak_vram = vram_after_load;
    int pos = 0;                      // running position across a case's turns
    std::string current_case;
    double total_decode_ms = 0.0;
    int total_gen = 0;

    for (const Job& j : jobs) {
        if (j.reset) {
            // A fresh conversation: the continuous KV is position-addressed, so
            // restarting at 0 overwrites the previous case's slots in place.
            engine.reset_state(0);
            pos = 0;
            current_case = j.case_name;
        }
        ASSERT_EQ(j.case_name, current_case)
            << "turn " << j.turn << " of " << j.case_name
            << " arrived without a reset and without a preceding turn of its case";
        ASSERT_LT(pos + j.prompt_ids.size() + static_cast<size_t>(j.max_new), kMaxSeqLen)
            << "case " << j.case_name << " would overrun the " << kMaxSeqLen
            << "-token context";

        std::cout << "[Chat] " << j.case_name << " turn " << j.turn << ": prefill "
                  << j.prompt_ids.size() << " tokens at pos " << pos
                  << (j.reset ? " (fresh)" : " (KV append)") << std::flush;

        // ---- prefill: every prompt token but the last only needs its KV row ----
        const auto prefill_t0 = clock_t_::now();
        int next = -1;
        for (size_t i = 0; i < j.prompt_ids.size(); ++i) {
            const bool last = (i + 1 == j.prompt_ids.size());
            if (last) {
                const auto st = engine.forward_status(j.prompt_ids[i], pos, 0.0f, 1.0f,
                                                      /*seq_id=*/0, &next);
                ASSERT_EQ(st, blackwell::EngineStatus::Success)
                    << "prefill(last) " << j.case_name << " turn " << j.turn << ": "
                    << blackwell::to_string(st);
            } else {
                const auto st = engine.get_impl()->run_token(j.prompt_ids[i], pos,
                                                             /*seq_id=*/0,
                                                             /*want_logits=*/false);
                ASSERT_EQ(st, blackwell::EngineStatus::Success)
                    << "prefill " << j.case_name << " turn " << j.turn << " at pos "
                    << pos << ": " << blackwell::to_string(st);
            }
            ++pos;
        }
        const double prefill_ms = ms_since(prefill_t0);
        peak_vram = std::max(peak_vram, vram_used_bytes());

        // ---- greedy decode ----------------------------------------------------
        std::vector<int> generated;
        std::string stop_reason = "max_new";
        const auto decode_t0 = clock_t_::now();
        for (int n = 0; n < j.max_new; ++n) {
            if (is_stop(stop_ids, next)) { stop_reason = "stop_token"; break; }
            generated.push_back(next);
            if (static_cast<size_t>(pos) + 1 >= kMaxSeqLen) {
                stop_reason = "context_full";
                break;
            }
            int following = -1;
            const auto st = engine.forward_status(next, pos, 0.0f, 1.0f, /*seq_id=*/0,
                                                  &following);
            ASSERT_EQ(st, blackwell::EngineStatus::Success)
                << "decode " << j.case_name << " turn " << j.turn << " step " << n
                << ": " << blackwell::to_string(st);
            ++pos;
            next = following;
        }
        const double decode_ms = ms_since(decode_t0);
        peak_vram = std::max(peak_vram, vram_used_bytes());
        total_decode_ms += decode_ms;
        total_gen += static_cast<int>(generated.size());

        const double tps = decode_ms > 0.0
                             ? generated.size() / (decode_ms / 1000.0) : 0.0;
        std::cout << " -> generated " << generated.size() << " tokens, " << stop_reason
                  << ", " << std::fixed << std::setprecision(2) << tps << " tok/s\n";

        out << j.case_name << "\t" << j.turn << "\t";
        for (size_t i = 0; i < generated.size(); ++i)
            out << (i ? "," : "") << generated[i];
        out << "\t" << stop_reason << "\t" << j.prompt_ids.size() << "\t"
            << generated.size() << "\t" << std::fixed << std::setprecision(1)
            << prefill_ms << "\t" << decode_ms << "\n";

        // A case's answer must not be empty -- that would mean the very first
        // sampled token was a stop token, i.e. a broken prompt or a broken prefill.
        EXPECT_FALSE(generated.empty())
            << j.case_name << " turn " << j.turn << " generated nothing (first "
            << "sampled token " << next << " was already a stop token)";
    }

    const double overall_tps = total_decode_ms > 0.0
                                 ? total_gen / (total_decode_ms / 1000.0) : 0.0;
    out << "# telemetry  weights load " << std::fixed << std::setprecision(1) << load_ms
        << " ms | peak VRAM " << std::setprecision(2)
        << peak_vram / (1024.0 * 1024.0) << " MiB (baseline "
        << vram_before / (1024.0 * 1024.0) << " MiB, after load "
        << vram_after_load / (1024.0 * 1024.0) << " MiB) | " << gpu_layers()
        << "/40 layers resident | " << total_gen << " tokens at "
        << std::setprecision(2) << overall_tps << " tok/s aggregate\n";
    out.close();

    std::cout << "[Chat] weights load " << std::setprecision(1) << load_ms
              << " ms, peak VRAM " << std::setprecision(2)
              << peak_vram / (1024.0 * 1024.0) << " MiB, aggregate "
              << overall_tps << " tok/s over " << total_gen << " tokens\n";
    std::cout << "[Chat] wrote " << cd << "/outputs.tsv -- decode it with "
                 "`python scripts/glm4_chat_regression.py --decode`\n";
}
