// GLM-4-9B-Chat-1M end to end in C++ alone: raw text in, raw text out.
//
// This is the test test_glm4_chat_regression.cpp could not be. That one speaks
// only token ids because the engine could not tokenize GLM-4, so a Python
// bridge (scripts/glm4_chat_regression.py) had to own both ends. With
// TiktokenTokenizer + GlmChatTemplate the whole loop is native: no jobs.tsv, no
// outputs.tsv, no interpreter. What is under test is therefore the *seam* the
// bridge used to hide -- prompt text -> chat template -> ids -> engine -> ids ->
// text -- which is exactly the path a shipping application takes.
//
// It stays a separate file from the bridge test on purpose: running the same
// four prompts through HuggingFace's tokenizer and through ours, and getting the
// same answers, is what makes the native path evidence rather than self-report.
//
// THE MULTI-TURN CASE. Turn 2 of a case does NOT re-prefill: only the new user
// delta is appended, at the position where turn 1's generation stopped. The
// model must answer from KV state alone, with RoPE phase continuing across the
// turn boundary -- a re-prefilling harness would pass with a broken cache.
//
// Requires the HF-native checkpoint (BLACKWELL_GLM4_INDEX); SKIPs otherwise.
#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
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

size_t gpu_layers() {
    const std::string v = engine_test::env_or("BLACKWELL_GLM4_GPU_LAYERS", "16");
    return static_cast<size_t>(std::strtoul(v.c_str(), nullptr, 10));
}

constexpr size_t kMaxSeqLen = 1024;   // longest case: 2 turns + both generations
constexpr int    kMaxNew    = 96;

// The four prompts, as TEXT. Two single-turn cases and one two-turn case whose
// second turn is answerable only from the first turn's KV.
struct Turn {
    const char* user;
    bool        reset;      // true: start a fresh conversation at position 0
};
struct Case {
    const char* name;
    const char* system;     // "" -> no system block
    std::vector<Turn> turns;
};

const std::vector<Case>& cases() {
    static const std::vector<Case> c{
        {"arithmetic", "You are a helpful assistant. Answer briefly.",
         {{"What is 17 plus 25? Reply with just the number.", true}}},
        {"capital", "",
         {{"What is the capital of France? Answer in one word.", true}}},
        {"followup", "You are a helpful assistant. Answer briefly.",
         {{"My favourite colour is teal. Remember it.", true},
          {"What is my favourite colour?", false}}},
    };
    return c;
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

// Lowercase ASCII copy, so an answer can be matched without pinning the model's
// exact phrasing (greedy decode is deterministic, but the wording is the
// checkpoint's business, not this test's).
std::string lower(std::string s) {
    for (char& ch : s)
        if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch + 32);
    return s;
}

} // namespace

TEST(Glm4NativeChat, FourPromptGreedyChatWithoutPython) {
    using namespace engine_test;

    if (!file_exists(model_index_path()))
        GTEST_SKIP() << "GLM-4 checkpoint not found at " << model_index_path()
                     << " (convert with scripts/convert_glm4_thudm_to_hf.py)";

    // ---- the tokenizer, built from the checkpoint directory alone -----------
    auto tok = blackwell::TokenizerFactory::create(model_dir());
    ASSERT_NE(tok, nullptr);
    ASSERT_NE(tok->chat_template(), nullptr)
        << "GLM-4's tokenizer_config.json should yield a glm4 chat template";
    ASSERT_EQ(tok->chat_template()->name(), "glm4");

    const auto& stop_ids = tok->special_tokens().stop_ids;
    ASSERT_FALSE(stop_ids.empty());

    std::cout << "\n[NativeChat] tokenizer: " << tok->vocab_size() << " tokens, stop ids:";
    for (int s : stop_ids) std::cout << " " << s;
    std::cout << "\n[NativeChat] engine: " << gpu_layers()
              << " of 40 layers VRAM-resident, greedy (temperature 0)\n";

    const auto load_t0 = clock_t_::now();
    BlackwellEngine engine(model_index_path(), kMaxSeqLen, gpu_layers(),
                           BlackwellEngine::KVCacheMode::Continuous);
    const double load_ms = ms_since(load_t0);

    size_t peak_vram = vram_used_bytes();
    double total_decode_ms = 0.0;
    int    total_gen = 0;
    std::vector<std::string> answers;   // flat, in case order then turn order

    for (const Case& c : cases()) {
        // A fresh conversation: the continuous KV is position-addressed, so
        // restarting at 0 overwrites the previous case's slots in place.
        engine.reset_state(0);
        int pos = 0;
        std::vector<blackwell::ChatMessage> history;
        if (*c.system) history.push_back({"system", c.system});

        for (size_t t = 0; t < c.turns.size(); ++t) {
            const Turn& turn = c.turns[t];
            history.push_back({"user", turn.user});

            // The delta this turn contributes. Turn 1 is the whole rendered
            // conversation; later turns are only the new marker + user text +
            // generation cue, because everything before it is already in the KV.
            std::vector<int> delta;
            if (turn.reset) {
                delta = tok->apply_chat_template(history, /*add_generation_prompt=*/true);
            } else {
                delta = tok->encode_chat_message(history.back());
                const auto cue = tok->encode_generation_prompt();
                delta.insert(delta.end(), cue.begin(), cue.end());
            }
            ASSERT_FALSE(delta.empty());
            ASSERT_LT(pos + delta.size() + static_cast<size_t>(kMaxNew), kMaxSeqLen)
                << "case " << c.name << " would overrun the " << kMaxSeqLen
                << "-token context";

            std::cout << "[NativeChat] " << c.name << " turn " << (t + 1) << ": \""
                      << turn.user << "\"\n              prefill " << delta.size()
                      << " tokens at pos " << pos
                      << (turn.reset ? " (fresh)" : " (KV append)") << std::flush;

            // ---- prefill: every token but the last only needs its KV row ----
            int next = -1;
            for (size_t i = 0; i < delta.size(); ++i) {
                const bool last = (i + 1 == delta.size());
                const auto st =
                    last ? engine.forward_status(delta[i], pos, 0.0f, 1.0f, /*seq_id=*/0, &next)
                         : engine.get_impl()->run_token(delta[i], pos, /*seq_id=*/0,
                                                        /*want_logits=*/false);
                ASSERT_EQ(st, blackwell::EngineStatus::Success)
                    << "prefill " << c.name << " turn " << (t + 1) << " at pos " << pos
                    << ": " << blackwell::to_string(st);
                ++pos;
            }

            // ---- greedy decode ----------------------------------------------
            std::vector<int> generated;
            std::string stop_reason = "max_new";
            const auto decode_t0 = clock_t_::now();
            for (int n = 0; n < kMaxNew; ++n) {
                if (tok->is_stop(next)) { stop_reason = "stop_token"; break; }
                generated.push_back(next);
                if (static_cast<size_t>(pos) + 1 >= kMaxSeqLen) {
                    stop_reason = "context_full";
                    break;
                }
                int following = -1;
                const auto st = engine.forward_status(next, pos, 0.0f, 1.0f, /*seq_id=*/0,
                                                      &following);
                ASSERT_EQ(st, blackwell::EngineStatus::Success)
                    << "decode " << c.name << " turn " << (t + 1) << " step " << n
                    << ": " << blackwell::to_string(st);
                ++pos;
                next = following;
            }
            const double decode_ms = ms_since(decode_t0);
            total_decode_ms += decode_ms;
            total_gen += static_cast<int>(generated.size());
            peak_vram = std::max(peak_vram, vram_used_bytes());

            // ---- decode to text, natively -----------------------------------
            const std::string answer = tok->decode(generated, /*render_special=*/false);
            answers.push_back(answer);
            history.push_back({"assistant", answer});

            const double tps =
                decode_ms > 0.0 ? generated.size() / (decode_ms / 1000.0) : 0.0;
            std::cout << " -> " << generated.size() << " tokens, " << stop_reason << ", "
                      << std::fixed << std::setprecision(2) << tps << " tok/s\n"
                      << "              answer: \"" << answer << "\"\n";

            // An empty answer means the very first sampled token was a stop
            // token: a broken prompt, a broken template, or a broken prefill.
            EXPECT_FALSE(generated.empty())
                << c.name << " turn " << (t + 1) << " generated nothing (first sampled "
                << "token " << next << " was already a stop token)";

            // Round-tripping the answer is what proves decode() produced real
            // text and not a byte salad that merely looks printable.
            if (!answer.empty())
                EXPECT_EQ(tok->decode(tok->encode(answer, false), false), answer)
                    << "answer does not survive a re-encode: \"" << answer << "\"";
        }
    }

    ASSERT_EQ(answers.size(), 4u);

    // Content checks. Greedy decode is deterministic, but the phrasing belongs
    // to the checkpoint -- assert the fact each prompt asks for, not the wording.
    EXPECT_NE(answers[0].find("42"), std::string::npos)
        << "arithmetic: expected 17+25=42 somewhere in \"" << answers[0] << "\"";
    EXPECT_NE(lower(answers[1]).find("paris"), std::string::npos)
        << "capital: expected Paris in \"" << answers[1] << "\"";
    EXPECT_NE(lower(answers[3]).find("teal"), std::string::npos)
        << "followup turn 2 answered \"" << answers[3]
        << "\" -- the colour from turn 1 did not survive in the KV cache";

    const double overall_tps =
        total_decode_ms > 0.0 ? total_gen / (total_decode_ms / 1000.0) : 0.0;
    std::cout << "[NativeChat] weights load " << std::fixed << std::setprecision(1)
              << load_ms << " ms, peak VRAM " << std::setprecision(2)
              << peak_vram / (1024.0 * 1024.0) << " MiB, " << gpu_layers()
              << "/40 resident, " << total_gen << " tokens at " << overall_tps
              << " tok/s aggregate\n"
              << "[NativeChat] NOTE: a Debug build compiles CUDA with -G -Od; use the\n"
                 "             x64-Release tree for any figure you intend to quote.\n";
}
