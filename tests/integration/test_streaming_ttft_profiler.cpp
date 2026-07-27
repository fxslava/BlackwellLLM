// =============================================================================
// Streaming TTFT latency profiler + stateless-rewind split test (Tier 2, GPU).
//
// Drives the REAL center-slice streaming stack — rt::RealEngineControl over the
// 8B AWQ backbone + the real Whisper encoder / Ultravox projector — through two
// utterances of real dataset audio, and instruments the exact phase breakdown
//   [VAD trigger -> (edge commit) -> (turn-close template prefill) -> TTFT]
// plus the speech-time cost that CenterSlice moves OFF the critical path
// (encoder + projector + center prefill run as warm hops DURING speech).
//
// Split-test assertions (the stateless-rewind contract):
//   1. finalize_turn (stateless) returns the sequence exactly to the retained
//      base — the generated text leaves zero KV residue.
//   2. A speech restart's do_rewind resolves its keep EXACTLY to
//      history_base_pos() (the system-prefix floor when no history is retained).
//   3. Chunk 2's prefill starts AT the base and costs exactly
//      [user header + 12 audio soft-tokens + turn close] forwards — an
//      incremental prefill, никакой re-prefill of the frozen system prompt.
//   4. resume_after_pause() (the barge-in micro-rollback) restores the
//      pre-edge-commit position pointer-only; its wall time is reported in µs.
//
// Wall-clock numbers are PRINTED as telemetry, not asserted (GPU timing in CI
// flakes); the latency contract is pinned through token-count arithmetic.
//
// External requirements (SKIPs when absent):
//   BLACKWELL_AWQ_INDEX       backbone index (default
//                             F:/AI/llama-3.1-8B-Instruct-AWQ-INT4/model.safetensors.index.json)
//   BLACKWELL_AUDIO_HEAD      Ultravox head dir (default F:/AI/ultravox-v0_5-llama-3_1-8b)
//   BLACKWELL_ULTRAVOX_DUMPS  golden-dump dir with input_features.bin [128,3000]
// =============================================================================
#include <gtest/gtest.h>

// Speech-stack deps (blackwell_bridge + whisper_dsp) are wired by the ROOT
// CMakeLists after audio_sandbox is added; without them this TU is empty.
#ifdef BLACKWELL_HAVE_SPEECH_STACK

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common.h"                    // CUDA_CHECK
#include "device_buffer.h"             // blackwell::DeviceBuffer
#include "blackwell/engine.h"
#include "blackwell/tokenizer.h"       // blackwell::TokenizerFactory
#include "blackwell/runtime_config.h"  // InferenceConfig / AudioStreamingMode
#include "real_engine_control.hpp"     // rt::RealEngineControl (white-box base)

using blackwell::DeviceBuffer;
using blackwell::EngineStatus;

namespace {

std::string env_or(const char* n, const std::string& f) {
    const char* v = std::getenv(n);
    return (v && *v) ? std::string(v) : f;
}
std::string awq_index_path() {
    return env_or("BLACKWELL_AWQ_INDEX",
                  "F:/AI/llama-3.1-8B-Instruct-AWQ-INT4/model.safetensors.index.json");
}
std::string model_dir_of(const std::string& index) {
    const auto cut = index.find_last_of("/\\");
    return cut == std::string::npos ? std::string(".") : index.substr(0, cut);
}
std::string audio_head_dir() {
    return env_or("BLACKWELL_AUDIO_HEAD", "F:/AI/ultravox-v0_5-llama-3_1-8b");
}
std::string dumps_dir() {
    return env_or("BLACKWELL_ULTRAVOX_DUMPS",
                  "D:/Projects/BlackwellLLM/tests/integration/golden_dumps/ultravox");
}
bool file_exists(const std::string& p) { return std::ifstream(p).good(); }

constexpr int kNumMel      = 128;    // input_features.bin geometry [128, 3000]
constexpr int kDumpFrames  = 3000;
constexpr int kChunkFrames = 192;    // one "2 s" chunk = 12 soft-tokens (192 * 10 ms)
constexpr int kChunkTokens = kChunkFrames / blackwell::audio::kMelFramesPerSoftToken;  // 12
constexpr int kMaxCtx      = 1024;

constexpr const char* kSystemPrompt =
    "You are a live speech translator. Transcribe the user's speech, then "
    "translate it to English. Answer as: [Speech] ... | [Translation] ...";

// Wall clock with a device sync at both ends (phase boundaries, not throughput).
using Clock = std::chrono::steady_clock;
Clock::time_point tick() {
    CUDA_CHECK(cudaDeviceSynchronize());
    return Clock::now();
}
double ms_between(const Clock::time_point& a, const Clock::time_point& b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// Slice mel columns [c0, c0+frames) out of the host dump and upload contiguous
// [kNumMel, frames] (mel-major) — one utterance chunk.
DeviceBuffer<float> upload_chunk(const std::vector<float>& mel, int c0, int frames) {
    std::vector<float> host(static_cast<std::size_t>(kNumMel) * frames);
    for (int m = 0; m < kNumMel; ++m)
        std::copy(mel.begin() + static_cast<std::size_t>(m) * kDumpFrames + c0,
                  mel.begin() + static_cast<std::size_t>(m) * kDumpFrames + c0 + frames,
                  host.begin() + static_cast<std::size_t>(m) * frames);
    DeviceBuffer<float> d(host.size());
    CUDA_CHECK(cudaMemcpy(d.get(), host.data(), host.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    return d;
}

// White-box control: re-exposes the protected center-slice primitives and adds
// the token-level prefill/decode drivers the phase timer brackets. Same single
// engine-owning thread as production (this test thread IS the engine thread).
class ProfilerControl : public rt::RealEngineControl {
public:
    using rt::RealEngineControl::RealEngineControl;
    using rt::RealEngineControl::prefill_audio_center_hop;
    using rt::RealEngineControl::commit_audio_right_edge;
    using rt::RealEngineControl::resume_after_pause;

    int position() const { return pos_; }

    rt::RealEngineControl::TurnFrame turn_frame() const { return make_turn_frame(); }

    // Greedy prefill of a token-id span; *next_out receives the sampled token
    // after the LAST id (the first decode token when ids end with the gen cue).
    int prefill_ids(const std::vector<int>& ids, int* next_out = nullptr) {
        int next = -1, n = 0;
        for (const int id : ids) {
            if (pos_ >= max_context_) break;
            if (engine_->forward_status(id, pos_, 0.0f, 1.0f, cfg_.seq_id, &next) !=
                EngineStatus::Success)
                break;
            ++pos_;
            ++n;
        }
        if (next_out) *next_out = next;
        return n;
    }

    // Greedy decode of up to n tokens starting from `first`; returns the text.
    std::string decode_n(int first, int n) {
        std::string out;
        int next = first;
        for (int i = 0; i < n && pos_ < max_context_; ++i) {
            if (tok_->is_stop(next)) break;
            out += tok_->decode(next, /*render_special=*/false);
            if (engine_->forward_status(next, pos_, 0.0f, 1.0f, cfg_.seq_id, &next) !=
                EngineStatus::Success)
                break;
            ++pos_;
        }
        return out;
    }

    // The full rewind path a speech restart triggers (Command marshaled by the
    // ring in production; direct dispatch here on the engine thread).
    EngineStatus rewind_now(uint32_t keep) {
        Command cmd{};
        cmd.type = CommandType::Rewind;
        cmd.keep_prompt_tokens = keep;
        return do_rewind(cmd);
    }
};

// One utterance through the center-slice flow with phase timing. Returns the
// phase breakdown; asserts the token-count contract inside.
struct UtteranceProfile {
    double warm_hops_ms = 0;   // encoder+projector+center prefill (DURING speech)
    double edge_ms      = 0;   // VAD trigger -> right-edge commit
    double template_ms  = 0;   // turn-close template prefill (incl. first sample)
    double ttft_ms      = 0;   // VAD trigger -> first decode token available
    int    first_token  = -1;
    int    pos_pre_edge = 0;   // pause-checkpoint position (resume target)
};

void run_utterance(ProfilerControl& control, const float* d_mel_chunk,
                   int base_pos, UtteranceProfile* out) {
    UtteranceProfile p;
    const auto frame = control.turn_frame();

    ASSERT_EQ(control.position(), base_pos)
        << "utterance must start AT the retained base (no stale KV)";

    // Speech begins: the user header opens the turn (live: first warm hop).
    control.prefill_ids(frame.user_hdr);

    // Warm hop DURING speech: encode+project the chunk, inject the stable center
    // tokens (12 - K). This is the work CenterSlice keeps OFF the TTFT path.
    const auto t_warm0 = tick();
    control.prefill_audio_center_hop(d_mel_chunk, kChunkFrames, /*mel_frame_offset=*/0);
    const auto t_warm1 = tick();
    p.warm_hops_ms = ms_between(t_warm0, t_warm1);
    p.pos_pre_edge = control.position();

    // ---- VAD trigger: everything below is the TTFT critical path -------------
    const auto t_vad = tick();
    control.commit_audio_right_edge();                 // K rows, zero encoder work
    const auto t_edge = tick();
    control.prefill_ids(frame.user_eot);
    control.prefill_ids(frame.gen_cue, &p.first_token);  // last forward samples token #1
    const auto t_first = tick();

    p.edge_ms     = ms_between(t_vad, t_edge);
    p.template_ms = ms_between(t_edge, t_first);
    p.ttft_ms     = ms_between(t_vad, t_first);

    // Token-count contract: header + ALL 12 audio soft-tokens + turn close, and
    // nothing else — the audio is fully in the KV, none of it re-prefilled.
    const int expected = static_cast<int>(frame.user_hdr.size()) + kChunkTokens +
                         static_cast<int>(frame.user_eot.size()) +
                         static_cast<int>(frame.gen_cue.size());
    ASSERT_EQ(control.position() - base_pos, expected);
    ASSERT_GE(p.first_token, 0);
    *out = p;
}

}  // namespace

TEST(StreamingTtftProfiler, CenterSliceTtftAndStatelessRewindSplitTest) {
    const std::string index = awq_index_path();
    const std::string head  = audio_head_dir();
    const std::string feats = dumps_dir() + "/input_features.bin";
    if (!file_exists(index))
        GTEST_SKIP() << "AWQ backbone absent: " << index << " (set BLACKWELL_AWQ_INDEX)";
    if (!file_exists(head + "/model.safetensors"))
        GTEST_SKIP() << "audio head absent: " << head << " (set BLACKWELL_AUDIO_HEAD)";
    if (!file_exists(feats))
        GTEST_SKIP() << "log-mel dump absent: " << feats << " (set BLACKWELL_ULTRAVOX_DUMPS)";

    // ---- setup: engine + tokenizer + audio head (center-slice plan) ----------
    blackwell::InferenceConfig req;
    req.max_context_length = kMaxCtx;
    req.audio_streaming.enable = true;
    req.audio_streaming.mode = blackwell::AudioStreamingMode::CenterSlice;
    // Defaults: window 2240 ms (14 tok), hop 320 ms (2 tok), L=2, K=3.

    std::cout << "[ttft] constructing 8B AWQ engine from " << index << " ...\n";
    BlackwellEngine engine(index, req);
    std::unique_ptr<blackwell::ITokenizer> tok =
        blackwell::TokenizerFactory::create(model_dir_of(index));
    ProfilerControl control(&engine, tok.get(), kMaxCtx);
    control.load_audio_head(head);

    std::vector<float> mel(static_cast<std::size_t>(kNumMel) * kDumpFrames);
    { std::ifstream f(feats, std::ios::binary);
      f.read(reinterpret_cast<char*>(mel.data()),
             static_cast<std::streamsize>(mel.size() * sizeof(float)));
      ASSERT_TRUE(bool(f)) << "short read: " << feats; }
    // Two DIFFERENT stretches of the real utterance as the two "2 s" chunks.
    DeviceBuffer<float> d_chunk1 = upload_chunk(mel, /*c0=*/0,   kChunkFrames);
    DeviceBuffer<float> d_chunk2 = upload_chunk(mel, /*c0=*/400, kChunkFrames);

    const auto t_sys0 = tick();
    const uint32_t nsys = control.prefill_system_prompt(kSystemPrompt);
    const auto t_sys1 = tick();
    const int base = control.history_base_pos();
    ASSERT_EQ(base, static_cast<int>(nsys));
    std::cout << std::fixed << std::setprecision(2)
              << "[ttft] system prefix: " << nsys << " tokens in "
              << ms_between(t_sys0, t_sys1) << " ms (prefilled ONCE, frozen)\n";

    // =========================================================================
    // Utterance 1: chunk 1 -> VAD timeout -> TTFT breakdown -> short decode.
    // =========================================================================
    UtteranceProfile u1;
    ASSERT_NO_FATAL_FAILURE(run_utterance(control, d_chunk1.get(), base, &u1));
    const auto t_dec0 = tick();
    const std::string reply1 = control.decode_n(u1.first_token, 8);
    const auto t_dec1 = tick();
    std::cout << "[ttft] ---- utterance 1 (chunk 1: " << kChunkTokens << " audio tokens) ----\n"
              << "[ttft]   speech-time warm hop (enc+proj+prefill): "
              << u1.warm_hops_ms << " ms  <- OFF the critical path\n"
              << "[ttft]   VAD trigger -> edge commit (K=3 rows):   " << u1.edge_ms << " ms\n"
              << "[ttft]   turn-close template prefill + sample:    " << u1.template_ms << " ms\n"
              << "[ttft]   TTFT (VAD trigger -> first token):       " << u1.ttft_ms << " ms\n"
              << "[ttft]   decode 8 tokens: " << ms_between(t_dec0, t_dec1) << " ms  reply=\""
              << reply1 << "\"\n";

    // Turn end: stateless text rewind — the whole turn (audio + generated text)
    // flushes back to the retained base; zero residue.
    control.finalize_turn(reply1.empty() ? "profiler" : reply1, /*completed=*/true);
    ASSERT_EQ(control.position(), base) << "stateless flush must return exactly to the base";
    ASSERT_EQ(control.history_base_pos(), base);

    // =========================================================================
    // Speech restarts (chunk 2): the rewind fires IMMEDIATELY and resolves to
    // the history base — proving no generated text leaked into the floor.
    // =========================================================================
    control.cancel_generation(2);
    const auto t_rw0 = Clock::now();
    ASSERT_EQ(control.rewind_now(/*keep=*/0), EngineStatus::Success);
    const double rewind_us = std::chrono::duration<double, std::micro>(
        Clock::now() - t_rw0).count();
    EXPECT_EQ(control.last_rewind_keep(), static_cast<uint32_t>(base))
        << "rewind keep must resolve EXACTLY to history_base_pos (system floor)";
    ASSERT_EQ(control.position(), base);
    std::cout << "[ttft] speech restart rewind -> keep=" << control.last_rewind_keep()
              << " (== history_base " << base << ") in " << rewind_us << " us\n";

    // Utterance 2: identical incremental cost — starts AT the base (asserted
    // inside run_utterance: the frozen system prompt is NOT re-prefilled).
    UtteranceProfile u2;
    ASSERT_NO_FATAL_FAILURE(run_utterance(control, d_chunk2.get(), base, &u2));
    std::cout << "[ttft] ---- utterance 2 (chunk 2 after restart) ----\n"
              << "[ttft]   speech-time warm hop:                    " << u2.warm_hops_ms << " ms\n"
              << "[ttft]   VAD trigger -> edge commit:              " << u2.edge_ms << " ms\n"
              << "[ttft]   turn-close template prefill + sample:    " << u2.template_ms << " ms\n"
              << "[ttft]   TTFT:                                    " << u2.ttft_ms << " ms\n"
              << "[ttft] chunk-2 prefill was INCREMENTAL: started at pos " << base
              << ", system prefix (" << nsys << " tok) untouched\n";

    // =========================================================================
    // Barge-in micro-resume: pointer-only rollback of the tentative edge +
    // template (+ any decoded tokens) back to the pause checkpoint.
    // =========================================================================
    (void)control.decode_n(u2.first_token, 4);          // a few tentative tokens
    const auto t_res0 = Clock::now();
    control.resume_after_pause();
    const double resume_us = std::chrono::duration<double, std::micro>(
        Clock::now() - t_res0).count();
    ASSERT_EQ(control.position(), u2.pos_pre_edge)
        << "resume must restore the pre-edge-commit position exactly";
    std::cout << "[ttft] barge-in resume rollback (pos " << u2.pos_pre_edge + 3
              << "+ -> " << u2.pos_pre_edge << "): " << resume_us
              << " us (pointer-only, zero GPU compute)\n";

    // Clean up the interrupted utterance: flush to the base.
    control.finalize_turn(std::string(), /*completed=*/false);
    ASSERT_EQ(control.position(), base);
    std::cout << "[ttft] interrupted utterance flushed -> pos " << control.position()
              << " (== base)\n";
}

#endif  // BLACKWELL_HAVE_SPEECH_STACK
