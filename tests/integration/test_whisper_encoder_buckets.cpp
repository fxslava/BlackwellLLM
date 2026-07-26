// =============================================================================
// Phase-1 bucketed WhisperEncoder + SlidingAudioWindow tests.
//
// Self-contained (no checkpoint / golden dumps): the encoder is loaded with
// correctly-sized RANDOM weights, so these run on any machine with the Blackwell
// GPU free. They validate the NEW Phase-1 mechanics — graph-bucket selection /
// geometry / finite output, the back-compat full-30 s path, and the sliding-window
// shift + tail-delta slice — NOT numerical parity (that stays on the 3000 bucket in
// test_whisper_encoder_full_pass.cpp against the PyTorch reference).
// =============================================================================
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "common.h"                 // CUDA_CHECK
#include "device_buffer.h"          // blackwell::DeviceBuffer
#include "whisper_encoder.h"        // WhisperEncoder{,Config}, WhisperWeights
#include "sliding_audio_window.h"   // SlidingAudioWindow (compile-checked here)

using blackwell::DeviceBuffer;
using blackwell::audio::WhisperEncoder;
using blackwell::audio::WhisperEncoderConfig;
using blackwell::audio::WhisperWeights;
using blackwell::audio::WhisperLayerWeights;

namespace {

std::vector<float> rand_vec(size_t n, float scale, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.0f, scale);
    std::vector<float> v(n);
    for (float& x : v) x = nd(rng);
    return v;
}
std::vector<float> const_vec(size_t n, float v) { return std::vector<float>(n, v); }

// Correctly-sized random weights for `cfg` (init scales that keep the 32-layer
// pre-LN stack finite: LN weight 1 / bias 0, linears ~N(0, 0.02)).
WhisperWeights make_random_weights(const WhisperEncoderConfig& cfg) {
    std::mt19937 rng(1234);
    const size_t D = cfg.d_model, mel = cfg.num_mel_bins, ffn = cfg.ffn_dim,
                 seq = cfg.max_source_positions;
    WhisperWeights w;
    w.conv1_w = rand_vec(D * mel * 3, 0.02f, rng);
    w.conv1_b = const_vec(D, 0.0f);
    w.conv2_w = rand_vec(D * D * 3, 0.02f, rng);
    w.conv2_b = const_vec(D, 0.0f);
    w.embed_positions = rand_vec(seq * D, 0.02f, rng);
    w.layer_norm_w = const_vec(D, 1.0f);
    w.layer_norm_b = const_vec(D, 0.0f);
    w.layers.resize(cfg.num_layers);
    for (auto& l : w.layers) {
        l.self_attn_layer_norm_w = const_vec(D, 1.0f);
        l.self_attn_layer_norm_b = const_vec(D, 0.0f);
        l.q_w = rand_vec(D * D, 0.02f, rng); l.q_b = const_vec(D, 0.0f);
        l.k_w = rand_vec(D * D, 0.02f, rng);
        l.v_w = rand_vec(D * D, 0.02f, rng); l.v_b = const_vec(D, 0.0f);
        l.out_w = rand_vec(D * D, 0.02f, rng); l.out_b = const_vec(D, 0.0f);
        l.final_layer_norm_w = const_vec(D, 1.0f);
        l.final_layer_norm_b = const_vec(D, 0.0f);
        l.fc1_w = rand_vec(ffn * D, 0.02f, rng); l.fc1_b = const_vec(ffn, 0.0f);
        l.fc2_w = rand_vec(D * ffn, 0.02f, rng); l.fc2_b = const_vec(D, 0.0f);
    }
    return w;
}

bool all_finite(const std::vector<float>& v) {
    for (float x : v) if (!std::isfinite(x)) return false;
    return true;
}

}  // namespace

// The bucket a forward() selects matches the smallest bucket covering mel_frames,
// its output length is conv_out_frames_of(bucket), and the output is finite.
TEST(WhisperEncoderBuckets, SelectionGeometryAndFiniteOutput) {
    WhisperEncoderConfig cfg;                       // turbo defaults + {150,300,500,3000}
    WhisperEncoder enc(cfg);
    enc.load_weights(make_random_weights(cfg));

    DeviceBuffer<float> d_mel(static_cast<size_t>(cfg.num_mel_bins) * cfg.conv_frames);
    {   std::mt19937 rng(7);
        auto h = rand_vec(d_mel.count(), 0.5f, rng);
        CUDA_CHECK(cudaMemcpy(d_mel.get(), h.data(), h.size() * sizeof(float),
                              cudaMemcpyHostToDevice)); }

    struct Case { int mel_frames; int want_seq; };
    // conv_out_frames_of: 150->75, 300->150, 500->250, 3000->1500; 200 rounds to 300.
    const Case cases[] = {{3000, 1500}, {500, 250}, {300, 150}, {200, 150}, {150, 75}};
    for (const Case& c : cases) {
        const float* d_out = enc.forward(d_mel.get(), c.mel_frames);
        CUDA_CHECK(cudaStreamSynchronize(enc.stream()));
        EXPECT_EQ(enc.output_frames(), c.want_seq) << "mel_frames=" << c.mel_frames;

        std::vector<float> out(static_cast<size_t>(c.want_seq) * cfg.d_model);
        CUDA_CHECK(cudaMemcpy(out.data(), d_out, out.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        EXPECT_TRUE(all_finite(out)) << "non-finite output at mel_frames=" << c.mel_frames;
    }

    // Valid (unpadded) frame accounting used by the sliding-window slicer.
    EXPECT_EQ(enc.valid_output_frames(224), 112);
    EXPECT_EQ(WhisperEncoderConfig::conv_out_frames_of(300), 150);
}

// The back-compat single-arg forward() is the full 30 s bucket.
TEST(WhisperEncoderBuckets, BackCompatForwardIsFullBucket) {
    WhisperEncoderConfig cfg;
    WhisperEncoder enc(cfg);
    enc.load_weights(make_random_weights(cfg));

    DeviceBuffer<float> d_mel(static_cast<size_t>(cfg.num_mel_bins) * cfg.conv_frames);
    CUDA_CHECK(cudaMemset(d_mel.get(), 0, d_mel.count() * sizeof(float)));
    (void)enc.forward(d_mel.get());
    CUDA_CHECK(cudaStreamSynchronize(enc.stream()));
    EXPECT_EQ(enc.output_frames(), cfg.max_source_positions);   // 1500
}

// Out-of-range mel_frames throws (INIT-tier contract).
TEST(WhisperEncoderBuckets, RejectsOutOfRangeFrames) {
    WhisperEncoderConfig cfg;
    WhisperEncoder enc(cfg);
    enc.load_weights(make_random_weights(cfg));
    DeviceBuffer<float> d_mel(static_cast<size_t>(cfg.num_mel_bins) * cfg.conv_frames);
    EXPECT_THROW((void)enc.forward(d_mel.get(), 0), std::runtime_error);
    EXPECT_THROW((void)enc.forward(d_mel.get(), cfg.conv_frames + 1), std::runtime_error);
}

// The absolute position offset threads through the captured graph: encoding the same
// window at a shifted offset applies a different embed_positions slice, so the output
// changes (and stays finite). offset 0 is the default whole-clip encode.
TEST(WhisperEncoderBuckets, PositionOffsetShiftsOutput) {
    WhisperEncoderConfig cfg;
    WhisperEncoder enc(cfg);
    enc.load_weights(make_random_weights(cfg));

    const int frames = 300;                                        // bucket 300 -> seq 150
    DeviceBuffer<float> d_mel(static_cast<size_t>(cfg.num_mel_bins) * frames);
    {   std::mt19937 rng(3);
        auto h = rand_vec(d_mel.count(), 0.5f, rng);
        CUDA_CHECK(cudaMemcpy(d_mel.get(), h.data(), h.size() * sizeof(float),
                              cudaMemcpyHostToDevice)); }

    const int seq = WhisperEncoderConfig::conv_out_frames_of(frames);   // 150
    const size_t n = static_cast<size_t>(seq) * cfg.d_model;

    const float* p0 = enc.forward(d_mel.get(), frames, /*enc_pos_offset=*/0);
    CUDA_CHECK(cudaStreamSynchronize(enc.stream()));
    std::vector<float> out0(n);
    CUDA_CHECK(cudaMemcpy(out0.data(), p0, n * sizeof(float), cudaMemcpyDeviceToHost));

    const float* pK = enc.forward(d_mel.get(), frames, /*enc_pos_offset=*/200);  // +200 <= 1500-150
    CUDA_CHECK(cudaStreamSynchronize(enc.stream()));
    std::vector<float> outK(n);
    CUDA_CHECK(cudaMemcpy(outK.data(), pK, n * sizeof(float), cudaMemcpyDeviceToHost));

    EXPECT_TRUE(all_finite(out0));
    EXPECT_TRUE(all_finite(outK));
    bool differs = false;
    for (size_t i = 0; i < n && !differs; ++i)
        if (std::abs(out0[i] - outK[i]) > 1e-4f) differs = true;
    EXPECT_TRUE(differs) << "position offset did not change the encoder output";
}

// SlidingAudioWindow: a push shifts the window left and lands the newest chunk at
// the tail; extract_delta returns exactly the tail hop_tokens rows.
TEST(SlidingAudioWindow, ShiftLandsNewestAtTailAndDeltaIsTail) {
    using blackwell::audio::SlidingAudioWindow;
    using blackwell::audio::SlidingWindowConfig;
    using blackwell::audio::kMelFramesPerSoftToken;

    SlidingWindowConfig scfg;
    scfg.num_mel_bins = 2;
    scfg.history_tokens = 1;
    scfg.hop_tokens = 1;                        // window = 2 tokens = 32 mel frames
    const int W = scfg.window_mel_frames();
    const int chunk = kMelFramesPerSoftToken;   // 16 frames == 1 hop
    const int text_hidden = 4;

    SlidingAudioWindow win(scfg, text_hidden);

    auto push_const = [&](float v) {
        std::vector<float> h(static_cast<size_t>(scfg.num_mel_bins) * chunk, v);
        DeviceBuffer<float> d(h.size());
        CUDA_CHECK(cudaMemcpy(d.get(), h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
        win.push_chunk(d.get(), chunk, /*stream=*/0);
        CUDA_CHECK(cudaStreamSynchronize(0));
    };
    push_const(1.0f);                           // window: [0..0, 1..1]
    push_const(2.0f);                           // window: [1..1, 2..2]

    std::vector<float> w(static_cast<size_t>(scfg.num_mel_bins) * W);
    CUDA_CHECK(cudaMemcpy(w.data(), win.window_mel(), w.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    // Row 0: first `chunk` frames == 1 (shifted history), last `chunk` == 2 (newest).
    for (int t = 0; t < W; ++t) {
        const float expect = (t < W - chunk) ? 1.0f : 2.0f;
        EXPECT_FLOAT_EQ(w[t], expect) << "row0 frame " << t;
    }
    EXPECT_TRUE(win.warm());                     // 32 frames pushed == full window

    // Fake projector output of 5 soft-tokens (row r filled with value r); the delta
    // must be the single tail row (index 4).
    const int out_soft = 5;
    std::vector<float> proj(static_cast<size_t>(out_soft) * text_hidden);
    for (int r = 0; r < out_soft; ++r)
        for (int c = 0; c < text_hidden; ++c) proj[r * text_hidden + c] = static_cast<float>(r);
    DeviceBuffer<float> d_proj(proj.size());
    CUDA_CHECK(cudaMemcpy(d_proj.get(), proj.data(), proj.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    const auto delta = win.extract_delta(d_proj.get(), out_soft);
    EXPECT_EQ(delta.count, scfg.hop_tokens);     // 1
    EXPECT_EQ(delta.text_hidden, text_hidden);
    EXPECT_EQ(delta.data, d_proj.get() + static_cast<size_t>(out_soft - 1) * text_hidden);
    std::vector<float> row(text_hidden);
    CUDA_CHECK(cudaMemcpy(row.data(), delta.data, row.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    for (float x : row) EXPECT_FLOAT_EQ(x, 4.0f);  // the newest (last) soft-token row
}
