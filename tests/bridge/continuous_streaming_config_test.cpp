// =============================================================================
// tests/bridge/continuous_streaming_config_test.cpp
//
// The re-translation pipeline's tunables (docs/CONTINUOUS_STREAMING.md). CPU-only.
//
// Two things are actually worth pinning here, and neither is "the default is 500":
//
//   1. clamp() is TOTAL and IDEMPOTENT. It is the single gate between a UI drag /
//      a CLI string and the pipeline, so "any input becomes a legal config" has to
//      hold for adversarial input, not just plausible input.
//   2. target <= high_water survives everything — clamping, a torn store observed
//      mid-drag, and the ordering inside LiveStreamingConfig. A low watermark above
//      the high watermark would evict on every commit, which is a pathology the
//      type system cannot express away.
// =============================================================================
#include "continuous_streaming_config.hpp"

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

namespace {

using blackwell::bridge::ContinuousStreamingConfig;
using blackwell::bridge::LiveStreamingConfig;

using Cfg = ContinuousStreamingConfig;

// ---- defaults ---------------------------------------------------------------

// Both of these moved once the Release profile landed (docs/CONTINUOUS_STREAMING.md
// §7): the redraft is ~20.5 ms/token, so 15 s utterances extrapolate to a ~4.6 s
// redraft and a 500 ms cadence names a rate the GPU cannot produce.
TEST(ContinuousStreamingConfig, DefaultsAreTheAgreedOperatingPoint) {
    const Cfg cfg{};
    EXPECT_EQ(cfg.partial_cadence_ms, 1000);
    EXPECT_EQ(cfg.max_utterance_ms, 8000);
    EXPECT_EQ(cfg.eviction_high_water_mark, 3000);
    EXPECT_EQ(cfg.eviction_target_tokens, 1000);
    EXPECT_EQ(cfg.pre_roll_ms, 250);
    EXPECT_EQ(cfg.hangover_ms, 400);
    EXPECT_TRUE(cfg.valid());
}

// ---- clamp(): total and idempotent ------------------------------------------

TEST(ContinuousStreamingConfig, ClampIsTotalOverAdversarialInput) {
    Cfg cfg{};
    cfg.pre_roll_ms = -100000;
    cfg.hangover_ms = 0;
    cfg.max_utterance_ms = 999999;
    cfg.partial_cadence_ms = -1;
    cfg.eviction_high_water_mark = -5;
    cfg.eviction_target_tokens = -5;
    cfg.clamp();

    EXPECT_TRUE(cfg.valid());
    EXPECT_EQ(cfg.pre_roll_ms, Cfg::kMinPreRollMs);
    EXPECT_EQ(cfg.hangover_ms, Cfg::kMinHangoverMs);
    EXPECT_EQ(cfg.max_utterance_ms, Cfg::kMaxMaxUtteranceMs);
    EXPECT_EQ(cfg.partial_cadence_ms, Cfg::kMinPartialCadenceMs);
    EXPECT_EQ(cfg.eviction_high_water_mark, Cfg::kMinHighWaterTokens);
    EXPECT_LE(cfg.eviction_target_tokens, cfg.eviction_high_water_mark);
}

TEST(ContinuousStreamingConfig, ClampIsIdempotent) {
    Cfg cfg{};
    cfg.pre_roll_ms = 1 << 20;
    cfg.hangover_ms = -7;
    cfg.eviction_high_water_mark = 200;
    cfg.eviction_target_tokens = 199999;
    cfg.clamp();
    const Cfg once = cfg;
    cfg.clamp();
    EXPECT_EQ(cfg, once);
}

TEST(ContinuousStreamingConfig, ValidAgreesWithClamp) {
    Cfg cfg{};
    EXPECT_TRUE(cfg.valid());
    cfg.eviction_target_tokens = cfg.eviction_high_water_mark + 1;
    EXPECT_FALSE(cfg.valid());
    cfg.clamp();
    EXPECT_TRUE(cfg.valid());
}

// ---- the watermark invariant ------------------------------------------------

// A low watermark above the high watermark would make eviction_min_delta() demand
// a negative reclaim, i.e. evict on every single commit.
TEST(ContinuousStreamingConfig, TargetIsPulledDownToTheHighWaterMark) {
    Cfg cfg{};
    cfg.eviction_high_water_mark = 2000;
    cfg.eviction_target_tokens = 8000;
    cfg.clamp();
    EXPECT_EQ(cfg.eviction_high_water_mark, 2000);
    EXPECT_EQ(cfg.eviction_target_tokens, 2000);
}

// Ordering inside clamp() matters: the per-field ranges run first, so lowering
// high_water into range cannot leave target stranded above it.
TEST(ContinuousStreamingConfig, ClampingHighWaterAlsoConstrainsTarget) {
    Cfg cfg{};
    cfg.eviction_high_water_mark = Cfg::kMaxHighWaterTokens * 4;   // out of range
    cfg.eviction_target_tokens = Cfg::kMaxHighWaterTokens * 3;
    cfg.clamp();
    EXPECT_EQ(cfg.eviction_high_water_mark, Cfg::kMaxHighWaterTokens);
    EXPECT_LE(cfg.eviction_target_tokens, cfg.eviction_high_water_mark);
    EXPECT_TRUE(cfg.valid());
}

// ---- eviction policy --------------------------------------------------------

TEST(ContinuousStreamingConfig, ShouldEvictFiresAtTheHighWaterMarkAndNotBefore) {
    Cfg cfg{};
    cfg.eviction_high_water_mark = 3000;
    EXPECT_FALSE(cfg.should_evict(2999u));
    EXPECT_TRUE(cfg.should_evict(3000u));
    EXPECT_TRUE(cfg.should_evict(9000u));
}

// One pass reclaims down TO the low watermark, so the gap between the two IS the
// hysteresis — that is what makes compactions rare and large instead of constant
// and small.
TEST(ContinuousStreamingConfig, MinDeltaReclaimsDownToTheTarget) {
    Cfg cfg{};
    cfg.eviction_high_water_mark = 3000;
    cfg.eviction_target_tokens = 1000;
    EXPECT_EQ(cfg.eviction_min_delta(3000u), 2000u);
    EXPECT_EQ(cfg.eviction_min_delta(3500u), 2500u);
    EXPECT_EQ(cfg.eviction_min_delta(1000u), 0u);
    EXPECT_EQ(cfg.eviction_min_delta(10u), 0u);   // never underflows
}

// ---- draft headroom ---------------------------------------------------------

// The draft zone is deliberately absent from the watermarks: eviction runs behind
// C, so the caller must leave the draft's worst case as headroom above the high
// water mark. This is that worst case, with one implementation.
TEST(ContinuousStreamingConfig, DraftHeadroomIsAudioCeilingPlusDecodeCap) {
    Cfg cfg{};
    cfg.max_utterance_ms = 15000;
    // 15000 ms / 160 ms per soft token = 93.75 -> 94 (ceil), + 128 decode cap.
    EXPECT_EQ(cfg.draft_headroom_tokens(128), 94u + 128u);
    EXPECT_EQ(cfg.draft_headroom_tokens(0), 94u);
    EXPECT_EQ(cfg.draft_headroom_tokens(-5), 94u);        // negative cap == none
    EXPECT_GT(cfg.draft_headroom_tokens(128, 1), 0u);     // no divide-by-zero
}

TEST(ContinuousStreamingConfig, FitsContextRejectsAWindowThatCannotHoldItsOwnDraft) {
    Cfg cfg{};
    cfg.eviction_high_water_mark = 3000;
    cfg.max_utterance_ms = 15000;
    const std::uint32_t need = 3000u + cfg.draft_headroom_tokens(128);

    EXPECT_TRUE(cfg.fits_context(static_cast<int>(need), 128));
    EXPECT_TRUE(cfg.fits_context(static_cast<int>(need) + 1, 128));
    EXPECT_FALSE(cfg.fits_context(static_cast<int>(need) - 1, 128));
    EXPECT_FALSE(cfg.fits_context(2048, 128));   // the app's current kMaxContext
}

// ---- derivation to the segmenter --------------------------------------------

TEST(ContinuousStreamingConfig, DerivesTheSegmenterGatingKnobs) {
    Cfg cfg{};
    cfg.pre_roll_ms = 300;
    cfg.hangover_ms = 450;
    cfg.max_utterance_ms = 12000;
    cfg.partial_cadence_ms = 750;

    const auto s = cfg.to_segmenter_config(16000, 160);
    EXPECT_EQ(s.sample_rate, 16000);
    EXPECT_EQ(s.block_samples, 160);
    EXPECT_EQ(s.preroll_ms, 300);
    EXPECT_EQ(s.hangover_ms, 450);
    EXPECT_EQ(s.max_utterance_ms, 12000);
    EXPECT_EQ(s.partial_period_ms, 750);

    // Detector policy is NOT user-facing streaming config and must keep the
    // segmenter's own defaults.
    const blackwell::vad::SegmenterConfig def{};
    EXPECT_FLOAT_EQ(s.onset_threshold, def.onset_threshold);
    EXPECT_EQ(s.tail_pad_ms, def.tail_pad_ms);
    EXPECT_EQ(s.min_utterance_ms, def.min_utterance_ms);
}

// ---- LiveStreamingConfig ----------------------------------------------------

TEST(LiveStreamingConfig, RoundTripsAValidConfig) {
    Cfg cfg{};
    cfg.pre_roll_ms = 300;
    cfg.hangover_ms = 500;
    cfg.max_utterance_ms = 10000;
    cfg.partial_cadence_ms = 250;
    cfg.eviction_high_water_mark = 4000;
    cfg.eviction_target_tokens = 1500;

    LiveStreamingConfig live(cfg);
    EXPECT_EQ(live.load(), cfg);
}

TEST(LiveStreamingConfig, StoreClampsSoAReaderNeverSeesAnIllegalConfig) {
    LiveStreamingConfig live;
    Cfg bad{};
    bad.eviction_high_water_mark = 1000;
    bad.eviction_target_tokens = 999999;
    bad.hangover_ms = -1;
    live.store(bad);

    const Cfg got = live.load();
    EXPECT_TRUE(got.valid());
    EXPECT_LE(got.eviction_target_tokens, got.eviction_high_water_mark);
}

TEST(LiveStreamingConfig, DefaultConstructedMatchesTheStructDefaults) {
    EXPECT_EQ(LiveStreamingConfig{}.load(), Cfg{});
}

// The panel stores field-wise while the engine thread loads, so a reader can land
// between two stores. Every individual field stays valid and the watermark pair is
// ordered so the one invariant that matters cannot tear — load() must therefore
// never hand back an illegal config, in either direction of change.
TEST(LiveStreamingConfig, WatermarkInvariantHoldsUnderConcurrentStores) {
    LiveStreamingConfig live;
    std::atomic<bool> stop{false};
    std::atomic<int> violations{0};
    std::atomic<int> reads{0};

    std::thread writer([&] {
        for (int i = 0; !stop.load(std::memory_order_relaxed); ++i) {
            Cfg cfg{};
            // Alternate between a large and a small window so the store path
            // exercises BOTH orderings (high water growing and shrinking).
            if ((i & 1) == 0) {
                cfg.eviction_high_water_mark = 8000;
                cfg.eviction_target_tokens = 7000;
            } else {
                cfg.eviction_high_water_mark = 200;
                cfg.eviction_target_tokens = 150;
            }
            live.store(cfg);
        }
    });

    for (int i = 0; i < 200000; ++i) {
        const Cfg got = live.load();
        if (got.eviction_target_tokens > got.eviction_high_water_mark)
            violations.fetch_add(1, std::memory_order_relaxed);
        if (!got.valid()) violations.fetch_add(1, std::memory_order_relaxed);
        reads.fetch_add(1, std::memory_order_relaxed);
    }
    stop.store(true, std::memory_order_relaxed);
    writer.join();

    EXPECT_EQ(violations.load(), 0);
    EXPECT_GT(reads.load(), 0);
}

}  // namespace
