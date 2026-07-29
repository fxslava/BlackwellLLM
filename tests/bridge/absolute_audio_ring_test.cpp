// =============================================================================
// tests/bridge/absolute_audio_ring_test.cpp
//
// The N-second FIFO the continuous pipeline redrafts from. CPU-only.
//
// The property that matters is the REFUSAL: a redraft asks for a whole utterance
// from its first sample, over and over, and the ring is finite. Handing back a
// truncated window would produce a confident translation of a sentence missing
// its opening word — the worst possible failure mode, because nothing downstream
// can detect it. So a range that is not fully resident is an error, never a clamp.
// =============================================================================
#include "absolute_audio_ring.hpp"

#include <cstdint>
#include <numeric>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

namespace {

using blackwell::bridge::AbsoluteAudioRing;

// Sample value == its absolute index, so any mis-addressing is visible directly.
std::vector<float> ramp(std::uint64_t begin, std::size_t n) {
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<float>(begin + i);
    return v;
}

void write_ramp(AbsoluteAudioRing& r, std::size_t n) {
    const std::vector<float> v = ramp(r.written(), n);
    r.write(v.data(), v.size());
}

TEST(AbsoluteAudioRing, StartsEmpty) {
    const AbsoluteAudioRing r(1024);
    EXPECT_EQ(r.written(), 0u);
    EXPECT_EQ(r.available_from(), 0u);
    EXPECT_FALSE(r.holds(0, 1));
}

TEST(AbsoluteAudioRing, ReadsBackWhatWasWrittenAtAbsoluteIndices) {
    AbsoluteAudioRing r(1024);
    write_ramp(r, 500);
    EXPECT_EQ(r.written(), 500u);

    std::vector<float> out;
    ASSERT_TRUE(r.read(100, 200, &out));
    ASSERT_EQ(out.size(), 100u);
    for (std::size_t i = 0; i < out.size(); ++i)
        ASSERT_FLOAT_EQ(out[i], static_cast<float>(100 + i));
}

TEST(AbsoluteAudioRing, ReadsAcrossTheWrapPoint) {
    AbsoluteAudioRing r(256);
    write_ramp(r, 200);
    write_ramp(r, 40);   // 240 written, wraps at 256

    std::vector<float> out;
    ASSERT_TRUE(r.read(190, 240, &out));
    ASSERT_EQ(out.size(), 50u);
    for (std::size_t i = 0; i < out.size(); ++i)
        ASSERT_FLOAT_EQ(out[i], static_cast<float>(190 + i));
}

// ---- the refusal ------------------------------------------------------------

TEST(AbsoluteAudioRing, RefusesAudioAlreadyOverwritten) {
    AbsoluteAudioRing r(256);
    write_ramp(r, 400);   // samples 0..143 are gone
    EXPECT_EQ(r.available_from(), 144u);

    std::vector<float> out{1.0f, 2.0f};
    EXPECT_FALSE(r.read(0, 100, &out));
    EXPECT_FALSE(r.read(100, 200, &out));     // straddles the eviction boundary
    EXPECT_EQ(out.size(), 2u) << "a refused read must not disturb the caller's buffer";

    EXPECT_TRUE(r.read(144, 400, &out));      // exactly what remains
    EXPECT_EQ(out.size(), 256u);
}

TEST(AbsoluteAudioRing, RefusesAudioNotYetWritten) {
    AbsoluteAudioRing r(1024);
    write_ramp(r, 100);
    std::vector<float> out;
    EXPECT_FALSE(r.read(50, 101, &out));
    EXPECT_FALSE(r.read(200, 300, &out));
    EXPECT_TRUE(r.read(50, 100, &out));
}

TEST(AbsoluteAudioRing, RefusesEmptyAndInvertedRanges) {
    AbsoluteAudioRing r(1024);
    write_ramp(r, 100);
    std::vector<float> out;
    EXPECT_FALSE(r.read(50, 50, &out));
    EXPECT_FALSE(r.read(60, 50, &out));
}

TEST(AbsoluteAudioRing, HoldsAgreesWithRead) {
    AbsoluteAudioRing r(256);
    write_ramp(r, 400);
    std::vector<float> out;
    for (std::uint64_t begin = 0; begin < 400; begin += 37) {
        const std::uint64_t end = begin + 60;
        EXPECT_EQ(r.holds(begin, end), r.read(begin, end, &out))
            << "holds() and read() disagreed at [" << begin << ", " << end << ")";
    }
}

// A whole maximal utterance must always be satisfiable, which is the sizing rule
// the app follows: capacity >= max_utterance + pre-roll.
TEST(AbsoluteAudioRing, HoldsAFullCapacityWindowAtAnyPoint) {
    AbsoluteAudioRing r(16000 * 20);   // 20 s @ 16 kHz
    for (int i = 0; i < 50; ++i) write_ramp(r, 16000);   // 50 s of audio

    const std::uint64_t end = r.written();
    const std::uint64_t begin = end - 16000 * 15;   // the last 15 s
    std::vector<float> out;
    ASSERT_TRUE(r.read(begin, end, &out));
    ASSERT_EQ(out.size(), static_cast<std::size_t>(16000 * 15));
    EXPECT_FLOAT_EQ(out.front(), static_cast<float>(begin));
    EXPECT_FLOAT_EQ(out.back(), static_cast<float>(end - 1));
}

// clear() must NOT rewind the absolute clock: segments already issued name
// absolute indices, and reusing them would silently hand back different audio.
TEST(AbsoluteAudioRing, ClearDropsHistoryWithoutRewindingTheClock) {
    AbsoluteAudioRing r(1024);
    write_ramp(r, 500);
    const std::uint64_t at_clear = r.written();

    r.clear();
    EXPECT_EQ(r.written(), at_clear);
    EXPECT_EQ(r.available_from(), at_clear);
    std::vector<float> out;
    EXPECT_FALSE(r.read(400, 500, &out));

    write_ramp(r, 100);
    ASSERT_TRUE(r.read(at_clear, at_clear + 100, &out));
    EXPECT_FLOAT_EQ(out.front(), static_cast<float>(at_clear));
}

// ---- threading --------------------------------------------------------------

TEST(AbsoluteAudioRing, ConcurrentWriterAndReaderNeverSeeTornAudio) {
    AbsoluteAudioRing r(16000);
    std::atomic<bool> done{false};

    std::thread producer([&] {
        for (int i = 0; i < 400; ++i) write_ramp(r, 160);   // 10 ms blocks
        done.store(true, std::memory_order_release);
    });

    int reads = 0, torn = 0;
    std::vector<float> out;
    while (!done.load(std::memory_order_acquire)) {
        const std::uint64_t end = r.written();
        if (end < 1000) continue;
        const std::uint64_t begin = end - 1000;
        if (!r.read(begin, end, &out)) continue;
        ++reads;
        // Every sample equals its absolute index, so one wrong element proves a
        // torn or mis-addressed read.
        for (std::size_t i = 0; i < out.size(); ++i) {
            if (out[i] != static_cast<float>(begin + i)) {
                ++torn;
                break;
            }
        }
    }
    producer.join();
    EXPECT_EQ(torn, 0);
    EXPECT_GT(reads, 0);
}

}  // namespace
