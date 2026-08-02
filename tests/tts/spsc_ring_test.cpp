// =============================================================================
// tests/tts/spsc_ring_test.cpp
//
// Phase 1 of docs/TTS_INTEGRATION_AUDIT.md: the lock-free primitive the whole
// audio graph is about to be rebuilt on. It ends up under the playback sink, the
// capture ring and the AEC reference tap, so a bug here is an audible artifact
// in three places at once — which is why this file is the most paranoid in the
// suite.
//
// CPU-only, no ONNXRuntime, no model, no audio device.
//
// The load-bearing assertions are Wraparound and ConcurrentStress. The first
// pins the split-copy path (a span crossing the end of the buffer is memcpy'd in
// two pieces, and getting the second piece wrong corrupts audio only when the
// write happens to straddle the boundary — i.e. rarely, and never in a short
// test that does not force it). The second pins the memory ordering: the
// producer publishes with release and the consumer acquires, and if that pairing
// is wrong the consumer reads slots whose contents are not yet visible.
// =============================================================================
#include "spsc_ring.hpp"

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

namespace {

using blackwell::audio_rt::SpscRing;

// A distinctive, exactly-representable float per index, so a corrupted or
// misordered sample is identifiable rather than merely "wrong". Integers up to
// 2^24 are exact in f32, which bounds how long a sequence check stays meaningful.
float sample_at(std::uint64_t i) { return static_cast<float>(i % (1u << 24)); }

// ---- geometry ---------------------------------------------------------------

TEST(SpscRing, CapacityRoundsUpToPowerOfTwo) {
    EXPECT_EQ(SpscRing<float>(1).capacity(), 2u);      // floored at 2
    EXPECT_EQ(SpscRing<float>(2).capacity(), 2u);
    EXPECT_EQ(SpscRing<float>(3).capacity(), 4u);
    EXPECT_EQ(SpscRing<float>(1000).capacity(), 1024u);
    EXPECT_EQ(SpscRing<float>(24000).capacity(), 32768u);   // ~500 ms @ 48 kHz
    EXPECT_EQ(SpscRing<float>(32768).capacity(), 32768u);
}

TEST(SpscRing, StartsEmpty) {
    SpscRing<float> ring(16);
    EXPECT_EQ(ring.available(), 0u);
    EXPECT_EQ(ring.space(), ring.capacity());
    EXPECT_EQ(ring.overruns(), 0u);
    EXPECT_EQ(ring.underruns(), 0u);
}

// ---- single-threaded correctness --------------------------------------------

TEST(SpscRing, WriteThenReadRoundTrips) {
    SpscRing<float> ring(16);
    const std::vector<float> in{1.0f, 2.0f, 3.0f, 4.0f};

    ASSERT_EQ(ring.write(in.data(), in.size()), in.size());
    EXPECT_EQ(ring.available(), in.size());

    std::vector<float> out(in.size(), -1.0f);
    ASSERT_EQ(ring.read(out.data(), out.size()), in.size());
    EXPECT_EQ(in, out);
    EXPECT_EQ(ring.available(), 0u);
}

TEST(SpscRing, NullAndZeroCountAreNoOps) {
    SpscRing<float> ring(16);
    float scratch = 0.0f;
    EXPECT_EQ(ring.write(nullptr, 4), 0u);
    EXPECT_EQ(ring.write(&scratch, 0), 0u);
    EXPECT_EQ(ring.write_or_drop(nullptr, 4), 0u);
    EXPECT_EQ(ring.write_or_drop(&scratch, 0), 0u);
    EXPECT_EQ(ring.read(nullptr, 4), 0u);
    EXPECT_EQ(ring.read(&scratch, 0), 0u);
    ring.read_or_silence(nullptr, 4);
    EXPECT_EQ(ring.available(), 0u);
    // A malformed call is a no-op, not a fault: neither counter may move, or a
    // null-guard would masquerade as lost audio.
    EXPECT_EQ(ring.overruns(), 0u);
    EXPECT_EQ(ring.underruns(), 0u);
}

// The split-copy path: force a span to straddle the end of the buffer, which a
// naive single-memcpy implementation gets wrong only in exactly this case.
TEST(SpscRing, WraparoundPreservesOrderAcrossManyLaps) {
    SpscRing<float> ring(8);              // small on purpose: laps come fast
    ASSERT_EQ(ring.capacity(), 8u);

    std::uint64_t produced = 0;
    std::uint64_t consumed = 0;
    std::vector<float> in(5);
    std::vector<float> out(5);

    // 5 in / 5 out against a capacity of 8 guarantees the offset walks the whole
    // buffer and every write eventually straddles the boundary.
    for (int iter = 0; iter < 200; ++iter) {
        for (std::size_t i = 0; i < in.size(); ++i) in[i] = sample_at(produced + i);
        ASSERT_EQ(ring.write(in.data(), in.size()), in.size()) << "iter " << iter;
        produced += in.size();

        ASSERT_EQ(ring.read(out.data(), out.size()), out.size()) << "iter " << iter;
        for (std::size_t i = 0; i < out.size(); ++i) {
            ASSERT_FLOAT_EQ(out[i], sample_at(consumed + i)) << "iter " << iter << " idx " << i;
        }
        consumed += out.size();
    }
    EXPECT_EQ(ring.overruns(), 0u);
    EXPECT_EQ(ring.underruns(), 0u);
    EXPECT_EQ(ring.total_written(), produced);
    EXPECT_EQ(ring.total_read(), consumed);
}

// ---- fault accounting -------------------------------------------------------

// The producer rejects the NEWEST samples rather than overwriting the oldest.
// That is the opposite of SampleRing's policy and it is deliberate: audio already
// queued is audio about to be heard.
TEST(SpscRing, WriteOrDropDiscardsNewestAndCounts) {
    SpscRing<float> ring(4);
    const std::vector<float> first{10.0f, 11.0f, 12.0f, 13.0f};
    ASSERT_EQ(ring.write_or_drop(first.data(), first.size()), 4u);
    EXPECT_EQ(ring.space(), 0u);

    const std::vector<float> overflow{99.0f, 98.0f};
    EXPECT_EQ(ring.write_or_drop(overflow.data(), overflow.size()), 0u);
    EXPECT_EQ(ring.overruns(), 2u);

    // The buffered audio is untouched — the drop cost us the new samples, not the
    // queued ones.
    std::vector<float> out(4, -1.0f);
    ASSERT_EQ(ring.read(out.data(), out.size()), 4u);
    EXPECT_EQ(out, first);
}

TEST(SpscRing, PartialWriteOrDropAcceptsWhatFitsAndCountsTheRest) {
    SpscRing<float> ring(4);
    const std::vector<float> two{1.0f, 2.0f};
    ASSERT_EQ(ring.write_or_drop(two.data(), two.size()), 2u);

    const std::vector<float> four{3.0f, 4.0f, 5.0f, 6.0f};
    EXPECT_EQ(ring.write_or_drop(four.data(), four.size()), 2u);   // only 2 slots left
    EXPECT_EQ(ring.overruns(), 2u);

    std::vector<float> out(4, -1.0f);
    ASSERT_EQ(ring.read(out.data(), out.size()), 4u);
    EXPECT_EQ(out, (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}));
}

// The asymmetry that matters: a producer which RETRIES lost nothing, so plain
// write() must not register a fault. Counting per call would make one retried
// block look like dozens of drops and render the metric useless for the case it
// exists to detect.
TEST(SpscRing, ShortWriteIsNotAnOverrun) {
    SpscRing<float> ring(4);
    const std::vector<float> six{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    EXPECT_EQ(ring.write(six.data(), six.size()), 4u);
    EXPECT_EQ(ring.overruns(), 0u);

    // Retrying the remainder after a drain places it intact.
    std::vector<float> out(4, -1.0f);
    ASSERT_EQ(ring.read(out.data(), out.size()), 4u);
    EXPECT_EQ(ring.write(six.data() + 4, 2), 2u);
    EXPECT_EQ(ring.overruns(), 0u);

    ASSERT_EQ(ring.read(out.data(), 2), 2u);
    EXPECT_FLOAT_EQ(out[0], 5.0f);
    EXPECT_FLOAT_EQ(out[1], 6.0f);
}

// A short plain read() is NOT a fault — that is the DSP worker's normal idle
// case, and counting it there would make the metric meaningless.
TEST(SpscRing, ShortReadIsNotAnUnderrun) {
    SpscRing<float> ring(16);
    const std::vector<float> in{1.0f, 2.0f};
    ASSERT_EQ(ring.write(in.data(), in.size()), 2u);

    std::vector<float> out(8, -1.0f);
    EXPECT_EQ(ring.read(out.data(), out.size()), 2u);
    EXPECT_EQ(ring.underruns(), 0u);
}

// read_or_silence() is the variant that DOES declare a short read a fault: a
// playback callback must return a full buffer every time.
TEST(SpscRing, ReadOrSilencePadsAndCounts) {
    SpscRing<float> ring(16);
    const std::vector<float> in{1.0f, 2.0f};
    ASSERT_EQ(ring.write(in.data(), in.size()), 2u);

    std::vector<float> out(5, -1.0f);
    ring.read_or_silence(out.data(), out.size());
    EXPECT_EQ(out, (std::vector<float>{1.0f, 2.0f, 0.0f, 0.0f, 0.0f}));
    EXPECT_EQ(ring.underruns(), 3u);

    // A fully-starved sink is all silence, and every sample is counted.
    ring.read_or_silence(out.data(), out.size());
    for (float v : out) EXPECT_FLOAT_EQ(v, 0.0f);
    EXPECT_EQ(ring.underruns(), 8u);
}

TEST(SpscRing, ResetClearsContentsAndCounters) {
    SpscRing<float> ring(4);
    const std::vector<float> in{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    ring.write_or_drop(in.data(), in.size());   // one sample dropped
    ASSERT_EQ(ring.overruns(), 1u);

    ring.reset();
    EXPECT_EQ(ring.available(), 0u);
    EXPECT_EQ(ring.space(), ring.capacity());
    EXPECT_EQ(ring.overruns(), 0u);
    EXPECT_EQ(ring.underruns(), 0u);
    EXPECT_EQ(ring.total_written(), 0u);
    EXPECT_EQ(ring.total_read(), 0u);
}

// ---- the real thing ---------------------------------------------------------

// Two threads, no locks, small ring, deliberately mismatched block sizes so the
// buffer runs both full and empty repeatedly. What is under test is not
// throughput but INTEGRITY: every sample the consumer sees must be the next one
// in the producer's sequence. A release/acquire mistake shows up here as a
// sample read before its contents were published.
TEST(SpscRing, ConcurrentStressPreservesTheSequence) {
    constexpr std::uint64_t kTotal = 2'000'000;   // ~40 s of 48 kHz mono
    SpscRing<float> ring(1024);

    std::atomic<bool> corrupt{false};
    std::atomic<std::uint64_t> bad_index{0};

    std::thread producer([&] {
        std::vector<float> block(97);             // coprime with the capacity
        std::uint64_t written = 0;
        while (written < kTotal) {
            const std::size_t n = static_cast<std::size_t>(
                block.size() < (kTotal - written) ? block.size() : (kTotal - written));
            for (std::size_t i = 0; i < n; ++i) block[i] = sample_at(written + i);
            std::size_t done = 0;
            while (done < n) {                    // spin until it fits: no drops here
                done += ring.write(block.data() + done, n - done);
            }
            written += n;
        }
    });

    std::thread consumer([&] {
        std::vector<float> block(53);             // also coprime, and != the writer's
        std::uint64_t read = 0;
        while (read < kTotal) {
            const std::size_t got = ring.read(block.data(), block.size());
            for (std::size_t i = 0; i < got; ++i) {
                if (block[i] != sample_at(read + i)) {
                    if (!corrupt.exchange(true)) bad_index.store(read + i);
                }
            }
            read += got;
        }
    });

    producer.join();
    consumer.join();

    EXPECT_FALSE(corrupt.load()) << "sequence broke at sample " << bad_index.load();
    EXPECT_EQ(ring.total_written(), kTotal);
    EXPECT_EQ(ring.total_read(), kTotal);
    EXPECT_EQ(ring.available(), 0u);
    EXPECT_EQ(ring.overruns(), 0u);   // the producer spun rather than dropping
}

}  // namespace
