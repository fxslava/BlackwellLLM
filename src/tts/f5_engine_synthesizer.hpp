#pragma once
// -----------------------------------------------------------------------------
// F5EngineSynthesizer — adapts F5TtsEngine to the ISynthesizer seam.
//
// This header is the ONLY place the duplex bridge and the CUDA engine meet, and
// it lives in blackwell_tts_f5 rather than blackwell_tts on purpose: naming
// F5TtsEngine from the bridge's own target would pull ONNXRuntime and cudart
// into a library whose other consumers are a text frontend and a CPU-only test
// suite (see the block at the top of src/tts/CMakeLists.txt).
//
// It carries the one piece of state the seam cannot: the total_frames_hint.
// F5's duration formula needs UTF-8 BYTE lengths of the reference and generated
// text, which only the caller has -- the seam passes token ids. So the caller
// sets the reference geometry once (SetReferenceGeometry) and this adapter
// reproduces
//     total = ref_len + ref_len * gen_bytes / ref_bytes / speed
// per chunk. Leave it unset and the engine's own token-count estimate is used,
// which is close for Cyrillic and wrong in general.
//
// THREADING: F5TtsEngine requires ONE owning thread. That thread is the TTS
// worker, which is also the only thread that calls PumpOnce, so the adapter adds
// no synchronisation of its own.
// -----------------------------------------------------------------------------
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "f5_tts_engine.hpp"
#include "tts_duplex_bridge.hpp"

namespace blackwell::tts {

class F5EngineSynthesizer final : public ISynthesizer {
public:
    explicit F5EngineSynthesizer(F5TtsEngine& engine) noexcept : engine_(engine) {}

    // Reference geometry for F5's byte-ratio duration formula.
    //   ref_samples   - length of the reference clip in samples (NOT mel frames;
    //                   F5 uses n_samples/hop, which is one less than the frame
    //                   count that center-padding produces)
    //   ref_text_utf8 - the reference transcript AS TOKENISED, including the
    //                   trailing space F5 appends when it ends in a 1-byte char
    void SetReferenceGeometry(std::size_t ref_samples,
                              std::string_view ref_text_utf8,
                              float speed = 1.0f) noexcept {
        ref_frames_ = ref_samples / static_cast<std::size_t>(kF5HopLength);
        ref_bytes_  = ref_text_utf8.size();
        speed_      = speed > 0.0f ? speed : 1.0f;
    }

    // Bytes of the chunk currently being synthesised. Set by the caller right
    // before the chunk is handed over; the seam only carries ids, and ids cannot
    // be converted back to a byte count.
    void SetChunkBytes(std::size_t gen_bytes) noexcept { gen_bytes_ = gen_bytes; }

    TtsStatus Synthesize(const std::vector<std::int64_t>& text_ids,
                         const std::atomic<bool>* cancel,
                         std::vector<float>& out_pcm) noexcept override {
        std::size_t hint = 0;
        if (ref_frames_ != 0 && ref_bytes_ != 0 && gen_bytes_ != 0) {
            hint = ref_frames_ +
                   static_cast<std::size_t>(static_cast<double>(ref_frames_) *
                                            static_cast<double>(gen_bytes_) /
                                            static_cast<double>(ref_bytes_) /
                                            static_cast<double>(speed_));
        }
        // GenerateAudio threads `cancel` into the ODE loop itself, so a barge-in
        // lands within one solver step rather than at the end of the chunk.
        return engine_.GenerateAudio(text_ids, cancel, out_pcm, hint);
    }

    int sample_rate() const noexcept override { return kF5SampleRate; }

private:
    F5TtsEngine& engine_;
    std::size_t  ref_frames_ = 0;
    std::size_t  ref_bytes_  = 0;
    std::size_t  gen_bytes_  = 0;
    float        speed_      = 1.0f;
};

}  // namespace blackwell::tts
