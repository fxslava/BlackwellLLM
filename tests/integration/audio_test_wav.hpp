#pragma once
// =============================================================================
// tests/integration/audio_test_wav.hpp — shared fixtures for the audio
// integration tests: asset paths, a minimal WAV reader, the reference clip's
// hardcoded split geometry, and word-level transcript comparison.
//
// ONE PLACE FOR THE MEL FILTERBANK PATH, and that is the reason this header
// exists. Two copies of mel_filters.bin live in this repo with IDENTICAL byte
// counts and TRANSPOSED layouts:
//
//   audio_sandbox/data/mel_filters.bin               [n_freqs=201, n_mels=128]
//                                                    freq-major — what
//                                                    WhisperDSP indexes. CORRECT.
//   tests/.../golden_dumps/ultravox/mel_filters.bin  [128, 201] mel-major — the
//                                                    layout the Python golden
//                                                    dumps compare against.
//
// Loading the second one does not fail, does not warn, and does not crash: it
// produces plausible-looking log-mel that makes the model emit fluent text about
// audio it never heard. WhisperDSP now rejects it (validate_mel_layout), but the
// real fix is that no test picks the path by hand any more.
// =============================================================================
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace blackwell_test_audio {

inline std::string env_or(const char* n, const std::string& f) {
    const char* v = std::getenv(n);
    return (v && *v) ? std::string(v) : f;
}
inline bool file_exists(const std::string& p) { return std::ifstream(p).good(); }

// ---- the shipping-default model pair (both 8B; a 1B projector against an 8B
//      backbone is a dimension mismatch the app rejects up front) -------------
inline std::string awq_index_path() {
    return env_or("BLACKWELL_AWQ_INDEX",
                  "F:/AI/llama-3.1-8B-Instruct-AWQ-INT4/model.safetensors.index.json");
}
inline std::string audio_head_dir() {
    return env_or("BLACKWELL_AUDIO_HEAD", "F:/AI/ultravox-v0_5-llama-3_1-8b");
}
inline std::string model_dir_of(const std::string& index) {
    const auto cut = index.find_last_of("/\\");
    return cut == std::string::npos ? std::string(".") : index.substr(0, cut);
}
inline std::string repo_root() {
    return env_or("BLACKWELL_REPO_ROOT", "D:/Projects/BlackwellLLM");
}
inline std::string dumps_dir() {
    return env_or("BLACKWELL_ULTRAVOX_DUMPS",
                  repo_root() + "/tests/integration/golden_dumps/ultravox");
}
// THE FREQ-MAJOR ONE. Never the dumps copy — see the header block.
inline std::string mel_filters_path() {
    return env_or("BLACKWELL_MEL_FILTERS", repo_root() + "/audio_sandbox/data/mel_filters.bin");
}
// The reference utterance the golden dumps were generated from (LibriSpeech
// dummy[0]); its transcript is recorded in golden_dumps/ultravox/meta.json:
//   "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES
//    AND WE ARE GLAD TO WELCOME HIS GOSPEL"
inline std::string reference_wav_path() { return dumps_dir() + "/test_audio.wav"; }

// ---- the reference clip's hardcoded VAD geometry ---------------------------
// kSplitSample was chosen from the clip's own energy profile: 3.24-3.32 s is the
// only true silence valley in the interior (-62 dB, the room-tone floor, against
// ~-20 dB of speech) and it is the pause between "...MIDDLE CLASSES" and "AND WE
// ARE...". A perfectly timed pause, exactly between words — and a literal
// constant, so a regression cannot move the goalposts.
//
// It is also arithmetically exact. The DSP emits one mel frame per 160 samples
// and the encoder one soft-token per 16 mel frames, so
//   whole 93680 -> 585 frames -> ceil(585/16) = 37 soft-tokens
//   half A 52800 -> 330 frames -> ceil(330/16) = 21
//   half B 40880 -> 255 frames -> ceil(255/16) = 16      and 21 + 16 == 37.
// The halves partition the clip's soft-token budget EXACTLY, which is what makes
// "nothing lost, nothing duplicated" an equation rather than a judgement call.
inline constexpr int         kSampleRate       = 16000;
inline constexpr std::size_t kClipSamples      = 93680;   // 5.855 s
inline constexpr std::size_t kSplitSample      = 52800;   // 3.300 s
inline constexpr int         kExpectTokensA    = 21;
inline constexpr int         kExpectTokensB    = 16;
inline constexpr int         kExpectTokensWhole = 37;

// Minimal mono-16k WAV reader (PCM16 / IEEE-float32). Local to the test tier:
// the repo's dr_wav wrapper lives in an app target the suites do not link, and
// the one file this reads is a fixed 16 kHz mono clip we generated ourselves.
// Returns empty on anything malformed or unsupported; callers SKIP.
inline std::vector<float> load_wav_mono16k(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    auto u16 = [](const unsigned char* p) {
        return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8);
    };
    auto u32 = [](const unsigned char* p) {
        return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
               (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
    };
    unsigned char hdr[12];
    f.read(reinterpret_cast<char*>(hdr), 12);
    if (!f || u32(hdr) != 0x46464952u /*RIFF*/ || u32(hdr + 8) != 0x45564157u /*WAVE*/)
        return {};

    uint32_t fmt = 0, channels = 0, rate = 0, bits = 0;
    while (f) {
        unsigned char ch[8];
        f.read(reinterpret_cast<char*>(ch), 8);
        if (!f) break;
        const uint32_t id = u32(ch), size = u32(ch + 4);
        std::vector<unsigned char> body(size);
        f.read(reinterpret_cast<char*>(body.data()), size);
        if (!f) break;
        if (id == 0x20746D66u /*"fmt "*/ && size >= 16) {
            fmt = u16(body.data());
            channels = u16(body.data() + 2);
            rate = u32(body.data() + 4);
            bits = u16(body.data() + 14);
        } else if (id == 0x61746164u /*"data"*/) {
            if (channels == 0 || rate != static_cast<uint32_t>(kSampleRate)) return {};
            const uint32_t bps = bits / 8;
            if (bps == 0) return {};
            const std::size_t frames = size / (bps * channels);
            std::vector<float> out(frames);
            for (std::size_t i = 0; i < frames; ++i) {
                double acc = 0.0;   // downmix by average (the clip is mono anyway)
                for (uint32_t c = 0; c < channels; ++c) {
                    const unsigned char* s = body.data() + (i * channels + c) * bps;
                    if (fmt == 3 && bits == 32) {
                        float v;
                        std::memcpy(&v, s, sizeof(v));
                        acc += v;
                    } else if (fmt == 1 && bits == 16) {
                        acc += static_cast<int16_t>(u16(s)) / 32768.0;
                    } else {
                        return {};   // unsupported encoding
                    }
                }
                out[i] = static_cast<float>(acc / channels);
            }
            return out;
        }
        if (size & 1u) f.seekg(1, std::ios::cur);   // RIFF word alignment
    }
    return {};
}

// Lowercase alphanumeric words — the only comparison a generative transcript
// supports honestly (casing, punctuation and "Mr."/"Mister" are the model's
// business; the CONTENT is what a broken frontend or boundary destroys).
inline std::vector<std::string> words_of(const std::string& s) {
    std::vector<std::string> w;
    std::string cur;
    for (const char c : s) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            cur += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        } else if (!cur.empty()) {
            w.push_back(cur);
            cur.clear();
        }
    }
    if (!cur.empty()) w.push_back(cur);
    return w;
}
inline bool has_word(const std::string& s, const char* word) {
    const std::vector<std::string> w = words_of(s);
    return std::find(w.begin(), w.end(), std::string(word)) != w.end();
}

}  // namespace blackwell_test_audio
