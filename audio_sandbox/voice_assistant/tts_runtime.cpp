// -----------------------------------------------------------------------------
// TtsRuntime::LoadWav24kMono — the one out-of-line member.
//
// A minimal RIFF/WAVE reader for the reference clip: 16-bit PCM and 32-bit
// float, any channel count (downmixed). Kept out of the header so <fstream> and
// the chunk walking do not land in every TU that includes tts_runtime.hpp.
// -----------------------------------------------------------------------------
#include "tts_runtime.hpp"

#include <cstring>
#include <fstream>

namespace rt {
namespace {

std::uint32_t rd32(const unsigned char* p) {
    return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}
std::uint16_t rd16(const unsigned char* p) {
    return static_cast<std::uint16_t>(static_cast<std::uint16_t>(p[0]) |
                                      (static_cast<std::uint16_t>(p[1]) << 8));
}

}  // namespace

std::vector<float> TtsRuntime::LoadWav24kMono(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("TtsRuntime: cannot open reference WAV: " + path);

    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(f)),
                                     std::istreambuf_iterator<char>());
    if (bytes.size() < 44 || std::memcmp(bytes.data(), "RIFF", 4) != 0 ||
        std::memcmp(bytes.data() + 8, "WAVE", 4) != 0) {
        throw std::runtime_error("TtsRuntime: not a RIFF/WAVE file: " + path);
    }

    // Walk the chunk list rather than assuming a 44-byte header: real files carry
    // LIST/INFO and fact chunks, and a fixed offset silently reads metadata as
    // samples (which sounds like a burst of noise at the start of the clip).
    std::uint16_t format = 0, channels = 0, bits = 0;
    std::uint32_t rate = 0;
    const unsigned char* data = nullptr;
    std::size_t data_bytes = 0;

    std::size_t pos = 12;
    while (pos + 8 <= bytes.size()) {
        const unsigned char* id = bytes.data() + pos;
        const std::uint32_t sz = rd32(bytes.data() + pos + 4);
        const std::size_t body = pos + 8;
        if (body + sz > bytes.size()) break;   // truncated chunk: stop, keep what we have

        if (std::memcmp(id, "fmt ", 4) == 0 && sz >= 16) {
            const unsigned char* b = bytes.data() + body;
            format   = rd16(b);
            channels = rd16(b + 2);
            rate     = rd32(b + 4);
            bits     = rd16(b + 14);
        } else if (std::memcmp(id, "data", 4) == 0) {
            data = bytes.data() + body;
            data_bytes = sz;
        }
        pos = body + sz + (sz & 1u);   // chunks are word-aligned
    }

    if (data == nullptr || channels == 0) {
        throw std::runtime_error("TtsRuntime: WAV has no fmt/data chunk: " + path);
    }
    if (rate != static_cast<std::uint32_t>(blackwell::tts::kF5SampleRate)) {
        // NOT resampled here on purpose: a wrong-rate reference does not fail,
        // it clones a voice pitched by the rate ratio, which sounds like a
        // different person and reads as a model problem.
        throw std::runtime_error(
            "TtsRuntime: reference WAV is " + std::to_string(rate) + " Hz; F5 needs " +
            std::to_string(blackwell::tts::kF5SampleRate) +
            " Hz. Convert it: ffmpeg -i \"" + path + "\" -ar 24000 -ac 1 ref_24k.wav");
    }

    std::vector<float> mono;
    const std::size_t ch = channels;

    if (format == 3 && bits == 32) {                 // IEEE float
        const std::size_t n = data_bytes / 4;
        mono.reserve(n / ch);
        for (std::size_t i = 0; i + ch <= n; i += ch) {
            float acc = 0.0f;
            for (std::size_t c = 0; c < ch; ++c) {
                float v = 0.0f;
                std::memcpy(&v, data + (i + c) * 4, 4);
                acc += v;
            }
            mono.push_back(acc / static_cast<float>(ch));
        }
    } else if (format == 1 && bits == 16) {          // signed 16-bit PCM
        const std::size_t n = data_bytes / 2;
        mono.reserve(n / ch);
        for (std::size_t i = 0; i + ch <= n; i += ch) {
            float acc = 0.0f;
            for (std::size_t c = 0; c < ch; ++c) {
                std::int16_t v = 0;
                std::memcpy(&v, data + (i + c) * 2, 2);
                acc += static_cast<float>(v) / 32768.0f;
            }
            mono.push_back(acc / static_cast<float>(ch));
        }
    } else {
        throw std::runtime_error(
            "TtsRuntime: unsupported WAV encoding (format=" + std::to_string(format) +
            ", bits=" + std::to_string(bits) + ") in " + path +
            ". Use 16-bit PCM or 32-bit float.");
    }

    if (mono.empty()) {
        throw std::runtime_error("TtsRuntime: reference WAV has no samples: " + path);
    }
    return mono;
}

}  // namespace rt
