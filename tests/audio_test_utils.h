#pragma once
// Shared helpers for the audio-frontend parity tests. The golden dumps are FP32
// (see scripts/generate_ultravox_audio_dumps.py), so load_bin_file reads float32;
// cosine similarity accumulates in double.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace audio_test {

inline std::vector<float> load_bin_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::fprintf(stderr, "FATAL: cannot open %s\n", path.c_str());
        std::exit(2);
    }
    const std::streamsize bytes = f.tellg();
    f.seekg(0);
    std::vector<float> v(static_cast<size_t>(bytes) / sizeof(float));
    f.read(reinterpret_cast<char*>(v.data()), bytes);
    return v;
}

inline double compute_cosine_similarity(const std::vector<float>& a,
                                        const std::vector<float>& b) {
    if (a.size() != b.size()) {
        std::fprintf(stderr, "FATAL: size mismatch %zu vs %zu\n", a.size(), b.size());
        std::exit(2);
    }
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double x = a[i], y = b[i];
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
}

}  // namespace audio_test
