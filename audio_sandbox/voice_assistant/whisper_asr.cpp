// -----------------------------------------------------------------------------
// whisper_asr.cpp — THE only TU that sees <whisper.h>. See whisper_asr.hpp.
// -----------------------------------------------------------------------------
#include "whisper_asr.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

// Angle brackets, deliberately: whisper.cpp's include dirs arrive as SYSTEM on
// blackwell::whisper_cpp, so /external:anglebrackets drops them to /W0 and the
// root's /W4 /WX budget keeps pointing at first-party code only.
#include <whisper.h>

#if defined(BLACKWELL_HAVE_WHISPER_CUDA)
#include <cuda_runtime.h>
#endif

namespace rt {
namespace {

// Whisper emits these INSTEAD of a transcript when it decides a window carries
// no speech. They are not transcripts and must never become intents: publishing
// "[BLANK_AUDIO]" would answer it as if the user had said it out loud.
//
// Matched case-insensitively against the WHOLE trimmed transcript, never as a
// substring -- a real sentence containing the word "silence" is a real sentence.
constexpr const char* kNonSpeechArtifacts[] = {
    "[blank_audio]", "(blank_audio)", "[silence]", "(silence)",
    "[ silence ]",   "[music]",       "(music)",   "[sound]",
    "[noise]",       "(noise)",       "[inaudible]", "(inaudible)",
    ".",             "..",            "...",       "…",
};

std::string trim(const std::string& s) {
    const auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return {};
    const auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

std::string to_lower_ascii(std::string s) {
    for (char& c : s) {
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    }
    return s;
}

bool is_non_speech(const std::string& trimmed) {
    if (trimmed.empty()) return true;
    const std::string lower = to_lower_ascii(trimmed);
    for (const char* artifact : kNonSpeechArtifacts) {
        if (lower == artifact) return true;
    }
    return false;
}

// ggml/whisper log sink. Installed process-wide (the API is global) so the
// library's chatter carries a prefix and the noisiest levels are dropped.
//
// THIS IS NOT COSMETIC. Without it ggml writes unprefixed lines to stderr
// interleaved with the pipeline's own diagnostics -- and this app's log is how
// a stalled turn gets diagnosed. INFO and DEBUG are dropped because ggml logs
// its entire tensor-by-tensor load at INFO, which buries everything else on
// every launch; WARN and ERROR are kept because those are the lines that explain
// a failed encode.
void whisper_log_sink(ggml_log_level level, const char* text, void* /*user*/) {
    if (text == nullptr) return;
    if (level != GGML_LOG_LEVEL_ERROR && level != GGML_LOG_LEVEL_WARN) return;
    std::fprintf(stderr, "[whisper.cpp] %s", text);
}

}  // namespace

struct WhisperAsr::Impl {
    whisper_context* ctx = nullptr;
    // OWNS the language string. whisper_full_params::language is a borrowed
    // const char*, read inside whisper_full, so it must outlive every call --
    // pointing it at a temporary is a use-after-free that usually "works".
    std::string language;
    std::string model_path;
    int  n_threads = 4;
    bool gpu = false;

    std::atomic<std::uint64_t> utterances{0};
    std::atomic<std::uint64_t> failures{0};
    std::atomic<double>        last_encode_ms{0.0};

    ~Impl() {
        if (ctx != nullptr) whisper_free(ctx);
    }
};

WhisperAsr::WhisperAsr(const WhisperAsrConfig& cfg) : impl_(std::make_unique<Impl>()) {
    if (cfg.model_path.empty()) {
        throw std::runtime_error("WhisperAsr: no model path configured");
    }

    whisper_log_set(&whisper_log_sink, nullptr);

    impl_->model_path = cfg.model_path;
    impl_->n_threads  = cfg.n_threads < 1 ? 1 : cfg.n_threads;
    // "auto" rather than an empty string or nullptr: that is the literal token
    // whisper.cpp's language table maps to "detect it yourself". An empty string
    // resolves to no language at all and the call fails.
    impl_->language = cfg.language.empty() ? std::string("auto") : cfg.language;

#if defined(BLACKWELL_HAVE_WHISPER_CUDA)
    impl_->gpu = cfg.use_gpu;
#else
    // Asked for the GPU on a CPU-backend build. Say so once rather than letting
    // it look like the GPU is in use and merely slow: this is a ~10x latency
    // difference and it is the first thing anyone will suspect.
    if (cfg.use_gpu) {
        std::fprintf(stderr,
                     "[whisper] NOTE: GPU requested but this build has the ggml CPU backend "
                     "(-DBLACKWELL_WHISPER_CUDA=OFF) -- transcription will run on the CPU.\n");
    }
    impl_->gpu = false;
#endif

#if defined(BLACKWELL_HAVE_WHISPER_CUDA)
    // CUDA's current device is PER-THREAD, and ggml creates its context lazily
    // on whichever device is current when it first allocates. Selecting it here
    // -- in the constructor, which the owner thread runs -- is what keeps the
    // model on the same device the rest of the stack is using. This is the same
    // trap rt::select_cuda_device documents for the engine thread.
    if (impl_->gpu) {
        const cudaError_t rc = cudaSetDevice(cfg.gpu_device);
        if (rc != cudaSuccess) {
            throw std::runtime_error(std::string("WhisperAsr: cudaSetDevice(") +
                                     std::to_string(cfg.gpu_device) +
                                     ") failed: " + cudaGetErrorString(rc));
        }
    }
#endif

    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu    = impl_->gpu;
    cparams.gpu_device = cfg.gpu_device;
    // NOTE: flash attention is deliberately left at the library default. Its
    // field has changed shape across whisper.cpp releases (bool -> enum), and
    // pinning it here would make a routine tag bump a compile error for a
    // few percent of encode time.

    impl_->ctx = whisper_init_from_file_with_params(cfg.model_path.c_str(), cparams);
    if (impl_->ctx == nullptr) {
        // The path is in the message because every realistic cause is about the
        // file: absent, unreadable, or not a GGML model at all (pointing this at
        // a .safetensors checkpoint is the predictable mistake).
        throw std::runtime_error("WhisperAsr: could not load the GGML model at '" +
                                 cfg.model_path + "' (missing, unreadable, or not a "
                                 "GGML/GGUF whisper model)");
    }
}

WhisperAsr::~WhisperAsr() = default;

std::string WhisperAsr::Transcribe(const float* pcm16k, std::size_t count) {
    if (impl_->ctx == nullptr || pcm16k == nullptr) return {};

    // Whisper's encoder works on a 30 s window and zero-pads anything shorter, so
    // a very brief clip is mostly padding and decodes badly. WHISPER_SAMPLE_RATE
    // is the library's own constant (16 kHz) -- using it rather than a literal is
    // what makes this line still correct if the caller ever resamples.
    constexpr std::size_t kMinSamples = WHISPER_SAMPLE_RATE / 2;   // 500 ms
    if (count < kMinSamples) return {};

    // 30 s ceiling: the encoder's positional embeddings stop there, so a longer
    // window cannot be encoded at its true offsets. Keep the TAIL rather than the
    // head -- the most recent speech is what the user is waiting on an answer to,
    // and the terminal prosody that carries the punctuation lives at the end.
    constexpr std::size_t kMaxSamples = static_cast<std::size_t>(WHISPER_SAMPLE_RATE) * 30;
    if (count > kMaxSamples) {
        pcm16k += (count - kMaxSamples);
        count = kMaxSamples;
    }

    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.n_threads        = impl_->n_threads;
    wparams.language         = impl_->language.c_str();   // owned by Impl; see the field
    // NO CARRIED CONTEXT. Each utterance is independent: with no_context off,
    // whisper conditions on the previous window's tokens and, on a conversational
    // stream of short turns, happily continues the previous sentence instead of
    // transcribing this one.
    wparams.no_context       = true;
    // One utterance in, one segment out. The caller already did the segmentation
    // with a VAD that knows about pre-roll and hangover; letting whisper re-split
    // would produce fragments the commit gate would then dispatch separately.
    wparams.single_segment   = true;
    // TRANSLATION IS THE BACKBONE'S JOB, not whisper's. whisper_full's translate
    // flag only ever targets ENGLISH, which is the wrong answer for a pipeline
    // whose whole point is that a real language model handles the target language.
    wparams.translate        = false;
    // The library prints to stdout by default, which would interleave with -- and
    // in places corrupt -- the pipeline diagnostics this app is read through.
    wparams.print_realtime   = false;
    wparams.print_progress   = false;
    wparams.print_timestamps = false;
    wparams.print_special    = false;
    // Timestamps cost decoder passes and nothing here consumes them.
    wparams.token_timestamps = false;
    // suppress_blank and the temperature fallback ladder are left at their
    // defaults ON PURPOSE. They are what let whisper punt on a genuinely
    // unintelligible window instead of confabulating a sentence, and turning
    // them off to "get more transcripts" gets more WRONG transcripts.

    const auto t0 = std::chrono::steady_clock::now();
    const int rc = whisper_full(impl_->ctx, wparams, pcm16k, static_cast<int>(count));
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0).count();
    impl_->last_encode_ms.store(ms, std::memory_order_relaxed);

    if (rc != 0) {
        impl_->failures.fetch_add(1, std::memory_order_relaxed);
        std::fprintf(stderr, "[whisper] encode FAILED (whisper_full returned %d) -- "
                             "utterance dropped\n", rc);
        std::fflush(stderr);
        return {};
    }

    std::string text;
    const int n = whisper_full_n_segments(impl_->ctx);
    for (int i = 0; i < n; ++i) {
        const char* seg = whisper_full_get_segment_text(impl_->ctx, i);
        if (seg != nullptr) text += seg;
    }

    // Whisper prefixes every segment with a space, and single_segment still
    // yields leading/trailing whitespace often enough to matter: an untrimmed
    // transcript becomes an intent with a leading space, which changes the
    // prompt bytes and therefore the cloud leg's prompt-cache key.
    text = trim(text);
    if (is_non_speech(text)) return {};

    impl_->utterances.fetch_add(1, std::memory_order_relaxed);
    return text;
}

bool WhisperAsr::gpu() const noexcept { return impl_->gpu; }
const std::string& WhisperAsr::model_path() const noexcept { return impl_->model_path; }

std::uint64_t WhisperAsr::utterances() const noexcept {
    return impl_->utterances.load(std::memory_order_relaxed);
}
std::uint64_t WhisperAsr::failures() const noexcept {
    return impl_->failures.load(std::memory_order_relaxed);
}
double WhisperAsr::last_encode_ms() const noexcept {
    return impl_->last_encode_ms.load(std::memory_order_relaxed);
}

}  // namespace rt
