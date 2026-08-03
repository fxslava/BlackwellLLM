#pragma once
// -----------------------------------------------------------------------------
// F5MelExtractor — the log-mel front end F5-TTS conditions on. Raw 24 kHz mono
// PCM in, [frames, 100] time-major natural-log mel out.
//
// WHAT THIS IS FOR. F5TtsEngine::SetReferenceAudio needs the reference clip as a
// mel in the model's own geometry before it can upload it as the conditioning
// tensor. That geometry is NOT the one this repo already has: WhisperDSP is
// 16 kHz / 80-128 bands / log10 with a Whisper-specific dynamic-range
// compression, and it LOADS its filterbank from a .bin dumped by Python. None of
// that transfers. So this is a second, independent front end, and it is
// deliberately standalone.
//
// PARITY TARGET, stated exactly, because every one of these is load-bearing.
// This reproduces f5_tts.model.modules.get_vocos_mel_spectrogram, which is
//     torchaudio.transforms.MelSpectrogram(
//         sample_rate=24000, n_fft=1024, win_length=1024, hop_length=256,
//         n_mels=100, power=1, center=True, normalized=False, norm=None)
//     followed by  mel.clamp(min=1e-5).log()
// and therefore:
//
//   * power=1 -> MAGNITUDE sqrt(re^2+im^2), NOT the power spectrum. This is the
//     single easiest thing to get wrong here, because the Whisper front end two
//     directories away uses power=2 and its inner loop otherwise looks
//     identical. Using power=2 does not fail: it produces a mel that is exactly
//     twice this one after the log, i.e. a plausible spectrogram that clones the
//     wrong voice.
//   * norm=None -> triangles are NOT area-normalised (no Slaney 2/(f[m+2]-f[m])
//     scaling). librosa's default DOES normalise, so a filterbank lifted from
//     librosa is wrong here even at identical geometry.
//   * mel_scale="htk" (torchaudio's default, which F5 does not override) ->
//     m = 2595*log10(1 + f/700). NOT the Slaney/librosa piecewise-linear-below-
//     1 kHz scale.
//   * center=True -> reflect-pad by n_fft/2 before framing, so frame t is
//     CENTRED on sample t*hop. Frame count is 1 + n_samples/hop.
//   * natural log with a 1e-5 floor -- not log10, and not the 1e-10 floor
//     Whisper uses.
//
// LAYOUT: time-major [frames, n_mels] -- frame 0's 100 bands, then frame 1's.
// torchaudio hands back mel-major [n_mels, frames] and F5's CFM permutes it;
// emitting it already permuted is what lets F5TtsEngine upload the conditioning
// with one memcpy and slice the generated tail as a pointer offset. See the
// MelExtractorFn contract in f5_tts_engine.hpp.
//
// DEPENDENCY-FREE BY CONSTRUCTION. No ONNXRuntime, no CUDA, no engine, and no
// third-party FFT: n_fft is a power of two, so the radix-2 transform is ~40
// lines and keeps blackwell_tts the zero-link-dependency leaf its CMakeLists
// says it is. (pocketfft is in-tree via blackwell::sandbox_headers and would
// have worked; it was not worth turning a math leaf into a target with a
// dependency for a transform this size.)
//
// THREADING: ONE thread calls Compute(). It writes to preallocated scratch
// members -- that is what keeps the frame loop allocation-free -- so Compute()
// is non-const and two threads must not share an instance. Construct one per
// thread; the ctor is cheap (a 1024-point twiddle table and a ~1 KB filterbank).
//
// ERROR TIER: this is a leaf with no live audio deadline behind it, so the
// awkward-input case throws (std::invalid_argument) rather than returning a
// sentinel. The one caller that matters, F5TtsEngine::SetReferenceAudio, already
// wraps the extractor in try/catch and maps a throw to TtsStatus::RuntimeFailure
// -- it does not trust an injected callable to be noexcept in fact.
// -----------------------------------------------------------------------------
#include <cstddef>
#include <vector>

namespace blackwell::tts {

// Geometry. Defaults ARE the F5-TTS / vocos operating point; they are exposed
// only so a test can shrink them, not because they are tuning knobs. Changing
// any of them without re-exporting the model produces confident nonsense.
struct F5MelConfig {
    int    sample_rate = 24000;
    int    n_fft       = 1024;
    int    hop_length  = 256;
    int    win_length  = 1024;   // == n_fft here; a shorter window is centred in the frame
    int    n_mels      = 100;
    double f_min       = 0.0;
    double f_max       = 12000.0;   // Nyquist at 24 kHz
    // torchaudio's clamp floor, applied BEFORE the natural log.
    double log_floor   = 1e-5;

    int n_freqs() const { return n_fft / 2 + 1; }   // 513
};

class F5MelExtractor {
public:
    // Precomputes the Hann window, the mel filterbank and the FFT tables.
    // Throws std::invalid_argument if the config is not usable (n_fft not a
    // power of two, non-positive geometry, f_max above Nyquist, win_length
    // greater than n_fft).
    explicit F5MelExtractor(const F5MelConfig& cfg = F5MelConfig{});

    // Frames produced for `n_samples` of input, matching torch.stft(center=True):
    //     1 + n_samples / hop_length      (integer division)
    // Note torchaudio does NOT drop the final frame -- transformers' Whisper
    // extractor does, and WhisperDSP::predict_num_frames therefore subtracts one.
    // Do not copy that here.
    std::size_t FrameCount(std::size_t n_samples) const;

    // The whole front end. `pcm` is mono float32 at cfg.sample_rate, nominally
    // in [-1, 1]. Returns FrameCount(pcm.size()) * n_mels values, TIME-MAJOR.
    //
    // Throws std::invalid_argument when the clip is too short to reflect-pad --
    // torch's reflect mode needs at least n_fft/2 + 1 samples, and silently
    // zero-padding instead would fabricate a spectrum for audio that is not
    // there. At 24 kHz that floor is 513 samples (~21 ms), far below any usable
    // reference clip.
    std::vector<float> Compute(const std::vector<float>& pcm);

    // Same, without allocating the result: `out` is resized and overwritten.
    // The form to call when cloning voices repeatedly.
    void ComputeInto(const float* pcm, std::size_t n_samples, std::vector<float>& out);

    const F5MelConfig& config() const { return cfg_; }

private:
    // In-place radix-2 decimation-in-time FFT over the scratch buffers, forward
    // sign convention (exp(-2*pi*i*k*n/N)), matching numpy/torch.
    void Fft();

    // One frame: FFT the already-windowed scratch -> magnitude -> filterbank ->
    // log. Writes n_mels floats to `out`.
    void MelFrame(float* out);

    F5MelConfig cfg_;

    std::vector<double> hann_;      // [win_length], periodic
    std::vector<int>    bitrev_;    // [n_fft]
    std::vector<double> tw_re_;     // [n_fft/2]
    std::vector<double> tw_im_;

    // Mel filterbank in COMPRESSED form: filter m covers frequency bins
    // [band_lo_[m], band_hi_[m]) with weights at weights_[band_off_[m] + i].
    // Triangles touch only a handful of bins each (~1000 nonzeros total against
    // 51300 dense), so this is both smaller and ~50x less inner-loop work than
    // the dense freq-major layout WhisperDSP uses.
    std::vector<int>   band_lo_;
    std::vector<int>   band_hi_;
    std::vector<int>   band_off_;
    std::vector<float> weights_;

    // Scratch, sized once in the ctor. The reason Compute()'s frame loop makes
    // no allocation at all.
    std::vector<double> re_;
    std::vector<double> im_;
    std::vector<double> mag_;       // [n_freqs]
};

}  // namespace blackwell::tts
