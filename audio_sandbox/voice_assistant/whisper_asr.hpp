#pragma once
// -----------------------------------------------------------------------------
// whisper_asr.hpp — the whisper.cpp (GGML) speech-to-text engine behind ONE
// method: PCM in, UTF-8 out.
//
// PIMPL, AND NOT AS A STYLE CHOICE. This is the only translation unit in the
// repo that may see <whisper.h> and the ggml headers behind it -- exactly the
// contract src/vad/silero_vad.hpp holds for <onnxruntime_cxx_api.h>. Two things
// depend on it: the /W4 /WX budget (ggml does not compile clean and is
// quarantined as a SYSTEM include, which only helps if it is included in one
// place), and the ability to build this app with -DUSE_WHISPER_CPP=OFF without
// every consumer of this header growing an #ifdef.
//
// WHAT IT IS FOR. Cascade mode: a VAD-bounded utterance is transcribed here and
// the resulting TEXT is what reaches the language model. Contrast the legacy
// Ultravox path, where audio becomes soft-tokens spliced directly into the
// backbone's KV cache -- there, no text exists until the model writes one.
//
// THREADING. ONE OWNER THREAD, and this is a hard requirement, not a
// recommendation: a whisper_context is not thread-safe and holds the graph's
// scratch buffers. WhisperCascadeMode gives it a dedicated ASR worker and never
// touches it from anywhere else. Transcribe() blocks for the whole encode+decode
// (tens of ms on the GPU, hundreds on the CPU), which is precisely why it must
// not run on the engine thread or the DSP worker.
//
// CUDA. With the ggml CUDA backend this is the THIRD CUDA context in the process
// (engine + ONNXRuntime CUDA EP + ggml). ggml owns its own streams and there is
// no API to hand it ours, so the isolation between it and the decode loop is by
// context and by SCHEDULING -- the ASR worker runs one utterance at a time, off
// the decode loop's critical path. The device is selected in the constructor, on
// the calling thread, because CUDA's current device is per-thread: construct
// this object ON its owner thread.
//
// ERROR TIER: INIT. The constructor throws (std::runtime_error) on a missing or
// unloadable model. Transcribe() does NOT throw -- a failed encode is a lost
// utterance, not a dead app, and it reports by returning an empty string and
// bumping failures().
// -----------------------------------------------------------------------------
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace rt {

struct WhisperAsrConfig {
    // The GGML model file (e.g. Whisper-Turbo-Platinum-F16.bin). Required.
    std::string model_path;
    // ISO-639-1 code ("ru", "en"); empty = whisper's own language ID. Naming the
    // language is worth doing whenever it is known: on a short utterance the
    // detector confuses acoustically adjacent languages, and a wrong transcript
    // is not recoverable downstream -- it is what gets committed and answered.
    std::string language;
    // CPU worker threads for the ops ggml keeps on the host. Small, not a core
    // count: on a CUDA build almost nothing runs here, and on a CPU build the
    // graph stops scaling well past a handful of threads.
    int  n_threads = 4;
    // ggml CUDA backend. Ignored (forced false) on a build without it.
    bool use_gpu = true;
    int  gpu_device = 0;
};

class WhisperAsr {
public:
    // INIT tier: throws std::runtime_error naming the path if the model cannot
    // be loaded. Call ON the thread that will own this object (the device
    // selection is per-thread).
    explicit WhisperAsr(const WhisperAsrConfig& cfg);
    ~WhisperAsr();

    WhisperAsr(const WhisperAsr&) = delete;
    WhisperAsr& operator=(const WhisperAsr&) = delete;

    // Transcribe one complete utterance of 16 kHz mono float32 PCM. OWNER THREAD
    // ONLY. Returns the trimmed UTF-8 transcript, or an EMPTY string when there
    // was nothing to say -- which covers a genuinely silent window, whisper's own
    // non-speech artefacts ("[BLANK_AUDIO]", "(silence)"), and a failed encode.
    // The caller cannot distinguish those three and does not need to: all three
    // mean "publish nothing". failures() separates the last one for telemetry.
    //
    // WHAT MUST BE PASSED IN, because it decides whether punctuation is right:
    // a WHOLE utterance, with the pre-roll ahead of its first phoneme and the
    // tail-pad after its last. Whisper infers terminal punctuation -- the '?' of
    // a question in particular -- from the prosodic contour of the final few
    // hundred milliseconds. Hand it a window clipped at the last loud sample and
    // that contour is gone, along with the question mark. Segment geometry is
    // WhisperCascadeMode's job; this method just refuses windows that are too
    // short to be worth the GPU time.
    std::string Transcribe(const float* pcm16k, std::size_t count);

    // Is the ggml CUDA backend actually in use? False on a CPU-backend build, or
    // when the caller asked for the CPU.
    bool gpu() const noexcept;
    const std::string& model_path() const noexcept;

    // ---- telemetry (owner thread writes, any thread reads) ------------------
    // Counted separately because they answer different questions: `utterances`
    // is how many transcripts were produced, `failures` is how many encodes
    // returned non-zero. A rising failures() with a healthy utterances() is a
    // model/VRAM problem; both flat while the user is speaking is a segmenter
    // problem, one layer up.
    std::uint64_t utterances() const noexcept;
    std::uint64_t failures() const noexcept;
    // Wall time of the most recent Transcribe(), in ms. THE number to read when
    // asking whether cascade latency is the ASR or the backbone: it is the whole
    // of what the cascade adds ahead of the LLM's own prefill.
    double last_encode_ms() const noexcept;

private:
    struct Impl;                    // hides whisper_context / whisper_full_params
    std::unique_ptr<Impl> impl_;
};

}  // namespace rt
