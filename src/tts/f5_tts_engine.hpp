#pragma once
// -----------------------------------------------------------------------------
// F5TtsEngine — flow-matching speech synthesis (F5-TTS: a DiT solved with an
// explicit ODE integrator, then a neural vocoder) over the ONNXRuntime **CUDA**
// execution provider.
//
// =============================================================================
// WHY THIS ONE IS ON THE GPU, when docs/TTS_INTEGRATION_AUDIT.md §1.3 argues at
// length for the CPU provider.
// =============================================================================
// That argument was made about a VITS-family model: ONE forward pass per
// sentence, ~15-30 M parameters, 10-25x realtime on two cores. Every word of it
// is still correct FOR THAT MODEL. F5-TTS is a different animal — a ~330 M
// parameter transformer evaluated `nfe_step` times per utterance (16-32), and
// twice per step if classifier-free guidance is not folded into the graph. That
// is 1-2 orders of magnitude more arithmetic than the audit priced, and it puts
// CPU synthesis at roughly 0.3-1x realtime: slower than speech, which means the
// sink starves permanently and the feature does not exist.
//
// So the audit's conclusion is OVERRIDDEN for this backend, and the costs it
// listed are accepted rather than avoided. State them plainly, because they are
// real and someone will have to measure them:
//   * SM CONTENTION IS NOW LIVE. At batch = 1 the LLM decode is latency-bound,
//     so a concurrent DiT solve steals SM time from exactly the number the
//     product measures. Synthesis of an already-emitted utterance now competes
//     with the generation of the next one. This is a genuine regression against
//     the CPU design and it is the thing to profile first.
//   * VRAM. Bounded and small — see the arithmetic on max_frames below (~5 MB of
//     latents) — but the WEIGHTS are ~1.3 GB in fp16, on top of the ~5.3 GB AWQ
//     backbone and the ~1.05 GB KV pool. Set gpu_mem_limit_mb deliberately; the
//     arena is pinned to kSameAsRequested precisely so ORT cannot quietly grab a
//     multi-gigabyte reservation the engine then fails to allocate around.
//   * A SECOND CUDA RUNTIME IN-PROCESS, and a 434 MB ORT-GPU archive instead of
//     the 75 MB CPU one. cmake/OnnxRuntime.cmake fetches CPU-only today; this
//     class does not build until that is addressed.
//
// THIS CLASS IS THEREFORE NOT A CUDA-FREE LEAF, unlike everything else currently
// in src/tts/. src/tts/CMakeLists.txt says the target is "ENGINE-FREE AND
// CUDA-FREE, and must stay that way" — that claim survives only if this TU lands
// in a SEPARATE target (blackwell_tts_f5) that the frontend does not link. Do
// not fold it into blackwell_tts.
//
// It is still ENGINE-free, which is the part the doctrine actually cares about:
// it never touches BlackwellEngine, VRAMArena, the KV cache, or the bridge. See
// the threading contract below.
//
// =============================================================================
// THE FOUR DESIGN DECISIONS THAT MATTER
// =============================================================================
//
// (1) THE ODE LOOP NEVER TOUCHES HOST MEMORY. The latent is [1, T, 100] fp32 —
//     about 1.1 MB at 30 s. Round-tripping it per step would be ~70 MB of PCIe
//     traffic and, far worse, ~32 forced synchronisations against a stream that
//     would otherwise stay busy. So the latent lives in device memory allocated
//     through ORT's own CUDA allocator, and Ort::IoBinding pins the graph's
//     input and output onto those device pointers. Two buffers, ping-ponged:
//     phase p reads latent[p] and writes latent[1-p]. Bindings for both phases
//     are built ONCE per utterance, so a step is "Run the phase, flip the phase"
//     with no tensor construction, no rebinding, and no allocation.
//
//     This mirrors, deliberately, the recurrent-state ping-pong in
//     src/vad/silero_vad.cpp — same reason (an ORT graph may not read and write
//     the same buffer), same shape, so it reads as native to anyone who knows
//     that file.
//
// (2) THE EULER UPDATE AND THE CFG COMBINE BELONG IN THE GRAPH, NOT HERE. A DiT
//     exported the obvious way returns the velocity field v_t, leaving C++ to
//     compute x + dt*v and, under guidance, v_cond + w*(v_cond - v_uncond).
//     Both are elementwise ops over a device tensor, so doing them here means
//     either a D2H round trip (kills decision 1) or hand-written CUDA kernels
//     and a cudart dependency in the hot path.
//
//     Neither is necessary. Both ops are a handful of lines in the export
//     wrapper, they fuse into the graph, and they cost nothing measurable on the
//     GPU. THE EXPORT CONTRACT IS THEREFORE: the DiT graph takes (x_t, t, dt,
//     cond_mel, text, ...) and returns x_{t+1} DIRECTLY, with guidance already
//     applied. F5DitContract below is the machine-readable statement of that,
//     and validate_signature() enforces it at load time rather than letting a
//     mismatched export produce plausible-sounding garbage.
//
//     If you are handed an export that emits raw velocity, do not work around it
//     in C++ — re-export it. That is a ten-line change on the Python side and it
//     is the difference between this design and a much worse one.
//
// (3) THE INTERRUPT CHECK ONLY WORKS BECAUSE EACH STEP SYNCHRONISES. This is the
//     subtle one, and getting it wrong produces a flag that looks correct and
//     does nothing. Run() against a CUDA-EP session ENQUEUES work and returns; it
//     does not wait. A loop that checks the flag and calls Run() would race
//     through all 32 iterations in microseconds, enqueue the entire solve, and
//     only then start waiting — by which point cancelling is meaningless because
//     everything is already submitted.
//
//     So solve_ode() calls SynchronizeOutputs() at the bottom of every
//     iteration. The cost is one stream sync (tens of microseconds) against a
//     step that runs for milliseconds: noise. What it buys is that the flag is
//     checked against WORK COMPLETED rather than work submitted, which makes
//     worst-case cancel latency exactly one DiT step — the number quoted below,
//     and the number the duplex design depends on.
//
//     The same reasoning is why the flag is checked again after the loop and
//     around the vocoder: an interrupt arriving during the (single, longer)
//     vocoder Run cannot be honoured by a loop check at all. See the note on
//     RunOptions::SetTerminate in the .cpp for the escalation path if that Run
//     ever grows past the barge-in budget.
//
// (4) CONDITIONING IS UPLOADED ONCE, NOT PER UTTERANCE AND CERTAINLY NOT PER
//     STEP. SetReferenceAudio() writes the reference mel into a device buffer
//     sized for max_frames and zero-filled beyond the reference — which is
//     exactly the padded conditioning tensor every step of every subsequent
//     utterance wants. The ODE loop reads it in place and never rewrites it.
//     Re-cloning a voice is the only thing that costs an upload.
//
// =============================================================================
// THREADING — the contract, not a suggestion
// =============================================================================
// EXACTLY ONE thread calls SetReferenceAudio() / GenerateAudio() / Reset(): the
// TTS worker (docs/TTS_INTEGRATION_AUDIT.md §2.2). Those methods share device
// buffers, ORT bindings and ORT's CUDA stream, none of which are internally
// synchronised, and two concurrent solves would interleave writes into the same
// latent with no diagnostic beyond audible garbage.
//
// The observer surface — state(), synthesis_errors(), interruptions(),
// last_solve_ms(), last_vocode_ms() — is atomic and callable from any thread.
// That is the UI-readout seam and, exactly as in SileroVAD, it is the ONLY
// cross-thread surface.
//
// The interrupt flag is the one input that crosses threads, and it does so by
// design: the barge-in detector (the DSP worker, via the speech segmenter) sets
// it; this class only ever reads it, with relaxed ordering, because a
// one-iteration delay in observing a cancel is indistinguishable from the
// cancel having arrived one iteration later.
//
// THIS CLASS IS NOT A PARTY TO THE SINGLE-ENGINE-THREAD DOCTRINE. It touches
// CUDA, but the doctrine governs BlackwellEngine state, not the CUDA driver —
// which is thread-safe per context. If a future feature needs synthesis to
// influence generation it marshals through the existing PostEngineTask seam; it
// does not reach across.
//
// =============================================================================
// ERROR DOCTRINE (CLAUDE.md extension pattern #4), split by phase as usual
// =============================================================================
//   INIT    — the ctor throws std::runtime_error: missing model, a graph whose
//             signature is not the one bound here, no CUDA device, ORT failure.
//             Callers construct inside try/catch and disable TTS, which is the
//             posture translator/main.cpp already takes for the neural VAD and
//             the audio head. A missing 1.3 GB model is a greyed-out button.
//   RUNTIME — SetReferenceAudio() / GenerateAudio() / Reset() are noexcept and
//             return TtsStatus. An ORT fault mid-solve is caught, counted
//             (synthesis_errors()), and reported as RuntimeFailure. Nothing on
//             this path may unwind: by Phase 4 of the audit an aborted call also
//             desynchronises the echo canceller from its reference stream.
// -----------------------------------------------------------------------------
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "spsc_ring.hpp"   // blackwell::audio_rt::SpscRing<float>
#include "tts_status.hpp"

namespace blackwell::tts {

// -----------------------------------------------------------------------------
// Model geometry. These are properties of the F5-TTS checkpoint family, not
// tuning knobs — every one of them must match the ONNX export and the vocoder,
// and a mismatch is silent (wrong-speed or noise-shaped audio), never a crash.
//
// NOTE FOR ANYONE REACHING FOR WhisperDSP: this is NOT that geometry and the two
// must not be confused. Whisper is 16 kHz / 80 mels / log10-with-clamp; F5 is
// 24 kHz / 100 mels / natural log, hop 256, n_fft 1024. Feeding a Whisper mel to
// this model produces confident nonsense. See MelExtractorFn below.
// -----------------------------------------------------------------------------
inline constexpr int    kF5SampleRate  = 24000;
inline constexpr int    kF5MelChannels = 100;
inline constexpr int    kF5HopLength   = 256;
inline constexpr int    kF5NFft        = 1024;

// 24000 / 256. Exact in binary, so the frame<->sample conversions below are not
// approximations; keep it that way if the hop is ever repinned.
inline constexpr double kF5FramesPerSecond =
    static_cast<double>(kF5SampleRate) / static_cast<double>(kF5HopLength);

// Padding id for the text-condition tensor, applied over the tail of the frame
// axis that carries no token. MUST match the export's convention — F5 exports
// commonly pad with -1 and offset real ids by +1 inside the graph, but this is
// an export decision, so it is a constant here and validated by ear, not
// inferred. If speech starts correctly and then degrades toward the end of long
// utterances, suspect this first.
inline constexpr std::int32_t kF5TextPadId = -1;

// -----------------------------------------------------------------------------
// F5DitContract — the tensor names the two graphs are bound by.
//
// Placeholders, per the brief: fill them in once the export is finalised. They
// are DATA rather than string literals in the .cpp so that fixing an export
// mismatch is a config change and not a recompile of the only TU that can see
// ONNXRuntime. validate_signature() checks every one of these against the loaded
// graph at INIT time — a mis-bound tensor does not fail, it produces
// plausible-looking garbage, which is far more expensive to diagnose than a
// refused load.
// -----------------------------------------------------------------------------
struct F5DitContract {
    // ---- DiT inputs ----------------------------------------------------------
    std::string latent_in   = "x_t";        // [1, T, 100] f32, DEVICE
    std::string cond_mel    = "cond";       // [1, T, 100] f32, DEVICE (cached)
    std::string text_ids    = "text";       // [1, T]      i32, DEVICE (per utterance)
    std::string time_step   = "t";          // scalar/[1]  f32, HOST (see below)
    std::string delta_t     = "dt";         // scalar/[1]  f32, HOST
    // ---- DiT output ----------------------------------------------------------
    // x_{t+1}, NOT the velocity field. See design decision (2).
    std::string latent_out  = "x_next";     // [1, T, 100] f32, DEVICE

    // ---- Vocoder ------------------------------------------------------------
    // Time-major [1, T_gen, 100] in, to match the DiT's output layout exactly so
    // the handoff is a pointer offset rather than a transpose. If your vocoder
    // export insists on channels-first [1, 100, T_gen], put the permute INSIDE
    // that graph — the same argument as decision (2), and it fuses away.
    std::string vocoder_in  = "mel";        // [1, T_gen, 100] f32, DEVICE
    std::string vocoder_out = "waveform";   // [1, T_gen * 256] f32, HOST
};

// -----------------------------------------------------------------------------
// F5TtsConfig — INIT-tier setup. Everything here is fixed for the lifetime of
// the engine; per-utterance variation goes through GenerateAudio's arguments.
// -----------------------------------------------------------------------------
struct F5TtsConfig {
    std::string   dit_model_path;      // the transformer, ~1.3 GB fp16
    std::string   vocoder_model_path;  // Vocos / BigVGAN, ~50-500 MB
    F5DitContract contract{};

    int device_id = 0;

    // Number of function evaluations — the DiT is run exactly this many times.
    // THE latency/quality dial, and the only one worth exposing to a user.
    // Below ~8 the flow is under-integrated and artefacts appear; above ~32 the
    // returns are inaudible. 16 is a reasonable default for a duplex assistant.
    int nfe_step = 16;

    // Classifier-free guidance weight, applied IN THE GRAPH as
    //     v = v_cond + cfg * (v_cond - v_uncond)
    // (F5's formulation; note it is not the Stable-Diffusion spelling, though it
    // is equivalent under a shift of scale). Baked into the export, so it is
    // recorded here for the log and the panel, NOT sent per step.
    //
    // The cost hides here: unless the export batches cond and uncond together
    // into one batch-2 pass, guidance DOUBLES the work per step. Batch it.
    float cfg_strength = 2.0f;

    // Sway sampling coefficient. F5 warps the otherwise-uniform time schedule by
    //     t <- t + c * (cos(pi/2 * t) - 1 + t)
    // With c < 0 this front-loads evaluations toward t = 0, where the flow is
    // least linear and Euler error is largest. -1.0 is the published value and
    // it is worth a surprising amount of quality at low nfe_step; 0 disables the
    // warp and gives a uniform schedule.
    float sway_coef = -1.0f;

    // Speech rate. Scales the ESTIMATED duration (and therefore the tensor
    // length), so it changes cost as well as tempo: >1 is faster and cheaper.
    float speed = 1.0f;

    // Hard cap on the frame axis, and the thing every device buffer is sized
    // from. A guard, not a tuning knob — DiT attention is quadratic in T, so a
    // runaway duration estimate would turn one utterance into a multi-second
    // stall on the TTS thread while holding the GPU.
    //
    // The VRAM arithmetic, so it can be checked rather than trusted:
    //   latent  1 x 3000 x 100 x 4 B = 1.2 MB, x2 for the ping-pong  = 2.4 MB
    //   cond    1 x 3000 x 100 x 4 B                                 = 1.2 MB
    //   text    1 x 3000 x 4 B                                       = 0.01 MB
    //   ~3.6 MB of latents. Rounding to 5 MB with slack, this is noise next to
    //   the weights and next to the backbone. Sizing generously is free; the
    //   real ceiling is the quadratic attention cost, which is why 3000 frames
    //   (= 32 s) rather than something larger.
    std::size_t max_frames = 3000;

    // Ceiling handed to the CUDA EP arena, in MiB. Bounds what ORT may reserve
    // so a synthesis burst cannot starve the backbone's allocations. 0 = no
    // limit, which on a box also hosting 5.3 GB of AWQ weights is a bad idea.
    std::size_t gpu_mem_limit_mb = 2048;

    // Seed for the initial noise x_0. FIXED BY DEFAULT, deliberately: flow
    // matching is stochastic in its initial condition, so an unpinned seed makes
    // every regression test either flaky or forced to assert on energy envelopes
    // instead of samples (docs/TTS_INTEGRATION_AUDIT.md §5.4.4 flags exactly
    // this). Set random_seed_per_utterance to get variation in production while
    // keeping tests reproducible.
    std::uint64_t noise_seed = 1234567891234567891ull;
    bool random_seed_per_utterance = false;

    // ---- diagnostics --------------------------------------------------------
    // Non-empty turns on ORT's per-node profiler, which writes
    // <prefix>_<timestamp>.json containing every kernel's duration AND the
    // execution provider it ran on. That file is the only way to answer "which
    // nodes fell back to CPU and did it cost anything" with numbers rather than
    // inference from a partition-time warning. Costs a few percent; leave empty
    // in production.
    std::string profile_prefix;

    // 0 = VERBOSE, 1 = INFO, 2 = WARNING (ORT's default here), 3 = ERROR.
    // VERBOSE prints the full node->EP assignment table at session build.
    int log_severity = 2;

    // ORT graph optimization level: 0 = DISABLE_ALL, 1 = BASIC, 2 = EXTENDED,
    // 3 = ALL (the default, and what production wants).
    //
    // A DIAGNOSTIC SEAM, not a tuning knob. Level 3 is where ORT fuses
    // LayerNorm/SkipLayerNorm/Attention, and a fusion that is correct in fp32
    // can be wrong in fp16 -- which presents as NaN in the output mel and
    // therefore SILENCE, with no error from Run(). When an fp16 DiT produces
    // silent or noisy audio, bisecting this value is what separates "the export
    // is wrong" from "a fused fp16 kernel is wrong": if level 1 is clean and
    // level 3 is not, the graph is fine and the fusion is not.
    int graph_opt_level = 3;
};

// -----------------------------------------------------------------------------
// Published for the UI badge (docs/TTS_INTEGRATION_AUDIT.md §4.5), as one
// atomic, the same treatment TranscriptView gives SpeechPipelineState.
// -----------------------------------------------------------------------------
enum class F5State : std::int32_t {
    Uninitialised = 0,
    Idle          = 1,
    Conditioning  = 2,  // SetReferenceAudio: mel extraction + the one upload
    Solving       = 3,  // the ODE loop; this is where an interrupt is honoured
    Vocoding      = 4,  // latent -> PCM; a single Run, not interruptible mid-call
};

inline const char* to_string(F5State s) noexcept {
    switch (s) {
        case F5State::Uninitialised: return "Uninitialised";
        case F5State::Idle:          return "Idle";
        case F5State::Conditioning:  return "Conditioning";
        case F5State::Solving:       return "Solving";
        case F5State::Vocoding:      return "Vocoding";
    }
    return "UnknownF5State";
}

// -----------------------------------------------------------------------------
// MelExtractorFn — the reference-audio front end, injected rather than built in.
//
// WHY A SEAM. F5's mel is 24 kHz / 100 bands / natural-log / hop 256, and the
// only mel front end in this repo is WhisperDSP's 16 kHz / 80 band / log10
// geometry, which is golden-parity-pinned against PyTorch dumps and must not be
// generalised to serve a second consumer. So this is either (a) a small
// dedicated STFT, or (b) — usually better — a third tiny ONNX graph, since most
// F5 exports already ship a preprocessing model that takes raw audio. Injecting
// it keeps that choice out of this class entirely.
//
// Contract: `pcm` is mono f32 at kF5SampleRate. On success fill `out_mel` with
// frames*kF5MelChannels values in TIME-MAJOR order (frame 0's 100 bands, then
// frame 1's — matching the tensor layout the graphs are bound with), set
// `out_frames`, and return true. Must be noexcept-in-practice: it is called
// from a RUNTIME-tier method.
// -----------------------------------------------------------------------------
using MelExtractorFn = std::function<bool(const float* pcm, std::size_t n_samples,
                                          std::vector<float>& out_mel,
                                          std::size_t& out_frames)>;

// =============================================================================
// F5TtsEngine
// =============================================================================
class F5TtsEngine {
public:
    // INIT tier. Loads both graphs onto the CUDA EP, validates their signatures
    // against config.contract, and allocates every device buffer the steady
    // state will ever need. Throws std::runtime_error on a missing file, a
    // signature mismatch, an absent CUDA device, or any ORT failure.
    //
    // Expensive — hundreds of milliseconds to seconds, dominated by the weight
    // upload. Construct once, off the UI thread, at startup.
    F5TtsEngine(F5TtsConfig config, MelExtractorFn mel_extractor);
    ~F5TtsEngine();

    // Movable; the sessions and device allocations travel with the PIMPL.
    // Standard moved-from semantics: only destruction and assignment are valid
    // on the source. NOT copyable — there is device memory and an ORT stream in
    // here, and neither has a sane copy.
    F5TtsEngine(F5TtsEngine&&) noexcept;
    F5TtsEngine& operator=(F5TtsEngine&&) noexcept;
    F5TtsEngine(const F5TtsEngine&) = delete;
    F5TtsEngine& operator=(const F5TtsEngine&) = delete;

    // -------------------------------------------------------------------------
    // Conditioning cache (design decision 4)
    // -------------------------------------------------------------------------
    // RUNTIME tier. Clones a voice: extracts the reference mel from `pcm_24k`,
    // uploads it ONCE into the device conditioning buffer, and stores the
    // reference transcript ids. Every subsequent GenerateAudio() reads that
    // buffer in place — this is the whole point, and the reason a solve costs
    // nothing per step for conditioning.
    //
    // `pcm_24k` is mono f32 at kF5SampleRate; resample before calling. Keep it
    // to 5-15 s: the reference occupies the FRONT of the same frame axis the
    // generated speech has to fit into, so a long reference directly shrinks
    // max_frames' usable remainder AND raises the quadratic attention cost of
    // every step.
    //
    // `ref_text_ids` is the transcript OF THAT AUDIO, tokenised by the same
    // ITokenizer that will tokenise the text to speak. It is not optional: the
    // duration estimate is a ratio between token count and frame count measured
    // on this pair, so a wrong or empty transcript yields wrong-length — and
    // therefore rushed or padded — speech.
    //
    // Invalidates any previously cached reference. Cheap enough to call on a
    // voice change; not cheap enough to call per utterance.
    TtsStatus SetReferenceAudio(const std::vector<float>& pcm_24k,
                                const std::vector<std::int64_t>& ref_text_ids) noexcept;

    // As above, but takes a mel that has already been computed elsewhere —
    // time-major, `frames * kF5MelChannels` values. Bypasses MelExtractorFn
    // entirely, which is what the tests use (no STFT, no ONNX preprocessor, a
    // deterministic mel) and what a pipeline that already has the mel should use.
    TtsStatus SetReferenceMel(const float* mel, std::size_t frames,
                              const std::vector<std::int64_t>& ref_text_ids) noexcept;

    bool HasReference() const noexcept;

    // -------------------------------------------------------------------------
    // Synthesis
    // -------------------------------------------------------------------------
    // RUNTIME tier. Solves the flow ODE for `text_ids` against the cached
    // reference, vocodes the result, and appends mono f32 PCM at kF5SampleRate
    // to `out_pcm` (which is CLEARED first, and is the caller's reusable buffer
    // so the steady state allocates only when an utterance grows past the
    // high-water mark).
    //
    // `interrupt` may be null (uninterruptible). When non-null it is polled once
    // per completed DiT step and again around the vocoder; if it is ever
    // observed true the solve is abandoned, the device stream is drained so no
    // work outlives the call, `out_pcm` is left EMPTY, and Interrupted is
    // returned. Worst-case cancel latency is ONE DiT step — see design decision
    // (3) for why that is a real number and not an aspiration.
    //
    // Interrupted is NOT a failure: it does not touch synthesis_errors() and the
    // caller must not retry (tts_status.hpp spells this out).
    //
    // Returns NotInitialized if no reference has been set, EmptyResult for empty
    // `text_ids`, RuntimeFailure if either graph faulted.
    // `total_frames_hint`, when non-zero, REPLACES the built-in duration
    // estimate with an exact frame count (reference prefix included).
    //
    // It exists because F5's own duration formula is not the one EstimateFrames
    // implements: infer_batch_process measures text in UTF-8 BYTES, not tokens,
    // and takes the reference length as n_samples/hop rather than the mel frame
    // count. Those are close for Cyrillic and wrong in general, and only the
    // caller knows the byte lengths. A caller reproducing F5 exactly computes
    //     ref_len  = ref_samples / hop
    //     total    = ref_len + ref_len * gen_bytes / ref_bytes / speed
    // and passes it here; anything else can pass 0 and get the estimate.
    TtsStatus GenerateAudio(const std::vector<std::int64_t>& text_ids,
                           const std::atomic<bool>* interrupt,
                           std::vector<float>& out_pcm,
                           std::size_t total_frames_hint = 0) noexcept;

    // As above, but pushes straight into the playback ring instead of
    // materialising a vector — the form the TtsWorker should actually use, since
    // it is the one that gets audio moving without a second copy.
    //
    // Uses SpscRing::write() (retry), NOT write_or_drop(): this producer runs far
    // faster than realtime and can afford to wait for space, so a full ring means
    // "the sink is still playing", not "audio was lost". Counting a drop here
    // would make the overrun metric meaningless — spsc_ring.hpp's fault-policy
    // block is explicit about this, and it is the exact mistake its concurrency
    // test was written to catch.
    //
    // THE CALLER MUST BE THE RING'S SOLE PRODUCER. The ring is SPSC; handing the
    // same ring to two synthesis calls on two threads corrupts it silently.
    //
    // While retrying, `interrupt` is still polled — a barge-in during playback
    // must stop the push, not wait for a sink nobody is listening to any more.
    TtsStatus GenerateAudio(const std::vector<std::int64_t>& text_ids,
                           const std::atomic<bool>* interrupt,
                           audio_rt::SpscRing<float>& sink,
                           std::size_t total_frames_hint = 0) noexcept;

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------
    // Drops the cached reference and drains the device stream, returning the
    // engine to Idle with no work in flight. Call on a voice change, a session
    // boundary, or after a fault. Does NOT reload the graphs — that is the ctor.
    TtsStatus Reset() noexcept;

    // Frames of reference currently cached, and the frame budget left for
    // generated speech (max_frames minus that). The panel wants both; a
    // remainder near zero is why an utterance came out truncated.
    std::size_t reference_frames() const noexcept;
    std::size_t available_frames() const noexcept;

    // Frames the duration heuristic would choose for `text_ids`, clamped to the
    // budget. Exposed for the same reason the estimate exists at all: it decides
    // tensor length and therefore cost, so a caller batching work — or a test
    // asserting the heuristic — needs to see it without running a solve.
    std::size_t EstimateFrames(std::size_t text_id_count) const noexcept;

    // ---- cross-thread observers (any thread) --------------------------------
    F5State state() const noexcept { return state_.load(std::memory_order_relaxed); }

    // Solves that FAULTED (ORT threw, a graph misbehaved). Monotone. Interrupts
    // are not failures and are counted separately.
    std::uint64_t synthesis_errors() const noexcept {
        return synthesis_errors_.load(std::memory_order_relaxed);
    }
    std::uint64_t interruptions() const noexcept {
        return interruptions_.load(std::memory_order_relaxed);
    }

    // Wall-clock of the last completed solve / vocode, milliseconds. The two
    // numbers the latency budget is actually made of — a solve climbing toward
    // the barge-in budget means nfe_step is too high for this GPU under load.
    double last_solve_ms() const noexcept {
        return last_solve_ms_.load(std::memory_order_relaxed);
    }
    double last_vocode_ms() const noexcept {
        return last_vocode_ms_.load(std::memory_order_relaxed);
    }

private:
    // The shared body of both GenerateAudio overloads: everything up to and
    // including the vocoder, leaving the PCM in the Impl's host output buffer.
    // The overloads differ only in where they then put it, and that difference
    // is not worth duplicating a solve for.
    TtsStatus synthesize_locked(const std::vector<std::int64_t>& text_ids,
                                const std::atomic<bool>* interrupt,
                                std::size_t total_frames_hint,
                                const float*& out_samples,
                                std::size_t& out_count) noexcept;

    struct Impl;                    // sessions, bindings, device allocations
    std::unique_ptr<Impl> impl_;    // PIMPL: keeps ONNXRuntime out of this header

    std::atomic<F5State>      state_{F5State::Uninitialised};
    std::atomic<std::uint64_t> synthesis_errors_{0};
    std::atomic<std::uint64_t> interruptions_{0};
    std::atomic<double>       last_solve_ms_{0.0};
    std::atomic<double>       last_vocode_ms_{0.0};
};

}  // namespace blackwell::tts
