// -----------------------------------------------------------------------------
// F5TtsEngine implementation — the second (and last) translation unit in the
// project that sees <onnxruntime_cxx_api.h>. Everything above it talks to the
// PIMPL'd header, so no consumer inherits ORT's include path or its exception
// types. src/vad/silero_vad.cpp is the other one, and this file follows its
// idioms deliberately: pre-built bindings, ping-ponged buffers, a signature
// validated at load, and a RUNTIME tier that counts faults instead of throwing.
//
// It is also the only TU here that includes <cuda_runtime.h>, and only for
// cudaMemcpy. See the block above upload_to_device() for why that dependency is
// the right trade and why it does NOT belong in the ODE loop.
//
// GRAPHS BOUND HERE (names are data — see F5DitContract):
//   DiT      x_t [1,T,100] f32 D | cond [1,T,100] f32 D | text [1,T] i32 D
//            t [1] f32 H | dt [1] f32 H            ->  x_next [1,T,100] f32 D
//   Vocoder  mel [1,T_gen,100] f32 D               ->  waveform [1,T_gen*256] f32 H
//   (D = device-resident and bound through IoBinding; H = host, ORT does the copy)
//
// The DiT returns x_{t+1}, not the velocity field, and applies CFG internally.
// That is a hard export contract, not a preference — f5_tts_engine.hpp design
// decision (2) has the argument.
// -----------------------------------------------------------------------------
#include "f5_tts_engine.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <map>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include <cuda_runtime.h>
#include <onnxruntime_cxx_api.h>

namespace blackwell::tts {
namespace {

constexpr double kPi = 3.14159265358979323846;

// UTF-8 std::string -> the wide path ORT wants on Windows. Goes through char8_t
// so a non-ASCII checkout path survives (the plain std::filesystem::path(
// std::string) ctor would reinterpret it in the active code page, and
// std::filesystem::u8path is deprecated in C++20). Lifted verbatim from
// silero_vad.cpp; if a third ORT consumer appears, hoist it rather than copy it
// a third time.
std::filesystem::path to_path(const std::string& utf8) {
    return std::filesystem::path(
        std::u8string(reinterpret_cast<const char8_t*>(utf8.c_str()), utf8.size()));
}

// RAII for OrtCUDAProviderOptionsV2, which the C API hands out as a raw pointer
// with a matching Release. The options are built by a sequence of calls that can
// each throw, so a bare pointer here leaks on the INIT-tier failure path — the
// one path where we are already unwinding and least want a second problem.
class CudaProviderOptions {
public:
    CudaProviderOptions() { Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&opts_)); }
    ~CudaProviderOptions() {
        if (opts_ != nullptr) Ort::GetApi().ReleaseCUDAProviderOptions(opts_);
    }
    CudaProviderOptions(const CudaProviderOptions&) = delete;
    CudaProviderOptions& operator=(const CudaProviderOptions&) = delete;

    OrtCUDAProviderOptionsV2* get() const noexcept { return opts_; }

private:
    OrtCUDAProviderOptionsV2* opts_ = nullptr;
};

// EPSS -- Empirically Pruned Step Sampling. THE time grid F5 actually uses.
//
// This is not a refinement, it is the schedule. f5_tts/model/cfm.py:
//     if t_start == 0 and use_epss:            # use_epss DEFAULTS TO TRUE
//         t = get_epss_timesteps(steps, ...)
//     else:
//         t = torch.linspace(t_start, 1, steps + 1, ...)
//     if sway_sampling_coef is not None:
//         t = t + sway_sampling_coef * (cos(pi/2 * t) - 1 + t)
//
// so for the step counts in the table below the grid is a hand-tuned, heavily
// front-loaded sequence over 32nds -- NOT a uniform linspace -- and sway is
// applied ON TOP of it. Integrating the same ODE on a uniform grid instead is
// well-formed and produces a completely different (much worse) solution: the
// latent overshoots, the mel leaves the range the vocoder was trained on, and
// the audio comes out as a metallic drone with no speech structure. Measured:
// the resulting x_final ranged [-13.1, 16.2] against the correct [-7.9, 4.8].
//
// Entries exist only for these step counts; anything else falls back to the
// uniform grid, exactly as get_epss_timesteps does. Values are 32nds.
const std::vector<int>* epss_table(int nfe_step) {
    static const std::map<int, std::vector<int>> kTable{
        {5,  {0, 2, 4, 8, 16, 32}},
        {6,  {0, 2, 4, 6, 8, 16, 32}},
        {7,  {0, 2, 4, 6, 8, 16, 24, 32}},
        {10, {0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32}},
        {12, {0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32}},
        {16, {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32}},
    };
    const auto it = kTable.find(nfe_step);
    return it == kTable.end() ? nullptr : &it->second;
}

// The time schedule: nfe_step + 1 knots from t=0 (pure noise) to t=1 (data),
// taken from the EPSS table when one exists and uniform otherwise, then warped
// by the sway coefficient
//     t <- t + c * (cos(pi/2 * t) - 1 + t)
// The warp fixes both endpoints for any c, and for c in [-1, 0] it is monotone,
// concentrating evaluations near t=0 where the flow is least linear and Euler's
// local error is largest.
//
// Computed ONCE at construction: it depends only on config, it is 33 floats, and
// recomputing trig inside the ODE loop would be the only host arithmetic in an
// otherwise device-resident step.
std::vector<float> build_time_schedule(int nfe_step, float sway_coef) {
    const std::vector<int>* epss = epss_table(nfe_step);
    std::vector<float> knots(static_cast<std::size_t>(nfe_step) + 1);
    for (int i = 0; i <= nfe_step; ++i) {
        // EPSS values are 32nds; the uniform fallback is i/nfe_step.
        const double u = epss != nullptr
                             ? static_cast<double>((*epss)[static_cast<std::size_t>(i)]) / 32.0
                             : static_cast<double>(i) / static_cast<double>(nfe_step);
        const double warped = u + static_cast<double>(sway_coef) *
                                      (std::cos(kPi * 0.5 * u) - 1.0 + u);
        knots[static_cast<std::size_t>(i)] = static_cast<float>(warped);
    }
    // A non-monotone schedule yields a negative dt, which integrates the flow
    // BACKWARDS for that step. The audio comes out muddy rather than broken, so
    // this is exactly the class of bug that survives listening tests — refuse it
    // at INIT instead. (Reachable only with |sway_coef| > 1, i.e. a bad config.)
    for (std::size_t i = 1; i < knots.size(); ++i) {
        if (!(knots[i] > knots[i - 1])) {
            throw std::runtime_error(
                "F5TtsEngine: sway_coef=" + std::to_string(sway_coef) +
                " produces a non-monotone time schedule (knot " + std::to_string(i) +
                " does not increase). Use a coefficient in [-1, 0]; -1.0 is the "
                "published value and 0 disables the warp.");
        }
    }
    return knots;
}

}  // namespace

// =============================================================================
// Impl — both sessions, every device allocation, and the pre-built bindings.
//
// DECLARATION ORDER IS DEPENDENCY ORDER (CLAUDE.md extension pattern #3), and
// here it is load-bearing rather than stylistic, because ORT's C++ types are
// non-owning views over each other:
//   env      -> outlives the sessions that were created from it
//   options  -> consumed by the session ctors
//   sessions -> the ALLOCATOR is constructed from a session
//   gpu_alloc-> the MemoryAllocations come from it and must not outlive it
//   *_dev    -> the Ort::Values are non-owning views onto these pointers
//   values   -> the IoBindings hold references to these
//   bindings -> last in, first destroyed
// Reordering any pair above turns teardown into a use-after-free that only
// reproduces on the destruction path. Add members deliberately.
// =============================================================================
struct F5TtsEngine::Impl {
    F5TtsConfig    config;
    MelExtractorFn mel_extractor;
    F5DitContract  names;   // copy of config.contract, so the c_str()s are stable

    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "blackwell_tts_f5"};
    Ort::SessionOptions dit_options;
    Ort::SessionOptions vocoder_options;

    std::unique_ptr<Ort::Session> dit;
    std::unique_ptr<Ort::Session> vocoder;

    // Device memory info for the CUDA EP, and the CPU one used for the scalars
    // and the final waveform. "Cuda" is the allocator name ORT registers the EP
    // under; it must match exactly or GetAllocation returns CPU memory and the
    // whole design silently degrades to what it was built to avoid.
    Ort::MemoryInfo cuda_mem{nullptr};
    Ort::MemoryInfo cpu_mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Device allocator, taken from the DiT session. The vocoder shares it: both
    // sessions target the same device, and one arena is one reservation instead
    // of two.
    std::unique_ptr<Ort::Allocator> gpu_alloc;

    // ---- device buffers, allocated ONCE at max_frames --------------------
    // Sized for the worst case so that a per-utterance shape change is a
    // narrower Ort::Value over the same pointer, never a reallocation. See the
    // VRAM arithmetic in F5TtsConfig::max_frames.
    std::optional<Ort::MemoryAllocation> latent_dev[2];  // ping-pong, [1,T,100] f32
    std::optional<Ort::MemoryAllocation> cond_dev;       // cached conditioning
    std::optional<Ort::MemoryAllocation> text_dev;       // [1,T] i32
    // [t, dt] as ONE 2-float device buffer, so a step refreshes both with a
    // single 8-byte H2D. See the note on scalars in the ODE loop for why these
    // are NOT bound as CPU tensors.
    std::optional<Ort::MemoryAllocation> scalars_dev;

    // ---- host staging ----------------------------------------------------
    // Sized once too; these are the only host-side per-utterance buffers, and
    // each is written at most ONCE per utterance (never per step).
    std::vector<float>        host_noise;    // x_0 ~ N(0, I)
    std::vector<std::int32_t> host_text;     // ref ids ++ gen ids, pad-filled
    std::vector<float>        host_wave;     // the vocoder's output lands here

    // ---- per-utterance tensor views + bindings ---------------------------
    // Rebuilt once per GenerateAudio (frames changes), never inside the loop.
    // Two complete bindings, one per ping-pong phase: phase p reads latent[p]
    // and writes latent[1-p], so a step is Run(binding[p]) and a flip. Same
    // structure, and the same reason, as SileroVAD's recurrent-state phases.
    std::optional<Ort::IoBinding> dit_binding[2];
    std::optional<Ort::IoBinding> vocoder_binding;

    // The Ort::Values are non-owning views and must outlive the bindings that
    // reference them, so they are members rather than locals in the builder.
    std::vector<Ort::Value> keepalive;

    // Bound as HOST scalars and mutated between steps. ORT inserts the 4-byte
    // H2D copy; at 4 bytes that is free, and it is what keeps the loop from
    // needing a cudaMemcpy of its own. These are the ONLY things that cross the
    // bus during a solve.
    float t_now = 0.0f;
    float dt_now = 0.0f;

    std::vector<float> schedule;   // nfe_step + 1 knots, precomputed
    std::mt19937_64 rng;

    // ---- cached conditioning --------------------------------------------
    std::size_t ref_frames = 0;
    std::vector<std::int64_t> ref_text_ids;
    bool has_reference = false;

    // Geometry of the utterance the bindings are currently shaped for; 0 means
    // "no valid bindings".
    std::size_t bound_frames = 0;
    std::size_t phase = 0;

    explicit Impl(F5TtsConfig cfg, MelExtractorFn mel)
        : config(std::move(cfg)),
          mel_extractor(std::move(mel)),
          names(config.contract),
          schedule(build_time_schedule(config.nfe_step, config.sway_coef)),
          rng(config.noise_seed) {
        validate_config();

        configure_session(dit_options);
        configure_session(vocoder_options);

        dit     = open_session(config.dit_model_path, dit_options, "DiT");
        vocoder = open_session(config.vocoder_model_path, vocoder_options, "vocoder");

        validate_signature();

        cuda_mem = Ort::MemoryInfo("Cuda", OrtDeviceAllocator,
                                   config.device_id, OrtMemTypeDefault);
        gpu_alloc = std::make_unique<Ort::Allocator>(*dit, cuda_mem);

        allocate_buffers();
    }

    void validate_config() const {
        if (config.nfe_step < 1) {
            throw std::runtime_error("F5TtsEngine: nfe_step must be >= 1 (got " +
                                     std::to_string(config.nfe_step) + ").");
        }
        if (config.max_frames == 0) {
            throw std::runtime_error("F5TtsEngine: max_frames must be > 0.");
        }
        if (!(config.speed > 0.0f)) {
            throw std::runtime_error("F5TtsEngine: speed must be > 0.");
        }
    }

    // Session options, pinned explicitly rather than taken from ORT's defaults.
    // The CPU-side knobs matter even on a GPU session: the EP falls back to CPU
    // for any node it cannot place, and ORT's default intra-op pool is sized to
    // the machine's CORE COUNT. Left alone, one synthesis burst spawns a thread
    // per core and preempts the DSP worker — the one thread in this process with
    // a genuine 10 ms deadline. Two threads is plenty for the residue.
    void configure_session(Ort::SessionOptions& options) const {
        options.SetIntraOpNumThreads(2);
        options.SetInterOpNumThreads(1);
        options.SetExecutionMode(ORT_SEQUENTIAL);

        // Defaults to ORT_ENABLE_ALL; see F5TtsConfig::graph_opt_level for why
        // the lower rungs exist and what bisecting them tells you.
        switch (config.graph_opt_level) {
            case 0:  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);     break;
            case 1:  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);    break;
            case 2:  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED); break;
            default: options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);      break;
        }

        CudaProviderOptions cuda;

        const std::string device_id = std::to_string(config.device_id);
        const std::string mem_limit =
            std::to_string(config.gpu_mem_limit_mb * 1024ull * 1024ull);

        std::vector<const char*> keys;
        std::vector<const char*> values;
        keys.push_back("device_id");                 values.push_back(device_id.c_str());

        // kNextPowerOfTwo by default — ORT's own default, restored from a
        // previous kSameAsRequested pin. NOT the fix for the BFCArena failure
        // that prompted the change: a measured 2x2 (see
        // F5TtsConfig::arena_extend_strategy) shows both strategies failing
        // under a cap and both succeeding without one. gpu_mem_limit_mb is the
        // knob that mattered.
        keys.push_back("arena_extend_strategy");
        values.push_back(config.arena_extend_strategy == 1 ? "kSameAsRequested"
                                                           : "kNextPowerOfTwo");

        // HEURISTIC, not EXHAUSTIVE. Exhaustive benchmarks every cuDNN algorithm
        // on the first call with a new shape — seconds of stall and a VRAM spike,
        // paid again per shape. Our frame count varies per utterance, so
        // EXHAUSTIVE would re-trigger constantly. This is the same reasoning the
        // Whisper encoder's bucketed CUDA graphs apply from the other direction.
        keys.push_back("cudnn_conv_algo_search");    values.push_back("HEURISTIC");

        // Keeps EP copies on the compute stream, so IoBinding's
        // SynchronizeOutputs() is a sufficient barrier for everything a step did
        // — which is precisely the guarantee the interrupt check leans on.
        keys.push_back("do_copy_in_default_stream"); values.push_back("1");

        // Omitted entirely at 0, which is the default — an absent gpu_mem_limit
        // lets the arena grow against physical VRAM, which is the only budget
        // that means anything once the attention intermediates are in play.
        // Passing a cap here is opt-in and deliberately rare; see the field.
        if (config.gpu_mem_limit_mb > 0) {
            keys.push_back("gpu_mem_limit");         values.push_back(mem_limit.c_str());
        }

        Ort::ThrowOnError(Ort::GetApi().UpdateCUDAProviderOptions(
            cuda.get(), keys.data(), values.data(), keys.size()));
        Ort::ThrowOnError(Ort::GetApi().SessionOptionsAppendExecutionProvider_CUDA_V2(
            static_cast<OrtSessionOptions*>(options), cuda.get()));
    }

    std::unique_ptr<Ort::Session> open_session(const std::string& model_path,
                                               Ort::SessionOptions& options,
                                               const char* role) const {
        if (model_path.empty()) {
            throw std::runtime_error(std::string("F5TtsEngine: no ") + role + " model path set.");
        }
        const std::filesystem::path path = to_path(model_path);
        if (!std::filesystem::exists(path)) {
            throw std::runtime_error(std::string("F5TtsEngine: ") + role +
                                     " model not found: " + model_path);
        }
        return std::make_unique<Ort::Session>(env, path.c_str(), options);
    }

    // Fails loudly on any graph that is not the one bound below. A mis-bound
    // tensor does not throw — it produces confident garbage (wrong-speed speech,
    // noise-shaped output, a voice that is not the reference), and diagnosing
    // that from audio costs far more than refusing to load.
    //
    // Names only, deliberately: shapes are dynamic on the frame axis by design,
    // so asserting them here would reject valid exports. The frame axis is
    // checked where it is actually known — rebuild_bindings().
    void validate_signature() const {
        require_io(*dit, "DiT", /*inputs=*/{names.latent_in, names.cond_mel, names.text_ids,
                                            names.time_step, names.delta_t},
                   /*outputs=*/{names.latent_out});
        require_io(*vocoder, "vocoder", {names.vocoder_in}, {names.vocoder_out});
    }

    static void require_io(const Ort::Session& session, const char* role,
                           const std::vector<std::string>& want_in,
                           const std::vector<std::string>& want_out) {
        Ort::AllocatorWithDefaultOptions alloc;
        auto collect = [&](bool inputs) {
            std::vector<std::string> got;
            const std::size_t n = inputs ? session.GetInputCount() : session.GetOutputCount();
            got.reserve(n);
            for (std::size_t i = 0; i < n; ++i) {
                const Ort::AllocatedStringPtr name =
                    inputs ? session.GetInputNameAllocated(i, alloc)
                           : session.GetOutputNameAllocated(i, alloc);
                got.emplace_back(name.get());
            }
            return got;
        };
        auto check = [&](const std::vector<std::string>& want,
                         const std::vector<std::string>& got, const char* kind) {
            for (const std::string& w : want) {
                if (std::find(got.begin(), got.end(), w) == got.end()) {
                    std::string listing;
                    for (const std::string& g : got) listing += (listing.empty() ? "" : ", ") + g;
                    throw std::runtime_error(
                        std::string("F5TtsEngine: ") + role + " graph has no " + kind + " named '" +
                        w + "'. Present: [" + listing + "]. Update F5DitContract to match the "
                        "export, or re-export to match the contract -- and remember the DiT must "
                        "return x_{t+1} with CFG already applied, not the raw velocity field.");
                }
            }
        };
        check(want_in,  collect(true),  "input");
        check(want_out, collect(false), "output");
    }

    // Every device and host buffer the steady state will ever need, taken once
    // at max_frames. After this returns, nothing on the synthesis path
    // allocates: a shorter utterance is a narrower Ort::Value over the same
    // pointer, which is the entire reason the ODE loop is allocation-free.
    void allocate_buffers() {
        const std::size_t frames = config.max_frames;
        const std::size_t mel_elems = frames * static_cast<std::size_t>(kF5MelChannels);

        latent_dev[0] = gpu_alloc->GetAllocation(mel_elems * sizeof(float));
        latent_dev[1] = gpu_alloc->GetAllocation(mel_elems * sizeof(float));
        cond_dev      = gpu_alloc->GetAllocation(mel_elems * sizeof(float));
        text_dev      = gpu_alloc->GetAllocation(frames * sizeof(std::int32_t));
        scalars_dev   = gpu_alloc->GetAllocation(2 * sizeof(float));

        host_noise.assign(mel_elems, 0.0f);
        host_text.assign(frames, kF5TextPadId);
        host_wave.assign(frames * static_cast<std::size_t>(kF5HopLength), 0.0f);

        // The conditioning buffer is read in full on EVERY step of EVERY
        // utterance, including the region past the reference, so it must start
        // defined rather than holding whatever the allocator handed back.
        zero_device(cond_dev->get(), mel_elems * sizeof(float));
    }

    // Samples the vocoder emits for `gen_frames` of mel. NOT gen_frames * hop.
    //
    // charactr/vocos-mel-24khz is exported with padding="center": the forward
    // STFT padded its input by n_fft/2 on each side, so the inverse trims that
    // much back off and the graph returns (T - 1) * hop samples. (A padding=
    // "same" vocoder would return T * hop -- the class-signature default, which
    // the released config overrides. Verified against the exported graph:
    // scripts/export_f5_tts_onnx.py writes the mode and this formula into
    // f5_tts_contract.json.)
    //
    // Getting this wrong is not a crash. Over-reporting by one hop leaves ~10.7
    // ms of stale buffer on the end of every utterance, which is an audible
    // click -- and clicks are the entire perceived quality of a TTS feature.
    static std::size_t vocoder_samples(std::size_t gen_frames) noexcept {
        return gen_frames == 0 ? 0
                               : (gen_frames - 1) * static_cast<std::size_t>(kF5HopLength);
    }

    // NOT const: Ort::MemoryAllocation::get() is non-const, and rightly so --
    // these hand out writable device pointers, so a const accessor would have
    // been lying about what the caller may do with the result.
    float* latent_ptr(std::size_t p) noexcept {
        return static_cast<float*>(latent_dev[p]->get());
    }
    float* cond_ptr() noexcept { return static_cast<float*>(cond_dev->get()); }
    std::int32_t* text_ptr() noexcept {
        return static_cast<std::int32_t*>(text_dev->get());
    }
    float* scalars_ptr() noexcept { return static_cast<float*>(scalars_dev->get()); }

    // ---- the cudart seam -------------------------------------------------
    // WHY cudaMemcpy AND NOT PURE ORT. ORT's public API can allocate device
    // memory (Ort::Allocator) and can bind it (IoBinding), but it exposes no way
    // to WRITE a device tensor out of band. The alternative is to bind the
    // conditioning and the text as HOST inputs and let ORT copy them in on every
    // Run — which is ~1.2 MB x nfe_step of PCIe traffic per utterance to move
    // data that has not changed, and it discards f5_tts_engine.hpp's design
    // decision (4) outright.
    //
    // So: three cudaMemcpy calls, all of them OUTSIDE the ODE loop (conditioning
    // once per voice, noise and text once per utterance). The loop itself makes
    // no CUDA calls at all.
    //
    // STREAM SAFETY. These are synchronous copies on the legacy default stream,
    // while ORT runs on its own (non-blocking) stream, so the two are NOT
    // implicitly ordered. Every caller therefore drains first — see drain(). The
    // copies only ever run between utterances, on the single owning thread, with
    // no Run in flight, which makes that drain sufficient and cheap.
    static void upload_to_device(void* dst, const void* src, std::size_t bytes) {
        const cudaError_t rc = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
        if (rc != cudaSuccess) {
            throw std::runtime_error(std::string("F5TtsEngine: H2D copy failed: ") +
                                     cudaGetErrorString(rc));
        }
    }

    static void zero_device(void* dst, std::size_t bytes) {
        const cudaError_t rc = cudaMemset(dst, 0, bytes);
        if (rc != cudaSuccess) {
            throw std::runtime_error(std::string("F5TtsEngine: device memset failed: ") +
                                     cudaGetErrorString(rc));
        }
    }

    // Blocks until nothing this engine submitted is still in flight. Called
    // before any out-of-band copy, on teardown, and on the interrupt path — an
    // abandoned solve must not leave kernels writing into buffers the next call
    // is about to overwrite, and must not leave them running past the
    // destructor.
    void drain() noexcept {
        for (auto& b : dit_binding) {
            if (b.has_value()) {
                try { b->SynchronizeOutputs(); } catch (const std::exception&) {}
            }
        }
        if (vocoder_binding.has_value()) {
            try { vocoder_binding->SynchronizeOutputs(); } catch (const std::exception&) {}
        }
        (void)cudaDeviceSynchronize();
    }

    // Duration heuristic. F5 has no duration predictor: the generated length is
    // extrapolated from the reference, on the assumption that the same voice
    // speaking the same language holds a roughly constant frames-per-token rate.
    //
    //   gen_frames = ref_frames * (gen_tokens / ref_tokens) / speed
    //
    // Crude, and it IS the quality ceiling on rhythm — too short and the speech
    // is rushed and clipped, too long and it trails into breaths or babble. The
    // ratio must be computed in the units the reference was measured in, which
    // is why ref_text_ids comes from the SAME tokenizer as the text to speak.
    std::size_t estimate_total_frames(std::size_t gen_tokens) const noexcept {
        if (ref_frames == 0 || ref_text_ids.empty() || gen_tokens == 0) return 0;

        const double ratio = static_cast<double>(gen_tokens) /
                             static_cast<double>(ref_text_ids.size());
        const double gen = static_cast<double>(ref_frames) * ratio /
                           static_cast<double>(config.speed);

        // At least one frame of speech, so a one-token utterance still produces
        // audio rather than an empty tensor the graphs would reject.
        const std::size_t gen_frames =
            std::max<std::size_t>(1, static_cast<std::size_t>(std::llround(gen)));
        return std::min(config.max_frames, ref_frames + gen_frames);
    }

    // Builds every per-utterance Ort::Value and both DiT bindings. Runs ONCE per
    // GenerateAudio; after it returns the ODE loop touches nothing but Run() and
    // two host floats.
    void rebuild_bindings(std::size_t frames) {
        if (frames < 2 || frames > config.max_frames) {
            throw std::runtime_error("F5TtsEngine: frame count " + std::to_string(frames) +
                                     " out of range [2, " + std::to_string(config.max_frames) +
                                     "].");
        }

        // Bindings BEFORE the values they were built from. ORT copies the
        // OrtValue into the binding, so the reverse order happens to work today
        // — but it encodes the dependency backwards, and this is the one file
        // where that habit gets expensive.
        dit_binding[0].reset();
        dit_binding[1].reset();
        vocoder_binding.reset();
        keepalive.clear();
        // LOAD-BEARING, not an optimisation: bind_vocoder() pushes two more
        // entries after the DiT bindings already hold views into this vector, so
        // a reallocation would move Ort::Values out from under them. Six here
        // plus two there fits in twelve with room to spare; grow this if the
        // contract grows.
        keepalive.reserve(12);

        const auto f = static_cast<std::int64_t>(frames);
        const std::array<std::int64_t, 3> mel_shape{1, f, kF5MelChannels};
        const std::array<std::int64_t, 2> text_shape{1, f};
        const std::array<std::int64_t, 1> scalar_shape{1};
        const std::size_t mel_elems = frames * static_cast<std::size_t>(kF5MelChannels);

        // Narrower views over the SAME max_frames allocations — this is the
        // whole trick that makes a variable-length model work with fixed,
        // pre-allocated device memory and zero per-utterance allocation.
        auto dev_mel = [&](float* p) {
            return Ort::Value::CreateTensor<float>(cuda_mem, p, mel_elems,
                                                   mel_shape.data(), mel_shape.size());
        };

        keepalive.push_back(dev_mel(latent_ptr(0)));                         // 0
        keepalive.push_back(dev_mel(latent_ptr(1)));                         // 1
        keepalive.push_back(dev_mel(cond_ptr()));                            // 2
        keepalive.push_back(Ort::Value::CreateTensor<std::int32_t>(          // 3
            cuda_mem, text_ptr(), frames, text_shape.data(), text_shape.size()));
        // t and dt live in DEVICE memory, not host.
        //
        // THIS IS NOT AN OPTIMISATION -- binding them as CPU tensors is silently
        // WRONG. Ort::IoBinding resolves a CPU-memory input for a CUDA session
        // by copying it to the device when it is BOUND, not on every Run. Since
        // the bindings are built once per utterance and the scalars are mutated
        // per step, the graph kept seeing the values they held at bind time
        // (0, 0) for the whole solve. x_next = x_t + v*0 = x_t, so the latent
        // never moved and the vocoder was handed the raw initial noise.
        //
        // Nothing about that fails: it produces a well-formed 9-second WAV of a
        // low-level drone whose statistics are indistinguishable from vocoded
        // Gaussian noise, and it is INDEPENDENT OF THE INPUT TEXT -- which is
        // what finally gave it away, since two different sentences yielded
        // byte-comparable audio under a fixed noise seed.
        keepalive.push_back(Ort::Value::CreateTensor<float>(                 // 4
            cuda_mem, scalars_ptr(), 1, scalar_shape.data(), scalar_shape.size()));
        keepalive.push_back(Ort::Value::CreateTensor<float>(                 // 5
            cuda_mem, scalars_ptr() + 1, 1, scalar_shape.data(), scalar_shape.size()));

        for (std::size_t p = 0; p < 2; ++p) {
            Ort::IoBinding b(*dit);
            b.BindInput(names.latent_in.c_str(),  keepalive[p]);
            b.BindInput(names.cond_mel.c_str(),   keepalive[2]);
            b.BindInput(names.text_ids.c_str(),   keepalive[3]);
            b.BindInput(names.time_step.c_str(),  keepalive[4]);
            b.BindInput(names.delta_t.c_str(),    keepalive[5]);
            // Phase p writes the OTHER latent. A graph may not alias its input
            // and output tensor, and binding both to one buffer is the classic
            // way to get output that is subtly wrong rather than an error.
            b.BindOutput(names.latent_out.c_str(), keepalive[1 - p]);
            dit_binding[p].emplace(std::move(b));
        }

        bound_frames = frames;
        phase = 0;
    }

    // The vocoder reads only the GENERATED tail of the solved latent — the
    // reference prefix was conditioning, not output. Because the latent is
    // row-major [1, T, 100] with time as the middle axis, that tail is
    // CONTIGUOUS at offset ref_frames * 100, so the trim is pointer arithmetic:
    // no copy, no kernel, no transpose. That contiguity is also why the header's
    // export contract insists the vocoder take time-major mel.
    void bind_vocoder(std::size_t result_phase, std::size_t frames) {
        const std::size_t gen_frames = frames - ref_frames;
        const std::size_t gen_elems  = gen_frames * static_cast<std::size_t>(kF5MelChannels);
        const std::size_t samples    = vocoder_samples(gen_frames);

        const std::array<std::int64_t, 3> mel_shape{
            1, static_cast<std::int64_t>(gen_frames), kF5MelChannels};
        const std::array<std::int64_t, 2> wave_shape{1, static_cast<std::int64_t>(samples)};

        float* gen_mel = latent_ptr(result_phase) +
                         ref_frames * static_cast<std::size_t>(kF5MelChannels);

        keepalive.push_back(Ort::Value::CreateTensor<float>(
            cuda_mem, gen_mel, gen_elems, mel_shape.data(), mel_shape.size()));
        // Output bound in HOST memory: ORT performs the D2H itself. This is the
        // one transfer the design wants, since the waveform is the result and
        // has to reach the CPU regardless. The buffer is pre-sized at max_frames
        // so even this costs no allocation.
        keepalive.push_back(Ort::Value::CreateTensor<float>(
            cpu_mem, host_wave.data(), samples, wave_shape.data(), wave_shape.size()));

        Ort::IoBinding b(*vocoder);
        b.BindInput(names.vocoder_in.c_str(),   keepalive[keepalive.size() - 2]);
        b.BindOutput(names.vocoder_out.c_str(), keepalive[keepalive.size() - 1]);
        vocoder_binding.emplace(std::move(b));
    }
};

// =============================================================================
// F5TtsEngine
// =============================================================================
F5TtsEngine::F5TtsEngine(F5TtsConfig config, MelExtractorFn mel_extractor)
    : impl_(std::make_unique<Impl>(std::move(config), std::move(mel_extractor))) {
    state_.store(F5State::Idle, std::memory_order_relaxed);
}

F5TtsEngine::~F5TtsEngine() {
    // Nothing may still be writing into buffers whose allocations are about to
    // be released. On the interrupt path in particular the caller may destroy
    // the engine promptly after abandoning a solve.
    if (impl_) impl_->drain();
}

// The atomics are not themselves movable, so the moves are hand-written: the
// sessions and device allocations travel with the PIMPL, the published scalars
// are transferred by value. Relaxed ordering is right — a move is not a
// concurrent operation (the source must already be quiescent), so these only
// need to be atomic, not ordered against anything. Same argument, same shape as
// SileroVAD's moves.
F5TtsEngine::F5TtsEngine(F5TtsEngine&& other) noexcept
    : impl_(std::move(other.impl_)),
      state_(other.state_.load(std::memory_order_relaxed)),
      synthesis_errors_(other.synthesis_errors_.load(std::memory_order_relaxed)),
      interruptions_(other.interruptions_.load(std::memory_order_relaxed)),
      last_solve_ms_(other.last_solve_ms_.load(std::memory_order_relaxed)),
      last_vocode_ms_(other.last_vocode_ms_.load(std::memory_order_relaxed)) {
    other.state_.store(F5State::Uninitialised, std::memory_order_relaxed);
}

F5TtsEngine& F5TtsEngine::operator=(F5TtsEngine&& other) noexcept {
    if (this != &other) {
        if (impl_) impl_->drain();
        impl_ = std::move(other.impl_);
        state_.store(other.state_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        synthesis_errors_.store(other.synthesis_errors_.load(std::memory_order_relaxed),
                                std::memory_order_relaxed);
        interruptions_.store(other.interruptions_.load(std::memory_order_relaxed),
                             std::memory_order_relaxed);
        last_solve_ms_.store(other.last_solve_ms_.load(std::memory_order_relaxed),
                             std::memory_order_relaxed);
        last_vocode_ms_.store(other.last_vocode_ms_.load(std::memory_order_relaxed),
                              std::memory_order_relaxed);
        other.state_.store(F5State::Uninitialised, std::memory_order_relaxed);
    }
    return *this;
}

// -----------------------------------------------------------------------------
// Conditioning
// -----------------------------------------------------------------------------
TtsStatus F5TtsEngine::SetReferenceAudio(const std::vector<float>& pcm_24k,
                                         const std::vector<std::int64_t>& ref_text_ids) noexcept {
    if (!impl_) return TtsStatus::NotInitialized;
    if (pcm_24k.empty() || ref_text_ids.empty()) return TtsStatus::InvalidArgument;
    if (!impl_->mel_extractor) {
        // No front end was injected, so there is no way to turn audio into the
        // 100-band/24 kHz mel this model wants. A caller in this position should
        // be using SetReferenceMel().
        return TtsStatus::NotInitialized;
    }

    state_.store(F5State::Conditioning, std::memory_order_relaxed);

    std::vector<float> mel;
    std::size_t frames = 0;
    try {
        if (!impl_->mel_extractor(pcm_24k.data(), pcm_24k.size(), mel, frames)) {
            state_.store(F5State::Idle, std::memory_order_relaxed);
            synthesis_errors_.fetch_add(1, std::memory_order_relaxed);
            return TtsStatus::RuntimeFailure;
        }
    } catch (const std::exception&) {
        // The extractor is caller-supplied and therefore not trusted to be
        // noexcept in fact, whatever its contract says.
        state_.store(F5State::Idle, std::memory_order_relaxed);
        synthesis_errors_.fetch_add(1, std::memory_order_relaxed);
        return TtsStatus::RuntimeFailure;
    }
    return SetReferenceMel(mel.data(), frames, ref_text_ids);
}

TtsStatus F5TtsEngine::SetReferenceMel(const float* mel, std::size_t frames,
                                       const std::vector<std::int64_t>& ref_text_ids) noexcept {
    if (!impl_) return TtsStatus::NotInitialized;
    if (mel == nullptr || frames == 0 || ref_text_ids.empty()) return TtsStatus::InvalidArgument;

    // The reference sits at the FRONT of the same frame axis the generated
    // speech must fit into, so it cannot consume the whole budget. Leaving at
    // least half is arbitrary but defensible: a reference longer than the speech
    // it conditions is both wasteful and quadratically expensive.
    if (frames > impl_->config.max_frames / 2) {
        return TtsStatus::InvalidArgument;
    }

    state_.store(F5State::Conditioning, std::memory_order_relaxed);
    try {
        impl_->drain();   // no Run may be reading cond_dev while we overwrite it

        const std::size_t mel_elems = static_cast<std::size_t>(kF5MelChannels) *
                                      impl_->config.max_frames;

        // Zero the WHOLE buffer, not just the tail past `frames`. The region
        // beyond the reference is the padding every step reads, and a shorter
        // reference replacing a longer one would otherwise leave the previous
        // voice's mel sitting in that padding — audible as a ghost of the old
        // speaker, and invisible in any test that only ever sets one reference.
        Impl::zero_device(impl_->cond_ptr(), mel_elems * sizeof(float));
        Impl::upload_to_device(impl_->cond_ptr(), mel,
                               frames * static_cast<std::size_t>(kF5MelChannels) * sizeof(float));

        impl_->ref_frames    = frames;
        impl_->ref_text_ids  = ref_text_ids;
        impl_->has_reference = true;
        impl_->bound_frames  = 0;   // geometry changed; bindings are stale

        state_.store(F5State::Idle, std::memory_order_relaxed);
        return TtsStatus::Success;
    } catch (const std::exception&) {
        impl_->has_reference = false;
        synthesis_errors_.fetch_add(1, std::memory_order_relaxed);
        state_.store(F5State::Idle, std::memory_order_relaxed);
        return TtsStatus::RuntimeFailure;
    }
}

bool F5TtsEngine::HasReference() const noexcept {
    return impl_ && impl_->has_reference;
}

std::size_t F5TtsEngine::reference_frames() const noexcept {
    return impl_ ? impl_->ref_frames : 0;
}

std::size_t F5TtsEngine::available_frames() const noexcept {
    if (!impl_) return 0;
    return impl_->config.max_frames - impl_->ref_frames;
}

std::size_t F5TtsEngine::EstimateFrames(std::size_t text_id_count) const noexcept {
    return impl_ ? impl_->estimate_total_frames(text_id_count) : 0;
}

// -----------------------------------------------------------------------------
// Synthesis
// -----------------------------------------------------------------------------
// The shared body. Leaves the PCM in impl_->host_wave and hands back a view; the
// two public overloads differ only in where they copy it, which is not worth
// duplicating a solve for.
TtsStatus F5TtsEngine::synthesize_locked(const std::vector<std::int64_t>& text_ids,
                                         const std::atomic<bool>* interrupt,
                                         std::size_t total_frames_hint,
                                         const float*& out_samples,
                                         std::size_t& out_count) noexcept {
    out_samples = nullptr;
    out_count = 0;

    if (!impl_) return TtsStatus::NotInitialized;
    if (!impl_->has_reference) return TtsStatus::NotInitialized;
    if (text_ids.empty()) return TtsStatus::EmptyResult;

    // Relaxed throughout: a one-iteration delay in observing a cancel is
    // indistinguishable from the cancel having arrived one iteration later, and
    // there is no other data being published alongside this flag that would need
    // ordering against it.
    const auto cancelled = [interrupt]() noexcept {
        return interrupt != nullptr && interrupt->load(std::memory_order_relaxed);
    };
    if (cancelled()) {
        interruptions_.fetch_add(1, std::memory_order_relaxed);
        return TtsStatus::Interrupted;
    }

    Impl& impl = *impl_;

    const std::size_t frames =
        total_frames_hint != 0 ? std::min(total_frames_hint, impl.config.max_frames)
                               : impl.estimate_total_frames(text_ids.size());
    if (frames <= impl.ref_frames + 1) {
        // The estimate left no room for generated speech: the reference has
        // eaten the frame budget. Not a fault, and retrying will not help.
        return TtsStatus::EmptyResult;
    }

    try {
        // ---- per-utterance setup (all of it OUTSIDE the loop) ---------------
        impl.drain();   // the out-of-band copies below need a quiet device

        if (impl.bound_frames != frames) {
            impl.rebuild_bindings(frames);
        }
        impl.phase = 0;

        // x_0 ~ N(0, I). Generated on the host and uploaded once: 1.2 MB of H2D
        // per utterance (~100 us), against which a device-side RNG would need a
        // curand dependency and a kernel launch to save nothing measurable.
        // Seeded from config so tests are reproducible by default — flow
        // matching is stochastic ONLY in this initial condition, so pinning it
        // pins the whole utterance.
        if (impl.config.random_seed_per_utterance) {
            impl.rng.seed(std::random_device{}());
        }
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        const std::size_t mel_elems = frames * static_cast<std::size_t>(kF5MelChannels);
        for (std::size_t i = 0; i < mel_elems; ++i) {
            impl.host_noise[i] = gauss(impl.rng);
        }
        Impl::upload_to_device(impl.latent_ptr(0), impl.host_noise.data(),
                               mel_elems * sizeof(float));

        // Text condition: [ reference transcript | text to speak | pad... ].
        // The reference ids must lead, because the model aligns them against the
        // reference mel occupying the matching frames of the conditioning
        // tensor — that pairing is what carries the voice.
        std::fill(impl.host_text.begin(), impl.host_text.begin() +
                      static_cast<std::ptrdiff_t>(frames), kF5TextPadId);
        std::size_t w = 0;
        for (const std::int64_t id : impl.ref_text_ids) {
            if (w >= frames) break;
            impl.host_text[w++] = static_cast<std::int32_t>(id);
        }
        for (const std::int64_t id : text_ids) {
            if (w >= frames) break;   // truncated; the duration estimate was optimistic
            impl.host_text[w++] = static_cast<std::int32_t>(id);
        }
        Impl::upload_to_device(impl.text_ptr(), impl.host_text.data(),
                               frames * sizeof(std::int32_t));

        // ---- the ODE solve --------------------------------------------------
        // Explicit Euler over the sway-warped schedule. Each iteration is
        // literally "publish t and dt, Run the phase, flip" — the update
        // x <- x + dt*v and the CFG combine both happen inside the graph, which
        // is why there is no arithmetic here and no device pointer arithmetic
        // either. See design decision (2).
        state_.store(F5State::Solving, std::memory_order_relaxed);
        const auto solve_begin = std::chrono::steady_clock::now();

        for (int step = 0; step < impl.config.nfe_step; ++step) {
            // Checked BEFORE submitting, so a cancel that arrived during the
            // previous step costs nothing further.
            if (cancelled()) {
                impl.drain();   // nothing may outlive this call
                interruptions_.fetch_add(1, std::memory_order_relaxed);
                state_.store(F5State::Idle, std::memory_order_relaxed);
                return TtsStatus::Interrupted;
            }

            const std::size_t k = static_cast<std::size_t>(step);
            impl.t_now  = impl.schedule[k];
            impl.dt_now = impl.schedule[k + 1] - impl.schedule[k];

            // Publish the step's scalars to the device. 8 bytes, and the ONLY
            // host->device traffic inside the loop. Safe against ORT's stream
            // without extra synchronisation because the previous iteration ended
            // with SynchronizeOutputs(), so no kernel is in flight here.
            const float host_scalars[2] = {impl.t_now, impl.dt_now};
            Impl::upload_to_device(impl.scalars_ptr(), host_scalars, sizeof(host_scalars));

            impl.dit->Run(Ort::RunOptions{nullptr}, *impl.dit_binding[impl.phase]);

            // THE LOAD-BEARING LINE. Run() only ENQUEUES; without this the loop
            // would submit all nfe_step steps in microseconds and the interrupt
            // check above would be decorative — see design decision (3). The
            // sync costs tens of microseconds against a step measured in
            // milliseconds, and it is what makes worst-case cancel latency one
            // step rather than one utterance.
            impl.dit_binding[impl.phase]->SynchronizeOutputs();

            impl.phase = 1 - impl.phase;   // the freshly written latent is now live
        }

        const auto solve_end = std::chrono::steady_clock::now();
        last_solve_ms_.store(
            std::chrono::duration<double, std::milli>(solve_end - solve_begin).count(),
            std::memory_order_relaxed);

        // ---- vocode ---------------------------------------------------------
        // Last chance to bail before committing to a Run that cannot be
        // interrupted from this thread. ORT does offer RunOptions::SetTerminate
        // for exactly this, but it must be called from ANOTHER thread while this
        // one is blocked inside Run, and it is sticky (UnsetTerminate is
        // required before the options object is reused). That is a real
        // escalation path if vocoding ever grows past the barge-in budget;
        // today it is a single sub-10 ms Run and the check here is enough.
        if (cancelled()) {
            impl.drain();
            interruptions_.fetch_add(1, std::memory_order_relaxed);
            state_.store(F5State::Idle, std::memory_order_relaxed);
            return TtsStatus::Interrupted;
        }

        state_.store(F5State::Vocoding, std::memory_order_relaxed);
        const auto vocode_begin = std::chrono::steady_clock::now();

        impl.bind_vocoder(impl.phase, frames);
        impl.vocoder->Run(Ort::RunOptions{nullptr}, *impl.vocoder_binding);
        // The waveform is bound in HOST memory, so this sync is what makes the
        // D2H visible to the read below. Skipping it yields a buffer that is
        // usually right and occasionally half-written.
        impl.vocoder_binding->SynchronizeOutputs();

        const auto vocode_end = std::chrono::steady_clock::now();
        last_vocode_ms_.store(
            std::chrono::duration<double, std::milli>(vocode_end - vocode_begin).count(),
            std::memory_order_relaxed);

        out_count   = Impl::vocoder_samples(frames - impl.ref_frames);
        out_samples = impl.host_wave.data();

        state_.store(F5State::Idle, std::memory_order_relaxed);
        return TtsStatus::Success;

    } catch (const std::exception&) {
        // RUNTIME tier: an ORT fault degrades this utterance and is counted; it
        // never unwinds into the TTS worker, which by Phase 4 of the audit is
        // also feeding an echo canceller that must not lose its reference
        // stream. Drain first — the fault may have left work queued.
        impl_->drain();
        impl_->bound_frames = 0;   // bindings are of unknown validity now
        synthesis_errors_.fetch_add(1, std::memory_order_relaxed);
        state_.store(F5State::Idle, std::memory_order_relaxed);
        return TtsStatus::RuntimeFailure;
    }
}

TtsStatus F5TtsEngine::GenerateAudio(const std::vector<std::int64_t>& text_ids,
                                     const std::atomic<bool>* interrupt,
                                     std::vector<float>& out_pcm,
                                     std::size_t total_frames_hint) noexcept {
    out_pcm.clear();

    const float* samples = nullptr;
    std::size_t count = 0;
    const TtsStatus status =
        synthesize_locked(text_ids, interrupt, total_frames_hint, samples, count);
    if (status != TtsStatus::Success) return status;   // out_pcm stays EMPTY

    try {
        out_pcm.assign(samples, samples + count);
    } catch (const std::bad_alloc&) {
        // The only allocation on this path. Reported rather than propagated:
        // noexcept is the contract, and a caller who ignores the status still
        // gets an empty buffer rather than a partial utterance.
        out_pcm.clear();
        synthesis_errors_.fetch_add(1, std::memory_order_relaxed);
        return TtsStatus::RuntimeFailure;
    }
    return TtsStatus::Success;
}

TtsStatus F5TtsEngine::GenerateAudio(const std::vector<std::int64_t>& text_ids,
                                     const std::atomic<bool>* interrupt,
                                     audio_rt::SpscRing<float>& sink,
                                     std::size_t total_frames_hint) noexcept {
    const float* samples = nullptr;
    std::size_t count = 0;
    const TtsStatus status =
        synthesize_locked(text_ids, interrupt, total_frames_hint, samples, count);
    if (status != TtsStatus::Success) return status;

    // write(), not write_or_drop(): this producer runs many times faster than
    // realtime, so a full ring means "the sink is still playing what we already
    // gave it", not "audio was lost". Counting an overrun on every retry would
    // make that metric report hundreds of thousands of drops for a stream that
    // lost nothing — spsc_ring.hpp's fault-policy block is explicit about this.
    std::size_t pushed = 0;
    while (pushed < count) {
        if (interrupt != nullptr && interrupt->load(std::memory_order_relaxed)) {
            // Barge-in mid-push. The already-written span is the consumer's
            // problem to discard (that is what the epoch tag at the layer above
            // is for); ours is to stop feeding a sink nobody is listening to.
            interruptions_.fetch_add(1, std::memory_order_relaxed);
            return TtsStatus::Interrupted;
        }
        const std::size_t n = sink.write(samples + pushed, count - pushed);
        if (n == 0) {
            // Ring full. Yield rather than spin: the consumer is a real-time
            // playback callback, and burning a core here is the one thing that
            // could make it miss the deadline we are waiting on.
            std::this_thread::yield();
            continue;
        }
        pushed += n;
    }
    return TtsStatus::Success;
}

TtsStatus F5TtsEngine::Reset() noexcept {
    if (!impl_) return TtsStatus::NotInitialized;
    impl_->drain();
    impl_->has_reference = false;
    impl_->ref_frames    = 0;
    impl_->ref_text_ids.clear();
    impl_->bound_frames  = 0;
    impl_->phase         = 0;
    state_.store(F5State::Idle, std::memory_order_relaxed);
    return TtsStatus::Success;
}

}  // namespace blackwell::tts
