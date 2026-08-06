#pragma once
// -----------------------------------------------------------------------------
// vram_budget.hpp — decide how much context this box can actually afford, BEFORE
// anything allocates it.
//
// THE FAILURE THIS EXISTS TO PREVENT. `max_context` is a persisted setting with a
// range of 512..131072 (settings_store.hpp) and no relationship whatsoever to the
// card it will run on. Nothing downstream re-checks it: bring_up_real_engine
// hands it to InferenceConfig::max_context_length, the arena sizes the KV pool
// from the resolved plan, and the allocation either succeeds or takes the process
// down inside a constructor -- after a 5.3 GB weight load, with a CUDA OOM whose
// message names a buffer nobody configured.
//
// AND ON WINDOWS IT IS WORSE THAN A CRASH. Under WDDM an over-committed
// allocation does NOT fail: the driver migrates pages to system RAM over PCIe.
// Throughput collapses by an order of magnitude, no error is reported anywhere,
// and the app simply appears to have frozen. report_vram() prints the numbers
// that would show it, but printing happens AFTER the commitment is made. This
// module is the half that runs BEFORE.
//
// THE MODEL, and it is deliberately the one the user's own accounting uses:
//
//     VRAM_total = VRAM_weights            backbone checkpoint + audio head
//                + VRAM_KV(max_context)    the one term we control
//                + VRAM_GGML               whisper.cpp, cascade mode only
//                + VRAM_DiT                F5-TTS diffusion transformer
//                + VRAM_ORT                the ONNXRuntime CUDA EP's arena
//                + kSafetyReserveBytes     OS / D3D / desktop compositor
//
// Everything but the KV term is FIXED for a given launch -- the checkpoints are
// what they are. So the budget resolves to one question: with the fixed terms
// paid and the reserve held back, how many KV tokens are left? That number is
// `granted_context`, and it is either the value the user asked for or the largest
// one that fits.
//
// MEASURED, NOT GUESSED, wherever measuring is possible. The weight terms are
// summed from the actual safetensors headers (a header parse, no tensor data
// read) and the GGML term from the model file's size on disk. Only the two ORT
// terms are constants, because they describe an allocator's behaviour rather than
// a file -- and each one carries the measurement it came from.
//
// ERROR TIER: INIT. plan_vram_budget() is noexcept and reports through the
// returned struct; a `!fits` budget is the caller's cue to abort startup with
// `failure` as the message, which is a clean exit through main's handler rather
// than a CUDA abort several seconds later.
// -----------------------------------------------------------------------------
#include <cstddef>
#include <string>

namespace rt {

// The safety buffer held back for the OS, the desktop compositor and D3D --
// including WebView2's own swap chain, which this app creates AFTER the engine.
// One gigabyte is the figure the task specifies; it is also roughly where a
// 12 GB card starts spilling in practice (see report_vram's 87.5% note).
inline constexpr std::size_t kSafetyReserveBytes = 1024ull * 1024ull * 1024ull;

// The floor the guard refuses to go below. Matches clamp_settings' own lower
// bound for max_context, so a clamped budget can never produce a value the
// settings validator would reject -- and a box that cannot afford even this has
// no business loading the checkpoint at all.
inline constexpr int kMinViableContext = 512;

// Granularity the granted context is rounded DOWN to. Nothing requires it; it
// exists so the clamped number reads as a decision rather than as the remainder
// of a division ("3584" is a context length, "3617" is an artefact).
inline constexpr int kContextGranularity = 256;

// What the app intends to load. Empty paths / false flags mean "this term is not
// part of this launch", and its contribution is exactly zero.
struct VramBudgetInputs {
    // The backbone checkpoint directory. Weights are summed from
    // model.safetensors.index.json (or model.safetensors).
    std::string model_dir;
    // The Ultravox audio head. Only audio_tower.* + multi_modal_projector.* are
    // uploaded, and they are widened BF16 -> fp32 on the way (2x); both facts are
    // accounted for. Ignored when load_audio_head is false (cascade mode).
    std::string audio_head_dir;
    bool        load_audio_head = false;
    // The GGML Whisper model whisper.cpp uploads in cascade mode.
    std::string whisper_model_path;
    // F5-TTS: the DiT graph plus the ONNXRuntime CUDA EP arena behind it.
    bool        tts_enabled = false;

    int requested_context = 2048;
    // paged_branch_factor: the KV pool is sized per branch, so isolated sessions
    // (chat + transcription) cost exactly twice a single linear context.
    int branch_factor = 1;
};

// The resolved plan. Every size is in BYTES so the arithmetic has one unit; the
// logging is where they become gigabytes.
struct VramBudget {
    // ---- the fixed terms -----------------------------------------------------
    std::size_t weights = 0;   // backbone + audio head
    std::size_t ggml = 0;      // whisper.cpp
    std::size_t dit = 0;       // F5 diffusion transformer
    std::size_t ort = 0;       // ORT CUDA EP arena / cuDNN workspace
    std::size_t reserve = kSafetyReserveBytes;

    // ---- the term the guard actually moves -----------------------------------
    std::size_t kv_bytes_per_token = 0;   // already multiplied by branch_factor
    std::size_t kv = 0;                   // kv_bytes_per_token * granted_context

    // ---- what the driver says ------------------------------------------------
    std::size_t device_free = 0;
    std::size_t device_total = 0;

    int  requested_context = 0;
    int  granted_context = 0;
    bool clamped = false;   // granted < requested: a WARN was (or should be) logged

    // fits == false is the FAIL-CLOSED verdict: not even kMinViableContext can be
    // afforded, and `failure` says which term ate the card. The caller must abort
    // startup rather than proceed with a smaller number.
    bool        fits = true;
    std::string failure;

    // Set when a term could not be MEASURED (an unreadable checkpoint header, a
    // driver that would not answer cudaMemGetInfo). The budget then degrades to
    // advisory: nothing is clamped, because clamping on a fabricated number can
    // break a configuration that was working. Says so in the log.
    bool        advisory_only = false;
    std::string advisory_reason;

    // Sum of every fixed term plus the reserve -- i.e. what is spoken for before
    // a single KV token is allocated.
    std::size_t fixed_total() const noexcept {
        return weights + ggml + dit + ort + reserve;
    }
};

// Compute the plan. MUST run after cudaSetDevice for the target device and
// BEFORE the engine (and therefore the KV pool) is constructed. Calling
// cudaMemGetInfo here also forces the CUDA context to exist, so `device_free`
// already has the context's own footprint subtracted from it -- which is the
// number we want, not the nameplate capacity.
//
// noexcept: a checkpoint that cannot be read is an advisory budget, not an
// exception. The real load a few lines later will produce a far better error
// message for a genuinely bad path than this function could.
VramBudget plan_vram_budget(const VramBudgetInputs& in) noexcept;

// Print the budget, one line per term, in the same [vram] voice report_vram uses.
// Emits the WARN when the context was clamped and the failure line when it was
// not affordable at all.
void log_vram_budget(const VramBudget& b) noexcept;

}  // namespace rt
