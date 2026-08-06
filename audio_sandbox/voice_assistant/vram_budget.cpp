// vram_budget.cpp — the measurement half of vram_budget.hpp.
//
// The only TU in voice_assistant that reads a checkpoint header without loading
// it. Everything here is cheap: a safetensors header parse, a config.json read,
// two stat() calls and one cudaMemGetInfo.
#include "vram_budget.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <nlohmann/json.hpp>

#include "safetensors.h"   // SafetensorsLoader, TensorEntry (engine white-box)

namespace rt {
namespace {

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

double to_gb(std::size_t bytes) noexcept {
    return static_cast<double>(bytes) / kGiB;
}

// ---- the two terms that describe an ALLOCATOR rather than a file -------------
//
// F5-TTS is loaded through the ONNXRuntime CUDA execution provider, so its cost
// is not a file size: the DiT weights are allocated THROUGH the EP's arena, and
// the arena also has to hold the attention intermediates a long utterance
// produces. Both figures below are measurements, recorded here so nobody
// re-derives them from guesses (main.cpp's [tts] block carries the full account):
//
//   * the fp32 DiT weights measure 1.31 GB;
//   * a gpu_mem_limit_mb sweep found a 2048 MiB cap costing 2219 MiB committed,
//     1792 MiB costing the same 2219, and 1536 MiB failing to initialise at all;
//   * a 335-token utterance asks ONE attention node for 385 MB, which is why the
//     cap is unbounded by default and why the arena term below is not just the
//     difference between the two numbers above.
//
// Split into two named terms rather than one, because they answer different
// questions: kTtsDitBytes is what the graph weighs, kTtsOrtArenaBytes is what
// running it needs on top. A future fp16 export moves the first and not the
// second.
constexpr std::size_t kTtsDitBytes      = 1434ull * 1024ull * 1024ull;   // ~1.40 GiB
constexpr std::size_t kTtsOrtArenaBytes = 1024ull * 1024ull * 1024ull;   // ~1.00 GiB

// whisper.cpp uploads the GGML model and then allocates its own compute buffers
// (KV for the 32-token decoder, the conv/encoder scratch). Measured at roughly a
// sixth of the model on Whisper-Turbo geometry; taken as a flat 20% so the term
// errs toward refusing a context rather than toward WDDM paging.
constexpr double kGgmlComputeOverhead = 0.20;

bool file_size_bytes(const std::string& path, std::size_t* out) noexcept {
    if (path.empty()) return false;
    std::error_code ec;
    const auto n = std::filesystem::file_size(std::filesystem::path(path), ec);
    if (ec) return false;
    *out = static_cast<std::size_t>(n);
    return true;
}

// Sum every tensor a safetensors checkpoint declares, WITHOUT reading tensor
// data -- SafetensorsLoader parses only the JSON header (and, for an index, the
// shard map). `prefixes` empty means "every tensor"; otherwise only the tensors
// whose name starts with one of them, which is how the audio head is sized
// without charging the app for the 15 GB of backbone weights sitting in the same
// directory that it never uploads.
//
// Returns false when nothing could be read at all, which the caller turns into
// an advisory budget rather than a fabricated number.
bool sum_safetensors(const std::string& file, const std::vector<std::string>& prefixes,
                     std::size_t* out) noexcept {
    try {
        const SafetensorsLoader st(file);
        std::size_t total = 0;
        for (const std::string& name : st.list_tensors()) {
            if (!prefixes.empty()) {
                const bool wanted =
                    std::any_of(prefixes.begin(), prefixes.end(),
                                [&name](const std::string& p) {
                                    return name.compare(0, p.size(), p) == 0;
                                });
                if (!wanted) continue;
            }
            total += st.get_tensor(name).byte_size;
        }
        if (total == 0) return false;
        *out = total;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

// The backbone. Prefer the shard index (a multi-file checkpoint's only complete
// description); fall back to a single-file checkpoint.
bool measure_backbone(const std::string& model_dir, std::size_t* out) noexcept {
    if (model_dir.empty()) return false;
    if (sum_safetensors(model_dir + "/model.safetensors.index.json", {}, out)) return true;
    return sum_safetensors(model_dir + "/model.safetensors", {}, out);
}

// The audio head, which costs TWICE what the checkpoint holds: audio_head_loader
// widens every BF16 tensor to fp32 on upload (a left shift by 16), and
// whisper_encoder.h carries its weights as `const float*` throughout, so there is
// no path on which the narrow form reaches the device.
bool measure_audio_head(const std::string& dir, std::size_t* out) noexcept {
    if (dir.empty()) return false;
    std::size_t bf16 = 0;
    if (!sum_safetensors(dir + "/model.safetensors",
                         {"audio_tower.", "multi_modal_projector."}, &bf16)) {
        return false;
    }
    *out = bf16 * 2;   // BF16 -> fp32
    return true;
}

// ---- KV geometry, straight from config.json ---------------------------------
// Not from the engine: it has not been constructed yet, and constructing it is
// exactly the act this budget exists to gate.
struct KvGeometry {
    int layers = 0;
    int kv_heads = 0;
    int head_dim = 0;
    bool valid() const noexcept { return layers > 0 && kv_heads > 0 && head_dim > 0; }
};

KvGeometry read_kv_geometry(const std::string& model_dir) noexcept {
    KvGeometry g;
    if (model_dir.empty()) return g;
    try {
        std::ifstream f(model_dir + "/config.json", std::ios::binary);
        if (!f) return g;
        nlohmann::json j;
        f >> j;
        // A multimodal wrapper (Ultravox) nests the backbone under text_config;
        // resolve_projector_params reads hidden_size the same way, so the two
        // agree on which object describes the model that owns the KV cache.
        const nlohmann::json& m =
            (j.contains("text_config") && j["text_config"].is_object()) ? j["text_config"] : j;

        g.layers = m.value("num_hidden_layers", 0);
        // GQA/MQA: num_key_value_heads is what the cache is sized by. A model
        // without it is MHA, where the KV head count equals the query head count.
        const int q_heads = m.value("num_attention_heads", 0);
        g.kv_heads = m.value("num_key_value_heads", q_heads);
        // head_dim is explicit on newer configs and derived on older ones.
        g.head_dim = m.value("head_dim", 0);
        if (g.head_dim <= 0 && q_heads > 0) {
            const int hidden = m.value("hidden_size", 0);
            if (hidden > 0) g.head_dim = hidden / q_heads;
        }
    } catch (const std::exception&) {
        return KvGeometry{};
    }
    return g;
}

}  // namespace

VramBudget plan_vram_budget(const VramBudgetInputs& in) noexcept {
    VramBudget b;
    b.requested_context = in.requested_context;
    b.granted_context = in.requested_context;

    // ---- what the driver actually has left ----------------------------------
    // `free`, not `total`: the desktop compositor, the browser and every other
    // process on the card have already taken their share, and a budget against
    // the nameplate capacity would be a budget against a machine nobody is using.
    {
        std::size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
            b.advisory_only = true;
            b.advisory_reason = "cudaMemGetInfo failed -- no device memory figure to budget against";
            return b;
        }
        b.device_free = free_b;
        b.device_total = total_b;
    }

    // ---- the fixed terms -----------------------------------------------------
    std::size_t backbone = 0;
    if (measure_backbone(in.model_dir, &backbone)) {
        b.weights += backbone;
    } else {
        b.advisory_only = true;
        b.advisory_reason = "could not read the backbone safetensors header in " + in.model_dir;
    }

    if (in.load_audio_head) {
        std::size_t head = 0;
        if (measure_audio_head(in.audio_head_dir, &head)) {
            b.weights += head;
        } else {
            b.advisory_only = true;
            b.advisory_reason = "could not read the audio head header in " + in.audio_head_dir;
        }
    }

    if (!in.whisper_model_path.empty()) {
        std::size_t ggml = 0;
        if (file_size_bytes(in.whisper_model_path, &ggml)) {
            b.ggml = ggml + static_cast<std::size_t>(static_cast<double>(ggml) *
                                                     kGgmlComputeOverhead);
        } else {
            b.advisory_only = true;
            b.advisory_reason = "could not stat the GGML model at " + in.whisper_model_path;
        }
    }

    if (in.tts_enabled) {
        b.dit = kTtsDitBytes;
        b.ort = kTtsOrtArenaBytes;
    }

    // ---- the term we control -------------------------------------------------
    // The paged KV cache is BF16: per token, per branch,
    //     layers * kv_heads * head_dim * 2 (K and V) * 2 bytes
    // which is 128 KB/token at 8B geometry (32 x 8 x 128 x 2 x 2) -- the figure
    // engine_bootstrap.hpp quotes.
    const KvGeometry g = read_kv_geometry(in.model_dir);
    const int branches = in.branch_factor > 0 ? in.branch_factor : 1;
    if (g.valid()) {
        b.kv_bytes_per_token = static_cast<std::size_t>(g.layers) *
                               static_cast<std::size_t>(g.kv_heads) *
                               static_cast<std::size_t>(g.head_dim) * 2ull * 2ull *
                               static_cast<std::size_t>(branches);
    } else {
        b.advisory_only = true;
        b.advisory_reason = "config.json in " + in.model_dir +
                            " does not describe the KV geometry (layers / kv heads / head dim)";
    }

    // An advisory budget REPORTS and does not clamp. Clamping on a term we failed
    // to measure would refuse contexts that fit, which is a worse failure than
    // the one this guard exists to prevent: it breaks a configuration that was
    // working, silently, on the basis of a number nobody can check.
    if (b.advisory_only) {
        b.kv = b.kv_bytes_per_token * static_cast<std::size_t>(b.granted_context);
        return b;
    }

    // ---- the decision --------------------------------------------------------
    const std::size_t fixed = b.fixed_total();
    if (fixed >= b.device_free) {
        b.fits = false;
        b.granted_context = 0;
        b.kv = 0;
        b.failure =
            "the fixed VRAM terms (weights " + std::to_string(b.weights >> 20) +
            " MiB + GGML " + std::to_string(b.ggml >> 20) + " MiB + DiT " +
            std::to_string(b.dit >> 20) + " MiB + ORT arena " + std::to_string(b.ort >> 20) +
            " MiB + " + std::to_string(kSafetyReserveBytes >> 20) + " MiB reserve) already "
            "exceed the " + std::to_string(b.device_free >> 20) +
            " MiB free on this device -- there is no room for a KV cache of any size. "
            "Free VRAM, or disable speech output / the cascade, or load a smaller checkpoint.";
        return b;
    }

    const std::size_t kv_budget = b.device_free - fixed;
    const std::size_t affordable_tokens = kv_budget / b.kv_bytes_per_token;

    if (affordable_tokens < static_cast<std::size_t>(kMinViableContext)) {
        b.fits = false;
        b.granted_context = 0;
        b.kv = 0;
        b.failure =
            "only " + std::to_string(affordable_tokens) + " KV tokens fit in the " +
            std::to_string(kv_budget >> 20) + " MiB left after the fixed terms, which is below "
            "the " + std::to_string(kMinViableContext) + "-token minimum. Free VRAM, or "
            "disable speech output / the cascade, or load a smaller checkpoint.";
        return b;
    }

    if (affordable_tokens < static_cast<std::size_t>(b.requested_context)) {
        // Rounded DOWN to the granularity, then re-floored: the rounding must
        // never push the result back above what was affordable, and it must never
        // push it below the minimum we just proved fits.
        int granted = static_cast<int>(affordable_tokens);
        granted -= granted % kContextGranularity;
        if (granted < kMinViableContext) granted = kMinViableContext;
        b.granted_context = granted;
        b.clamped = true;
    }

    b.kv = b.kv_bytes_per_token * static_cast<std::size_t>(b.granted_context);
    return b;
}

void log_vram_budget(const VramBudget& b) noexcept {
    std::printf("[vram-budget] device: %.2f GB free / %.2f GB total\n", to_gb(b.device_free),
                to_gb(b.device_total));

    if (b.advisory_only) {
        // Named as advisory on its own line, because the numbers below are still
        // printed and would otherwise read as a guarantee they are not making.
        std::fprintf(stderr,
                     "[vram-budget] WARN: advisory only -- %s. max_context is NOT being "
                     "guarded on this launch.\n",
                     b.advisory_reason.c_str());
    }

    std::printf("[vram-budget]   weights (backbone + audio head) : %6.2f GB\n", to_gb(b.weights));
    if (b.ggml > 0) std::printf("[vram-budget]   GGML (whisper.cpp)              : %6.2f GB\n",
                                to_gb(b.ggml));
    if (b.dit > 0)  std::printf("[vram-budget]   DiT (F5-TTS)                    : %6.2f GB\n",
                                to_gb(b.dit));
    if (b.ort > 0)  std::printf("[vram-budget]   ORT CUDA EP arena               : %6.2f GB\n",
                                to_gb(b.ort));
    std::printf("[vram-budget]   OS / D3D reserve                : %6.2f GB\n", to_gb(b.reserve));

    if (b.kv_bytes_per_token > 0) {
        // Padded to the same column as the fixed terms above: the whole point of
        // this block is that the reader compares the KV line against them.
        char label[48] = {};
        std::snprintf(label, sizeof(label), "KV @ %d tokens", b.granted_context);
        std::printf("[vram-budget]   %-32s: %6.2f GB (%zu KB/token)\n", label, to_gb(b.kv),
                    b.kv_bytes_per_token >> 10);
    }

    if (!b.fits) {
        std::fprintf(stderr, "[vram-budget] FATAL: %s\n", b.failure.c_str());
        std::fflush(stdout);
        return;
    }

    if (b.clamped) {
        // THE WARN THE GUARD EXISTS TO EMIT. Loud, and it says both numbers: a
        // user who set 32768 deliberately needs to know they are running 3584,
        // and needs it to be obvious that the app decided rather than ignored.
        std::fprintf(stderr,
                     "[vram-budget] WARN: max_context %d does not fit in VRAM -- clamped to "
                     "%d (the largest that fits with a %.2f GB reserve held back). The "
                     "persisted setting is unchanged; free VRAM or lower it in Settings to "
                     "silence this.\n",
                     b.requested_context, b.granted_context, to_gb(b.reserve));
    } else {
        std::printf("[vram-budget] max_context %d fits (%.2f GB of headroom beyond the "
                    "reserve)\n",
                    b.granted_context,
                    to_gb(b.device_free - b.fixed_total() - b.kv));
    }
    std::fflush(stdout);
}

}  // namespace rt
