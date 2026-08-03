// -----------------------------------------------------------------------------
// test_f5_tts_cuda_cpp — the end-to-end F5-TTS smoke tool: reference WAV + text
// in, synthesised speech out, on the GPU, through the real C++ stack.
//
// WHAT IT PROVES. Every piece of this subsystem has been verified in isolation
// (mel against torchaudio, graphs against PyTorch at two sequence lengths, the
// CUDA EP against a link probe). This is the first thing that runs them in
// series, which is the only way to catch the errors that live BETWEEN them:
// a mel handed over mel-major, a reference prefix counted twice, a duration
// estimate that produces a tensor the graph rejects.
//
// It is deliberately a CLI rather than a ctest case: it needs a ~1.3 GB model,
// a GPU, and a reference clip, none of which belong in the fast `validation`
// suite. Wire it into `integration` once the assets have a pinned home.
//
//   test_f5_tts_cuda_cpp --ckpt-dir models/f5_tts --vocab D:/TTF/vocab.txt
//                        --ref-audio ref.wav --ref-text "..." --text "..."
//
// THREADING: single-threaded by construction. F5TtsEngine requires one owning
// thread and this tool has exactly one.
// -----------------------------------------------------------------------------
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
// For the UTF-16 -> UTF-8 argv conversion below, and to put the console into
// UTF-8 so the Cyrillic this tool exists to print is not mangled on the way out.
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <nlohmann/json.hpp>

// dr_wav DECLARATIONS only. The implementation lives in dr_wav_impl.cpp, which
// is the single TU in the root build that defines DR_WAV_IMPLEMENTATION -- see
// that file for why it has to be separate (a codegen-time C4701 that no pragma
// around the include can suppress).
#include "dr_wav.h"

#include "f5_mel_extractor.hpp"
#include "f5_tokenizer.hpp"
#include "f5_tts_engine.hpp"

namespace {

using blackwell::tts::F5MelConfig;
using blackwell::tts::F5MelExtractor;
using blackwell::tts::F5Tokenizer;
using blackwell::tts::F5TtsConfig;
using blackwell::tts::F5TtsEngine;
using blackwell::tts::TtsStatus;

// F5's target_rms in infer_batch_process. See the normalisation block in run().
constexpr double kTargetRms = 0.1;

struct Args {
    std::string ckpt_dir  = "models/f5_tts";
    std::string vocab;
    std::string ref_audio;
    std::string ref_text;
    std::string text;
    std::string out = "output_f5_cpp.wav";
    int   nfe   = 16;
    float speed = 1.0f;
    float cfg   = 2.0f;
    int   device = 0;
    std::size_t max_frames = 3000;
    int   opt_level = 3;   // ORT graph optimization; see F5TtsConfig
    std::size_t gpu_mem_limit_mb = 2048;   // ORT CUDA arena ceiling; see F5TtsConfig
    // Debug seam: writes the ref+gen id sequences as raw int32 so they can be
    // diffed against f5_tts' own list_str_to_idx. Character ids are LINE INDICES
    // into vocab.txt, so an off-by-one or a mis-parsed vocab yields fluent
    // nonsense rather than an error -- the only way to know they are right is to
    // compare them.
    std::string dump_ids;
};

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "\n[FATAL] %s\n\n", msg.c_str());
    std::exit(1);
}

void usage() {
    std::printf(
        "test_f5_tts_cuda_cpp — end-to-end F5-TTS synthesis on the CUDA EP\n\n"
        "  --ckpt-dir DIR    folder with f5_tts_dit.onnx / f5_tts_vocoder.onnx /\n"
        "                    f5_tts_contract.json          (default: models/f5_tts)\n"
        "  --vocab FILE      vocab.txt THAT SHIPPED WITH THE CHECKPOINT (required)\n"
        "  --ref-audio FILE  reference WAV, 24 kHz (required)\n"
        "  --ref-text  TEXT  transcript OF THAT AUDIO (required)\n"
        "  --text      TEXT  text to synthesise (required)\n"
        "  --out FILE        output WAV            (default: output_f5_cpp.wav)\n"
        "  --nfe N           solver steps          (default: 16)\n"
        "  --speed F         speech rate           (default: 1.0)\n"
        "  --max-frames N    frame-axis cap        (default: 3000)\n"
        "  --device N        CUDA device           (default: 0)\n"
        "  --gpu-mem-limit N ORT CUDA arena ceiling, MiB (default: 2048, 0 = none).\n"
        "                    Bounds the ARENA ONLY -- weights are allocated outside\n"
        "                    it, so total VRAM is ~1.3 GB + what this permits.\n"
        "  --opt-level N     ORT graph opt level    (default: 3)\n"
        "                    0=off 1=basic 2=extended 3=all. Bisect this when an\n"
        "                    fp16 DiT yields silence: a fused fp16 kernel can be\n"
        "                    wrong where the graph is fine.\n");
}

Args parse_args(const std::vector<std::string>& argv) {
    Args a;
    const int argc = static_cast<int>(argv.size());
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[static_cast<std::size_t>(i)];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) die(std::string("missing value for ") + what);
            return argv[static_cast<std::size_t>(++i)];
        };
        if (k == "--help" || k == "-h")      { usage(); std::exit(0); }
        else if (k == "--ckpt-dir")          a.ckpt_dir = next("--ckpt-dir");
        else if (k == "--vocab")             a.vocab = next("--vocab");
        else if (k == "--ref-audio")         a.ref_audio = next("--ref-audio");
        else if (k == "--ref-text")          a.ref_text = next("--ref-text");
        else if (k == "--text")              a.text = next("--text");
        else if (k == "--out")               a.out = next("--out");
        else if (k == "--nfe")               a.nfe = std::stoi(next("--nfe"));
        else if (k == "--speed")             a.speed = std::stof(next("--speed"));
        else if (k == "--max-frames")        a.max_frames =
                                                 static_cast<std::size_t>(std::stoul(next("--max-frames")));
        else if (k == "--device")            a.device = std::stoi(next("--device"));
        else if (k == "--opt-level")         a.opt_level = std::stoi(next("--opt-level"));
        else if (k == "--gpu-mem-limit")     a.gpu_mem_limit_mb =
                                                 static_cast<std::size_t>(std::stoul(next("--gpu-mem-limit")));
        else if (k == "--dump-ids")          a.dump_ids = next("--dump-ids");
        else die("unknown argument: " + k);
    }
    if (a.vocab.empty())     die("--vocab is required");
    if (a.ref_audio.empty()) die("--ref-audio is required");
    if (a.ref_text.empty())  die("--ref-text is required (F5 conditions on an (audio, TEXT) pair)");
    if (a.text.empty())      die("--text is required");
    return a;
}

bool exists(const std::string& p) { return std::ifstream(p).good(); }

// Mono float32 at exactly kF5SampleRate, or die explaining what to do.
std::vector<float> load_wav_24k_mono(const std::string& path) {
    unsigned int channels = 0, rate = 0;
    drwav_uint64 total = 0;
    float* raw = drwav_open_file_and_read_pcm_frames_f32(path.c_str(), &channels, &rate,
                                                         &total, nullptr);
    if (raw == nullptr) die("cannot read WAV: " + path);

    std::vector<float> mono(static_cast<std::size_t>(total));
    if (channels <= 1) {
        std::memcpy(mono.data(), raw, mono.size() * sizeof(float));
    } else {
        // Plain average downmix, matching what miniaudio/torchaudio do for a
        // reference clip. Nothing here is quality-critical: the model conditions
        // on timbre, not on stereo image.
        for (std::size_t i = 0; i < mono.size(); ++i) {
            float acc = 0.0f;
            for (unsigned int c = 0; c < channels; ++c) {
                acc += raw[i * channels + c];
            }
            mono[i] = acc / static_cast<float>(channels);
        }
    }
    drwav_free(raw, nullptr);

    if (rate != static_cast<unsigned int>(blackwell::tts::kF5SampleRate)) {
        // NOT resampled here on purpose. A wrong-rate reference does not fail —
        // it clones a voice pitched by the rate ratio, which sounds like a
        // different person and reads as a model problem. Resampling badly would
        // be its own quality bug, so the tool refuses and names the fix.
        die("reference WAV is " + std::to_string(rate) + " Hz; F5 needs " +
            std::to_string(blackwell::tts::kF5SampleRate) + " Hz.\n"
            "  Convert it first, e.g.:\n"
            "    ffmpeg -i \"" + path + "\" -ar 24000 -ac 1 ref_24k.wav");
    }
    if (mono.empty()) die("reference WAV has no samples: " + path);
    return mono;
}

// 16-bit PCM out: what every player opens without complaint. The engine's f32 is
// the lossless artefact; this file is for listening.
void write_wav_16(const std::string& path, const std::vector<float>& pcm, int rate) {
    drwav_data_format fmt{};
    fmt.container     = drwav_container_riff;
    fmt.format        = DR_WAVE_FORMAT_PCM;
    fmt.channels      = 1;
    fmt.sampleRate    = static_cast<drwav_uint32>(rate);
    fmt.bitsPerSample = 16;

    drwav w;
    if (!drwav_init_file_write(&w, path.c_str(), &fmt, nullptr)) {
        die("cannot open for writing: " + path);
    }
    std::vector<drwav_int16> s16(pcm.size());
    for (std::size_t i = 0; i < pcm.size(); ++i) {
        const float v = pcm[i] < -1.0f ? -1.0f : (pcm[i] > 1.0f ? 1.0f : pcm[i]);
        s16[i] = static_cast<drwav_int16>(v * 32767.0f);
    }
    drwav_write_pcm_frames(&w, s16.size(), s16.data());
    drwav_uninit(&w);
}

// The contract JSON the exporter writes. Read rather than assumed: the tensor
// names are the one thing that can drift between a re-export and this binary,
// and a mismatch is caught by F5TtsEngine's signature validation at load with a
// much worse message than this one.
void apply_contract(const std::string& json_path, F5TtsConfig& cfg) {
    if (!exists(json_path)) {
        std::printf("  contract      : (absent — using built-in F5DitContract defaults)\n");
        return;
    }
    std::ifstream f(json_path);
    nlohmann::json j;
    f >> j;
    const auto& d = j.at("dit");
    cfg.contract.latent_in  = d.at("latent_in").get<std::string>();
    cfg.contract.cond_mel   = d.at("cond_mel").get<std::string>();
    cfg.contract.text_ids   = d.at("text_ids").get<std::string>();
    cfg.contract.time_step  = d.at("time_step").get<std::string>();
    cfg.contract.delta_t    = d.at("delta_t").get<std::string>();
    cfg.contract.latent_out = d.at("latent_out").get<std::string>();
    const auto& v = j.at("vocoder");
    cfg.contract.vocoder_in  = v.at("vocoder_in").get<std::string>();
    cfg.contract.vocoder_out = v.at("vocoder_out").get<std::string>();

    std::printf("  contract      : %s\n", json_path.c_str());
    if (j.contains("baked_in")) {
        const auto& b = j["baked_in"];
        std::printf("  baked into DiT: cfg=%.2f (%s), euler=%s, opset=%d\n",
                    b.value("cfg_strength", 0.0), b.value("cfg_convention", "?").c_str(),
                    b.value("euler_update", false) ? "yes" : "no", b.value("opset", 0));
    }
    if (j.contains("geometry")) {
        const auto& g = j["geometry"];
        // Cross-check the graph's geometry against what this binary was compiled
        // for. These are compile-time constants on the C++ side, so a mismatch
        // cannot be fixed by configuration — better to say so now than to
        // produce a tensor the graph silently reinterprets.
        const int mel = g.value("mel_channels", blackwell::tts::kF5MelChannels);
        const int sr  = g.value("sample_rate", blackwell::tts::kF5SampleRate);
        const int hop = g.value("hop_length", blackwell::tts::kF5HopLength);
        if (mel != blackwell::tts::kF5MelChannels || sr != blackwell::tts::kF5SampleRate ||
            hop != blackwell::tts::kF5HopLength) {
            die("contract geometry (sr=" + std::to_string(sr) + " mel=" + std::to_string(mel) +
                " hop=" + std::to_string(hop) + ") disagrees with this build (sr=" +
                std::to_string(blackwell::tts::kF5SampleRate) + " mel=" +
                std::to_string(blackwell::tts::kF5MelChannels) + " hop=" +
                std::to_string(blackwell::tts::kF5HopLength) + "). Rebuild or re-export.");
        }
    }
}

}  // namespace

// =============================================================================
// ENTRY POINT — and on Windows this is NOT a formality.
//
// A plain `int main(int, char**)` receives argv in the process ANSI code page.
// Cyrillic is not representable in the typical Windows-125x/866 console page, so
// the CRT substitutes a literal '?' for EVERY Cyrillic character before this
// program sees a byte of it.
//
// That failure is worse than mojibake, because '?' IS in the F5 vocabulary
// (id 30). The tokenizer then reports ZERO unknown characters, the engine
// happily synthesises a 9-second monotone drone from sixty question marks, and
// every check short of listening to the output passes. Measured before this fix:
// every id came out 30, and the audio had 4.8 dB of dynamic range against the
// 30-40 dB real speech shows.
//
// So the arguments are taken as UTF-16 (wmain, which MSVC wires to
// wmainCRTStartup) and converted to UTF-8 exactly once, here. Everything below
// this line is UTF-8 end to end, matching /utf-8 on the sources and the vocab
// file's own encoding.
// =============================================================================
int run(const std::vector<std::string>& argv_utf8);

#if defined(_WIN32)
namespace {
std::string utf16_to_utf8(const wchar_t* w) {
    if (w == nullptr || *w == L'\0') return std::string();
    const int need = ::WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    if (need <= 1) return std::string();
    std::string out(static_cast<std::size_t>(need - 1), '\0');
    ::WideCharToMultiByte(CP_UTF8, 0, w, -1, out.data(), need, nullptr, nullptr);
    return out;
}
}  // namespace

int wmain(int argc, wchar_t** wargv) {
    // So the Cyrillic we print comes back out intact too.
    ::SetConsoleOutputCP(CP_UTF8);
    std::vector<std::string> args;
    args.reserve(static_cast<std::size_t>(argc));
    for (int i = 0; i < argc; ++i) args.push_back(utf16_to_utf8(wargv[i]));
    return run(args);
}
#else
int main(int argc, char** argv) {
    return run(std::vector<std::string>(argv, argv + argc));
}
#endif

int run(const std::vector<std::string>& argv_utf8) {
    const Args args = parse_args(argv_utf8);
    const std::string dit = args.ckpt_dir + "/f5_tts_dit.onnx";
    const std::string voc = args.ckpt_dir + "/f5_tts_vocoder.onnx";

    std::printf("\n=== F5-TTS / C++ / CUDA EP ===\n");
    if (!exists(dit)) die("missing DiT graph: " + dit);
    if (!exists(voc)) die("missing vocoder graph: " + voc);
    std::printf("  dit           : %s\n", dit.c_str());
    std::printf("  vocoder       : %s\n", voc.c_str());

    F5TtsConfig cfg;
    cfg.dit_model_path     = dit;
    cfg.vocoder_model_path = voc;
    cfg.device_id          = args.device;
    cfg.nfe_step           = args.nfe;
    cfg.cfg_strength       = args.cfg;
    cfg.speed              = args.speed;
    cfg.max_frames         = args.max_frames;
    cfg.graph_opt_level    = args.opt_level;
    cfg.gpu_mem_limit_mb   = args.gpu_mem_limit_mb;
    apply_contract(args.ckpt_dir + "/f5_tts_contract.json", cfg);

    // ---- reference audio -> mel -------------------------------------------
    const std::vector<float> ref_pcm = load_wav_24k_mono(args.ref_audio);
    std::printf("\n  ref audio     : %s\n", args.ref_audio.c_str());
    std::printf("                  %.2f s @ %d Hz mono\n",
                static_cast<double>(ref_pcm.size()) / blackwell::tts::kF5SampleRate,
                blackwell::tts::kF5SampleRate);

    // ---- reference RMS normalisation --------------------------------------
    // F5's infer_batch_process scales the reference to target_rms = 0.1 BEFORE
    // taking its mel, and scales the generated audio back down by the inverse
    // afterwards. Skipping it is not a loudness nicety: the mel is a LOG
    // magnitude, so a gain of g shifts every conditioning value by ln(g)
    // uniformly. This clip sits at RMS 0.024, i.e. +12.3 dB / +1.41 in log-mel
    // -- far outside anything the model saw in training, and it degrades the
    // output badly while still producing well-formed audio.
    std::vector<float> ref_norm = ref_pcm;
    double sumsq_ref = 0.0;
    for (const float v : ref_norm) sumsq_ref += static_cast<double>(v) * v;
    const double ref_rms = std::sqrt(sumsq_ref / static_cast<double>(ref_norm.size()));
    double rms_gain = 1.0;
    if (ref_rms > 0.0 && ref_rms < kTargetRms) {
        rms_gain = kTargetRms / ref_rms;
        for (float& v : ref_norm) v = static_cast<float>(v * rms_gain);
    }
    std::printf("  ref rms       : %.5f -> gain %.4f (%+.2f dB), log-mel shift %+.4f\n",
                ref_rms, rms_gain, 20.0 * std::log10(rms_gain), std::log(rms_gain));

    F5MelExtractor mel_ex{F5MelConfig{}};
    const auto t_mel0 = std::chrono::steady_clock::now();
    const std::vector<float> ref_mel = mel_ex.Compute(ref_norm);
    const auto t_mel1 = std::chrono::steady_clock::now();
    const std::size_t ref_frames = mel_ex.FrameCount(ref_pcm.size());
    std::printf("  ref mel       : %zu frames x %d bands (%.1f ms)\n", ref_frames,
                blackwell::tts::kF5MelChannels,
                std::chrono::duration<double, std::milli>(t_mel1 - t_mel0).count());

    // ---- tokenise ----------------------------------------------------------
    F5Tokenizer tok(args.vocab);
    std::size_t unk_ref = 0, unk_gen = 0;
    const std::vector<std::int32_t> ref_ids32 = tok.Tokenize(args.ref_text, &unk_ref);
    const std::vector<std::int32_t> gen_ids32 = tok.Tokenize(args.text, &unk_gen);
    std::printf("  vocab         : %s (%zu entries)\n", args.vocab.c_str(), tok.vocab_size());
    std::printf("  ref text      : %zu chars -> %zu ids%s\n", args.ref_text.size(),
                ref_ids32.size(), unk_ref ? "  [UNKNOWNS]" : "");
    std::printf("  gen text      : %zu chars -> %zu ids%s\n", args.text.size(),
                gen_ids32.size(), unk_gen ? "  [UNKNOWNS]" : "");

    if (unk_ref || unk_gen) {
        // Loud, because this is what "Russian text against the base EN/ZH vocab"
        // looks like from here, and the audio it produces is confident nonsense
        // rather than anything that sounds broken.
        std::printf("\n  !! %zu character(s) are not in this voice's alphabet and were\n"
                    "  !! replaced with the space id. Output will be wrong. Unknown:",
                    unk_ref + unk_gen);
        for (const auto& s : tok.UnknownCharacters(args.ref_text + args.text)) {
            std::printf(" '%s'", s.c_str());
        }
        std::printf("\n  !! Check that --vocab is the file that shipped with this checkpoint.\n");
    }

    if (!args.dump_ids.empty()) {
        // Layout: [int32 n_ref][int32 n_gen][n_ref int32][n_gen int32].
        std::ofstream f(args.dump_ids, std::ios::binary);
        const std::int32_t nr = static_cast<std::int32_t>(ref_ids32.size());
        const std::int32_t ng = static_cast<std::int32_t>(gen_ids32.size());
        f.write(reinterpret_cast<const char*>(&nr), sizeof(nr));
        f.write(reinterpret_cast<const char*>(&ng), sizeof(ng));
        f.write(reinterpret_cast<const char*>(ref_ids32.data()),
                static_cast<std::streamsize>(ref_ids32.size() * sizeof(std::int32_t)));
        f.write(reinterpret_cast<const char*>(gen_ids32.data()),
                static_cast<std::streamsize>(gen_ids32.size() * sizeof(std::int32_t)));
        std::printf("  dumped ids    : %s\n", args.dump_ids.c_str());
    }

    // The engine takes int64 ids (the tokenizer emits int32, which is what the
    // graph's text tensor wants; the widening happens once, here).
    const std::vector<std::int64_t> ref_ids(ref_ids32.begin(), ref_ids32.end());
    const std::vector<std::int64_t> gen_ids(gen_ids32.begin(), gen_ids32.end());

    // ---- engine ------------------------------------------------------------
    std::printf("\n  loading graphs onto CUDA device %d ...\n", args.device);
    const auto t_load0 = std::chrono::steady_clock::now();
    F5TtsEngine engine(cfg, nullptr);   // no MelExtractorFn: we pass the mel directly
    const auto t_load1 = std::chrono::steady_clock::now();
    std::printf("  load          : %.2f s\n",
                std::chrono::duration<double>(t_load1 - t_load0).count());

    TtsStatus st = engine.SetReferenceMel(ref_mel.data(), ref_frames, ref_ids);
    if (st != TtsStatus::Success) {
        die(std::string("SetReferenceMel failed: ") + blackwell::tts::to_string(st) +
            "\n  reference is " + std::to_string(ref_frames) + " frames; the engine caps it "
            "at max_frames/2 = " + std::to_string(cfg.max_frames / 2) +
            ". Use a shorter clip or raise --max-frames.");
    }
    std::printf("  reference     : %zu frames cached, %zu frames left for speech\n",
                engine.reference_frames(), engine.available_frames());

    // F5's own duration formula, reproduced exactly (infer_batch_process):
    //   ref_audio_len = n_samples // hop      <- NOT the mel frame count
    //   ref/gen text measured in UTF-8 BYTES  <- NOT token counts
    // Both differ from the engine's built-in estimate; for Cyrillic the gap is
    // small (1599 vs 1582 frames here) but it is free to be exact, and the
    // formula is only computable where the raw text is, which is here.
    //
    // F5 also appends a space to ref_text when it ends in a 1-byte character,
    // BEFORE measuring it -- so the byte count below must match the text that
    // was actually tokenised.
    std::string ref_text_f5 = args.ref_text;
    if (!ref_text_f5.empty() &&
        static_cast<unsigned char>(ref_text_f5.back()) < 0x80u) {
        ref_text_f5 += ' ';
    }
    const std::size_t ref_audio_len = ref_pcm.size() /
                                      static_cast<std::size_t>(blackwell::tts::kF5HopLength);
    const std::size_t total_frames =
        ref_audio_len + static_cast<std::size_t>(
            static_cast<double>(ref_audio_len) *
            static_cast<double>(args.text.size()) /
            static_cast<double>(ref_text_f5.size()) / static_cast<double>(args.speed));
    std::printf("  duration      : %zu total frames (F5 byte-ratio formula; "
                "engine estimate would be %zu)\n",
                total_frames, engine.EstimateFrames(gen_ids.size()));

    // ---- synthesise --------------------------------------------------------
    std::printf("\n  synthesising (nfe=%d, cfg=%.1f, speed=%.2f) ...\n",
                args.nfe, static_cast<double>(args.cfg), static_cast<double>(args.speed));
    std::atomic<bool> interrupt{false};   // never set here; the barge-in seam
    std::vector<float> out_pcm;

    // Warm-up is deliberately NOT done: this measurement is time-to-first-audio
    // for a cold utterance, which is what a duplex assistant actually pays. The
    // engine's own counters below split it into solve vs vocode.
    const auto t0 = std::chrono::steady_clock::now();
    st = engine.GenerateAudio(gen_ids, &interrupt, out_pcm, total_frames);
    const auto t1 = std::chrono::steady_clock::now();

    if (st != TtsStatus::Success) {
        die(std::string("GenerateAudio failed: ") + blackwell::tts::to_string(st) +
            "  (synthesis_errors=" + std::to_string(engine.synthesis_errors()) + ")");
    }
    if (out_pcm.empty()) die("GenerateAudio returned Success but produced no samples");

    const double wall     = std::chrono::duration<double>(t1 - t0).count();
    const double audio_s  = static_cast<double>(out_pcm.size()) / blackwell::tts::kF5SampleRate;

    // Undo the reference normalisation, as F5 does, so the output sits at the
    // reference clip's original loudness instead of ~4x hotter.
    if (rms_gain != 1.0) {
        const float inv = static_cast<float>(1.0 / rms_gain);
        for (float& v : out_pcm) v *= inv;
    }

    float peak = 0.0f;
    double sumsq = 0.0;
    for (const float v : out_pcm) {
        const float a = v < 0.0f ? -v : v;
        if (a > peak) peak = a;
        sumsq += static_cast<double>(v) * v;
    }

    write_wav_16(args.out, out_pcm, blackwell::tts::kF5SampleRate);

    std::printf("\n=== RESULT ===\n");
    std::printf("  wrote         : %s  (%zu samples, %.3f s)\n",
                args.out.c_str(), out_pcm.size(), audio_s);
    std::printf("  peak / rms    : %.4f / %.4f\n", static_cast<double>(peak),
                std::sqrt(sumsq / static_cast<double>(out_pcm.size())));
    std::printf("\n  TTFB          : %.3f s  (== full synthesis; F5 is not streaming)\n", wall);
    std::printf("  solve  (%2d st): %.3f s   (%.1f ms/step)\n", args.nfe,
                engine.last_solve_ms() / 1000.0, engine.last_solve_ms() / args.nfe);
    std::printf("  vocode        : %.3f s\n", engine.last_vocode_ms() / 1000.0);
    std::printf("  RTF           : %.3f   (%.1fx realtime)\n",
                wall / audio_s, audio_s / wall);
    return 0;
}
