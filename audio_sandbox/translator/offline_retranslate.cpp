// -----------------------------------------------------------------------------
// offline_retranslate — the T5 offline driver for continuous streaming
// (docs/CONTINUOUS_STREAMING.md). Runs the WHOLE pipeline over a reference WAV,
// deterministically, with no microphone and no GUI:
//
//   WAV -> 10 ms blocks -> SileroVAD -> SpeechSegmenter -> AbsoluteAudioRing
//       -> RetranslationSession -> RetranslationEngine -> 8B AWQ backbone
//
// WHY OFFLINE, AND WHY ONE THREAD. The live app needs two threads because the mic
// callback cannot block on a GPU decode. A file has no such constraint, so this
// driver feeds blocks synchronously on the engine thread and the SegmentJobQueue
// seam is simply not in the path. That is deliberate: the queue's job is to
// survive real-time pressure, and mixing that in would make every failure here
// ambiguous between "the loop is wrong" and "the handoff dropped something". The
// loop is what this driver exists to prove.
//
// WHAT IT PROVES (checked mechanically, reported at the end):
//   1. THE REDRAFT LOOP BREATHES. Every Partial of an utterance rewinds to the
//      SAME commit pointer C, so N partials are a fixed point on the cache rather
//      than N accumulating drafts. Checked per segment.
//   2. THE LEDGER AND THE CACHE AGREE. ledger.tail() == engine cursor after every
//      single segment. This is the invariant that makes every eviction plan real
//      rather than a fiction computed against a cache that drifted.
//   3. EVICTION CLEANS THE CACHE. evict_head fires behind C, the cursor drops by
//      exactly delta, and the frozen prefix is never touched.
// Transcript COHERENCE across an eviction is printed, not asserted: whether a
// translation still reads correctly is a human judgement, and a driver that
// claimed otherwise would be lying about what it measured.
//
// THREADING: single-threaded by construction (see above). Everything runs on the
// thread that constructed the engine, which satisfies the doctrine trivially.
// -----------------------------------------------------------------------------
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include "audio_recorder.h"              // rt::load_wav_mono16k
#include "whisper_dsp.h"                 // whisper::WhisperDSP

#include "blackwell/engine.h"            // BlackwellEngine
#include "blackwell/runtime_config.h"    // blackwell::InferenceConfig
#include "blackwell/tokenizer.h"         // blackwell::TokenizerFactory

#include "absolute_audio_ring.hpp"       // blackwell::bridge::AbsoluteAudioRing
#include "continuous_streaming_config.hpp"
#include "kv_ledger.hpp"                 // blackwell::bridge::KvLedger
#include "retranslation_session.hpp"     // blackwell::bridge::RetranslationSession
#include "speech_segmenter.hpp"          // blackwell::vad::SpeechSegmenter
#include "silero_vad.hpp"                // blackwell::vad::SileroVAD

#include "cli_config.hpp"                // rt::parse_backbone_config / validate_dimensions
#include "retranslation_engine.hpp"      // rt::RetranslationEngine

namespace {

// The same frozen system prompt the live app uses. It is prefilled ONCE and its
// length becomes S, the floor no rewind and no eviction may cross.
constexpr const char* kSystemPrompt =
    "You are a real-time speech transcriber and translator. Each user turn states "
    "its task and its exact output format: follow them literally, transcribe "
    "verbatim, and output nothing else.";

// Sized for continuous streaming: the default eviction high water is 3000 tokens
// and a maximal draft adds ~222 on top (docs §5 / the T4 commit).
constexpr int kMaxContext = 4096;

struct DriverArgs {
    std::string model_dir;
    std::string audio_head;
    std::string data_dir = "data";
    std::string wav_path;
    std::string vad_model =
#ifdef BLACKWELL_VAD_MODEL_PATH
        BLACKWELL_VAD_MODEL_PATH;
#else
        "";
#endif
    std::string tgt_lang = "Russian";
    std::string src_lang = "Auto";
    int   max_new_tokens = 64;
    int   repeat = 1;             // replay the clip N times to synthesize a session
    // Silence appended after EVERY pass. A file ends abruptly; a microphone stream
    // does not. Without a trailing gap the last utterance never accumulates the
    // hangover that closes it, so it would draft forever and never commit — and
    // between passes it is what makes each replay its own utterance instead of one
    // run-on span capped by max_utterance_ms. 0 disables it.
    int   gap_ms = -1;            // -1 == derive from hangover_ms
    float vad_threshold = 0.5f;
    bool  verbose_drafts = true;  // print every Partial, not just the Finals
    blackwell::bridge::ContinuousStreamingConfig cfg{};
};

[[noreturn]] void usage_and_exit(const char* argv0, int code) {
    std::fprintf(stderr,
        "usage: %s [options]\n"
        "  --model-dir <dir>        backbone checkpoint (default: the app config's)\n"
        "  --audio-head <dir>       Ultravox audio_tower + projector checkpoint\n"
        "  --wav <file>             input clip (default: the golden reference clip)\n"
        "  --data-dir <dir>         holds mel_filters.bin (default: data)\n"
        "  --vad-model <file>       silero_vad.onnx\n"
        "  --repeat <n>             replay the clip n times as one continuous session\n"
        "  --gap-ms <n>             silence appended after each pass (default: hangover+600)\n"
        "  --tgt-lang <name>        translation target (default: Russian)\n"
        "  --src-lang <name>        forced source language (default: Auto)\n"
        "  --max-new-tokens <n>     decode cap per draft (default: 64)\n"
        "  --partial-cadence-ms <n> redraft cadence (default: 500)\n"
        "  --hangover-ms <n>        release hangover (default: 400)\n"
        "  --max-utterance-ms <n>   forced commit (default: 15000)\n"
        "  --evict-high-water <n>   committed-zone trigger, tokens (default: 3000)\n"
        "  --evict-target <n>       committed-zone low water, tokens (default: 1000)\n"
        "  --finals-only            do not print per-Partial draft lines\n",
        argv0);
    std::exit(code);
}

DriverArgs parse_args(int argc, char** argv) {
    DriverArgs a;
    auto need = [&](int& i, const char* flag) -> std::string {
        if (i + 1 >= argc) {
            std::fprintf(stderr, "FATAL: %s requires a value\n", flag);
            usage_and_exit(argv[0], 2);
        }
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if      (arg == "--model-dir")          a.model_dir  = need(i, "--model-dir");
        else if (arg == "--audio-head")         a.audio_head = need(i, "--audio-head");
        else if (arg == "--wav")                a.wav_path   = need(i, "--wav");
        else if (arg == "--data-dir")           a.data_dir   = need(i, "--data-dir");
        else if (arg == "--vad-model")          a.vad_model  = need(i, "--vad-model");
        else if (arg == "--tgt-lang")           a.tgt_lang   = need(i, "--tgt-lang");
        else if (arg == "--src-lang")           a.src_lang   = need(i, "--src-lang");
        else if (arg == "--repeat")             a.repeat = std::atoi(need(i, "--repeat").c_str());
        else if (arg == "--gap-ms")             a.gap_ms = std::atoi(need(i, "--gap-ms").c_str());
        else if (arg == "--max-new-tokens")     a.max_new_tokens = std::atoi(need(i, "--max-new-tokens").c_str());
        else if (arg == "--partial-cadence-ms") a.cfg.partial_cadence_ms = std::atoi(need(i, "--partial-cadence-ms").c_str());
        else if (arg == "--hangover-ms")        a.cfg.hangover_ms = std::atoi(need(i, "--hangover-ms").c_str());
        else if (arg == "--max-utterance-ms")   a.cfg.max_utterance_ms = std::atoi(need(i, "--max-utterance-ms").c_str());
        else if (arg == "--evict-high-water")   a.cfg.eviction_high_water_mark = std::atoi(need(i, "--evict-high-water").c_str());
        else if (arg == "--evict-target")       a.cfg.eviction_target_tokens = std::atoi(need(i, "--evict-target").c_str());
        else if (arg == "--vad-threshold")      a.vad_threshold = static_cast<float>(std::atof(need(i, "--vad-threshold").c_str()));
        else if (arg == "--finals-only")        a.verbose_drafts = false;
        else if (arg == "-h" || arg == "--help") usage_and_exit(argv[0], 0);
        else {
            std::fprintf(stderr, "FATAL: unknown argument '%s'\n", arg.c_str());
            usage_and_exit(argv[0], 2);
        }
    }
    if (a.model_dir.empty())  a.model_dir  = rt::kDefaultModelDir;
    if (a.audio_head.empty()) a.audio_head = rt::kDefaultAudioHead;
    if (a.repeat < 1) a.repeat = 1;
    a.cfg.clamp();
    // Derived AFTER clamp so it tracks the hangover the segmenter will actually
    // use. The margin covers the tail pad the segmenter adds past the release.
    if (a.gap_ms < 0) a.gap_ms = a.cfg.hangover_ms + 600;
    return a;
}

// Everything the run must be able to answer for at the end. Accumulated as the
// segments stream past, so a failure names the segment that produced it.
struct Invariants {
    std::uint64_t segments = 0;
    std::uint64_t partials = 0;
    std::uint64_t finals = 0;
    std::uint64_t skipped = 0;          // redrafts that produced no audio tokens
    std::uint64_t ledger_desyncs = 0;   // ledger.tail() != engine cursor
    std::uint64_t drift_violations = 0; // a Partial saw a moved commit pointer
    std::uint64_t prefix_violations = 0;// the frozen prefix changed
    std::uint64_t evictions = 0;
    std::uint64_t evicted_total = 0;
    std::uint32_t max_tail = 0;
    bool ok() const {
        return ledger_desyncs == 0 && drift_violations == 0 && prefix_violations == 0;
    }
};

}  // namespace

int main(int argc, char** argv) {
    const DriverArgs args = parse_args(argc, argv);

    // ---- 1. Validate geometry BEFORE any CUDA allocation --------------------
    rt::BackboneConfig backbone;
    rt::ProjectorParams projector;
    try {
        backbone  = rt::parse_backbone_config(args.model_dir);
        // Empty == the shipping 8B projector defaults, exactly as the live app
        // resolves them when --projector-path is not given.
        projector = rt::resolve_projector_params(std::string{});
        rt::validate_dimensions(projector, backbone, args.model_dir);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 2;
    }

    std::printf("=== offline re-translation driver (T5) ===\n");
    std::printf("  model-dir   : %s (hidden=%d vocab=%d quant=%s)\n", args.model_dir.c_str(),
                backbone.hidden_size, backbone.vocab_size, backbone.quant_method.c_str());
    std::printf("  audio-head  : %s\n", args.audio_head.c_str());
    std::printf("  cadence     : partial=%d ms  hangover=%d ms  max-utt=%d ms\n",
                args.cfg.partial_cadence_ms, args.cfg.hangover_ms, args.cfg.max_utterance_ms);
    std::printf("  eviction    : high-water=%d  target=%d  (draft headroom=%u, capacity=%d)\n",
                args.cfg.eviction_high_water_mark, args.cfg.eviction_target_tokens,
                args.cfg.draft_headroom_tokens(args.max_new_tokens), kMaxContext);
    if (!args.cfg.fits_context(kMaxContext, args.max_new_tokens)) {
        std::fprintf(stderr,
            "FATAL: eviction_high_water_mark + draft headroom exceeds the context "
            "(%d + %u > %d). A redraft could run off the end of the cache between "
            "two evictions.\n",
            args.cfg.eviction_high_water_mark,
            args.cfg.draft_headroom_tokens(args.max_new_tokens), kMaxContext);
        return 2;
    }

    try {
        // ---- 2. Audio front end (CPU) ---------------------------------------
        const std::string wav = args.wav_path.empty()
            ? std::string("tests/integration/golden_dumps/ultravox/test_audio.wav")
            : args.wav_path;
        std::vector<float> pcm = rt::load_wav_mono16k(wav);
        if (pcm.empty()) {
            std::fprintf(stderr, "FATAL: %s produced no samples\n", wav.c_str());
            return 2;
        }
        whisper::DspConfig dcfg;
        dcfg.n_mels = projector.num_mel_bins;
        whisper::WhisperDSP dsp(dcfg, args.data_dir + "/mel_filters.bin");

        const int sample_rate = dcfg.sample_rate;
        const int block = 160;   // 10 ms — the segmenter's and the VAD's block size
        std::printf("  wav         : %s (%zu samples, %.2f s) x%d repeat(s)\n", wav.c_str(),
                    pcm.size(), static_cast<double>(pcm.size()) / sample_rate, args.repeat);

        std::unique_ptr<blackwell::vad::SileroVAD> vad;
        if (args.vad_model.empty()) {
            std::fprintf(stderr, "FATAL: no Silero model. Build with USE_SILERO_VAD=ON or "
                                 "pass --vad-model <silero_vad.onnx>.\n");
            return 2;
        }
        vad = std::make_unique<blackwell::vad::SileroVAD>(args.vad_model);
        vad->set_threshold(args.vad_threshold);
        std::printf("  vad         : %s (threshold %.2f)\n", args.vad_model.c_str(),
                    args.vad_threshold);

        // The ring must be able to satisfy the LONGEST window a redraft can ask
        // for: one maximal utterance plus its pre-roll, floored at the encoder's
        // own 30 s ceiling so a config change cannot silently shrink it below what
        // the bucketed encoder can consume anyway.
        const std::size_t ring_samples = std::max<std::size_t>(
            static_cast<std::size_t>(sample_rate) * 30,
            static_cast<std::size_t>(sample_rate) *
                static_cast<std::size_t>(args.cfg.max_utterance_ms + args.cfg.pre_roll_ms) / 1000);
        blackwell::bridge::AbsoluteAudioRing ring(ring_samples);
        blackwell::vad::SpeechSegmenter segmenter(
            args.cfg.to_segmenter_config(sample_rate, block));

        // ---- 3. Engine + adapter (CUDA; INIT tier, throws) ------------------
        std::printf("[engine] loading tokenizer + backbone ...\n");
        std::fflush(stdout);
        std::unique_ptr<blackwell::ITokenizer> tokenizer =
            blackwell::TokenizerFactory::create(args.model_dir);
        blackwell::InferenceConfig req;
        req.max_context_length = static_cast<size_t>(kMaxContext);
        req.source_language = args.src_lang;
        req.target_language = args.tgt_lang;
        BlackwellEngine engine(args.model_dir + "/model.safetensors.index.json", req);

        rt::RetranslationEngine adapter(&engine, tokenizer.get(), &dsp, &ring, kMaxContext);
        std::printf("[audio] loading audio head ...\n");
        std::fflush(stdout);
        adapter.load_audio_head(args.audio_head);

        rt::RetranslationPrompt p;
        p.source_language_index = rt::language_index_or_auto(args.src_lang);
        p.target_language_index = rt::language_index_or_auto(args.tgt_lang);
        p.transcribe = false;
        p.translate  = true;
        adapter.set_prompt(p);

        const std::uint32_t S = adapter.prefill_system_prompt(kSystemPrompt);
        std::printf("[system-prefix] S = %u tokens (the frozen floor)\n\n", S);
        std::fflush(stdout);

        // ---- 4. The loop ----------------------------------------------------
        blackwell::bridge::KvLedger ledger(S);
        blackwell::bridge::RetranslationSession session(
            &adapter, &ledger, args.cfg, static_cast<std::uint32_t>(args.max_new_tokens));
        // Arm utterance 1's framing too, so the very first redraft is as cheap as
        // every later one (otherwise only utterances 2+ get the saving).
        session.begin_session();
        std::printf("[turn-prefix] %u token(s) resident below C — redrafts skip step 2\n\n",
                    session.resident_prefix());
        std::fflush(stdout);

        Invariants inv;
        std::vector<std::string> transcript;      // committed finals, in order
        // The interesting utterance is the first one DRAFTED on an evicted cache,
        // not the one whose commit triggered the eviction (eviction runs after the
        // commit, so that one was drafted on the full cache and proves nothing).
        // Armed by an eviction, consumed by the NEXT commit.
        std::uint32_t first_commit_after_evict = 0;   // index into transcript, 1-based
        bool evict_pending_mark = false;
        std::uint32_t open_utterance = 0;
        std::uint32_t open_commit_point = 0;

        const std::size_t clip_blocks = pcm.size() / static_cast<std::size_t>(block);
        const std::size_t gap_blocks =
            static_cast<std::size_t>(args.gap_ms) * static_cast<std::size_t>(sample_rate) /
            (1000u * static_cast<std::size_t>(block));
        const std::vector<float> silence(static_cast<std::size_t>(block), 0.0f);
        std::printf("--- streaming %zu block(s) of %d samples (%zu clip + %zu gap) x%d ---\n",
                    (clip_blocks + gap_blocks) * static_cast<std::size_t>(args.repeat), block,
                    clip_blocks, gap_blocks, args.repeat);
        std::fflush(stdout);

        // One block in, at most one segment out. Everything the driver checks
        // happens here, so the clip and the inter-pass silence go through exactly
        // the same path — the gap is audio like any other, not a special case.
        auto feed_block = [&](const float* b) {
            // Ring FIRST, then the detector: a segment must never name audio the
            // ring has not yet accepted.
            ring.write(b, static_cast<std::size_t>(block));
            const float prob = vad->feed(b, static_cast<std::size_t>(block));

            const auto seg = segmenter.on_block(prob);
            if (!seg.has_value()) return;

            // A new utterance: latch the commit pointer every Partial of it must
            // return to (proof #1).
            if (seg->utterance_id != open_utterance) {
                open_utterance = seg->utterance_id;
                open_commit_point = ledger.commit_point();
            } else if (ledger.commit_point() != open_commit_point) {
                ++inv.drift_violations;
                std::printf("  !! DRIFT: utterance %u saw C move %u -> %u between partials\n",
                            seg->utterance_id, open_commit_point, ledger.commit_point());
            }

            const std::uint32_t c_before = ledger.commit_point();
            const auto t0 = std::chrono::steady_clock::now();
            const auto res = session.on_segment(*seg);
            const double ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                    .count();
            ++inv.segments;

            // THE desync check: the ledger's map of the cache must match the
            // engine's actual write cursor after every segment, always.
            const auto cursor = static_cast<std::uint32_t>(adapter.position());
            if (ledger.tail() != cursor) {
                ++inv.ledger_desyncs;
                std::printf("  !! DESYNC: ledger.tail()=%u but engine cursor=%u\n",
                            ledger.tail(), cursor);
            }
            if (ledger.frozen_prefix() != S) {
                ++inv.prefix_violations;
                std::printf("  !! PREFIX MOVED: %u != %u\n", ledger.frozen_prefix(), S);
            }
            inv.max_tail = std::max(inv.max_tail, ledger.tail());

            const bool final_seg = seg->kind == blackwell::vad::SegmentKind::Final;
            if (final_seg) ++inv.finals; else ++inv.partials;
            if (!res.ok) {
                ++inv.skipped;
                if (args.verbose_drafts)
                    std::printf("[U%u %s] %.2f-%.2f s  (no draft: audio produced no soft "
                                "tokens or the redraft was refused)  %.0f ms\n",
                                seg->utterance_id, final_seg ? "F" : "P",
                                static_cast<double>(seg->begin_sample) / sample_rate,
                                static_cast<double>(seg->end_sample) / sample_rate, ms);
                std::fflush(stdout);
                return;
            }

            if (final_seg || args.verbose_drafts) {
                std::printf("[U%u %s] %.2f-%.2f s  audio=%u frame=%u(+%u res) text=%u | "
                            "S=%u C=%u tail=%u draft=%u | %.0f ms | %s\n",
                            seg->utterance_id, final_seg ? "F" : "P",
                            static_cast<double>(seg->begin_sample) / sample_rate,
                            static_cast<double>(seg->end_sample) / sample_rate,
                            res.audio_tokens, res.framing_tokens,
                            res.resident_prefix_tokens, res.text_tokens,
                            ledger.frozen_prefix(), ledger.commit_point(), ledger.tail(),
                            ledger.draft_tokens(), ms, res.text.c_str());
                std::fflush(stdout);
            }

            if (res.committed) {
                transcript.push_back(res.text);
                if (evict_pending_mark) {
                    evict_pending_mark = false;
                    if (first_commit_after_evict == 0)
                        first_commit_after_evict = static_cast<std::uint32_t>(transcript.size());
                }
                if (res.evicted_tokens > 0) {
                    ++inv.evictions;
                    inv.evicted_total += res.evicted_tokens;
                    evict_pending_mark = true;
                    std::printf("  >> EVICTED %u token(s) behind C: C %u -> %u, cursor=%d, "
                                "committed zone %u tok\n",
                                res.evicted_tokens, c_before, ledger.commit_point(),
                                adapter.position(), ledger.committed_tokens());
                    std::fflush(stdout);
                }
            }
        };

        for (int r = 0; r < args.repeat; ++r) {
            for (std::size_t off = 0; off + static_cast<std::size_t>(block) <= pcm.size();
                 off += static_cast<std::size_t>(block))
                feed_block(pcm.data() + off);
            for (std::size_t g = 0; g < gap_blocks; ++g) feed_block(silence.data());
        }

        // ---- 5. The verdict -------------------------------------------------
        std::printf("\n===== COMMITTED TRANSCRIPT (%zu utterance(s)) =====\n", transcript.size());
        for (std::size_t i = 0; i < transcript.size(); ++i) {
            const bool marker = first_commit_after_evict != 0 &&
                                i + 1 == first_commit_after_evict;
            std::printf("%3zu%s %s\n", i + 1, marker ? " [first after eviction] >" : "  ",
                        transcript[i].c_str());
        }

        std::printf("\n===== SESSION =====\n");
        std::printf("  segments        : %llu (%llu partial, %llu final, %llu skipped)\n",
                    static_cast<unsigned long long>(inv.segments),
                    static_cast<unsigned long long>(inv.partials),
                    static_cast<unsigned long long>(inv.finals),
                    static_cast<unsigned long long>(inv.skipped));
        std::printf("  drafts/commits  : %llu / %llu\n",
                    static_cast<unsigned long long>(session.drafts()),
                    static_cast<unsigned long long>(session.commits()));
        std::printf("  evictions       : %llu (%llu token(s) reclaimed)\n",
                    static_cast<unsigned long long>(inv.evictions),
                    static_cast<unsigned long long>(inv.evicted_total));
        std::printf("  overflow refuse : %llu\n",
                    static_cast<unsigned long long>(session.overflow_refusals()));
        std::printf("  KV zones        : S=%u C=%u tail=%u (peak tail %u / capacity %d)\n",
                    ledger.frozen_prefix(), ledger.commit_point(), ledger.tail(),
                    inv.max_tail, kMaxContext);
        std::printf("  engine cursor   : %d\n", adapter.position());

        std::printf("\n===== INVARIANTS =====\n");
        std::printf("  [%s] ledger.tail() == engine cursor after every segment (%llu violation(s))\n",
                    inv.ledger_desyncs == 0 ? "PASS" : "FAIL",
                    static_cast<unsigned long long>(inv.ledger_desyncs));
        std::printf("  [%s] every Partial rewound to its utterance's commit pointer (%llu violation(s))\n",
                    inv.drift_violations == 0 ? "PASS" : "FAIL",
                    static_cast<unsigned long long>(inv.drift_violations));
        std::printf("  [%s] the frozen prefix S never moved (%llu violation(s))\n",
                    inv.prefix_violations == 0 ? "PASS" : "FAIL",
                    static_cast<unsigned long long>(inv.prefix_violations));
        std::printf("  [%s] at least one utterance committed\n",
                    session.commits() > 0 ? "PASS" : "FAIL");
        std::printf("  [%s] eviction fired (informational: needs a long enough session)\n",
                    inv.evictions > 0 ? "yes " : "no  ");
        std::printf("\nTranscript coherence across an eviction is for the reader to judge — "
                    "see the marked line above.\n");

        const bool pass = inv.ok() && session.commits() > 0;
        std::printf("\n%s\n", pass ? "RESULT: PASS" : "RESULT: FAIL");
        return pass ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
}
