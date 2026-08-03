// =============================================================================
// Two-session context isolation (Tier 2, GPU) — the engine's NATIVE CoW
// branching, as the voice assistant uses it.
//
// The assistant runs two jobs with two different system prompts: turn speech
// into text, then answer it. Sharing one linear sequence made each poison the
// other — the transcriber inherited the assistant persona and answered the audio
// instead of writing it down, and the chat context filled up with the
// transcriber's "[Speech] ... | [Translation] ..." framing. RealEngineControl
// now forks a second sequence (paged KV) so the two contexts are physically
// distinct: chat on seq 0 (persistent), transcription on seq 1 (ephemeral).
//
// What is asserted, in the order it matters:
//   1. CAPABILITY. Branching is available only under the paged KV cache; a
//      require_branching engine must report supports_cow_branching and enough
//      branch capacity, and enable_isolated_sessions() must take it.
//   2. INDEPENDENT PREFIXES. The two system prompts are prefilled to DIFFERENT
//      lengths on their own sequences, and neither position moves the other's.
//   3. NO KV LEAK — the decisive one. Decoding is deterministic under greedy
//      sampling, so the chat session's first sampled token for a fixed prompt
//      must be bit-identical before and after the audio session is loaded with
//      unrelated tokens. If the two shared KV, the extra context would shift the
//      logits and the token would change.
//   4. EPHEMERALITY. A finalized transcription turn returns seq 1 exactly to its
//      own frozen prefix in EVERY context mode (including BoundedHistory, where
//      the chat session does retain history) — a misheard transcript must never
//      become the next utterance's context.
//   5. The chat session survives all of it: its position is untouched by
//      everything the audio session does.
//
// External requirements (SKIPs when absent):
//   BLACKWELL_AWQ_INDEX  backbone index (default
//                        F:/AI/llama-3.1-8B-Instruct-AWQ-INT4/model.safetensors.index.json)
// No audio head and no golden dumps: isolation is a KV property and is proven
// with text turns, which keeps this test fast enough to run on every change.
// =============================================================================
#include <gtest/gtest.h>

// Speech-stack deps (blackwell_bridge + whisper_dsp) are wired by the ROOT
// CMakeLists after audio_sandbox is added; without them this TU is empty.
#ifdef BLACKWELL_HAVE_SPEECH_STACK

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "blackwell/engine.h"
#include "blackwell/runtime_config.h"  // InferenceConfig / RuntimeOverrides
#include "blackwell/tokenizer.h"       // blackwell::TokenizerFactory
#include "real_engine_control.hpp"     // rt::RealEngineControl (white-box base)

namespace {

std::string env_or(const char* n, const std::string& f) {
    const char* v = std::getenv(n);
    return (v && *v) ? std::string(v) : f;
}
std::string awq_index_path() {
    return env_or("BLACKWELL_AWQ_INDEX",
                  "F:/AI/llama-3.1-8B-Instruct-AWQ-INT4/model.safetensors.index.json");
}
std::string model_dir_of(const std::string& index) {
    const auto cut = index.find_last_of("/\\");
    return cut == std::string::npos ? std::string(".") : index.substr(0, cut);
}
bool file_exists(const std::string& p) { return std::ifstream(p).good(); }

constexpr int kMaxCtx = 1024;

// The two prompts the product actually ships, and the point of the whole
// exercise: they say opposite things about what to do with speech, which is why
// they cannot share a sequence.
constexpr const char* kPersonaPrompt =
    "You are a warm, concise voice assistant. Answer the user's question "
    "directly in one or two sentences.";
constexpr const char* kAudioTaskPrompt =
    "You are a speech transcription engine. Write down exactly what was said "
    "and output nothing else.";

// White-box control: re-exposes the session primitives (protected in production,
// because nothing above the engine thread may touch them) and adds the token
// drivers this test brackets. Same single engine-owning thread as production —
// the test thread IS the engine thread.
class IsolationControl : public rt::RealEngineControl {
public:
    using rt::RealEngineControl::RealEngineControl;
    using rt::RealEngineControl::activate;
    using rt::RealEngineControl::audio_;
    using rt::RealEngineControl::chat_;

    int position() const { return pos_; }

    // Prefill `text` as a user turn on the ACTIVE session and return the token
    // the last forward sampled — i.e. the first token of the reply. Greedy, so
    // the value is a deterministic function of that session's KV, which is
    // exactly what makes it usable as a leak detector.
    int first_reply_token(const std::string& text) {
        std::vector<int> turn =
            tok_->encode_chat_message(blackwell::ChatMessage{"user", text});
        const std::vector<int> cue = tok_->encode_generation_prompt();
        turn.insert(turn.end(), cue.begin(), cue.end());
        int next = -1;
        for (const int id : turn) {
            EXPECT_LT(pos_, max_context_);
            EXPECT_EQ(engine_->forward_status(id, pos_, 0.0f, 1.0f, active_seq_, &next),
                      blackwell::EngineStatus::Success);
            ++pos_;
        }
        return next;
    }

    // Greedily extend the active session by n tokens (the "turn" whose residue
    // the ephemerality assertion then looks for).
    void decode_n(int first_token, int n) {
        int next = first_token;
        for (int i = 0; i < n && pos_ < max_context_; ++i) {
            if (tok_->is_stop(next)) break;
            ASSERT_EQ(engine_->forward_status(next, pos_, 0.0f, 1.0f, active_seq_, &next),
                      blackwell::EngineStatus::Success);
            ++pos_;
        }
    }
};

}  // namespace

TEST(SessionIsolation, TranscriptionAndChatDoNotShareKv) {
    const std::string index = awq_index_path();
    if (!file_exists(index))
        GTEST_SKIP() << "AWQ backbone absent: " << index << " (set BLACKWELL_AWQ_INDEX)";

    // ---- 1. CAPABILITY: branching exists only under the paged KV cache -------
    blackwell::InferenceConfig req;
    req.max_context_length = kMaxCtx;
    req.require_branching = true;              // -> KVCacheMode::Paged
    blackwell::RuntimeOverrides overrides;
    overrides.paged_branch_factor = 2;         // exactly the two sequences we run

    std::cout << "[isolation] constructing 8B AWQ engine (paged/branching) from "
              << index << " ...\n";
    BlackwellEngine engine(index, req, overrides);
    ASSERT_TRUE(engine.get_capabilities().supports_cow_branching)
        << "require_branching resolved to a cache that cannot fork";
    ASSERT_GE(engine.branch_capacity(), 2);

    std::unique_ptr<blackwell::ITokenizer> tok =
        blackwell::TokenizerFactory::create(model_dir_of(index));
    IsolationControl control(&engine, tok.get(), kMaxCtx);

    ASSERT_TRUE(control.enable_isolated_sessions())
        << "fork(0 -> 1) refused on a branching-capable engine";
    ASSERT_TRUE(control.isolated());

    // ---- 2. INDEPENDENT PREFIXES --------------------------------------------
    // prefill_system_prompt lays BOTH: the persona on seq 0, and the audio task
    // (read from the control) on seq 1.
    control.set_audio_task_prompt(kAudioTaskPrompt);
    const uint32_t n_persona = control.prefill_system_prompt(kPersonaPrompt);
    ASSERT_GT(n_persona, 0u);

    const int chat_base  = control.chat_position();
    const int audio_base = control.audio_position();
    EXPECT_EQ(chat_base, static_cast<int>(n_persona));
    EXPECT_GT(audio_base, 0);
    // Different prompts, different lengths — proof each prefix landed on its own
    // sequence rather than one being appended after the other.
    EXPECT_NE(chat_base, audio_base);
    std::cout << "[isolation] prefixes: chat(seq 0)=" << chat_base
              << " tok, audio(seq 1)=" << audio_base << " tok\n";

    // Laying a prefix is IDEMPOTENT: a persona rebuild re-runs setup for both
    // sequences, and a second audio prefix stacked on top of the first would
    // leave seq 1 decoding under two contradictory system prompts.
    ASSERT_EQ(control.rebuild_system_prompt(kPersonaPrompt), n_persona);
    EXPECT_EQ(control.chat_position(), chat_base);
    EXPECT_EQ(control.audio_position(), audio_base) << "the audio prefix was laid twice";
    ASSERT_EQ(control.rebuild_audio_task_prompt(kAudioTaskPrompt),
              static_cast<uint32_t>(audio_base));
    EXPECT_EQ(control.audio_position(), audio_base);
    EXPECT_EQ(control.chat_position(), chat_base) << "an audio prefix rebuild moved the chat";

    // ---- 3. NO KV LEAK -------------------------------------------------------
    // Baseline: the chat session's first sampled token for a fixed question,
    // with the audio session holding nothing but its own prefix.
    control.activate(control.chat_);
    const int chat_pos_before = control.position();
    const int baseline_token = control.first_reply_token("What is the capital of France?");
    ASSERT_GE(baseline_token, 0);
    // Roll that probe back off the chat session so the second run starts from an
    // identical chat KV — the ONLY difference between the two runs must be what
    // the audio session holds.
    control.kv_cache_rollback({chat_pos_before, 0}, "isolation probe rollback");
    ASSERT_EQ(control.position(), chat_pos_before);

    // Now load the audio session with a long, unrelated turn. Under a shared
    // sequence these tokens would sit directly beneath the chat probe.
    control.activate(control.audio_);
    const int audio_first = control.first_reply_token(
        "Zanzibar zeppelin quartz. Transcribe the preceding words verbatim.");
    control.decode_n(audio_first, 24);
    const int audio_loaded = control.audio_position();
    EXPECT_GT(audio_loaded, audio_base) << "the audio session did not actually grow";

    // The chat session must be untouched by all of that...
    EXPECT_EQ(control.chat_position(), chat_pos_before);

    // ...and must sample the IDENTICAL token, because it cannot see any of it.
    control.activate(control.chat_);
    const int probed_token = control.first_reply_token("What is the capital of France?");
    EXPECT_EQ(probed_token, baseline_token)
        << "chat logits shifted after loading the transcription session -- the two "
           "sequences are sharing KV";
    std::cout << "[isolation] chat first token stable at id " << baseline_token
              << " with " << (audio_loaded - audio_base)
              << " unrelated token(s) resident on seq 1\n";
    control.kv_cache_rollback({chat_pos_before, 0}, "isolation probe rollback");

    // ---- 4. EPHEMERALITY, even in BoundedHistory -----------------------------
    // The chat session retains turns in this mode; the transcription session
    // must not, or a misheard word becomes the next utterance's prior.
    control.set_context_mode(rt::RealEngineControl::ContextMode::BoundedHistory);
    control.set_history_budget_tokens(512);

    control.activate(control.audio_);
    const int t2 = control.first_reply_token("Transcribe: the quick brown fox.");
    control.decode_n(t2, 16);
    EXPECT_GT(control.audio_position(), audio_base);
    control.finalize_turn("[Speech] the quick brown fox", /*completed=*/true);
    EXPECT_EQ(control.audio_position(), audio_base)
        << "a finalized transcription turn left KV residue on seq 1";

    // The chat session, by contrast, DOES retain a completed turn.
    control.activate(control.chat_);
    const int t3 = control.first_reply_token("Say hello.");
    control.decode_n(t3, 8);
    control.finalize_turn("Hello there.", /*completed=*/true);
    EXPECT_GT(control.chat_position(), chat_base)
        << "BoundedHistory dropped a completed chat turn";

    // ---- 5. and the two never converged ------------------------------------
    EXPECT_EQ(control.audio_position(), audio_base);
    std::cout << "[isolation] final: chat(seq 0)=" << control.chat_position()
              << " tok (history retained), audio(seq 1)=" << control.audio_position()
              << " tok (back at its prefix)\n";
}

#endif  // BLACKWELL_HAVE_SPEECH_STACK
