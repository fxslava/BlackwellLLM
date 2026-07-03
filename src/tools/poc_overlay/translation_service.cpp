#include "translation_service.h"

#include <windows.h>  // OutputDebugStringW

#include <string_view>
#include <utility>
#include <vector>

#include "blackwell_llm_adapter.h"  // playground::BlackwellLLMAdapter
#include "config.h"                 // ToUtf8 / FromUtf8
#include "orchestrator.h"           // agent::orch::{AgentOrchestrator, Message, ...}
#include "tool_parser.h"            // agent::orch::ToolInvocation

// ---------------------------------------------------------------------------
// AOT prompt-cache warm start (system_prompt.bkv) -- integration status.
//
// The paging substrate is complete and tested standalone (src/paging/:
// PrefixCacheManager, TieredMemoryPager, AOTCacheWarmer, .bkv serializer), but
// BlackwellEngine does not yet EXPORT an accessor to the PrefixCacheManager
// that owns its live paged KV pools, and the production IPrefillDriver binding
// (tokenizer + chunked prefill sweep) has not landed. Until that seam exists,
// warm-starting from a foreign PrefixCacheManager would build KV pages the
// engine's attention kernels never read -- worse than useless.
//
// So the warm start compiles only under BLACKWELL_ENGINE_HAS_PREFIX_CACHE
// (define it when the engine grows `paging::PrefixCacheManager& prefix_cache()`),
// and today degrades to a logged no-op: the system prompt is deliberately tiny
// (~90 tokens) and gets prefilled once per cold start through the adapter's
// own KV-reuse path instead.
// ---------------------------------------------------------------------------
#if defined(BLACKWELL_ENGINE_HAS_PREFIX_CACHE)
#include "../../paging/aot_cache_warmer.h"
#endif

namespace {

void Log(const std::wstring& message) {
    OutputDebugStringW((L"[translation_service] " + message + L"\n").c_str());
}

// The strict 1-step preview protocol (docs/REACT_1STEP_ASSESSMENT.md §2.2):
// finish-only, no tool manifest, one anchoring example. Kept ASCII-only so the
// source file needs no /utf-8 gymnastics.
std::string BuildPreviewPrompt(const std::string& lang) {
    return "You translate text into " + lang +
           ".\n"
           "Reply with exactly one tag and nothing else:\n"
           "<finish>TRANSLATION</finish>\n"
           "\n"
           "Rules:\n"
           "- Translate the user's message into " + lang + ".\n"
           "- Keep numbers, names, code identifiers, URLs and emoji unchanged.\n"
           "- Match the source's punctuation and casing style.\n"
           "- If the text is already in " + lang + ", return it unchanged.\n"
           "\n"
           "Example:\n"
           "User: Bonjour le monde\n"
           "Assistant: <finish>Hello world</finish>\n";
}

// Commit protocol (§2.3): same skeleton plus ONE optional glossary round-trip.
// The act-then-finish orchestrator upgrade lets a well-behaved model put the
// tool call and the <finish> in a single completion; a lookup that genuinely
// needs its observation costs the second (and last) iteration.
std::string BuildCommitPrompt(const std::string& lang) {
    return "You are a translation agent. Translate the user's message into " + lang +
           ".\n"
           "\n"
           "You may check ONE domain term first:\n"
           "<tool_call name=\"glossary_lookup\"><arg name=\"term\">TERM</arg></tool_call>\n"
           "The result arrives as an OBSERVATION message.\n"
           "\n"
           "Then reply with exactly one tag and nothing else:\n"
           "<finish>TRANSLATION</finish>\n"
           "\n"
           "Rules:\n"
           "- At most one tool call, and only if a term's established translation matters.\n"
           "- If no lookup is needed, output <finish>TRANSLATION</finish> immediately.\n"
           "- Keep numbers, names, code identifiers, URLs and emoji unchanged.\n"
           "\n"
           "Example:\n"
           "User: Bonjour le monde\n"
           "Assistant: <finish>Hello world</finish>\n";
}

// Chop a trailing INCOMPLETE UTF-8 sequence so a mid-codepoint streaming
// boundary never flashes U+FFFD in the overlay. (The dropped bytes reappear
// with the next delta.)
void TrimIncompleteUtf8(std::string& s) {
    size_t i = s.size();
    size_t continuation = 0;
    while (i > 0 && (static_cast<unsigned char>(s[i - 1]) & 0xC0) == 0x80 &&
           continuation < 3) {
        --i;
        ++continuation;
    }
    if (i == 0) return;
    const auto lead = static_cast<unsigned char>(s[i - 1]);
    size_t expected = 1;
    if (lead >= 0xF0) expected = 4;
    else if (lead >= 0xE0) expected = 3;
    else if (lead >= 0xC0) expected = 2;
    if (expected > 1 && continuation + 1 < expected) s.resize(i - 1);
}

}  // namespace

TranslationService::TranslationService(Settings settings, PreviewSink previewSink)
    : settings_(std::move(settings)),
      previewSink_(std::move(previewSink)),
      previewPrompt_(BuildPreviewPrompt(settings_.targetLang)),
      commitPrompt_(BuildCommitPrompt(settings_.targetLang)) {
    // Prototype glossary: no store yet, but the tool is REAL -- registered,
    // dispatchable, and the first production validation of a ReAct tool
    // round-trip. Failure mode is text, per the dispatcher contract.
    tools_.register_tool(
        "glossary_lookup",
        "Look up the pinned translation for a domain term. Args: term.",
        [](const agent::orch::ToolInvocation& call) -> std::string {
            const auto it = call.args.find("term");
            if (it == call.args.end() || it->second.empty()) {
                return "ERROR: glossary_lookup needs <arg name=\"term\">TERM</arg>.";
            }
            return "No glossary entry for '" + it->second +
                   "'. Use your best judgment and keep it consistent with the "
                   "context.";
        });
    // Decoy from the assessment (F-2 insurance): a model that hallucinates a
    // translate_text tool gets steered back onto the <finish> protocol instead
    // of an opaque "tool not found".
    tools_.register_tool(
        "translate_text",
        "Do not call this. Output the translation as <finish>TRANSLATION</finish>.",
        [](const agent::orch::ToolInvocation&) -> std::string {
            return "Output the translation as <finish>TRANSLATION</finish> now.";
        });

    worker_ = std::thread(&TranslationService::ThreadMain, this);
}

TranslationService::~TranslationService() {
    stop_.store(true, std::memory_order_relaxed);
    currentGen_.fetch_add(1, std::memory_order_relaxed);  // abort in-flight decode
    {
        std::lock_guard<std::mutex> lock(mutex_);
    }  // pairs the flag with the cv (no wakeup may fall between check and wait)
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
    // Unblock a TranslateBlocking() caller that is still waiting: the worker
    // has exited, so its promise will never be fulfilled otherwise.
    std::lock_guard<std::mutex> lock(mutex_);
    if (pendingCommit_ && pendingCommit_->reply) {
        pendingCommit_->reply->set_value(std::nullopt);
    }
}

void TranslationService::RequestPreview(std::wstring segment,
                                        std::wstring inferenceContext) {
    if (state() == State::Error || segment.empty()) return;
    Job job;
    job.kind = JobKind::Preview;
    job.segment = std::move(segment);
    job.context = std::move(inferenceContext);
    job.gen = currentGen_.fetch_add(1, std::memory_order_relaxed) + 1;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pendingPreview_ = std::move(job);  // latest wins; older slot content dies
    }
    cv_.notify_all();
}

std::optional<std::wstring> TranslationService::TranslateBlocking(
    const std::wstring& segment, const std::wstring& inferenceContext,
    std::chrono::milliseconds timeout) {
    if (state() == State::Error || segment.empty()) return std::nullopt;

    Job job;
    job.kind = JobKind::Commit;
    job.segment = segment;
    job.context = inferenceContext;
    // Bumping the generation aborts any in-flight preview decode -- the commit
    // is about to replace that text anyway, and it frees the engine sooner.
    job.gen = currentGen_.fetch_add(1, std::memory_order_relaxed) + 1;
    job.reply = std::make_shared<std::promise<std::optional<std::wstring>>>();
    auto future = job.reply->get_future();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (pendingCommit_) return std::nullopt;  // re-entrant commit: refuse
        pendingPreview_.reset();                  // superseded by the commit
        pendingCommit_ = std::move(job);
    }
    cv_.notify_all();

    // Bounded wait on the STA thread (see header for why this is acceptable in
    // the prototype). On timeout the worker keeps decoding; its late result is
    // published into an abandoned shared state and simply evaporates.
    if (future.wait_for(timeout) != std::future_status::ready) {
        Log(L"commit translation timed out; injecting source unchanged");
        return std::nullopt;
    }
    return future.get();
}

void TranslationService::CancelPending() {
    currentGen_.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(mutex_);
    pendingPreview_.reset();
}

void TranslationService::ThreadMain() {
    LoadEngine();

    while (true) {
        Job job;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] {
                return stop_.load(std::memory_order_relaxed) || pendingCommit_ ||
                       pendingPreview_;
            });
            if (stop_.load(std::memory_order_relaxed)) break;
            if (pendingCommit_) {
                job = std::move(*pendingCommit_);
                pendingCommit_.reset();
            } else {
                job = std::move(*pendingPreview_);
                pendingPreview_.reset();
            }
        }

        if (state() == State::Error) {
            if (job.reply) job.reply->set_value(std::nullopt);
            continue;
        }

        // A preview that was superseded while queued is dead on arrival; skip
        // the prefill instead of letting the stream callback cancel it later.
        if (job.kind == JobKind::Preview &&
            job.gen != currentGen_.load(std::memory_order_relaxed)) {
            continue;
        }

        state_.store(State::Generating, std::memory_order_relaxed);
        std::optional<std::wstring> result = RunTranslation(job);
        state_.store(State::Ready, std::memory_order_relaxed);

        if (job.kind == JobKind::Commit) {
            job.reply->set_value(std::move(result));
        } else if (result && previewSink_ &&
                   job.gen == currentGen_.load(std::memory_order_relaxed)) {
            previewSink_(job.gen, *result, /*done=*/true);
        }
    }
}

void TranslationService::LoadEngine() {
    if (settings_.modelDir.empty()) {
        Log(L"no model directory configured (tray -> Settings); translation disabled");
        state_.store(State::Error, std::memory_order_relaxed);
        return;
    }
    try {
        Log(L"loading model from " + settings_.modelDir + L" ...");
        // Continuous KV mode for the dense-model prototype. The .bkv/prefix-
        // cache path will require use_paged_attention=true once the engine
        // exports its paging substrate -- flip it together with the warm start.
        adapter_ = std::make_unique<playground::BlackwellLLMAdapter>(
            ToUtf8(settings_.modelDir), settings_.maxSeqLen,
            /*use_paged_attention=*/false);
        WarmStartPromptCache();
        state_.store(State::Ready, std::memory_order_relaxed);
        Log(L"model ready");
    } catch (const std::exception& e) {
        Log(L"model load FAILED: " + FromUtf8(e.what()));
        adapter_.reset();
        state_.store(State::Error, std::memory_order_relaxed);
    } catch (...) {
        Log(L"model load FAILED: unknown error");
        adapter_.reset();
        state_.store(State::Error, std::memory_order_relaxed);
    }
}

void TranslationService::WarmStartPromptCache() {
    if (settings_.promptCacheDir.empty()) return;
#if defined(BLACKWELL_ENGINE_HAS_PREFIX_CACHE)
    // Deploy-side Text-to-Cache: restore the AOT-compiled prompt library
    // (system_prompt.bkv and friends) with LoadPolicy::ColdRam -- every branch
    // sits in pinned RAM from startup and page-faults into VRAM on its first
    // acquire, so warm start costs no VRAM until a prompt is actually used.
    try {
        auto& pc = adapter_->engine().prefix_cache();  // the engine seam
        const int tokens = blackwell::paging::AOTCacheWarmer::warm_start(
            pc, ToUtf8(settings_.promptCacheDir),
            blackwell::paging::PrefixCacheManager::LoadPolicy::ColdRam);
        Log(L"prompt cache warm start: " + std::to_wstring(tokens) +
            L" tokens now zero-prefill-servable from pinned RAM");
    } catch (const std::exception& e) {
        // Non-fatal by design: a missing/foreign .bkv directory must never
        // block translation -- the prompt falls back to a one-time prefill.
        Log(L"prompt cache warm start failed (falling back to cold prefill): " +
            FromUtf8(e.what()));
    }
#else
    Log(L"prompt cache warm start SKIPPED: this engine build does not export "
        L"its PrefixCacheManager yet (BLACKWELL_ENGINE_HAS_PREFIX_CACHE unset); "
        L"the system prompt will be prefilled once per cold start instead");
#endif
}

std::optional<std::wstring> TranslationService::RunTranslation(const Job& job) {
    using namespace agent::orch;
    const bool isPreview = (job.kind == JobKind::Preview);

    playground::BlackwellLLMAdapter::Params params;
    params.temperature = settings_.temperature;
    params.top_p = settings_.topP;
    params.max_new_tokens =
        isPreview ? settings_.previewMaxNewTokens : settings_.commitMaxNewTokens;
    adapter_->set_params(params);

    // The stream callback is the cancellation AND streaming plane in one:
    //   * stale generation (a newer request arrived)  -> cooperative abort;
    //   * service shutdown                            -> cooperative abort;
    //   * "</finish>" fully streamed                  -> hard stop at the
    //     protocol boundary (the adapter's checkpoint-level stop strings are
    //     fixed; this is the supported way to cut decode at our tag);
    //   * preview only: push the partial translation to the sink live.
    std::string acc;
    std::wstring lastPosted;
    adapter_->set_stream_callback([&, this](const std::string& delta) -> bool {
        if (stop_.load(std::memory_order_relaxed)) return false;
        if (isPreview && job.gen != currentGen_.load(std::memory_order_relaxed)) {
            return false;
        }
        acc += delta;
        if (isPreview) StreamPreviewPartial(acc, job.gen, lastPosted);
        return acc.find("</finish>") == std::string::npos;
    });
    // The callback captures locals of THIS frame; never let it outlive them.
    struct CallbackGuard {
        playground::BlackwellLLMAdapter& adapter;
        ~CallbackGuard() { adapter.set_stream_callback(nullptr); }
    } guard{*adapter_};

    OrchestratorConfig cfg;
    cfg.max_iterations = isPreview ? 1 : 2;
    cfg.system_prompt = isPreview ? previewPrompt_ : commitPrompt_;

    // Fresh orchestrator per request (transcripts are per-run state); the KV
    // cache lives one level down in the adapter and survives across runs.
    AgentOrchestrator orch(*adapter_, tools_, cfg);
    if (!job.context.empty()) {
        orch.seed_history({{Message::Role::User,
                            "Text committed earlier in this field (already "
                            "translated; keep the new translation consistent "
                            "with it):\n" +
                                ToUtf8(job.context)}});
    }

    const RunResult r = orch.run(ToUtf8(job.segment));

    if (r.finished && !r.answer.empty()) return FromUtf8(r.answer);

    // Cap exit (F-2 gate): never surface protocol markup. A nudged pure-prose
    // preview is still a usable best effort; a commit must be clean or nothing,
    // because its output is injected into the user's document.
    if (isPreview && !r.finished && !r.answer.empty() &&
        r.answer.find('<') == std::string::npos) {
        return FromUtf8(r.answer);
    }
    Log(isPreview ? L"preview run did not finish cleanly; dropped"
                  : L"commit run did not finish cleanly; injecting source unchanged");
    return std::nullopt;
}

void TranslationService::StreamPreviewPartial(const std::string& acc,
                                              std::uint64_t gen,
                                              std::wstring& lastPosted) {
    static constexpr std::string_view kOpen = "<finish>";
    static constexpr std::string_view kClose = "</finish>";

    const size_t open = acc.find(kOpen);
    if (open == std::string::npos) return;  // tag not reached yet: nothing to show

    std::string body = acc.substr(open + kOpen.size());
    if (const size_t close = body.find(kClose); close != std::string::npos) {
        body.erase(close);
    } else {
        // Trim a partially-streamed "</finis" tail so it never flashes onscreen.
        if (const size_t lt = body.rfind('<'); lt != std::string::npos) {
            const std::string_view tail = std::string_view(body).substr(lt);
            if (tail.size() < kClose.size() && kClose.substr(0, tail.size()) == tail) {
                body.erase(lt);
            }
        }
        TrimIncompleteUtf8(body);
    }

    std::wstring text = FromUtf8(body);
    if (text.empty() || text == lastPosted || !previewSink_) return;
    lastPosted = std::move(text);
    previewSink_(gen, lastPosted, /*done=*/false);
}
