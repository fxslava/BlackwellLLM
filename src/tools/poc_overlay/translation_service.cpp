#include "translation_service.h"

// windows.h's min/max macros would shred the std::min/std::max calls inside
// the paging substrate headers included below.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>  // OutputDebugStringW

#include <filesystem>
#include <string_view>
#include <utility>
#include <vector>

#include "blackwell_llm_adapter.h"  // playground::BlackwellLLMAdapter
#include "config.h"                 // ToUtf8 / FromUtf8
#include "orchestrator.h"           // agent::orch::{AgentOrchestrator, Message, ...}
#include "tool_parser.h"            // agent::orch::ToolInvocation

// ---------------------------------------------------------------------------
// AOT prompt-cache warm start (system_prompt.bkv).
//
// The engine now exports its prefix-cache substrate (BlackwellEngine::
// has_prefix_cache() / prefix_cache() / prefill_driver(), composed in
// engine.cpp over the PagedKVManager's own SequenceManager), so warm-started
// .bkv pages are the very pages the paged-flash attention kernels read.
// BLACKWELL_ENGINE_HAS_PREFIX_CACHE is defined by this target's CMakeLists;
// the compile-time gate remains so the file still builds against an older
// engine checkout. At runtime the substrate exists only for Paged-mode dense
// uniform full-attention models -- has_prefix_cache() is the authority.
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

TranslationService::TranslationService(Settings settings)
    : settings_(std::move(settings)),
      previewPrompt_(BuildPreviewPrompt(settings_.targetLang)) {
    // F-2 insurance from the assessment: a model that hallucinates a
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
    Shutdown();
}

void TranslationService::Shutdown() {
    stop_.store(true, std::memory_order_relaxed);
    currentGen_.fetch_add(1, std::memory_order_relaxed);  // abort in-flight decode
    {
        std::lock_guard<std::mutex> lock(mutex_);
    }  // pairs the flag with the cv (no wakeup may fall between check and wait)
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
}

void TranslationService::SetPreviewSink(PreviewSink sink) {
    std::lock_guard<std::mutex> lock(mutex_);
    previewSink_ = std::move(sink);
}

void TranslationService::DeliverToSink(std::uint64_t gen, const std::wstring& text,
                                       bool done) {
    PreviewSink sink;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        sink = previewSink_;  // copy under the lock; call outside it
    }
    if (sink) sink(gen, text, done);
}

void TranslationService::RequestPreview(std::wstring segment,
                                        std::wstring inferenceContext) {
    if (stop_.load(std::memory_order_relaxed) || state() == State::Error ||
        segment.empty()) {
        return;
    }
    Job job;
    job.segment = std::move(segment);
    job.context = std::move(inferenceContext);
    job.gen = currentGen_.fetch_add(1, std::memory_order_relaxed) + 1;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pendingPreview_ = std::move(job);  // latest wins; older slot content dies
    }
    cv_.notify_all();
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
                return stop_.load(std::memory_order_relaxed) || pendingPreview_;
            });
            if (stop_.load(std::memory_order_relaxed)) break;
            job = std::move(*pendingPreview_);
            pendingPreview_.reset();
        }

        if (state() == State::Error) continue;

        // Superseded while queued: dead on arrival; skip the prefill instead of
        // letting the stream callback cancel it mid-decode.
        if (job.gen != currentGen_.load(std::memory_order_relaxed)) continue;

        state_.store(State::Generating, std::memory_order_relaxed);
        const std::optional<std::wstring> result = RunTranslation(job);
        state_.store(State::Ready, std::memory_order_relaxed);

        // ALWAYS close the request out (empty text = failed/dropped) so the
        // consumer's state machine can leave its loading state -- but only if
        // this job is still the current one; a superseded job's slot in the
        // consumer was already taken over by its successor.
        if (job.gen == currentGen_.load(std::memory_order_relaxed)) {
            DeliverToSink(job.gen, result.value_or(std::wstring()), /*done=*/true);
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
        // Paged KV mode: required by the prefix-cache substrate (radix-tree
        // prefix reuse + tiered demotion). The tier budget comes straight from
        // the user's settings; the engine validates it (a non-empty spill path
        // is required whenever the disk tier is on).
        blackwell::RuntimeOverrides overrides;
        overrides.kv_vram_cache_pages = settings_.vramCacheBlocks;
        overrides.kv_ram_slots = settings_.ramTierBlocks;
        overrides.kv_disk_slots =
            settings_.diskSpillEnabled ? settings_.diskSpillBlocks : 0;
        if (settings_.diskSpillEnabled) {
            overrides.kv_spill_path = ToUtf8(settings_.spillFilePath);
            // The tier backend opens (doesn't create) the directory; make sure
            // %LOCALAPPDATA%\Blackwell (or wherever the user pointed) exists.
            std::error_code ec;
            std::filesystem::create_directories(
                std::filesystem::path(settings_.spillFilePath).parent_path(), ec);
            if (ec) {
                Log(L"warning: could not create the spill directory (" +
                    settings_.spillFilePath + L"); the engine will report the error");
            }
        }
        adapter_ = std::make_unique<playground::BlackwellLLMAdapter>(
            ToUtf8(settings_.modelDir), settings_.maxSeqLen,
            /*use_paged_attention=*/true,
            /*num_gpu_layers=*/static_cast<size_t>(-1), overrides);
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
    // Runtime capability gate: Paged-mode dense uniform full-attention models
    // only. Hybrid SSM / gated-attention checkpoints have no prefix cache and
    // must not be treated as an error -- they just prefill normally.
    if (!adapter_->engine().has_prefix_cache()) {
        Log(L"prompt cache warm start skipped: the loaded model has no prefix-"
            L"cache substrate (hybrid SSM / gated attention / non-paged mode)");
        return;
    }
    // Deploy-side Text-to-Cache: restore the AOT-compiled prompt library
    // (system_prompt.bkv and friends) with LoadPolicy::ColdRam -- every branch
    // sits in pinned RAM from startup and page-faults into VRAM on its first
    // acquire, so warm start costs no VRAM until a prompt is actually used.
    try {
        auto& pc = adapter_->engine().prefix_cache();
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
    Log(L"prompt cache warm start SKIPPED: built without "
        L"BLACKWELL_ENGINE_HAS_PREFIX_CACHE; the system prompt will be "
        L"prefilled once per cold start instead");
#endif
}

std::optional<std::wstring> TranslationService::RunTranslation(const Job& job) {
    using namespace agent::orch;

    playground::BlackwellLLMAdapter::Params params;
    params.temperature = settings_.temperature;
    params.top_p = settings_.topP;
    params.max_new_tokens = settings_.previewMaxNewTokens;
    adapter_->set_params(params);

    // The stream callback is the cancellation AND streaming plane in one:
    //   * stale generation (a newer request / an interrupt) -> cooperative abort;
    //   * service shutdown                                  -> cooperative abort;
    //   * "</finish>" fully streamed                        -> hard stop at the
    //     protocol boundary (the adapter's checkpoint-level stop strings are
    //     fixed; this is the supported way to cut decode at our tag);
    //   * push the growing partial translation to the sink live.
    std::string acc;
    std::wstring lastPosted;
    adapter_->set_stream_callback([&, this](const std::string& delta) -> bool {
        if (stop_.load(std::memory_order_relaxed)) return false;
        if (job.gen != currentGen_.load(std::memory_order_relaxed)) return false;
        acc += delta;
        StreamPreviewPartial(acc, job.gen, lastPosted);
        return acc.find("</finish>") == std::string::npos;
    });
    // The callback captures locals of THIS frame; never let it outlive them.
    struct CallbackGuard {
        playground::BlackwellLLMAdapter& adapter;
        ~CallbackGuard() { adapter.set_stream_callback(nullptr); }
    } guard{*adapter_};

    OrchestratorConfig cfg;
    cfg.max_iterations = 1;  // the strict 1-step preview budget
    cfg.system_prompt = previewPrompt_;

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

    // Cap exit (F-2 gate): never surface protocol markup. Nudged pure prose is
    // still a usable best effort for a preview.
    if (!r.finished && !r.answer.empty() && r.answer.find('<') == std::string::npos) {
        return FromUtf8(r.answer);
    }
    Log(L"preview run did not finish cleanly; dropped");
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
    if (text.empty() || text == lastPosted) return;
    lastPosted = std::move(text);
    DeliverToSink(gen, lastPosted, /*done=*/false);
}
