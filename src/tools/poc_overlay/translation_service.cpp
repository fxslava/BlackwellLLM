#include "translation_service.h"

// windows.h's min/max macros would shred the std::min/std::max calls inside
// the paging substrate headers included below.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>  // OutputDebugStringW

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <utility>

#include "blackwell_llm_adapter.h"  // playground::BlackwellLLMAdapter
#include "config.h"                 // ToUtf8 / FromUtf8
#include "live_translation_tracker.h"

// ---------------------------------------------------------------------------
// JIT prompt cache (per-model .bkv, generated on the end user's machine).
//
// The user picks the checkpoint at runtime (Qwen / ALMA / Llama / ...), and KV
// page geometry + token ids are strictly model-specific -- so pre-compiled
// .bkv files cannot ship with the app. Instead the service self-bootstraps on
// first run: it builds the WarmupSpec in C++ (the chat-template-rendered
// system-prompt prefix), drives the engine's own EnginePrefillCoordinator
// (the production IPrefillDriver, tokenizer already bound by the adapter)
// through AOTCacheWarmer::compile(), and serializes the result to a directory
// keyed by the substrate's model hash -- the same hash warm_start() validates
// manifests against, so a cache can never be replayed onto the wrong model.
// Subsequent runs find the manifest and go straight to the ColdRam warm start.
//
// BLACKWELL_ENGINE_HAS_PREFIX_CACHE is defined by this target's CMakeLists;
// the compile-time gate remains so the file still builds against an older
// engine checkout. At runtime the substrate exists only for Paged-mode dense
// uniform full-attention models -- has_prefix_cache() is the authority.
// ---------------------------------------------------------------------------
#if defined(BLACKWELL_ENGINE_HAS_PREFIX_CACHE)
#include "engine_prefill_coordinator.h"  // EnginePrefillCoordinator (IPrefillDriver)
#include "paging/aot_cache_warmer.h"     // AOTCacheWarmer (compile + warm_start)
#include "paging/warmup_spec.h"          // WarmupSpec / WarmupNode (built in C++)
#endif

namespace {

void Log(const std::wstring& message) {
    OutputDebugStringW((L"[translation_service] " + message + L"\n").c_str());
}

// %LOCALAPPDATA%\Blackwell\Cache -- the default per-model prompt-cache root.
std::wstring DefaultPromptCacheRoot() {
    wchar_t local[MAX_PATH] = {};
    if (GetEnvironmentVariableW(L"LOCALAPPDATA", local, MAX_PATH) == 0) {
        return L"BlackwellCache";  // last resort: relative to the working dir
    }
    return std::wstring(local) + L"\\Blackwell\\Cache";
}

// 16-digit lowercase hex of the substrate's model hash: the cache directory
// name AND the value the .bkv manifest is validated against.
std::wstring HashDirName(unsigned long long hash) {
    wchar_t buf[17] = {};
    swprintf(buf, 17, L"%016llx", hash);
    return buf;
}

// The strict 1-step preview protocol (docs/REACT_1STEP_ASSESSMENT.md §2.2):
// finish-only, no tool manifest, one anchoring example. Kept ASCII-only so the
// source file needs no /utf-8 gymnastics. Shared verbatim by the JIT cache
// compiler (StableServingPrefix) and LiveTranslationTracker (BuildTokens) --
// both must address the identical radix-tree pages.
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

// Steady-clock milliseconds -- the lifecycle's activity stamp unit.
std::int64_t NowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

}  // namespace

TranslationService::TranslationService(Settings settings) : settings_(std::move(settings)) {
    // One system prompt per configured direction. The set is fixed for the
    // session; the switcher only changes WHICH one is active.
    for (const std::string& lang : settings_.targetLanguages) {
        previewPrompts_.push_back(BuildPreviewPrompt(lang));
    }
    if (previewPrompts_.empty()) {
        previewPrompts_.push_back(BuildPreviewPrompt("English"));
    }
    int active = settings_.activeLanguage;
    if (active < 0 || active >= static_cast<int>(previewPrompts_.size())) {
        active = 0;
    }
    activeLanguage_.store(active, std::memory_order_relaxed);
    lastActivityMs_.store(NowMs(), std::memory_order_relaxed);

    if (settings_.deferLoad) {
        // Start-inactive mode: the backend exists (this object, its queues,
        // its prompts) but VRAM stays untouched until EnsureLoaded().
        state_.store(State::Idle, std::memory_order_relaxed);
    } else {
        loadStarted_ = true;
        worker_ = std::thread(&TranslationService::ThreadMain, this);
    }
}

void TranslationService::EnsureLoaded() {
    // UI-thread only (like the ctor's calling thread), so no lock around
    // loadStarted_/worker_ is needed -- Shutdown() also runs there.
    if (loadStarted_ || stop_.load(std::memory_order_relaxed)) {
        return;
    }
    loadStarted_ = true;
    state_.store(State::Loading, std::memory_order_relaxed);
    worker_ = std::thread(&TranslationService::ThreadMain, this);
}

void TranslationService::SetActiveLanguage(int index) {
    activeLanguage_.store(index, std::memory_order_relaxed);  // remembered for the handoff
    std::lock_guard<std::mutex> lock(mutex_);
    if (tracker_) tracker_->SetActiveLanguage(index);  // clamps internally
}

void TranslationService::SetDeveloperMode(bool enabled) {
    developerMode_.store(enabled, std::memory_order_relaxed);  // remembered for the handoff
    std::lock_guard<std::mutex> lock(mutex_);
    if (tracker_) tracker_->SetCollectProbabilities(enabled);
}

void TranslationService::UpdateLanguagePairs(std::vector<std::string> targetLanguages,
                                             int activeLanguage) {
    // Rebuild the per-direction system prompts OUTSIDE the lock (pure string
    // work), then swap everything in atomically with respect to the load
    // thread (tracker construction / JIT compile snapshot the same state).
    std::vector<std::string> prompts;
    prompts.reserve(targetLanguages.size());
    for (const std::string& lang : targetLanguages) {
        prompts.push_back(BuildPreviewPrompt(lang));
    }
    if (prompts.empty()) {
        targetLanguages = {"English"};
        prompts.push_back(BuildPreviewPrompt("English"));
    }
    if (activeLanguage < 0 || activeLanguage >= static_cast<int>(prompts.size())) {
        activeLanguage = 0;
    }
    activeLanguage_.store(activeLanguage, std::memory_order_relaxed);

    std::lock_guard<std::mutex> lock(mutex_);
    settings_.targetLanguages = std::move(targetLanguages);
    previewPrompts_ = prompts;
    if (tracker_) {
        // Live hot-swap: the very next request addresses the new set. A brand-
        // new direction has no compiled .bkv branch yet -- it prefills cold
        // once and the radix tree commits it, exactly like any other prompt.
        tracker_->UpdatePrompts(std::move(prompts), activeLanguage);
    }
    // No tracker yet (deferred / still loading): the load thread constructs it
    // from previewPrompts_ under this same mutex, so it picks the new set up.
}

void TranslationService::PrecacheLanguage(const std::string& targetLanguage,
                                          std::function<void(bool, int)> done) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        int index = -1;
        for (size_t i = 0; i < settings_.targetLanguages.size(); ++i) {
            if (settings_.targetLanguages[i] == targetLanguage) {
                index = static_cast<int>(i);
                break;
            }
        }
        if (index >= 0 && tracker_ &&
            state_.load(std::memory_order_relaxed) == State::Ready) {
            // The compile drives the engine's prefill coordinator, so it MUST
            // run on the tracker's engine-owning worker. `this` outlives the
            // task: Shutdown joins that worker before this service unwinds.
            tracker_->PostEngineTask([this, index, done = std::move(done)] {
                bool ok = false;
                int tokens = 0;
                try {
                    const std::wstring dir = ResolvePromptCacheDir();
                    ok = !dir.empty() && CompilePromptCacheBranch(index, dir, &tokens);
                } catch (const std::exception& e) {
                    Log(L"pre-cache failed: " + FromUtf8(e.what()));
                }
                if (done) done(ok, tokens);
            });
            return;
        }
    }
    // Unknown target (unsaved pair) or the engine isn't serving yet: fail
    // fast -- outside the lock -- so the UI says so instead of spinning.
    if (done) done(false, 0);
}

TranslationService::~TranslationService() {
    Shutdown();
}

void TranslationService::Shutdown() {
    stop_.store(true, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (tracker_) tracker_->Cancel();  // unblock any in-flight decode promptly
    }
    if (worker_.joinable()) worker_.join();  // finishes LoadEngine (+ tracker handoff)
    // Join the tracker's worker OUTSIDE mutex_: its thread takes mutex_ itself
    // (DeliverToSink / DeliverLifecycle / a pre-cache task's prompt snapshot),
    // so destroying it under the lock could deadlock the join against a worker
    // blocked on the very mutex we hold.
    std::unique_ptr<LiveTranslationTracker> tracker;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        tracker = std::move(tracker_);
    }
    tracker.reset();  // joins LiveTranslationTracker's own worker thread
}

void TranslationService::SetPreviewSink(PreviewSink sink) {
    std::lock_guard<std::mutex> lock(mutex_);
    previewSink_ = std::move(sink);
}

void TranslationService::SetLifecycleSink(LifecycleSink sink) {
    std::lock_guard<std::mutex> lock(mutex_);
    lifecycleSink_ = std::move(sink);
}

void TranslationService::DeliverLifecycle(LifecycleEvent event) {
    LifecycleSink sink;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        sink = lifecycleSink_;  // copy under the lock; call outside it
    }
    if (sink) sink(event);
}

void TranslationService::DeliverToSink(std::uint64_t gen, const std::wstring& text,
                                       const TokenHeatmap& tokens, bool done) {
    PreviewSink sink;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        sink = previewSink_;  // copy under the lock; call outside it
    }
    if (sink) sink(gen, text, tokens, done);
}

void TranslationService::MarkActivity() {
    lastActivityMs_.store(NowMs(), std::memory_order_relaxed);
    lifecycleStage_.store(0, std::memory_order_relaxed);  // re-arm both stages
}

void TranslationService::LifecycleTick(int kvSpillTimeoutSec, int hibernateTimeoutSec) {
    if (state_.load(std::memory_order_relaxed) != State::Ready ||
        stop_.load(std::memory_order_relaxed)) {
        return;  // nothing resident yet (Idle/Loading) or already torn down
    }
    const std::int64_t idleMs = NowMs() - lastActivityMs_.load(std::memory_order_relaxed);
    const std::int64_t idleSec = idleMs / 1000;
    const int stage = lifecycleStage_.load(std::memory_order_relaxed);

    if (stage < 2 && hibernateTimeoutSec > 0 && idleSec >= hibernateTimeoutSec) {
        lifecycleStage_.store(2, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(mutex_);
        if (tracker_) {
            Log(L"lifecycle: " + std::to_wstring(idleSec) +
                L" s idle -> stage 2 (soft hibernation)");
            tracker_->RequestHibernate();
        }
    } else if (stage < 1 && kvSpillTimeoutSec > 0 && idleSec >= kvSpillTimeoutSec) {
        lifecycleStage_.store(1, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(mutex_);
        if (tracker_) {
            Log(L"lifecycle: " + std::to_wstring(idleSec) +
                L" s idle -> stage 1 (KV disk spill)");
            tracker_->RequestKvSpill();
        }
    }
}

void TranslationService::TrackUpdate(std::wstring segment, std::wstring inferenceContext) {
    if (stop_.load(std::memory_order_relaxed) || segment.empty()) return;
    MarkActivity();
    std::lock_guard<std::mutex> lock(mutex_);
    if (!tracker_) return;  // still loading (or failed) -- nothing to warm yet
    tracker_->TrackUpdate(std::move(segment), std::move(inferenceContext));
}

void TranslationService::TriggerGeneration(std::wstring segment,
                                           std::wstring inferenceContext) {
    if (stop_.load(std::memory_order_relaxed) || segment.empty()) return;
    MarkActivity();
    std::lock_guard<std::mutex> lock(mutex_);
    if (!tracker_) return;
    const std::uint64_t gen = requestSeq_.fetch_add(1, std::memory_order_relaxed) + 1;
    tracker_->TriggerGeneration(
        std::move(segment), std::move(inferenceContext),
        [this, gen](const std::wstring& text, const TokenHeatmap& tokens, bool done) {
            DeliverToSink(gen, text, tokens, done);
        });
}

void TranslationService::Cancel() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (tracker_) tracker_->Cancel();
}

void TranslationService::ThreadMain() {
    LoadEngine();
    if (state_.load(std::memory_order_relaxed) != State::Ready) {
        return;  // failed / no model configured -- nothing left for this thread to do
    }

    // LiveTranslationTracker owns the ONLY thread that touches the engine from
    // here on. Construct it locally first (its constructor starts its own
    // thread immediately) so a shutdown race can simply let it fall out of
    // scope -- its destructor joins that thread before this function returns,
    // so `tracker_` never gets published in a half-torn-down state.
    // Prompts are snapshotted under mutex_: a Settings save during the load
    // may have hot-reloaded the pair set since the constructor ran.
    std::vector<std::string> prompts;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        prompts = previewPrompts_;
    }
    auto tracker = std::make_unique<LiveTranslationTracker>(
        *adapter_, std::move(prompts), activeLanguage_.load(std::memory_order_relaxed),
        settings_.previewMaxNewTokens, settings_.temperature, settings_.topP);
    // Wire the lifecycle observer BEFORE publishing: LifecycleTick can only
    // reach the tracker through tracker_, so no event can fire un-forwarded.
    tracker->SetLifecycleSink(
        [this](LifecycleEvent event) { DeliverLifecycle(event); });
    // Carry the Developer-Mode flag across the handoff (it may have been toggled
    // while the model was still loading).
    tracker->SetCollectProbabilities(developerMode_.load(std::memory_order_relaxed));
    std::lock_guard<std::mutex> lock(mutex_);
    if (!stop_.load(std::memory_order_relaxed)) {
        tracker_ = std::move(tracker);
    }
    // else: Shutdown() raced construction -- `tracker` destructs here (still
    // under the lock is fine; its destructor only joins its own thread, never
    // touches mutex_), and tracker_ stays null.
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
        // Weight-load progress: the arena calls this per tensor on THIS load
        // thread; the atomic percent is polled by the UI-thread readiness
        // timer, which renders "Initializing... (N%)" into the center HUD.
        overrides.load_progress = [this](size_t done, size_t total) {
            const int pct = total ? static_cast<int>((done * 100) / total) : 0;
            loadProgressPct_.store(pct, std::memory_order_relaxed);
        };
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
#if defined(BLACKWELL_ENGINE_HAS_PREFIX_CACHE)
    // Runtime capability gate: Paged-mode dense uniform full-attention models
    // only. Hybrid SSM / gated-attention checkpoints have no prefix cache and
    // must not be treated as an error -- they just prefill normally (and
    // LiveTranslationTracker falls back to the adapter's ordinary generate()
    // for them too -- see live_translation_tracker.cpp).
    if (!adapter_->engine().has_prefix_cache()) {
        Log(L"prompt cache skipped: the loaded model has no prefix-cache "
            L"substrate (hybrid SSM / gated attention / non-paged mode)");
        return;
    }
    // Everything below is best-effort by design: a cache failure of ANY kind
    // must never block translation -- the prompt falls back to a one-time
    // prefill through the ordinary decode path.
    try {
        auto& pc = adapter_->engine().prefix_cache();

        // (1) Model-specific cache directory (see ResolvePromptCacheDir).
        const std::wstring dir = ResolvePromptCacheDir();
        if (dir.empty()) {
            return;  // unreachable here (has_prefix_cache checked above)
        }
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        if (ec) {
            Log(L"prompt cache disabled: cannot create " + dir);
            return;
        }

        // (2) Conditional JIT generation. The manifest is the compile's last
        // artifact (AOTCacheWarmer writes it after every .bkv), so its
        // presence implies a complete cache; a torn earlier run left no
        // manifest and recompiles here.
        const bool have_cache = std::filesystem::exists(
            std::filesystem::path(dir) / L"manifest.json");
        if (!have_cache) {
            Log(L"first-run optimization: compiling the prompt cache for this "
                L"model into " + dir + L" ...");
            if (!CompilePromptCache(dir)) {
                return;  // nothing cacheable (reason already logged)
            }
        }

        // (3) Cold load: every compiled branch goes to pinned RAM
        // (LoadPolicy::ColdRam) and page-faults into VRAM on first acquire,
        // so the warm start costs no VRAM until a prompt is actually used.
        // A stale/corrupt directory (e.g. a page-size change across engine
        // versions) is healed by one recompile-and-retry before giving up.
        int tokens = 0;
        try {
            tokens = blackwell::paging::AOTCacheWarmer::warm_start(
                pc, ToUtf8(dir),
                blackwell::paging::PrefixCacheManager::LoadPolicy::ColdRam);
        } catch (const std::exception& e) {
            Log(L"prompt cache rejected (" + FromUtf8(e.what()) +
                L"); recompiling once ...");
            if (!CompilePromptCache(dir)) return;
            tokens = blackwell::paging::AOTCacheWarmer::warm_start(
                pc, ToUtf8(dir),
                blackwell::paging::PrefixCacheManager::LoadPolicy::ColdRam);
        }
        Log(L"prompt cache ready: " + std::to_wstring(tokens) +
            L" tokens zero-prefill-servable from pinned RAM");
    } catch (const std::exception& e) {
        Log(L"prompt cache unavailable (falling back to cold prefill): " +
            FromUtf8(e.what()));
    }
#else
    Log(L"prompt cache SKIPPED: built without BLACKWELL_ENGINE_HAS_PREFIX_CACHE; "
        L"the system prompt will be prefilled once per cold start instead");
#endif
}

#if defined(BLACKWELL_ENGINE_HAS_PREFIX_CACHE)
bool TranslationService::CompilePromptCache(const std::wstring& dir) {
    using blackwell::paging::AOTCacheWarmer;
    using blackwell::paging::WarmupNode;
    using blackwell::paging::WarmupSpec;

    // The WarmupSpec, constructed in C++ (no spec.json on disk). One ROOT PER
    // translation direction: each holds that direction's full, chat-template-
    // rendered system block (role framing, translation rules, the <finish>
    // protocol -- and, critically, the target language). We emit them as
    // sibling roots rather than a hand-built base+children tree because the
    // radix tree dedups the shared LEADING tokens automatically at commit /
    // serve time: "You translate text into " is identical across directions and
    // shares pages; the branches diverge only at the language token onward. No
    // fragile mid-sentence BPE seam to get right.
    //
    // The cacheable text per branch is the byte-exact serving PREFIX, not the
    // raw prompt: radix-tree hits happen on token ids, and the serving path
    // (LiveTranslationTracker's BuildTokens) tokenizes the RENDERED transcript.
    // An unrendered/misaligned node would compile fine and then never match.
    // Human-readable direction name for the logs (index-aligned with
    // previewPrompts_ / targetLanguages), so the multi-branch compile is legible.
    // Snapshot under mutex_: a Settings save can hot-reload the pair set at
    // any moment, and this compile runs off the UI thread (load thread on
    // first run, tracker worker for the on-demand pre-cache).
    std::vector<std::string> prompts;
    std::vector<std::string> targets;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        prompts = previewPrompts_;
        targets = settings_.targetLanguages;
    }
    const auto branchName = [&targets](size_t i) -> std::wstring {
        if (i < targets.size()) {
            return L"->" + FromUtf8(targets[i]);
        }
        return L"#" + std::to_wstring(i);
    };

    WarmupSpec spec;
    spec.name = "overlay-translator";
    spec.emit_mode = WarmupSpec::EmitMode::All;
    for (size_t i = 0; i < prompts.size(); ++i) {
        const std::string prefix = StableServingPrefix(prompts[i]);
        if (prefix.empty()) {
            Log(L"prompt cache: no stable serving prefix for direction " + branchName(i) +
                L"; skipping that branch");
            continue;
        }
        WarmupNode node;
        node.id = "system_prompt_" + std::to_string(i);
        node.text = prefix;
        Log(L"prompt cache: queuing branch " + branchName(i) + L" (" +
            std::to_wstring(prefix.size()) + L" prefix bytes)");
        spec.roots.push_back(std::move(node));
    }
    if (spec.roots.empty()) {
        Log(L"prompt cache: no stable serving prefix could be derived for this "
            L"chat template; skipping compilation");
        return false;
    }

    // The engine's own prefill coordinator IS the production IPrefillDriver;
    // the adapter bound the checkpoint's tokenizer to it at construction, so
    // compile() tokenizes and prefills with exactly the serving stack.
    AOTCacheWarmer::Options opt;
    opt.out_dir = ToUtf8(dir);
    // The serving path encodes the rendered prompt with add_special_tokens =
    // false (the template's literal special-token strings carry the framing),
    // so the compiled tokens must do the same or the very first token differs.
    opt.add_bos = false;
    AOTCacheWarmer warmer(adapter_->engine().prefix_cache(),
                          adapter_->engine().prefill_driver());
    const AOTCacheWarmer::Report rep = warmer.compile(spec, opt);

    if (rep.files == 0) {
        // Shorter than one KV page (16 tokens) -- legal, just not worth caching.
        Log(L"prompt cache: the system prompt is shorter than one KV page; "
            L"nothing was emitted");
        return false;
    }
    Log(L"first-run optimization done: " + std::to_wstring(rep.files) +
        L" branch(es), " + std::to_wstring(rep.prefilled_tokens) +
        L" tokens prefilled once and serialized to .bkv");
    return true;
}

std::string TranslationService::StableServingPrefix(const std::string& previewPrompt) const {
    using Adapter = playground::BlackwellLLMAdapter;

    // Render two otherwise-identical transcripts that diverge only in the user
    // turn; their longest common prefix is everything the chat template emits
    // before request-specific content -- template-agnostically.
    const std::string head = "[SYSTEM]\n" + previewPrompt + "\n\n[USER]\n";
    const std::string a =
        Adapter::apply_chat_template(head + "A\n\n", adapter_->chat_template());
    const std::string b =
        Adapter::apply_chat_template(head + "B\n\n", adapter_->chat_template());
    size_t n = 0;
    const size_t limit = std::min(a.size(), b.size());
    while (n < limit && a[n] == b[n]) ++n;
    std::string common = a.substr(0, n);

    // BPE seam safety: cut right after the template's end-of-turn marker. The
    // marker is a special token -- a hard segmentation boundary -- so
    // encode(prefix) is guaranteed to be a strict token-prefix of
    // encode(full prompt). Cutting anywhere later (e.g. inside "<|im_start|>
    // user\n") risks the tokenizer merging across the seam at serve time.
    const std::string marker =
        adapter_->chat_template() == Adapter::ChatTemplate::Llama3 ? "<|eot_id|>"
                                                                   : "<|im_end|>";
    const size_t end = common.rfind(marker);
    if (end == std::string::npos) return {};  // template surprise: no safe cut
    return common.substr(0, end + marker.size());
}

bool TranslationService::CompilePromptCacheBranch(int index, const std::wstring& dir,
                                                  int* prefilledTokens) {
    using blackwell::paging::AOTCacheWarmer;
    using blackwell::paging::WarmupNode;
    using blackwell::paging::WarmupSpec;
    using nlohmann::json;

    if (prefilledTokens) *prefilledTokens = 0;
    if (!adapter_->engine().has_prefix_cache()) {
        Log(L"pre-cache skipped: this model has no prefix-cache substrate");
        return false;
    }

    std::string prompt;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index < 0 || index >= static_cast<int>(previewPrompts_.size())) {
            return false;  // pair set changed under the queued task
        }
        prompt = previewPrompts_[index];
    }
    const std::string prefix = StableServingPrefix(prompt);
    if (prefix.empty()) {
        Log(L"pre-cache: no stable serving prefix for this chat template");
        return false;
    }

    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const std::filesystem::path manifestPath = std::filesystem::path(dir) / L"manifest.json";

    // The warmer rewrites manifest.json with ONLY the entries of THIS compile,
    // which would strand every other branch on the next warm start -- so hold
    // the pre-compile manifest and merge it back below.
    json oldManifest;
    if (std::ifstream in{manifestPath, std::ios::binary}) {
        try {
            oldManifest = json::parse(in);
        } catch (...) {
            oldManifest = json();  // torn/corrupt: nothing worth merging
        }
    }

    WarmupSpec spec;
    spec.name = "overlay-translator";
    spec.emit_mode = WarmupSpec::EmitMode::All;
    WarmupNode node;
    node.id = "system_prompt_" + std::to_string(index);
    node.text = prefix;
    spec.roots.push_back(std::move(node));

    AOTCacheWarmer::Options opt;
    opt.out_dir = ToUtf8(dir);
    opt.add_bos = false;  // must match the serving encode (see CompilePromptCache)
    AOTCacheWarmer warmer(adapter_->engine().prefix_cache(),
                          adapter_->engine().prefill_driver());
    // NOTE: the compile itself acquires/prefills/commits through the LIVE
    // radix tree, so the branch is zero-prefill-servable the moment this
    // returns -- the .bkv file is only the cross-run persistence on top.
    const AOTCacheWarmer::Report rep = warmer.compile(spec, opt);
    if (prefilledTokens) {
        *prefilledTokens = static_cast<int>(rep.prefilled_tokens + rep.reused_tokens);
    }

    // Merge: keep every previously-manifested branch this compile didn't just
    // re-emit. Best-effort -- the live tree already has the branch either way.
    try {
        if (oldManifest.contains("entries") && oldManifest["entries"].is_array()) {
            std::ifstream in{manifestPath, std::ios::binary};
            json merged = json::parse(in);
            in.close();
            auto& entries = merged["entries"];
            for (const auto& e : oldManifest["entries"]) {
                const std::string file = e.value("file", std::string());
                bool present = false;
                for (const auto& n : entries) {
                    if (n.value("file", std::string()) == file) {
                        present = true;
                        break;
                    }
                }
                if (!present && !file.empty() &&
                    std::filesystem::exists(std::filesystem::path(dir) / file)) {
                    entries.push_back(e);
                }
            }
            std::ofstream out{manifestPath, std::ios::binary | std::ios::trunc};
            out << merged.dump(2) << '\n';
        }
    } catch (const std::exception& e) {
        Log(L"pre-cache: manifest merge failed (branch still live): " + FromUtf8(e.what()));
    }

    if (rep.files == 0) {
        Log(L"pre-cache: branch shorter than one KV page; committed to the live "
            L"tree but nothing was serialized");
        return true;  // still a successful warm-up -- just nothing to persist
    }
    Log(L"pre-cache done: branch " + std::to_wstring(index) + L", " +
        std::to_wstring(rep.prefilled_tokens) + L" tokens prefilled, " +
        std::to_wstring(rep.reused_tokens) + L" reused");
    return true;
}

std::wstring TranslationService::ResolvePromptCacheDir() const {
    // adapter_ is stable here by construction: set once on the load thread
    // before the tracker exists, and both callers (WarmStartPromptCache on the
    // load thread, the pre-cache task on the tracker worker) run strictly
    // after that.
    if (!adapter_ || !adapter_->engine().has_prefix_cache()) {
        return {};
    }
    const std::wstring root = settings_.promptCacheRoot.empty()
                                  ? DefaultPromptCacheRoot()
                                  : settings_.promptCacheRoot;
    return root + L"\\" + HashDirName(adapter_->engine().prefix_cache().model_hash());
}
#else   // !BLACKWELL_ENGINE_HAS_PREFIX_CACHE -- keep the unconditional callers linking
bool TranslationService::CompilePromptCacheBranch(int, const std::wstring&, int* tokens) {
    if (tokens) *tokens = 0;
    return false;
}
std::wstring TranslationService::ResolvePromptCacheDir() const { return {}; }
#endif  // BLACKWELL_ENGINE_HAS_PREFIX_CACHE
