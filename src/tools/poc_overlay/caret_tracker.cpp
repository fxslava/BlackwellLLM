#include "caret_tracker.h"

#include <objbase.h>
#include <UIAutomation.h>
#include <wrl/client.h>

#include <cwctype>

#include "text_injector.h"

using Microsoft::WRL::ComPtr;

namespace {

// Extracts the caret point (screen coords) from a text range's bounding
// rectangles. The range is expected to be collapsed to the caret, so there is
// a single (x, y, w, h) rect; we take its left/top as the caret's leading edge.
bool CaretPointFromRange(IUIAutomationTextRange* range, POINT& out) {
    SAFEARRAY* rectsArray = nullptr;
    if (FAILED(range->GetBoundingRectangles(&rectsArray)) || !rectsArray) {
        return false;
    }

    bool found = false;
    double* rects = nullptr;
    if (SUCCEEDED(SafeArrayAccessData(rectsArray, reinterpret_cast<void**>(&rects)))) {
        long lower = 0;
        long upper = -1;
        SafeArrayGetLBound(rectsArray, 1, &lower);
        SafeArrayGetUBound(rectsArray, 1, &upper);
        const long count = upper - lower + 1;
        if (count >= 4) {
            // Packed as (x, y, width, height). Use the LAST rect's left/top: for
            // a collapsed caret that is the caret glyph; if a provider ever hands
            // back a multi-rect selection, the last rect sits at the active end.
            const long base = count - 4;
            out.x = static_cast<LONG>(rects[base + 0]);
            out.y = static_cast<LONG>(rects[base + 1]);
            found = true;
        }
        SafeArrayUnaccessData(rectsArray);
    }
    SafeArrayDestroy(rectsArray);
    return found;
}

// Primary path: read the document up to the caret via TextPattern, and the
// caret's screen point. Returns true only if the text was successfully read.
// When `outRange` is provided it also receives the [doc-start .. caret] range,
// so the commit path can Select() the exact text it is about to replace.
bool ReadTextPattern(IUIAutomationElement* focused, std::wstring& outText, POINT& outCaret,
                     bool& outCaretFound,
                     ComPtr<IUIAutomationTextRange>* outRange = nullptr) {
    ComPtr<IUnknown> patternUnknown;
    if (FAILED(focused->GetCurrentPattern(UIA_TextPatternId, &patternUnknown)) || !patternUnknown) {
        return false;
    }
    ComPtr<IUIAutomationTextPattern> textPattern;
    if (FAILED(patternUnknown.As(&textPattern)) || !textPattern) {
        return false;
    }

    // The current selection of a caret with no highlight is a degenerate range
    // sitting exactly at the caret -- the standard UIA "where is the caret" idiom.
    ComPtr<IUIAutomationTextRangeArray> selection;
    if (FAILED(textPattern->GetSelection(&selection)) || !selection) {
        return false;
    }
    int rangeCount = 0;
    selection->get_Length(&rangeCount);
    if (rangeCount == 0) {
        return false;
    }
    ComPtr<IUIAutomationTextRange> caret;
    if (FAILED(selection->GetElement(0, &caret)) || !caret) {
        return false;
    }

    // Collapse any selection to its active (end) endpoint so we anchor on the
    // caret regardless of whether text is selected.
    caret->MoveEndpointByRange(TextPatternRangeEndpoint_Start, caret.Get(),
                               TextPatternRangeEndpoint_End);
    outCaretFound = CaretPointFromRange(caret.Get(), outCaret);

    // Build a range spanning [start of DOCUMENT .. caret] and read its text. Using
    // the whole document (not just the current line) means multiline input is
    // captured in full -- both for display and for commit.
    ComPtr<IUIAutomationTextRange> document;
    if (FAILED(textPattern->get_DocumentRange(&document)) || !document) {
        return false;
    }
    ComPtr<IUIAutomationTextRange> upToCaret;
    if (FAILED(document->Clone(&upToCaret)) || !upToCaret) {
        return false;
    }
    upToCaret->MoveEndpointByRange(TextPatternRangeEndpoint_End, caret.Get(),
                                   TextPatternRangeEndpoint_Start);

    BSTR bstr = nullptr;
    if (FAILED(upToCaret->GetText(-1, &bstr)) || !bstr) {
        return false;
    }
    outText.assign(bstr, SysStringLen(bstr));
    SysFreeString(bstr);
    if (outRange) {
        *outRange = upToCaret;  // hand the exact range to the commit path
    }
    return true;
}

// Fetches the focused element's ValuePattern, if any (Tier 1 candidate).
ComPtr<IUIAutomationValuePattern> GetValuePattern(IUIAutomationElement* focused) {
    ComPtr<IUIAutomationValuePattern> valuePattern;
    ComPtr<IUnknown> patternUnknown;
    if (SUCCEEDED(focused->GetCurrentPattern(UIA_ValuePatternId, &patternUnknown)) &&
        patternUnknown) {
        patternUnknown.As(&valuePattern);
    }
    return valuePattern;
}

// Secondary path: whole-control value via ValuePattern (no caret offset).
bool ReadValuePattern(IUIAutomationElement* focused, std::wstring& outText) {
    ComPtr<IUnknown> patternUnknown;
    if (FAILED(focused->GetCurrentPattern(UIA_ValuePatternId, &patternUnknown)) || !patternUnknown) {
        return false;
    }
    ComPtr<IUIAutomationValuePattern> valuePattern;
    if (FAILED(patternUnknown.As(&valuePattern)) || !valuePattern) {
        return false;
    }
    BSTR value = nullptr;
    if (FAILED(valuePattern->get_CurrentValue(&value)) || !value) {
        return false;
    }
    outText.assign(value, SysStringLen(value));
    SysFreeString(value);
    return true;
}

// True if `c` ends the capture area under granularity `g`. Boundaries nest:
// newlines always break; Sentence adds . ! ?; Clause additionally , ; :.
bool IsBoundaryChar(wchar_t c, CaptureGranularity g) {
    if (c == L'\n' || c == L'\r') return true;
    if (g == CaptureGranularity::Paragraph) return false;
    if (c == L'.' || c == L'!' || c == L'?') return true;
    if (g == CaptureGranularity::Sentence) return false;
    return c == L',' || c == L';' || c == L':';
}

}  // namespace

std::wstring CaretTracker::ExtractCapture(const std::wstring& segment,
                                          CaptureGranularity granularity) {
    // Skip the trailing run of whitespace + boundary characters first: the
    // terminator the user JUST typed belongs to the clause being captured, not
    // to a (still empty) next one. "hello world," must capture "hello world,",
    // not "".
    size_t scanEnd = segment.size();
    while (scanEnd > 0 && (std::iswspace(segment[scanEnd - 1]) ||
                           IsBoundaryChar(segment[scanEnd - 1], granularity))) {
        --scanEnd;
    }
    if (scanEnd == 0) {
        return {};  // nothing but separators -> nothing worth capturing
    }

    // Walk back to the previous boundary, then left-trim the separator gap.
    size_t begin = scanEnd;
    while (begin > 0 && !IsBoundaryChar(segment[begin - 1], granularity)) {
        --begin;
    }
    while (begin < segment.size() && std::iswspace(segment[begin])) {
        ++begin;
    }
    // Capture runs to the REAL end (trailing punctuation/whitespace included),
    // so the commit-time ends_with verification is exact.
    return segment.substr(begin);
}

CaretTracker::CaretTracker(Callbacks callbacks, CaptureGranularity granularity,
                           int idleTimerMs)
    : callbacks_(std::move(callbacks)),
      granularity_(static_cast<int>(granularity)),
      idleTimerMs_(idleTimerMs > 0 ? idleTimerMs : 1) {
    thread_ = std::thread(&CaretTracker::ThreadMain, this);
}

CaretTracker::~CaretTracker() {
    stop_ = true;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        hasPending_ = true;  // wake the worker even with nothing pending
    }
    cv_.notify_all();
    if (thread_.joinable()) {
        thread_.join();
    }
}

void CaretTracker::RequestUpdate(std::wstring fallbackText, bool /*wordBoundary*/) {
    // Called from the keyboard hook -- must stay O(1) and never touch COM/UIA.
    // Overwrites the pending text with the latest state so a burst of keystrokes
    // costs a single UIA round-trip (and a single idle-timer reset).
    std::lock_guard<std::mutex> lock(mutex_);
    pendingFallback_ = std::move(fallbackText);
    hasPendingUpdate_ = true;
    hasPending_ = true;
    cv_.notify_all();
}

void CaretTracker::RequestCommit(std::wstring /*fallbackText*/) {
    // O(1) and hook-safe. The commit consumes the state machine's OWN snapshot
    // (source_raw / translation_raw); it no longer needs the hook buffer.
    std::lock_guard<std::mutex> lock(mutex_);
    pendingCommit_ = true;
    hasPending_ = true;
    cv_.notify_all();
}

void CaretTracker::RequestReset() {
    std::lock_guard<std::mutex> lock(mutex_);
    pendingReset_ = true;
    hasPending_ = true;
    cv_.notify_all();
}

void CaretTracker::OnPreviewResult(std::wstring text, bool done) {
    // Called from the TranslationService worker thread: marshal into this
    // worker's serialized event loop. Latest-wins is correct here too -- a
    // newer partial supersedes an older one, and a final overwrites partials.
    std::lock_guard<std::mutex> lock(mutex_);
    pendingTranslation_.emplace(std::move(text), done);
    hasPending_ = true;
    cv_.notify_all();
}

void CaretTracker::ThreadMain() {
    // UI Automation client calls require an STA on the calling thread. This
    // worker exists solely to own that apartment so the UIA text/caret round-trip
    // never runs on the hook (UI) thread or the overlay repaint path.
    const HRESULT hrCoInit = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);

    ComPtr<IUIAutomation> automation;
    const HRESULT hrAutomation = CoCreateInstance(CLSID_CUIAutomation, nullptr, CLSCTX_INPROC_SERVER,
                                                   IID_PPV_ARGS(&automation));

    while (true) {
        std::wstring fallback;
        bool haveUpdate = false;
        bool doCommit = false;
        bool doReset = false;
        bool timerFired = false;
        std::optional<std::pair<std::wstring, bool>> translation;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            const auto pred = [this] {
                return hasPending_ || stop_.load(std::memory_order_relaxed);
            };
            // The idle debounce IS this wait deadline: no separate timer thread,
            // no async callback -- expiry is just another event in the one
            // serialized loop, so it can never race a keystroke or a commit.
            if (idleArmed_) {
                if (!cv_.wait_until(lock, idleDeadline_, pred)) {
                    timerFired = true;
                }
            } else {
                cv_.wait(lock, pred);
            }
            if (stop_.load(std::memory_order_relaxed)) {
                break;
            }
            if (pendingReset_) {
                doReset = true;
                pendingReset_ = false;
            }
            if (pendingCommit_) {
                doCommit = true;
                pendingCommit_ = false;
            }
            if (hasPendingUpdate_) {
                haveUpdate = true;
                fallback = std::move(pendingFallback_);
                hasPendingUpdate_ = false;
            }
            translation.swap(pendingTranslation_);
            hasPending_ = false;
        }

        if (timerFired) {
            idleArmed_ = false;  // consumed; HandleKeystroke re-arms as needed
        }
        if (doReset) {
            ResetToIdle();
            continue;
        }
        // A commit consumes the snapshot and hides the overlay, so it supersedes
        // any coalesced keystroke/translation event in the same wake-up.
        if (doCommit) {
            if (SUCCEEDED(hrAutomation)) {
                HandleCommit(automation.Get());
            }
            continue;
        }
        if (haveUpdate && SUCCEEDED(hrAutomation)) {
            HandleKeystroke(automation.Get(), fallback);
        }
        if (translation) {
            HandleTranslation(translation->first, translation->second);
        }
        // A keystroke in the same wake-up already re-armed the timer; only a
        // "pure" expiry starts inference.
        if (timerFired && !haveUpdate) {
            HandleIdleExpired();
        }
    }

    automation.Reset();
    if (SUCCEEDED(hrCoInit)) {
        CoUninitialize();
    }
}

// ---------------------------------------------------------------------------
// State-machine event handlers (worker thread only)
// ---------------------------------------------------------------------------

void CaretTracker::HandleKeystroke(IUIAutomation* automation, const std::wstring& fallback) {
    // INTERRUPT: any keystroke during Translating/Ready invalidates the
    // translation -- cancel the in-flight decode and clear what was shown.
    if (phase_ == Phase::Translating || phase_ == Phase::Ready) {
        if (callbacks_.cancelGeneration) {
            callbacks_.cancelGeneration();
        }
    }
    translationPartial_.clear();
    translationRaw_.clear();

    std::wstring segment;
    bool fromUia = false;
    const bool resolved = ResolveSegment(automation, fallback, segment, fromUia);

    const auto granularity =
        static_cast<CaptureGranularity>(granularity_.load(std::memory_order_relaxed));
    sourceRaw_ = resolved ? ExtractCapture(segment, granularity) : std::wstring();

    if (sourceRaw_.empty() || !anchorValid_) {
        phase_ = Phase::Idle;  // nothing captured (or nowhere to anchor)
        idleArmed_ = false;
        Render();
        return;
    }

    phase_ = Phase::Typing;
    idleArmed_ = true;
    idleDeadline_ = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(idleTimerMs_.load(std::memory_order_relaxed));
    // Speculative background prefill: warm the engine's radix tree token-by-
    // token as the user types, so the debounce-triggered generation below has
    // (ideally) nothing left to prefill. Fire-and-forget; LiveTranslationTracker
    // owns its own thread and never blocks this call.
    if (callbacks_.trackUpdate) {
        callbacks_.trackUpdate(sourceRaw_, inferenceContext_);
    }
    Render();
}

void CaretTracker::HandleIdleExpired() {
    if (phase_ != Phase::Typing || sourceRaw_.empty()) {
        return;  // expiry raced a state change; nothing to translate
    }
    phase_ = Phase::Translating;
    translationPartial_.clear();
    if (callbacks_.triggerGeneration) {
        callbacks_.triggerGeneration(sourceRaw_, inferenceContext_);
    }
    Render();
}

void CaretTracker::HandleTranslation(const std::wstring& text, bool done) {
    if (phase_ != Phase::Translating) {
        return;  // stale delivery: the user typed (or committed) meanwhile
    }
    if (!done) {
        translationPartial_ = text;  // streamed partial
    } else if (!text.empty()) {
        translationRaw_ = text;
        translationPartial_.clear();
        phase_ = Phase::Ready;
    } else {
        // Empty final = the run failed / was dropped. Fall back to Typing
        // WITHOUT re-arming the timer -- retrying on a persistent failure
        // would spin; the next keystroke re-enters the normal flow.
        phase_ = Phase::Typing;
        translationPartial_.clear();
    }
    Render();
}

void CaretTracker::HandleCommit(IUIAutomation* automation) {
    // Surgical commit is valid ONLY from Ready: there must be a finished
    // translation on screen. In any other state Ctrl+Enter is a no-op (the
    // overlay already shows what stage we are in).
    if (phase_ != Phase::Ready || sourceRaw_.empty() || translationRaw_.empty()) {
        return;
    }

    // Flag the injection window so the keyboard hook treats the upcoming
    // synthetic input as ours (belt-and-suspenders alongside the dwExtraInfo tag).
    if (callbacks_.injectionGuard) {
        callbacks_.injectionGuard(true);
    }

    // Gather the UIA context HERE, freshly, so the interfaces stay in their
    // owning apartment and we verify against the field's CURRENT state.
    ComPtr<IUIAutomationTextRange> range;
    ComPtr<IUIAutomationValuePattern> valuePattern;
    std::wstring fullText;
    bool fromUia = false;

    ComPtr<IUIAutomationElement> focused;
    if (SUCCEEDED(automation->GetFocusedElement(&focused)) && focused) {
        POINT caretIgnored{};
        bool caretFoundIgnored = false;
        std::wstring text;
        if (ReadTextPattern(focused.Get(), text, caretIgnored, caretFoundIgnored, &range)) {
            fullText = text;  // entire document up to the caret (all lines)
            fromUia = true;
        }
        valuePattern = GetValuePattern(focused.Get());
        if (!fromUia && valuePattern) {
            BSTR value = nullptr;
            if (SUCCEEDED(valuePattern->get_CurrentValue(&value)) && value) {
                fullText.assign(value, SysStringLen(value));
                SysFreeString(value);
                fromUia = true;
            }
        }
    }

    bool injected = false;
    if (fromUia) {
        // The snapshot must still match reality. Typing would have reset the
        // state machine, but the field can change under us regardless (focus
        // swaps, IME composition, another process) -- verify, never guess.
        if (fullText.ends_with(sourceRaw_)) {
            TextInjector::Request request;
            request.source = sourceRaw_;
            request.replacement = translationRaw_;
            if (range) {
                // Narrow [doc-start .. caret] to exactly the captured source:
                // everything before it is preserved by Tier 2's range select.
                int moved = 0;
                range->MoveEndpointByUnit(
                    TextPatternRangeEndpoint_Start, TextUnit_Character,
                    static_cast<int>(fullText.size() - sourceRaw_.size()), &moved);
                request.selectionRange = range.Get();
            }
            // SetValue replaces the WHOLE control value, so only offer Tier 1
            // when the capture IS the whole value.
            request.valuePattern =
                (fullText == sourceRaw_) ? valuePattern.Get() : nullptr;
            injected = TextInjector::Replace(request) != TextInjector::Tier::None;
            if (injected) {
                // The field now holds prefix + translation; hide all of it from
                // future captures.
                committedPrefix_ =
                    fullText.substr(0, fullText.size() - sourceRaw_.size()) +
                    translationRaw_;
            }
        }
    } else {
        // UIA-opaque control (e.g. Telegram/Qt): no authoritative text to
        // verify or track a prefix against; Tier 3 selects source.size() chars
        // by keystroke and pastes over them.
        TextInjector::Request request;
        request.source = sourceRaw_;
        request.replacement = translationRaw_;
        injected = TextInjector::Replace(request) != TextInjector::Tier::None;
    }

    if (injected) {
        // The AI history keeps the ORIGINAL text (what the user actually wrote).
        if (!inferenceContext_.empty()) {
            inferenceContext_ += L' ';
        }
        inferenceContext_ += sourceRaw_;
        OutputDebugStringW(
            (L"[poc_overlay] InferenceContext: \"" + inferenceContext_ + L"\"\n").c_str());
    }

    if (callbacks_.injectionGuard) {
        callbacks_.injectionGuard(false);
    }

    // Injected or aborted-stale, this interaction is over; the next keystroke
    // starts a fresh capture.
    phase_ = Phase::Idle;
    sourceRaw_.clear();
    translationRaw_.clear();
    translationPartial_.clear();
    idleArmed_ = false;
    Render();
}

void CaretTracker::ResetToIdle() {
    // Unconditional (not gated on phase): a queued-but-not-yet-started
    // speculative TrackUpdate should also be dropped when focus/caret resets,
    // not just an active Translating decode. Cheap and idempotent either way.
    if (callbacks_.cancelGeneration) {
        callbacks_.cancelGeneration();
    }
    phase_ = Phase::Idle;
    sourceRaw_.clear();
    translationRaw_.clear();
    translationPartial_.clear();
    idleArmed_ = false;
    Render();
}

void CaretTracker::Render() const {
    if (!callbacks_.render) {
        return;
    }
    OverlaySnapshot snapshot;
    snapshot.anchor = anchor_;
    snapshot.anchorValid = anchorValid_;
    switch (phase_) {
        case Phase::Idle:
            snapshot.phase = OverlayPhase::Hidden;
            break;
        case Phase::Typing:
            snapshot.phase = OverlayPhase::Typing;
            snapshot.source = sourceRaw_;
            break;
        case Phase::Translating:
            snapshot.phase = OverlayPhase::Translating;
            snapshot.source = sourceRaw_;
            snapshot.translation = translationPartial_;
            break;
        case Phase::Ready:
            snapshot.phase = OverlayPhase::Ready;
            snapshot.translation = translationRaw_;
            break;
    }
    callbacks_.render(snapshot);
}

// ---------------------------------------------------------------------------
// Text resolution
// ---------------------------------------------------------------------------

bool CaretTracker::ResolveSegment(IUIAutomation* automation, const std::wstring& fallbackText,
                                  std::wstring& segment, bool& fromUia) {
    anchorValid_ = false;
    fromUia = false;

    ComPtr<IUIAutomationElement> focused;
    if (FAILED(automation->GetFocusedElement(&focused)) || !focused) {
        segment = fallbackText;  // no focus -> no anchor; caller idles
        return !segment.empty();
    }

    // (1) Primary: TextPattern document-up-to-caret. This also yields the caret.
    std::wstring uiaText;
    bool haveText = ReadTextPattern(focused.Get(), uiaText, anchor_, anchorValid_);

    // (1) Secondary: ValuePattern whole-control value.
    if (!haveText) {
        haveText = ReadValuePattern(focused.Get(), uiaText);
    }

    // Caret position fallback: some providers (or the ValuePattern path) give no
    // caret rect -- anchor to the focused element's top-left instead.
    if (!anchorValid_) {
        RECT rect{};
        if (SUCCEEDED(focused->get_CurrentBoundingRectangle(&rect)) &&
            (rect.right > rect.left || rect.bottom > rect.top)) {
            anchor_.x = rect.left;
            anchor_.y = rect.top;
            anchorValid_ = true;
        }
    }

    if (haveText) {
        // Track only what was typed since the last commit; the committed text
        // stays hidden (but lives on in inferenceContext_).
        segment = StripCommittedPrefix(uiaText);
        fromUia = true;
    } else {
        // (2) UIA-opaque app: the hook's ToUnicodeEx buffer is the current
        // segment as-is.
        segment = fallbackText;
    }
    return true;
}

std::wstring CaretTracker::StripCommittedPrefix(const std::wstring& fullText) {
    if (!committedPrefix_.empty() && fullText.rfind(committedPrefix_, 0) == 0) {
        return fullText.substr(committedPrefix_.size());
    }
    // The field no longer begins with what we committed (user edited into it, or
    // focus moved elsewhere) -- drop the boundary and its history.
    committedPrefix_.clear();
    inferenceContext_.clear();
    return fullText;
}
