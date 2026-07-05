#include "caret_tracker.h"

#include <objbase.h>
#include <UIAutomation.h>
#include <wrl/client.h>

#include <cwctype>
#include <vector>

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

// Reads the focused element's current UIA text SELECTION (the highlighted
// range). Returns true -- filling `outRange` + `outText` -- only when a NON-empty
// range is selected; a bare caret is a degenerate range whose GetText is empty,
// which we reject (a plain click is not a translation request).
bool ReadSelection(IUIAutomationElement* focused, ComPtr<IUIAutomationTextRange>& outRange,
                   std::wstring& outText) {
    ComPtr<IUnknown> patternUnknown;
    if (FAILED(focused->GetCurrentPattern(UIA_TextPatternId, &patternUnknown)) || !patternUnknown) {
        return false;
    }
    ComPtr<IUIAutomationTextPattern> textPattern;
    if (FAILED(patternUnknown.As(&textPattern)) || !textPattern) {
        return false;
    }
    ComPtr<IUIAutomationTextRangeArray> selection;
    if (FAILED(textPattern->GetSelection(&selection)) || !selection) {
        return false;
    }
    int rangeCount = 0;
    selection->get_Length(&rangeCount);
    if (rangeCount == 0) {
        return false;
    }
    ComPtr<IUIAutomationTextRange> range;
    if (FAILED(selection->GetElement(0, &range)) || !range) {
        return false;
    }
    BSTR bstr = nullptr;
    if (FAILED(range->GetText(-1, &bstr)) || !bstr) {
        return false;
    }
    outText.assign(bstr, SysStringLen(bstr));
    SysFreeString(bstr);
    outRange = range;
    return !outText.empty();
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
    // Smart left-trim: strip ALL leading whitespace, tabs, newlines AND
    // punctuation from the head of the capture. Without this, a leading "- ",
    // ", " or stray space the user typed before the new phrase would ride along
    // into source_raw, and the injected translation would physically stick to
    // the previous output ("Hello.- Привет" instead of "Hello. Привет"). Bounded
    // by scanEnd so a capture that is nothing but separators collapses to empty.
    while (begin < scanEnd &&
           (std::iswspace(segment[begin]) || std::iswpunct(segment[begin]))) {
        ++begin;
    }
    if (begin >= scanEnd) {
        return {};  // only leading separators/punctuation -> nothing to translate
    }
    // Capture runs to the REAL end (trailing punctuation/whitespace included),
    // so the commit-time ends_with verification is exact.
    return segment.substr(begin);
}

CaretTracker::CaretTracker(Callbacks callbacks, CaptureGranularity granularity,
                           int idleTimerMs, LanguageRouting routing)
    : callbacks_(std::move(callbacks)),
      granularity_(static_cast<int>(granularity)),
      idleTimerMs_(idleTimerMs > 0 ? idleTimerMs : 1),
      routing_(routing) {
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

void CaretTracker::RequestSelectionCheck() {
    std::lock_guard<std::mutex> lock(mutex_);
    pendingSelectionCheck_ = true;
    hasPending_ = true;
    cv_.notify_all();
}

void CaretTracker::SetActive(bool active) {
    std::lock_guard<std::mutex> lock(mutex_);
    pendingActive_ = active;  // latest wins
    hasPending_ = true;
    cv_.notify_all();
}

void CaretTracker::SetLanguageOverride(int index) {
    std::lock_guard<std::mutex> lock(mutex_);
    pendingOverride_ = index;  // latest wins; -1 clears the override (back to auto)
    hasPending_ = true;
    cv_.notify_all();
}

void CaretTracker::ShowHud(std::wstring message, bool fade) {
    std::lock_guard<std::mutex> lock(mutex_);
    pendingHud_.emplace(std::move(message), fade);  // latest wins
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
        bool doSelectionCheck = false;
        bool timerFired = false;
        std::optional<bool> setActive;
        std::optional<int> setOverride;
        std::optional<std::pair<std::wstring, bool>> hud;
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
            if (pendingSelectionCheck_) {
                doSelectionCheck = true;
                pendingSelectionCheck_ = false;
            }
            if (hasPendingUpdate_) {
                haveUpdate = true;
                fallback = std::move(pendingFallback_);
                hasPendingUpdate_ = false;
            }
            setActive.swap(pendingActive_);
            setOverride.swap(pendingOverride_);
            hud.swap(pendingHud_);
            translation.swap(pendingTranslation_);
            hasPending_ = false;
        }

        if (timerFired) {
            idleArmed_ = false;  // consumed; HandleKeystroke re-arms as needed
        }
        // Master-mode + HUD events first: they set the gate / banner the rest
        // of this wake-up's handlers respect.
        if (setActive) {
            HandleSetActive(*setActive);
        }
        // A manual direction override (dropdown / cycle) is applied before any
        // coalesced keystroke this wake-up, so the capture below uses it.
        if (setOverride) {
            HandleSetOverride(*setOverride);
        }
        if (hud) {
            HandleShowHud(hud->first, hud->second);
        }
        // Reset is NOT terminal: a fast click coalesces mouse-down (reset) with
        // mouse-up (selection check), and the check must still run after it.
        if (doReset) {
            ResetToIdle();
        }
        // A commit consumes the snapshot and hides the overlay, so it supersedes
        // any coalesced keystroke/translation event in the same wake-up.
        if (doCommit) {
            if (SUCCEEDED(hrAutomation)) {
                HandleCommit(automation.Get());
            }
            continue;
        }
        // A selection is an explicit action superseding typing/timer this wake-up.
        if (doSelectionCheck && active_ && SUCCEEDED(hrAutomation)) {
            HandleSelectionCheck(automation.Get());
            continue;
        }
        // Typing / debounce only while Translation Mode is active (so a HUD or an
        // OFF state is never clobbered by stray input).
        if (haveUpdate && active_ && SUCCEEDED(hrAutomation)) {
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
    // INTERRUPT: any keystroke during a translation phase (typing OR selection)
    // invalidates it -- cancel the in-flight decode and clear what was shown.
    if (phase_ == Phase::Translating || phase_ == Phase::Ready ||
        phase_ == Phase::SelTranslating || phase_ == Phase::SelReady) {
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
        sessionLanguage_ = -1;  // session ended; the next capture re-detects the layout
        Render();
        return;
    }

    // A "new session" is any keystroke that did NOT arrive mid-typing-pipeline
    // (i.e. we were Idle / in a HUD / in a selection popup): only THEN do we
    // re-read the OS keyboard layout and lock the direction. Once locked, an
    // accidental layout tap mid-sentence can no longer flip it.
    const bool newSession = (phase_ != Phase::Typing && phase_ != Phase::Translating &&
                             phase_ != Phase::Ready);
    PublishLanguage(ResolveTypingLanguage(newSession));

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
    const bool typingFlow = (phase_ == Phase::Translating);
    const bool selectionFlow = (phase_ == Phase::SelTranslating);
    if (!typingFlow && !selectionFlow) {
        return;  // stale delivery: the user typed / committed / reselected meanwhile
    }
    if (!done) {
        translationPartial_ = text;  // streamed partial
    } else if (!text.empty()) {
        translationRaw_ = text;
        translationPartial_.clear();
        phase_ = typingFlow ? Phase::Ready : Phase::SelReady;
    } else {
        // Empty final = the run failed / was dropped. The typing flow falls back
        // to Typing (next keystroke retries); the selection flow just hides the
        // popup (there is no debounce loop to fall back into). Neither re-arms.
        translationPartial_.clear();
        phase_ = typingFlow ? Phase::Typing : Phase::Idle;
    }
    Render();
}

void CaretTracker::HandleCommit(IUIAutomation* automation) {
    // Passive-selection commit: overwrite the still-highlighted selection with
    // its translation. Separate path -- the OS already owns the range.
    if (phase_ == Phase::SelReady) {
        HandleSelectionCommit(automation);
        return;
    }

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

    // CONTEXT BLEED FIX: this is a STATELESS translation prosthetic. Do NOT
    // accumulate committed source into inferenceContext_ -- the assistant's
    // reply was injected into the field WITHOUT chat-template turn markers, so
    // feeding "<prev source><prev translation><new source>" back as one user
    // turn made the model re-emit the previous output. Every commit ends the
    // conversation: the next capture is a fresh single user turn vs. the system
    // prompt. (committedPrefix_ still hides the applied text from the capture.)
    inferenceContext_.clear();

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
    // A CenterHud banner owns the overlay and manages its own fade -- a stray
    // click/nav-key must not hide it.
    if (phase_ == Phase::Hud) {
        return;
    }
    phase_ = Phase::Idle;
    sourceRaw_.clear();
    translationRaw_.clear();
    translationPartial_.clear();
    inferenceContext_.clear();  // manual reset -> fresh conversation (context fix)
    idleArmed_ = false;
    sessionLanguage_ = -1;      // the next capture re-detects the OS layout
    Render();
}

void CaretTracker::HandleSelectionCheck(IUIAutomation* automation) {
    ComPtr<IUIAutomationElement> focused;
    ComPtr<IUIAutomationTextRange> range;
    std::wstring selected;

    // Primary: the focused element's UIA text SELECTION. Fails silently on
    // UIA-opaque controls (Qt/Telegram, WinUI3/Notepad) -- they either expose no
    // TextPattern at all or hand back an empty range even with text highlighted.
    if (SUCCEEDED(automation->GetFocusedElement(&focused)) && focused) {
        ReadSelection(focused.Get(), range, selected);  // leaves `selected` empty on failure
    }

    // Bulletproof fallback: no usable UIA selection -> back up the clipboard,
    // synthesize Ctrl+C against the focused control, read what it copied, and
    // restore the clipboard. This yields the SOURCE text for a read-mode popup
    // even in Qt/WinUI3. `range` stays null: there is no UIA range to Select() at
    // commit time, so a Ctrl+Enter over such a selection is a best-effort no-op
    // (the reading use-case does not write back) -- see HandleSelectionCommit.
    if (selected.empty()) {
        std::wstring clip;
        if (ReadSelectionViaClipboard(clip)) {
            selected = std::move(clip);
        }
    }
    if (selected.empty()) {
        return;  // genuinely just a click / caret move -- no highlighted text
    }

    // Supersede anything in flight (a prior popup or typing translation).
    if (phase_ == Phase::Translating || phase_ == Phase::Ready ||
        phase_ == Phase::SelTranslating || phase_ == Phase::SelReady) {
        if (callbacks_.cancelGeneration) {
            callbacks_.cancelGeneration();
        }
    }

    // Anchor the popup at the mouse cursor -- the user just released it at the
    // selection's active end.
    GetCursorPos(&anchor_);
    anchorValid_ = true;
    sourceRaw_ = std::move(selected);
    translationPartial_.clear();
    translationRaw_.clear();
    phase_ = Phase::SelTranslating;
    idleArmed_ = false;
    sessionLanguage_ = -1;  // a selection interrupts any locked typing session

    // Asymmetric direction: a selection ignores the OS layout entirely and uses
    // the reading default (e.g. EN->RU) -- read foreign text while your layout
    // stays ready to reply -- unless a manual override is pinned.
    PublishLanguage(ResolveSelectionLanguage());

    // A selection is an EXPLICIT request: warm the tree AND fire generation now
    // (no debounce, unlike the typing flow).
    if (callbacks_.trackUpdate) {
        callbacks_.trackUpdate(sourceRaw_, inferenceContext_);
    }
    if (callbacks_.triggerGeneration) {
        callbacks_.triggerGeneration(sourceRaw_, inferenceContext_);
    }
    Render();
}

void CaretTracker::HandleSelectionCommit(IUIAutomation* automation) {
    if (sourceRaw_.empty() || translationRaw_.empty()) {
        phase_ = Phase::Idle;
        Render();
        return;
    }
    if (callbacks_.injectionGuard) {
        callbacks_.injectionGuard(true);
    }

    TextInjector::Request request;
    request.source = sourceRaw_;
    request.replacement = translationRaw_;
    // The OS/user still has the range highlighted, so a bare Ctrl+V overwrites it
    // even in UIA-opaque apps (Telegram/Notepad) that expose no Selectable range.
    request.selectionActive = true;

    ComPtr<IUIAutomationElement> focused;
    if (SUCCEEDED(automation->GetFocusedElement(&focused)) && focused) {
        ComPtr<IUIAutomationTextRange> range;
        std::wstring current;
        // Prefer a verified UIA range: re-query and, only if the selection is
        // STILL exactly the text we translated, offer the range for the precise
        // Tier-2 Select()+paste. Otherwise Replace() falls through to the Tier-4
        // paste-over-live-selection (a plain click that collapsed the range would
        // have reset us out of SelReady before ever reaching here).
        if (ReadSelection(focused.Get(), range, current) && current == sourceRaw_) {
            request.selectionRange = range.Get();
        }
    }
    TextInjector::Replace(request);
    // Stateless prosthetic: a selection translation is a one-shot, so the
    // context never carries across it (see HandleCommit for the full rationale).
    inferenceContext_.clear();

    if (callbacks_.injectionGuard) {
        callbacks_.injectionGuard(false);
    }

    phase_ = Phase::Idle;
    sourceRaw_.clear();
    translationRaw_.clear();
    translationPartial_.clear();
    idleArmed_ = false;
    Render();
}

void CaretTracker::HandleSetActive(bool active) {
    active_ = active;
    if (active) {
        return;  // going active: wait for real input; the ACTIVE banner (if any) shows
    }
    // Leaving Translation Mode: stop paying for any in-flight decode and drop the
    // tracking state. Clear the pill (Render) so nothing lingers; the OFF banner
    // (ShowHud, issued right after by main) then takes over the overlay.
    if (callbacks_.cancelGeneration) {
        callbacks_.cancelGeneration();
    }
    if (phase_ != Phase::Hud) {
        phase_ = Phase::Idle;
        Render();
    }
    sourceRaw_.clear();
    translationRaw_.clear();
    translationPartial_.clear();
    idleArmed_ = false;
}

void CaretTracker::HandleShowHud(const std::wstring& message, bool fade) {
    phase_ = Phase::Hud;
    hudMessage_ = message;
    hudFade_ = fade;
    idleArmed_ = false;
    // A banner supersedes any tracking pill; drop stale capture/translation text.
    sourceRaw_.clear();
    translationPartial_.clear();
    translationRaw_.clear();
    Render();
}

void CaretTracker::Render() const {
    if (!callbacks_.render) {
        return;
    }
    OverlaySnapshot snapshot;
    snapshot.anchor = anchor_;
    snapshot.anchorValid = anchorValid_;
    // The header bar + override dropdown only exist for the caret-anchored typing
    // pipeline; the locked session direction drives them. `languagePinned`
    // distinguishes a manual override ("[Pinned]") from OS-layout auto-routing
    // ("[Auto]") so the header can explain WHY this direction is active.
    snapshot.language = sessionLanguage_;
    snapshot.languagePinned = languageOverride_.has_value();
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
        case Phase::Hud:
            snapshot.phase = OverlayPhase::CenterHud;
            snapshot.message = hudMessage_;
            snapshot.fade = hudFade_;
            snapshot.anchorValid = true;  // centered; needs no caret/cursor anchor
            break;
        case Phase::SelTranslating:
            // No source shown -- the OS already highlights it. Anchor = cursor.
            snapshot.phase = OverlayPhase::SelectionTranslating;
            snapshot.translation = translationPartial_;
            break;
        case Phase::SelReady:
            snapshot.phase = OverlayPhase::SelectionReady;
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

// ---------------------------------------------------------------------------
// OS-aware language routing (worker thread only)
// ---------------------------------------------------------------------------

int CaretTracker::DetectTypingLanguage() const {
    // The FOREGROUND thread's active keyboard layout is what actually produces
    // the glyphs the user is typing (our own STA thread's layout is irrelevant).
    const DWORD fgThread = GetWindowThreadProcessId(GetForegroundWindow(), nullptr);
    const HKL layout = GetKeyboardLayout(fgThread);
    const auto lang = static_cast<LANGID>(LOWORD(reinterpret_cast<UINT_PTR>(layout)));
    // A Cyrillic (Russian) layout means the user is typing Russian -> translate
    // AWAY from it (RU->EN); any other (Latin) layout -> EN->RU. PRIMARYLANGID
    // keeps this robust across every Russian sublanguage.
    if (PRIMARYLANGID(lang) == LANG_RUSSIAN) {
        return routing_.typingCyrillic;
    }
    return routing_.typingLatin;
}

int CaretTracker::ResolveTypingLanguage(bool newSession) {
    if (!newSession && sessionLanguage_ >= 0) {
        return sessionLanguage_;  // locked for the whole session -- no mid-sentence flip
    }
    // A manual override pins the direction regardless of the OS layout; otherwise
    // read the layout fresh and lock it in for the rest of this typing session.
    sessionLanguage_ = languageOverride_ ? *languageOverride_ : DetectTypingLanguage();
    return sessionLanguage_;
}

int CaretTracker::ResolveSelectionLanguage() const {
    // Asymmetric: never the OS layout -- the reading default unless pinned.
    return languageOverride_ ? *languageOverride_ : routing_.selectionReading;
}

void CaretTracker::PublishLanguage(int index) {
    if (index < 0 || index >= routing_.count) {
        return;  // unconfigured / out of range: keep the engine's current branch
    }
    if (index == publishedLanguage_) {
        return;  // unchanged: a burst of same-direction keystrokes costs one store
    }
    publishedLanguage_ = index;
    if (callbacks_.setActiveLanguage) {
        callbacks_.setActiveLanguage(index);
    }
}

void CaretTracker::HandleSetOverride(int index) {
    if (index >= 0 && index < routing_.count) {
        languageOverride_ = index;
    } else {
        languageOverride_.reset();  // -1 (or garbage) clears the pin -> OS-aware auto
    }
    // Apply immediately to a live typing session: relock to the new direction and
    // push it so the very next keystroke's speculative prefill (and the header)
    // use it. Idle/selection/HUD states pick it up on their next action.
    if (phase_ == Phase::Typing || phase_ == Phase::Translating || phase_ == Phase::Ready) {
        sessionLanguage_ = languageOverride_ ? *languageOverride_ : DetectTypingLanguage();
        PublishLanguage(sessionLanguage_);
        Render();
    }
}

// ---------------------------------------------------------------------------
// Clipboard-fallback selection capture (worker thread only)
// ---------------------------------------------------------------------------

namespace {

// Open the clipboard, tolerating the brief windows another process holds it.
bool OpenClipboardRetry() {
    for (int i = 0; i < 12; ++i) {
        if (OpenClipboard(nullptr)) return true;
        Sleep(8);
    }
    return false;
}

// A byte-exact snapshot of every HGLOBAL-backed clipboard format, so an injected
// Ctrl+C can borrow the clipboard and then hand it back untouched. Non-HGLOBAL
// formats (CF_BITMAP / CF_METAFILEPICT / CF_HDROP as a handle) are skipped --
// adequate for text-selection contexts, where CF_UNICODETEXT and any HTML/RTF
// payloads are all global-memory backed.
struct ClipboardBackup {
    struct Entry {
        UINT format = 0;
        std::vector<BYTE> bytes;
    };
    std::vector<Entry> entries;
    bool captured = false;
};

ClipboardBackup BackupClipboard() {
    ClipboardBackup backup;
    if (!OpenClipboardRetry()) return backup;
    for (UINT fmt = EnumClipboardFormats(0); fmt != 0; fmt = EnumClipboardFormats(fmt)) {
        HANDLE handle = GetClipboardData(fmt);
        if (!handle) continue;
        const SIZE_T size = GlobalSize(handle);
        if (size == 0) continue;  // not an HGLOBAL (or empty) -- skip
        if (const void* src = GlobalLock(handle)) {
            ClipboardBackup::Entry entry;
            entry.format = fmt;
            entry.bytes.assign(static_cast<const BYTE*>(src),
                               static_cast<const BYTE*>(src) + size);
            GlobalUnlock(handle);
            backup.entries.push_back(std::move(entry));
        }
    }
    CloseClipboard();
    backup.captured = true;
    return backup;
}

void RestoreClipboard(const ClipboardBackup& backup) {
    if (!backup.captured || !OpenClipboardRetry()) return;
    EmptyClipboard();
    for (const auto& entry : backup.entries) {
        HGLOBAL mem = GlobalAlloc(GMEM_MOVEABLE, entry.bytes.size());
        if (!mem) continue;
        if (void* dst = GlobalLock(mem)) {
            memcpy(dst, entry.bytes.data(), entry.bytes.size());
            GlobalUnlock(mem);
            if (!SetClipboardData(entry.format, mem)) {
                GlobalFree(mem);  // ownership not transferred
            }
        } else {
            GlobalFree(mem);
        }
    }
    CloseClipboard();
}

bool ReadClipboardUnicode(std::wstring& out) {
    if (!OpenClipboardRetry()) return false;
    bool ok = false;
    if (HANDLE handle = GetClipboardData(CF_UNICODETEXT)) {
        if (const auto* text = static_cast<const wchar_t*>(GlobalLock(handle))) {
            out.assign(text);
            GlobalUnlock(handle);
            ok = true;
        }
    }
    CloseClipboard();
    return ok;
}

// Synthesize a clean Ctrl+C: drop any physically-held modifier first (a stray
// Shift/Alt would turn it into a different chord), then Ctrl down / C / Ctrl up.
// Every event is tagged with the injector signature so our own low-level
// keyboard hook recognizes and passes it straight through.
void SendCopyChord() {
    INPUT seq[8] = {};
    int n = 0;
    const auto key = [&](WORD vk, bool up) {
        seq[n].type = INPUT_KEYBOARD;
        seq[n].ki.wVk = vk;
        seq[n].ki.dwFlags = up ? KEYEVENTF_KEYUP : 0;
        seq[n].ki.dwExtraInfo = TextInjector::kInjectedSignature;
        ++n;
    };
    key(VK_LSHIFT, /*up=*/true);
    key(VK_RSHIFT, /*up=*/true);
    key(VK_LMENU, /*up=*/true);
    key(VK_RMENU, /*up=*/true);
    key(VK_CONTROL, /*up=*/false);
    key('C', /*up=*/false);
    key('C', /*up=*/true);
    key(VK_CONTROL, /*up=*/true);
    SendInput(static_cast<UINT>(n), seq, sizeof(INPUT));
}

}  // namespace

bool CaretTracker::ReadSelectionViaClipboard(std::wstring& out) {
    // Bracket the whole round-trip with the injection guard so the synthetic
    // Ctrl+C never re-enters our keyboard hook as user input (or a chord reset).
    if (callbacks_.injectionGuard) callbacks_.injectionGuard(true);

    const ClipboardBackup backup = BackupClipboard();
    const DWORD seqBefore = GetClipboardSequenceNumber();

    SendCopyChord();

    // The target services the copy on ITS OWN thread, so poll the clipboard
    // sequence number (bumped by any SetClipboardData -- even copying identical
    // text) rather than sleeping a fixed, guessed interval. An unchanged number
    // means the control copied nothing: there was no real selection.
    std::wstring text;
    for (int i = 0; i < 25; ++i) {  // up to ~250 ms, then give up
        Sleep(10);
        if (GetClipboardSequenceNumber() != seqBefore) {
            ReadClipboardUnicode(text);
            break;
        }
    }

    RestoreClipboard(backup);
    if (callbacks_.injectionGuard) callbacks_.injectionGuard(false);

    if (text.empty()) return false;
    out = std::move(text);
    return true;
}
