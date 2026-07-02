#include "caret_tracker.h"

#include <objbase.h>
#include <UIAutomation.h>
#include <wrl/client.h>

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

// Primary path: read the active line up to the caret via TextPattern, and the
// caret's screen point. Returns true only if the text was successfully read.
// When `outRange` is provided it also receives the [line-start .. caret] range,
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

}  // namespace

CaretTracker::CaretTracker(UpdateCallback callback, TransformCallback commitTransform,
                           InjectionGuard injectionGuard)
    : callback_(std::move(callback)),
      transform_(std::move(commitTransform)),
      injectionGuard_(std::move(injectionGuard)) {
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

void CaretTracker::RequestUpdate(std::wstring fallbackText, bool wordBoundary) {
    // Called from the keyboard hook -- must stay O(1) and never touch COM/UIA.
    // Overwrites the pending text with the latest state so a burst of keystrokes
    // costs a single UIA round-trip, but OR-accumulates the word-boundary flag so
    // a separator coalesced behind a later keystroke is not lost.
    std::lock_guard<std::mutex> lock(mutex_);
    pendingFallback_ = std::move(fallbackText);
    pendingWordBoundary_ = pendingWordBoundary_ || wordBoundary;
    hasPending_ = true;
    cv_.notify_all();
}

void CaretTracker::RequestCommit(std::wstring fallbackText) {
    // Also O(1) and hook-safe: hand the commit to the STA worker, where the UIA
    // interfaces live and where the (blocking, clipboard-sleeping) replacement
    // can run without stalling the UI thread.
    std::lock_guard<std::mutex> lock(mutex_);
    pendingCommit_ = true;
    pendingCommitFallback_ = std::move(fallbackText);
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
        bool wordBoundary = false;
        bool doCommit = false;
        std::wstring commitFallback;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return hasPending_ || stop_; });
            if (stop_) {
                break;
            }
            if (pendingCommit_) {
                doCommit = true;
                commitFallback = std::move(pendingCommitFallback_);
                pendingCommit_ = false;
            }
            fallback = std::move(pendingFallback_);
            wordBoundary = pendingWordBoundary_;
            pendingWordBoundary_ = false;
            hasPending_ = false;
        }

        // A commit replaces the text and hides the overlay, so it supersedes any
        // coalesced display update in the same wake-up.
        if (doCommit) {
            if (SUCCEEDED(hrAutomation)) {
                PerformCommit(automation.Get(), commitFallback);
            }
            continue;
        }

        CaretUpdate update;
        if (SUCCEEDED(hrAutomation)) {
            Resolve(automation.Get(), fallback, update);
        } else {
            update.text = fallback;  // UIA unavailable -- nothing to anchor to
        }
        // Stamp the boundary flag onto the resolved (authoritative) text so the
        // consumer can gate inference on the real string, not the fallback.
        update.wordBoundary = wordBoundary;

        if (callback_) {
            callback_(update);
        }
    }

    automation.Reset();
    if (SUCCEEDED(hrCoInit)) {
        CoUninitialize();
    }
}

bool CaretTracker::Resolve(IUIAutomation* automation, const std::wstring& fallbackText,
                           CaretUpdate& update) {
    update.text.clear();
    update.caretFound = false;

    ComPtr<IUIAutomationElement> focused;
    if (FAILED(automation->GetFocusedElement(&focused)) || !focused) {
        update.text = fallbackText;
        return false;  // no focus -> no anchor
    }

    // (1) Primary: TextPattern document-up-to-caret. This also yields the caret.
    std::wstring uiaText;
    bool haveText =
        ReadTextPattern(focused.Get(), uiaText, update.caretScreenPos, update.caretFound);

    // (1) Secondary: ValuePattern whole-control value.
    if (!haveText) {
        haveText = ReadValuePattern(focused.Get(), uiaText);
    }

    // Caret position fallback: some providers (or the ValuePattern path) give no
    // caret rect -- anchor to the focused element's top-left instead.
    if (!update.caretFound) {
        RECT rect{};
        if (SUCCEEDED(focused->get_CurrentBoundingRectangle(&rect)) &&
            (rect.right > rect.left || rect.bottom > rect.top)) {
            update.caretScreenPos.x = rect.left;
            update.caretScreenPos.y = rect.top;
            update.caretFound = true;
        }
    }

    if (haveText) {
        // Show only what was typed since the last commit; the committed text stays
        // hidden (but lives on in inferenceContext_).
        update.text = StripCommittedPrefix(uiaText);
    } else {
        // (2) UIA-opaque app: the hook's ToUnicodeEx buffer already resets on
        // commit, so it is the current segment as-is.
        update.text = fallbackText;
    }
    return update.caretFound;
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

void CaretTracker::PerformCommit(IUIAutomation* automation, const std::wstring& fallbackText) {
    // Flag the injection window so the keyboard hook treats the upcoming
    // synthetic input as ours (belt-and-suspenders alongside the dwExtraInfo tag).
    if (injectionGuard_) {
        injectionGuard_(true);
    }

    // Gather the UIA context HERE, on the STA thread, so the interfaces we pass to
    // TextInjector stay in their owning apartment. Re-resolving (rather than
    // reusing a cached range) keeps us in sync with the field's current state.
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

    // Work out the segment typed since the last commit -- that is what we replace,
    // preserving any already-committed prefix.
    std::wstring segment;
    bool haveBoundary = false;
    if (fromUia) {
        if (!committedPrefix_.empty() && fullText.rfind(committedPrefix_, 0) == 0) {
            segment = fullText.substr(committedPrefix_.size());
            haveBoundary = true;
        } else {
            committedPrefix_.clear();
            inferenceContext_.clear();
            segment = fullText;
        }
    } else {
        segment = fallbackText;  // UIA-opaque: the hook buffer is the current segment
    }

    if (!segment.empty()) {
        const std::wstring target = transform_ ? transform_(segment) : segment;

        TextInjector::Request request;
        request.source = segment;
        request.replacement = target;
        if (fromUia) {
            // Narrow the range to just the new segment [prefix-end .. caret] so the
            // earlier committed prefix is preserved by Tier 2.
            if (haveBoundary && range) {
                int moved = 0;
                range->MoveEndpointByUnit(TextPatternRangeEndpoint_Start, TextUnit_Character,
                                          static_cast<int>(committedPrefix_.size()), &moved);
            }
            request.selectionRange = range.Get();
            // SetValue replaces the WHOLE value, so only offer it when nothing is
            // preserved ahead of the segment.
            request.valuePattern = haveBoundary ? nullptr : valuePattern.Get();
        }

        const TextInjector::Tier tier = TextInjector::Replace(request);
        if (tier != TextInjector::Tier::None && fromUia) {
            committedPrefix_ += target;    // field now holds prefix + translated segment
            inferenceContext_ += segment;  // AI history retains the ORIGINAL text
            OutputDebugStringW(
                (L"[poc_overlay] InferenceContext: \"" + inferenceContext_ + L"\"\n").c_str());
        }
    }

    if (injectionGuard_) {
        injectionGuard_(false);
    }
}