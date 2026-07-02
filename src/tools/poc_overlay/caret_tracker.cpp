#include "caret_tracker.h"

#include <objbase.h>
#include <UIAutomation.h>
#include <wrl/client.h>

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
bool ReadTextPattern(IUIAutomationElement* focused, std::wstring& outText, POINT& outCaret,
                     bool& outCaretFound) {
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

    // Build a range spanning [start of current line .. caret] and read its text.
    ComPtr<IUIAutomationTextRange> line;
    if (FAILED(caret->Clone(&line)) || !line) {
        return false;
    }
    line->ExpandToEnclosingUnit(TextUnit_Line);

    ComPtr<IUIAutomationTextRange> upToCaret;
    if (FAILED(line->Clone(&upToCaret)) || !upToCaret) {
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
    return true;
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

CaretTracker::CaretTracker(UpdateCallback callback) : callback_(std::move(callback)) {
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
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return hasPending_ || stop_; });
            if (stop_) {
                break;
            }
            fallback = std::move(pendingFallback_);
            wordBoundary = pendingWordBoundary_;
            pendingWordBoundary_ = false;
            hasPending_ = false;
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

    // (1) Primary: TextPattern line-up-to-caret. This also yields the caret point.
    bool haveText = ReadTextPattern(focused.Get(), update.text, update.caretScreenPos,
                                    update.caretFound);

    // (1) Secondary: ValuePattern whole-control value.
    if (!haveText) {
        haveText = ReadValuePattern(focused.Get(), update.text);
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

    // (2) Text fallback: the hook's ToUnicodeEx buffer, for UIA-opaque apps.
    if (!haveText) {
        update.text = fallbackText;
    }
    return update.caretFound;
}
