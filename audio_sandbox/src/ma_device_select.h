#pragma once
// -----------------------------------------------------------------------------
// ma_device_select.h — resolving a configured device NAME to a miniaudio device
// id. PRIVATE to audio_sandbox/src: it names miniaudio types, so it must never
// be reachable from include/, which is the boundary the two device classes PIMPL
// themselves to protect.
//
// Header-only and inline because its two callers (audio_capture.cpp and
// audio_playback.cpp) are the only ones there will ever be, and giving it a TU
// would mean adding a source file to three targets to share fifteen lines.
//
// =============================================================================
// THE ID MUST COME FROM THE CONTEXT THE DEVICE IS OPENED ON
// =============================================================================
// ma_device_id is not a portable handle -- on WASAPI it wraps an endpoint ID
// string owned by the enumerating context. Passing an id obtained from context A
// to a device initialised on context B is undefined, and on Windows it usually
// "works" right up until the device list changes. Both callers therefore own an
// explicit ma_context, enumerate on it, and open on the same one; neither uses
// the implicit per-device context that ma_device_init(NULL, ...) creates.
//
// =============================================================================
// MATCHING IS DELIBERATELY FORGIVING, FALLBACK IS DELIBERATELY LOUD
// =============================================================================
// Endpoint names are long, punctuated and hardware-dependent
// ("Speakers (Realtek(R) Audio)"), and this value is hand-edited in a JSON file.
// An exact-match-only resolver would reject "Realtek" and leave the user
// guessing which of the three visually similar strings they mistyped. So:
// case-insensitive exact match first, then case-insensitive substring, and a
// substring that matches more than one device is treated as NO match -- picking
// one arbitrarily would silently open a device the user did not ask for, which
// is worse than falling back to the default and saying so.
// -----------------------------------------------------------------------------
#include <cstddef>
#include <string>

#include "miniaudio.h"

namespace rt::detail {

inline char ascii_lower(char c) noexcept {
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

inline std::string ascii_lowered(const std::string& s) {
    std::string out(s.size(), '\0');
    for (std::size_t i = 0; i < s.size(); ++i) out[i] = ascii_lower(s[i]);
    return out;
}

// Why the result is an enum and not a bool: these four failures need four
// different messages. A typo, an under-specified name, and an index past the end
// of the list are three different mistakes, and telling a user to fix the wrong
// one wastes their time.
enum class DeviceLookup {
    Matched,
    NotFound,
    Ambiguous,
    OutOfRange,   // an index was given, and there is no device at it
};

// `playback_list` selects WHICH list to search -- and it is not always the same
// as the device type being opened. WASAPI loopback captures the output of a
// PLAYBACK endpoint, so a loopback capture device is identified by a playback
// id. Getting this backwards yields MA_NO_DEVICE from ma_device_init with no
// hint as to why.
inline DeviceLookup resolve_device_id(ma_context* ctx, bool playback_list,
                                      const std::string& requested, ma_device_id* out_id,
                                      std::string* out_name) {
    if (ctx == nullptr || out_id == nullptr || requested.empty()) return DeviceLookup::NotFound;

    ma_device_info* playback = nullptr;
    ma_uint32 playback_count = 0;
    ma_device_info* capture = nullptr;
    ma_uint32 capture_count = 0;
    if (ma_context_get_devices(ctx, &playback, &playback_count, &capture, &capture_count) !=
        MA_SUCCESS) {
        return DeviceLookup::NotFound;
    }

    ma_device_info* list = playback_list ? playback : capture;
    const ma_uint32 count = playback_list ? playback_count : capture_count;
    if (list == nullptr || count == 0) return DeviceLookup::NotFound;

    const std::string want = ascii_lowered(requested);

    // Exact wins outright, and wins over any number of substring hits: a user
    // who typed the full name of one device must get that device even when its
    // name is a prefix of another's.
    for (ma_uint32 i = 0; i < count; ++i) {
        if (ascii_lowered(list[i].name) == want) {
            *out_id = list[i].id;
            if (out_name != nullptr) *out_name = list[i].name;
            return DeviceLookup::Matched;
        }
    }

    const ma_device_info* hit = nullptr;
    unsigned hits = 0;
    for (ma_uint32 i = 0; i < count; ++i) {
        if (ascii_lowered(list[i].name).find(want) != std::string::npos) {
            hit = &list[i];
            ++hits;
        }
    }
    if (hits == 1 && hit != nullptr) {
        *out_id = hit->id;
        if (out_name != nullptr) *out_name = hit->name;
        return DeviceLookup::Matched;
    }
    return hits > 1 ? DeviceLookup::Ambiguous : DeviceLookup::NotFound;
}

// Selection by zero-based INDEX into the list print_audio_devices() shows.
//
// =============================================================================
// THE INDEX IS A POSITION IN AN ENUMERATION, NOT AN IDENTITY
// =============================================================================
// It is only meaningful against the list the user just read, and miniaudio's
// order is the backend's order -- so plugging in a headset, or Windows
// re-ordering its endpoints, silently re-points an index at a different device.
// A name survives that; an index does not. This exists because an index is the
// thing a user can actually type without transcribing
// "Speakers (2- High Definition Audio Device)" by hand, and because the failure
// it can produce (the wrong device, audibly) is recoverable in a way that a
// misspelled name (silence, with no clue why) is not.
//
// Consequence for the caller: the list MUST be printed from the same backend
// this resolves against, and printed at the same startup that reads the setting.
// print_audio_devices() enumerates on a temporary context of its own, which is a
// different ma_context from the one passed here -- the orders agree because both
// come from the same backend enumerating the same machine, not because anything
// guarantees it. That is why an out-of-range index falls back and warns rather
// than clamping to the nearest valid one: if the lists have drifted, the nearest
// valid index is not "close", it is arbitrary.
//
// Negative index means "not selected by index" and is NOT an error -- it is the
// default, and the caller is expected to try a name next.
inline DeviceLookup resolve_device_by_index(ma_context* ctx, bool playback_list, int index,
                                            ma_device_id* out_id, std::string* out_name) {
    if (ctx == nullptr || out_id == nullptr || index < 0) return DeviceLookup::NotFound;

    ma_device_info* playback = nullptr;
    ma_uint32 playback_count = 0;
    ma_device_info* capture = nullptr;
    ma_uint32 capture_count = 0;
    if (ma_context_get_devices(ctx, &playback, &playback_count, &capture, &capture_count) !=
        MA_SUCCESS) {
        return DeviceLookup::NotFound;
    }

    ma_device_info* list = playback_list ? playback : capture;
    const ma_uint32 count = playback_list ? playback_count : capture_count;
    if (list == nullptr || static_cast<ma_uint32>(index) >= count) return DeviceLookup::OutOfRange;

    *out_id = list[static_cast<ma_uint32>(index)].id;
    if (out_name != nullptr) *out_name = list[static_cast<ma_uint32>(index)].name;
    return DeviceLookup::Matched;
}

// The one entry point both device classes use, so the PRECEDENCE between the two
// settings is decided once instead of twice.
//
// INDEX WINS over name when both are set. An index is an unambiguous ordinal the
// user picked off a printed list; a name is a fuzzy substring match. Letting the
// weaker one override the stronger would mean a stale `output_device_name` left
// in settings.json quietly defeats the index the user just typed to get away
// from it.
//
// A failed index does NOT fall through to the name. Two selectors disagreeing is
// exactly the state where guessing is worst: the user gets the device they did
// not ask for, and the warning about the one they did ask for scrolls away. Fall
// back to the system default and say which selector failed.
//
// `out_reason` receives a caller-printable clause ("index 7 is past the end of
// the list") whenever the result is not Matched; the device classes prefix it
// with their own direction so one string serves input and output.
inline DeviceLookup resolve_device_selection(ma_context* ctx, bool playback_list, int index,
                                             const std::string& name, ma_device_id* out_id,
                                             std::string* out_name, std::string* out_reason) {
    const auto reason = [out_reason](const char* text) {
        if (out_reason != nullptr) *out_reason = text;
    };

    if (index >= 0) {
        const DeviceLookup r = resolve_device_by_index(ctx, playback_list, index, out_id, out_name);
        if (r != DeviceLookup::Matched) {
            reason(r == DeviceLookup::OutOfRange
                       ? "index is past the end of the device list"
                       : "device list could not be read");
        }
        return r;
    }

    if (name.empty()) return DeviceLookup::NotFound;   // neither selector set: the default

    const DeviceLookup r = resolve_device_id(ctx, playback_list, name, out_id, out_name);
    if (r != DeviceLookup::Matched) {
        reason(r == DeviceLookup::Ambiguous ? "name matches more than one endpoint"
                                            : "name not found");
    }
    return r;
}

}  // namespace rt::detail
