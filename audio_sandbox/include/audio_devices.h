#pragma once
// -----------------------------------------------------------------------------
// audio_devices.h — enumerating the machine's audio endpoints.
//
// WHY THIS IS A SEPARATE HEADER AND NOT A METHOD ON EITHER DEVICE CLASS.
// Enumeration has to be available BEFORE anything is opened -- that is the whole
// point of `--list-audio-devices`, and it is what makes a misspelled device name
// diagnosable rather than merely fatal. Hanging it off AudioCapture would mean
// constructing a capture object to ask what capture objects could exist.
//
// STL-ONLY BY CONSTRUCTION, like the two device headers beside it: the
// miniaudio types stay inside audio_capture.cpp, which is the single TU in this
// project that compiles miniaudio's implementation. `name` is the string a user
// puts in settings.json, and it is the string the resolver matches against, so
// what is printed here is exactly what is configurable -- if those two ever
// drift, a name that the UI shows stops being a name the app accepts.
//
// THE POSITION IN THE RETURNED VECTOR IS THE SETTABLE INDEX. It is what
// output_device_index / input_device_index mean, and what the [N] in the printed
// listing shows. There is no `index` field because there must not be one: an
// index that is stored rather than derived can disagree with the position it was
// stored at, and then the number the user reads is not the number the resolver
// uses. Anything enumerating for display and anything enumerating for selection
// must both take the position from this same call ordering.
//
// LIFETIME: the returned vector OWNS its strings. miniaudio's own
// ma_device_info array is context-owned and is invalidated by the next
// enumeration on that context, which is precisely the kind of dangling this
// copy exists to prevent.
// -----------------------------------------------------------------------------
#include <string>
#include <vector>

namespace rt {

enum class AudioDeviceKind {
    Playback,   // speakers / headphones
    Capture,    // microphones
};

struct AudioDeviceInfo {
    std::string name;
    // The endpoint Windows would pick on its own. Reported rather than assumed,
    // because "the default" is a user setting that changes under a running app.
    bool is_default = false;
};

// Enumerates on a TEMPORARY context of its own, so it is safe to call at any
// time -- before, during or after a device is open. Returns an empty vector if
// the backend cannot be initialised; that is a "nothing to show" condition for a
// diagnostic listing, not an error worth throwing over.
std::vector<AudioDeviceInfo> enumerate_audio_devices(AudioDeviceKind kind);

// Prints both lists to stdout in the form the settings file wants: a zero-based
// [N] the user can put in *_device_index, and the endpoint name they can put in
// *_device_name. This is the output a user copies out of, so names print
// VERBATIM -- no truncation, no prettifying -- and the index printed is the
// position the resolver will index with.
//
// Called unconditionally at startup, not only under --list-audio-devices. A
// device list is the first thing needed to diagnose silence, and silence is the
// failure mode that gives the user no other clue that a device setting is even
// involved.
void print_audio_devices();

}  // namespace rt
