#pragma once
// -----------------------------------------------------------------------------
// translator/language_table.hpp — the ONE forced-language table shared by the
// CLI (--src-lang/--tgt-lang), the ImGui dropdowns, and RealEngineControl's
// per-turn prompt builder. A selection crosses threads as a bare table index in
// an atomic, so every consumer must agree on the same ordering: never reorder or
// remove entries mid-flight; append only. Index 0 is the "Auto" sentinel (no
// forcing — model-side behaviour).
//
// Deliberately dependency-free (no json/CUDA/ImGui) so any translator TU can
// include it.
// -----------------------------------------------------------------------------
#include <cctype>
#include <cstring>
#include <string>

namespace rt {

inline constexpr const char* kLanguages[] = {
    "Auto",    "Russian",    "English",  "German",   "French",  "Spanish",
    "Italian", "Portuguese", "Chinese",  "Japanese", "Korean",  "Ukrainian",
};
inline constexpr int kLanguageCount =
    static_cast<int>(sizeof(kLanguages) / sizeof(kLanguages[0]));

// Case-insensitive name -> table index; unknown / empty -> -1 (callers choose
// whether that is an error (CLI) or the Auto fallback (config pass-through)).
inline int language_index(const std::string& name) {
    auto ieq = [](const std::string& a, const char* b) {
        if (a.size() != std::strlen(b)) return false;
        for (std::size_t i = 0; i < a.size(); ++i) {
            if (std::tolower(static_cast<unsigned char>(a[i])) !=
                std::tolower(static_cast<unsigned char>(b[i])))
                return false;
        }
        return true;
    };
    for (int i = 0; i < kLanguageCount; ++i)
        if (ieq(name, kLanguages[i])) return i;
    return -1;
}

// Config-tier fallback: an unknown/unset name degrades to Auto (index 0).
inline int language_index_or_auto(const std::string& name) {
    const int i = language_index(name);
    return i < 0 ? 0 : i;
}

}  // namespace rt
