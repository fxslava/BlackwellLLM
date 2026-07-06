#pragma once
#include <string>
#include <vector>

// One decoded (or source) token plus the model's confidence in it, for the
// Developer-Mode "thermograd" heatmap. `text` is the UTF-16 piece exactly as it
// contributes to the displayed string (concatenating a run of TokenInfo::text
// reproduces that run of the string, so the overlay can measure per-token
// character ranges by cumulative offset). `probability` is the softmax
// probability of the token under the model, in [0, 1]:
//   translation tokens -> the decode-time probability of the sampled token
//                         (low = the model was uncertain here);
//   source tokens      -> the teacher-forced prefill probability (perplexity),
//                         when the engine can supply it.
struct TokenInfo {
    std::wstring text;
    float probability = 1.0f;  // 1.0 = fully confident (renders "cold")
};

using TokenHeatmap = std::vector<TokenInfo>;
