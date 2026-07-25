#pragma once
#include <cstddef>
// -----------------------------------------------------------------------------
// Prompt injector: splice the projector's audio embeddings into the text
// embedding sequence at the `<|audio|>` placeholder (token id 128256). This is
// the engine's multimodal seam -- it replaces ONE placeholder row with the
// num_audio projected soft-token rows, leaving the text decode untouched:
//
//   out = concat(text[:audio_pos], audio[:num_audio], text[audio_pos+1:])
//   out rows = seq_len - 1 + num_audio
//
// HARD CONTRACT: no device allocation here (caller owns d_out); d_out must NOT
// alias d_text/d_audio; hidden % 4 == 0 (float4 copy); 0 <= audio_pos < seq_len.
// -----------------------------------------------------------------------------
void inject_audio_embeddings(const float* d_text,
                             const float* d_audio,
                             float* d_out,
                             int seq_len,
                             int audio_pos,
                             int num_audio,
                             int hidden);
