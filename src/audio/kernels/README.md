# `blackwell_audio` CUDA kernels

Audio-only SIMT kernels for the Ultravox frontend. **Independent** of
`src/kernels/` — nothing here modifies the text LLM kernels. One `.cu` + `.cuh`
pair per NEW kernel, hard launch contracts documented in the `.cuh` (alignment,
dim multiples, buffer validity), matching the `src/kernels/` house style.
Stateless text kernels (RMSNorm, SwiGLU, batched GEMM) are **reused by calling
them**, not copied.

Pipeline order (implemented + validated — see `ULTRAVOX_AUDIO_PLAN.md` §4):

| Kernel | New/Reuse | Op | Validation |
|---|---|---|---|
| `log_mel` | new | STFT power → mel filterbank (128) → `log10`/clamp (Whisper DSP) | vs `WhisperFeatureExtractor` `input_features` |
| `conv_subsample` | new | conv1(k3,s1)+GELU → conv2(k3,s2)+GELU, time ÷2 | vs `conv_out` dump |
| `layer_norm` | new | true LayerNorm (mean-subtract + bias) — Whisper uses LN, **not** RMSNorm | folded into encoder-layer checks |
| `audio_attention` | new | bidirectional (non-causal) multi-head self-attention, 20 heads | vs per-layer `enc_layer_i` |
| `encoder_mlp` | reuse GEMM + new GELU | `fc1`(1280→5120) → GELU → `fc2`(5120→1280) | vs per-layer `enc_layer_i` |
| `stack_audio_frames` | new | pad time to %8, reshape [T,1280]→[T',10240] | vs `stacked` dump |
| `projector` | reuse RMSNorm+GEMM + new swiglu-chunk | ln_pre → linear_1(10240→4096) → SwiGLU(→2048) → ln_mid → linear_2(2048→2048) | vs `audio_embeds` dump |

Parity bar for every kernel: **cosine similarity > 0.999** against the PyTorch
golden tensor (plus a max-relative-error check), per project rule.

**First kernel to write (Phase 1): `log_mel`.**
