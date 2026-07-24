# `blackwell_audio` — Ultravox multimodal frontend

An **independent, preprocessor-only** subsystem. It converts raw audio into
*audio embeddings* (rows in the text model's embedding space) that the engine
splices into `<|audio|>` placeholder slots (token id **128256**) before the
ordinary text decode. Target model: `fixie-ai/ultravox-v0_5-llama-3_2-1b`
(whisper-large-v3-turbo encoder + Ultravox SwiGLU projector + Llama-3.2-1B).

## The two hard rules

1. **Do not modify the text LLM kernels (`src/kernels/`), the KV cache, the decode
   loop, or `Impl::run_token`.** The frontend runs *before* the first text token
   is embedded and hands the engine a `[num_tokens, hidden]` buffer. The only
   engine-side change is a splice at the embedding step (see
   `ULTRAVOX_AUDIO_PLAN.md` §6). *Calling* a stateless text kernel (RMSNorm,
   SwiGLU, batched GEMM) is fine — that is reuse, not modification.
2. **Strict decoupling from LLM state.** The frontend holds NO engine/KV/sequence
   state. The pipeline is split at the encoder/projector boundary
   (`encode_features()` vs `project()`) so the expensive Whisper encoder runs once
   and cheap projectors run per model — the substrate for future **speculative
   decoding** (a Llama-3.2-1B *draft* and an ~8B *target* over the same audio).

## Layout

| Path | What |
|---|---|
| `kernels/` | Audio-only CUDA `.cu`/`.cuh` pairs (log-mel DSP, conv subsample, LayerNorm, bidirectional attention, encoder MLP, stack-frames, SwiGLU projector). Independent from `src/kernels/`. |
| `whisper_encoder.{h,cpp}` | 32-layer bidirectional encoder orchestration (`ModifiedWhisperEncoder`). |
| `ultravox_projector.{h,cpp}` | StackFrames → RMSNorm → linear → SwiGLU → RMSNorm → linear; bound to one text model. |
| `audio_frontend.{h,cpp}` | PIMPL implementation of `include/audio/audio_frontend.h`. |
| `audio_weight_loader.{h,cpp}` | Loads encoder + projector tensors from the Ultravox checkpoint; resolves `AudioEncoderGeometry`. |
| `CMakeLists.txt` | `blackwell_audio` static lib. **Not yet wired** into `src/CMakeLists.txt` — activated in Phase 0 once the first kernel lands (keeps the build green). |

## Doctrine inherited from the engine

- **Single owning thread.** Construct + call from the engine-owning thread only.
- **Hybrid error doctrine.** Loading/setup = INIT tier → `CUDA_CHECK_THROW`
  (throws `blackwell::cuda_error`, RAII unwinds). The `encode()/project()` path
  runs off the decode loop; a throw at the frontend boundary is converted to
  `EngineStatus` at the splice site.
- **`DeviceBuffer<T>` for all device memory**; declaration order = construction
  order; destructors never throw.
- **Kernels carry hard launch contracts in their `.cuh`**, matching the
  `src/kernels/` house style.

See `ULTRAVOX_AUDIO_PLAN.md` at the repo root for the full roadmap.
