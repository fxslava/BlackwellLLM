# Ultravox Audio Frontend — Implementation Plan

**Branch:** `feature/ultravox-audio-frontend`
**Target model:** [`fixie-ai/ultravox-v0_5-llama-3_2-1b`](https://huggingface.co/fixie-ai/ultravox-v0_5-llama-3_2-1b)
— a `whisper-large-v3-turbo` audio encoder + an Ultravox SwiGLU projector +
a `Llama-3.2-1B-Instruct` text backbone.

**Goal:** add speech input to BlackwellLLM by building an **independent audio
frontend** that converts raw audio into *audio embeddings* — rows in the text
model's embedding space — which the engine splices into `<|audio|>` placeholder
slots (token id **128256**) before the ordinary text decode. The text LLM
kernels, KV cache, and decode loop are **not modified**; the frontend is a strict
preprocessor.

**Parity bar (project rule):** every stage is validated tensor-for-tensor
against a PyTorch golden dump at **cosine similarity > 0.999** (plus a
max-relative-error guard).

> ### ⚠️ Architectural note — strict decoupling from LLM state (load-bearing)
> The frontend is a **pure function** `audio → embeddings`. It holds **no** KV
> cache, no sequence position, no per-decode state, and no reference to any
> `BlackwellEngine` instance. It is reentrant and its output is a plain value
> buffer.
>
> **Why this matters:** the roadmap targets **speculative decoding**, where the
> same audio must condition **two independent models at once** — a `Llama-3.2-1B`
> **draft** and an `~8B` **target**. Their hidden sizes differ (2048 vs 4096), so
> the shared, expensive artifact is the **Whisper encoder output** (`[T, 1280]`),
> computed **once**; the **projector is a thin, per-model adapter** run twice
> (once → 2048 for the draft, once → 4096 for the target). The frontend API
> therefore exposes the encoder features *and* the projected embeddings
> separately, and a frontend instance carries a swappable projector — never a
> baked-in assumption of a single consumer model. See §2.1 and §6.

---

## 1. Architecture research — Ultravox v0.5 (Llama-3.2-1B)

Sourced from the Ultravox model source (`UltravoxModel` / `UltravoxProjector` /
`ModifiedWhisperEncoder`) and the checkpoint `config.json`. Re-verify against the
installed package before generating dumps (golden-dumps geometry contract).

### 1.1 Audio encoder — `ModifiedWhisperEncoder` (whisper-large-v3-turbo)

The "turbo" variant only shrinks the Whisper *decoder* (32→4 layers); the
**encoder is unchanged from large-v3** and is the only part Ultravox uses. The
"Modified" wrapper removes the fixed 30 s padding requirement (variable-length
audio) and allows longer positional ranges — geometry below is otherwise stock
Whisper.

| Field | Value |
|---|---|
| `num_mel_bins` | 128 |
| `d_model` (hidden) | 1280 |
| `encoder_layers` | 32 |
| `encoder_attention_heads` | 20 (head_dim 64) |
| `encoder_ffn_dim` | 5120 |
| conv1 | `Conv1d(128 → 1280, k=3, s=1, p=1)` + GELU |
| conv2 | `Conv1d(1280 → 1280, k=3, s=2, p=1)` + GELU (**time ÷2**) |
| positional | learned `embed_positions`, added post-conv |
| layer | **pre-norm, bidirectional (non-causal)**: `h += Attn(LN(h))`; `h += FC2(GELU(FC1(LN(h))))` |
| final | `layer_norm` |

Note: **Ultravox does NOT use Whisper's `avg_pooler`.** Temporal downsampling
happens in the projector via `StackAudioFrames` (§1.2).

### 1.2 Projector — `UltravoxProjector` (the connector, exact structure)

Config for this checkpoint: `stack_factor=8`, `projector_act="swiglu"`,
`projector_ln_mid=true`, projector `hidden_size=4096`, `norm_init=0.4`. All norms
are **`RMSNorm`** (LlamaRMSNorm, no unit-offset), all linears are **`bias=False`**.

```
audio encoder out:  [B, T, 1280]                       (T = mel_frames / 2)
 └─ StackAudioFrames(stack_factor=8):                   pad T to %8, then reshape
        [B, T, 1280] → [B, ceil(T/8), 1280*8] = [B, T', 10240]   # time ÷8
 └─ ln_pre   : RMSNorm(10240)
 └─ linear_1 : Linear(10240 → 4096, bias=False)         # -> [B, T', 4096]
 └─ swiglu   : chunk 4096 into (x|gate) halves; F.silu(gate) * x   -> [B, T', 2048]
 └─ ln_mid   : RMSNorm(2048)          # because projector_ln_mid=True
 └─ linear_2 : Linear(2048 → 2048, bias=False)          # -> [B, T', 2048]  (Llama-1B hidden)
 └─ ln_post  : Identity               # because projector_ln_mid=True
     audio embeddings: [B, T', 2048]
```

Dimension formulas (general): `dim_in = audio_d_model * stack_factor`;
`linear_1: dim_in → proj_hidden`; `swiglu: proj_hidden → proj_hidden/2`;
`linear_2: proj_hidden/2 → text_hidden`. **v0.5 places the mid-norm after
linear_1** (`projector_ln_mid=True` → `ln_mid=RMSNorm`, `ln_post=Identity`); v0.4.1
and below did the opposite — key it off the config flag, never hard-code.

**Total downsampling:** mel frames → conv2 (÷2) → stack (÷8) = **÷16**. ~1500 mel
→ ~94 audio embeddings per 30 s (exact count = `ceil((mel/2)/8)`).

### 1.3 Injection — `<|audio|>` placeholder replacement

`audio_token_index = 128256` (an added special token, one past Llama-3.2's 128256
base vocab). The processor emits N placeholder tokens per clip, N = number of
audio embeddings. At merge, text embeddings at those positions are **overwritten**
by the audio embeddings (per-position replacement / `masked_scatter`). No
attention-mask trickery, no causal-mask change: post-merge the sequence is
ordinary embeddings the decoder never distinguishes.

---

## 2. Design — independent frontend as a preprocessor

**The seam.** After `embed_tokens` fills `[seq, hidden]`, overwrite the rows at
`<|audio|>` (128256) positions with the projector output, *before* layer 0.
Everything upstream (DSP → encoder → projector) is the self-contained
`blackwell_audio` library. It never links or calls into the text decode loop.

Honors the engine's extension patterns:
- **No text-kernel changes** (`src/kernels/` untouched). Stateless kernels *may*
  be **reused by calling them** (RMSNorm, SwiGLU, batched GEMM) — calling ≠
  modifying; new files only for audio-specific ops (§4).
- **Capability-gated** (pattern #1): `ModelCapabilities::accepts_audio_input`;
  `require_audio_input()` throws with "load an Ultravox checkpoint" guidance.
- **Tiered config** (pattern #2): `enable_audio` / `audio_weights_dir` enter via
  `InferenceConfig` → validated in `build_and_validate_runtime()` → `RuntimeConfig`.
- **Single owning thread**: constructed + driven from the engine-owning thread;
  `encode()` runs as an engine task.
- **Declaration order** (pattern #3): frontend `unique_ptr` after `arena`, null
  for text-only models (conditional-subsystem pattern, like `ssm_state`).

### 2.1 Decoupling for speculative decoding (design consequence)

The frontend interface splits the pipeline at the encoder/projector boundary:

```
encode_features(pcm)  -> AudioFeatures { [T, 1280] }     # shared, expensive, run ONCE
project(features, target)  -> AudioEmbeddings { [T', H] }# cheap, per text model
```

- `AudioFeatures` is model-agnostic (Whisper space, 1280-dim). Cache/reuse it.
- A **projector is bound to one text model** (its `text_hidden` + its projector
  weights). Speculative decoding constructs **two** projectors — draft (H=2048)
  and target (H=4096) — over the **same** `AudioFeatures`.
- The frontend owns **no** engine/KV state, so the two model instances consume the
  embeddings independently and concurrently. This is why the API must not fold
  "encode" and "project" into one call, and why the projector weights are not a
  singleton inside the encoder.

`create_audio_frontend()` stays a convenience for the single-model case (encode +
project fused); the split methods are the substrate for spec-dec.

---

## 3. Directory & build layout (scaffolded on this branch)

```
include/audio/audio_frontend.h      # light C++ contract (no CUDA), encode/project split
src/audio/
  README.md                         # subsystem doctrine
  CMakeLists.txt                    # blackwell_audio (NOT yet add_subdirectory'd)
  kernels/                          # audio-only .cu/.cuh pairs (§4)
  audio_weight_loader.{h,cpp}       # checkpoint → geometry + device weights
  whisper_encoder.{h,cpp}           # 32-layer bidirectional encoder orchestration
  ultravox_projector.{h,cpp}        # StackFrames → RMSNorm → linear → swiglu → RMSNorm → linear
  audio_frontend.{h,cpp}            # PIMPL impl of the public contract
scripts/generate_ultravox_audio_dumps.py  # PyTorch golden-dump generator
tests/validation/test_audio_*.cpp   # per-kernel FP64/FP32 CPU-reference parity
tests/integration/test_ultravox_frontend.cpp  # end-to-end vs golden dumps
```

**Build wiring rule:** `src/audio/CMakeLists.txt` exists but is deliberately
**not** referenced by `add_subdirectory(audio)` in `src/CMakeLists.txt` yet — a
zero-source target fails to configure, so activation is **Phase 0 step 1**, in the
same commit as the first kernel. Follows `src/kernels/`: explicit source list, no
exported include paths.

---

## 4. CUDA kernel sequence (the exact kernels to write, in order)

One `.cu` + `.cuh` pair per NEW kernel under `src/audio/kernels/`, hard launch
contract in the `.cuh` (alignment, dim multiples, buffer validity), `launch_*`
free function, SIMT design following the existing kernels.

| # | Kernel | New/Reuse | Op | Validated against |
|---|---|---|---|---|
| K1 | `log_mel` | **new** | STFT power → mel filterbank (128) → `log10`/clamp/normalize (Whisper DSP) | `input_features` dump |
| K2 | `conv_subsample` | **new** | conv1(k3,s1,p1)+GELU → conv2(k3,s2,p1)+GELU (time ÷2) | `conv_out` dump |
| K3a | `layer_norm` | **new** | true LayerNorm (mean-subtract + bias) — Whisper uses LN, **not** RMSNorm; do NOT reuse `rmsnorm.cu` | folded into K3/K4 checks |
| K3 | `audio_attention` | **new** | bidirectional (non-causal) MHA, 20 heads, seqlen≈T; no KV cache | per-layer `enc_layer_i` |
| K4 | `encoder_mlp` | reuse `batched_bf16_gemm` + **new** GELU epilogue | fc1(1280→5120)+GELU, fc2(5120→1280) | per-layer `enc_layer_i` |
| K5 | `stack_audio_frames` | **new** | pad T to %8, reshape [T,1280]→[T',10240] | `stacked` dump |
| K6 | `projector` | reuse `rmsnorm.cu` (stateless call) + `batched_bf16_gemm` + **new** swiglu-chunk | ln_pre→lin1→swiglu→ln_mid→lin2 | `audio_embeds` dump |

Reuse policy: **calling** a stateless text kernel (`launch_rmsnorm_kernel`,
`batched_bf16_gemm`, `launch_swiglu_*`) is allowed and encouraged — it does not
modify the text path. `blackwell_audio` links `blackwell_kernels` for this.
Whisper **LayerNorm ≠ RMSNorm**, so K3a is genuinely new. The projector's RMSNorm
matches Llama semantics (no unit-offset), so it reuses `launch_rmsnorm_kernel`
directly.

**Numerics:** encoder + projector run FP16/BF16 weights, FP32 accumulate (matching
the repo's GEMV/GEMM). Parity target is the FP32 PyTorch pass; cosine > 0.999
absorbs the low-bit rounding, as the text kernels' parity tests do.

---

## 5. Golden-dump parity protocol

Generator: `scripts/generate_ultravox_audio_dumps.py` (scaffolded). Loads the
Ultravox model (trust-remote-code), runs the processor + `ModifiedWhisperEncoder`
+ `UltravoxProjector` on a deterministic clip, dumps float32 `.bin` per stage
(`input_features`, `conv_out`, `enc_layer_{i}`, `encoder_final_norm`, `stacked`,
`proj_ln_pre`, `proj_linear_1`, `proj_swiglu`, `audio_embeds`) + `meta.json`.
Paths follow the golden-dumps skill (`BLACKWELL_MODELS_DIR`, repo-anchored out
dir, committed under `tests/integration/golden_dumps/ultravox/` via Git LFS).

Two tiers, mirroring the existing suites:
- **`tests/validation/test_audio_*.cpp`** — each kernel vs an FP64 CPU reference
  on real shapes (`rel_frobenius`/`max_rel_error` from `tests/common/`) **plus** a
  new `cosine_similarity` helper asserting `> 0.999`.
- **`tests/integration/test_ultravox_frontend.cpp`** — whole frontend vs golden
  dumps, stage by stage then end-to-end `audio_embeds`; `env_or()` override
  (`BLACKWELL_ULTRAVOX_DIR`); skips loudly when dumps absent (a skip is not a pass).

---

## 6. Engine integration (the splice)

1. **Config/capability.** `ModelCapabilities::accepts_audio_input`; `enable_audio`
   / `audio_weights_dir` on `InferenceConfig`; `require_audio_input()` gate.
2. **Ownership.** `Impl` gains `std::unique_ptr<audio::AudioFrontend>` after
   `arena`, constructed only when audio is enabled (null otherwise). `~Impl()`
   stays `= default`.
3. **Prompt API.** Caller supplies audio clips + the `<|audio|>`-annotated token
   sequence. Engine calls `predict_num_tokens()` to size the placeholder run at
   tokenization time, then `encode_features()` + `project()` on the engine thread.
4. **The splice.** After `embed_tokens`, overwrite the placeholder (128256) rows
   with the projector output (device scatter — NOT a text-kernel change).
5. **Spec-dec readiness (§2.1).** Because the frontend is engine-state-free and
   `encode_features`/`project` are split, the future draft+target setup runs the
   encoder once and projects per model. Keep this split intact — do not let the
   engine cache "the" audio embeddings as a singleton.
6. **Error tiers.** Loading = INIT (throws → `E_OUTOFMEMORY`/`E_INVALIDARG` at the
   COM edge). `encode()`/`project()` at the splice are wrapped so a throw becomes
   an `EngineStatus` (`status_from_current_exception()`), like `forward_status`.
7. **DLL boundary.** If COM consumers need audio, add a matching `IBlackwellEngine`
   method (HRESULT + POD out-params) in the same change; else keep it white-box.

---

## 7. Phased roadmap

| Phase | Deliverable | Exit criterion |
|---|---|---|
| **0** | Scaffold (this branch): dirs, contract header, CMake, dump script, this plan. Wire `add_subdirectory(audio)` when Phase 1 lands. | Repo configures; build green (audio target inert). |
| **1** | `audio_weight_loader` + geometry from checkpoint; **`log_mel` (K1)** ← *first kernel*. | K1 cosine > 0.999 vs `input_features`. |
| **2** | `conv_subsample` (K2) + `layer_norm` helper (K3a). | K2 cosine > 0.999 vs `conv_out`. |
| **3** | `audio_attention` (K3) + `encoder_mlp` (K4) → full 32-layer encoder. | per-layer + final-norm cosine > 0.999. |
| **4** | `stack_audio_frames` (K5) + `projector` (K6, swiglu). | `audio_embeds` cosine > 0.999. |
| **5** | `AudioFrontend` PIMPL (encode/project split) + integration test. | `test_ultravox_frontend` passes on real checkpoint. |
| **6** | Engine splice + capability/config/ownership; spec-dec-ready API. | text-only byte-identical; audio prompt decodes coherently. |

Each phase: `ctest --preset validation` for kernels, `ctest --preset integration`
end-to-end (regenerate dumps first if geometry moved).

---

## 8. Risks & open questions

- **DSP fidelity (K1).** Matching `WhisperFeatureExtractor` bit-for-bit (window,
  mel filterbank, log/normalize). Mitigation: dump the filterbank matrix + window
  from the processor and load as constants rather than reconstruct on device.
- **Variable-length audio.** `ModifiedWhisperEncoder` supports non-30 s inputs;
  decide whether Phase 1 caps at one 30 s window or handles arbitrary length +
  the `StackAudioFrames` padding-to-%8 at the tail.
- **Bidirectional attention footprint.** `[T,T]` scores per head × 20 × 32 layers;
  confirm the batch=1 desktop VRAM/latency budget; tile if needed.
- **Projector variant flag.** `projector_ln_mid` flips norm placement between
  v0.4.1 and v0.5 — always read it from config; our target is v0.5 (`True`).
- **Spec-dec projector duality.** The 8B target's hidden size (4096) and its
  projector weights are a *future* checkpoint; Phase 6 designs the two-projector
  path but only the 1B projector ships first. Keep `project(features, model)`
  parameterized by the target from day one.
- **Checkpoint + deps.** Need the Ultravox checkpoint under `BLACKWELL_MODELS_DIR`
  and `transformers` (+ trust-remote-code) + `soundfile` for dump generation.

---

### Research sources
- Ultravox model source (`UltravoxModel`, `UltravoxProjector`,
  `ModifiedWhisperEncoder`, `StackAudioFrames`) — fixie-ai/ultravox.
- Checkpoint `config.json` — `fixie-ai/ultravox-v0_5-llama-3_2-1b`.
- vLLM Ultravox implementation (cross-check of projector + merge).
