# GLM-4-9B + TurboQuant/QJL compressed KV — reconnaissance & integration blueprint

Status: **Phase 1 complete and verified end-to-end against the real checkpoint
(2026-09-24).** Phases 2–6 (TurboQuant itself) are still a plan. Produced from an audit of
`D:\Projects\BlackwellLLM`, `D:\Projects\qjl-lab` and
`D:\Projects\vllm-ascend\csrc\attention\turboquant`.

Scope note on provenance: `qjl-lab` is a **CUDA** harness (RTX 5070, `sm_120`) implementing
*True QJL 3+1* attention — a 4-bit (1 sign + 3 Lloyd-Max magnitude bits) KV store consumed
**in the projected basis**. The Ascend `turboquant` tree is the same *shape* of algorithm
(rotate → per-vector scale → nibble codebook → decode inside the attention kernel) with
different codebooks and an NPU-specific memory layout. Both are mapped below; the port
target is `qjl-lab`'s kernels, with the codec made a trait so the Ascend tables drop in
for cross-platform quality comparison.

Values marked **(†)** were taken from model documentation when this was written and have NOT
been re-checked; anything marked **(verified)** was read from
`F:/AI/models/GLM-4-9B-Chat-1M` on 2026-09-24. The one † that turned out wrong is recorded
in place: the -1M checkpoint has **4** KV heads, not 2.

---

## 0. Executive summary — the five findings that decide this project

1. **GLM-4 is one kernel away from running.** Its geometry (head_dim 128, GQA 32/4, SwiGLU,
   RMSNorm, QKV bias) is entirely inside what `blackwell_core` already does. The single
   *mathematical* gap is RoPE: GLM-4 rotates **interleaved pairs `(2j, 2j+1)` over the first
   `head_dim/2` channels**, while every RoPE kernel in `src/kernels/` is half-split
   (`(k, k+half)`). Everything else is config parsing, a fused-projection split, and two
   optional sandwich norms.
2. **There is a zero-kernel escape hatch for that gap.** q·k is invariant under a channel
   permutation applied to *both* Q and K, and interleaved↔half-split is exactly such a
   permutation. Permuting the first 64 rows of `q_proj`/`k_proj` (and their biases) at load
   time makes the existing `launch_rope_partial_inplace` bit-correct for GLM-4. Recommended
   plan: implement the honest interleaved kernel (it is ~30 lines and testable against a CPU
   reference), and keep the permutation as the fallback/cross-check.
3. **Pure 4-bit KV does not work; the exact sink+recent window is mandatory.** Your own
   `qjl-lab/reports/bench.json` is unambiguous on passkey retrieval at 32k/49k:
   `QJL m=1 (pure)` scores **0.10 / 0.00**, while `m=1 + sink4/recent128` scores **1.00 /
   1.00** for +1.5% bytes, and `m=2 (pure)` (8.25 bits) scores 1.00 / 0.80. The exact window
   is not a tuning knob — it is part of the operator, and it must be in the engine design
   from commit one.
4. **The KV-manager seam is already the right shape.** `IKVCacheManager::attention_decode()`
   encapsulates *RoPE → append → attention → context* behind one per-layer virtual call
   ([src/core/kv_cache/ikv_cache_manager.h:66](../src/core/kv_cache/ikv_cache_manager.h)).
   A third strategy — `TurboQuantKVManager` — slots in beside Continuous/Paged with no change
   to `engine.cpp`'s layer loop. The `void*` return of `get_layer_k_ptr` even anticipates a
   non-float payload.
5. **Two hot-path hazards to design out up front.** (a) `qjl_accumulate_cuda` allocates a
   `[B,Hkv,nblocks,G,R]` fp32 partials tensor **per call** and the two-pass shape
   materialises an `O(n)` scores buffer (134 MB at 1M context) — both violate the
   zero-allocation decode contract and must become manager-owned `DeviceBuffer`s, then be
   fused into an online-softmax single pass (which is precisely what Ascend's
   `turboquant_fused_decode.cpp` already is). (b) QJL storage is **incompatible with the
   existing RoPE-aware KV eviction** (`launch_kv_evict_head`): the Hadamard rotation mixes
   channels, so the "rotations about the same axis compose" argument that makes eviction
   exact no longer holds. Continuous-streaming sessions must evict whole compressed pages or
   stay on bf16.

VRAM arithmetic for GLM-4-9B-Chat-1M (40 layers, **4** KV heads, head_dim 128), all
layers, K+V. Verified against the checkpoint 2026-09-24: `multi_query_group_num` is 4, so
every figure below is double what the 2-KV-head recollection in the first draft predicted.

| KV format | B/token | 128k ctx | 1M ctx | vs fp32 |
|---|---|---|---|---|
| fp32 continuous (today's default) | 163 840 | 20.00 GiB | 160.00 GiB | 1.00x |
| bf16 paged (today's Paged mode) | 81 920 | 10.00 GiB | 80.00 GiB | 2.00x |
| **QJL 3+1, R=D=128 (m=1)** | **21 120** | **2.58 GiB** | **20.62 GiB** | **7.76x** |
| QJL 3+1, m=2 (8.25 bits) | 41 600 | 5.08 GiB | 40.62 GiB | 3.94x |

The exact sink(4)+recent(128) window costs **10.3 MiB total** across all 40 layers. GLM-4 at
128k context therefore needs ~2.6 GiB of compressed KV — but note the bf16 weights alone are
**17.7 GiB**, so *nothing* about this model is a single-12-GB-card configuration until the
weights are quantized (AWQ-int4 would be ~5.5 GiB). Today the engine runs it with layer
offloading, and the tiered RAM/NVMe pager (`docs/TIERED_KV_AND_AOT.md`) is what carries the
KV toward 1M.

---

## 1. GLM-4 architectural gap matrix

### 1.1 The matrix

| Property | Llama-3-8B (supported) | Qwen-2.5-7B (supported) | **GLM-4-9B / -1M** | Engine gap |
|---|---|---|---|---|
| Layers / hidden / intermediate | 32 / 4096 / 14336 | 28 / 3584 / 18944 | 40 / 4096 / **13696** (†) | none (config-driven) |
| Heads (Q / KV), head_dim | 32 / 8, 128 | 28 / 4, 128 | **32 / 4**, 128 (verified) | none; G=8 — inside the kernel G limits (§2.6) |
| Vocab | 128256 | 152064 | **151552** (†) | none |
| `tie_word_embeddings` | false | false | false (†) | none |
| RMSNorm eps | 1e-5 | 1e-6 | **1.5625e-07** (†) | none — but it is 2 orders below the current default; must be parsed, never defaulted |
| Norm style | plain `weight` | plain `weight` | plain `weight` | none (`norm_add_unit_offset=false`) |
| Extra norms | — | — | **`post_self_attn_layernorm`, `post_mlp_layernorm`** on the GLM-4-0414 (`glm4`) family only | **NEW**: two extra RMSNorms per layer, applied to the *sub-layer output before the residual add* |
| Attention bias | none | q/k/v bias | **q/k/v bias, no o_proj bias** (†) | none (`has_qkv_bias`) |
| q_norm / k_norm | no | no | no | none |
| Attention output gate | no | no | no | none |
| RoPE rotary span | full 128 | full 128 | **partial: 64 = head_dim × 0.5** (†) | `rotary_dim` already exists in `ModelConfig`; only the *gated* full-attn path reads it today |
| **RoPE pairing** | half-split `(k,k+64)` | half-split | **interleaved `(2j,2j+1)`** | **THE REAL GAP** — see §1.2 |
| RoPE theta | 500000 | 1000000 | **1e8** = 10000 × `rope_ratio`(10000) (verified, -1M) | none (float) |
| RoPE scaling | llama3 bands (supported) | none | none (plain theta stretch) | none |
| Softmax scale | `1/sqrt(128)` | `1/sqrt(128)` | `1/sqrt(128)` — `apply_query_key_layer_scaling` **cancels** | none, but see the trap in §1.4 |
| MLP | separate `gate_proj`/`up_proj` | separate | **fused `gate_up_proj` [2I, H]** (HF-native) / `dense_h_to_4h` (THUDM) | **NEW**: one GEMV + split, or two views into one tensor |
| Activation | SwiGLU (silu) | SwiGLU | **SwiGLU (silu)** — first half = gate, second = up | none |
| Sliding window | no | no | no | none (`use_sliding_window` still throws) |
| Max position | 8192 (+scaling) | 32768 | **1048576** (verified, `seq_length`) | none, but `max_position_embeddings` gates `build_and_validate_runtime` |

### 1.2 RoPE — the one real kernel gap

HF `Glm*` models build `inv_freq` over `head_dim * partial_rotary_factor = 64` dimensions
(32 frequencies, `inv_freq[j] = theta^(-2j/64)`), `repeat_interleave(2)` the cos/sin to 64
lanes, then rotate **adjacent** channel pairs and pass channels `[64,128)` through
untouched.

[src/kernels/full_attention.cu:64](../src/kernels/full_attention.cu) (`rope_partial_kernel`)
uses the *same 32 frequencies in the same order* — `powf(rope_theta, 2k/rotary_dim)` — but
pairs `(k, k+32)` inside the rotary block. So the two differ **only by a channel
permutation** of the first 64 lanes:

```
pi : interleaved index 2j -> j,  2j+1 -> j + 32      (identity on channels 64..127)
half_split_rope(pi(x)) == pi(interleaved_rope(x))        for every position
```

Two ways to close it, both exact:

* **(A) New kernel — recommended.** `launch_rope_interleaved_partial_inplace(...)` plus a
  fused `launch_fused_rope_interleaved_kv_kernel(...)` for the continuous path. ~30 lines
  each, mirrors the existing kernels, testable against a CPU reference at fp32, and keeps
  the checkpoint on disk byte-faithful (golden dumps compare tensor-for-tensor).
* **(B) Load-time channel permutation — zero new kernels.** Permute the first `rotary_dim`
  **rows** of `q_proj.weight`/`k_proj.weight` and their biases by `pi`. `q·k` is invariant
  (same permutation on both sides), V/o_proj are untouched, and the existing half-split
  partial kernel becomes bit-correct. Costs one load-time permutation hook the `VRAMArena`
  does not have today (it stages tensors verbatim), and makes the VRAM image differ from
  the checkpoint — which is why (A) is primary and (B) is the cross-check.

**Follow-on gap:** `launch_kv_evict_head` ([src/kernels/kv_evict.cuh](../src/kernels/kv_evict.cuh))
has **no `rotary_dim` parameter** — it re-phases all `head_dim/2` channel pairs. Run against
a partial-rotary model it corrupts the non-rotary half. GLM-4 + continuous streaming
(`docs/CONTINUOUS_STREAMING.md`) therefore needs `rotary_dim` threaded through that kernel
too, independently of TurboQuant.

### 1.3 Tensor-name maps (three checkpoint flavours)

`VRAMArena::allocate_weights_pool` is name-agnostic — it stages *every* tensor the
safetensors index lists and registers it by name, so extra tensors cost VRAM but never
fail. Only the `step_*` functions' hard-coded name strings need to match.

| Role | engine expects today | THUDM native (`chatglm`) | HF `glm` / `glm4` |
|---|---|---|---|
| prefix | `model.` | `transformer.encoder.` | `model.` |
| embeddings | `model.embed_tokens.weight` | `transformer.embedding.word_embeddings.weight` | `model.embed_tokens.weight` |
| Q/K/V | `...self_attn.{q,k,v}_proj.weight` | `...layers.N.self_attention.query_key_value.{weight,bias}` **fused [4608, 4096]** | `...self_attn.{q,k,v}_proj.{weight,bias}` |
| out | `...self_attn.o_proj.weight` | `...self_attention.dense.weight` | `...self_attn.o_proj.weight` |
| MLP in | `...mlp.{gate,up}_proj.weight` | `...mlp.dense_h_to_4h.weight` **[27392, 4096]** | `...mlp.gate_up_proj.weight` **[27392, 4096]** |
| MLP out | `...mlp.down_proj.weight` | `...mlp.dense_4h_to_h.weight` | `...mlp.down_proj.weight` |
| norms | `input_layernorm`, `post_attention_layernorm`, `model.norm` | `input_layernorm`, `post_attention_layernorm`, `transformer.encoder.final_layernorm` | same as engine + `post_self_attn_layernorm`, `post_mlp_layernorm` (`glm4` only) |
| head | `lm_head.weight` | `transformer.output_layer.weight` | `lm_head.weight` |

**Recommendation: support the HF-native `glm`/`glm4` checkpoints only.** They align with the
engine's existing naming to within two deltas (fused `gate_up_proj`, optional sandwich
norms). Supporting the THUDM native layout means a fused QKV split *and* a whole alternate
prefix scheme for no evaluation benefit. Record that as a rejected alternative, and have the
loader throw the house-style actionable error ("checkpoint is THUDM-native `chatglm`;
convert with `transformers` ≥ 4.46 or use the `-hf` variant") rather than mis-loading.

### 1.4 Traps

* **`apply_query_key_layer_scaling`** (THUDM configs): `norm_factor *= layer_number`, then
  the scores are multiplied by `layer_number` again. Net scale is `1/sqrt(head_dim)` — the
  same as everyone else. Do **not** implement a per-layer scale; do add a comment saying why
  it was deliberately ignored, or a future reader will "fix" it.
* **`rms_norm_eps = 1.5625e-07` (†)** — the current `t.value("rms_norm_eps", 1e-6f)` default
  is 6x larger. Parsed correctly it is fine; silently defaulted it shifts every norm.
* **Fused `gate_up_proj` ordering**: HF `Glm4MLP` does
  `gate, up = gate_up_proj(x).chunk(2, dim=-1)` — **first** half is the gate. The naive
  guess (up first) is silently wrong and will pass a shape check.
* **Sandwich norms (`glm4` / 0414 family)** are applied to the *sub-layer output* before the
  residual add: `h = x + post_self_attn_layernorm(attn(norm(x)))`. That inverts the
  engine's current `dispatcher.forward(..., d_residual_accum=d_X_accum)` fusion, which adds
  into the residual *inside* the projection kernel. Either drop the residual fusion for
  these layers (o_proj → scratch → norm → add) or add a norm-then-accumulate epilogue.
  **Check whether the target checkpoint is 0414 at all** — plain GLM-4-9B-Chat(-1M) is the
  older `glm`/`chatglm` architecture with no sandwich norms, and the whole item disappears.
* **`multi_query_group_num = 2`** gives G = 16 queries per KV head. That is the largest GQA
  ratio in the repo (Llama 4, Qwen 7) and it is what sizes the ported kernels' shared
  memory and block shape (§2.6).

### 1.5 Confirm before coding

```bash
python -c "import json,sys; c=json.load(open(sys.argv[1])); print(json.dumps({k:c.get(k) for k in ['model_type','architectures','hidden_size','intermediate_size','num_hidden_layers','num_attention_heads','num_key_value_heads','head_dim','vocab_size','rms_norm_eps','rope_theta','partial_rotary_factor','attention_bias','mlp_bias','tie_word_embeddings','max_position_embeddings','seq_length','multi_query_group_num','kv_channels','ffn_hidden_size','rope_ratio','layernorm_epsilon','add_qkv_bias','add_bias_linear'] if k in c}, indent=1))" F:/AI/models/GLM-4-9B-Chat-1M/config.json
```

```bash
python -c "import json,sys; ix=json.load(open(sys.argv[1]))['weight_map']; print('\n'.join(sorted(k for k in ix if 'layers.0.' in k)))" F:/AI/models/GLM-4-9B-Chat-1M/model.safetensors.index.json
```

---

### 1.6 Tokenizer and chat-template status

**The engine cannot tokenize this checkpoint, and the gap is structural.** Verified
2026-09-24 and pinned by `Glm4ChatRegression.TokenizerFactoryRejectsGlm4Checkpoint`.

| Concern | What the engine supports natively | What GLM-4-9B-Chat-1M ships | Verdict |
|---|---|---|---|
| Vocabulary file | `tokenizer.json` — HF byte-level-BPE dump (`model.vocab` + `model.merges`); the factory **throws** when absent | `tokenizer.model` — 151 329 lines of `base64(token) rank`, a **tiktoken rank file**; no `tokenizer.json` at all | **bridged** |
| Tokenizer class | `ByteLevelBpeTokenizer`, driven entirely from JSON sidecars | `ChatGLM4Tokenizer` in `tokenization_chatglm.py` (remote code), wrapping `tiktoken.Encoding` | **bridged** |
| Pre-tokenizer regex | parsed out of `tokenizer.json`'s `Split` node (`parse_max_digit_run`) | a `pat_str` literal in the Python class; `\p{N}{1,3}`, i.e. the same tiktoken family, digit-run 3 | compatible *in principle* |
| Chat template | two families only — ChatML (`<\|im_start\|>`) and Llama-3 (`<\|start_header_id\|>`); anything else **throws** | a third family: `[gMASK]<sop>` then `<\|{role}\|>{metadata}\n{content}` per message, `<\|assistant\|>` as the generation prompt (**no** trailing newline — the model emits token 198 itself) | **bridged** |
| Special tokens | resolved from `added_tokens` / `tokenizer_config.json` by name | 151329 `<\|endoftext\|>`, 151331 `[gMASK]`, 151333 `<sop>`, 151335 `<\|system\|>`, 151336 `<\|user\|>`, 151337 `<\|assistant\|>`, 151338 `<\|observation\|>` | resolvable, but only *after* a vocab loads |
| Stop set | `generation_config.json` → `config.json` → tokenizer EOS | `eos_token_id = [151329, 151336, 151338]` | **natively supported** — this part needs no work |

So the engine's id→behaviour plumbing (stop sets, position handling, sampling) is
already correct for GLM-4; what is missing is purely the text↔id boundary.

**The bridge** (`scripts/glm4_chat_regression.py`): Python owns tokenization and
detokenization, and the engine is handed nothing but token ids. The ids crossing that
boundary are the same ids a native tokenizer would produce — the script's
`verify_against_template` asserts its hand-built turns equal
`tokenizer.apply_chat_template(...)` before emitting any — so the generation being
measured is entirely the engine's.

**What native support would take**, if the tray app ever needs to run GLM-4 without
Python: (1) a tiktoken loader — decode base64 ranks into the existing `BpeModelData`
(`vocab` + implied merges by rank) and hardcode nothing else, since the pre-tokenizer
regex is the same family the factory already parses; (2) a `GlmTemplate` class beside
`ChatMLTemplate` / `Llama3Template`, ~20 lines. Neither is on the TurboQuant critical
path, which is why this is a documented bridge rather than a port.

## 2. TurboQuant digital-twin audit and translation map

### 2.1 What `qjl-lab` actually is

| File | Role | What the port needs from it |
|---|---|---|
| `qjl/lloyd_max.py` + `qjl/centroids_3bit.json` | offline 8-level Lloyd-Max codebook on the half-normal; distortion 0.009501 (SNR 20.22 dB) | **the 16 signed constants** become a `constexpr float lut[16]` |
| `qjl/projection.py` | `P_i = diag(s_i) H_D / sqrt(D)`, stacked `m` blocks, `P P^T = I` exactly | the sign vectors + the FWHT factorisation (§2.4) |
| `qjl/codec.py` | nibble = `sign<<3 \| mag(0..7)`, two coords per byte (even coord in the **low** nibble); per-token scale = `rms(x_proj)`, **rounded to fp16 before code selection** | the exact encoder, including that rounding order |
| `qjl/cache.py` | pre-allocated `[B,Hkv,S,R/2]` codes + `[B,Hkv,S]` fp16 scales, plus optional exact fp16 sink/recent windows in the **projected** basis | the storage layout and the window semantics |
| `qjl/attention.py` | the 5-step operator: project Q → scores on codes → softmax over (quantised ∪ exact) → accumulate on codes → **one** inverse projection per layer per step | the control flow of `TurboQuantKVManager::attention_decode` |
| `csrc/qjl_kernels.cu` | 3 kernels: `qjl_scores`, `qjl_accumulate`, `qjl_reduce_partials`; nibble→centroid expansion **in registers**, codes staged through shared memory with a padded `(R/8 + 1)`-word row stride | **the port target** |
| `qjl/patch.py` | HF monkey-patch; prefill attends exactly in fp16 then encodes and drops the layer's K/V | the prefill contract: **quantisation error enters at decode only** |

Measured on RTX 5070 (`reports/microbench.json`, per-operator, CUDA-graph captured, Hq=16,
Hkv=2, D=R=128 — includes the Q projection):

| n | fp16 attn | fused QJL | speedup | effective BW |
|---|---|---|---|---|
| 4 096 | 0.0213 ms | 0.0110 ms | 1.94x | 98 GB/s |
| 16 384 | 0.0638 ms | 0.0130 ms | 4.89x | 332 GB/s |
| 32 768 | 0.1213 ms | 0.0220 ms | 5.51x | 393 GB/s |
| 65 536 | 0.3688 ms | 0.0253 ms | **14.58x** | 684 GB/s (**102% of the 672 GB/s peak** — L2 hits) |

End-to-end (`reports/bench.json`, Qwen2.5-3B, 36 layers, eager PyTorch) the same operator is
*slower* than fp16 (38–58 ms/token vs 27–32 ms): the win is entirely eaten by per-layer
Python/launch overhead around a ~4 ms kernel budget. In `blackwell_core` — C++ launches,
no Python dispatcher, and a candidate for CUDA-graph capture — the microbench column is the
honest prediction, which is the whole reason to port rather than to tune the twin.

### 2.2 The Ascend triad vs the twin

Ascend registers three operators (`op_adapter/turboquant_torch_ops.h`), plus a Cube-fused
decode. The twin has the same decomposition with the cache write folded into the Python
driver:

| Ascend operator | `qjl-lab` equivalent | Ported name (proposed) |
|---|---|---|
| `npu_turboquant_rotate_q(query, pi_signs, codec_tables, hadamard16, query_rot)` | `QJLProjection.project(q)` | `launch_tq_project` |
| `npu_turboquant_reshape_and_cache(key, value, key_cache, value_cache, scale_cache, slot_mapping, pi_signs, codec_tables)` | `QJLRuntime.write_prefill/write_decode` (project → encode → `cache.write`) | `launch_tq_reshape_and_cache[_paged]` |
| `npu_turboquant_paged_attention(query_rot, key_cache, value_cache, scale_cache, block_tables, context_lens, codec_tables, workspace, ...)` | `qjl_scores` + softmax + `qjl_accumulate` + `unproject` | `launch_tq_flash_decode` (Phase B) |
| `npu_turboquant_cube_decode(..., output_stage)` — fuses the query rotation *and* the epilogue, with `kRotatedBasis / kUnrotated / kGated` output stages | — (the twin always unprojects) | the Phase-B kernel's `TqOutputStage` |
| `npu_turboquant_workspace_size(...)`, `PlanPagedAttention` / `PlanFusedDecode` (`op_host/turboquant_tiling.h`) | implicit in the launchers | **host-side `tq_flash_workspace_floats(...)` — required by the zero-alloc rule** |

Two Ascend design decisions worth importing wholesale:

* **Split-and-reduce online softmax.** `PlanPagedAttention`/`PlanFusedDecode` split a context
  above `kFusedContextLimit = 4096` into at most `kMaxSequenceSplits = 8` pieces and reduce
  partials (`kPartialMaxLane`/`kPartialSumLane` = running max + sum) — i.e. flash-style
  online softmax, no `O(n)` score buffer. It also splits for **bandwidth** above 8192 tokens
  (`kBandwidthSplitContext`, one split per `kBandwidthSplitRows = 2048` rows) because a single
  DMA stream cannot saturate the memory controller. This is exactly the Phase-B shape for
  `launch_tq_flash_decode`.
* **`Pi = D H D` and `Pi^2 = I`** (`turboquant_layout.h`, `TurboQuantOutputStage::kUnrotated`):
  the epilogue is the *same* operator as the prologue, and the rotation can be folded into
  `o_proj` offline (`kRotatedBasis`) to remove the epilogue entirely. Note the twin uses
  `P = diag(s) H` (one sign vector): `P P^T = I` but `P^2 != I`, so the engine needs both
  directions (§2.4). Folding `P^T` into `o_proj`'s weights is still available and removes one
  kernel per layer per step — a Phase-C optimisation, not a Phase-A assumption.

### 2.3 Which codec to port

| | `qjl-lab` QJL 3+1 | Ascend `KV3_FP4` | Ascend `KV4_FP8` | Ascend `KV5_FP8` |
|---|---|---|---|---|
| bits / levels | 4 / 16 (1 sign + 3 mag) | 3 / 8 | 4 / 16 | 5 / 32 |
| elems per group / bytes | 2 / 1 | 8 / 3 | 2 / 1 | 8 / 5 |
| codebook | Lloyd-Max on half-normal, ±{0.128 … 2.733} | ±{0.5,2,4,6}, gain 2.788 | ±{0.5…7.5}, gain 2.983, **affine bias 7.5** | 32 levels, gain 2.206 |
| distortion | 0.009501 | 0.038444 | 0.011543 | 0.002869 |
| scale statistic | per-token **RMS** of the projected vector, fp16 | per-vector amax folded through `1/kGain` (`scoreScale_ = scale / kGain`) | same | same |
| store layout | row-major `[.., S, R/2]` | row-major | **NZ-tiled fractal** (`NzTiledPackedByte`) for the Cube | row-major |

`KV4_FP8` and QJL 3+1 are the same bit budget with distortions within 20% of each other
(0.0115 vs 0.0095 — the Lloyd-Max table is slightly better, as it should be), so **port QJL
3+1 as the default mode** and expose the codec as a POD trait:

```cpp
// src/kernels/turboquant_codec.cuh
namespace blackwell::tq {
enum class CodecMode : int32_t { QJL3p1 = 0, KV3_FP4 = 3, KV4_FP8 = 4, KV5_FP8 = 5 };
struct CodecDesc {                 // POD: host + device, no CUDA types
    int32_t bits, levels;          // 4, 16 for QJL3p1
    int32_t elems_per_group;       // 2  (two nibbles per byte)
    int32_t bytes_per_group;       // 1
    float   gain;                  // 1.0 for QJL (RMS-normalised); Ascend kGain otherwise
    float   affine_bias;           // 0.0 for QJL
    float   distortion;            // the theoretical error floor, for the test assertions
    float   lut[16];               // signed alphabet indexed BY THE STORED NIBBLE
};
constexpr CodecDesc codec_of(CodecMode m);   // mirrors Ascend's TurboQuantModeConfigOf
} // namespace blackwell::tq
```

Only `QJL3p1` needs a kernel path in Phase A; the enum exists so `KV3_FP4`/`KV5_FP8` become a
table entry plus a pack-width template parameter later, and so an evaluation can compare the
Ascend and CUDA codecs on identical BlackwellLLM plumbing. Do **not** port the NZ-tiled
layout — it is a Cube-fractal requirement with no CUDA analogue; the CUDA equivalent concern
is the padded shared-memory row stride the twin already uses.

### 2.4 Port table: what lands in `src/kernels/`

The projection must **not** be a GEMM. With `P = diag(s) H_D / sqrt(D)` and `H` symmetric:

```
project  (x -> x P)    =  FWHT(x (*) s) / sqrt(m D)     # sign flip, then butterflies
unproject(y -> y P^T)  =  FWHT(y) (*) s / sqrt(m D)     # butterflies, then sign flip
```

(verified numerically to 2.2e-16 during this audit). That replaces two 128×128 GEMMs per head
with a 7-stage in-register/shared butterfly — the CUDA counterpart of Ascend's `H16 Mmad +
AIV butterflies`. For `m > 1`, block `i` uses sign vector `s_i`, and `unproject` sums the `m`
blocks.

| New file | Operators | Proposed launch signature (house style: `launch_*`, fp32 in/out, `cudaStream_t` last) |
|---|---|---|
| `turboquant_codec.cuh` | `CodecMode`, `CodecDesc`, `codec_of()`, `__device__` encode/decode helpers, `TqProjDesc{head_dim, expansion, proj_dim, scale, const uint32_t* d_sign_bits}` | header only, no launcher |
| `turboquant_project.cu/.cuh` | FWHT rotation, both directions (the `rotate_q` analogue) | `void launch_tq_project(const float* d_X, float* d_Xp, size_t num_tokens, size_t num_heads, int head_dim, int expansion, const uint32_t* d_signs, cudaStream_t stream = 0);` and `launch_tq_unproject(const float* d_Yp, float* d_Y, ...)` |
| `turboquant_cache.cu/.cuh` | fused **project → RMS scale → nibble encode → scatter** (the `reshape_and_cache` analogue), continuous and paged | `void launch_tq_reshape_and_cache(const float* d_K, const float* d_V, uint8_t* d_k_codes, uint8_t* d_v_codes, __half* d_k_scale, __half* d_v_scale, int start_pos, size_t num_tokens, size_t kv_heads, int head_dim, int expansion, size_t max_seq_len, const uint32_t* d_signs, blackwell::tq::CodecMode mode, cudaStream_t stream = 0);` + `_paged(..., const int32_t* d_block_table, int page, int slot, ...)` |
| `turboquant_attention.cu/.cuh` | **Phase A** faithful two-pass port of `qjl_scores_kernel` / `qjl_accumulate_kernel` / `qjl_reduce_partials_kernel`; **Phase B** fused online-softmax decode with split-K partials and the exact sink/recent window folded in | A: `launch_tq_decode_scores(const float* d_Qp, const uint8_t* d_k_codes, const __half* d_k_scale, float* d_scores, int n, size_t q_heads, size_t kv_heads, int proj_dim, size_t max_seq_len, tq::CodecMode, cudaStream_t)` and `launch_tq_decode_accumulate(const float* d_probs, const uint8_t* d_v_codes, const __half* d_v_scale, float* d_partials, float* d_Op, int n, ..., int nblocks, cudaStream_t)`. B: `launch_tq_flash_decode(const float* d_Qp, const uint8_t* d_k_codes, const uint8_t* d_v_codes, const __half* d_k_scale, const __half* d_v_scale, const float* d_exact_k, const float* d_exact_v, int sink, int recent, int n, float* d_Op, float* d_workspace, ...)` + host-side `size_t tq_flash_workspace_floats(int n, int q_heads, int kv_heads, int proj_dim)` |
| (optional) `turboquant_prefill.cu` | batched chunk variant of `reshape_and_cache`, plus exact attention over the *unquantised* chunk so quantisation enters at decode only (the twin's contract) | reuses `launch_paged_flash_attention_prefill` over a transient bf16 buffer — no new attention kernel in Phase A |

Register the new `.cu` files in the explicit `set(SOURCES ...)` list in
`src/kernels/CMakeLists.txt` — that file deliberately forbids `file(GLOB)`.

### 2.5 Memory contract

* **Codes**: `uint8[kv_heads][max_seq_len][R/2]`, even projected coordinate in the **low**
  nibble. `R % 8 == 0` (the kernel reads `uint32` words), and `R` must be a power of two for
  the Sylvester Hadamard (`R = m * 128` ✓ for GLM-4).
* **Scales**: `__half[kv_heads][max_seq_len]`, one per token per KV head per tensor. Keep
  fp16: the encoder *quantises against the fp16-rounded scale* (`codec.py`), so widening it
  to fp32 silently changes the codes at cell boundaries.
* **Exact windows**: `sink` + `recent` tokens stored **in the projected basis** at bf16,
  `[kv_heads][sink+recent][R]`. Defaults `sink=4, recent=128` — the configuration that scores
  1.00 on passkey.
* **Alignment / paging**: with `PAGE_SIZE = 16` tokens, one GLM-4 page of codes is
  `16 × 2 × 64 = 2048 B` per tensor per layer plus `16 × 2 × 2 = 64 B` of scales — both
  128B-aligned, so the compressed store is **page-compatible** with `SequenceManager`'s
  `[total_pages][kv_heads][PAGE_SIZE][dim]` addressing, with `dim` reinterpreted as bytes and
  the scale plane carried as a parallel pool. That is the migration path from Phase A
  (continuous, single sequence) to a paged compressed cache; the tiered pager
  (`TieredMemoryPager`) then works unchanged because it moves pages, not elements.
* **Workspace**: `d_partials` is `[kv_heads][nblocks<=128][G][R]` fp32 = 2.1 MB at GLM-4's
  G=16, R=128 — allocate **once** in the manager's ctor.
* **Dependencies the twin has that the port must not inherit**: `torch/extension.h`,
  `c10/cuda/CUDAException.h`, `PYBIND11_MODULE`, `TORCH_CHECK`, `at::Half`. Replace with
  `cuda_fp16.h` + `cuda_bf16.h` (already used repo-wide), `CUDA_CHECK_RETURN` at the launcher
  call sites in `src/core`, and hard launch-contract comments in the `.cuh` — the house
  convention (`paged_flash_attention.cuh`) is that contracts are **documented, not
  runtime-checked**, with validation pushed up to `build_and_validate_runtime`.
* One simplification the port gets for free: the twin takes `q_proj` as `__half` and
  immediately converts to float in shared memory. BlackwellLLM's activations are already
  fp32, so the ported kernel takes `const float*` and drops the cast.

### 2.6 Shape feasibility at GLM-4's geometry

The twin's two hard block-shape constraints, evaluated at the checkpoint's real
`Hkv=4, G=8, R=128, words=16`:

* `qjl_accumulate`: `blockDim = G * words = 128` ✓ (limit 1024 → G ≤ 64 at R=128).
* `qjl_scores` shared memory: `4*(G*R + 16) + 4*64*(words+1)` = **8.4 KB** ✓.
* `qjl_accumulate` shared memory: `4*(16 + G*64) + 4*64*17` = **6.4 KB** ✓.

GLM-4 is comfortably inside both. Qwen2.5-7B (G=7) and Llama-3-8B (G=4) are smaller still. A
model with G=64 at R=256 would break the accumulate block shape — state that as the
documented launch contract rather than discovering it later.

---

## 3. Engine modifications blueprint (file by file)

### 3.1 Config and loading

* **`include/blackwell/config.h`** — add to `ModelConfig`:
  `enum class RopePairing { HalfSplit, Interleaved };` plus
  `RopePairing rope_pairing = RopePairing::HalfSplit;` (the default keeps every existing
  checkpoint byte-identical); `bool mlp_fused_gate_up = false;` ;
  `bool has_sandwich_norms = false;`. The existing `rotary_dim`, `has_qkv_bias` and
  `norm_add_unit_offset` already cover the rest of GLM-4.
* **`src/core/config.cpp`** — in `load_from_json`: recognise `model_type` in
  `{"glm","glm4","chatglm"}`; read `partial_rotary_factor` from the **document root** (GLM
  puts it there, not under `rope_parameters`); set `rope_pairing = Interleaved`,
  `mlp_fused_gate_up = true`, `has_sandwich_norms = (model_type == "glm4")`; map the THUDM
  spellings (`add_qkv_bias`, `layernorm_epsilon`, `kv_channels`, `ffn_hidden_size`,
  `multi_query_group_num`, `rope_ratio`) onto the canonical fields; throw the actionable
  error for `chatglm`-native tensor naming (§1.3). Keep the `use_sliding_window` guard.
* **`src/core/memory_pool.cpp`** — no change. The arena is name-agnostic; the extra GLM
  tensors load and register for free.

### 3.2 Kernels

* **`src/kernels/full_attention.cu/.cuh`** — add `launch_rope_interleaved_partial_inplace`.
* **`src/kernels/kv_evict.cu/.cuh`** — thread `rotary_dim` through `launch_kv_evict_head`
  (the independent streaming fix from §1.2).
* **`src/kernels/rope.cu/.cuh`** — add the interleaved variant of the **fused** rope+KV-append
  used by the continuous path, or route GLM-4 through the in-place kernel + `launch_kv_append`
  pair (one extra launch per layer; measure before optimising).
* **`src/kernels/turboquant_*.{cu,cuh}`** — §2.4.
* **`src/kernels/CMakeLists.txt`** — extend the explicit `SOURCES` list.

### 3.3 The new KV strategy

New files `src/core/kv_cache/turboquant_kv_manager.{h,cpp}` implementing `IKVCacheManager`:

```cpp
class TurboQuantKVManager : public IKVCacheManager {
public:
    TurboQuantKVManager(const ModelConfig&, const blackwell::RuntimeConfig&);  // INIT tier: throws
    void prepare_decode_step(SeqId seq, int pos) override;       // single-sequence: seq must be 0
    void prepare_prefill_step(SeqId, int, int) override;
    void attention_decode(int layer, int pos, float* dQ, float* dK, float* dV, float* dO) override;
    void attention_prefill(int layer, int start, int n, float* dQ, float* dK, float* dV, float* dO) override;
    void* get_layer_k_ptr(int layer) override;   // uint8 code plane (void* is already the contract)
    void* get_layer_v_ptr(int layer) override;
    void fork(SeqId, SeqId) override;            // throws: "compressed KV has no CoW; use KVCacheMode::Paged"
    void rewind(SeqId, int target_pos) override; // supported: position-addressed, bookkeeping only
    const char* name() const override { return "turboquant"; }
    bool supports_branching() const override { return false; }
};
```

`attention_decode` body (per layer, all launches on the compute stream, **zero allocation**):

1. `launch_rope_{interleaved_,}partial_inplace(d_Q / d_K)` — RoPE stays in the original basis.
2. `launch_tq_reshape_and_cache(d_K, d_V, …, pos, 1, …)` — project + encode + scatter, and
   refresh the exact sink/recent windows (the `push_recent` slide).
3. `launch_tq_project(d_Q, d_Qp, 1, q_heads, …)`.
4. Phase A: `launch_tq_decode_scores` → mask the sink/recent span → exact GEMV over the
   windows → softmax → `launch_tq_decode_accumulate` → reduce.
   Phase B: one `launch_tq_flash_decode`.
5. `launch_tq_unproject(d_Op, d_O, …)` — one epilogue per layer per step.

Members, all `blackwell::DeviceBuffer<T>` sized once in the ctor (declaration order =
dependency order, per extension pattern #3): `d_k_codes`, `d_v_codes` (`uint8`), `d_k_scale`,
`d_v_scale` (`__half`), `d_exact_k`, `d_exact_v`, `d_signs` (`uint32` sign bitmask, `m*D`
bits), `d_Qp`, `d_Op`, `d_scores`, `d_probs`, `d_partials`, `d_workspace`. Nothing is
allocated after construction.

### 3.4 Runtime config, capabilities, validation

* **`include/blackwell/engine.h`** — `enum class KVCacheMode { Continuous, Paged, TurboQuant };`
  (append — the enum crosses the DLL boundary, so appending is the ABI-safe move).
* **`include/blackwell/runtime_config.h`** —
  tier 2 `InferenceConfig`: `bool compress_kv = false; int kv_compression_expansion = 1;
  int kv_exact_sink = 4; int kv_exact_recent = 128;`
  tier 3 `RuntimeConfig`: resolved `tq_mode` (`tq::CodecMode`), `tq_expansion`,
  `tq_sink_tokens`, `tq_recent_tokens`, `tq_proj_seed`.
  `RuntimeOverrides`: `std::optional<>` twins of each (the `-x264-params` seam — no new loose
  constructor arguments anywhere).
  `KernelLimits`: `kTurboQuantMaxGroupsTimesWords = 1024`, `kTurboQuantProjDimMax = 256`.
* **`src/core/runtime_config.cpp`** — in `build_and_validate_runtime`, add a `TurboQuant`
  branch that throws actionable `std::invalid_argument` / `std::runtime_error` when:
  `head_dim` is not a power of two (Sylvester Hadamard); `R = expansion * head_dim` violates
  `R % 8 == 0` or the `G * R/8 <= 1024` block shape; `require_branching` is set (→ "compressed
  KV has no CoW fork; use KVCacheMode::Paged or drop require_branching");
  `sink + recent > max_seq_len`; **`kv_exact_recent == 0`** — refuse a pure-4-bit plan by
  default, with a message citing the measured passkey collapse, and require an explicit
  `RuntimeOverrides` opt-out. That is the one place the §0-item-3 finding becomes a checked
  invariant instead of a footnote.
* **`derive_capabilities` / `ModelCapabilities`** — add `bool supports_compressed_kv`, and keep
  `supports_cow_branching = false` for `TurboQuant` (finalised in the engine ctor exactly as
  it is today for Continuous).

### 3.5 Engine

* **`src/core/engine_impl.h`** — no new members if all TurboQuant state lives in the manager
  (preferred: it keeps the composition root honest and the destruction order trivial). If
  `d_Qp`/`d_Op` must be shared with the hybrid path, declare them *after* `arena` and *before*
  `kv_mgr`, with the usual "why here" comment.
* **`src/core/engine.cpp`** —
  (a) the composition root: a third `case KVCacheMode::TurboQuant:` building
  `TurboQuantKVManager`;
  (b) `step_attention_math` is **unchanged** — that is the point of the seam;
  (c) `step_mlp_projections`: when `mlp_fused_gate_up`, one
  `dispatcher.forward(prefix + "gate_up_proj", d_X_norm, d_GateUp, 2*intermediate_dim,
  hidden_dim)` and then point `launch_fused_swiglu_kernel` at the two halves of that buffer
  (gate = `d_GateUp`, up = `d_GateUp + intermediate_dim`) — no split kernel, no copy;
  (d) sandwich norms (only if the target is the 0414 family): `step_attention_out` /
  `step_mlp_out` must stop fusing the residual add into the projection for those layers;
  (e) a `require_compressed_kv(caps, op)` gate in the style of the existing
  `require_branching` / `require_prefix_cache` helpers.
* **`src/core/engine_prefill_coordinator.*`** — Phase A can run the prompt through the
  existing per-token path; the coordinator's COMPUTE phase reports via the `status` fields on
  `Result`/`UpdateStats` and needs no new vocabulary. Chunked compressed prefill is Phase C.

### 3.6 Memory pool / paging

* Phase A: the compressed store lives **in the manager**, not in `VRAMArena` — the arena's
  fp32 `d_k_cache`/`d_v_cache` are simply not allocated for this mode. That needs one guard in
  `allocate_dynamic_pool` (skip the KV slabs when the plan says TurboQuant) so the 10 GiB fp32
  slab is not reserved and immediately wasted; without it a 128k plan OOMs before the
  compressed cache is even built.
* Phase C: move the code/scale planes into `paging::SequenceManager` as a third pool pair so
  CoW fork, the radix prefix cache, `.bkv` serialisation and the tiered RAM/NVMe pager apply
  unchanged (§2.5 shows the page geometry works out).
* `hibernate()`/`wakeup()`: the compressed planes are pure device memory with no host mirror
  in Phase A, so they must either be evacuated alongside the weights or explicitly documented
  as dropped on hibernate (a re-prefill on wake). Decide this deliberately — the overlay's
  hibernation path is a shipped feature.

### 3.7 Doctrine compliance checklist

| Doctrine | How the design satisfies it |
|---|---|
| Single-threaded control plane | The manager adds no threads and no streams; every launcher takes `cudaStream_t stream = 0` and is called from the engine-owning thread. `EngineCom`'s `BLACKWELL_VERIFY_OWNING_THREAD()` covers the new mode for free — no new COM methods are needed. |
| Hybrid error doctrine | Ctor + `build_and_validate_runtime` = INIT tier, `throw` (`cuda_error` / `std::invalid_argument`); `attention_decode` and every launcher = RUNTIME tier, `CUDA_CHECK_RETURN` and `EngineStatus` returns, no exceptions, **no new `EngineStatus` values** (`OutOfVram` / `InvalidConfig` / `CudaRuntimeError` already cover it, so `hresult_from_status` in `engine_com.cpp` is untouched). |
| Zero allocation on the decode hot path | Every buffer is a ctor-time `DeviceBuffer`; `nblocks` is capped at 128 and `d_partials`/`d_workspace` are sized from `tq_flash_workspace_floats(max_seq_len, …)` at construction. **This is the single biggest deviation from the twin** — `qjl_accumulate_cuda` calls `torch::empty` twice per layer per token. |
| Capability gating | `fork` / `release_sequence` / batched decode throw the house-style "what to do instead" messages; `supports_branching() == false`. |
| Tiered configuration | Intent (`compress_kv`) → validated plan (`tq_*`) → members sized from the plan, never from the raw request. |
| Declaration order = dependency order | Manager members ordered codes → scales → windows → signs → scratch → workspace, each with a "why here" comment. |
| Zero-warning `/W4 /WX` | New `.cpp`/`.h` are first-party CXX and must compile clean; `.cu` is exempt. Watch `size_t`→`int` narrowing at the launcher boundaries (the repo's most common C4267). |

### 3.8 CMake

`src/kernels/CMakeLists.txt` (explicit source list) and `src/core/CMakeLists.txt`
(`CORE_SOURCES += kv_cache/turboquant_kv_manager.cpp`). No new targets, no new include
directories, no new dependencies — per skill `cmake-hygiene`.

---

## 4. Validation strategy

The house pattern is: CPU reference in `tests/common/*.h` → tolerance-checked kernel test in
`tests/validation/` → PyTorch golden dump compared end-to-end in `tests/integration/`.
Bit-exactness is asserted only where it is genuinely achievable (integer / packing paths);
everything with float reductions gets a documented tolerance.

### 4.1 `tests/common/` — new CPU references

* `turboquant_reference.h`: `tq_hadamard_project` / `tq_unproject` (naive `x @ P` with the
  explicit matrix — the independent check on the FWHT kernel), `tq_encode_nibbles` (including
  the fp16 scale-rounding order), `tq_decode_values`, `tq_decode_attention` (the full 5-step
  operator in fp64), and `tq_bitplane_attention` (the `{-1,0,+1}` masked-popcount form — the
  invariant-strict reference that never materialises a centroid value, ported from the twin's
  `bitplane` backend).
* `cpu_reference.h` additions: `cpu_rope_interleaved_partial` beside the existing half-split
  reference, plus `cpu_rope_permutation_equivalence` asserting the §1.2 identity.

### 4.2 `tests/validation/` — new suites (label `validation`, fast, no checkpoint)

| File | Asserts | Tolerance |
|---|---|---|
| `test_rope_interleaved.cpp` | interleaved partial RoPE vs the CPU reference at GLM-4 shapes (`Hq=32/Hkv=2`, `D=128`, `rot=64`) at `pos` ∈ {0, 1, 7, 4095, 131071}; `pos=0` is identity; and the permutation identity of §1.2 | `1e-5` relative (fp32 `sincosf`) |
| `test_turboquant_codec.cpp` | encode→decode round-trip SNR on `N(0,1)` lands on the Lloyd-Max floor (**distortion 0.009501 ⇒ 20.22 dB**, assert > 20.0); nibble packing is **bit-exact** vs the reference packer; the 16-entry LUT matches `centroids_3bit.json` to `1e-12`; `bits_per_source_element == 4.125` | exact for packing, 0.2 dB for SNR |
| `test_turboquant_projection.cpp` | FWHT `project`/`unproject` vs the explicit-matrix reference; `max abs(P P^T - I) < 1e-6`; `unproject(project(x)) == x`; `m=2` stacking | `1e-5` |
| `test_turboquant_attention.cpp` | the ported kernels vs (a) the fp64 CPU operator, (b) the bitplane reference, (c) exact fp32 attention with the per-score RMS error asserted **at the theoretical floor** (< 11%, expect ~9.7%); `n` ∈ {1, 63, 64, 127, 1000, 8192} (the twin's non-tile-multiple cases); sink/recent masking correctness | `5e-3` relative Frobenius (the twin's own threshold) plus the RMS ceiling |
| `test_turboquant_zero_alloc.cpp` | drives 64 decode steps and asserts device free memory is **unchanged** after warm-up (`cudaMemGetInfo`) — the hot-path allocation invariant made executable | exact |
| `test_runtime_config.cpp` (extend) | the new validator branches: non-power-of-two `head_dim`, `G*R/8 > 1024`, `require_branching + TurboQuant`, `recent == 0` without the explicit override, `sink+recent > max_seq_len` | message-substring assertions, as the file already does |

### 4.3 `scripts/generate_glm4_dumps.py` — golden dumps

Model it on `scripts/generate_qwen_dumps.py`: `BLACKWELL_MODELS_DIR`-anchored default model
dir, out dir anchored to the **repo root** (`tests/integration/golden_dumps/glm4_9b/`),
single-token BOS-style prompt for the linear-stack parity plus a short real prompt for the
attention path. Dump, per probed layer (`0`, `10`, `20`, `39`) in fp32 little-endian `.bin`
(the existing convention):

```
embed_out.bin
layer_N_input_norm.bin
layer_N_{q,k,v}_proj.bin        # pre-RoPE, post-bias
layer_N_{q,k}_rope.bin          # POST interleaved RoPE  <-- the new gap's witness
layer_N_attn_math.bin           # context, pre-o_proj
layer_N_attn_out.bin  layer_N_accum_out.bin
layer_N_post_attn_norm.bin
layer_N_gate_up.bin             # the fused [2I] tensor, so the split order is witnessed
layer_N_mlp_out.bin
final_norm_out.bin  logits.bin
# TurboQuant-specific (a second script, or a --turboquant flag):
tq_signs.bin                    # the m*D sign vector, so C++ and Python share one projection
tq_lut.bin                      # the 16 signed centroids
layer_N_k_codes.bin  layer_N_k_scales.bin    # bit-exact encoder witness
layer_N_tq_context.bin                       # the 5-step operator's output, fp32
```

Emitting `tq_signs.bin` / `tq_lut.bin` as *inputs* to both sides is what makes the comparison
a test of the kernels rather than of two independent RNGs — the same discipline
`generate_qwen_dumps.py` uses when it dequantises AWQ with the engine's exact convention.

### 4.4 `tests/integration/` (label `integration`, skips when the checkpoint is absent)

* `test_glm4_engine.cpp` — the `test_qwen_engine.cpp` pattern: per-stage cosine > 0.999 and
  max-abs deltas against the dumps, walking embed → norms → projections → **RoPE** →
  attention → MLP → logits. This is the test that proves the interleaved-RoPE gap is closed.
* `test_turboquant_decode_equivalence.cpp` — one engine in `Continuous` (or `Paged`) mode and
  one in `TurboQuant` mode over the same prompt: assert top-1 agreement over N greedy steps
  and a KL bound on the logits, and report the KV byte ratio. The honest end-to-end quality
  gate.
* `test_turboquant_passkey.cpp` — the twin's passkey harness (`tools/passkey.py`) ported to the
  engine, at 8k/32k/128k and depths {0, .25, .5, .75, 1}. **Assert the pure-4-bit collapse and
  the sink/recent recovery** so the §0-item-3 finding cannot silently regress.
* Extend `test_kv_evict_logit_equivalence.cpp` for `rotary_dim`, and add an explicit
  "TurboQuant + eviction is refused" case.

### 4.5 Parity-vs-tolerance policy (state it in the PR; it will be asked)

Bit-exact: nibble packing/unpacking, the LUT table, the sign vector, slot addressing.
Tolerance-checked: everything downstream of a float reduction — the FWHT (`1e-5`), the
operator vs fp64 (`5e-3` Frobenius), engine-vs-PyTorch (cosine > 0.999, as the repo already
does). *Not* asserted: agreement with fp32 attention beyond the Lloyd-Max floor — the ~9.7%
per-score RMS error is the design, and a test that demands less is a test that will be
"fixed" by disabling compression.

---

## 5. Phase-by-phase roadmap

Each bullet is one reviewable commit. Phases 1 and 2 are independent and can land in either
order; Phase 3 needs both.

**Phase 0 — confirm the target (no code)**

1. Run the §1.5 one-liners against the actual GLM-4 checkpoint; record the real `rope_theta`,
   `rms_norm_eps`, `partial_rotary_factor`, layer count, and whether the checkpoint is `glm`
   (no sandwich norms) or `glm4`/0414 (sandwich norms). Decide the quantisation flavour
   (BF16 / AWQ-int4 / FP8) — that decides which existing dispatcher path GLM-4 rides and
   whether `weight_scale` / `qzeros` naming lines up.

**Phase 1 — GLM-4 geometry, no compression** — steps 2, 3, 4 and 6 **landed 2026-09-24**.

2. **[done]** `config.h` / `config.cpp`: GLM family recognition, `RopePairing`,
   `mlp_fused_gate_up`, `has_sandwich_norms`, the THUDM parameter aliases, the
   `chatglm`-native rejection error, + 5 `ConfigLoaderGlm4` cases in
   `tests/validation/test_runtime_config.cpp`.
3. **[done]** `cpu_rope_interleaved_partial` reference +
   `launch_rope_interleaved_partial_inplace` (and its batched prefill sibling) +
   `tests/validation/test_rope_interleaved.cpp` — 5 cases including the permutation
   identity run kernel-against-kernel and the resulting q·k equality.
4. **[done]** Fused `gate_up_proj` (one buffer, two views — plus a strided SwiGLU for the
   batched rows) and the sandwich norms (`launch_rmsnorm_accum_kernel`) across all three
   forward paths. The **RoPE pairing dispatch lives in the KV managers, not in
   `step_attention_math`** — that function only delegates to `IKVCacheManager`, and RoPE is
   inside `attention_decode`/`attention_prefill`, so the dispatch had to go where the
   rotation actually happens (§3.3's seam, unchanged).
5. **[done]** `scripts/generate_glm4_dumps.py` (reference forward over the checkpoint's own
   modeling code, cross-checked against its `forward()` at cosine 0.99999654) +
   `scripts/convert_glm4_thudm_to_hf.py` (the THUDM tree is not loadable: rename + fused-QKV
   row split, no arithmetic) + `tests/integration/test_glm4_engine.cpp` on the shared
   `tests/common/engine_test_harness.h`.
   **Gate MET: every probed stage ≥ 0.9997 cosine, q_rope/k_rope ≥ 0.99993, attention entropy
   within 0.21%, final logits cosine 0.99993 with an identical top-5 and exact top-1.**
6. **[done]** Streaming fix: `launch_kv_evict_head` now takes `rotary_dim` **and the
   pairing**. `rotary_dim` alone would have left a kernel that looks GLM-4-ready while still
   re-phasing the wrong channel couples inside the rotary block; both are covered by new
   `KvEvictValidation` cases that build the cache through the interleaved append path.

**Phase 2 — TurboQuant primitives (no engine wiring)**

7. `turboquant_codec.cuh` (traits, LUT, `codec_of`) + `turboquant_reference.h` +
   `test_turboquant_codec.cpp`. **Gate: the encoder is bit-exact vs the reference and hits
   20.2 dB.**
8. `turboquant_project.cu` (FWHT both directions) + `test_turboquant_projection.cpp`.
9. `turboquant_cache.cu` (fused project + encode + scatter, continuous layout).
10. `turboquant_attention.cu` — faithful two-pass port of the three twin kernels +
    `test_turboquant_attention.cpp` against the fp64 and bitplane references.
    **Gate: `5e-3` Frobenius vs the CPU operator at every `n` in the twin's parameter set.**

**Phase 3 — engine integration (Continuous, single sequence)**

11. `KVCacheMode::TurboQuant` + the tier-2/tier-3 knobs + the validator branch (including the
    `recent == 0` refusal) + `derive_capabilities`.
12. `TurboQuantKVManager` with ctor-time `DeviceBuffer`s, the exact sink/recent windows and the
    `push_recent` slide; the `allocate_dynamic_pool` guard that stops reserving the fp32 KV
    slab. + `test_turboquant_zero_alloc.cpp`.
13. Composition root in `engine.cpp`. **Gate: first working GLM-4 + TurboQuant decode step,
    with the KV byte ratio printed and matching the §0 table.**
14. `test_turboquant_decode_equivalence.cpp` (top-1 agreement vs bf16 over N steps).

**Phase 4 — quality and the headline claim**

15. `test_turboquant_passkey.cpp` at 8k/32k/128k × 5 depths; the pure-vs-windowed collapse
    assertion.
16. A `tests/benchmarks/` entry reproducing the microbench table inside the engine:
    per-operator CUDA-event timings vs the bf16 paged path at n ∈ {4k, 16k, 32k, 64k, 128k},
    plus whole-step ms/token and peak VRAM.
    **Gate: the 5–14x operator speedup survives the port, or we learn why it does not.**

**Phase 5 — fuse and page**

17. `launch_tq_flash_decode`: online-softmax single pass with split-K partials (Ascend's
    `PlanFusedDecode` tiering: `kFusedContextLimit = 4096`, `kMaxSequenceSplits = 8`, the
    bandwidth split at 8192/2048) and the exact windows folded in. Deletes the `O(n)` scores
    buffer — the prerequisite for 1M context.
18. Move the code/scale planes into `paging::SequenceManager` as a third pool pair; CoW fork,
    the radix prefix cache, `.bkv` and the tiered pager come along for free. Flip
    `supports_branching()`.
19. Chunked compressed prefill through `EnginePrefillCoordinator`.

**Phase 6 — the evaluation this is all for**

20. `KV3_FP4` / `KV5_FP8` codec-trait entries + a sweep harness: (codec × expansion ×
    sink/recent × context) → passkey accuracy, logit KL vs bf16, ms/token, peak VRAM. This is
    the table that answers "does low-bit KV + FP8/AWQ weights hold up on GLM-4-9B-1M".
21. Docs: fold the outcome into this file, `docs/03_memory_and_kv_modes.md` (a third KV mode),
    `CLAUDE.md` (project map + roadmap register), and the `<evolution_protocol>` blocks of
    skills `engine-extension` and `cmake-hygiene`.

---

## 6. Risks and open questions

1. **Does the operator speedup survive without CUDA graphs?** The twin's 14.6x is
   graph-captured; its eager numbers are ~10x worse and its end-to-end decode is *slower* than
   fp16. `blackwell_core` launches from C++ (no Python dispatch) but does not capture graphs.
   Phase 4 step 16 exists to answer this before Phase 5 is funded. Mitigation: the fused
   Phase-5 kernel collapses ~6 launches per layer into 2.
2. **1M context is a paging problem, not a compression problem.** 10.3 GiB of codes at 1M
   exceeds a 12 GB card once weights are resident. The compressed cache must land in the
   tiered pager (Phase 5 step 18) for the "-1M" in the model name to mean anything.
3. **RoPE-aware eviction is lost.** Re-phasing does not commute with the Hadamard rotation, so
   `launch_kv_evict_head` cannot operate on codes. Continuous-streaming sessions
   (`docs/CONTINUOUS_STREAMING.md`) must either drop whole compressed pages or stay on bf16.
   This needs an explicit product decision, since the overlay translator depends on it.
4. **Prefill still needs exact K/V.** The twin attends exactly in fp16 during prefill and
   encodes afterwards, so quantisation error enters only at decode. That requires one layer's
   worth of unquantised K/V live at once (for GLM-4-9B at 128k: `2 × 2 × 128 × 131072 × 2 B`
   = 134 MB for one layer). Budget it, or accept quantisation during prefill and re-measure
   quality.
5. **Codebook / projection provenance must be pinned.** The sign vector (`seed = 0x5EED`), the
   LUT and the scale statistic are part of the *cache format*: a `.bkv` file or an AOT-warmed
   prefix written with one projection is garbage under another. If Phase 5 step 18 lands,
   `compute_kv_model_hash` (`src/core/engine.cpp`) must absorb the codec mode, the expansion
   and the projection seed.
6. **`qjl-lab`'s `reports/REPORT.md` (referenced by its README) does not exist** — only the raw
   JSON. Anyone re-deriving the quality claims should regenerate it (`python tools/report.py`)
   rather than trusting a summary from memory.
