# E8W5 in BlackwellLLM — architecture audit, integration, and measurements

Native engine support for the 5-bit companded-E8 lattice format
([`docs/E8W5_FORMAT_SPEC.md`](E8W5_FORMAT_SPEC.md)): what the audit found, what was built on
top of it, and what the numbers actually are.

| Piece | Location |
|---|---|
| Device primitives | [`src/kernels/e8w5_dequant.cuh`](../src/kernels/e8w5_dequant.cuh) |
| GEMV + dequantize kernels | [`src/kernels/e8w5_linear.cu`](../src/kernels/e8w5_linear.cu) / [`.cuh`](../src/kernels/e8w5_linear.cuh) |
| Strategy + dispatch | `include/blackwell/config.h`, `src/core/config.cpp`, `src/core/ops_dispatcher.cpp` |
| Weight residency | `src/core/memory_pool.{h,cpp}` (`E8W5TensorPtrs`, `get_e8w5_pointers`) |
| Offline converter | [`scripts/e8w5_convert.py`](../scripts/e8w5_convert.py) |
| Correctness | [`tests/validation/test_e8w5_linear.cpp`](../tests/validation/test_e8w5_linear.cpp) |
| Benchmark | [`tests/benchmarks/bench_e8w5_gemv.cpp`](../tests/benchmarks/bench_e8w5_gemv.cpp) |

---

## 1. Audit: the five facts that determined the design

**1.1 Activations are FP32 end to end.** `LinearDispatcher::forward` takes
`const float* d_in` and writes `float* d_out`; every GEMV in `src/kernels/` consumes raw
FP32 and accumulates in FP32, and `bf16_linear.cu` comments that this is deliberate
("Пишем чистый FP32 без искусственного урезания до BF16"). The golden-dump integration
tests pin that precision tensor-for-tensor.

*Consequence:* the "A16" in E8W5-A16 does not describe this engine, and the brief's
`__hfma2` inner loop does not fit it. The decode hands back FP32 and the dot product uses
`fmaf`. Converting activations to FP16 to use `__hfma2` would change numerics the golden
dumps pin, to save arithmetic that is not the bottleneck (§4).

**1.2 Two different GEMV shapes already exist, and the format picks one.**
`bf16_linear.cu` is warp-per-output-row: lanes stride along K, `__shfl_down_sync` reduction,
weights laid out `W[out_features][in_features]`. `awq_linear.cu` is thread-per-8-output-features
with a K-major `qweight[in_features][out_features/8]` layout, because that is AutoAWQ's
checkpoint convention.

*Consequence:* E8W5 must follow the BF16 shape, and it is not a preference. An E8 block is 8
coordinates of **one** output row, so there is nothing to coalesce along `out_features` — the
AWQ shape has no analogue here. E8W5 therefore stores `[out_features][in_features/8]`,
matching the engine's dominant convention; AWQ is the outlier.

**1.3 Dispatch is a single strategy switch.** `ModelConfig::quant_strategy` selects the
backend in `forward_row` (batch=1) and `forward` (batched). `forward()` routes
`num_tokens < batched_gemm_threshold` (default 16) to a per-row GEMV sweep and above it to a
Tensor-Core GEMM; `COMPRESSED_TENSORS_INT4` has no batched kernel and always sweeps.

*Consequence:* adding a format is one enum value plus two switch cases, and there is
precedent for shipping a GEMV-only strategy.

**1.4 The arena is dtype-agnostic but name-driven.** `VRAMArena` walks every tensor in the
safetensors metadata and copies `entry.byte_size` bytes at 16-byte alignment — it never
interprets dtype. Access is by name (`get_weight_ptr`), with `get_quantized_pointers` hard-wired
to AWQ's `.qweight/.scales/.qzeros` triple and gated on `quant_method`. Layers at or above
`num_gpu_layers` are packed per layer into pinned host RAM and streamed through two ping-pong
device slots.

*Consequence:* new tensors need no loader changes, only a new accessor. And because offloading
packs *by layer name*, a projection's four E8W5 tensors land in the same layer block
automatically — the offload path works without being taught about the format.

**1.5 GLM-4's biased q/k/v do not bypass the dispatcher when quantized.** `step_qkv_projections`
has a fused BF16 GEMV+bias fast path, but it is gated on `quant_strategy == NONE`; any
quantized strategy routes through `dispatcher.forward()` and then
`launch_fused_qkv_bias_kernel`, with `BiasDType::BF16` for everything except AWQ.

*Consequence:* no separate biased-qkv work was needed, and a BF16-source checkpoint's biases
copy through unchanged and are read at the right dtype.

Build context: the VS 2022 generator with `CMAKE_CUDA_ARCHITECTURES=120-real` — the engine
builds **SM120 only**, so the brief's SM80/89/90 targets are cross-compilation checks here
rather than shipped configurations (§4 reports all four). `src/kernels/CMakeLists.txt` is an
explicit source list, deliberately not a GLOB.

---

## 2. What was built

**Strategy.** `QuantStrategy::E8W5_LATTICE`, resolved from a `quantization_config` with
`quant_method="e8w5"`, `bits=5`. `group_size` is validated to be exactly 128 and rejected
otherwise: 128 weights is 16 E8 blocks is 640 bits, the smallest unit at which a 5-bit stream
is 16-byte aligned, and the kernel indexes scales off that assumption (`gb >> 4`). It is a
property of the bit layout, not a tunable.

**Residency.** `E8W5TensorPtrs {plane_lo, plane_hi, scales, codebook}` and
`VRAMArena::get_e8w5_pointers()`, mirroring `get_quantized_pointers` including the
throw-on-missing behaviour — a half-populated projection would decode to plausible garbage
silently, which is worse than failing the load.

**Kernels.** `launch_e8w5_gemv_kernel`, `launch_e8w5_gemv_residual_kernel` (o_proj / down_proj
land on the residual stream), and `launch_e8w5_dequantize` (off the hot path; it exists so the
test can compare the device decode against the reference elementwise, and so a batched path can
be built before a fused MMA kernel exists).

**Converter.** `scripts/e8w5_convert.py` streams a HF checkpoint tensor by tensor, quantizes
the six projections per layer on the GPU, copies everything else through byte-faithfully, and
writes shards at a byte budget so the resident set stays at roughly one shard. Planes are
emitted as **int32**, not uint32/uint8: AWQ already ships I32 planes and F16 scales through this
loader, so those are the dtypes the path is known to carry, and the bytes are identical either
way (`plane_hi` as `int32[in/32]` is the same little-endian image as `uint8[in/8]`, which is
exactly what the kernel reads).

It lives in `scripts/`, not the `tools/` the brief suggested, because `scripts/` is this repo's
Python home (CLAUDE.md) and `src/tools/` holds C++ GUI tools; adding a third root directory for
one file would contradict the roadmap item that cleaned the root up. It is the one place a
shipping artifact imports from `research/`, so that the quantizer in the converter is literally
the one the research measured — `src/` and `tests/` still never do.

---

## 3. The performance bug worth documenting, because it cost 24x

The first working kernel followed the brief's instruction to use 128-bit `uint4` loads for
plane L. It was **correct and 24x off the bandwidth floor**: 379 µs for a 4096x4096 projection
whose 10.5 MB of weights should take 15.6 µs at 672 GB/s, at 4.2% of peak.

The cause is a chain that starts in the format itself:

1. u_7's low bit is recovered from the parity of the other seven coordinates, so **one lane
   must own a whole block** — this is the same constraint that forces the MMA permutation in
   spec §9.2.
2. A `uint4` of plane L is 4 consecutive blocks, so a lane owns **32 consecutive activations**.
3. Consecutive lanes are then 128 B apart in `x`, so **every x load instruction touches 32
   distinct cache lines instead of one**.

The weight loads were perfectly coalesced the whole time; the activation loads were a 32-way
scatter. Two corrections:

- **Drop `uint4` for the GEMV.** One block per lane, `uint32` loads. This is not a concession:
  32 lanes x 4 B is already a full 128 B transaction per instruction, so the 128-bit load buys
  no additional coalescing here — it only buys memory-level parallelism, which 8 warps per block
  across 512 blocks already supply. (`uint4` remains right for a *batched* kernel reading the
  MMA-permuted pack, where a lane's 8 coordinates are not 8 consecutive k.)
- **Stage `x` in shared memory, padded to 9 floats per 8-float group.** A lane reads
  `xs[9*l + i]`; 9 is coprime with 32 banks, so every lane hits a different bank for every `i`
  — conflict-free. Padding to the natural 8 would put all 32 lanes in the **same** bank, a
  32-way conflict, which is the trap this layout sets.

Then a second, smaller fix: `e8w5_block_dot8` is an 8-deep dependent FMA chain fed by
data-dependent shared loads, so one accumulator left the warp latency-bound. Two independent
accumulator chains took the largest projection from 427 µs to 333 µs.

| stage | q_proj 4096x4096 | gate_up 4096x27392 | down 13696x4096 |
|---|---|---|---|
| `uint4`, x from global | 379.6 µs | 1526.2 µs | 951.5 µs |
| 1 block/lane, x in shared | 88.2 µs | 427.3 µs | 257.1 µs |
| + dual accumulators | **85.9 µs** | **333.4 µs** | **248.2 µs** |

---

## 4. Measurements

### 4.1 Correctness (`ctest -L validation`, or `--gtest_filter=*E8W5*`)

11 tests, all passing; the full 148-test validation suite passes unchanged after the config,
arena and dispatcher edits.

| Claim | Result |
|---|---|
| Device dequantization is **bit-exact** vs an independent C++ reference unpacker, K in {4096, 11008, 13696, 14336} | pass (exact float equality, not a tolerance) |
| GEMV vs FP64 accumulation of the same decoded weights | pass, relative Frobenius < 1e-6 |
| Residual variant accumulates rather than overwrites | pass |
| All 32 coordinate values at all 8 positions in both cosets | pass |
| `in_features` not a multiple of 128 is rejected | pass (throws `std::invalid_argument`) |

The reference unpacker in the test is a deliberate restatement of the spec's bit layout rather
than a call into the CUDA primitives, which is what made it able to catch a real bug: the first
version of `e8w5_decode_block_f32` pre-shifted the spread MSB plane by 4, putting **coordinate
7's MSB at bit 32** where it fell off the word. It affected 1 coordinate in 8 and produced
plausible-looking output. Note what did *not* catch it: `cuda_decode_mirror()` in
`research/pack_e8w5.py` extracts then shifts, so it mirrored the format's *intent* and passed
while the CUDA diverged. A mirror written from the same understanding as the code is not an
independent check — the same defect was present in `research/kernels/e8w5_dequant.cuh` and is
now fixed in both.

### 4.2 Register pressure (`nvcc --ptxas-options=-v`, CUDA 13.2)

Zero spill stores and zero spill loads on all four architectures; 128 B shared memory for the
codebook, plus 4.6 KB for the staged `x` tile.

| arch | GEMV | dequantize | brief's ≤32/≤40 target |
|---|---|---|---|
| sm_80 | 34 | 48 | met (≤40) |
| sm_89 | 40 | 48 | met (≤40) |
| sm_90 | 32 | 32 | met (≤32) |
| sm_120 (shipped) | 40 | 40 | met (≤40) |

The engine itself builds SM120 only; the other three are cross-compilation checks.

### 4.3 Decode throughput, RTX 5070 (48 SMs, 672 GB/s theoretical peak)

`ctest -L benchmark`, GLM-4-9B decode shapes. **Read the caveat below before quoting a ratio.**

| projection | E8W5 | BF16 | speedup |
|---|---|---|---|
| q_proj 4096x4096 | 85.9 µs (125 GB/s, 18.7%) | 78.4 µs | 0.91x |
| kv_proj 4096x256 | 7.4 µs (94 GB/s, 13.9%) | 5.6 µs | 0.76x |
| o_proj 4096x4096 | 111.5 µs (97 GB/s, 14.4%) | 19.4 µs | 0.17x |
| **gate_up 4096x27392** | **333.4 µs (216 GB/s, 32.1%)** | 859.2 µs | **2.58x** |
| **down 13696x4096** | **248.2 µs (145 GB/s, 21.6%)** | 565.3 µs | **2.28x** |

**The BF16 column for the small shapes is a benchmark artifact, not a real baseline.** o_proj's
BF16 row reports 1729 GB/s — 257% of theoretical peak — because a 32 MB weight matrix stays
resident in cache across the timing loop's iterations. In a real 40-layer decode nothing is
resident: 8.24 B parameters stream past once per token. The rows to trust are the two large MLP
projections, where the working set defeats the cache, and those are also the ones that matter:
`gate_up` + `down` are 168 M of the 206 M parameters in a layer, **82% of the weight bytes**.
BF16 rows also move 15-20% run to run; E8W5's do not.

E8W5 reaches 32% of peak at best. It reads 3.11x fewer bytes than BF16 but delivers ~2.4x the
speed on the bandwidth-bound shapes, so roughly a quarter of the theoretical advantage is still
on the table. The remaining gap is not DRAM: at 10.5 MB, q_proj's weights also fit in cache, and
it still only reaches 125 GB/s, which points at the LDS + dependent-FMA pipeline rather than
memory. §6 lists what to try.

### 4.4 Conversion quality — and why there is no outlier side-car

The converter puts **every** K column into the lattice bulk. The research's b=5 figures all
include 0.1% of channels retained in BF16, and Part II found that retention worth +18.8 dB on a
spiky layer, so dropping it needed justification rather than a shrug. The justification is
measured: those numbers were obtained with **per-row** scales, and per-group-128 scales absorb
what sparse retention was compensating for — a 128-wide group containing a spiky channel gets
its own step instead of one step per 4096-wide row being dragged by the worst channel.

GLM-4-9B layer 0, W-SQNR over the whole tensor:

| tensor | research b=5 (per-row + 0.1% BF16) | converter (G=128, no outliers) | delta |
|---|---|---|---|
| self_attn.q_proj | 21.36 dB | **23.65 dB** | +2.29 |
| self_attn.k_proj | 20.84 dB | **24.25 dB** | +3.41 |
| self_attn.v_proj | 26.09 dB | **26.70 dB** | +0.61 |
| self_attn.o_proj | 25.10 dB | **26.60 dB** | +1.50 |
| mlp.gate_up_proj | 25.39 dB | **26.63 dB** | +1.24 |
| mlp.down_proj | 25.96 dB | **26.92 dB** | +0.96 |

Group scaling wins on every tensor, and by the largest margins exactly where NF5 collapsed
(q_proj 12.88 dB, k_proj 15.64 dB with per-row scales). So v1 is both simpler and better: no
activation gather, no padding, no side-car tensors, no double-counting hazard for a future
kernel that adds them.

This is W-SQNR on six tensors of one layer; the end-to-end number is §4.5.

### 4.5 Full conversion and end-to-end execution

`scripts/e8w5_convert.py` on GLM-4-9B-Chat-1M, RTX 5070:

| | |
|---|---|
| tensors quantized / copied through | 240 / 203 |
| wall time | 850 s (14 min), ~21 s per layer |
| size | 17.67 GiB -> **7.23 GiB** (2.44x overall) |
| W-SQNR over the 240 quantized tensors | min 23.65 / mean **26.91** / max 27.08 dB |
| per-tensor pack round-trip check | passed on all 240 |

The 2.44x is lower than the format's 3.11x because embeddings, `lm_head` and the norms stay
BF16 — 2.5 GiB of the 7.23 GiB is unquantized, and `lm_head` alone is 1.24 GiB.

**The engine runs it natively.** `blackwell_llm --model F:/AI/models/GLM-4-9B-e8w5 --ctx 512`
loads 7.23 GiB through DirectStorage (1469 aligned requests, no Win32 fallback) and generates:

> *Prompt:* "Explain in three sentences why lattice quantization can beat scalar quantization
> at the same bit rate."
>
> *E8W5 GLM-4-9B:* "Lattice quantization can outperform scalar quantization at the same bit
> rate because it allows for more efficient representation of the signal by exploiting the
> structure of the signal space, leading to better reconstruction quality. It achieves this by
> mapping the signal to a discrete set of points on a lattice, which can be more closely spaced
> than the uniform quantization levels used in scalar quantization. [...]"

Coherent and on-topic across multiple sentences, which is a much stronger signal than "did not
crash": a decode bug in a 5-bit weight path degrades into fluent nonsense long before it
segfaults.

**Perplexity A/B**, same eval text and same code path, via `blackwell_bench` (which gained
`--model` and `--gpu-layers`; the default checkpoint is unchanged):

| checkpoint | bpw of the linear layers | PPL |
|---|---|---|
| GLM-4-9B BF16 (`--gpu-layers 17`) | 16 | **12.4701** |
| GLM-4-9B E8W5 (all resident) | 5.14 | **13.0102** |
| | | **+0.540 (+4.33%)** |

**Do not quote this as a perplexity result.** The bench scores a single hardcoded 83-token
paragraph, so the sampling error on 83 log-probs is large and the number is indicative of "the
format works end to end", not of the model's quality on a corpus. A real evaluation needs
wikitext-2 over thousands of tokens, which this bench is not built for (item 4 in §6). What it
does establish is that 5.14 bpw weights cost single-digit percent PPL here rather than breaking
the model, and that the whole path — converter, loader, dispatcher, kernel — is consistent.

Timing note: 83 sequential `ForwardEval` calls plus the 7.4 GiB load complete in 17.7 s wall.
Load dominates; the linear layers account for roughly 40 x 794 µs = 32 ms per token by the §4.3
kernel sums, i.e. ~2.6 s of the run. End-to-end tok/s was not isolated from load time.

---

## 5. Batched / prefill path: why it sweeps rows

`num_tokens >= batched_gemm_threshold` falls through to the per-row GEMV sweep, sharing
`COMPRESSED_TENSORS_INT4`'s precedent. For E8W5 the obstruction is **structural, not merely
unwritten**: an `mma.sync.aligned.m16n8k16` B-fragment hands one lane four k values per k16
tile, while an E8 block needs all eight of its coordinates in one lane to recover u_7's parity
bit. A batched kernel therefore requires the MMA-permuted pack of spec §9.2 — which is measured
to be quality-neutral (−0.008 dB) but which the converter does not yet emit. Until then, a
batched launch would decode each block four times and discard six coordinates of eight.

Prefill is consequently slower than BF16 prefill per token. For this repo's stated workload —
batch=1, latency-critical, hotkey-driven translation — decode dominates, but a long first
prompt will feel it.

**Partially mitigated since (2026-09).** `IBlackwellEngine::PrefillTokens` skips the 1.24 GiB
`lm_head` read for every prompt token whose logits are discarded, and tiles the prompt through
`Impl::run_chunk` under Paged mode. The linear projections still sweep per row — that needs the
MMA-permuted pack below — but the batched attention/norm/RoPE path cuts the quadratic
context term 21x, which is 4.7x on a 16k-token prompt. Measured in
docs/LONGBENCH_E8W5_BASELINE.md §6.

---

## 6. Verification register and what to do next

| # | Claim | Status |
|---|---|---|
| 1 | Bit-exact device decode vs independent reference, 4 shapes | **verified** |
| 2 | GEMV within 1e-6 of FP64 over the same weights | **verified** |
| 3 | No register spills, ≤40 registers, 4 architectures | **verified** |
| 4 | Full 148-test validation suite unaffected | **verified** |
| 5 | Converter round-trips every packed tensor bit-exactly (per-tensor check, on by default) | **verified** |
| 6 | Group-128 scaling beats per-row + 0.1% outliers on W-SQNR | **verified**, 6 tensors of layer 0 |
| 7 | Decode throughput vs BF16 | **measured**, 2.3–2.6x on the weight-dominant projections; slower on the small attention ones, where the BF16 baseline is cache-inflated |
| 8 | Full 240-tensor conversion, every tensor round-trip verified | **verified** (§4.5) |
| 9 | End-to-end generation on the converted checkpoint, coherent output | **verified** (§4.5) |
| 10 | Perplexity vs BF16 | **measured**, 12.4701 -> 13.0102 (+4.33%) — on an 83-token text, so indicative only |
| 11 | Act-SNR / logit-KL of the G=128 no-outlier quantizer | **NOT measured** |
| 12 | Prefill / batched throughput | **measured, 13.75-15.37 ms/token (~65-73 tok/s)** on GLM-4-9B E8W5 at ~2k-token prompts, Paged + chunked prefill with `want_logits=false` for interior tokens (docs/LONGBENCH_E8W5_BASELINE.md §4). Still NO batched E8W5 GEMM — the linear projections sweep per row either way — but batching attention/norms/RoPE/KV-append cuts the context-dependent quadratic term 21x: `15.6*N + 6.26e-4*N^2` ms vs the token-by-token `17.2*N + 6.43e-3*N^2`, i.e. 4.7x on a 16k prompt, greedy output byte-identical within a KV mode |
| 13 | End-to-end tok/s isolated from load time | **measured** (docs/LONGBENCH_E8W5_BASELINE.md §4): prefill 13.75-15.37 ms/token, **decode 63.5-68.1 ms/token (~15.6 tok/s)**, timed separately per sample after the DirectStorage load over 200 LongBench prompts. Decode costs 4.6x more per token than prefill; ~13% of a decode step is the 1.24 GiB `lm_head` read, the rest is **unattributed** — ranked hypotheses (Paged-decode block-table overhead, `sample_top_p`'s per-token `cudaMalloc`/`cudaFree` + synchronising D2H copy, batch-1 SM occupancy) are listed in §4, and the Continuous-vs-Paged A/B that would separate them was cancelled before it ran — NOT yet diagnosed |

Next, in order of value:

1. **A real perplexity evaluation.** §4.5's +4.33% comes from 83 tokens. Point the bench at
   wikitext-2 (the research harness already downloads it) and score thousands, on both
   checkpoints, before anyone treats the format as production-ready.
2. **Close the bandwidth gap.** E8W5 sits at 32% of peak where BF16 reaches 62%. Candidates, in
   the order I would try them: widen the tile so each lane owns 2–4 blocks *strided by 32* (an
   offline block interleave, which restores `uint4` loads without recreating the x scatter);
   hoist the codebook into registers as a `half2[32]` select tree for the common coset; and
   process two output rows per warp to reuse the staged `x`.
3. **The MMA-permuted pack**, which unlocks both a batched kernel and prefill parity. The
   permutation is already implemented and measured in `research/pack_e8w5.py`; the converter
   needs a flag and the kernel needs writing.
4. **A CTest integration test** that loads a small converted checkpoint and compares logits
   against the BF16 original, so the format is covered end to end and not only at the kernel
   boundary.
