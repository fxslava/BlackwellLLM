# E8W5-A16 — weight format specification

5-bit companded-E8 weights, 16-bit activations. Byte-exact storage layout, decode algebra,
and thread mapping.

- Reference packer + verification: [`research/pack_e8w5.py`](../research/pack_e8w5.py)
- Device primitives: [`research/kernels/e8w5_dequant.cuh`](../research/kernels/e8w5_dequant.cuh)
- Where the quantizer comes from: [`research/MULTIRATE_SWEEP.md`](../research/MULTIRATE_SWEEP.md)
  (Part V) and [`research/COMPANDED_E8.md`](../research/COMPANDED_E8.md) (Part IV)

> **This is a research format, not an engine subsystem.** Nothing in `src/` or `tests/`
> depends on it, and no kernel has been written against it. It is specified and verified to
> the point where a kernel *can* be written; §11 is the register of what is measured and
> what is still assumed. The repo's own build targets `sm_120` only and is batch=1
> latency-critical, whereas this spec also covers SM80/89/90 and the tensor-core path,
> which that workload does not exercise.

---

## 1. Scope, and the three corrections to the brief

What the format stores, exactly: a per-tensor 64-entry codebook, per-group FP16 scales, and
two bit-planes of lattice coordinates. What it decodes to: FP16/BF16 register pairs.

Three things in the originating brief do not survive contact with the measured quantizer,
and the format below is built on the corrected versions.

| Brief says | Actually | Consequence |
|---|---|---|
| "32-entry Gaussian de-warp table" | **64 entries.** The codebook is indexed by `(coset, u)`, not by `u` — E8 is `D8 ∪ (D8 + ½·1)`, so each coordinate value has two reconstruction levels. `2 × 2^b = 64` at b=5. The "32" is Part IV's b=4 number (`2 × 16`) carried over to b=5 unchanged. | 6-bit index, 128 B table. Changes the select-tree depth, the smem footprint and the bank analysis (§7). |
| "≤ 2 ALU instructions per parameter" | **Not reachable for any companded format**, and the dB lives in the companding. ~5.9 int ops + 1 shared load per weight (§6). The table-free alternative (uniform E8) still costs ~3.1 int ALU ops/weight by the existing audit's count, and *loses* to NF5 at b=5 by −0.97 dB W-SQNR. | The op-count target and the dB target are mutually exclusive. §6 says why it does not matter at batch=1 and where it starts to. |
| "+1.2 to +2.7 dB over NF5 … completely outperforming" | Correct for `down_proj` (Part V: +2.714 dB layer 3, +1.218 dB layer 10, Act-SNR). Across all 12 tensors the partial full-model sweep has finished, the *typical* margin is ≈ +0.5 dB, with two +15 dB wins on layer-0 q/k_proj (where NF5 itself collapses to ~15 dB) and one **loss** of −0.881 dB on layer-0 o_proj. Weight-domain margin sits at the +0.522 dB packing ceiling. | The format is worth building; expect ~+0.5 dB typical plus robustness on pathological tensors, not a uniform +2 dB. §11. |

---

## 2. The code: 40 bits per 8 weights

Eight weights of one output row form one **block**, quantized as a point of box-constrained
E8. The stored fields per block:

| Field | Range | Bits |
|---|---|---|
| `u_i = k_i + 16`, i = 0..6 | `[0, 31]` | 7 × 5 = 35 |
| `u_7` high 4 bits | `[0, 15]` | 4 |
| `coset` c | `{0, 1}` | 1 |
| **total** | | **40** |

`u_7`'s low bit is **not stored**: D8 requires `Σ k_i` even, and `Σ u_i ≡ Σ k_i (mod 2)`
because the offset `8 × 16` is even, so

```
u_7 bit 0  =  XOR of (u_i bit 0) for i = 0..6        // == __popc(L & 0x01111111) & 1
```

This is the parity trick from Part II generalized: `1 + 7b + (b−1) = 8b` bits, i.e. exactly
**b bits per weight with the coset carried free**, at any coordinate width. At b=5 the
payload is exactly **5.000 bpw**.

Reconstruction of coordinate i, with a per-group scale `δ` and the per-tensor codebook `C`:

```
idx_i = (c << 5) | u_i                    // 6 bits, 0..63
w_i   = C[idx_i] * δ
```

`C` holds *conditional means* of the group-normalized weights, so it is the inverse
companding warp refined to centroids. It absorbs the companding strength λ, which is why λ
is not a runtime field — and it must not be hardcoded to 1.0: the full-model sweep picks
λ=0.25 for layer-0 q_proj and k_proj and λ=1.0 for the other ten tensors.

---

## 3. Bit-exact layout: two planes

**Plane L — one `uint32` per block.**

| Bits | Contents |
|---|---|
| 3:0 | `u_0 & 0xF` |
| 7:4 | `u_1 & 0xF` |
| … | … |
| 27:24 | `u_6 & 0xF` |
| 28 | `coset` |
| 31:29 | `u_7` bits 3:1 |

Nibble 7 is `(u_7 & 0xE) | coset`: the coset flag takes the bit that parity frees. It has to
live here — plane H is exactly 8 bits wide with no spare, and 8 coordinate MSBs + 1 coset
bit would be 9.

**Plane H — one `uint8` per block.** Bit i = `u_i >> 4`, for i = 0..7.

Total 5 bytes per block, and per scale group of 128 weights (16 blocks):

| Plane | Bytes/group | As 16-byte vectors |
|---|---|---|
| L | 16 × 4 = **64** | 4 × `uint4` |
| H | 16 × 1 = **16** | 1 × `uint4` |
| | 80 | 5 × `uint4` |

Both planes are independently 16-byte aligned with **zero padding**, which is the hard
constraint the brief opens with.

---

## 4. Why two planes, and why G=128 falls out of the arithmetic

40 bits per block and 128-bit transactions have `lcm(40, 128) = 640 bits = 16 blocks = 128
weights`. So **G=128 is the smallest group size at which a packed 5-bit stream is 16-byte
aligned at all** — the brief's requested group size is exactly the alignment quantum, not a
convention inherited from AWQ.

That alignment holds for a single interleaved stream too, so it does not by itself decide the
layout. What decides it is the per-block extraction pattern:

| Layout | Block field extraction | Verdict |
|---|---|---|
| **Chunk-32** (brief's option A): 4 blocks in 5 × `uint32` | Block j occupies bits 40j…40j+39, so 3 of every 4 blocks straddle a word boundary. Needs `shf.r.wrap` funnel shifts, **four different extraction patterns** with period 5 words, and 5 live words to decode 4 blocks. | rejected |
| **Chunk-64** (option B): 8 blocks in 10 × `uint32` | Same straddling, period 10 words, 8 patterns. Worse. | rejected |
| **Two planes** | Every block: one whole `uint32` + one whole byte. **One** extraction pattern, uniform across all blocks and all coordinates after the §6 fixup. | **adopted** |

Secondary reasons the split wins: each plane is a contiguous stream that prefetches
independently; a 64 B plane-L group and a 16 B plane-H group both divide a 128 B cache line
evenly, whereas an 80 B interleaved group puts 1.6 groups per line and crosses lines at a
non-power-of-two stride; and plane H can be skipped entirely by a b=4 variant of the same
format, making the two widths share a decoder.

---

## 5. Tensor-level layout

Per quantized `[N, K]` weight tensor, with `Kp` = padded bulk columns (§8):

| Tensor | dtype | Shape | Notes |
|---|---|---|---|
| `qweight_lo` | `uint32` | `[N, Kp/8]` | plane L, row-major, block order |
| `qweight_hi` | `uint8` | `[N, Kp/8]` | plane H, same block order |
| `scales` | `float16` | `[N, Kp/128]` | one per group, contiguous separate tensor |
| `codebook` | `float16` | `[64]` | per tensor |
| `outlier_cols` | `int32` | `[n_out]` | ascending K indices held in BF16 |
| `outlier_vals` | `bfloat16` | `[N, n_out]` | the retained channels verbatim |

Scales are a **separate contiguous tensor, not interleaved** with the weight planes. Reasons:
a GEMV warp reads 8 scales per 1024 weights, so interleaving would inject a 2-byte field
into an otherwise perfectly vectorized stream to save a load that is already 1/512th of the
traffic; and the scale tensor is small enough (`N × Kp/128 × 2` B = 0.9 MB for
`[4096, 13696]`) to stay resident while the 28 MB of planes stream past.

`Kp` is always a multiple of 128, so `Kp/8` is a multiple of 16 and **every row of both
planes starts and ends 16-byte aligned**. That is also exactly the condition Hopper TMA box
copies need on the innermost stride, so both planes are TMA-copyable by construction (§9.3).

---

## 6. Decode, and the instruction count

The kernel-side algebra, mirrored op-for-op and verified bit-exact by
`cuda_decode_mirror()` in the packer:

```c
coset  = (L >> 28) & 1;                                  // 2 ops
parity = __popc(L & 0x01111111) & 1;                     // 2 ops
L      = (L & 0xEFFFFFFF) | (parity << 28);              // 1 lop3
msb    = spread8(H) << 4;                                // 9 ops  (3 x shift/or/and)
base   = coset << 5;                                     // 1 op
// then per coordinate i:
idx_i  = base | ((L >> 4i) & 0xF) | ((msb >> 4i) & 0x10); // bfe + shift + and + lop3
w_i    = C[idx_i];                                        // LDS.u16
```

The bit-28 fixup is the load-bearing line: after it, **nibble i equals `u_i & 0xF` for all
eight coordinates**, so coordinate 7 stops being a special case and the per-coordinate inner
code is uniform. Three ops buy a uniform unrolled loop.

Static cost, same convention as `research/COMPANDED_E8.md`'s silicon audit (counted, not
measured):

| Item | Per block | Per weight |
|---|---|---|
| coset + parity + fixup + spread + base | 15 | 1.9 |
| per-coordinate index | 4 × 8 = 32 | 4.0 |
| **integer ALU total** | 47 | **5.9** |
| shared-memory loads | 8 | 1.0 |
| `HFMA2` | 4 | 0.5 |
| global loads | 1 × `uint32` + 1 B | 0.16 B/weight |

**The ≤2 ops/weight target is unreachable here, and the reason is structural**: the 64-entry
table lookup *is* the companding, and the companding is where the entire margin over NF5
comes from. The table-free option is uniform (non-companded) E8, which `COMPANDED_E8.md`'s audit prices at 25 int ALU per 8 weights (~3.1/weight — so not 2 either) and which
Part V measures at −0.968 dB W-SQNR *against* NF5 on layer 10, i.e. it loses to the scalar
baseline it is supposed to beat. There is no third option at this coordinate width.

Whether 5.9 ops/weight costs anything depends on the regime, and the research already priced
it: one `[4096, 13696]` tensor is 28.1 MB at this rate, and reading it at 564 GB/s takes
50 µs. At **batch=1** the GEMV is bandwidth-bound and the whole dequant chain hides under
the memory time — which is the regime this repo actually ships. It begins to matter only
when the weight read is amortized over enough activation columns to make the kernel
compute-bound, i.e. the tensor-core GEMM case the brief targets. That crossover is
**not measured here**.

Measured register use (`nvcc --ptxas-options=-v`, CUDA 13.2, **zero spills**, 128 B smem on
all four):

| arch | decode probe | full GEMV row kernel | ≤32-register target |
|---|---|---|---|
| sm_80 | 20 | 32 | **met** |
| sm_89 | 21 | 40 | missed (→80% occupancy) |
| sm_90 | 21 | 32 | **met** |
| sm_120 | 22 | 40 | missed (→80% occupancy) |

The occupancy target is therefore an Ampere/Hopper property of this source, not a property
of the format. Ada and Blackwell need 40 registers for the same code.

---

## 7. Codebook residency, and why FP16 is a memory decision

The codebook is 64 conditional means fitted **per tensor** — it absorbs λ, which varies
across tensors — so it can be neither a compile-time constant nor one `__constant__` table
shared by the model. Stage it in shared memory once per block launch: 128 B, `e8w5_stage_codebook()`.

Bank behaviour, which is the reason the table is FP16 and not FP32:

- 64 halves = 128 B = 32 four-byte words, so **word w sits in bank w and every bank holds
  exactly one word**. Two lanes reading different entries either hit different banks or hit
  the *same word* (an in-word broadcast). A 32-lane gather into this table is therefore
  **conflict-free for any index pattern whatsoever**.
- The same table in FP32 is 64 words over 32 banks — two words per bank — and an arbitrary
  gather can two-way conflict.

`__constant__` is the wrong home despite the size: constant memory serializes across
differing addresses within a warp, and the indices here are data-dependent and uncorrelated
across lanes, which is the access pattern constant memory is worst at.

---

## 8. Scales, outliers, padding

**Scales.** One FP16 per 128 weights, `[N, Kp/128]`. In a dot product the group scale comes
straight out of the sum — `Σ (δ v_j) x_j = δ Σ v_j x_j` — so it costs **one multiply per 128
weights**, not one per weight. Group scaling is free in the inner loop; it costs 0.125 bpw in
the file (§10), and that is the whole price.

**Outliers are part of the format, not an optional extra.** Every dB quoted in Part V
includes 0.1% of K retained as BF16 channels, and the research is emphatic about why: at b=5
on layer 3, direct E8 scores 9.08 dB and the same code with 0.5% sparse retention scores
27.91 dB. Quoting this format's quality without the outlier side-car is quoting a number
that does not exist. Selection is by **activation** energy on a calibration split
(`topk(x_cal² .sum(0))`), not by weight magnitude — so a packer must be handed the
calibration-chosen columns; `pack_e8w5.py`'s `--pack-tensor` path uses a weight-magnitude
stand-in and says so.

**Padding.** Outlier columns are *gathered out* before quantizing, so a retained channel does
not also burn a lattice code (rate is `(1−a)·bulk + a·16`, not `bulk + a·16`). The bulk is
then zero-padded up to a multiple of 128. This is why `K % 128 == 0` is not sufficient:
`K − n_out` generally is not. For `[4096, 13696]` with `n_out = 13`, bulk 13683 pads to
13696 — 13 dead columns, 0.0047 bpw.

---

## 9. Coordinate-to-thread mapping

### 9.1 GEMV (batch=1) — `LAYOUT_K_LINEAR`

Block j covers K positions `[8j, 8j+8)`. Warp tile = 1024 weights of one output row = 128
blocks = 8 groups. Lane *l* owns blocks `4l … 4l+3` — 32 weights, entirely inside group
`l/4`:

| Load | Per lane | Per warp |
|---|---|---|
| plane L | one `uint4` at block 4l | 32 × 16 B = 512 B, fully coalesced |
| plane H | one `uint32` at block 4l | 32 × 4 B = 128 B, fully coalesced |
| scales | group `l/4` | 8 scales, 4-lane broadcast each |

Two loads per lane, both naturally aligned, no shuffles, no wasted decode.

### 9.2 Tensor cores — `LAYOUT_MMA_M16N8K16`, and the parity obstruction

**The pack is not decodable coordinate-by-coordinate.** `u_7`'s low bit is recovered from the
parity of the other seven, so whichever thread owns one coordinate of a block must own all
eight. This is a hard constraint that the naive layout violates:

For `mma.sync.aligned.m16n8k16` with `.f16`, the B fragment gives lane *t* the four k values
`{2j, 2j+1, 2j+8, 2j+9}` for `j = t % 4`, at `n = t >> 2`. A block of 8 *consecutive* k is
therefore split across four lanes, each holding 2 of its 8 coordinates — and none of them can
decode its own two. The options are:

1. every lane decodes the whole block and discards 6 of 8 coordinates (4× decode waste),
2. `__shfl_sync` the parity across 4 lanes (the brief rules shuffles out, rightly),
3. drop the parity trick, store 41 bits per block → 5.125 bpw payload,
4. **choose which 8 weights form a block so that one lane owns all of them.** ← adopted

Two consecutive k16 tiles hand one lane exactly eight k values, and eight is the E8 block
size. So inside each aligned 32-column window, block `j` (j = 0..3) is defined as

```
K offsets = {2j, 2j+1, 2j+8, 2j+9} ∪ {16+2j, 16+2j+1, 16+2j+8, 16+2j+9}
```

baked into the offline packer (`mma_m16n8k16_perm()`). Block ownership and lane ownership
become the same thing: no shuffle, no waste, parity local. All 32 offsets stay inside one
32-column window, hence inside one 128-weight scale group, which is what keeps the block's
single scale well defined.

**This is only legitimate if regrouping is statistically neutral**, since the encoder's cells
are 8-dimensional and it is now quantizing a different 8-tuple. It is, measured:

| scales | layout | E8W5 W-SQNR | Δ vs k_linear |
|---|---|---|---|
| per-row | `k_linear` | 25.986 dB | — |
| per-row | `mma_m16n8k16` | 25.978 dB | **−0.008 dB** |
| group-128 | `k_linear` | 26.864 dB | — |
| group-128 | `mma_m16n8k16` | 26.860 dB | **−0.004 dB** |

(layer-10 `down_proj`, 512 × 4096 slice, 20-point step grid.) The permutation is free to
three decimal places, as exchangeability predicts: the warp is per-coordinate and shared, and
the lattice treats its eight coordinates symmetrically.

The same argument supplies the recipe for any other fragment shape, including `k32` 8-bit
tiles: **find the 8 k-positions one lane owns and declare those a block.** If a target
fragment gives a lane a number of k values that is not a multiple of 8, this format cannot
serve it without option 3 above.

### 9.3 Hopper TMA / WGMMA — what carries over and what is not claimed

Carries over, and checked: both planes are 2-D row-major tensors whose innermost byte stride
is a multiple of 16 for any legal `Kp` (§5), so TMA box copies into shared memory are
well-formed with no repacking. The MMA permutation is baked offline, so TMA copies tiles
verbatim and no runtime transpose appears.

**Not claimed, and deliberately not guessed:** WGMMA takes its operands from shared memory
under specific swizzle patterns, and a table-decoded weight lands in *registers*, so a
`wgmma` path needs the decoded FP16 written back to shared memory in the swizzle layout — an
extra shared round-trip per tile. Whether that round-trip is hidden by the async pipeline, and
what the resulting occupancy is, cannot be established from here. This spec follows Part IV's
precedent of leaving a column empty rather than inventing instruction counts for hardware it
has not measured.

---

## 10. Rate accounting

Exact, for `[4096, 13696]` (GLM-4-9B `down_proj`) at G=128 with 13 BF16 outlier channels.
Every stored bit counted once.

| Component | Bits | bpw |
|---|---|---|
| plane L (`qweight_lo`) | 224 395 264 | 4.000000 |
| plane H (`qweight_hi`) | 56 098 816 | 1.000000 |
| scales, FP16 per 128 | 7 012 352 | 0.125000 |
| outlier values, BF16 | 851 968 | 0.015187 |
| codebook, 64 × FP16 | 1 024 | 0.000018 |
| outlier column indices | 416 | 0.000007 |
| **total** | **288 359 840** | **5.140213** |

of which payload 5.015187 and metadata 0.125026. Dead padding columns: 13, worth 0.004746 bpw
(already inside plane L/H above). Total footprint 36 044 980 B (34.4 MiB) against 112 197 632 B (107.0 MiB) for BF16 — **3.11×**, i.e. 16 / 5.1402.

**The scale granularity is the whole difference from Part V's published rate**, and it is a
real trade rather than a free win:

| Scale granularity | bpw | W-SQNR (layer-10 slice) | margin over NF5 |
|---|---|---|---|
| per-row (Part V's choice) | 5.0164 | 25.986 dB | +0.611 dB |
| **per-group-128** | **5.1402** | **26.864 dB** | **+0.753 dB** |
| Δ | +0.1238 | **+0.878 dB** | +0.142 dB |

Spending 0.124 bpw on finer scales buys +0.878 dB. Spending the same 0.124 bpw on coordinate
width would buy ≈0.75 dB (interpolating Part V's b=4→b=5 step, +6.03 dB per bpw on this
tensor). **Group scaling is a slightly better use of those bits than rate is** — and NF5
gains from it too (+0.736 dB), so the margin over the baseline survives the change and
widens marginally. That last point matters: the comparison re-uses Part III's rule that a
baseline gets the same tuning as the candidate.

---

## 11. Verification register

| # | Claim | Status | Evidence |
|---|---|---|---|
| 1 | Pack round-trips bit-exactly | **verified** | 66 048 blocks / 528 384 coordinates: all 32 coordinate values at all 8 positions in both cosets, plus 65 536 random encoder outputs. `--self-test` |
| 2 | The kernel's integer decode reproduces the reference index exactly | **verified** | `cuda_decode_mirror()` vs the reference on the same set; index range exactly [0, 63] |
| 3 | Both planes 16-byte aligned per group, zero padding | **verified** | 64 B and 16 B per group; arithmetic asserted in `--self-test` |
| 4 | Compiles clean, no spills, 128 B smem | **verified** | sm_80/89/90/120, CUDA 13.2 |
| 5 | ≤32 registers | **partly** | met on sm_80/sm_90 (32), missed on sm_89/sm_120 (40) |
| 6 | MMA permutation is quality-neutral | **verified** | −0.008 / −0.004 dB on a real tensor slice (§9.2) |
| 7 | The fit reproduces Part V | **verified** | per-row `k_linear` slice: E8W5 25.986 dB / NF5 25.375 dB vs Part V's full-tensor 25.98 / 25.39 |
| 8 | Group-128 scales do not cost the margin | **verified** | +0.753 dB at G=128 vs +0.611 dB per-row (§10) |
| 9 | Exact achieved BPW | **verified** | §10, every term counted; the packer's measured total matches the analytic table to the last printed digit (5.140213) |
| 9b | The stored planes decode to the reconstruction that was scored, at full K with real padding | **verified** | `--pack-tensor` on layer 10, K=13696, bulk 13683 + 13 pad columns, 107 groups/row: exact tensor equality |
| 10 | End-to-end Act-SNR / logit-KL at G=128 | **NOT measured** | Part V's Act-SNR and KL are per-row-scale numbers on `down_proj` only. §12 |
| 11 | The decode runs correctly on a GPU | **NOT measured** | verified only through the bit-exact Python mirror and compilation; the GPU was fully occupied by the 40-layer sweep |
| 12 | Any throughput, latency or occupancy figure | **NOT measured** | no kernel has been benchmarked; §6's counts are static |

The brief's acceptance criterion "model outputs match the Phase B reference to within 1e-4"
is met more strongly than asked in the weight domain — the pack is **bit-exact**, not
1e-4-close — but is **not** established at the model-output level, because the reconstruction
being matched is itself a new one (G=128 scales), so there is no Phase B run to compare
against yet. Item 10 is that run.

## 12. What to measure next, in order

1. **Act-SNR and logit-KL at G=128**, by adding the group-scaled fitter to
   `eval_multirate_sweep.py`'s Tier 2 and re-running both layers. This is the one gap between
   "verified format" and "verified format at the quality it claims". Needs the GPU.
2. **A GPU round-trip test** of `e8w5_dequant.cuh` against `pack_e8w5.py`'s tensors —
   cheap, and it closes item 11.
3. **Where dequant stops hiding.** Sweep batch size on a real GEMM until the 5.9 ops/weight
   stop being free; that number decides whether the MMA layout is worth using at all for
   this repo's batch=1 workload.
4. **The full-model sweep**, when it finishes. §1's per-tensor spread is 2 layers of 40; the
   −0.881 dB loss on layer-0 o_proj and the two +15 dB wins on layer-0 q/k_proj are the
   interesting cases, and whether they are depth-0 artifacts is currently unknown.
