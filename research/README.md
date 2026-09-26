# research/ — E8 lattice VQ + QJL evaluation

An isolated experimental harness, deliberately outside the build: pure PyTorch, no CMake
target, nothing in `src/` or `tests/` depends on it. It measures a quantization *algorithm*
against the engine's real weights, activations, GPU and codec — it does not run through the
engine's kernels. See §5 of [FINDINGS.md](FINDINGS.md) for exactly where that line falls.

**Read [FINDINGS.md](FINDINGS.md) for the results.** Raw numbers land in `bench_report.json`.

## Run it

```bat
python research/run_full_engine_eval.py
```

~25 s end to end on an RTX 5070, plus ~20 s the first time for the activation capture.
Needs the GLM-4-9B HF checkpoint (`BLACKWELL_MODELS_DIR`, default `F:/AI/models`), and
downloads wikitext-2 on first use.

```bat
python research/eval_multirate_sweep.py                            :: Part V, ~7 min
python research/eval_multirate_sweep.py --tier1-only                :: ceiling per b, ~20 s
python research/eval_companded_e8.py                               :: Part IV, ~3 min
python research/eval_1d_vs_e8_kl.py                                :: Part III, ~90 s
python research/eval_fixed_rate_e8.py                              :: Part II, ~40 s
python research/eval_fixed_rate_e8.py --self-test                  :: pack + boxed decode
python research/run_full_engine_eval.py --skip-nvcomp --layers 10   :: quantization only
python research/e8_lattice_engine.py                               :: E8 decoder self-test
python research/capture_activations.py --num-seqs 8 --seq-len 256   :: bigger calib set
python research/nvcomp_stream_bench.py --payload e8=some.bin        :: codecs standalone
python research/pack_e8w5.py --self-test                           :: E8W5 pack/decode, no model
python research/pack_e8w5.py --neutrality                          :: what G=128 + the MMA perm cost
python research/pack_e8w5.py --pack-tensor --full-rows 4096        :: pack a tensor, print exact BPW
```

## Modules

| File | Role |
|---|---|
| `capture_activations.py` | **A** — hooks `layers.{3,10}.mlp.down_proj` on a truncated GLM-4-9B and saves the real input activations. Verifies the truncated load against the safetensors shards bit-for-bit. |
| `e8_lattice_engine.py` | **B** — exact Conway-Sloane E8 decoder (with a brute-force self-test), the five rate accountings, the scalar baselines, bit-exact payload packing, and the SQNR / Act-SNR metrics. Shared core; the others import it. |
| `qjl_residual.py` | **C** — 1-bit JL residual sketching, the closed-form noise/signal prediction, and the m-sweep that tests it. |
| `nvcomp_stream_bench.py` | **D** — nvCOMP v5 warm decompression latency and throughput, with round-trip verification and a d2d-bandwidth reference. |
| `run_full_engine_eval.py` | Orchestration → `bench_report.json` + the CLI matrix. |
| `eval_fixed_rate_e8.py` | **Part II** — zero-entropy fixed-rate E8: 8 coordinates in one 32-bit register (the coset flag paid for by D8's parity), box-constrained decode, FWHT incoherence vs sparse outlier retention, cosine metrics. → `FIXED_RATE_E8.md` + `fixed_rate_e8_report.json`. |
| `eval_1d_vs_e8_kl.py` | **Part III** — the head-to-head that decides whether to write the kernel: 1-D RTN/Lloyd-Max/NF4 against 8-D E8 at a matched 4.013 bpw, identical outliers and identical per-row scale search, scored all the way to logit KL by streaming the model tail. → `KL_1D_VS_E8.md` + `kl_comparison_report.json`. |
| `eval_multirate_sweep.py` | **Part V** — the b in {3,4,5} sweep that finds the cross-over. Tier 1 measures the packing ceiling per width with no model; Tier 2 runs the full logit-KL chain. Fits the companding strength lambda rather than assuming it. → `MULTIRATE_SWEEP.md` + `multirate_report.json`. |
| `eval_companded_e8.py` | **Part IV** — can the lattice capture companding gain *and* packing gain? Radial shell scaling (a centroid condition) vs Cartesian 32-LUT companding (a real warp), against an NF4-centroid control. Includes `boundary_diagnostic()`, which measures the +0.405 dB ceiling in seconds without touching the model. → `COMPANDED_E8.md` + `companded_e8_report.json`. |
| `pack_e8w5.py` | **Format** — the b=5 companded-E8 quantizer turned into a byte-exact kernel weight format: two 16-byte-aligned bit-planes, a 64-entry codebook, per-group-128 scales, and the MMA coordinate permutation. Verifies the pack round-trip and the *kernel's own integer decode* bit-exactly, and measures the two things Part V did not (group scales, block regrouping). → [`docs/E8W5_FORMAT_SPEC.md`](../docs/E8W5_FORMAT_SPEC.md) + `kernels/e8w5_dequant.cuh`. |

`artifacts/` (gitignored) holds the captured activations; they are re-derivable from the
checkpoint, so they are not tracked.

## Two things to know before extending this

**Rate is a choice, not a property.** An E8 quantizer has no intrinsic bit rate, and the
five definitions in `e8_rates` span 3.5 → 8.125 bpw *for the same quantized tensor*. Always
say which one a number is. `PRIMARY_RATE_KEY` in `run_full_engine_eval.py` is the
parity-credited coordinate entropy, because that is the one that reproduces the prior
offline analysis.

**Feed nvCOMP the bytes a kernel would really read.** Compressing a padded or dequantized
payload flatters the ratio badly — per-block byte padding alone moved GDeflate on E8 from
1.26 to 1.89.

**Give every baseline the same tuning you give the candidate.** Part I and Part II both
overstated E8 because the scalar baselines did not get the per-row MSE-optimal scale search
that the lattice got; with it, Lloyd-Max gains ~3.4 dB and the ranking flips (Part III). Any
new scheme added here must be scored against baselines fitted the same way.

**Screen a lattice idea with `boundary_diagnostic()` before building the plumbing.** It
measures, in seconds and with no model, what a lattice can win over the cubic lattice at a
given coordinate width: +0.202 dB at b=3, +0.405 at b=4, +0.522 at b=5, against an asymptotic
+0.654. Z^8 is the one lattice that tiles the code box exactly, so a non-tiling lattice
forfeits part of its packing gain to overload, and that loss halves with every added bit.
Run `eval_multirate_sweep.py --tier1-only` first; it would have predicted Part IV in advance.

**Never judge a lattice format at one bitwidth, and fit the companding strength.** The
lattice's advantage is non-monotone in b with a trough at exactly b=4 (Part V section 30), and
pinning the companding strength at "full" instead of fitting it cost 4.5 dB at b=3. Both
mistakes are in this repo's history.

**A format is not the quantizer.** `pack_e8w5.py` reuses Part IV's encoder, warp and codebook unchanged, so Part V's dB transfer — but per-group-128 scales and the MMA block permutation are *new*, and a dB measured with per-row scales does not describe them. Anything that changes scale granularity, block grouping or codebook precision has to be re-measured, not inherited; `--neutrality` is where that happens.

**Accumulate the scale-search MSE in float64.** Adjacent grid points are often near-tied for
a row, so float32 reduction noise flips which scale that row picks and moves the downstream
logit-KL effect size by ~25% run to run. Part III's determinism depends on this.
