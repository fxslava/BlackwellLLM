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
python research/eval_companded_e8.py                               :: Part IV, ~3 min
python research/eval_1d_vs_e8_kl.py                                :: Part III, ~90 s
python research/eval_fixed_rate_e8.py                              :: Part II, ~40 s
python research/eval_fixed_rate_e8.py --self-test                  :: pack + boxed decode
python research/run_full_engine_eval.py --skip-nvcomp --layers 10   :: quantization only
python research/e8_lattice_engine.py                               :: E8 decoder self-test
python research/capture_activations.py --num-seqs 8 --seq-len 256   :: bigger calib set
python research/nvcomp_stream_bench.py --payload e8=some.bin        :: codecs standalone
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
| `eval_companded_e8.py` | **Part IV** — can the lattice capture companding gain *and* packing gain? Radial shell scaling (a centroid condition) vs Cartesian 32-LUT companding (a real warp), against an NF4-centroid control. Includes `boundary_diagnostic()`, which measures the +0.405 dB ceiling in seconds without touching the model. → `COMPANDED_E8.md` + `companded_e8_report.json`. |

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
given coordinate width. A non-tiling lattice forfeits 0.25 dB of its asymptotic packing gain
to overload in a bounded 4-bit code, and Z^8 is the one lattice that tiles the code box
exactly. That single number would have predicted all of Part IV in advance.

**Accumulate the scale-search MSE in float64.** Adjacent grid points are often near-tied for
a row, so float32 reduction noise flips which scale that row picks and moves the downstream
logit-KL effect size by ~25% run to run. Part III's determinism depends on this.
