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
python research/run_full_engine_eval.py --skip-nvcomp --layers 10   # quantization only
python research/e8_lattice_engine.py                               # E8 decoder self-test
python research/capture_activations.py --num-seqs 8 --seq-len 256   # bigger calib set
python research/nvcomp_stream_bench.py --payload e8=some.bin        # codecs standalone
```

## Modules

| File | Role |
|---|---|
| `capture_activations.py` | **A** — hooks `layers.{3,10}.mlp.down_proj` on a truncated GLM-4-9B and saves the real input activations. Verifies the truncated load against the safetensors shards bit-for-bit. |
| `e8_lattice_engine.py` | **B** — exact Conway-Sloane E8 decoder (with a brute-force self-test), the five rate accountings, the scalar baselines, bit-exact payload packing, and the SQNR / Act-SNR metrics. Shared core; the others import it. |
| `qjl_residual.py` | **C** — 1-bit JL residual sketching, the closed-form noise/signal prediction, and the m-sweep that tests it. |
| `nvcomp_stream_bench.py` | **D** — nvCOMP v5 warm decompression latency and throughput, with round-trip verification and a d2d-bandwidth reference. |
| `run_full_engine_eval.py` | Orchestration → `bench_report.json` + the CLI matrix. |

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
