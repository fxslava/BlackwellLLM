---
name: golden-dumps
description: Regenerate the PyTorch golden reference dumps that BlackwellLLM's CTest integration suite compares against. Use when integration tests report missing/stale dumps, when engine or model geometry changes (SSM state shape, KV layout, new checkpoint family), or when adding a new integration parity test.
---

# Golden Dumps — PyTorch ↔ Engine Parity Protocol

The `integration` CTest suite proves the CUDA engine bit-for-bit (FP32 tolerance) against
HuggingFace/PyTorch reference tensors. The dumps are the **ground truth**. Two storage
conventions exist:

- `tests/integration/golden_dumps/**` — small per-layer probe tensors, **tracked in git
  via Git LFS** (`.gitattributes`: `tests/integration/golden_dumps/**/*.bin`). A clone
  needs `git lfs install` once, or the tests read 127-byte pointer files and fail loudly.
- `dumps/` — the large Llama full-pass dumps, **gitignored**, regenerated locally.

## Script → checkpoint → test mapping

| Generator | Checkpoint folder (under `BLACKWELL_MODELS_DIR`) | Consumed by |
|---|---|---|
| `scripts/generate_golden_dumps.py` | `llama3-8b-fp8` (Llama-3 8B **FP8** — intercepts `torch._scaled_mm` to capture the *true* FP8 activation bytes + dynamic scales) | Llama integration tests, via `dumps/` |
| `scripts/generate_qwen_dumps.py` | `Qwen2.5-Coder-7B-Instruct-AWQ` (**AWQ 4-bit**) | `tests/integration/test_qwen_engine.cpp` |
| `scripts/generate_qwen35_dumps.py` | `Qwen3.5-9B-AWQ-4bit` (**hybrid**: 24 GatedDeltaNet linear + 8 gated full-attention layers) — single-token decode, zero recurrent state | `tests/integration/test_qwen3_5_hybrid_integration.cpp` |
| `scripts/generate_qwen35_multistep.py` | same — multi-step decode (recurrent state evolution) | multistep hybrid parity test |
| `tests/integration/generate_tokenizer_goldens.py` | llama3 + qwen25 | `tests/integration/test_tokenizer_golden.cpp` |

Run from anywhere — output paths are anchored to the repo root, not the CWD. Requires
`torch` + `transformers` (CUDA build).

## Path conventions (both sides must agree)

- **Python side:** checkpoint roots resolve as
  `os.environ.get("BLACKWELL_MODELS_DIR", "F:/AI")` + the folder names above (the Llama
  scripts default to the CWD for the legacy `./llama3-8b-fp8` layout). Everything stays
  overridable per-run via `--model-dir` / `--out-dir`.
- **C++ side:** the integration tests use per-test env overrides with the same defaults —
  `BLACKWELL_LLAMA_INDEX`, `BLACKWELL_LLAMA_DIR`, `BLACKWELL_QWEN_DIR` (see `env_or()` in
  `tests/integration/`). When adding a test, follow that `env_or()` pattern; never bake in
  an absolute path without an env escape hatch.

## The geometry contract (why regeneration is ever needed)

Each generator documents, in its module docstring, the exact geometry the C++ side reads —
e.g. `scripts/generate_qwen35_dumps.py` must match `SsmGeometry::from_config` in
`src/core/ssm/ssm_state_pool.h` (recurrent state `[H][Dk][Dv]`, conv ring buffer `[conv_dim][K-1]`
— note HF keeps the full `K` window, the engine keeps `K-1`; the script synthesizes init
buffers from the *engine* geometry deliberately).

**Protocol when geometry changes on either side:**
1. Identify which side moved (engine struct vs HF modeling code — the docstrings cite the
   exact `transformers` source they verified against).
2. Update the generator's geometry block AND its docstring to re-verify against the
   installed `transformers` version.
3. Regenerate; then rebuild + run `ctest -L integration` and confirm the affected test
   *ran* (it skips silently when dumps are absent — a skip is not a pass).
4. If tensor names/shapes changed, update the reader side in `tests/integration/` in the
   same change. Regenerated dumps under `tests/integration/golden_dumps/` are committed
   (LFS handles the binaries transparently — just `git add`).

## Sharp edges

- The FP8 generator's `torch._scaled_mm` spy is version-sensitive: a `torch` upgrade that
  changes the private API breaks capture silently — sanity-check captured byte counts.
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before importing torch on this
  machine (the scripts that need it already do; copy the pattern into new ones).
- `meta.json` inside each golden_dumps folder records the `model_dir` it was generated
  from — check it when parity failures look like a checkpoint mismatch.

<migration_context>
Roadmap #5 (root cleanup) landed 2026-07: scripts moved from the repo root to `scripts/`,
default paths re-anchored to the repo root, `BLACKWELL_MODELS_DIR` introduced, and the
previously plain-tracked 126 MB of golden dumps migrated to Git LFS (no history rewrite —
pre-migration commits still carry the raw blobs; a fresh partial clone with LFS skips
them). If old notes/scripts reference `generate_*.py` at the repo root, the `scripts/`
copies are the live ones.
</migration_context>
