---
name: golden-dumps
description: Regenerate the PyTorch golden reference dumps that BlackwellLLM's CTest integration suite compares against. Use when integration tests report missing/stale dumps, when engine or model geometry changes (SSM state shape, KV layout, new checkpoint family), or when adding a new integration parity test.
---

# Golden Dumps — PyTorch ↔ Engine Parity Protocol

The `integration` CTest suite proves the CUDA engine bit-for-bit (FP32 tolerance) against
HuggingFace/PyTorch reference tensors. The dumps are the **ground truth**; they are
regenerable artifacts (gitignored under `dumps/`), never committed.

## Script → checkpoint → test mapping

| Generator (repo root) | Checkpoint family | Consumed by |
|---|---|---|
| `generate_golden_dumps.py` | Llama-3 8B **FP8** — intercepts `torch._scaled_mm` to capture the *true* FP8 activation bytes + dynamic scales the framework computed | Llama integration tests in `tests/integration/` |
| `generate_qwen_dumps.py` | Qwen2.5 **AWQ 4-bit** | Qwen AWQ integration tests |
| `generate_qwen35_dumps.py` | Qwen3.5 **hybrid** (24 GatedDeltaNet linear layers + 8 gated full-attention, `full_attention_interval=4`) — single-token decode, zero recurrent state | `tests/integration/test_qwen3_5_hybrid_integration.cpp` |
| `generate_qwen35_multistep.py` | Qwen3.5 hybrid — multi-step decode (recurrent state evolution) | multistep hybrid parity test |

Run with the project venv's Python (needs `torch` + `transformers`, CUDA build); dumps are
raw little-endian FP32 tensors.

## The geometry contract (why regeneration is ever needed)

Each generator documents, in its module docstring, the exact geometry the C++ side reads —
e.g. `generate_qwen35_dumps.py` must match `SsmGeometry::from_config` in
`src/ssm/ssm_state_pool.h` (recurrent state `[H][Dk][Dv]`, conv ring buffer `[conv_dim][K-1]`
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
   same change.

## Sharp edges

- Model paths are currently **hardcoded absolute local paths** (e.g.
  `F:/AI/Qwen3.5-9B-AWQ-4bit`, `./llama3-8b-fp8`) — the scripts only run on a machine with
  those checkpoints at those locations.
- The FP8 generator's `torch._scaled_mm` spy is version-sensitive: a `torch` upgrade that
  changes the private API breaks capture silently — sanity-check captured byte counts.
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before importing torch on this
  machine (the scripts that need it already do; copy the pattern into new ones).

<evolution_protocol>
**Current workaround:** hardcoded absolute checkpoint paths (`F:/AI/...`) inside each
generator, scripts living loose in the repo root.
**Target state:** scripts under `scripts/`, checkpoint locations resolved as
`os.environ["BLACKWELL_MODELS_DIR"]` + a per-script relative model folder name, with a
clear error message when the env var or folder is missing. Dump output stays `dumps/`.
**When the repo cleanup lands (Roadmap #5 in CLAUDE.md):** rewrite this skill —
(1) update every script path in the table to `scripts/<name>.py`; (2) replace the
hardcoded-paths sharp edge with the `BLACKWELL_MODELS_DIR` convention and document the
expected folder names per checkpoint family; (3) when performing the migration itself,
change only path resolution — do not touch geometry blocks or the `_scaled_mm` spy, and
re-run one generator + `ctest -L integration` as the acceptance check; (4) update the
integration-test C++ side if it also hardcodes dump/model paths, keeping both sides on the
same env-var convention.
</evolution_protocol>
