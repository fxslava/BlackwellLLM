"""Golden-dump generator for the 8B FP8 Llama backbone, INJECTED-EMBEDDING path.

Target: local llama3-8b-fp8 (Meta-Llama-3-8B, FP8 checkpoint). Validates our
engine's FP8 transformer prefill INDEPENDENTLY of the audio projector: instead of
driving the model with token ids, we build a [seq_len, hidden] embedding matrix
and run the model with inputs_embeds= that matrix, so the C++ engine can inject
the exact same embeddings into d_X_accum (bypassing its embed_tokens kernel) and
compare per-position logits.

CREDENTIAL-FREE / OFFLINE: loads only the local checkpoint (no HF network). The
FP8 weights are consumed via transformers' native scaled_mm path (dtype=bf16
compute), which is the same reference the existing generate_golden_dumps.py uses.

Embeddings are IN-DISTRIBUTION by construction: we embed a deterministic,
seeded sequence of real token ids through model.model.embed_tokens, so the
injected activations sit in the FP8 dynamic range the model expects (a purely
random [194,4096] matrix would blow the first-layer scales and make parity
meaningless). The sequence itself is arbitrary -- this is a backbone stress test,
not a semantic one.

Dumps (into <repo>/dumps, gitignored -- these are large, ~100 MB for the logits):
  08b_input_embeds.bin   [seq_len, hidden]        FP32  (the injected embeddings)
  08b_final_logits.bin   [seq_len, vocab_size]    FP32  (per-position logits)
  08b_expected_tokens.txt  top-5 ids+strings at the LAST position
  08b_meta.txt           "seq_len hidden vocab_size"  (sidecar for the C++ reader)

Usage:
    python scripts/dump_llama8b_fp8.py
    python scripts/dump_llama8b_fp8.py --seq-len 194 --model-dir F:/AI/llama3-8b-fp8
"""

import argparse
import os
import sys

# Windows consoles default to cp1252; decoded BPE pieces can contain characters
# (e.g. U+FFFD) that crash a naive print(). Emit UTF-8 with a safe fallback.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
except Exception:
    pass

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_DIR = os.path.join(
    os.environ.get("BLACKWELL_MODELS_DIR", "F:/AI"), "llama3-8b-fp8")
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "dumps")
SEED = 1234


def dump_fp32(out_dir, name, t):
    arr = t.detach().cpu().float().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    arr.tofile(os.path.join(out_dir, f"{name}.bin"))
    return list(arr.shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--seq-len", type=int, default=194)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[setup] model : {args.model_dir}")
    print(f"[setup] dumps : {args.out_dir}")
    print(f"[setup] seq_len={args.seq_len}")

    print("[load] loading FP8 checkpoint (bf16 compute baseline)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, device_map="cuda:0", dtype=torch.bfloat16,
        low_cpu_mem_usage=True)
    model.eval()
    tok = AutoTokenizer.from_pretrained(args.model_dir)

    hidden = int(model.config.hidden_size)
    vocab = int(model.config.vocab_size)
    seq_len = args.seq_len
    print(f"[geom] hidden={hidden} vocab={vocab} layers={model.config.num_hidden_layers}")

    # --- Build the injected embedding matrix [seq_len, hidden] ----------------
    # Deterministic, seeded, in-distribution: embed real token ids so the
    # activations land in the FP8 range. Anchor the front with a readable prompt,
    # then fill to seq_len with seeded ids so the sequence is fully reproducible.
    prompt_ids = tok("<|begin_of_text|>Describe the audio.")["input_ids"]
    g = torch.Generator().manual_seed(SEED)
    fill = torch.randint(0, vocab, (max(0, seq_len - len(prompt_ids)),), generator=g).tolist()
    token_ids = (prompt_ids + fill)[:seq_len]
    ids = torch.tensor([token_ids], device="cuda:0")

    with torch.no_grad():
        embeds = model.model.embed_tokens(ids)            # [1, seq_len, hidden] bf16
        emb_shape = dump_fp32(args.out_dir, "08b_input_embeds", embeds[0])
        print(f"[embed] 08b_input_embeds {emb_shape}")

        # --- Forward from the injected embeddings (bypasses embed_tokens) -----
        out = model(inputs_embeds=embeds)
        logits = out.logits[0]                            # [seq_len, vocab] bf16
        log_shape = dump_fp32(args.out_dir, "08b_final_logits", logits)
        print(f"[logits] 08b_final_logits {log_shape}")

    # --- Top-5 at the LAST position (the next-token prediction) ---------------
    # Write the sidecars FIRST (UTF-8), so a console-encoding hiccup can never
    # leave the .bin dumps without their metadata.
    last = logits[-1].float()
    top5 = torch.topk(last, 5)
    ranked = list(zip(range(1, 6), top5.indices.tolist(), top5.values.tolist()))
    with open(os.path.join(args.out_dir, "08b_expected_tokens.txt"), "w", encoding="utf-8") as f:
        f.write("# rank\ttoken_id\tlogit\tstring  (LAST position, seq_len-1)\n")
        for rank, idx, val in ranked:
            f.write(f"{rank}\t{idx}\t{val:.6f}\t{tok.decode([idx])!r}\n")
    with open(os.path.join(args.out_dir, "08b_meta.txt"), "w") as f:
        f.write(f"{seq_len} {hidden} {vocab}\n")

    for rank, idx, val in ranked:
        print(f"[top5] rank {rank}: id={idx} logit={val:.4f} {tok.decode([idx])!r}")
    print(f"[done] wrote 08b_* dumps to {args.out_dir}")


if __name__ == "__main__":
    main()
