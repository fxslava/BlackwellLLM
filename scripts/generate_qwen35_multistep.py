"""Multi-token DECODE parity reference for the Qwen3.5 hybrid checkpoint.

The single-step generator (generate_qwen35_dumps.py) only exercises pos=0, where
RoPE is the identity. This one feeds a short REAL token sequence through the HF
model in one causal forward and dumps the per-position logits, so the C++ test
can decode the same tokens one at a time (pos 0,1,2,...) and verify the engine's
autoregressive path -- crucially the partial-RoPE full-attention layers and the
recurrent SSM layers -- stays in parity for pos > 0.

Outputs (raw little-endian, into the same golden_dumps dir):
    multistep_tokens.bin   int32  [seq_len]            the prompt token ids
    multistep_logits.bin   fp32   [seq_len][vocab]     logits at every position

Usage:
    python scripts/generate_qwen35_multistep.py
    python scripts/generate_qwen35_multistep.py --prompt "The quick brown fox jumps"
"""

import argparse
import os
import numpy as np
import torch

# Checkpoint root comes from BLACKWELL_MODELS_DIR; out dir is anchored to the
# repo root (parent of scripts/), NOT the CWD -- same convention as the
# single-step generator, and the SAME golden_dumps dir.
DEFAULT_MODEL_DIR = os.path.join(
    os.environ.get("BLACKWELL_MODELS_DIR", "F:/AI"), "Qwen3.5-9B-AWQ-4bit")
DEFAULT_OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests", "integration", "golden_dumps", "qwen3.5_hybrid",
)


def load_model(model_dir, device):
    import transformers
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_dir)
    print(f"[load] arch={cfg.architectures}")
    last_err = None
    for cls_name in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
        try:
            cls = getattr(transformers, cls_name)
            model = cls.from_pretrained(model_dir, dtype=torch.bfloat16,
                                        device_map=device, low_cpu_mem_usage=True)
            print(f"[load] loaded via {cls_name}")
            return model.eval(), cfg
        except Exception as e:  # noqa: BLE001
            print(f"[load] {cls_name} failed: {type(e).__name__}: {e}")
            last_err = e
    raise RuntimeError(f"could not load {model_dir}") from last_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog.")
    ap.add_argument("--max-tokens", type=int, default=12,
                    help="cap the PROMPT length")
    ap.add_argument("--generate", type=int, default=0,
                    help="greedy-generate this many tokens and append them (tests a long "
                         "single autoregressive run, like the playground's CoT)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model, cfg = load_model(args.model_dir, args.device)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    ids = tok(args.prompt, return_tensors="pt").input_ids[0].tolist()
    ids = ids[: args.max_tokens]
    dev = next(model.parameters()).device

    # Optional greedy continuation so the dumped sequence resembles a real (long)
    # decode. Done with HF's own KV cache -- if THIS loops, looping is the model's
    # nature, not an engine bug.
    if args.generate > 0:
        gen = model.generate(torch.tensor([ids], device=dev),
                             max_new_tokens=args.generate, do_sample=False,
                             temperature=None, top_p=None, top_k=None)
        ids = gen[0].tolist()
        print(f"[generate] HF greedy continuation -> {len(ids)} total tokens")
        print(f"[generate] text: {tok.decode(ids)!r}")

    input_ids = torch.tensor([ids], device=dev)
    print(f"[tokens] {ids}")

    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True, return_dict=True)
    logits = out.logits[0].float().cpu().numpy()   # [seq_len, vocab]

    np.asarray(ids, dtype="<i4").tofile(os.path.join(args.out_dir, "multistep_tokens.bin"))
    logits.astype("<f4", copy=False).tofile(os.path.join(args.out_dir, "multistep_logits.bin"))
    print(f"[dump] multistep_tokens.bin [{len(ids)}]")
    print(f"[dump] multistep_logits.bin {logits.shape}")

    # Per-position argmax for a quick eyeball.
    am = logits.argmax(axis=-1).tolist()
    print(f"[argmax/pos] {am}")


if __name__ == "__main__":
    main()
