"""Module A -- capture the real input activations of GLM-4-9B's layer-10 down_proj.

Weight-only SQNR is a property of the weights alone; Act-SNR is a property of the
weights *and* the activations that hit them. This script produces the missing half:
the exact tensor X that ``model.layers.<L>.mlp.down_proj`` sees on real tokens.

WHY A TRUNCATED MODEL
---------------------
A transformer is strictly feed-forward across layers, so layer L's input depends on
layers 0..L-1 and on nothing above. Instantiating only ``num_hidden_layers = L+1``
therefore yields *bit-identical* activations at layer L while fitting the 11-layer
prefix of a 9.4 B-parameter bf16 model (5.8 GB) into the 11.9 GB local card -- the
full model is 18.9 GB and would have to be CPU-offloaded. The truncation is
verified, not assumed: ``_assert_weights_match`` compares the loaded down_proj
against the raw safetensors tensor, so a silent ``from_pretrained`` key mismatch
fails here rather than being mistaken for a quantization result later.

WHY TWO LAYERS BY DEFAULT
-------------------------
Layer 10 is the layer under study. Layer 3 is captured in the *same* forward pass at
zero extra cost, and exists only so that a conclusion which happens to be an
artifact of one layer's activation statistics can be caught.

WHAT IS SAVED
-------------
X only (bf16, [tokens, 13696] -- 14 MB per layer), not W. The evaluation reads W
straight from the checkpoint shards via ``e8_lattice_engine.load_down_proj``, which
keeps a 112 MB tensor out of the artifact and the two steps independent.

USAGE
    python research/capture_activations.py                  # 4 x 128 = 512 tokens
    python research/capture_activations.py --num-seqs 8 --seq-len 256 --layers 10
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Dict, List

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e8_lattice_engine import DEFAULT_MODEL_DIR, load_down_proj  # noqa: E402

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


def build_token_batch(model_dir: str, num_seqs: int, seq_len: int,
                      dataset: str, split: str, seed: int) -> torch.Tensor:
    """One [num_seqs, seq_len] batch of real prose tokens.

    The corpus is concatenated into a single stream and the batch is drawn from
    ``num_seqs`` well-separated offsets rather than from the first N tokens, so the
    calibration set is not one passage about one topic.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    ds = load_dataset(dataset, "wikitext-2-raw-v1", split=split)
    text = "\n".join(t for t in ds["text"] if t.strip())
    # tokenize a generous prefix: ~6 chars/token is a safe overestimate of what we need
    need_chars = num_seqs * seq_len * 24
    ids = tok(text[:need_chars], return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    total = num_seqs * seq_len
    if ids.numel() < total:
        raise RuntimeError(f"corpus gave {ids.numel()} tokens, need {total}")
    stride = ids.numel() // num_seqs
    rows: List[torch.Tensor] = []
    for i in range(num_seqs):
        start = i * stride
        rows.append(ids[start:start + seq_len])
    return torch.stack(rows)


def _assert_weights_match(model, model_dir: str, layers: List[int]) -> None:
    """The truncated load must reproduce the checkpoint's tensors exactly."""
    for layer in layers:
        ref = load_down_proj(model_dir, layer).to(torch.float32)
        got = model.layers[layer].mlp.down_proj.weight.detach().to("cpu", torch.float32)
        if ref.shape != got.shape or not torch.equal(ref, got):
            raise RuntimeError(
                f"layer {layer} down_proj does not match the checkpoint "
                f"(shapes {tuple(ref.shape)} vs {tuple(got.shape)}) -- the truncated "
                f"from_pretrained load mapped keys wrongly, so captured activations "
                f"would be meaningless")
    print(f"  weight check OK: layers {layers} match the safetensors shards bit-for-bit")


def capture(model_dir: str, layers: List[int], num_seqs: int, seq_len: int,
            dataset: str, split: str, seed: int, device: str) -> Dict[int, dict]:
    warnings.filterwarnings("ignore")
    from transformers.models.glm import GlmConfig, GlmModel

    max_layer = max(layers)
    cfg = GlmConfig.from_pretrained(model_dir)
    full_layers = cfg.num_hidden_layers
    cfg.num_hidden_layers = max_layer + 1
    print(f"instantiating layers 0..{max_layer} of {full_layers} "
          f"(hidden={cfg.hidden_size}, intermediate={cfg.intermediate_size})")
    model = GlmModel.from_pretrained(model_dir, config=cfg, dtype=torch.bfloat16)
    _assert_weights_match(model, model_dir, layers)
    model = model.to(device).eval()
    print(f"  resident on {device}: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
          if device.startswith("cuda") else "  resident on cpu")

    grabbed: Dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(_mod, args):
            # pre-forward hook: args[0] is the down_proj input, i.e. the
            # post-activation hidden state of width `intermediate_size`.
            grabbed[layer_idx] = args[0].detach().reshape(-1, args[0].shape[-1]).clone()
        return hook

    handles = [model.layers[i].mlp.down_proj.register_forward_pre_hook(make_hook(i))
               for i in layers]

    ids = build_token_batch(model_dir, num_seqs, seq_len, dataset, split, seed)
    print(f"  calibration batch {tuple(ids.shape)} = {ids.numel()} tokens "
          f"from {dataset}[{split}]")
    with torch.no_grad():
        model(input_ids=ids.to(device))
    for h in handles:
        h.remove()

    out: Dict[int, dict] = {}
    for layer in layers:
        x = grabbed[layer]
        xf = x.to(torch.float32)
        out[layer] = {
            "x": x.to(torch.bfloat16).cpu(),
            "layer": layer,
            "tokens": int(ids.numel()),
            "num_seqs": num_seqs,
            "seq_len": seq_len,
            "dataset": f"{dataset}/wikitext-2-raw-v1[{split}]",
            "model_dir": model_dir,
            "x_rms": float(xf.pow(2).mean().sqrt()),
            "x_absmax": float(xf.abs().max()),
            "x_kurtosis": float(xf.pow(4).mean() / xf.pow(2).mean() ** 2),
        }
        print(f"  layer {layer:2d}: X {tuple(x.shape)} rms={out[layer]['x_rms']:.4f} "
              f"absmax={out[layer]['x_absmax']:.4f} kurt={out[layer]['x_kurtosis']:.1f}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--layers", type=int, nargs="+", default=[3, 10],
                    help="down_proj layers to hook; the deepest one sets the load size")
    ap.add_argument("--num-seqs", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--dataset", default="Salesforce/wikitext")
    ap.add_argument("--split", default="train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default=ARTIFACT_DIR)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    caps = capture(args.model_dir, sorted(args.layers), args.num_seqs, args.seq_len,
                   args.dataset, args.split, args.seed, args.device)
    for layer, blob in caps.items():
        path = os.path.join(args.out_dir, f"calib_act_layer{layer}.pt")
        torch.save(blob, path)
        print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
