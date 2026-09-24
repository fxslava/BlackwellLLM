"""Convert a THUDM-native GLM-4 checkpoint to the HF-native layout the engine serves.

WHY THIS EXISTS
---------------
THUDM ships glm-4-9b-chat(-1m) with `trust_remote_code` weights in its own tensor
namespace:

    transformer.embedding.word_embeddings.weight
    transformer.encoder.layers.N.self_attention.query_key_value.{weight,bias}   FUSED
    transformer.encoder.layers.N.self_attention.dense.weight
    transformer.encoder.layers.N.mlp.dense_h_to_4h.weight                        FUSED
    transformer.encoder.layers.N.mlp.dense_4h_to_h.weight
    transformer.encoder.final_layernorm.weight
    transformer.output_layer.weight

BlackwellLLM serves the HF-native GLM layout (`model.layers.N.self_attn.*` +
`mlp.gate_up_proj`), and `ConfigLoader` refuses the THUDM one by name rather than
mis-loading it -- see docs/GLM4_TURBOQUANT_INTEGRATION.md §1.3. This script is the
bridge: it renames tensors and SLICES the fused QKV into q/k/v. No arithmetic is
performed on any value, so the converted checkpoint is bit-identical to the
original modulo tensor naming and row partitioning -- which is what makes a parity
comparison against dumps taken from the ORIGINAL checkpoint legitimate.

THE TWO FUSED TENSORS
---------------------
`query_key_value` is `[q_dim + 2*kv_dim, hidden]` and SelfAttention.forward splits
its OUTPUT as [q | k | v] in that order, so q/k/v are contiguous ROW blocks of the
weight (and of the bias). `dense_h_to_4h` is `[2*ffn, hidden]` and MLP.swiglu does
`chunk(2)[0] -> silu` , i.e. gate first -- the identical convention to HF's
`gate_up_proj`, so that tensor is a pure rename with NO reordering.

Usage:
    python scripts/convert_glm4_thudm_to_hf.py
    python scripts/convert_glm4_thudm_to_hf.py \
        --src F:/AI/models/GLM-4-9B-Chat-1M --dst F:/AI/models/GLM-4-9B-Chat-1M-hf
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

DEFAULT_SRC = os.path.join(
    os.environ.get("BLACKWELL_MODELS_DIR", "F:/AI/models"), "GLM-4-9B-Chat-1M")


def build_config(src_cfg: dict) -> dict:
    """The HF-native GLM config the engine's ConfigLoader expects.

    model_type "glm" (not "glm4"): this checkpoint has no post_self_attn_layernorm /
    post_mlp_layernorm, so sandwich norms must stay OFF. rope_theta is written out
    explicitly (10000 * rope_ratio) rather than leaving the engine to re-derive it.
    """
    head_dim = src_cfg.get("kv_channels") or (
        src_cfg["hidden_size"] // src_cfg["num_attention_heads"])
    return {
        "architectures": ["GlmForCausalLM"],
        "model_type": "glm",
        "hidden_size": src_cfg["hidden_size"],
        "intermediate_size": src_cfg["ffn_hidden_size"],
        "num_hidden_layers": src_cfg.get("num_layers") or src_cfg["num_hidden_layers"],
        "num_attention_heads": src_cfg["num_attention_heads"],
        "num_key_value_heads": src_cfg["multi_query_group_num"],
        "head_dim": head_dim,
        "vocab_size": src_cfg["padded_vocab_size"],
        "rms_norm_eps": src_cfg["layernorm_epsilon"],
        # GLM rotates only head_dim/2 channels, as adjacent pairs.
        "partial_rotary_factor": 0.5,
        "rope_theta": 10000.0 * src_cfg.get("rope_ratio", 1),
        "attention_bias": bool(src_cfg.get("add_qkv_bias", False)),
        "mlp_bias": bool(src_cfg.get("add_bias_linear", False)),
        "tie_word_embeddings": bool(src_cfg.get("tie_word_embeddings", False)),
        "max_position_embeddings": src_cfg.get("seq_length", 131072),
        "torch_dtype": src_cfg.get("torch_dtype", "bfloat16"),
        "eos_token_id": src_cfg.get("eos_token_id"),
        "pad_token_id": src_cfg.get("pad_token_id"),
        # Provenance, so a converted tree is never mistaken for an upstream one.
        "_converted_from": src_cfg.get("_name_or_path", "THUDM GLM-4 (chatglm)"),
        "_converted_by": "scripts/convert_glm4_thudm_to_hf.py",
    }


# Non-parameter buffers the engine recomputes on device and must NOT carry over.
# Listed explicitly so that an unrecognised tensor still aborts the conversion:
# silently dropping a real weight is exactly the failure this script must not have.
DROPPED_BUFFERS = (
    "transformer.rotary_pos_emb.inv_freq",   # RoPE ladder; rebuilt from rope_theta
)


def rename(name: str) -> str | None:
    """THUDM tensor name -> HF-native name. None for the two fused tensors, which
    the caller handles, and for anything unrecognised (reported, not dropped)."""
    if name == "transformer.embedding.word_embeddings.weight":
        return "model.embed_tokens.weight"
    if name == "transformer.encoder.final_layernorm.weight":
        return "model.norm.weight"
    if name == "transformer.output_layer.weight":
        return "lm_head.weight"

    prefix = "transformer.encoder.layers."
    if not name.startswith(prefix):
        return None
    rest = name[len(prefix):]
    layer, _, tail = rest.partition(".")
    base = f"model.layers.{layer}."
    table = {
        "input_layernorm.weight": "input_layernorm.weight",
        "post_attention_layernorm.weight": "post_attention_layernorm.weight",
        "self_attention.dense.weight": "self_attn.o_proj.weight",
        "mlp.dense_h_to_4h.weight": "mlp.gate_up_proj.weight",
        "mlp.dense_4h_to_h.weight": "mlp.down_proj.weight",
    }
    if tail in table:
        return base + table[tail]
    if tail.startswith("self_attention.query_key_value."):
        return None            # fused: split by the caller
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--dst", default=None, help="default: <src>-hf")
    ap.add_argument("--shard-bytes", type=int, default=4 * 1024**3,
                    help="target bytes per output shard (default 4 GiB)")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst) if args.dst else src.parent / (src.name + "-hf")
    if not src.is_dir():
        print(f"[error] source checkpoint not found: {src}")
        return 2

    with open(src / "config.json", encoding="utf-8") as f:
        src_cfg = json.load(f)
    if src_cfg.get("model_type") != "chatglm":
        print(f"[error] {src/'config.json'} is model_type "
              f"{src_cfg.get('model_type')!r}, not 'chatglm' -- nothing to convert.")
        return 2

    hidden = src_cfg["hidden_size"]
    n_heads = src_cfg["num_attention_heads"]
    n_kv = src_cfg["multi_query_group_num"]
    head_dim = src_cfg.get("kv_channels") or (hidden // n_heads)
    q_dim, kv_dim = n_heads * head_dim, n_kv * head_dim

    index_path = src / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path, encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        shard_of = {k: src / v for k, v in weight_map.items()}
    else:
        only = src / "model.safetensors"
        with safe_open(only, framework="pt") as fh:
            shard_of = {k: only for k in fh.keys()}

    dst.mkdir(parents=True, exist_ok=True)
    print(f"[convert] {src.name} -> {dst}")
    print(f"[convert] geometry: hidden={hidden} heads={n_heads} kv_heads={n_kv} "
          f"head_dim={head_dim} (qkv rows {q_dim}+{kv_dim}+{kv_dim})")

    # Group source tensors by shard so each file opens exactly once.
    by_shard: dict[Path, list[str]] = {}
    for name, path in shard_of.items():
        by_shard.setdefault(path, []).append(name)

    out_index: dict[str, str] = {}
    pending: dict[str, torch.Tensor] = {}
    pending_bytes = 0
    shard_no = 0
    shard_files: list[tuple[str, dict[str, torch.Tensor]]] = []
    unhandled: list[str] = []
    total_out = 0

    def flush() -> None:
        nonlocal pending, pending_bytes, shard_no
        if not pending:
            return
        shard_no += 1
        fname = f"model-{shard_no:05d}.safetensors"
        shard_files.append((fname, pending))
        pending = {}
        pending_bytes = 0

    def emit(name: str, tensor: torch.Tensor) -> None:
        nonlocal pending_bytes, total_out
        pending[name] = tensor
        pending_bytes += tensor.numel() * tensor.element_size()
        total_out += 1
        if pending_bytes >= args.shard_bytes:
            flush()

    for shard_path in sorted(by_shard):
        with safe_open(shard_path, framework="pt") as fh:
            for name in sorted(by_shard[shard_path]):
                if name in DROPPED_BUFFERS:
                    print(f"[convert]   dropping recomputed buffer {name}")
                    continue
                t = fh.get_tensor(name)
                if ".self_attention.query_key_value." in name:
                    layer = name.split("layers.")[1].split(".")[0]
                    kind = "bias" if name.endswith(".bias") else "weight"
                    if t.shape[0] != q_dim + 2 * kv_dim:
                        print(f"[error] {name} has leading dim {t.shape[0]}, expected "
                              f"{q_dim + 2*kv_dim}; refusing to guess the split.")
                        return 3
                    q, k, v = t[:q_dim], t[q_dim:q_dim + kv_dim], t[q_dim + kv_dim:]
                    base = f"model.layers.{layer}.self_attn."
                    emit(base + f"q_proj.{kind}", q.clone())
                    emit(base + f"k_proj.{kind}", k.clone())
                    emit(base + f"v_proj.{kind}", v.clone())
                    continue
                new = rename(name)
                if new is None:
                    unhandled.append(name)
                    continue
                emit(new, t)
        print(f"[convert]   read {shard_path.name}")
    flush()

    if unhandled:
        print(f"[error] {len(unhandled)} source tensor(s) had no mapping, e.g. "
              f"{unhandled[:5]} -- refusing to write a checkpoint with holes.")
        return 3

    for fname, tensors in shard_files:
        save_file(tensors, str(dst / fname), metadata={"format": "pt"})
        for k in tensors:
            out_index[k] = fname
        print(f"[convert]   wrote {fname}  ({len(tensors)} tensors)")

    total_bytes = sum((dst / f).stat().st_size for f, _ in shard_files)
    with open(dst / "model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump({"metadata": {"total_size": total_bytes}, "weight_map": out_index},
                  f, indent=1)
    with open(dst / "config.json", "w", encoding="utf-8") as f:
        json.dump(build_config(src_cfg), f, indent=2)

    # Tokenizer assets travel along so the converted tree is self-contained.
    for asset in ("tokenizer.model", "tokenizer_config.json", "tokenizer.json",
                  "generation_config.json", "special_tokens_map.json"):
        if (src / asset).exists():
            shutil.copy2(src / asset, dst / asset)

    print(f"\n[convert] {total_out} tensors, {total_bytes / 2**30:.2f} GiB, "
          f"{len(shard_files)} shard(s)")
    print(f"[convert] index: {dst / 'model.safetensors.index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
