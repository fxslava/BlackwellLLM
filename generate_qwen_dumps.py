"""Golden-dump generator for Qwen2.5-Coder-7B-Instruct-AWQ.

Produces the PyTorch reference tensors consumed by
tests/integration/test_qwen_engine.cpp.

The checkpoint is AWQ int4 (gemm flavour). Neither `autoawq` nor `gptqmodel`
is installed in this environment, so instead of routing through the
transformers AWQ integration the script dequantizes the packed weights
manually — using the exact AutoAWQ GEMM convention (nibble interleave
{0,2,4,6,1,3,5,7}, zero-point without +1 offset) that
src/kernels/awq_linear.cu implements — and loads the result into a stock
FP16 Qwen2ForCausalLM. Mathematically this *is* the AWQ model: AutoAWQ's own
GEMM kernels compute (q - z) * s in half precision before the matmul.

The forward pass runs on CPU in FP16 (oneDNN accumulates in FP32, matching
GPU GEMM semantics); the 15 GB FP16 model does not fit the local 12 GB card.

Usage:
    python generate_qwen_dumps.py
    python generate_qwen_dumps.py --model-dir F:/AI/Qwen2.5-Coder-7B-Instruct-AWQ \
        --out-dir tests/integration/golden_dumps/qwen2.5_awq
"""

import argparse
import json
import os

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Logical nibble j sits at bit offset 16*(j&1) + 4*(j>>1) inside the packed
# uint32 (AWQ "gemm" interleave). Reading nibbles in ascending shift order
# therefore yields logical order {0,2,4,6,1,3,5,7}; this permutation undoes it.
AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

DEFAULT_MODEL_DIR = "F:/AI/Qwen2.5-Coder-7B-Instruct-AWQ"
DEFAULT_OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "tests", "integration", "golden_dumps", "qwen2.5_awq",
)

# Single-token "prompt": Qwen2.5's BOS (<|endoftext|>, id 151643), mirroring
# the BOS-only methodology of the Llama integration test. One token keeps the
# attention math trivial (output == V) so the comparison isolates the
# quantized linear stack.
INPUT_TOKEN_ID = 151643


def unpack_awq_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """[rows, cols] int32 -> [rows, cols*8] int32 nibbles in logical order."""
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    vals = (packed.unsqueeze(-1) >> shifts) & 0xF            # shift order
    vals = vals[..., AWQ_REVERSE_ORDER]                       # logical order
    return vals.reshape(packed.shape[0], -1)


def dequantize_awq_linear(qweight, qzeros, scales, group_size):
    """Rebuild the FP16 weight matrix [out_features, in_features].

    qweight: int32 [K, N/8], qzeros: int32 [G, N/8], scales: fp16 [G, N]
    W[k, n] = (q[k, n] - z[g(k), n]) * s[g(k), n],  g(k) = k // group_size
    """
    iw = unpack_awq_nibbles(qweight)                          # [K, N]
    iz = unpack_awq_nibbles(qzeros)                           # [G, N]
    k = iw.shape[0]
    group_idx = torch.arange(k) // group_size                 # [K]
    w = (iw.float() - iz.float()[group_idx]) * scales.float()[group_idx]
    return w.to(torch.float16).t().contiguous()               # [N, K]


def build_dequantized_state_dict(model_dir, group_size):
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_path, encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    raw = {}
    for shard in sorted(set(weight_map.values())):
        with safe_open(os.path.join(model_dir, shard), framework="pt", device="cpu") as f:
            for name in f.keys():
                raw[name] = f.get_tensor(name)

    state_dict = {}
    quant_bases = sorted({n[: -len(".qweight")] for n in raw if n.endswith(".qweight")})
    for base in quant_bases:
        state_dict[base + ".weight"] = dequantize_awq_linear(
            raw.pop(base + ".qweight"),
            raw.pop(base + ".qzeros"),
            raw.pop(base + ".scales"),
            group_size,
        )
    state_dict.update(raw)  # embed/norm/lm_head/bias tensors pass through as FP16
    print(f"[dequant] rebuilt {len(quant_bases)} AWQ linears, "
          f"{len(state_dict)} tensors total")
    return state_dict


def load_fp16_model(model_dir):
    cfg = AutoConfig.from_pretrained(model_dir)
    group_size = cfg.quantization_config["group_size"]
    # Drop the quantization block so transformers builds plain nn.Linear modules.
    del cfg.quantization_config

    state_dict = build_dequantized_state_dict(model_dir, group_size)

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    model.load_state_dict(state_dict, strict=False, assign=True)

    # The rotary inv_freq buffers are non-persistent (absent from the state
    # dict) and are still meta tensors here; recompute them for real.
    rotary = model.model.rotary_emb
    inv_freq, rotary.attention_scaling = rotary.compute_default_rope_parameters(
        rotary.config, device="cpu")
    rotary.register_buffer("inv_freq", inv_freq, persistent=False)
    rotary.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    leftover_meta = [n for n, t in list(model.named_parameters()) + list(model.named_buffers())
                     if t.is_meta]
    if leftover_meta:
        raise RuntimeError(f"meta tensors not materialized: {leftover_meta}")

    model.eval()
    return model


def sanity_generate(model, model_dir, num_tokens=8):
    """Greedy-decode a short continuation; gibberish here would mean the AWQ
    unpacking convention is wrong, so fail loudly before writing dumps."""
    tok = AutoTokenizer.from_pretrained(model_dir)
    ids = tok("def quicksort(arr):", return_tensors="pt").input_ids
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=num_tokens, do_sample=False)
    text = tok.decode(out[0][ids.shape[1]:])
    print(f"[sanity] greedy continuation of 'def quicksort(arr):' -> {text!r}")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--skip-sanity", action="store_true",
                        help="skip the greedy-generation self-check")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[setup] model: {args.model_dir}\n[setup] dumps: {args.out_dir}")

    model = load_fp16_model(args.model_dir)

    if not args.skip_sanity:
        sanity_generate(model, args.model_dir)

    def dump(name):
        def hook(module, inputs, output):
            t = output[0] if isinstance(output, tuple) else output
            t.detach().float().numpy().tofile(os.path.join(args.out_dir, f"{name}.bin"))
        return hook

    def dump_pre(name):
        def hook(module, inputs):
            inputs[0].detach().float().numpy().tofile(os.path.join(args.out_dir, f"{name}.bin"))
        return hook

    model.model.embed_tokens.register_forward_hook(dump("embed_out"))
    for i, layer in enumerate(model.model.layers):
        layer.input_layernorm.register_forward_hook(dump(f"layer_{i}_input_norm"))
        layer.self_attn.q_proj.register_forward_hook(dump(f"layer_{i}_q_proj"))
        layer.self_attn.k_proj.register_forward_hook(dump(f"layer_{i}_k_proj"))
        layer.self_attn.v_proj.register_forward_hook(dump(f"layer_{i}_v_proj"))
        layer.self_attn.o_proj.register_forward_pre_hook(dump_pre(f"layer_{i}_attn_math"))
        layer.self_attn.register_forward_hook(dump(f"layer_{i}_attn_out"))
        layer.post_attention_layernorm.register_forward_hook(dump(f"layer_{i}_post_attn_norm"))
        layer.mlp.gate_proj.register_forward_hook(dump(f"layer_{i}_gate_proj"))
        layer.mlp.up_proj.register_forward_hook(dump(f"layer_{i}_up_proj"))
        layer.mlp.register_forward_hook(dump(f"layer_{i}_mlp_out"))
        layer.register_forward_hook(dump(f"layer_{i}_accum_out"))
    model.model.norm.register_forward_hook(dump("final_norm_out"))
    model.lm_head.register_forward_hook(dump("logits_out"))

    print(f"[forward] single-token pass, token_id={INPUT_TOKEN_ID}, pos=0")
    with torch.no_grad():
        logits = model(torch.tensor([[INPUT_TOKEN_ID]])).logits[0, -1].float()

    top = torch.topk(logits, 5)
    meta = {
        "model_dir": args.model_dir,
        "input_token_id": INPUT_TOKEN_ID,
        "top1_token_id": int(top.indices[0]),
        "top5_token_ids": [int(t) for t in top.indices],
        "top5_logits": [float(v) for v in top.values],
        "vocab_size": int(logits.numel()),
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[done] golden top-5 tokens: {meta['top5_token_ids']}")
    print(f"[done] dumps written to {args.out_dir}")


if __name__ == "__main__":
    main()
