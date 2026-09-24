"""Golden-dump generator for GLM-4-9B-Chat-1M (THUDM native checkpoint).

Produces the PyTorch reference tensors consumed by
tests/integration/test_glm4_engine.cpp.

WHY A MANUAL LAYER SWEEP
------------------------
The dumps the C++ test needs include tensors that are *local variables* inside
``SelfAttention.forward`` -- the post-RoPE query and key, and the attention
probability row -- which no module forward-hook can reach. So instead of hooking,
this script drives the checkpoint's OWN submodules (``layer.input_layernorm``,
``layer.self_attention.query_key_value``, ``apply_rotary_pos_emb``, ...) in the
order ``GLMBlock.forward`` runs them. Every weight and every op is the
checkpoint's; only the call sequence is ours, which is what makes the
intermediates observable.

The sweep is validated against the model's own ``forward()`` in the same run
(``manual_vs_hf_logits_cosine`` in meta.json, asserted >= 0.9999), so a mistake in
the replicated call order fails here rather than being mistaken for an engine bug.

TRANSFORMERS COMPATIBILITY
--------------------------
The bundled modeling_chatglm.py targets transformers 4.44; this environment runs
5.8. Two class/config hooks that 5.x requires and the 4.44-era code predates are
injected before loading (``all_tied_weights_keys``, the generation fields
``PretrainedConfig`` no longer carries). Nothing about the math is patched.

DEVICE
------
CPU, deliberately: 9.48 B bf16 parameters are 18.9 GB and the local card has
11.9 GB. Same reason generate_qwen_dumps.py runs its reference on CPU. A 6-token
forward takes well under a minute.

SEQUENCE LENGTH
---------------
The prompt is several tokens long on purpose. At a single position attention has
exactly one key, so its softmax is the constant 1.0 and its entropy is
identically 0 -- an entropy comparison at pos=0 tests nothing. Dumps hold the
LAST token's row of every tensor, i.e. the row whose attention sees the whole
prefix, which is also the row whose logits decide the next token.

Usage:
    python scripts/generate_glm4_dumps.py
    python scripts/generate_glm4_dumps.py --model-dir F:/AI/models/GLM-4-9B-Chat-1M \
        --out-dir tests/integration/golden_dumps/glm4_9b --probe-layers 0 19 39
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_DIR = os.path.join(
    os.environ.get("BLACKWELL_MODELS_DIR", "F:/AI/models"), "GLM-4-9B-Chat-1M")
DEFAULT_OUT_DIR = REPO_ROOT / "tests" / "integration" / "golden_dumps" / "glm4_9b"

# [gMASK] <sop> <|user|> \n "Hello" <|assistant|> -- a real GLM-4 chat prefix shape.
# Ids are resolved from the tokenizer when it loads; this is the fallback and the
# documented default so the dumps are reproducible without the remote tokenizer.
FALLBACK_TOKEN_IDS = [151331, 151333, 151336, 198, 9707, 151337]


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_model(model_dir: str, dtype: torch.dtype):
    """Load the THUDM checkpoint under a transformers version its code predates."""
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    # transformers 5.x moved these off PretrainedConfig; modeling_chatglm.py (4.44)
    # still reads them during __init__/generation setup.
    for key, value in dict(
        max_length=8192, max_new_tokens=None, min_length=0, do_sample=False,
        early_stopping=False, num_beams=1, temperature=1.0, top_k=50, top_p=1.0,
        repetition_penalty=1.0, bos_token_id=None, use_cache=True,
    ).items():
        if not hasattr(cfg, key):
            setattr(cfg, key, value)

    cls = get_class_from_dynamic_module(
        "modeling_chatglm.ChatGLMForConditionalGeneration", model_dir)
    if not hasattr(cls, "all_tied_weights_keys"):
        cls.all_tied_weights_keys = {}     # 5.x reads it; GLM-4 ties nothing

    model = cls.from_pretrained(model_dir, config=cfg, dtype=dtype, low_cpu_mem_usage=True)
    model.eval()

    # apply_rotary_pos_emb is a module-level @torch.jit.script function in the same
    # remote file; fetch it from the already-imported module rather than reimplement.
    rope_module = sys.modules[cls.__module__]
    return model, cfg, rope_module.apply_rotary_pos_emb


def resolve_tokens(model_dir: str, prompt: str | None) -> tuple[list[int], str]:
    if prompt is None:
        return FALLBACK_TOKEN_IDS, "fallback ids (chat prefix shape)"
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        ids = tok.encode(prompt)
        return ids, f"tokenizer.encode({prompt!r})"
    except Exception as exc:                      # noqa: BLE001 - diagnostic only
        print(f"[warn] tokenizer unavailable ({type(exc).__name__}: {exc}); "
              f"using fallback ids", flush=True)
        return FALLBACK_TOKEN_IDS, "fallback ids (tokenizer load failed)"


# --------------------------------------------------------------------------- #
# dumping
# --------------------------------------------------------------------------- #
class Dumper:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        self.written: dict[str, int] = {}

    def save(self, name: str, tensor: torch.Tensor) -> None:
        """Flat fp32 little-endian -- the format load_golden_bin() expects."""
        arr = tensor.detach().to(torch.float32).contiguous().cpu().numpy().ravel()
        arr.astype("<f4").tofile(self.out_dir / name)
        self.written[name] = arr.size

    def save_i32(self, name: str, values: list[int]) -> None:
        np.asarray(values, dtype="<i4").tofile(self.out_dir / name)
        self.written[name] = len(values)


def attention_probs_last_row(q_rope: torch.Tensor, k_rope: torch.Tensor,
                             head_dim: int, group: int) -> torch.Tensor:
    """Softmax row the LAST query position produces, per query head.

    q_rope: [1, n_q_heads, sq, hn]; k_rope: [1, n_kv_heads, sq, hn] (pre-expand).
    Computed in fp32 from the bf16 operands: SDPA itself accumulates in bf16, and
    the difference is far inside the 5% entropy tolerance the test applies -- while
    an fp32 softmax is the unambiguous reference for "what distribution is this".
    Returns [n_q_heads, sq].
    """
    n_q = q_rope.shape[1]
    sq = q_rope.shape[2]
    q_last = q_rope[0, :, -1, :].to(torch.float32)                   # [n_q, hn]
    k_all = k_rope[0].to(torch.float32)                              # [n_kv, sq, hn]
    k_exp = k_all.repeat_interleave(group, dim=0)                    # [n_q, sq, hn]
    scores = torch.einsum("hd,hsd->hs", q_last, k_exp) / math.sqrt(head_dim)
    # The last row of a causal mask is all-visible, so no masking is needed.
    assert scores.shape == (n_q, sq)
    return torch.softmax(scores, dim=-1)


def entropy_nats(probs: torch.Tensor) -> torch.Tensor:
    """-sum p ln p along the last dim, with the 0 ln 0 -> 0 convention."""
    p = probs.clamp_min(0.0)
    logp = torch.where(p > 0, p.log(), torch.zeros_like(p))
    return -(p * logp).sum(dim=-1)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--probe-layers", type=int, nargs="+", default=[0, 19, 39])
    ap.add_argument("--prompt", default=None,
                    help="text to tokenize; omitted => the documented fallback ids")
    ap.add_argument("--skip-hf-crosscheck", action="store_true",
                    help="skip the model.forward() comparison (halves the runtime)")
    args = ap.parse_args()

    model_dir = args.model_dir
    if not os.path.isdir(model_dir):
        print(f"[error] checkpoint directory not found: {model_dir}")
        return 2

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    dtype = torch.bfloat16
    print(f"[glm4-dumps] loading {model_dir} (cpu, {dtype})...", flush=True)
    model, cfg, apply_rotary_pos_emb = load_model(model_dir, dtype)

    token_ids, token_source = resolve_tokens(model_dir, args.prompt)
    ids = torch.tensor([token_ids], dtype=torch.long)
    seq_len = ids.shape[1]

    n_heads = cfg.num_attention_heads
    n_kv = cfg.multi_query_group_num
    head_dim = cfg.kv_channels
    group = n_heads // n_kv
    hidden = cfg.hidden_size
    inter = cfg.ffn_hidden_size
    q_dim = n_heads * head_dim
    kv_dim = n_kv * head_dim
    transformer = model.transformer
    probe = sorted(set(int(l) for l in args.probe_layers))
    n_layers = cfg.num_layers
    for l in probe:
        if not 0 <= l < n_layers:
            print(f"[error] probe layer {l} outside [0, {n_layers})")
            return 2

    dumper = Dumper(out_dir)
    meta: dict = {
        "model_dir": model_dir,
        "model_type": cfg.model_type,
        "token_ids": token_ids,
        "token_source": token_source,
        "seq_len": seq_len,
        "probe_layers": probe,
        "geometry": {
            "hidden_size": hidden, "ffn_hidden_size": inter,
            "num_attention_heads": n_heads, "multi_query_group_num": n_kv,
            "head_dim": head_dim, "gqa_group": group,
            "num_layers": n_layers, "padded_vocab_size": cfg.padded_vocab_size,
            "q_dim": q_dim, "kv_dim": kv_dim,
            "layernorm_epsilon": cfg.layernorm_epsilon,
            "rope_ratio": cfg.rope_ratio,
            "rope_theta": 10000.0 * cfg.rope_ratio,
            "seq_length_trained": cfg.seq_length,
        },
        "dump_row": "last token (index seq_len-1)",
        "layers": {},
    }

    dumper.save_i32("tokens.bin", token_ids)

    with torch.no_grad():
        # ---- rope cache, exactly as ChatGLMModel.forward builds it -------------
        # RotaryEmbedding(rotary_dim // 2) => n_elem = kv_channels // 2 = 64, so the
        # cache holds 32 (cos, sin) pairs and apply_rotary_pos_emb rotates the first
        # 64 channels as adjacent couples. Latched to bf16 by forward_impl.
        rope_cache_full = transformer.rotary_pos_emb(transformer.seq_length)
        rope_cache = rope_cache_full[None, :seq_len]
        meta["geometry"]["rope_pairs"] = int(rope_cache.shape[-2])
        meta["geometry"]["rope_rotary_dim"] = int(rope_cache.shape[-2]) * 2

        # ---- embedding --------------------------------------------------------
        h = transformer.embedding(ids)                      # [1, sq, hidden]
        dumper.save("embed_out.bin", h[0, -1])

        for idx, layer in enumerate(transformer.encoder.layers):
            is_probe = idx in probe
            tag = f"layer_{idx}_"

            ln1 = layer.input_layernorm(h)
            if is_probe:
                dumper.save(tag + "input_norm.bin", ln1[0, -1])

            sa = layer.self_attention
            qkv = sa.query_key_value(ln1)                   # [1, sq, q_dim + 2*kv_dim]
            q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)
            if is_probe:
                dumper.save(tag + "q_proj.bin", q[0, -1])
                dumper.save(tag + "k_proj.bin", k[0, -1])
                dumper.save(tag + "v_proj.bin", v[0, -1])

            q = q.view(1, seq_len, n_heads, head_dim).transpose(1, 2)   # [1,nq,sq,hn]
            k = k.view(1, seq_len, n_kv, head_dim).transpose(1, 2)
            v = v.view(1, seq_len, n_kv, head_dim).transpose(1, 2)

            q_rope = apply_rotary_pos_emb(q, rope_cache)
            k_rope = apply_rotary_pos_emb(k, rope_cache)
            if is_probe:
                # Head-major [n_heads, head_dim] for the last position -- the engine's
                # d_Q / d_K layout after its own RoPE.
                dumper.save(tag + "q_rope.bin", q_rope[0, :, -1, :])
                dumper.save(tag + "k_rope.bin", k_rope[0, :, -1, :])

            # GQA expand, then causal SDPA (attention_mask is None for an unpadded
            # prompt, so SdpaAttention takes its is_causal=True branch).
            k_exp = k_rope.unsqueeze(2).expand(-1, -1, group, -1, -1).reshape(
                1, n_heads, seq_len, head_dim)
            v_exp = v.unsqueeze(2).expand(-1, -1, group, -1, -1).reshape(
                1, n_heads, seq_len, head_dim)
            ctx = F.scaled_dot_product_attention(q_rope, k_exp, v_exp, is_causal=True)
            ctx = ctx.transpose(1, 2).contiguous().reshape(1, seq_len, q_dim)
            if is_probe:
                dumper.save(tag + "attn_math.bin", ctx[0, -1])

                probs = attention_probs_last_row(q_rope, k_rope, head_dim, group)
                ent = entropy_nats(probs)                    # [n_heads]
                dumper.save(tag + "attn_probs.bin", probs)    # [n_heads, sq]
                meta["layers"][str(idx)] = {
                    "attn_entropy_mean_nats": float(ent.mean()),
                    "attn_entropy_min_nats": float(ent.min()),
                    "attn_entropy_max_nats": float(ent.max()),
                    "attn_entropy_per_head_nats": [float(x) for x in ent],
                    "max_entropy_nats": float(math.log(seq_len)),
                }

            attn_out = sa.dense(ctx)
            h = h + attn_out                                 # residual 1
            if is_probe:
                dumper.save(tag + "attn_accum.bin", h[0, -1])

            ln2 = layer.post_attention_layernorm(h)
            if is_probe:
                dumper.save(tag + "post_attn_norm.bin", ln2[0, -1])

            gate_up = layer.mlp.dense_h_to_4h(ln2)           # [1, sq, 2*inter]
            if is_probe:
                dumper.save(tag + "gate_up.bin", gate_up[0, -1])

            gate, up = torch.chunk(gate_up, 2, dim=-1)       # chunk[0] IS the gate
            mlp_out = layer.mlp.dense_4h_to_h(F.silu(gate) * up)
            if is_probe:
                dumper.save(tag + "mlp_out.bin", mlp_out[0, -1])

            h = h + mlp_out                                  # residual 2
            if is_probe:
                dumper.save(tag + "accum_out.bin", h[0, -1])

        final = transformer.encoder.final_layernorm(h)
        dumper.save("final_norm_out.bin", final[0, -1])
        logits = transformer.output_layer(final)
        dumper.save("logits_out.bin", logits[0, -1])

        logits_f32 = logits[0, -1].to(torch.float32)
        top5 = torch.topk(logits_f32, 5)
        meta["greedy_top5"] = {
            "ids": [int(i) for i in top5.indices],
            "logits": [float(x) for x in top5.values],
        }
        meta["logits_entropy_nats"] = float(
            entropy_nats(torch.softmax(logits_f32, dim=-1)))

        # ---- chain of trust: the manual sweep vs the model's own forward -------
        if not args.skip_hf_crosscheck:
            print("[glm4-dumps] cross-checking against model.forward()...", flush=True)
            ref = model(input_ids=ids, return_dict=True).logits[0, -1].to(torch.float32)
            cos = float(F.cosine_similarity(ref, logits_f32, dim=0))
            max_abs = float((ref - logits_f32).abs().max())
            meta["manual_vs_hf_logits_cosine"] = cos
            meta["manual_vs_hf_logits_max_abs"] = max_abs
            meta["manual_vs_hf_top1_match"] = int(ref.argmax()) == int(logits_f32.argmax())
            print(f"           cosine {cos:.8f}  max|d| {max_abs:.6f}  "
                  f"top1 {'match' if meta['manual_vs_hf_top1_match'] else 'MISMATCH'}")
            if cos < 0.9999 or not meta["manual_vs_hf_top1_match"]:
                print("[error] the manual layer sweep does not reproduce the model's own "
                      "forward: the dumps would encode OUR mistake, not the model.")
                return 3

    meta["files"] = dumper.written
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)

    print(f"\n[glm4-dumps] wrote {len(dumper.written)} files to {out_dir}")
    print(f"             tokens={token_ids} (seq_len={seq_len}, {token_source})")
    for l in probe:
        e = meta["layers"][str(l)]
        print(f"             layer {l:>2}: attention entropy "
              f"{e['attn_entropy_mean_nats']:.4f} nats (max possible "
              f"{e['max_entropy_nats']:.4f})")
    print(f"             greedy top-1: {meta['greedy_top5']['ids'][0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
