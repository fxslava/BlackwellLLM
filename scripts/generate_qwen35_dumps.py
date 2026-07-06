"""Golden-dump generator for the Qwen3.5 hybrid checkpoint
(F:/AI/Qwen3.5-9B-AWQ-4bit, arch ``Qwen3_5ForConditionalGeneration``).

Produces the raw little-endian FP32 reference tensors consumed by
``tests/integration/test_qwen3_5_hybrid_integration.cpp`` (single-token DECODE
parity, zero recurrent state -> one step).

Why this model is special
-------------------------
Qwen3.5 is a *hybrid* stack: 24 GatedDeltaNet "linear attention" layers + 8
periodic full-attention layers (``full_attention_interval=4``). The recurrent
linear layers carry two pieces of state that, on transformers >=5, live inside
the generic ``DynamicCache`` *per layer*:

    cache.layers[i].recurrent_states   # GatedDeltaNet matrix S
    cache.layers[i].conv_states        # causal-conv1d ring buffer

(verified in transformers/models/qwen3_5/modeling_qwen3_5.py: the mixer reads
``cache_params.layers[self.layer_idx].conv_states / .recurrent_states`` and
writes back via ``update_conv_state`` / ``update_recurrent_state``.)

Full-attention layers store K/V in those same ``cache.layers[i]`` slots instead,
so we select linear layers from ``config.text_config.layer_types`` and read the
recurrent attributes only there.

Geometry (must match SsmGeometry::from_config in src/ssm/ssm_state_pool.h)
-------------------------------------------------------------------------
    num_value_heads (H) = 32,  key_head_dim (Dk) = 128,  value_head_dim (Dv) = 128
    recurrent state S : [H][Dk][Dv]            = 32*128*128 = 524288 / layer
    conv_dim          : 2*(num_key_heads*key_head_dim) + (num_value_heads*value_head_dim)
                      = 2*(16*128) + (32*128)  = 8192
    conv ring buffer  : [conv_dim][K-1]        = 8192*3     = 24576  / layer   (K=4)
    num_linear_layers = 24

NOTE on the conv buffer width: HF keeps the *full* K=4 window in
``conv_states``; the engine's ``init_conv_state`` is K-1=3 wide. For the very
first token both are all-zeros, so we synthesize the init buffers directly from
the engine geometry (guaranteeing the exact element count the C++ test reads)
rather than reshaping HF's tensor.

Output files (engine contract, see the test header)
---------------------------------------------------
    input_embedding.bin   [hidden]                       embedding of the decode token
    init_ssm_state.bin     [n_lin][H][Dk][Dv]   ZEROS    pre-step recurrent state S
    init_conv_state.bin    [n_lin][conv_dim][K-1] ZEROS   pre-step conv ring buffer
    expected_logits.bin    [vocab]                        logits after one decode step
Extra (requested for inspection; the test does not assert these):
    expected_ssm_state.bin   [n_lin][H][Dk][Dv]           post-step recurrent state S
    expected_conv_state.bin  [n_lin][conv_dim][K]         post-step conv window (native HF, K wide)
    meta.json                                              token id, shapes, top-5 logits

Usage
-----
    python scripts/generate_qwen35_dumps.py
    python scripts/generate_qwen35_dumps.py --token-id 0 --device cuda
    python scripts/generate_qwen35_dumps.py --model-dir F:/AI/Qwen3.5-9B-AWQ-4bit \
        --out-dir tests/integration/golden_dumps/qwen3.5_hybrid
"""

import argparse
import json
import os

import numpy as np
import torch

# Checkpoint root comes from BLACKWELL_MODELS_DIR; out dir is anchored to the
# repo root (parent of scripts/), NOT the CWD, so the dumps always land where
# the C++ integration test reads them.
DEFAULT_MODEL_DIR = os.path.join(
    os.environ.get("BLACKWELL_MODELS_DIR", "F:/AI"), "Qwen3.5-9B-AWQ-4bit")
DEFAULT_OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests", "integration", "golden_dumps", "qwen3.5_hybrid",
)

# The engine's test drives forward(decode_token=0, pos=0) and (currently) does
# NOT overwrite d_X_accum with input_embedding.bin, so the HF reference MUST run
# the SAME token id for logit parity. Keep this in lockstep with the C++ side.
DEFAULT_TOKEN_ID = 0


def _to_f32_bin(t: torch.Tensor, path: str) -> tuple:
    """Save a tensor as raw little-endian FP32 (no header). Returns its shape."""
    arr = t.detach().to(torch.float32).cpu().contiguous().numpy()
    # numpy is native-endian; on x86/ARM that is little-endian, matching the
    # C++ std::ifstream reader. Be explicit anyway for portability.
    arr.astype("<f4", copy=False).tofile(path)
    return tuple(arr.shape)


def load_model(model_dir: str, device: str):
    """Load the hybrid VLM. compressed-tensors keeps the int4 weights packed and
    dequantizes per-forward, so VRAM stays near the packed footprint."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_dir)
    print(f"[load] arch={cfg.architectures} dtype={getattr(cfg, 'dtype', '?')}")

    last_err = None
    for cls_name in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
        try:
            import transformers
            cls = getattr(transformers, cls_name)
            model = cls.from_pretrained(
                model_dir,
                dtype=torch.bfloat16,
                device_map=device,
                low_cpu_mem_usage=True,
            )
            print(f"[load] loaded via {cls_name}")
            return model.eval(), cfg
        except Exception as e:  # noqa: BLE001 - report and try the next auto class
            print(f"[load] {cls_name} failed: {type(e).__name__}: {e}")
            last_err = e
    raise RuntimeError(f"could not load {model_dir}") from last_err


def find_decoder_layers(model, num_hidden_layers: int):
    """Locate the text decoder ModuleList (model.language_model.layers, possibly
    nested under the multimodal wrapper). Returns (module_path, ModuleList)."""
    import torch.nn as nn

    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) == num_hidden_layers:
            child = mod[0]
            if hasattr(child, "linear_attn") or hasattr(child, "self_attn"):
                return name, mod
    raise RuntimeError("could not find the text decoder layer list")


def get_layer_cache(cache, layer_idx: int):
    """transformers>=5 layered cache: per-layer object at cache.layers[i].
    Older fallbacks (dict/list attrs) are handled defensively."""
    layers = getattr(cache, "layers", None)
    if layers is not None:
        return layers[layer_idx]
    # Defensive fallbacks for alternate cache layouts.
    for rec_attr, conv_attr in (("recurrent_states", "conv_states"),):
        rec = getattr(cache, rec_attr, None)
        conv = getattr(cache, conv_attr, None)
        if rec is not None or conv is not None:
            class _Shim:  # expose the per-layer view uniformly
                pass
            s = _Shim()
            s.recurrent_states = rec[layer_idx] if rec is not None else None
            s.conv_states = conv[layer_idx] if conv is not None else None
            return s
    raise RuntimeError("unrecognized cache structure; inspect cache attributes")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--token-id", type=int, default=DEFAULT_TOKEN_ID,
                    help="single decode token id (MUST match the engine's decode_token)")
    ap.add_argument("--device", default="cuda",
                    help="'cuda', 'cuda:0', or 'cpu' (falls back to cpu on CUDA OOM)")
    args = ap.parse_args()

    torch.manual_seed(0)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[setup] model : {args.model_dir}")
    print(f"[setup] dumps : {args.out_dir}")
    print(f"[setup] token : {args.token_id}  pos=0  (zero initial state)")

    try:
        model, cfg = load_model(args.model_dir, args.device)
    except torch.cuda.OutOfMemoryError:
        print("[load] CUDA OOM -> retrying on CPU (slow)")
        model, cfg = load_model(args.model_dir, "cpu")
        args.device = "cpu"

    # ---- resolve geometry from text_config (mirror SsmGeometry::from_config) --
    tc = cfg.text_config
    hidden = tc.hidden_size
    vocab = tc.vocab_size
    n_layers = tc.num_hidden_layers
    layer_types = list(tc.layer_types)
    H = tc.linear_num_value_heads          # 32  (state / value heads)
    Dk = tc.linear_key_head_dim            # 128
    Dv = tc.linear_value_head_dim          # 128
    K = tc.linear_conv_kernel_dim          # 4
    key_dim = tc.linear_num_key_heads * tc.linear_key_head_dim      # 16*128 = 2048
    val_dim = tc.linear_num_value_heads * tc.linear_value_head_dim  # 32*128 = 4096
    conv_dim = 2 * key_dim + val_dim       # q + k + v  -> 8192

    linear_layer_indices = [i for i, t in enumerate(layer_types)
                            if t == "linear_attention"]
    n_lin = len(linear_layer_indices)
    print(f"[geom] hidden={hidden} vocab={vocab} layers={n_layers} "
          f"(linear={n_lin}, full={n_layers - n_lin})")
    print(f"[geom] H={H} Dk={Dk} Dv={Dv} conv_dim={conv_dim} K={K} "
          f"=> rec/layer={H * Dk * Dv}, conv(ring)/layer={conv_dim * (K - 1)}")

    # ---- input embedding ------------------------------------------------------
    embed_layer = model.get_input_embeddings()
    dev = next(embed_layer.parameters()).device
    input_ids = torch.tensor([[args.token_id]], device=dev)
    with torch.no_grad():
        embed_vec = embed_layer(input_ids)[0, 0]          # [hidden]
    shp = _to_f32_bin(embed_vec, os.path.join(args.out_dir, "input_embedding.bin"))
    print(f"[dump] input_embedding.bin {shp}")

    # ---- PRE-step states are exactly zero for the first token -----------------
    # Build them straight from the engine geometry so the element counts match
    # the C++ injection (init_conv_state is K-1 wide, NOT HF's K-wide window).
    init_ssm = np.zeros((n_lin, H, Dk, Dv), dtype="<f4")
    init_conv = np.zeros((n_lin, conv_dim, K - 1), dtype="<f4")
    init_ssm.tofile(os.path.join(args.out_dir, "init_ssm_state.bin"))
    init_conv.tofile(os.path.join(args.out_dir, "init_conv_state.bin"))
    print(f"[dump] init_ssm_state.bin  {init_ssm.shape} (zeros)")
    print(f"[dump] init_conv_state.bin {init_conv.shape} (zeros)")

    # ---- optional per-layer telemetry: capture each GatedDeltaNet output -------
    _, layers = find_decoder_layers(model, n_layers)
    captured = {}

    def make_hook(li):
        def hook(_m, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            captured[li] = t.detach()
        return hook

    handles = []
    for li, model_idx in enumerate(linear_layer_indices):
        layer = layers[model_idx]
        if hasattr(layer, "linear_attn"):
            handles.append(layer.linear_attn.register_forward_hook(make_hook(li)))

    # ---- the single decode step (zero state -> one token) ---------------------
    print(f"[forward] one decode step, token_id={args.token_id}")
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True, return_dict=True)
    for h in handles:
        h.remove()

    logits = out.logits[0, -1].float()                    # [vocab]
    shp = _to_f32_bin(logits, os.path.join(args.out_dir, "expected_logits.bin"))
    print(f"[dump] expected_logits.bin {shp}")

    # ---- POST-step recurrent / conv states from the cache ---------------------
    cache = out.past_key_values
    if cache is None:
        raise RuntimeError("no past_key_values returned; use_cache failed")
    print(f"[cache] type={type(cache).__name__} "
          f"layers={len(getattr(cache, 'layers', [])) or '?'}")

    post_ssm = np.zeros((n_lin, H, Dk, Dv), dtype="<f4")
    post_conv = np.zeros((n_lin, conv_dim, K), dtype="<f4")   # HF keeps full K window
    for li, model_idx in enumerate(linear_layer_indices):
        lc = get_layer_cache(cache, model_idx)
        rec = getattr(lc, "recurrent_states", None)
        conv = getattr(lc, "conv_states", None)
        if li == 0:
            print(f"[cache] linear layer {model_idx}: "
                  f"recurrent_states={None if rec is None else tuple(rec.shape)}, "
                  f"conv_states={None if conv is None else tuple(conv.shape)}")
        if rec is not None:
            post_ssm[li] = rec[0].detach().to(torch.float32).cpu().numpy()
        if conv is not None:
            c = conv[0].detach().to(torch.float32).cpu().numpy()
            post_conv[li, :, -c.shape[-1]:] = c   # right-align in case width < K
    post_ssm.tofile(os.path.join(args.out_dir, "expected_ssm_state.bin"))
    post_conv.tofile(os.path.join(args.out_dir, "expected_conv_state.bin"))
    print(f"[dump] expected_ssm_state.bin  {post_ssm.shape}")
    print(f"[dump] expected_conv_state.bin {post_conv.shape} (K-wide, HF native)")

    # per-layer GatedDeltaNet output telemetry (cosine sanity, not asserted)
    for li, t in sorted(captured.items()):
        shp = _to_f32_bin(t.reshape(-1),
                          os.path.join(args.out_dir, f"linear_attn_out_layer{li}.bin"))
        if li == 0:
            print(f"[dump] linear_attn_out_layer{li}.bin {shp} (+{len(captured) - 1} more)")

    # ---- meta / sanity --------------------------------------------------------
    top = torch.topk(logits, 5)
    meta = {
        "model_dir": args.model_dir,
        "token_id": args.token_id,
        "hidden": hidden, "vocab": vocab,
        "num_linear_layers": n_lin, "num_full_layers": n_layers - n_lin,
        "H": H, "Dk": Dk, "Dv": Dv, "conv_dim": conv_dim, "K": K,
        "linear_layer_indices": linear_layer_indices,
        "rec_elems_per_layer": H * Dk * Dv,
        "conv_ring_elems_per_layer": conv_dim * (K - 1),
        "top5_token_ids": [int(i) for i in top.indices],
        "top5_logits": [float(v) for v in top.values],
        "ssm_state_after_norm": float(np.linalg.norm(post_ssm)),
        "conv_state_after_norm": float(np.linalg.norm(post_conv)),
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[done] top-1 next token = {meta['top5_token_ids'][0]} "
          f"(logit {meta['top5_logits'][0]:.4f})")
    print(f"[done] |S_after|={meta['ssm_state_after_norm']:.4f} "
          f"|conv_after|={meta['conv_state_after_norm']:.4f}")
    print(f"[done] dumps written to {args.out_dir}")


if __name__ == "__main__":
    main()
