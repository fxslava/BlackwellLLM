"""Golden-dump generator for the FULL Ultravox-8B multimodal pipeline.

Chains the three real inference stages and dumps the PyTorch reference at each
boundary (the C++ test asserts cosine > 0.999 against these):

  encoder_last_hidden [1500,1280]                 (reused Whisper-large-v3 output)
    -> Ultravox-8B projector  ---> uv8b_audio_embeds   [188, 4096]   (Stage 1)
    -> prompt injector splice ---> uv8b_spliced_embeds [194, 4096]   (Stage 2)
    -> local llama3-8b-fp8     ---> uv8b_final_logits   [194, vocab]  (Stage 3)

CREDENTIAL-FREE / OFFLINE. Two local checkpoints, no HF network:
  * projector : F:/AI/ultravox-v0_5-llama-3_1-8b   (audio_tower + multi_modal_projector)
  * backbone  : F:/AI/llama3-8b-fp8                (FP8 Llama-3 8B; scaled_mm reference)
The Ultravox checkpoint bundles NO Llama backbone, so Stage 3 uses the local FP8
model -- the SAME checkpoint the C++ engine loads, so parity is well-defined.

REAL 8B PROJECTOR GEOMETRY (from the safetensors header -- NOT the 10240->8192
figure in some briefs; the 8B projector keeps the 1B intermediate dims and only
widens linear_2's OUTPUT to the 4096 backbone hidden):
  ln_pre[10240] -> linear_1[4096,10240] (10240->4096) -> SwiGLU(4096->2048)
  -> ln_mid[2048] -> linear_2[4096,2048] (2048->4096) -> audio_embeds[.,4096]

Dumps land in <repo>/dumps (gitignored -- ~100 MB for the logits, regenerated
locally). Usage:
    python scripts/dump_ultravox8b_pipeline.py
"""

import argparse
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
except Exception:
    pass

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.environ.get("BLACKWELL_MODELS_DIR", "F:/AI")
DEFAULT_PROJECTOR_DIR = os.path.join(MODELS, "ultravox-v0_5-llama-3_1-8b")
DEFAULT_LLAMA_DIR = os.path.join(MODELS, "llama3-8b-fp8")
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "dumps")
# Reused Whisper encoder output (same audio_tower family, [1500,1280]).
ENCODER_INPUT = os.path.join(
    REPO_ROOT, "tests", "integration", "golden_dumps", "ultravox", "encoder_last_hidden.bin")

AUDIO_TOKEN_ID = 128256  # config.audio_token_index (the <|audio|> placeholder)
STACK = 8
WHISPER_DIM = 1280
SEED = 1234


def dump_fp32(out_dir, name, t):
    arr = t.detach().cpu().float().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    arr.tofile(os.path.join(out_dir, f"{name}.bin"))
    return list(arr.shape)


def rmsnorm(x, weight, eps=1e-6):
    var = x.pow(2).mean(-1, keepdim=True)
    return weight * (x * torch.rsqrt(var + eps))


def stack_audio_frames(x, stack_factor=8):
    B, T, C = x.shape
    T_pad = (T + stack_factor - 1) // stack_factor * stack_factor
    x = F.pad(x, (0, 0, 0, T_pad - T))
    return x.view(B, T_pad // stack_factor, C * stack_factor)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--projector-dir", default=DEFAULT_PROJECTOR_DIR)
    ap.add_argument("--llama-dir", default=DEFAULT_LLAMA_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--encoder-input", default=ENCODER_INPUT)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[setup] projector: {args.projector_dir}")
    print(f"[setup] backbone : {args.llama_dir}")
    print(f"[setup] dumps    : {args.out_dir}")

    # --- Stage 0: encoder output (reused) ------------------------------------
    enc = np.fromfile(args.encoder_input, dtype=np.float32)
    n_frames = enc.size // WHISPER_DIM
    enc = torch.from_numpy(enc.reshape(1, n_frames, WHISPER_DIM))
    dump_fp32(args.out_dir, "uv8b_encoder_input", enc[0])
    print(f"[stage0] encoder_last_hidden {list(enc.shape)}")

    # --- Stage 1: Ultravox-8B projector --------------------------------------
    proj = os.path.join(args.projector_dir, "model.safetensors")
    with safe_open(proj, framework="pt", device="cpu") as f:
        w_ln_pre   = f.get_tensor("multi_modal_projector.ln_pre.weight").float()
        w_linear_1 = f.get_tensor("multi_modal_projector.linear_1.weight").float()
        w_ln_mid   = f.get_tensor("multi_modal_projector.ln_mid.weight").float()
        w_linear_2 = f.get_tensor("multi_modal_projector.linear_2.weight").float()
    print(f"[stage1] weights linear_1{list(w_linear_1.shape)} linear_2{list(w_linear_2.shape)} "
          f"ln_pre[{w_ln_pre.shape[0]}] ln_mid[{w_ln_mid.shape[0]}]")
    for name, w in [("uv8b_proj_ln_pre", w_ln_pre), ("uv8b_proj_linear_1", w_linear_1),
                    ("uv8b_proj_ln_mid", w_ln_mid), ("uv8b_proj_linear_2", w_linear_2)]:
        dump_fp32(args.out_dir, name, w)

    with torch.no_grad():
        h = stack_audio_frames(enc, STACK)          # [1, 188, 10240]
        h = rmsnorm(h, w_ln_pre)
        h = F.linear(h, w_linear_1)                 # [1, 188, 4096]
        a, gate = h.chunk(2, dim=-1)                # SwiGLU (Ultravox: gate = 2nd half)
        h = F.silu(gate) * a                        # [1, 188, 2048]
        h = rmsnorm(h, w_ln_mid)
        h = F.linear(h, w_linear_2)                 # [1, 188, 4096]
    audio = h[0]
    hidden = audio.shape[-1]
    num_audio = audio.shape[0]
    emb_shape = dump_fp32(args.out_dir, "uv8b_audio_embeds", audio)
    print(f"[stage1] uv8b_audio_embeds {emb_shape}")

    # --- Stage 2: prompt-injector splice -------------------------------------
    # Synthetic text embeddings (the injector is a pure row-copy, value-independent);
    # the audio rows are the REAL projector output above.
    tok = AutoTokenizer.from_pretrained(args.projector_dir)
    ids = tok("<|audio|>\nDescribe the audio.")["input_ids"]
    if AUDIO_TOKEN_ID not in ids:  # be robust to tokenizer variance
        ids = [AUDIO_TOKEN_ID] + ids
    seq_len = len(ids)
    audio_pos = ids.index(AUDIO_TOKEN_ID)
    rng = np.random.default_rng(SEED)
    text = torch.from_numpy((rng.standard_normal((seq_len, hidden)) * 0.05).astype(np.float32))
    dump_fp32(args.out_dir, "uv8b_text_embeds", text)

    spliced = torch.cat([text[:audio_pos], audio, text[audio_pos + 1:]], dim=0)
    spl_shape = dump_fp32(args.out_dir, "uv8b_spliced_embeds", spliced)
    with open(os.path.join(args.out_dir, "uv8b_injector_meta.txt"), "w") as f:
        f.write(f"{seq_len} {audio_pos} {num_audio} {hidden}\n")
    print(f"[stage2] prompt ids={ids} audio_pos={audio_pos} -> uv8b_spliced_embeds {spl_shape}")

    # --- Stage 3: local llama3-8b-fp8 prefill from spliced embeds -------------
    print("[stage3] loading FP8 backbone (bf16 compute)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.llama_dir, device_map="cuda:0", dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval()
    vocab = int(model.config.vocab_size)
    with torch.no_grad():
        emb = spliced.unsqueeze(0).to("cuda:0", dtype=torch.bfloat16)  # [1,194,4096]
        logits = model(inputs_embeds=emb).logits[0]                    # [194, vocab]
    log_shape = dump_fp32(args.out_dir, "uv8b_final_logits", logits)
    print(f"[stage3] uv8b_final_logits {log_shape}")

    ltok = AutoTokenizer.from_pretrained(args.llama_dir)
    top5 = torch.topk(logits[-1].float(), 5)
    ranked = list(zip(range(1, 6), top5.indices.tolist(), top5.values.tolist()))
    with open(os.path.join(args.out_dir, "uv8b_expected_tokens.txt"), "w", encoding="utf-8") as f:
        f.write("# rank\ttoken_id\tlogit\tstring  (LAST position)\n")
        for rank, idx, val in ranked:
            f.write(f"{rank}\t{idx}\t{val:.6f}\t{ltok.decode([idx])!r}\n")
    with open(os.path.join(args.out_dir, "uv8b_pipeline_meta.txt"), "w") as f:
        f.write(f"{spl_shape[0]} {hidden} {vocab}\n")   # final seq_len, hidden, vocab

    for rank, idx, val in ranked:
        print(f"[top5] rank {rank}: id={idx} logit={val:.4f} {ltok.decode([idx])!r}")
    print(f"[done] wrote uv8b_* pipeline dumps to {args.out_dir}")


if __name__ == "__main__":
    main()
