"""Golden-dump generator for the Ultravox audio frontend (encoder + projector).

Target: fixie-ai/ultravox-v0_5-llama-3_2-1b
  (whisper-large-v3-turbo encoder + Ultravox SwiGLU projector + Llama-3.2-1B).

Produces the PyTorch reference tensors consumed by the audio parity tests
(tests/integration/test_ultravox_frontend.cpp — added in Phase 5). The dumps are
the ground truth for the "cosine similarity > 0.999" parity bar; each stage of
the CUDA frontend is compared against the matching .bin here.

STATUS: SCAFFOLD. Ultravox ships as a *remote-code* model — its modules live in
the checkpoint's ultravox_model.py (UltravoxModel / UltravoxProjector /
ModifiedWhisperEncoder / StackAudioFrames), loaded via trust_remote_code=True.
Before first use, VERIFY every submodule path and the geometry block against the
downloaded modeling file (golden-dumps geometry-contract protocol). A renamed
submodule silently drops that dump and the C++ test then skips (a skip is NOT a
pass). Re-verify after any transformers / checkpoint upgrade.

Projector structure being captured (v0.5, projector_ln_mid=True):
    encoder [T,1280] -> StackAudioFrames(8) [T',10240] -> ln_pre(RMSNorm)
      -> linear_1(10240->4096) -> SwiGLU(->2048) -> ln_mid(RMSNorm)
      -> linear_2(2048->2048) -> audio_embeds [T',2048]

Path conventions match scripts/generate_qwen_dumps.py:
  * checkpoint root  <- BLACKWELL_MODELS_DIR (default F:/AI) + folder below
  * out dir          <- anchored to the repo root (parent of scripts/), NOT CWD

Usage:
    python scripts/generate_ultravox_audio_dumps.py
    python scripts/generate_ultravox_audio_dumps.py \
        --model-dir F:/AI/ultravox-v0_5-llama-3_2-1b \
        --out-dir tests/integration/golden_dumps/ultravox --audio path/to/clip.wav
"""

import argparse
import json
import os

# Set before importing torch on this machine (golden-dumps skill "Sharp edges").
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

DEFAULT_MODEL_DIR = os.path.join(
    os.environ.get("BLACKWELL_MODELS_DIR", "F:/AI"), "ultravox-v0_5-llama-3_2-1b")
DEFAULT_OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests", "integration", "golden_dumps", "ultravox",
)


def dump_tensor(out_dir, name, t):
    """Write a tensor as flat float32 little-endian (the C++ readers' format)."""
    arr = t.detach().to(torch.float32).cpu().numpy()
    arr.tofile(os.path.join(out_dir, f"{name}.bin"))
    return list(arr.shape)


def synth_audio(sample_rate, seconds=4.0):
    """Deterministic synthetic clip so the dump is reproducible without a WAV.
    Sum of tones + light noise exercises the full mel band. Use --audio for a
    real clip when a more representative parity signal is wanted."""
    rng = np.random.default_rng(0)
    n = int(sample_rate * seconds)
    t = np.arange(n) / sample_rate
    sig = (0.5 * np.sin(2 * np.pi * 220 * t)
           + 0.3 * np.sin(2 * np.pi * 440 * t)
           + 0.2 * np.sin(2 * np.pi * 880 * t)
           + 0.02 * rng.standard_normal(n))
    return sig.astype(np.float32)


def load_audio(path, target_sr):
    import soundfile as sf  # optional dep; only with --audio
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)  # downmix to mono
    if sr != target_sr:
        raise SystemExit(f"resample {sr}->{target_sr} Hz externally; not done here")
    return wav


def find_submodule(root, candidates):
    """Return the first attribute path that resolves, else raise — remote-code
    module trees drift between Ultravox versions, so fail loudly with guidance."""
    for path in candidates:
        obj = root
        ok = True
        for attr in path.split("."):
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok:
            return obj, path
    raise RuntimeError(
        f"none of {candidates} found on the model — inspect the checkpoint's "
        f"ultravox_model.py and update the hook paths (geometry contract).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--audio", default=None, help="mono 16 kHz WAV; else synthetic")
    args = parser.parse_args()

    from transformers import AutoModel, AutoProcessor

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[setup] model: {args.model_dir}\n[setup] dumps: {args.out_dir}")

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_dir, trust_remote_code=True,
        torch_dtype=torch.float32, device_map="cpu").eval()

    audio = load_audio(args.audio, 16000) if args.audio else synth_audio(16000)

    # DSP: the processor's feature_extractor yields the log-mel the encoder eats.
    feats = processor.feature_extractor(
        audio, sampling_rate=16000, return_tensors="pt")
    input_features = feats["input_features"]        # [1, 128, frames]
    print(f"[dsp] input_features {dump_tensor(args.out_dir, 'input_features', input_features)}")

    # Locate the audio tower + projector (remote-code paths differ by version).
    tower, tower_path = find_submodule(
        model, ["audio_tower", "audio_model", "model.audio_tower"])
    projector, proj_path = find_submodule(
        model, ["multi_modal_projector", "projector", "model.multi_modal_projector"])
    print(f"[modules] encoder='{tower_path}'  projector='{proj_path}'")

    captured = {}

    def cap(name):
        def hook(_m, _in, out):
            captured[name] = out[0] if isinstance(out, tuple) else out
        return hook

    # Stage probes. Adjust submodule names to the installed modeling file first.
    if hasattr(tower, "conv2"):
        tower.conv2.register_forward_hook(cap("conv_out"))
    layers = getattr(tower, "layers", None)
    if layers is not None:
        for i, layer in enumerate(layers):
            layer.register_forward_hook(cap(f"enc_layer_{i}"))
    if hasattr(tower, "layer_norm"):
        tower.layer_norm.register_forward_hook(cap("encoder_final_norm"))
    # Projector internals (names per UltravoxProjector; verify).
    for attr, tag in [("ln_pre", "proj_ln_pre"), ("linear_1", "proj_linear_1"),
                      ("act", "proj_swiglu"), ("ln_mid", "proj_ln_mid"),
                      ("linear_2", "proj_linear_2")]:
        if hasattr(projector, attr):
            getattr(projector, attr).register_forward_hook(cap(tag))

    with torch.no_grad():
        enc_out = tower(input_features)
        audio_hidden = getattr(enc_out, "last_hidden_state", enc_out)
        audio_embeds = projector(audio_hidden)

    for name, t in captured.items():
        dump_tensor(args.out_dir, name, t)
    embeds_shape = dump_tensor(args.out_dir, "audio_embeds", audio_embeds)

    cfg = model.config
    acfg = getattr(cfg, "audio_config", cfg)
    tcfg = getattr(cfg, "text_config", cfg)
    meta = {
        "model_dir": args.model_dir,
        "audio": args.audio or "synthetic-tones-4s",
        "encoder_path": tower_path,
        "projector_path": proj_path,
        "num_mel_bins": int(input_features.shape[1]),
        "encoder_layers": len(layers) if layers is not None else None,
        "d_model": int(getattr(acfg, "d_model", getattr(acfg, "hidden_size", -1))),
        "stack_factor": int(getattr(cfg, "stack_factor", -1)),
        "projector_hidden": int(getattr(cfg, "hidden_size", -1)),
        "projector_ln_mid": bool(getattr(cfg, "projector_ln_mid", False)),
        "text_hidden_size": int(getattr(tcfg, "hidden_size", -1)),
        "audio_token_index": int(getattr(cfg, "audio_token_index", -1)),
        "audio_embeds_shape": embeds_shape,
        "dumps": sorted(captured) + ["input_features", "audio_embeds"],
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] audio_embeds {embeds_shape} -> {args.out_dir}")


if __name__ == "__main__":
    main()
