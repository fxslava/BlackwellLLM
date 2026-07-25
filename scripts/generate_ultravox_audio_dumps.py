"""Golden-dump generator for the Ultravox audio frontend (encoder + projector).

Target: fixie-ai/ultravox-v0_5-llama-3_2-1b
  (whisper-large-v3-turbo encoder + Ultravox SwiGLU projector + Llama-3.2-1B).

Produces the PyTorch reference tensors consumed by the audio parity tests
(tests/integration/test_ultravox_frontend.cpp — added in Phase 5). The dumps are
the ground truth for the "cosine similarity > 0.999" parity bar.

Reference audio is REAL human speech — the standard Whisper/LibriSpeech dummy
clip (hf-internal-testing/librispeech_asr_dummy), NOT a synthetic tone — so the
DSP/encoder parity is exercised on a representative spectrum. The exact clip is
written to test_audio.wav so the C++ engine loads bit-identical samples.

CREDENTIAL-FREE BY DESIGN. The Ultravox checkpoint bundles ONLY the audio_tower
(Whisper encoder) + multi_modal_projector weights; the Llama backbone is not
included, and instantiating UltravoxModel/UltravoxConfig triggers a *gated* fetch
of meta-llama/Llama-3.2-1B-Instruct. We never need Llama for the audio frontend,
so this script sidesteps it entirely:
  * DSP        -> WhisperFeatureExtractor.from_pretrained(<local dir>)  (local cfg)
  * encoder    -> a stock, version-matched transformers WhisperEncoder built from
                  the embedded audio_config, with the bundled audio_tower.* weights
  * projector  -> the exact UltravoxProjector math, reconstructed functionally
                  from the bundled multi_modal_projector.* weights + config:
                    StackAudioFrames(8) -> RMSNorm(1e-6) -> linear_1(10240->4096)
                    -> SwiGLU(->2048) -> RMSNorm -> linear_2(2048->2048)
                  (verified against fixie-ai/ultravox ultravox_model.py).

NOTE ON THE ENCODER REFERENCE: stock WhisperEncoder requires the canonical 30 s
(3000-mel-frame) input, so the log-mel is padded to 3000 before the encoder. This
matches canonical Whisper. Ultravox ships a *ModifiedWhisperEncoder* that also
supports unpadded variable-length audio; for the padded 30 s case the forward math
is the same. Phase 3 should re-verify the per-layer dumps against
ModifiedWhisperEncoder if variable-length parity becomes load-bearing (flagged in
ULTRAVOX_AUDIO_PLAN.md §8).

Usage:
    python scripts/generate_ultravox_audio_dumps.py
    python scripts/generate_ultravox_audio_dumps.py --skip-encoder   # DSP only
"""

import argparse
import io
import json
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

DEFAULT_MODEL_DIR = os.path.join(
    os.environ.get("BLACKWELL_MODELS_DIR", "F:/AI"), "ultravox-v0_5-llama-3_2-1b")
DEFAULT_OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests", "integration", "golden_dumps", "ultravox",
)
TARGET_SR = 16000
WHISPER_FRAMES = 3000   # canonical 30 s: max_source_positions(1500) * conv2 stride(2)


def dump(out_dir, name, t):
    arr = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    arr.tofile(os.path.join(out_dir, f"{name}.bin"))
    return list(arr.shape)


def load_librispeech_sample():
    """Canonical Whisper test utterance (real speech, 16 kHz mono). Decoded via
    soundfile with Audio(decode=False) so we never depend on the datasets audio
    backend (torchcodec/ffmpeg), which drifts across releases."""
    from datasets import Audio, load_dataset
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean",
                      split="validation")
    ds = ds.cast_column("audio", Audio(decode=False))   # -> {"path","bytes"}
    entry = ds[0]["audio"]
    src = io.BytesIO(entry["bytes"]) if entry.get("bytes") else entry["path"]
    wav, sr = sf.read(src, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        raise SystemExit(f"librispeech sample is {sr} Hz, expected {TARGET_SR}")
    return wav.astype(np.float32), sr, ds[0].get("text", "")


def rmsnorm(x, weight, eps=1e-6):
    """LlamaRMSNorm (Ultravox projector norm), fp32 throughout."""
    var = x.pow(2).mean(-1, keepdim=True)
    return weight * (x * torch.rsqrt(var + eps))


def stack_audio_frames(x, stack_factor=8):
    """Ultravox StackAudioFrames: zero-pad time to a multiple of stack_factor,
    then reshape [B,T,C] -> [B, T/sf, C*sf]."""
    B, T, C = x.shape
    T_pad = (T + stack_factor - 1) // stack_factor * stack_factor
    x = F.pad(x, (0, 0, 0, T_pad - T))
    return x.view(B, T_pad // stack_factor, C * stack_factor)


def dump_projector_weights(out_dir, model_dir):
    """Dump the multi_modal_projector WEIGHT matrices (distinct from the per-stage
    ACTIVATION dumps proj_*.bin). Cheap: safe_open fetches only these 4 tensors.
        w_ln_pre   [10240]         w_linear_1 [4096, 10240]
        w_ln_mid   [2048]          w_linear_2 [2048, 2048]
    FP32 (checkpoint is fp16); these feed the C++ end-to-end projector parity."""
    from safetensors import safe_open
    names = {"ln_pre.weight": "w_ln_pre", "linear_1.weight": "w_linear_1",
             "ln_mid.weight": "w_ln_mid", "linear_2.weight": "w_linear_2"}
    path = os.path.join(model_dir, "model.safetensors")
    with safe_open(path, framework="pt", device="cpu") as f:
        for src, dst in names.items():
            shp = dump(out_dir, dst, f.get_tensor("multi_modal_projector." + src).float())
            print(f"[weights] {dst} {shp}")


AUDIO_TOKEN_ID = 128256  # `<|audio|>` placeholder (config.audio_token_index)


def dump_injector_tensors(out_dir, model_dir):
    """Reference for the prompt-injector kernel: splice the projector's audio
    embeddings into the text-embedding sequence at the `<|audio|>` position.

    The real Llama-3.2-1B embed_tokens matrix is gated/absent, so the TEXT
    embeddings are deterministic synthetic values. This is faithful for a splice
    parity test: the injector is a pure row-copy (out = concat(text[:pos], audio,
    text[pos+1:])), independent of the embedding VALUES; the audio rows are the
    real projector output (audio_embeds.bin)."""
    from transformers import AutoTokenizer
    hidden = 2048
    audio_embeds = np.fromfile(
        os.path.join(out_dir, "audio_embeds.bin"), dtype=np.float32).reshape(-1, hidden)
    num_audio = audio_embeds.shape[0]

    tok = AutoTokenizer.from_pretrained(model_dir)
    ids = tok("<|audio|>\nDescribe the audio.")["input_ids"]
    seq_len = len(ids)
    audio_pos = ids.index(AUDIO_TOKEN_ID)

    rng = np.random.default_rng(1234)
    text_embeds = (rng.standard_normal((seq_len, hidden)) * 0.05).astype(np.float32)
    dump(out_dir, "text_embeds", text_embeds)

    spliced = np.concatenate(
        [text_embeds[:audio_pos], audio_embeds, text_embeds[audio_pos + 1:]], axis=0)
    dump(out_dir, "spliced_embeds_ref", spliced)

    # Plain-int sidecar so the C++ test needs no JSON parser.
    with open(os.path.join(out_dir, "injector_meta.txt"), "w") as f:
        f.write(f"{seq_len} {audio_pos} {num_audio} {hidden}\n")
    print(f"[injector] prompt ids={ids}")
    print(f"[injector] text_embeds [{seq_len},{hidden}] audio_pos={audio_pos} "
          f"num_audio={num_audio} -> spliced_embeds_ref [{spliced.shape[0]},{hidden}]")


def run_encoder_and_projector(out_dir, model_dir, cfg, input_features, captured_meta):
    """Stock WhisperEncoder + functional Ultravox projector over bundled weights."""
    from safetensors.torch import load_file
    from transformers import WhisperConfig
    from transformers.models.whisper.modeling_whisper import WhisperEncoder

    weights = load_file(os.path.join(model_dir, "model.safetensors"))

    # --- Whisper encoder (stock, version-matched) ---
    wcfg = WhisperConfig(**cfg["audio_config"])
    enc = WhisperEncoder(wcfg).eval()
    enc_sd = {k[len("audio_tower."):]: v.float()
              for k, v in weights.items() if k.startswith("audio_tower.")}
    missing, unexpected = enc.load_state_dict(enc_sd, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected encoder keys: {unexpected[:5]}")
    print(f"[encoder] WhisperEncoder loaded ({len(enc_sd)} tensors, "
          f"{len(missing)} non-weight buffers left default)")

    # Pad/trim log-mel to the canonical 3000 frames the stock encoder expects.
    feats = input_features.float()
    T = feats.shape[-1]
    if T < WHISPER_FRAMES:
        feats = F.pad(feats, (0, WHISPER_FRAMES - T))
    elif T > WHISPER_FRAMES:
        feats = feats[..., :WHISPER_FRAMES]
    print(f"[encoder] input log-mel {list(input_features.shape)} -> "
          f"encoder input {list(feats.shape)}")

    captured = {}

    def cap(name):
        def hook(_m, _i, o):
            captured[name] = (o[0] if isinstance(o, tuple) else o).detach()
        return hook

    for i, layer in enumerate(enc.layers):
        layer.register_forward_hook(cap(f"enc_layer_{i}"))
    enc.layer_norm.register_forward_hook(cap("encoder_final_norm"))

    with torch.no_grad():
        # True conv subsample reference (post-GELU, permuted) = encoder inputs_embeds.
        c1 = F.gelu(F.conv1d(feats, enc.conv1.weight, enc.conv1.bias, padding=1))
        c2 = F.gelu(F.conv1d(c1, enc.conv2.weight, enc.conv2.bias, stride=2, padding=1))
        conv_out = c2.permute(0, 2, 1)                       # [B, 1500, 1280]
        dump(out_dir, "conv_out", conv_out)

        out = enc(feats)
        audio_features = getattr(out, "last_hidden_state",
                                 out[0] if isinstance(out, tuple) else out)

    for name, t in captured.items():
        dump(out_dir, name, t)
    dump(out_dir, "encoder_last_hidden", audio_features)

    # --- Ultravox projector (functional, exact) ---
    def w(name):
        return weights[f"multi_modal_projector.{name}"].float()

    with torch.no_grad():
        h = stack_audio_frames(audio_features, cfg["stack_factor"])   # [B,T',10240]
        dump(out_dir, "proj_stacked", h)
        h = rmsnorm(h, w("ln_pre.weight"))
        dump(out_dir, "proj_ln_pre", h)
        h = F.linear(h, w("linear_1.weight"))                         # [B,T',4096]
        dump(out_dir, "proj_linear_1", h)
        a, gate = h.chunk(2, dim=-1)                                  # SwiGLU
        h = F.silu(gate) * a                                          # [B,T',2048]
        dump(out_dir, "proj_swiglu", h)
        h = rmsnorm(h, w("ln_mid.weight"))                            # projector_ln_mid=True
        dump(out_dir, "proj_ln_mid", h)
        h = F.linear(h, w("linear_2.weight"))                         # [B,T',2048]
        embeds_shape = dump(out_dir, "audio_embeds", h)

    print(f"[projector] audio_embeds shape {embeds_shape}")
    captured_meta.update({
        "encoder_layers": len(enc.layers),
        "d_model": wcfg.d_model,
        "stack_factor": cfg["stack_factor"],
        "projector_hidden": cfg["hidden_size"],
        "projector_ln_mid": cfg["projector_ln_mid"],
        "text_hidden_size": embeds_shape[-1],
        "audio_token_index": cfg["audio_token_index"],
        "audio_embeds_shape": embeds_shape,
        "encoder_reference": "stock WhisperEncoder, 30s-padded (see module docstring)",
        "encoder_dumps": sorted(captured) + [
            "conv_out", "encoder_last_hidden", "proj_stacked", "proj_ln_pre",
            "proj_linear_1", "proj_swiglu", "proj_ln_mid", "audio_embeds"],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--skip-encoder", action="store_true",
                        help="only produce the DSP artifacts (wav/mel/log-mel)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[setup] model: {args.model_dir}")
    print(f"[setup] dumps: {args.out_dir}")

    cfg = json.load(open(os.path.join(args.model_dir, "config.json")))

    # --- 1. Reference audio -> test_audio.wav --------------------------------
    audio, sr, text = load_librispeech_sample()
    wav_path = os.path.join(args.out_dir, "test_audio.wav")
    sf.write(wav_path, audio, sr)
    print(f"[audio] librispeech: {audio.shape[0]} samples @ {sr} Hz "
          f"({audio.shape[0]/sr:.2f} s); transcript={text!r}")
    print(f"[audio] wrote {wav_path}")

    # --- 2. DSP: mel filterbank + log-mel (feature extractor only, LOCAL) ----
    # The checkpoint's preprocessor_config.json has a STALE feature_size=80; the
    # real UltravoxProcessor ignores it and loads the feature extractor from the
    # audio model (openai/whisper-large-v3-turbo -> 128 mel bins), which is what
    # the 128-channel conv1 requires. Override feature_size=128 to reproduce the
    # whisper-large-v3-turbo filterbank offline (all other params already match:
    # n_fft=400, hop=160, chunk_length=30).
    from transformers import WhisperFeatureExtractor
    fe = WhisperFeatureExtractor.from_pretrained(args.model_dir, feature_size=128)
    assert fe.feature_size == 128, fe.feature_size

    mel = np.asarray(fe.mel_filters, dtype=np.float32)
    mel_128x201 = mel if mel.shape[0] < mel.shape[1] else mel.T.copy()   # want [128,201]
    mel_shape = dump(args.out_dir, "mel_filters", mel_128x201)
    print(f"[dsp] mel_filters {mel_shape}  (raw fe.mel_filters {list(mel.shape)})")

    feats = fe(audio, sampling_rate=sr, return_tensors="pt")
    input_features = feats["input_features"]                   # [1,128,T]
    log_mel_shape = dump(args.out_dir, "expected_log_mel", input_features)
    dump(args.out_dir, "input_features", input_features)       # alias
    print(f"[dsp] LOG-MEL SPECTROGRAM shape {log_mel_shape} -> expected_log_mel.bin")
    print("[ok] DSP artifacts: test_audio.wav, mel_filters.bin, expected_log_mel.bin")

    # --- 2b. Projector WEIGHT matrices (for the C++ end-to-end parity) --------
    dump_projector_weights(args.out_dir, args.model_dir)

    # --- 2c. Prompt-injector reference (needs audio_embeds.bin from an encoder run)
    if os.path.exists(os.path.join(args.out_dir, "audio_embeds.bin")):
        dump_injector_tensors(args.out_dir, args.model_dir)
    else:
        print("[injector] skipped: audio_embeds.bin absent "
              "(run once without --skip-encoder first)")

    meta = {
        "model_dir": args.model_dir,
        "audio_source": "hf-internal-testing/librispeech_asr_dummy[validation][0]",
        "transcript": text,
        "sample_rate": int(sr),
        "num_mel_bins": int(input_features.shape[1]),
        "mel_filters_shape": mel_shape,
        "log_mel_shape": log_mel_shape,
    }

    # --- 3. Encoder + projector stage dumps (credential-free) ----------------
    if not args.skip_encoder:
        try:
            run_encoder_and_projector(args.out_dir, args.model_dir, cfg,
                                      input_features, meta)
            print("[ok] encoder + projector stage dumps written")
        except Exception as e:  # never let this block the DSP trio
            import traceback
            print(f"[warn] encoder/projector dumps skipped: {type(e).__name__}: {e}")
            traceback.print_exc()
            meta["encoder_error"] = f"{type(e).__name__}: {e}"

    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] dumps -> {args.out_dir}")


if __name__ == "__main__":
    main()
