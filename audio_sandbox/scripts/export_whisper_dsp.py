#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# export_whisper_dsp.py
#
# Golden-parity dumper for the Whisper (large-v3-turbo) audio front-end that
# `fixie-ai/ultravox-v0_5-llama-3_2-1b` uses. Produces three artifacts the C++
# sandbox consumes:
#
#   data/test_audio.wav        16 kHz mono 16-bit PCM dummy clip (deterministic).
#   data/mel_filters.bin       float32, shape (N_FREQS=201, N_MELS=128), row-major.
#                              Pulled straight from transformers' feature extractor
#                              so the (Slaney-normalised) filterbank is authentic —
#                              C++ NEVER recomputes it, it just loads this.
#   data/expected_log_mel.bin  float32, shape (N_MELS=128, N_FRAMES), row-major.
#                              The reference log-mel the C++ pipeline must match.
#
# PARITY CONTRACT (both sides MUST agree bit-for-bit on the algorithm):
#   * waveform fed to the golden == int16 / 32768.0  (exactly what dr_wav's f32
#     read path yields), so the input samples are identical on both sides.
#   * periodic Hann window of length n_fft (== torch.hann_window default).
#   * torch.stft center=True  -> reflect-pad the signal by n_fft//2 each side.
#   * drop the LAST stft time frame  (transformers does stft[..., :-1]).
#   * power spectrum = |stft|**2 ; mel = mel_filters.T @ power.
#   * log10(clamp(mel, 1e-10)) ; floor at (global_max - 8) ; (x + 4) / 4.
#
# This is exactly WhisperFeatureExtractor._torch_extract_fbank_features, run on
# the RAW (un-padded-to-30s) waveform so the C++ processes the same samples.
# -----------------------------------------------------------------------------
import argparse
import os
import sys
import wave

import numpy as np

# Whisper-large-v3-turbo DSP geometry (verify against the loaded extractor below).
SR = 16000
N_FFT = 400
HOP = 160
N_MELS = 128
N_FREQS = N_FFT // 2 + 1  # 201


def generate_dummy_wav(path: str, sr: int = SR, seconds: float = 3.0) -> np.ndarray:
    """Deterministic non-trivial mono clip: two tones + a chirp + light noise.

    Returns the int16 PCM samples actually written (so the caller can derive the
    exact float waveform dr_wav will hand the C++ side)."""
    n = int(round(sr * seconds))
    t = np.arange(n, dtype=np.float64) / sr
    sig = 0.40 * np.sin(2 * np.pi * 220.0 * t)
    sig += 0.30 * np.sin(2 * np.pi * 440.0 * t)
    f0, f1 = 200.0, 3000.0
    sig += 0.30 * np.sin(2 * np.pi * (f0 * t + (f1 - f0) / (2 * seconds) * t * t))
    rng = np.random.default_rng(0)
    sig += 0.01 * rng.standard_normal(n)
    sig = np.clip(sig, -1.0, 1.0)

    pcm16 = (sig * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm16.tobytes())
    return pcm16


def torch_fbank(waveform: np.ndarray, mel_filters: np.ndarray) -> np.ndarray:
    """Reference log-mel via torch.stft — identical to transformers' internal
    `_torch_extract_fbank_features`. Preferred path (exact library semantics)."""
    import torch

    wf = torch.from_numpy(waveform).to(torch.float32)
    window = torch.hann_window(N_FFT)  # periodic by default
    stft = torch.stft(wf, N_FFT, HOP, window=window, return_complex=True)  # center=True
    magnitudes = stft[..., :-1].abs() ** 2                 # drop last frame, power
    mf = torch.from_numpy(mel_filters).to(torch.float32)   # (N_FREQS, N_MELS)
    mel_spec = mf.T @ magnitudes                           # (N_MELS, N_FRAMES)
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.numpy().astype(np.float32)


def numpy_fbank(waveform: np.ndarray, mel_filters: np.ndarray) -> np.ndarray:
    """torch-free fallback replicating the SAME math (reflect pad, periodic Hann,
    rfft power, drop last frame, log compression). Used only if torch is absent."""
    pad = N_FFT // 2
    wf = np.pad(waveform.astype(np.float64), pad, mode="reflect")
    win = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(N_FFT) / N_FFT)  # periodic Hann
    n_frames_full = 1 + (len(wf) - N_FFT) // HOP
    frames = np.stack([wf[i * HOP:i * HOP + N_FFT] * win for i in range(n_frames_full)], axis=0)
    spec = np.fft.rfft(frames, n=N_FFT, axis=1)            # (n_frames_full, N_FREQS)
    power = (np.abs(spec) ** 2)[:-1]                       # drop last frame
    mel_spec = (power @ mel_filters).T                     # (N_MELS, N_FRAMES)
    log_spec = np.log10(np.clip(mel_spec, 1e-10, None))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.astype(np.float32)


def load_mel_filters(model_id: str) -> np.ndarray:
    """Pull the authentic filterbank + verify DSP geometry from transformers."""
    from transformers import WhisperFeatureExtractor

    fe = WhisperFeatureExtractor.from_pretrained(model_id)
    # Fail loud on any geometry drift — parity is meaningless otherwise.
    assert fe.feature_size == N_MELS, f"feature_size {fe.feature_size} != {N_MELS}"
    assert fe.n_fft == N_FFT, f"n_fft {fe.n_fft} != {N_FFT}"
    assert fe.hop_length == HOP, f"hop_length {fe.hop_length} != {HOP}"
    assert fe.sampling_rate == SR, f"sampling_rate {fe.sampling_rate} != {SR}"
    mel = np.asarray(fe.mel_filters, dtype=np.float32)     # (N_FREQS, N_MELS)
    assert mel.shape == (N_FREQS, N_MELS), f"mel_filters shape {mel.shape}"
    return mel


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    default_out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    ap.add_argument("--model-id", default="openai/whisper-large-v3-turbo",
                    help="HF id of the encoder whose feature extractor to mirror "
                         "(Ultravox v0.5 uses whisper-large-v3-turbo).")
    ap.add_argument("--out-dir", default=default_out)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--force-numpy", action="store_true",
                    help="Skip torch; use the numpy STFT fallback for the golden.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    wav_path = os.path.join(args.out_dir, "test_audio.wav")
    mel_path = os.path.join(args.out_dir, "mel_filters.bin")
    exp_path = os.path.join(args.out_dir, "expected_log_mel.bin")

    print(f"[1/4] loading feature extractor: {args.model_id}")
    mel_filters = load_mel_filters(args.model_id)
    mel_filters.tofile(mel_path)
    print(f"      mel_filters {mel_filters.shape} -> {mel_path}")

    print(f"[2/4] writing dummy wav ({args.seconds}s @ {SR} Hz)")
    pcm16 = generate_dummy_wav(wav_path, SR, args.seconds)
    # EXACT samples dr_wav's f32 path produces for 16-bit PCM: x / 32768.0.
    waveform = pcm16.astype(np.float32) / 32768.0
    print(f"      {len(waveform)} samples -> {wav_path}")

    use_numpy = args.force_numpy
    if not use_numpy:
        try:
            import torch  # noqa: F401
        except Exception:
            print("      torch unavailable -> numpy STFT fallback")
            use_numpy = True

    print(f"[3/4] computing golden log-mel ({'numpy' if use_numpy else 'torch'})")
    log_mel = numpy_fbank(waveform, mel_filters) if use_numpy else torch_fbank(waveform, mel_filters)
    assert log_mel.shape[0] == N_MELS, log_mel.shape

    print(f"[4/4] writing golden log-mel {log_mel.shape} -> {exp_path}")
    log_mel.astype(np.float32).tofile(exp_path)

    print("done.")
    print(f"  n_frames        = {log_mel.shape[1]}")
    print(f"  log_mel min/max = {log_mel.min():.6f} / {log_mel.max():.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
