# audio_sandbox — Whisper log-mel front-end (parity sandbox + real-time viz)

Standalone C++ sandbox for the **audio front-end (DSP + feature extraction)** of
`fixie-ai/ultravox-v0_5-llama-3_2-1b`, whose encoder is
`openai/whisper-large-v3-turbo`. It extracts **128-bin log-mel spectrograms** from
16 kHz mono audio. **No neural-network weights, no CUDA, no engine dependencies** —
pure DSP correctness, to be ported to CUDA later.

Two products share one **verified** DSP core (`whisper_dsp`):

| Target | What |
|---|---|
| `parity_check` | Offline: `.wav` → log-mel → cosine similarity vs the PyTorch golden dump. |
| `audio_realtime` | **Windows-only**: live mic/loopback capture → scrolling Direct2D spectrogram. |

> Separate from the in-engine `src/audio/` + `include/audio/` (Qwen2-Audio CUDA)
> subsystem. This is a self-contained `project()` you build on its own.

## DSP geometry

`sample_rate=16000`, `n_fft=400` (25 ms), `hop_length=160` (10 ms), `n_mels=128`,
`n_freqs = n_fft/2 + 1 = 201`.

## Parity contract (Python golden ⇔ C++ offline path agree exactly)

1. Input waveform = `int16 / 32768.0` — the exact samples `dr_wav`'s f32 read path
   yields, so both sides see identical inputs.
2. **Periodic** Hann window of length `n_fft` (`torch.hann_window` default).
3. `torch.stft(center=True)` ⇒ **reflect-pad** the signal by `n_fft/2` each side.
4. **Drop the last** STFT time frame (`stft[..., :-1]`).
5. Power spectrum `|stft|²`; mel = `mel_filters.T @ power`.
6. `log10(clamp(mel, 1e-10))` → floor at `global_max − 8` → `(x + 4) / 4`.

The **mel filterbank is loaded** from `mel_filters.bin` (dumped from
`transformers`), never recomputed in C++ — the Slaney-normalised triangular
filters are the single biggest source of float mismatch.

### Real-time vs. offline (what necessarily differs, and why it stays parity-safe)

The verified inner math (steps 2, 5, 6-`log10`) lives in **one** function,
`WhisperDSP::compute_log_mel_column()`, called by *both* paths. The streaming path
omits only the three steps that **cannot** exist in a live stream:

- **center reflect-pad / drop-last** (steps 3–4) — they need the clip boundaries;
  the stream just takes contiguous 400/160 windows off the live signal.
- **global `(max − 8)` compression** — it needs the whole-clip maximum. The viz
  applies the same 8-decade dynamic-range shape as a **rolling** normalisation
  over the visible history. This is a display-only step; it never touches the DSP,
  and `parity_check` still matches the golden at **cosine = 1.0**.

## Architecture (real-time)

```
miniaudio callback ─push─▶ SampleRing (mutex+deque) ─pop─▶ RealTimeDSP worker
   ─push─▶ SpectrogramBuffer (mutex, rolling N cols) ─snapshot─▶ Direct2D UI (~30 FPS)
```

miniaudio is configured `ma_format_f32` / `channels=1` / `sampleRate=16000`, so it
does the resampling + downmix internally — the ring already holds Whisper-ready
samples. Loopback uses WASAPI (`ma_device_type_loopback`).

## Usage

```bash
# 1) Generate golden data (needs: pip install transformers torch numpy)
#    torch is optional — a numpy STFT fallback produces the same math.
python audio_sandbox/scripts/export_whisper_dsp.py
#    -> audio_sandbox/data/{test_audio.wav, mel_filters.bin, expected_log_mel.bin}

# 2) Configure + build (pulls dr_wav + pocketfft + miniaudio via FetchContent)
cmake -S audio_sandbox -B audio_sandbox/out -DCMAKE_BUILD_TYPE=Release
cmake --build audio_sandbox/out --config Release

# 3a) Offline parity check (exit 0 iff cosine > 0.9999)
audio_sandbox/out/Release/parity_check.exe audio_sandbox/data

# 3b) Real-time visualiser (needs mel_filters.bin from step 1)
audio_sandbox/out/Release/audio_realtime.exe audio_sandbox/data
#     then press M (microphone) or L (system loopback) at the prompt
```

## Layout

| Path | What |
|---|---|
| `include/whisper_dsp.h` · `src/whisper_dsp.cpp` | `WhisperDSP` — the verified core. `process()` (offline) + `compute_log_mel_column()` (streaming) share `log_mel_frame_()`. |
| `src/parity_check.cpp` | dr_wav load → `process()` → cosine-similarity parity check. |
| `include/audio_capture.h` · `src/audio_capture.cpp` | miniaudio capture (mic/loopback) + thread-safe `SampleRing`. |
| `include/realtime_dsp.h` · `src/realtime_dsp.cpp` | DSP worker thread + rolling `SpectrogramBuffer`. |
| `include/window_d2d.h` · `src/window_d2d.cpp` | Win32 + Direct2D scrolling heatmap (inferno colormap, Y-inverted). |
| `src/main.cpp` | Real-time app entry: console mode prompt → wires capture → DSP → UI. |
| `scripts/export_whisper_dsp.py` | Golden dumper (authentic mel filterbank from `transformers`). |
| `data/` | Generated artifacts (git-ignored). |
