#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_f5_reference.py — build the F5-TTS reference clip, with a transcript that
is actually true.

    python scripts/make_f5_reference.py            # writes models/f5_tts/ref_female_24k.wav

===============================================================================
WHY THIS SCRIPT EXISTS AT ALL
===============================================================================
F5 conditions on an (audio, TEXT) PAIR and treats synthesis as infilling. The
text is not a label and not documentation -- it is an input the model reasons
about. If it does not match the clip, the model must reconcile the two, and it
does so by INVENTING content: the reference bleeds into every utterance and the
output sounds possessed rather than merely wrong.

So the hard part of "get a reference clip" is not the audio. It is being able to
state, with certainty, what the audio says. That rules out grabbing a nice-
sounding clip from anywhere and typing out what you think you hear -- a
transcript that is merely PLAUSIBLE produces exactly the bug this fixes.

Hence a dataset with GROUND TRUTH: google/fleurs (CC-BY-4.0) ships
`raw_transcription` with real case and punctuation, plus speaker gender, and its
rows are reachable one at a time through the HF datasets-server API, so only the
selected clip is ever downloaded.

===============================================================================
WHAT IS DONE TO THE SIGNAL, AND WHAT IS DELIBERATELY NOT
===============================================================================
  resample   -> exactly 24 000 Hz (kF5SampleRate), soxr_hq
  channels   -> mono
  format     -> 16-bit PCM WAV
  trim       -> LEADING/TRAILING SILENCE ONLY, at a threshold far below speech,
                keeping 40 ms of padding. Trimming silence cannot invalidate the
                transcript. Trimming to hit a duration target WOULD, so nothing
                here cuts by length -- the length filter is applied when
                SELECTING the clip, never by editing one.
  level      -> peak-normalise to 0.95. No compression, no EQ, no denoise: every
                one of those changes the timbre being cloned.

Selection filters (all measured, none by ear -- this script cannot listen and
neither could the agent that wrote it):
  female (gender == 1), 3.0-5.5 s, no clipping (peak < 0.999), audible
  (peak >= 0.05), pure Cyrillic + punctuation, ranked by measured SNR.

The pure-Cyrillic filter is not cosmetic: F5's vocab is character-level, and a
Latin token in the REFERENCE ("WiFi") puts a script in the conditioning that
later utterances never revisit. Every character of the chosen text is checked
against the checkpoint's vocab.txt before the file is written.

===============================================================================
IF YOU CHANGE THE CLIP, CHANGE THE TRANSCRIPT WITH IT
===============================================================================
This script prints the transcript it selected. That exact string must land in
BOTH places or the pair desynchronises and the bleeding comes back:
  * audio_sandbox/voice_assistant/settings_store.hpp  (tts_ref_text default)
  * %LOCALAPPDATA%/BlackwellVoiceAssistant/settings.json  (the running value,
    which OVERRIDES the default on any machine that has launched the app once)
"""
import argparse
import json
import os
import re
import sys
import urllib.request

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO_ROOT, "models", "f5_tts", "ref_female_24k.wav")
DEFAULT_VOCAB = os.path.join(REPO_ROOT, "models", "f5_tts", "vocab.txt")

TARGET_SR = 24000          # kF5SampleRate (src/tts/f5_tts_engine.hpp)
API = ("https://datasets-server.huggingface.co/rows"
       "?dataset=google%2Ffleurs&config=ru_ru&split=validation"
       "&offset={off}&length={n}")
FLEURS_SR = 16000
CYRILLIC_OK = re.compile(r"^[А-Яа-яЁё0-9\s\.,\!\?\:\;\-—«»\"'()]+$")


def frame_rms(x, sr, ms):
    fl = max(1, int(ms / 1000.0 * sr))
    fr = x[: len(x) // fl * fl].reshape(-1, fl)
    return np.sqrt((fr * fr).mean(axis=1) + 1e-12), fl


def measure(x, sr):
    """peak / rms / crude SNR. Noise floor = quietest 10% of 20 ms frames."""
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    rms = float(np.sqrt(np.mean(x * x)))
    fe, _ = frame_rms(x, sr, 20.0)
    noise = float(np.sort(fe)[: max(1, len(fe) // 10)].mean())
    snr = 20.0 * np.log10((rms + 1e-12) / (noise + 1e-12))
    return peak, rms, snr


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--vocab", default=DEFAULT_VOCAB,
                   help="checkpoint vocab.txt; every character of the chosen "
                        "transcript must appear in it.")
    p.add_argument("--rows", type=int, default=300, help="dataset rows to scan.")
    p.add_argument("--min-sec", type=float, default=3.0)
    p.add_argument("--max-sec", type=float, default=5.5)
    args = p.parse_args()

    try:
        import soundfile as sf
        import librosa
    except ImportError as e:
        sys.exit(f"needs soundfile + librosa: {e}")

    rows = []
    for off in range(0, args.rows, 100):
        with urllib.request.urlopen(API.format(off=off, n=100), timeout=120) as r:
            rows += json.load(r)["rows"]
    print(f"scanned {len(rows)} dataset rows")

    cands = []
    for rr in rows:
        row = rr["row"]
        if row.get("gender") != 1:                       # 1 == female
            continue
        dur = int(row.get("num_samples") or 0) / float(FLEURS_SR)
        if not (args.min_sec <= dur <= args.max_sec):
            continue
        text = (row.get("raw_transcription") or "").strip()
        if not text or not CYRILLIC_OK.match(text):
            continue
        aud = row.get("audio") or []
        if aud and "src" in aud[0]:
            cands.append((text, aud[0]["src"]))
    print(f"{len(cands)} candidates (female, {args.min_sec}-{args.max_sec}s, Cyrillic)")

    scored = []
    probe = os.path.join(os.path.dirname(os.path.abspath(args.out)), "_probe.tmp.wav")
    for text, url in cands[:14]:
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                open(probe, "wb").write(r.read())
            x, sr = sf.read(probe, dtype="float32", always_2d=False)
        except Exception as e:
            print("  skip:", str(e)[:70])
            continue
        if x.ndim > 1:
            x = x.mean(axis=1)
        peak, rms, snr = measure(x, sr)
        if peak >= 0.999 or peak < 0.05:                 # clipped / inaudible
            continue
        scored.append(dict(x=x, sr=sr, text=text, snr=snr, peak=peak))
    if os.path.exists(probe):
        os.remove(probe)
    if not scored:
        sys.exit("no usable candidate")

    scored.sort(key=lambda c: -c["snr"])
    best = scored[0]
    x, sr, text = best["x"], best["sr"], best["text"]
    print(f"selected: {len(x)/sr:.2f}s, SNR {best['snr']:.1f} dB, peak {best['peak']:.3f}")

    # ---- vocab coverage: a miss here corrupts the CONDITIONING, so it is fatal.
    if os.path.exists(args.vocab):
        vs = {l.rstrip("\n") for l in open(args.vocab, encoding="utf-8")}
        missing = sorted({c for c in text if c not in vs})
        if missing:
            sys.exit(f"transcript uses characters absent from {args.vocab}: {missing}")
        print(f"vocab: all {len(set(text))} unique characters present")
    else:
        print(f"vocab: {args.vocab} not found -- coverage NOT checked")

    # ---- trim true silence only (see the header) ---------------------------
    fe, fl = frame_rms(x, sr, 10.0)
    noise = float(np.sort(fe)[: max(1, len(fe) // 10)].mean())
    thr = max(noise * 4.0, float(np.percentile(fe, 90)) * 0.02)
    above = np.where(fe > thr)[0]
    if len(above) == 0:
        sys.exit("no speech detected -- refusing to write")
    pad = int(0.040 * sr)
    x = x[max(0, above[0] * fl - pad): min(len(x), (above[-1] + 1) * fl + pad)]

    y = librosa.resample(x, orig_sr=sr, target_sr=TARGET_SR, res_type="soxr_hq")
    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = np.clip(y * (0.95 / peak), -1.0, 1.0)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sf.write(args.out, y, TARGET_SR, subtype="PCM_16")
    info = sf.info(args.out)
    print(f"\nwrote {args.out}")
    print(f"  {info.samplerate} Hz, {info.channels} ch, {info.subtype}, "
          f"{info.duration:.3f}s")
    print("\nref_text -- copy VERBATIM into settings_store.hpp and settings.json:")
    print(f"  {text}")


if __name__ == "__main__":
    main()
