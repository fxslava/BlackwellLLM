#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_f5_tts_cuda.py — F5-TTS zero-shot synthesis probe: acoustic quality + CUDA latency.

Purpose: decide whether F5-TTS is worth a C++ ONNX/CUDA integration BEFORE writing any
of that pipeline. Two questions only — how does it sound, and how long does it take on
this GPU.

===============================================================================
INSTALL
===============================================================================
!! READ THIS FIRST — `pip install f5-tts` can break your GPU setup. !!

This box runs a PyTorch nightly built for CUDA 13.0, which is what gives the RTX 5070
(Blackwell, sm_120) working kernels. `f5-tts` depends on torch + torchaudio, and pip
resolving those from the DEFAULT index will happily install a stable build whose kernel
set stops at sm_90. Everything then imports fine and every CUDA call dies with
"no kernel image is available for execution on the device".

So: install into a DEDICATED venv, and put the right torch in it FIRST.

    python -m venv .venv-f5
    .venv-f5\\Scripts\\activate                    # Windows;  source .venv-f5/bin/activate on Linux

    # 1. torch + torchaudio WITH sm_120 kernels. Must come before f5-tts so pip sees
    #    them as already satisfied and does not substitute its own.
    pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130

    # 2. F5-TTS itself (pulls vocos, transformers, soundfile, ...).
    pip install f5-tts

    # optional: bleeding edge instead of the PyPI release
    # pip install git+https://github.com/SWivid/F5-TTS.git

If you insist on installing into an existing environment, use `--no-deps` and add the
remaining requirements by hand. Either way the arch check below will tell you
immediately if torch got swapped — that check exists precisely because this failure
mode is silent until the first kernel launch.

===============================================================================
WHAT YOU MUST PROVIDE
===============================================================================
1. reference_voice.wav — 3-10 s of ONE speaker, clean, no music/noise, ideally ending
   on a complete word. This is the voice that gets cloned; its quality caps the output
   quality entirely. Mono, any sample rate (F5-TTS resamples to 24 kHz internally).

2. The TRANSCRIPT of that reference audio (--ref-text). F5-TTS is not "audio in, voice
   out": the model conditions on an (audio, text) pair and treats generation as infilling
   the continuation. A wrong or missing transcript degrades output badly. Leaving it
   empty makes F5-TTS auto-transcribe with Whisper, which works but downloads ~1.6 GB
   and can mis-hear — set it explicitly for a fair quality evaluation.

===============================================================================
!! RUSSIAN — THE THING THAT WILL OTHERWISE WASTE YOUR AFTERNOON !!
===============================================================================
The official F5-TTS checkpoints (F5TTS_Base / F5TTS_v1_Base) are trained on Emilia:
ENGLISH AND CHINESE ONLY. Their character vocabulary has no Cyrillic. Feeding Russian
to the base model does not produce accented Russian — it produces confident nonsense,
because the text simply is not representable.

Judging "F5-TTS quality" from that run would be judging the wrong thing.

For Russian you need a community fine-tune AND its matching vocab file, passed together:

    python test_f5_tts_cuda.py --ckpt <path/to/model.safetensors> --vocab <path/to/vocab.txt>

The known community Russian fine-tune is the HF repo `Misha24-10/F5-TTS_RUSSIAN`.
I have NOT verified its current file names — open the repo's file listing and point
--ckpt / --vocab at the actual checkpoint and vocab it ships. The vocab is not optional:
a Russian checkpoint loaded against the base EN/ZH vocab gives garbage just as surely as
the base model does.

The script detects Cyrillic in the generation text and warns if no custom checkpoint was
given. It still runs — hearing the failure is itself informative — but it will not let
you mistake it for a quality result.

===============================================================================
USAGE
===============================================================================
    python test_f5_tts_cuda.py
    python test_f5_tts_cuda.py --ref-text "текст который звучит в reference_voice.wav"
    python test_f5_tts_cuda.py --nfe 16,32,64 --runs 3     # latency/quality sweep
"""

import argparse
import inspect
import os
import sys
import time

# -----------------------------------------------------------------------------
# Configuration — everything hardcoded lives here.
# -----------------------------------------------------------------------------

# Rich punctuation on purpose: exclamation, comma, period. Prosody is the thing under
# evaluation, and flat declarative text would not show whether the model does anything
# interesting with intonation.
GEN_TEXT = (
    "Привет! Это тестовая генерация голоса с использованием нейросети F5-TTS "
    "на видеокарте. Скажи, а получается ли естественная интонация?"
)

REF_AUDIO_NAME = "reference_voice.wav"
OUTPUT_WAV = "f5_output_test.wav"

# Default model id for the current f5-tts release. Overridden entirely when --ckpt is given.
DEFAULT_MODEL = "F5TTS_v1_Base"


def fail(msg: str, code: int = 1) -> None:
    """Print to stderr and exit. Used for every unrecoverable precondition."""
    print(f"\n[FATAL] {msg}\n", file=sys.stderr)
    sys.exit(code)


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


# -----------------------------------------------------------------------------
# Step 1 — CUDA, strictly. No silent CPU fallback: a CPU number would be useless for
# estimating the latency of a GPU pipeline, and F5-TTS on CPU is minutes per utterance.
# -----------------------------------------------------------------------------
def require_cuda() -> "torch.device":
    try:
        import torch
    except ImportError:
        fail("PyTorch is not installed. See the INSTALL block at the top of this file.")

    if not torch.cuda.is_available():
        fail(
            "CUDA is not available — this script is CUDA-only by design.\n"
            f"  torch {torch.__version__}, built against CUDA {torch.version.cuda}\n"
            "  Check the driver, and that this torch is a CUDA build (not the CPU wheel)."
        )

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    device_arch = f"sm_{major}{minor}"
    arch_list = torch.cuda.get_arch_list()

    print(f"  torch          : {torch.__version__}  (CUDA {torch.version.cuda})")
    print(f"  device         : {name}  ({device_arch})")
    print(f"  kernel archs   : {', '.join(arch_list)}")

    # THE check this script exists to make early. A torch without kernels for this GPU
    # imports cleanly, reports cuda.is_available() == True, and then fails at the first
    # launch with an error that reads like a model bug. Catch it here instead.
    if device_arch not in arch_list:
        fail(
            f"This torch has NO kernels for {device_arch} ({name}).\n"
            f"  It ships: {', '.join(arch_list)}\n"
            "  Almost certainly pip replaced your torch while installing f5-tts.\n"
            "  Reinstall the Blackwell-capable build (see the INSTALL block), ideally\n"
            "  into a clean venv, and install torch BEFORE f5-tts."
        )

    torch.cuda.reset_peak_memory_stats()
    return torch.device("cuda")


# -----------------------------------------------------------------------------
# Step 2 — locate the reference audio.
# -----------------------------------------------------------------------------
def find_reference_audio() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (os.path.join(os.getcwd(), REF_AUDIO_NAME), os.path.join(here, REF_AUDIO_NAME)):
        if os.path.isfile(candidate):
            return candidate

    fail(
        f"'{REF_AUDIO_NAME}' not found (looked in {os.getcwd()} and {here}).\n"
        "  Record 3-10 seconds of ONE speaker: clean, no background music or noise,\n"
        "  ending on a complete word. That clip IS the cloned voice — its quality is\n"
        "  the ceiling on everything the model produces."
    )
    return ""  # unreachable; keeps type checkers quiet


def report_reference(path: str) -> float:
    """Print reference-clip properties and return its duration in seconds."""
    try:
        import soundfile as sf
    except ImportError:
        print("  (soundfile missing — skipping reference audio inspection)")
        return 0.0

    info = sf.info(path)
    dur = info.frames / float(info.samplerate)
    print(f"  reference      : {path}")
    print(f"                   {dur:.2f} s, {info.samplerate} Hz, {info.channels} ch")

    # Not fatal — F5-TTS handles these — but both measurably hurt cloning quality, and a
    # bad reference is the single most common cause of "the model sounds terrible".
    if dur < 3.0:
        print("  [WARN] shorter than 3 s: too little material to characterise the voice.")
    elif dur > 12.0:
        print("  [WARN] longer than ~10 s: F5-TTS clips the prompt; the tail is ignored.")
    if info.channels != 1:
        print("  [WARN] not mono — it will be downmixed.")
    return dur


# -----------------------------------------------------------------------------
# Step 3 — model construction, tolerant of the package's API drift.
#
# f5-tts has renamed constructor and infer() arguments across releases (model_type ->
# model, seed defaults, etc.). Rather than pin one spelling and break on the next
# release, build the kwargs we want and keep only those the installed signature
# actually accepts.
# -----------------------------------------------------------------------------
def filter_kwargs(fn, desired: dict) -> dict:
    """Keep only the kwargs `fn` actually declares; report what got dropped."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return desired
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return desired
    kept = {k: v for k, v in desired.items() if k in params}
    dropped = sorted(set(desired) - set(kept))
    if dropped:
        print(f"  [note] installed f5-tts does not accept: {', '.join(dropped)} — ignored")
    return kept


def build_model(args, device):
    try:
        from f5_tts.api import F5TTS
    except ImportError as e:
        fail(
            f"Could not import f5_tts ({e}).\n"
            "  pip install f5-tts   — see the INSTALL block, and mind the torch hazard."
        )

    # Both spellings of the model-name argument; filter_kwargs keeps whichever exists.
    desired = {
        "model": DEFAULT_MODEL,
        "model_type": "F5-TTS",
        "ckpt_file": args.ckpt or "",
        "vocab_file": args.vocab or "",
        "device": str(device),
    }
    kwargs = filter_kwargs(F5TTS.__init__, desired)

    if args.ckpt:
        print(f"  checkpoint     : {args.ckpt}")
        print(f"  vocab          : {args.vocab or '(none — likely WRONG for a fine-tune)'}")
    else:
        print(f"  checkpoint     : {DEFAULT_MODEL} (downloads from HF on first run)")

    t0 = time.perf_counter()
    model = F5TTS(**kwargs)
    load_s = time.perf_counter() - t0
    return model, load_s


# -----------------------------------------------------------------------------
# Step 4 — one generation, timed.
# -----------------------------------------------------------------------------
def synthesize(model, ref_audio, ref_text, gen_text, nfe, seed):
    """Run one inference. Returns (wav_float32, sample_rate, wall_seconds)."""
    import torch

    desired = {
        "ref_file": ref_audio,
        "ref_text": ref_text,
        "gen_text": gen_text,
        "nfe_step": nfe,
        "cfg_strength": 2.0,
        "sway_sampling_coef": -1.0,
        "speed": 1.0,
        "remove_silence": False,
        "seed": seed,          # fixed seed => runs are comparable
        "show_info": lambda *a, **k: None,   # keep the timing output clean
    }
    kwargs = filter_kwargs(model.infer, desired)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = model.infer(**kwargs)
    torch.cuda.synchronize()   # CUDA is async; without this we would time the launch, not the work
    elapsed = time.perf_counter() - t0

    # infer() returns (wav, sample_rate, spectrogram) — unpack defensively in case a
    # future release appends to the tuple.
    wav, sr = result[0], result[1]
    return wav, sr, elapsed


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(
        description="F5-TTS zero-shot quality + CUDA latency probe.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ref-text", default="",
                   help="Transcript of reference_voice.wav. Empty => Whisper auto-transcribes "
                        "(~1.6 GB download). Set it for a fair quality evaluation.")
    p.add_argument("--text", default=GEN_TEXT, help="Text to synthesize.")
    p.add_argument("--ckpt", default="", help="Custom checkpoint (REQUIRED for Russian).")
    p.add_argument("--vocab", default="", help="Custom vocab.txt — must pair with --ckpt.")
    p.add_argument("--nfe", default="32",
                   help="Flow-matching steps; comma-separated to sweep, e.g. 16,32,64. "
                        "Latency is roughly linear in this; quality saturates.")
    p.add_argument("--runs", type=int, default=2,
                   help="Timed runs per NFE (after one untimed warmup).")
    p.add_argument("--seed", type=int, default=1234, help="Fixed seed so runs are comparable.")
    args = p.parse_args()

    try:
        nfe_values = [int(x) for x in args.nfe.split(",") if x.strip()]
    except ValueError:
        fail(f"--nfe must be integers, got '{args.nfe}'")
    if not nfe_values:
        fail("--nfe produced no values")

    banner("ENVIRONMENT")
    device = require_cuda()
    import torch  # safe now: require_cuda already proved it imports

    banner("INPUTS")
    ref_audio = find_reference_audio()
    ref_dur = report_reference(ref_audio)

    if args.ref_text:
        print(f"  ref transcript : {args.ref_text!r}")
    else:
        print("  ref transcript : (empty — Whisper will auto-transcribe)")
        print("  [WARN] F5-TTS conditions on an (audio, TEXT) pair; it infills a continuation.")
        print("         An ASR mistake here shows up as degraded output that looks like a")
        print("         model failure. Pass --ref-text for an honest quality read.")

    print(f"  text           : {args.text!r}")

    # The Russian trap. Check before spending a minute on model download + generation.
    has_cyrillic = any("Ѐ" <= ch <= "ӿ" for ch in args.text)
    if has_cyrillic and not args.ckpt:
        print("\n  " + "!" * 74)
        print("  !! CYRILLIC TEXT ON THE BASE (ENGLISH/CHINESE) CHECKPOINT")
        print("  !!")
        print("  !! F5TTS_Base / F5TTS_v1_Base are trained on Emilia (EN + ZH). Their")
        print("  !! character vocabulary contains no Cyrillic, so this will not produce")
        print("  !! accented Russian — it will produce nonsense. Do not read the result")
        print("  !! as a verdict on F5-TTS quality.")
        print("  !!")
        print("  !! For Russian, pass a fine-tune AND its vocab together:")
        print("  !!     --ckpt <model.safetensors> --vocab <vocab.txt>")
        print("  !! Known community fine-tune: HF repo Misha24-10/F5-TTS_RUSSIAN")
        print("  !! (verify its current file names yourself — they are not pinned here).")
        print("  " + "!" * 74)
    if args.ckpt and not args.vocab:
        print("\n  [WARN] --ckpt without --vocab. A fine-tune with an extended alphabet needs")
        print("         its own vocab.txt; loading it against the base vocab gives garbage.")

    banner("MODEL LOAD")
    model, load_s = build_model(args, device)
    print(f"  load time      : {load_s:.2f} s   (one-time; first run also downloads weights)")

    banner("GENERATION")
    # Warmup: the first call pays cuDNN/cuBLAS algorithm selection, lazy module init and
    # allocator growth. Including it would inflate the steady-state figure a real-time
    # pipeline actually cares about.
    print("  warmup run (untimed) ...")
    wav, sr, warm_s = synthesize(model, ref_audio, args.ref_text, args.text,
                                 nfe_values[0], args.seed)
    print(f"  warmup took    : {warm_s:.2f} s (discarded)")

    results = []   # (nfe, mean_seconds, audio_seconds)
    for nfe in nfe_values:
        times = []
        for i in range(max(1, args.runs)):
            wav, sr, elapsed = synthesize(model, ref_audio, args.ref_text, args.text,
                                          nfe, args.seed)
            times.append(elapsed)
            print(f"  nfe={nfe:<3} run {i + 1}/{args.runs}: {elapsed:.3f} s")
        mean_s = sum(times) / len(times)
        audio_s = len(wav) / float(sr)
        results.append((nfe, mean_s, audio_s))

    # Save the LAST generation (highest NFE swept = best quality) for listening.
    try:
        import soundfile as sf
        out_path = os.path.join(os.getcwd(), OUTPUT_WAV)
        sf.write(out_path, wav, sr)
        print(f"\n  wrote          : {out_path}  ({len(wav) / float(sr):.2f} s @ {sr} Hz)")
    except Exception as e:                                    # noqa: BLE001
        fail(f"generation succeeded but writing {OUTPUT_WAV} failed: {e}")

    banner("LATENCY")
    print(f"  {'NFE':>5}  {'gen (s)':>9}  {'audio (s)':>10}  {'RTF':>7}  {'xRT':>7}")
    for nfe, mean_s, audio_s in results:
        rtf = mean_s / audio_s if audio_s > 0 else float("nan")
        print(f"  {nfe:>5}  {mean_s:>9.3f}  {audio_s:>10.2f}  {rtf:>7.3f}  {1.0 / rtf:>6.2f}x")
    print("  RTF = generation time / audio duration. Below 1.0 is faster than real time.")

    # With two or more NFE points, split the cost into the part that scales with the
    # flow-matching steps (the DiT) and the fixed part (vocoder + pre/post-processing).
    # That is the model a latency budget is actually built from.
    if len(results) >= 2:
        xs = [r[0] for r in results]
        ys = [r[1] for r in results]
        n = len(xs)
        mean_x, mean_y = sum(xs) / n, sum(ys) / n
        denom = sum((x - mean_x) ** 2 for x in xs)
        if denom > 0:
            per_step = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
            fixed = mean_y - per_step * mean_x
            print(f"\n  cost model     : ~{fixed * 1000:.0f} ms fixed + "
                  f"~{per_step * 1000:.1f} ms per NFE step")
            print("                   (fixed ~= vocoder + pre/post; per-step ~= one DiT pass)")

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"\n  peak VRAM      : {peak:.2f} GiB (torch allocator; excludes context ~0.3-0.6 GiB)")

    banner("READ THIS BEFORE BUDGETING FOR A REAL-TIME PIPELINE")
    slowest = max(r[1] for r in results)
    print(f"  TTFB == total generation time == {slowest:.2f} s (worst NFE swept).")
    print()
    print("  F5-TTS IS NOT STREAMING, and that is architectural, not an implementation gap:")
    print("    * it is a flow-matching model — the DiT denoises the WHOLE mel-spectrogram")
    print("      jointly over N steps, so no prefix of the output exists until the last step;")
    print("    * total duration is decided UP FRONT, before any audio is produced.")
    print("  There is therefore no first chunk to emit early. Time-to-first-audio equals")
    print("  full synthesis time for the entire utterance, and the only lever is splitting")
    print("  the text into smaller pieces before calling it.")
    print()
    print("  Compare against the alternative: a VITS-family model emits a whole sentence in")
    print("  ONE forward pass, typically ~120-250 ms to first audio. Sentence-chunking gets")
    print("  F5-TTS to the same structure but each chunk still costs the figure above, and")
    print("  chunk boundaries in a flow-matching model are also where prosody continuity")
    print("  breaks, since each chunk is re-conditioned on the reference rather than on")
    print("  what was just spoken.")
    print()
    print("  Judge the trade honestly: F5-TTS buys zero-shot voice cloning from a few")
    print("  seconds of reference audio, which VITS cannot do at all. The questions are")
    print("  whether the latency above fits a conversational turn, and whether the VRAM")
    print("  fits alongside the 8B AWQ backbone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
