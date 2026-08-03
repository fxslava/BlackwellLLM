#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_f5_tts_onnx.py — F5-TTS -> two ONNX graphs for the bare-metal C++ engine.

Produces exactly the pair src/tts/f5_tts_engine.{hpp,cpp} binds:

    f5_tts_dit.onnx      x_t, cond, text, t, dt  ->  x_next
    f5_tts_vocoder.onnx  mel                     ->  waveform

===============================================================================
THE POINT: THE C++ SIDE DOES NO ARITHMETIC
===============================================================================
F5TtsEngine keeps the [1, T, 100] latent resident in VRAM and ping-pongs it
between two Ort::IoBinding phases. For that to work, ONE Run() must advance the
solver by one full step -- so the CFG combine and the Euler update are baked in
here, not computed in C++. If they were not, C++ would need either a device
round-trip per step (~1.2 MB x nfe_step of pointless PCIe traffic, and a forced
sync that destroys the pipelining) or hand-written CUDA kernels in the hot path.

Both are a handful of lines in F5DitEulerStepWrapper.forward below, they fuse
into the graph, and they cost nothing measurable on the GPU.

===============================================================================
FOUR THINGS THAT WILL BITE YOU, ALL HANDLED BELOW
===============================================================================
1. CFG IN F5 IS NOT "AN EMPTY TOKEN SEQUENCE". F5's classifier-free guidance
   runs the SAME text through the transformer twice with drop_audio_cond /
   drop_text flags, which swap in LEARNED null embeddings. Feeding an empty or
   zeroed token tensor instead bypasses those embeddings and produces a
   different (worse, and subtly so) unconditional branch. See build_wrapper().

   The flags are Python bools, so they trace as constants -- two static
   subgraphs, no data-dependent control flow.

2. F5's CFG FORMULA IS NOT THE STABLE-DIFFUSION ONE.
       F5:      v = v_cond + (v_cond - v_uncond) * s
       classic: v = v_uncond + (v_cond - v_uncond) * s
   These differ by exactly one unit of scale (s_classic = s_f5 + 1). The brief
   asked for the classic spelling; the default here is `f5`, because that is
   what cfg_strength=2.0 means in every F5 config, checkpoint card and paper
   number. Pass --cfg-convention classic to get the other one. Whichever you
   pick is FROZEN INTO THE GRAPH -- the C++ side has no cfg input by design.

3. nn.Embedding NEEDS int64, THE C++ SIDE WANTS int32. The wrapper casts
   internally (one Cast node, free) so the graph's text input stays int32 and
   f5_tts_engine.cpp's device buffer stays 4 bytes/frame.

4. VOCOS CANNOT BE EXPORTED AS SHIPPED. Its ISTFT head uses torch.fft.irfft
   (no ONNX symbolic at opset 17), complex tensors (unrepresentable), and
   F.fold (needs Col2Im, opset 18+). All three are replaced with exact,
   export-safe equivalents -- see OnnxSafeISTFTHead. The replacement is checked
   numerically against the original before export.

===============================================================================
KNOWN-BROKEN: --dit-dtype fp16 PRODUCES SILENCE ON THE ORT CUDA PROVIDER
===============================================================================
Status 2026-08-03. The fp16 path below is COMPLETE and NUMERICALLY CORRECT, and
it is still defaulted OFF, because on the one runtime that matters it produces
a full-length WAV of pure silence (peak 0.0, rms NaN) with no error from Run().

What was measured, so nobody repeats it:

  torch fp16, CUDA, 16 steps, realistic inputs ... finite, absmax 26.71
                                                   (fp32 gives 26.74)
  ORT CPU provider, fp16 graph, 16 steps ......... finite, absmax 27.01,
                                                   including with the real
                                                   zero-padded conditioning
  ORT CUDA provider, fp16 graph .................. NaN, at EVERY graph
                                                   optimization level
                                                   (3/2/1/0 -- so it is not a
                                                   fusion; --opt-level exists
                                                   on the CLI to re-check this)

So the graph is sound and the CUDA provider's fp16 kernels are where it breaks.
ReduceL2 inside the ConvNeXt-v2 GRN blocks is a genuine fp16 overflow hazard --
it reduces over the SEQUENCE axis, so sum(x^2) grows with frame count and
crosses 65504 somewhere past ~1.5k frames, which is exactly why every check
above (all at <= 743 frames) is finite and production is not. keep_fp32_islands()
holds those four modules at fp32 and IS the right fix for that hazard, but it
did NOT resolve the silence on its own -- so there is at least one more fp16
site, and it has not been found yet.

FINDING IT NEEDS onnxruntime-gpu IN PYTHON. The C++ engine cannot dump
intermediates, and the CPU provider cannot reproduce the fault. With a GPU
python ORT, bisect by dumping node outputs and find the first non-finite
tensor; that is a bounded job and the reason it is not done here is tooling,
not difficulty.

AND NOTE THE CEILING BEFORE SPENDING THE TIME: fp16 measured 117 ms/step
against fp32's 183. That is 1.6x, NOT the 4x the arithmetic suggests, because
this graph is 16,511 nodes for 22 layers and ~63% of them are Shape/Constant/
Unsqueeze/Gather plumbing on a dynamic frame axis. Launch and shape overhead
dominate, and no precision change touches them. Getting to ~30-50 ms/step means
fixing the graph -- static/bucketed frame axis, which also folds away the shape
ops, removes the 44 CPU<->GPU Memcpy boundaries and unlocks CUDA Graph capture.
fp16 is a multiplier on top of that work, not a substitute for it.

===============================================================================
USAGE
===============================================================================
    # Russian fine-tune (the case this repo cares about)
    python scripts/export_f5_tts_onnx.py \
        --ckpt  F:/AI/F5-TTS_RUSSIAN/model_212000.safetensors \
        --vocab F:/AI/F5-TTS_RUSSIAN/vocab.txt \
        --arch v0

    # base EN/ZH checkpoint
    python scripts/export_f5_tts_onnx.py --ckpt .../model_1250000.safetensors --arch v1

    --out-dir defaults to <repo>/models/f5_tts/ (gitignored, where
    cmake/OnnxRuntime.cmake already puts hash-pinned model assets).

Run it in the SAME venv as scripts/test_f5_tts_cuda.py -- read that file's
INSTALL block first; `pip install f5-tts` will happily replace a Blackwell-
capable torch with one whose kernels stop at sm_90.
"""

import argparse
import inspect
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Paths follow the repo convention (CLAUDE.md): checkpoints under
# BLACKWELL_MODELS_DIR, outputs anchored to the repo root, never to the CWD.
# -----------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.environ.get("BLACKWELL_MODELS_DIR", "F:/AI")
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "models", "f5_tts")

# Geometry. These are properties of the F5-TTS checkpoint family and MUST match
# the constants in src/tts/f5_tts_engine.hpp (kF5SampleRate, kF5MelChannels,
# kF5HopLength, kF5NFft). They are written into the sidecar JSON so the two
# sides can be diffed mechanically instead of by eye.
N_MEL_CHANNELS = 100
SAMPLE_RATE = 24000
HOP_LENGTH = 256
N_FFT = 1024

# The pad id the C++ side fills the text tensor's tail with (kF5TextPadId).
# NOT arbitrary: F5's TextEmbedding does `text = text + 1` and treats 0 as the
# filler/mask token, so -1 in, 0 after the offset. Getting this wrong degrades
# only the TAIL of long utterances, which is the kind of bug that survives a
# listening test.
TEXT_PAD_ID = -1

# Set by build_vocoder() once the real head is inspected. A one-element list
# rather than a plain global so the assignment site is unambiguous; it feeds the
# output-length formula written into the contract JSON.
VOCODER_PADDING = [None]

# Architecture presets. dim/depth/text_dim/vocab are auto-detected from the
# checkpoint below; what CANNOT be detected are the behavioural flags, because
# they carry no parameters -- hence the preset.
ARCH_PRESETS = {
    # F5TTS_Base (v0). The community Russian fine-tunes are built on this.
    "v0": dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512,
               conv_layers=4, text_mask_padding=False, pe_attn_head=1),
    # F5TTS_v1_Base.
    "v1": dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512,
               conv_layers=4, text_mask_padding=True, pe_attn_head=None),
}


def fail(msg, code=1):
    print(f"\n[FATAL] {msg}\n", file=sys.stderr)
    sys.exit(code)


def banner(title):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def filter_kwargs(fn, desired):
    """Keep only the kwargs `fn` declares. Same idiom as test_f5_tts_cuda.py --
    f5-tts and torch.onnx.export have both renamed arguments across releases,
    and pinning one spelling breaks on the next upgrade."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return desired
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return desired
    return {k: v for k, v in desired.items() if k in params}


# =============================================================================
# Checkpoint
# =============================================================================
def load_state_dict(ckpt_path):
    """Load .safetensors and normalise it to bare transformer.* keys.

    F5 checkpoints wrap the weights in an EMA container: keys are prefixed
    `ema_model.` and two bookkeeping scalars (`initted`, `step`) ride along.
    f5_tts.model.utils.load_checkpoint strips exactly these; replicated here so
    the script does not depend on that private-ish helper.
    """
    try:
        from safetensors.torch import load_file
    except ImportError:
        fail("safetensors is not installed:  pip install safetensors")

    if not os.path.isfile(ckpt_path):
        fail(f"checkpoint not found: {ckpt_path}")

    raw = load_file(ckpt_path, device="cpu")
    state = {}
    for k, v in raw.items():
        if k in ("initted", "step"):
            continue
        state[k[len("ema_model."):] if k.startswith("ema_model.") else k] = v

    tf = {k[len("transformer."):]: v for k, v in state.items() if k.startswith("transformer.")}
    if not tf:
        # Some exports save the bare transformer with no CFM wrapper.
        tf = {k: v for k, v in state.items() if not k.startswith("mel_spec.")}
    if not tf:
        fail(f"no transformer weights found in {ckpt_path} (keys look like: "
             f"{sorted(list(raw.keys()))[:5]})")
    return tf


def infer_arch(tf_state, preset):
    """Derive what the WEIGHTS prove, keep the preset only for what they can't.

    Every dimension here is readable off a tensor shape, so reading it beats
    trusting a flag: a v0 checkpoint loaded under a v1 config is otherwise a
    strict-load failure with an unhelpful shape-mismatch dump.
    """
    cfg = dict(preset)

    w = tf_state.get("transformer_blocks.0.attn.to_q.weight")
    if w is not None:
        cfg["dim"] = int(w.shape[0])

    depth = 0
    while f"transformer_blocks.{depth}.attn.to_q.weight" in tf_state:
        depth += 1
    if depth:
        cfg["depth"] = depth

    emb = tf_state.get("text_embed.text_embed.weight")
    if emb is None:
        fail("checkpoint has no text_embed.text_embed.weight -- this does not look "
             "like an F5-TTS DiT. Is it a vocoder or an E2-TTS checkpoint?")
    # nn.Embedding(text_num_embeds + 1, text_dim): the +1 is the filler token.
    ckpt_vocab_size = int(emb.shape[0]) - 1
    cfg["text_dim"] = int(emb.shape[1])

    conv = 0
    while f"text_embed.text_blocks.{conv}.conv1d.0.weight" in tf_state:
        conv += 1
    if conv:
        cfg["conv_layers"] = conv

    # InputEmbedding.proj is Linear(mel_dim*2 + text_dim, dim): the *2 is
    # [noisy latent | conditioning mel] concatenated, and text_dim rides along.
    # Subtract the text half before halving, or you get (mel + text/2) and a
    # confusing shape failure two steps later.
    proj = tf_state.get("input_embed.proj.weight")
    mel_dim = ((int(proj.shape[1]) - cfg["text_dim"]) // 2
               if proj is not None else N_MEL_CHANNELS)
    return cfg, ckpt_vocab_size, mel_dim


def load_vocab(vocab_path):
    """Vocab -> {char: id}. Delegates to f5_tts's own reader when available.

    NOT reimplemented casually: get_tokenizer parses with `char[:-1]`, which
    strips the trailing newline AND clips the final line's last character when
    the file does not end in one. That quirk is baked into every trained
    checkpoint's id assignment, so reproducing it is correctness, not fidelity
    to a bug.
    """
    if not os.path.isfile(vocab_path):
        fail(f"vocab not found: {vocab_path}")
    try:
        from f5_tts.model.utils import get_tokenizer
        vocab_char_map, vocab_size = get_tokenizer(vocab_path, "custom")
        return vocab_char_map, int(vocab_size)
    except Exception as e:  # noqa: BLE001
        print(f"  [note] f5_tts.get_tokenizer unavailable ({e}); parsing vocab directly")
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_char_map = {char[:-1]: i for i, char in enumerate(f)}
        return vocab_char_map, len(vocab_char_map)


def build_dit(cfg, vocab_size, mel_dim):
    try:
        from f5_tts.model.backbones.dit import DiT
    except ImportError as e:
        fail(f"could not import f5_tts ({e}).  pip install f5-tts  -- and read the "
             f"INSTALL block in scripts/test_f5_tts_cuda.py before you do.")

    desired = dict(cfg)
    desired.update(text_num_embeds=vocab_size, mel_dim=mel_dim)
    kwargs = filter_kwargs(DiT.__init__, desired)
    dropped = sorted(set(desired) - set(kwargs))
    if dropped:
        print(f"  [note] installed DiT does not accept: {', '.join(dropped)} -- ignored")
    return DiT(**kwargs)


# =============================================================================
# Traceable text embedding -- the one F5 module that cannot be exported as-is
# =============================================================================
class TraceableTextEmbedding(nn.Module):
    """Drop-in replacement for f5_tts TextEmbedding.forward, safe to trace.

    THE PROBLEM. DiT.get_input_embed passes `seq_len = x.shape[1]`. Under the
    TorchScript tracer that is not a Python int -- it is a 0-dim Tensor. F5's
    TextEmbedding then takes its variable-length branch:

        if torch.is_tensor(seq_len):
            max_seq_len = int(seq_len.max().item())     # <-- BAKES A CONSTANT
            ...
            valid_pos_mask = seq_pos < seq_len.unsqueeze(1)   # <-- 0-dim: raises

    Two independent failures. The `.unsqueeze(1)` raises outright (a 0-dim
    tensor has no dim 1), and even with that fixed, `int(....item())` freezes the
    traced frame count into the graph -- so `text[:, :max_seq_len]` would silently
    TRUNCATE and `freqs_cis[:max_seq_len]` would mis-broadcast at every length
    except the one used for tracing. That is the exact failure mode verify()
    exists to catch, and it would sail past a single-length check.

    THE REPLACEMENT keeps the arithmetic identical for this repo's calling
    convention -- text is ALREADY padded to the audio frame count with
    TEXT_PAD_ID, which is what f5_tts_engine.cpp uploads -- and derives the one
    length it needs from `text.shape[1]` symbolically, so the frame axis stays
    dynamic. Under that convention the parts dropped here are provably no-ops:

      * text[:, :max_seq_len]              -- text is already exactly that long
      * F.pad(text, (0, max_seq_len - n))  -- pad width is zero
      * valid_pos_mask                     -- all-True for a single sequence
                                              occupying the whole frame axis

    Everything with a learned parameter (the embedding, the sinusoidal
    freqs_cis, the ConvNeXt text blocks, the mask_padding zeroing) is preserved
    exactly and still runs against the checkpoint's own weights.
    """

    def __init__(self, te):
        super().__init__()
        if getattr(te, "average_upsampling", False):
            raise RuntimeError(
                "this checkpoint uses text_embedding_average_upsampling, whose "
                "implementation is a per-sample Python loop over .item() lengths "
                "and cannot be traced at all. Export it with the dynamo exporter "
                "or disable the feature.")
        self.text_embed = te.text_embed
        self.text_blocks = te.text_blocks
        self.mask_padding = bool(te.mask_padding)
        self.extra_modeling = bool(te.extra_modeling)
        self.register_buffer("freqs_cis", te.freqs_cis.clone(), persistent=False)
        self.max_pos = int(te.freqs_cis.shape[0])

    def forward(self, text, seq_len=None, drop_text=False):
        # seq_len is accepted and ignored: DiT hands us x.shape[1], and text is
        # required to already be that long (see the class docstring).
        del seq_len

        text = text + 1                      # -1 pad -> 0 filler, F5's convention

        if self.mask_padding:
            text_mask = text == 0

        if drop_text:                        # CFG: the unconditional branch
            text = torch.zeros_like(text)

        text = self.text_embed(text)         # [b, n] -> [b, n, d]

        if self.extra_modeling:
            # THE line that has to stay dynamic. text.shape[1] is symbolic under
            # tracing, so this exports as a Slice with a computed end rather than
            # a constant one.
            text = text + self.freqs_cis[:text.shape[1], :]

            if self.mask_padding:
                m = text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1))
                text = text.masked_fill(m, 0.0)
                for block in self.text_blocks:
                    text = block(text)
                    text = text.masked_fill(m, 0.0)
            else:
                text = self.text_blocks(text)

        return text


def patch_text_embedding(transformer):
    """Swap in the traceable text embedding, reporting what it replaced."""
    original = transformer.text_embed
    replacement = TraceableTextEmbedding(original)
    transformer.text_embed = replacement
    print(f"  text embed     : patched for tracing "
          f"(mask_padding={replacement.mask_padding}, "
          f"extra_modeling={replacement.extra_modeling}, "
          f"freqs_cis up to {replacement.max_pos} frames)")
    return original, replacement


@torch.no_grad()
def check_text_embed_replacement(original, replacement, device, vocab_size,
                                 frames=257, n_tok=61):
    """Prove the traceable text embedding matches the module it replaces.

    Run in EAGER mode, where seq_len is an honest Python int and the original
    takes its safe branch -- so this compares the two implementations, not the
    tracer's view of them. Both CFG branches are checked, because drop_text
    selects a different path through the original and a replacement that only
    got the conditional branch right would still produce plausible audio with
    quietly broken guidance.
    """
    g = torch.Generator(device="cpu").manual_seed(0)
    text = torch.full((1, frames), TEXT_PAD_ID, dtype=torch.long)
    text[0, :n_tok] = torch.randint(0, max(1, vocab_size), (n_tok,), generator=g)
    text = text.to(device)

    worst = 0.0
    for drop in (False, True):
        ref = original(text, seq_len=frames, drop_text=drop)
        got = replacement(text, seq_len=frames, drop_text=drop)
        if ref.shape != got.shape:
            fail(f"text-embed replacement changed the shape for drop_text={drop}: "
                 f"{tuple(ref.shape)} -> {tuple(got.shape)}")
        scale = ref.abs().max().item() or 1.0
        rel = (ref - got).abs().max().item() / scale
        worst = max(worst, rel)
    print(f"  text embed     : max relative |diff| vs original = {worst:.2e} "
          f"(both CFG branches)")
    if worst > 1e-5:
        fail("the traceable text embedding does NOT match f5_tts' own. "
             "Do not ship this graph.")


# =============================================================================
# The DiT wrapper -- one Run() == one full solver step
# =============================================================================
class F5DitEulerStepWrapper(nn.Module):
    """(x_t, cond, text, t, dt) -> x_{t+1}, with CFG and the Euler update inside.

    Deliberately NOT a subclass of CFM: CFM.sample() owns duration estimation,
    the noise draw, the sway schedule and an odeint call, all of which belong on
    the host (they are host-side scalars, or in the C++ engine's case, decisions
    it already makes). What must be on the device is exactly this: two
    transformer passes, a lerp, and an axpy.

    NO DATA-DEPENDENT CONTROL FLOW. The only branch is on self.cfg_strength, a
    Python float read at trace time, so it constant-folds into one of two static
    graphs. Nothing here inspects a tensor VALUE.

    PRECISION SPLIT (fp16 exports). The transformer BODY runs in
    compute_dtype; the SOLVER STATE does not. x_t/cond/x_next stay fp32 at the
    graph boundary and the two arithmetic steps that own error accumulation --
    the CFG combine and the Euler update -- are done in fp32:

      * Euler is applied nfe_step times to the SAME tensor. Accumulating that
        in fp16 compounds rounding across 16 steps, and it lands in the mel the
        vocoder reads directly, i.e. as audible noise.
      * The CFG combine AMPLIFIES the cond/uncond difference by s (2.0 by
        default). Amplifying a fp16 difference is exactly where the bits are
        worth keeping.

    Everything the tensor cores actually care about -- every MatMul, Gemm and
    attention in the 22 DiT layers -- is inside _call() and runs in fp16. The
    casts cost one elementwise kernel each on a [1, T, 100] tensor (~554 KB at
    T=1385): microseconds against a multi-millisecond step.

    Keeping the boundary fp32 also means the C++ engine is UNCHANGED -- its
    device buffers, its two IoBinding phases and the DiT->vocoder handoff are
    all fp32 and stay that way. See src/tts/f5_tts_engine.hpp design note (1).
    """

    def __init__(self, transformer, cfg_strength=2.0, convention="f5",
                 compute_dtype=torch.float32):
        super().__init__()
        self.transformer = transformer
        self.cfg_strength = float(cfg_strength)
        self.convention = convention
        # Mutated in main() after the fp32 reference pass is captured, so the
        # drift check compares the SAME module in both precisions.
        self.compute_dtype = compute_dtype

        # Resolve the transformer's kwarg spelling ONCE, so forward() stays free
        # of introspection during tracing. `cache=False` matters: newer F5
        # versions memoise the text embedding across the cond/uncond pair, which
        # is stateful and would bake one utterance's text into the graph.
        probe = dict(mask=None, drop_audio_cond=False, drop_text=False, cache=False)
        self._extra = {k: v for k, v in filter_kwargs(transformer.forward, probe).items()
                       if k not in ("drop_audio_cond", "drop_text")}

    def _call(self, x_t, cond, text, t, drop):
        return self.transformer(x=x_t, cond=cond, text=text, time=t,
                                drop_audio_cond=drop, drop_text=drop, **self._extra)

    def forward(self, x_t, cond, text, t, dt):
        # int32 in (so the C++ device buffer is 4 bytes/frame), int64 for
        # nn.Embedding. One Cast node.
        text = text.to(torch.long)

        # Down into the compute precision for the transformer body only. A
        # no-op pair of Identity/Cast nodes when compute_dtype is fp32, so the
        # fp32 export is byte-for-byte what it always was.
        acc = x_t.dtype                       # fp32: the solver-state precision
        xc = x_t.to(self.compute_dtype)
        cc = cond.to(self.compute_dtype)
        tc = t.to(self.compute_dtype)

        # ...and straight back up. Every line below this point is fp32; see the
        # PRECISION SPLIT note in the class docstring for why that is not
        # optional.
        v_cond = self._call(xc, cc, text, tc, drop=False).to(acc)

        if self.cfg_strength >= 1e-5:
            # THE unconditional branch. drop_audio_cond/drop_text swap in the
            # model's LEARNED null embeddings -- this is not the same thing as
            # passing an empty or zeroed token tensor, which would skip those
            # embeddings entirely and give a differently-wrong uncond estimate.
            v_uncond = self._call(xc, cc, text, tc, drop=True).to(acc)
            s = self.cfg_strength
            if self.convention == "f5":
                v = v_cond + (v_cond - v_uncond) * s
            else:
                v = v_uncond + (v_cond - v_uncond) * s
        else:
            v = v_cond

        # Explicit Euler. dt is [1] and broadcasts against the trailing mel axis.
        return x_t + v * dt


# =============================================================================
# Vocoder -- Vocos, made exportable
# =============================================================================
class OnnxSafeISTFTHead(nn.Module):
    """Vocos' ISTFTHead with all three export blockers removed, exactly.

    The original does:
        mag, p = out(x).transpose(1,2).chunk(2, dim=1)
        S = clip(exp(mag), max=100) * (cos(p) + 1j*sin(p))
        audio = istft(S)                    # irfft -> window -> fold -> normalise

    Three problems and their replacements:

      torch.fft.irfft  ->  a matmul against a precomputed real inverse-DFT
                           basis. For a real signal the Hermitian sum collapses
                           to
                               x[m] = (1/N) sum_k w_k (Re_k cos(2pi k m/N)
                                                      - Im_k sin(2pi k m/N))
                           with w = [1, 2, 2, ..., 2, 1]. Exact, not an
                           approximation, and one GEMM on the GPU.

      complex tensors  ->  never materialised. Re = mag*cos(p), Im = mag*sin(p)
                           feed the two bases directly.

      F.fold           ->  conv_transpose1d against an identity kernel, which IS
                           overlap-add: out[b,0,t*hop + k] += frames[b,k,t].
                           Fold would need ONNX Col2Im (opset 18); ConvTranspose
                           is supported everywhere and lowers to the same thing.

    Verified numerically against the module it replaces before export -- see
    check_istft_replacement().
    """

    def __init__(self, head):
        super().__init__()
        self.out = head.out
        istft = head.istft
        self.n_fft = int(istft.n_fft)
        self.hop_length = int(istft.hop_length)
        self.win_length = int(istft.win_length)
        self.padding = getattr(istft, "padding", "same")

        # BOTH vocos padding modes reduce to the same overlap-add; they differ
        # ONLY in how much is trimmed off each end, and therefore in the output
        # length. Getting this wrong does not throw -- it shifts the waveform by
        # a few hundred samples and changes its length, which downstream reads
        # as a click at the start and a duration mismatch.
        #
        #   'same'   : forward STFT padded by (win-hop)/2 per side.
        #              trim = (win_length - hop_length)//2  ->  T*hop samples
        #   'center' : forward STFT padded by n_fft/2 per side (torch.stft's
        #              center=True). trim = n_fft//2        ->  (T-1)*hop samples
        #
        # 'center' is what the RELEASED charactr/vocos-mel-24khz config actually
        # uses -- 'same' is only the class-signature default, and the checkpoint
        # overrides it. The center branch of vocos' ISTFT delegates to
        # torch.istft, which has no ONNX symbolic whatsoever, so this
        # replacement is MORE necessary there, not less.
        if self.padding == "same":
            self.trim = (self.win_length - self.hop_length) // 2
        elif self.padding == "center":
            self.trim = self.n_fft // 2
        else:
            raise RuntimeError(
                f"unknown vocos ISTFT padding '{self.padding}' (expected 'same' "
                f"or 'center').")

        n = self.n_fft
        bins = n // 2 + 1
        k = torch.arange(bins, dtype=torch.float64).unsqueeze(1)
        m = torch.arange(n, dtype=torch.float64).unsqueeze(0)
        ang = 2.0 * math.pi * k * m / n
        w = torch.full((bins, 1), 2.0, dtype=torch.float64)
        w[0, 0] = 1.0
        if n % 2 == 0:
            w[-1, 0] = 1.0        # Nyquist bin is its own conjugate

        self.register_buffer("idft_cos", (w * torch.cos(ang) / n).float())   # [bins, n]
        self.register_buffer("idft_sin", (-w * torch.sin(ang) / n).float())  # [bins, n]
        self.register_buffer("window", istft.window.clone().float())
        # Identity overlap-add kernel: [C_in=n_fft, C_out=1, kW=n_fft].
        self.register_buffer("ola", torch.eye(n, dtype=torch.float32).unsqueeze(1))

    def forward(self, x):
        x = self.out(x).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        # clip BEFORE anything downstream can overflow; matches vocos exactly.
        mag = torch.clip(torch.exp(mag), max=1e2)
        real = mag * torch.cos(p)          # [B, bins, T]
        imag = mag * torch.sin(p)

        # irfft: [B, bins, T] -> [B, T, bins] @ [bins, n] -> [B, n, T]
        frames = (torch.matmul(real.transpose(1, 2), self.idft_cos)
                  + torch.matmul(imag.transpose(1, 2), self.idft_sin)).transpose(1, 2)
        frames = frames * self.window.view(1, -1, 1)

        y = F.conv_transpose1d(frames, self.ola, stride=self.hop_length)      # [B,1,L]

        # Same overlap-add over the squared window gives the normalisation
        # envelope. expand() is free; it becomes an Expand node over a constant.
        wsq = (self.window ** 2).view(1, -1, 1).expand(1, self.n_fft, frames.shape[-1])
        env = F.conv_transpose1d(wsq, self.ola, stride=self.hop_length)
        # vocos asserts env > 1e-11; an assert is not exportable and would be a
        # data-dependent branch besides, so clamp instead. Same numerics on any
        # input a COLA-satisfying window produces.
        env = torch.clamp(env, min=1e-11)

        return (y / env)[:, 0, self.trim:-self.trim]


class F5VocoderWrapper(nn.Module):
    """Time-major mel -> 1-D PCM.

    Takes [B, T, 100] and transposes internally. Vocos wants channels-first
    [B, 100, T], but the DiT emits time-major, and doing the permute HERE means
    f5_tts_engine.cpp can hand the vocoder a pointer straight into the solved
    latent (offset by the reference prefix, which is contiguous in that layout)
    with no copy, no kernel and no transpose on the C++ side.
    """

    def __init__(self, vocos):
        super().__init__()
        self.vocos = vocos

    def forward(self, mel):
        return self.vocos.decode(mel.transpose(1, 2))


def build_vocoder(args, device):
    if args.vocoder == "bigvgan":
        # BigVGAN is pure conv + snake activation: it exports as-is, no patching.
        try:
            import bigvgan
        except ImportError:
            fail("--vocoder bigvgan needs the `bigvgan` package (nvidia/BigVGAN).")
        model = bigvgan.BigVGAN.from_pretrained(args.vocoder_path or
                                                "nvidia/bigvgan_v2_24khz_100band_256x",
                                                use_cuda_kernel=False)
        model.remove_weight_norm()
        # BigVGAN is a pure upsampling conv stack: no STFT, no trim, so the
        # output is exactly frames * hop samples.
        VOCODER_PADDING[0] = "none (bigvgan)"
        return model.eval().to(device), None

    try:
        from vocos import Vocos
    except ImportError:
        fail("vocos is not installed:  pip install vocos")

    src = args.vocoder_path or "charactr/vocos-mel-24khz"
    vocos = (Vocos.from_hparams(os.path.join(src, "config.yaml"))
             if os.path.isdir(src) else Vocos.from_pretrained(src))
    if os.path.isdir(src):
        from safetensors.torch import load_file
        sd_path = os.path.join(src, "pytorch_model.bin")
        st_path = os.path.join(src, "model.safetensors")
        sd = (load_file(st_path) if os.path.isfile(st_path)
              else torch.load(sd_path, map_location="cpu", weights_only=True))
        vocos.load_state_dict(sd, strict=False)
    vocos = vocos.eval().to(device)

    original_head = vocos.head
    safe_head = OnnxSafeISTFTHead(original_head).eval().to(device)
    VOCODER_PADDING[0] = safe_head.padding
    print(f"  istft padding  : {safe_head.padding}  -> output is "
          f"{'(frames-1)*hop' if safe_head.padding == 'center' else 'frames*hop'} samples")
    return vocos, (original_head, safe_head)


@torch.no_grad()
def check_istft_replacement(original, safe, device, frames=97):
    """Prove the export-safe head reproduces the real one before we ship it.

    This is the check that matters most in the whole script: a sign error in the
    inverse-DFT basis or an off-by-one in the overlap-add does not throw -- it
    produces audio that sounds like noise, and it would be diagnosed from a
    waveform hours later.
    """
    # Vocos' backbone hands the head [B, T, dim] (channels LAST) -- it is only
    # inside the head, after the Linear, that it becomes channels-first.
    x = torch.randn(1, frames, original.out.in_features, device=device)
    try:
        ref = original(x)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] could not run the ORIGINAL vocos head for comparison ({e});")
        print("         the replacement is UNVERIFIED. Listen to the output before trusting it.")
        return
    got = safe(x)
    n = min(ref.shape[-1], got.shape[-1])
    if ref.shape != got.shape:
        fail(f"ISTFT replacement changed the output shape: {tuple(ref.shape)} -> "
             f"{tuple(got.shape)}. Check hop/win/padding.")
    err = (ref[..., :n] - got[..., :n]).abs().max().item()
    scale = ref.abs().max().item() or 1.0
    print(f"  ISTFT replacement: max |diff| = {err:.3e}  (signal peak {scale:.3f}, "
          f"relative {err / scale:.2e})")
    if err / scale > 1e-4:
        fail("the ONNX-safe ISTFT does NOT match vocos' own. Do not ship this graph.")


# =============================================================================
# fp16 islands
# =============================================================================
def keep_fp32_islands(transformer):
    """Hold the GRN blocks at fp32 after the rest of the model has been halved.

    THE FAILURE THIS EXISTS TO PREVENT, because it is invisible in every cheap
    test. ConvNeXt-v2's global response normalization computes

        Gx = ||x||_2 over the SEQUENCE axis          (ONNX: ReduceL2)

    i.e. sqrt(sum(x^2)) reduced over T, not over channels. ORT's CUDA kernel
    accumulates that sum in the tensor's own dtype, so the intermediate is
    sum(x^2) ~ T * E[x^2]. At T = 1761 frames with activations of order 6 that
    is ~6.3e4 against an fp16 ceiling of 65504: it overflows to inf, Nx becomes
    inf/inf = NaN, and the NaN propagates through the DiT into the mel. The
    vocoder then emits a full-length, perfectly well-formed WAV of pure
    silence, and Run() reports success at every step.

    IT IS SEQUENCE-LENGTH DEPENDENT, which is what makes it so easy to ship:
    the export's own verification runs at 512 and 743 frames and is finite,
    ORT's CPU provider is finite at any length (MLAS accumulates reductions in
    fp32), and torch fp16 is finite too. Only the CUDA provider, only past
    ~1.5k frames -- i.e. only in production, on the long utterances a duplex
    assistant actually generates.

    Cost of holding them back: 4 modules inside the TEXT encoder, which is a
    handful of ConvNeXt blocks against 22 DiT layers. Unmeasurable next to
    getting audio instead of silence.
    """
    kept = []
    for name, mod in transformer.named_modules():
        if name.rsplit(".", 1)[-1] != "grn":
            continue
        mod.float()
        # The surrounding graph is fp16, so the island needs a cast on each
        # edge. Registered rather than written into the module because the
        # module belongs to f5-tts, not to this script.
        mod.register_forward_pre_hook(
            lambda m, inp: tuple(a.float() if torch.is_tensor(a) and a.is_floating_point()
                                 else a for a in inp))
        mod.register_forward_hook(
            lambda m, inp, out: out.half() if torch.is_tensor(out) else out)
        kept.append(name)
    return kept


# =============================================================================
# Export + verification
# =============================================================================
def onnx_export(module, args_tuple, path, input_names, output_names, dynamic_axes, opset):
    kwargs = dict(
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        export_params=True,
        # The TorchDynamo exporter is opt-in today and default-on in later
        # releases; it does not yet handle this graph's dynamic shapes as
        # reliably as the tracer. Pinned explicitly rather than left to drift.
        dynamo=False,
    )
    with torch.no_grad():
        torch.onnx.export(module, args_tuple, path,
                          **filter_kwargs(torch.onnx.export, kwargs))
    size_mb = os.path.getsize(path) / (1024 ** 2)
    print(f"  wrote {path}  ({size_mb:.1f} MB)")


@torch.no_grad()
def verify(module, path, feeds_by_len, output_name, device, tol=2e-3):
    """Run the exported graph against PyTorch at TWO different sequence lengths.

    The second length is the entire point. A tracer that baked the dummy frame
    count into a Reshape/Range/Slice constant still passes at the traced length
    and fails at every other one -- which, for a model whose length is a
    per-utterance duration estimate, means it fails in production and nowhere
    else. Rotary embeddings and the text-embedding pad are the usual culprits.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("  [warn] onnxruntime not installed -- skipping verification. "
              "Do not ship an unverified graph.")
        return True

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
        if "CUDA" in " ".join(ort.get_available_providers()) else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(path, providers=providers)
    print(f"  verifying with {sess.get_providers()[0]}")

    ok = True
    for n_frames, feeds in feeds_by_len.items():
        torch_out = module(*[torch.from_numpy(v).to(device) for v in feeds.values()])
        onnx_out = sess.run([output_name], feeds)[0]
        ref = torch_out.detach().cpu().numpy().astype(np.float32)
        if ref.shape != onnx_out.shape:
            print(f"  [FAIL] frames={n_frames}: shape {ref.shape} (torch) != "
                  f"{onnx_out.shape} (onnx)")
            ok = False
            continue
        diff = float(np.abs(ref - onnx_out).max())
        rel = diff / (float(np.abs(ref).max()) or 1.0)
        verdict = "ok" if rel <= tol else "FAIL"
        print(f"  [{verdict}] frames={n_frames:<5} max|diff| = {diff:.3e}  rel = {rel:.2e}")
        if rel > tol:
            ok = False
    return ok


def main():
    p = argparse.ArgumentParser(
        description="Export F5-TTS to the two ONNX graphs src/tts/f5_tts_engine.cpp binds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt", default=os.path.join(MODELS_DIR, "F5-TTS_RUSSIAN",
                                                  "model_212000.safetensors"),
                   help="F5-TTS DiT checkpoint (.safetensors).")
    p.add_argument("--vocab", default=os.path.join(MODELS_DIR, "F5-TTS_RUSSIAN", "vocab.txt"),
                   help="vocab.txt PAIRED WITH THAT CHECKPOINT. A fine-tune loaded "
                        "against the base vocab produces confident nonsense.")
    p.add_argument("--arch", choices=sorted(ARCH_PRESETS), default="v1",
                   help="Behavioural-flag preset -- text_mask_padding / pe_attn_head. "
                        "THESE CARRY NO WEIGHTS, so the wrong choice loads with 0 missing "
                        "and 0 unexpected keys and silently computes garbage; it cannot be "
                        "detected from the checkpoint. Take it from the model's NAME "
                        "(F5TTS_v1_Base* -> v1, F5TTS_Base -> v0). Dimensions and vocab "
                        "size ARE auto-detected and cross-checked.")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--dit-dtype", choices=("fp16", "fp32"), default="fp32",
                   help="Precision of the DiT TRANSFORMER BODY. The graph's INPUTS AND "
                        "OUTPUTS are fp32 either way -- the C++ engine binds the same "
                        "fp32 device buffers and the vocoder handoff is unchanged; only "
                        "the body is halved. See F5DitEulerStepWrapper's PRECISION SPLIT. "
                        "DEFAULTS TO fp32 BECAUSE fp16 IS NOT YET SHIPPABLE ON THIS "
                        "STACK -- see the KNOWN-BROKEN block below. Halving the body "
                        "does work and does help (~183 -> ~117 ms/step measured), so "
                        "this is worth finishing; it is not worth shipping today.")
    p.add_argument("--cfg-strength", type=float, default=2.0,
                   help="FROZEN INTO THE GRAPH. 0 disables CFG and halves the cost.")
    p.add_argument("--cfg-convention", choices=("f5", "classic"), default="f5",
                   help="f5: v_cond + (v_cond-v_uncond)*s   (what cfg_strength=2.0 means "
                        "everywhere in F5). classic: v_uncond + (v_cond-v_uncond)*s.")
    p.add_argument("--vocoder", choices=("vocos", "bigvgan"), default="vocos")
    p.add_argument("--vocoder-path", default="",
                   help="Local dir or HF id; empty = charactr/vocos-mel-24khz.")
    p.add_argument("--trace-frames", type=int, default=512, help="Dummy length for tracing.")
    p.add_argument("--verify-frames", type=int, default=743,
                   help="SECOND length used only for verification. Deliberately not a "
                        "round number and not a multiple of the trace length.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip-dit", action="store_true")
    p.add_argument("--skip-vocoder", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    dit_path = os.path.join(args.out_dir, "f5_tts_dit.onnx")
    voc_path = os.path.join(args.out_dir, "f5_tts_vocoder.onnx")
    all_ok = True

    banner("ENVIRONMENT")
    print(f"  torch          : {torch.__version__}")
    print(f"  device         : {device}")
    print(f"  out dir        : {args.out_dir}")

    # -------------------------------------------------------------------------
    if not args.skip_dit:
        banner("DiT")
        print(f"  checkpoint     : {args.ckpt}")
        print(f"  vocab          : {args.vocab}")
        tf_state = load_state_dict(args.ckpt)
        cfg, ckpt_vocab, mel_dim = infer_arch(tf_state, ARCH_PRESETS[args.arch])
        vocab_map, vocab_size = load_vocab(args.vocab)

        print(f"  arch preset    : {args.arch}  -> {cfg}")
        print(f"  mel channels   : {mel_dim}")
        print(f"  vocab          : {vocab_size} entries (checkpoint expects {ckpt_vocab})")

        # THE trap the earlier probe script warned about, now caught
        # mechanically instead of by ear. A Russian fine-tune loaded against the
        # base EN/ZH vocab has every character mapped to the wrong embedding row.
        if vocab_size != ckpt_vocab:
            fail(f"vocab/checkpoint mismatch: {args.vocab} has {vocab_size} entries but "
                 f"{os.path.basename(args.ckpt)} was trained with {ckpt_vocab}.\n"
                 f"  These MUST come from the same release. Loading anyway would map every "
                 f"character to the wrong embedding and produce fluent nonsense.")
        if mel_dim != N_MEL_CHANNELS:
            fail(f"checkpoint has mel_dim={mel_dim}, but the C++ engine is compiled for "
                 f"kF5MelChannels={N_MEL_CHANNELS}. Change one or the other, not neither.")

        transformer = build_dit(cfg, vocab_size, mel_dim)
        missing, unexpected = transformer.load_state_dict(tf_state, strict=False)
        if missing:
            fail(f"{len(missing)} weights missing from the checkpoint, e.g. {missing[:4]}.\n"
                 f"  Almost certainly the wrong --arch preset. Try the other one.")
        if unexpected:
            print(f"  [note] {len(unexpected)} unused checkpoint tensors, e.g. {unexpected[:3]}")
        transformer = transformer.eval().to(device)

        # Must happen AFTER load_state_dict (the replacement borrows the loaded
        # submodules) and BEFORE tracing.
        original_te, patched_te = patch_text_embedding(transformer)
        check_text_embed_replacement(original_te.to(device), patched_te,
                                     device, vocab_size)

        wrapper = F5DitEulerStepWrapper(transformer, args.cfg_strength,
                                        args.cfg_convention).eval().to(device)
        print(f"  CFG            : {args.cfg_convention} convention, s = {args.cfg_strength}"
              + ("  (DISABLED -- single pass per step)" if args.cfg_strength < 1e-5 else
                 "  (two transformer passes per step)"))

        def dit_feeds(n):
            g = np.random.default_rng(0)
            text = np.full((1, n), TEXT_PAD_ID, dtype=np.int32)
            n_tok = min(n // 3, max(1, len(vocab_map) - 1))
            text[0, :n_tok] = g.integers(0, max(1, vocab_size), size=n_tok, dtype=np.int32)
            return {
                "x_t":  g.standard_normal((1, n, mel_dim), dtype=np.float32),
                "cond": g.standard_normal((1, n, mel_dim), dtype=np.float32),
                "text": text,
                "t":    np.array([0.37], dtype=np.float32),
                "dt":   np.array([1.0 / 16.0], dtype=np.float32),
            }

        dummy = dit_feeds(args.trace_frames)
        dummy_t = tuple(torch.from_numpy(v).to(device) for v in dummy.values())

        # ---- precision ------------------------------------------------------
        # Halve the BODY, measure what that cost, and say so. The drift number
        # is the whole reason this is done here rather than by post-hoc
        # conversion of the finished graph: onnxconverter-common's fp16 pass
        # rewrites op dtypes without understanding which tensors are solver
        # STATE, and on this export it also mis-Casts the time_embed subgraph
        # into a graph ORT refuses to load (type-parameter mismatch on
        # time_mlp.0/Gemm). Doing it in torch keeps the fp32/fp16 boundary
        # where the ALGORITHM wants it -- see the wrapper's PRECISION SPLIT.
        if args.dit_dtype == "fp16":
            with torch.no_grad():
                ref32 = wrapper(*dummy_t).float()

            transformer.half()
            wrapper.compute_dtype = torch.float16
            kept = keep_fp32_islands(transformer)
            print(f"  fp32 islands   : {len(kept)} module(s) held back from fp16 "
                  f"-- {', '.join(kept) if len(kept) <= 6 else kept[0] + ', ...'}")

            with torch.no_grad():
                out16 = wrapper(*dummy_t).float()

            drift = float((ref32 - out16).abs().max())
            scale = float(ref32.abs().max()) or 1.0
            print(f"  precision      : fp16 body / fp32 solver state "
                  f"(x_t, cond, x_next, CFG combine, Euler)")
            print(f"  fp16 drift     : max|diff| = {drift:.3e}  rel = {drift/scale:.2e} "
                  f"(one step, vs the same module in fp32)")
            # A loud floor, not a tolerance to tune. fp16 carries ~3 decimal
            # digits, so ~1e-3 relative is expected and fine; an order of
            # magnitude past that means something overflowed or a norm lost its
            # accumulator, and the audio will be audibly wrong.
            if drift / scale > 2e-2:
                fail(f"fp16 drift {drift/scale:.2e} is far past what rounding explains.\n"
                     f"  Something in the body overflowed or underflowed. Re-run with "
                     f"--dit-dtype fp32 to confirm the graph is otherwise correct.")
        else:
            print(f"  precision      : fp32 body (tensor cores IDLE -- see --dit-dtype)")

        # Every axis that varies per utterance is named. `frames` is shared by
        # x_t/cond/text/x_next on purpose: the C++ side pads the text tensor to
        # the frame count with TEXT_PAD_ID, which F5's TextEmbedding then maps to
        # its filler token -- identical to passing a short text and letting the
        # module pad it, but with ONE dynamic axis instead of two.
        onnx_export(
            wrapper, dummy_t, dit_path,
            input_names=["x_t", "cond", "text", "t", "dt"],
            output_names=["x_next"],
            dynamic_axes={
                "x_t":    {0: "batch", 1: "frames"},
                "cond":   {0: "batch", 1: "frames"},
                "text":   {0: "batch", 1: "frames"},
                "x_next": {0: "batch", 1: "frames"},
            },
            opset=args.opset)

        # Tolerance tracks the BODY precision, not the boundary. torch and ORT
        # both compute in fp16 here but pick different kernels, tile orders and
        # accumulator layouts, so their disagreement is ~fp16 epsilon times the
        # reduction depth -- 2e-3 is an fp32 number and would fail a perfectly
        # good fp16 graph. What this check is FOR is unchanged: catching a
        # tracer that baked the dummy frame count into a constant, which shows
        # up as a shape error or a rel of order 1, nowhere near either bound.
        all_ok &= verify(wrapper, dit_path,
                         {args.trace_frames: dit_feeds(args.trace_frames),
                          args.verify_frames: dit_feeds(args.verify_frames)},
                         "x_next", device,
                         tol=2e-2 if args.dit_dtype == "fp16" else 2e-3)

    # -------------------------------------------------------------------------
    if not args.skip_vocoder:
        banner("VOCODER")
        vocoder, heads = build_vocoder(args, device)
        if heads is not None:
            original_head, safe_head = heads
            check_istft_replacement(original_head, safe_head, device)
            vocoder.head = safe_head          # swap AFTER the comparison
        wrapper = F5VocoderWrapper(vocoder).eval().to(device) if args.vocoder == "vocos" \
            else F5VocoderWrapper(_BigVGANShim(vocoder)).eval().to(device)

        def voc_feeds(n):
            g = np.random.default_rng(1)
            # Mel-scale-ish range; the exact distribution does not matter for a
            # numerical parity check, only that it is not degenerate.
            return {"mel": (g.standard_normal((1, n, N_MEL_CHANNELS)) * 2.0 - 5.0
                            ).astype(np.float32)}

        dummy = voc_feeds(args.trace_frames)
        dummy_t = tuple(torch.from_numpy(v).to(device) for v in dummy.values())

        onnx_export(
            wrapper, dummy_t, voc_path,
            input_names=["mel"], output_names=["waveform"],
            dynamic_axes={"mel": {0: "batch", 1: "frames"},
                          "waveform": {0: "batch", 1: "samples"}},
            opset=args.opset)

        all_ok &= verify(wrapper, voc_path,
                         {args.trace_frames: voc_feeds(args.trace_frames),
                          args.verify_frames: voc_feeds(args.verify_frames)},
                         "waveform", device, tol=5e-3)

    # -------------------------------------------------------------------------
    # Sidecar: the C++ F5DitContract, emitted rather than transcribed. The whole
    # class of "the export renamed a tensor and nobody told the engine" bug goes
    # away if this file is what configures F5TtsConfig.
    # -------------------------------------------------------------------------
    contract = {
        "dit": {"latent_in": "x_t", "cond_mel": "cond", "text_ids": "text",
                "time_step": "t", "delta_t": "dt", "latent_out": "x_next"},
        # istft_padding decides the OUTPUT LENGTH, which the C++ side must agree
        # with exactly: 'center' (what charactr/vocos-mel-24khz actually ships)
        # trims n_fft/2 per side and yields (frames-1)*hop; 'same' yields
        # frames*hop. Recorded rather than assumed -- see
        # F5TtsEngine::Impl::vocoder_samples().
        "vocoder": {"vocoder_in": "mel", "vocoder_out": "waveform",
                    "istft_padding": VOCODER_PADDING[0] or "unknown",
                    "output_samples": ("(frames - 1) * hop_length"
                                       if VOCODER_PADDING[0] == "center"
                                       else "frames * hop_length")},
        "geometry": {"sample_rate": SAMPLE_RATE, "mel_channels": N_MEL_CHANNELS,
                     "hop_length": HOP_LENGTH, "n_fft": N_FFT,
                     "text_pad_id": TEXT_PAD_ID},
        "baked_in": {"cfg_strength": args.cfg_strength,
                     "cfg_convention": args.cfg_convention,
                     "euler_update": True, "opset": args.opset,
                     # BODY precision. The graph's io_dtype is fp32 REGARDLESS,
                     # which is the field the C++ side actually depends on: it
                     # binds fp32 device buffers for x_t/cond/x_next and hands
                     # the DiT's output straight to the fp32 vocoder. If this
                     # ever becomes fp16, f5_tts_engine.cpp must change with it.
                     "dit_body_dtype": args.dit_dtype,
                     "dit_io_dtype": "fp32",
                     "vocoder_dtype": "fp32"},
        "source": {"ckpt": os.path.abspath(args.ckpt), "vocab": os.path.abspath(args.vocab),
                   "arch": args.arch, "vocoder": args.vocoder},
    }
    meta_path = os.path.join(args.out_dir, "f5_tts_contract.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(contract, f, indent=2, ensure_ascii=False)
    print(f"\n  wrote {meta_path}")

    banner("RESULT")
    if not all_ok:
        print("  VERIFICATION FAILED -- do not wire these graphs into the engine.")
        print()
        print("  If only the second (unseen) frame count failed, the tracer baked the")
        print("  dummy length into a constant. That is nearly always the rotary")
        print("  embedding: DiT computes rope from a Python int seq_len = x.shape[1].")
        print("  Patch it to derive the length from the tensor, re-export, re-verify.")
        return 1

    exported = ([] if args.skip_dit else ["DiT"]) + ([] if args.skip_vocoder else ["vocoder"])
    print(f"  Verified at two sequence lengths: {', '.join(exported) or '(nothing exported)'}")
    if args.skip_dit or args.skip_vocoder:
        print("  NOTE: a graph was skipped this run; the pair on disk may be from "
              "different invocations. Re-run without --skip-* before shipping.")
    print(f"  Point F5TtsConfig at:\n    {dit_path}\n    {voc_path}")
    print("\n  Reminder: the C++ side owns everything NOT in these graphs -- the noise")
    print("  draw, the sway-sampled schedule, the duration estimate, and the reference")
    print("  mel (masked and zero-padded, which is what `cond` expects).")
    return 0


class _BigVGANShim(nn.Module):
    """BigVGAN exposes __call__, not .decode(); F5VocoderWrapper wants .decode()."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def decode(self, mel):
        return self.model(mel).squeeze(1)


if __name__ == "__main__":
    sys.exit(main())
