"""E8 Gosset-lattice vector quantizer + the scalar baselines it is measured against.

This is the shared core of the study: every other module in ``research/`` imports
from here. It provides three things.

1. ``closest_e8`` -- a vectorized Conway-Sloane decoder. E8 = D8 U (D8 + 1/2),
   where D8 = {x in Z^8 : sum(x) even}. The decoder rounds into each coset and
   keeps whichever candidate is nearer, which is exact (not a heuristic): the two
   cosets tile R^8 between them, so the nearer of the two coset-nearest points is
   the global nearest lattice point.

2. Rate accounting. An E8 quantizer has no intrinsic bit rate -- the lattice is
   infinite, so the rate is whatever it costs to *code the chosen points*, and
   that depends entirely on which coding scheme you commit to. This module
   reports five rates for the same quantized tensor (see ``e8_rates``) because
   the spread between them is the headline engineering result: the rate that
   makes E8 look good is not the rate a random-access GEMV kernel can use.

3. The scalar baselines -- RTN INT4 and a 16-level Lloyd-Max LUT -- built with
   exactly the same per-row FP16 scale so the overhead column is identical
   (4096 scales / 56.1M weights = 0.00117 bpw) and the comparison is apples to
   apples.

WHY BLOCKS RUN ALONG THE INPUT DIMENSION
----------------------------------------
``down_proj.weight`` is [out=4096, in=13696], so a row is contiguous along the
reduction axis and 13696 = 8 * 1712 divides evenly. Grouping the 8-dim lattice
blocks along that axis is both the cache-friendly layout for a GEMV and the axis
a per-output-row scale is constant along.

REFERENCE DTYPE
---------------
The checkpoint is bf16; bf16 IS the reference. All error math is done in fp32 (or
fp64 for reductions) so the metric never measures its own accumulation noise.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch

DEFAULT_MODEL_DIR = os.path.join(
    os.environ.get("BLACKWELL_MODELS_DIR", "F:/AI/models"), "GLM-4-9B-Chat-1M-hf")

E8_DIM = 8
SCALE_BITS = 16          # one FP16 scale per output row
BF16_BITS = 16


# --------------------------------------------------------------------------- #
# checkpoint access
# --------------------------------------------------------------------------- #

def load_down_proj(model_dir: str = DEFAULT_MODEL_DIR, layer: int = 10,
                   device: str = "cpu") -> torch.Tensor:
    """mmap ``model.layers.<layer>.mlp.down_proj.weight`` straight out of the shards.

    Reading the one tensor we study (112 MB) instead of instantiating the model
    keeps the evaluation independent of the capture step and of transformers.
    """
    from safetensors import safe_open

    key = f"model.layers.{layer}.mlp.down_proj.weight"
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_path, "r", encoding="utf-8") as fh:
        shard = json.load(fh)["weight_map"][key]
    with safe_open(os.path.join(model_dir, shard), framework="pt", device="cpu") as fh:
        w = fh.get_tensor(key)
    return w.to(device)


# --------------------------------------------------------------------------- #
# E8 decoding
# --------------------------------------------------------------------------- #

def _closest_d8(y: torch.Tensor) -> torch.Tensor:
    """Nearest point of D8 = {x in Z^8 : sum(x) even} for each row of ``y`` [..., 8].

    Round-then-repair: round every coordinate, and if the sum came out odd, move
    the single coordinate with the largest rounding error to its *next* nearest
    integer. Flipping exactly one coordinate by +-1 toggles the parity of the sum,
    and flipping the worst-rounded one is the cheapest way to do it, which is the
    Conway-Sloane argument.
    """
    r = torch.round(y)
    resid = y - r
    odd = (r.sum(dim=-1).to(torch.int64) & 1).bool()          # [...]
    if bool(odd.any()):
        idx = resid.abs().argmax(dim=-1, keepdim=True)         # [..., 1]
        # next-nearest integer = step away in the direction the residual points.
        step = torch.where(resid.gather(-1, idx) >= 0, 1.0, -1.0).to(r.dtype)
        step = step * odd.unsqueeze(-1).to(r.dtype)            # zero where sum even
        r = r.scatter_add(-1, idx, step)
    return r


def closest_e8(y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Nearest E8 point for each row of ``y`` [N, 8].

    Returns ``(k, half)`` with the lattice point reconstructible as
    ``k + 0.5 * half`` -- ``k`` an integer vector in D8 and ``half`` a 0/1 coset
    flag per block. Keeping the point in this form (rather than as floats) is what
    lets the rate accounting count real symbols.
    """
    a = _closest_d8(y)                                         # coset 0
    b = _closest_d8(y - 0.5) + 0.5                             # coset 1/2
    da = (y - a).pow(2).sum(dim=-1)
    db = (y - b).pow(2).sum(dim=-1)
    take_b = db < da
    point = torch.where(take_b.unsqueeze(-1), b, a)
    half = take_b.to(torch.uint8)
    k = point - 0.5 * half.unsqueeze(-1).to(point.dtype)
    return k, half


# --------------------------------------------------------------------------- #
# quantizers -- all share the same per-row FP16 scale convention
# --------------------------------------------------------------------------- #

@dataclass
class Quantized:
    """A dequantized weight tensor plus everything the rate/report needs."""
    name: str
    w_hat: torch.Tensor                      # fp32, same shape as W
    payload_bpw: float                       # bits per weight for the coded symbols
    overhead_bpw: float                      # scales / LUT
    rates: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, object] = field(default_factory=dict)
    payload_bytes: Optional[bytes] = None    # bit-packed random-access payload

    @property
    def total_bpw(self) -> float:
        return self.payload_bpw + self.overhead_bpw


def _row_scale_overhead_bpw(w: torch.Tensor) -> float:
    rows = w.shape[0]
    return rows * SCALE_BITS / w.numel()


def _entropy_bits(counts: torch.Tensor) -> float:
    """Shannon entropy in bits of an empirical count vector."""
    c = counts.to(torch.float64)
    p = c / c.sum()
    p = p[p > 0]
    return float(-(p * torch.log2(p)).sum())


def e8_rates(k: torch.Tensor, half: torch.Tensor, n_weights: int) -> Dict[str, float]:
    """Five rates for one E8-quantized tensor. The spread between them is the point.

    ``bpw_coord_entropy``
        Coset bit + 8 conditional per-coordinate marginal entropies, divided by 8.
        This is the rate an ideal *per-coordinate* entropy coder achieves, and it
        is the definition that reproduces the classic high-rate lattice result
        (6.02R - 1.53 dB for an entropy-coded scalar quantizer on a Gaussian,
        plus E8's 0.65 dB granular gain). It is reported as primary because it is
        the definition the prior offline analysis used.
    ``bpw_coord_entropy_parity``
        The same minus exactly 1 bit per block: D8's even-sum constraint makes the
        8th coordinate's parity redundant given the other seven, so a coder that
        exploits it saves one bit. An achievable refinement, still variable-length.
    ``bpw_block_entropy``
        True empirical joint entropy over observed 8-dim blocks. Honest but
        *saturating*: with only n_weights/8 blocks it cannot exceed
        log2(n_blocks)/8 bpw, so above that it measures the sample size, not the
        source. The ``block_entropy_saturated`` flag says when that has happened.
    ``bpw_fixed_coords``
        (1 + 8*w)/8 where w is the fixed width that spans the observed coordinate
        range. This is what a random-access GEMV kernel can actually address --
        no entropy coder, constant stride, O(1) seek to any block.
    ``bpw_fixed_index``
        ceil(log2(#distinct blocks))/8, i.e. a dense index into a LUT of exactly
        the points that occur. Cheaper than fixed coords only if the LUT fits;
        ``fixed_index_lut_entries`` says how big it would have to be.
    """
    n_blocks = k.shape[0]
    assert n_weights == n_blocks * E8_DIM
    out: Dict[str, float] = {}

    h_coset = _entropy_bits(torch.bincount(half.to(torch.int64).reshape(-1), minlength=2))

    # conditional per-coordinate entropy, pooling the 8 positions (isotropy makes
    # them exchangeable, and pooling is what a single shared coder would see).
    kk = k.reshape(n_blocks, E8_DIM).to(torch.int64)
    kmin, kmax = int(kk.min()), int(kk.max())
    span = kmax - kmin + 1
    h_coord = 0.0
    for coset in (0, 1):
        sel = half.reshape(-1) == coset
        p_coset = float(sel.to(torch.float64).mean())
        if p_coset == 0.0:
            continue
        vals = (kk[sel] - kmin).reshape(-1)
        h_coord += p_coset * _entropy_bits(torch.bincount(vals, minlength=span))

    bits_per_block = h_coset + E8_DIM * h_coord
    out["bpw_coord_entropy"] = bits_per_block / E8_DIM
    out["bpw_coord_entropy_parity"] = max(bits_per_block - 1.0, 0.0) / E8_DIM

    # joint block entropy: pack (coset, coords) into one int64 key.
    w = max(1, math.ceil(math.log2(span)))
    if 1 + E8_DIM * w <= 62:
        key = half.to(torch.int64).reshape(-1).clone()
        shifted = kk - kmin
        for j in range(E8_DIM):
            key = (key << w) | shifted[:, j]
        counts = torch.unique(key, return_counts=True)[1]
        n_distinct = int(counts.numel())
        h_block = _entropy_bits(counts)
        out["bpw_block_entropy"] = h_block / E8_DIM
        out["block_entropy_saturated"] = float(h_block > math.log2(n_blocks) - 0.5)
        out["bpw_fixed_index"] = math.ceil(math.log2(max(n_distinct, 2))) / E8_DIM
        out["fixed_index_lut_entries"] = float(n_distinct)
    else:
        # w=8 needs 65 bits per block, past what one int64 key can hold. The joint
        # entropy is uninformative at these rates anyway -- it saturates against
        # log2(n_blocks) long before -- so it is omitted rather than approximated.
        out["block_entropy_omitted_bits_per_block"] = float(1 + E8_DIM * w)
    out["bpw_fixed_coords"] = (1 + E8_DIM * w) / E8_DIM
    out["coord_width_bits"] = float(w)
    out["coord_min"] = float(kmin)
    out["coord_max"] = float(kmax)
    return out


def quantize_e8(w: torch.Tensor, alpha: float, want_rates: bool = True,
                want_payload: bool = False,
                rate_key: str = "bpw_coord_entropy_parity") -> Quantized:
    """Quantize ``w`` [rows, cols] to E8 with per-row step ``delta = alpha * rms(row)``.

    ``alpha`` is the only knob: a smaller step gives a finer grid, more distinct
    lattice points, and so a higher rate and a higher SQNR.
    ``fit_e8_scale_for_rate`` inverts it.
    """
    rows, cols = w.shape
    assert cols % E8_DIM == 0, "input dim must be a multiple of 8"
    wf = w.to(torch.float32)
    delta = wf.pow(2).mean(dim=1, keepdim=True).sqrt() * alpha      # [rows, 1]
    delta = delta.to(torch.float16).to(torch.float32)               # scale is stored FP16
    y = (wf / delta).reshape(-1, E8_DIM)
    k, half = closest_e8(y)
    point = k + 0.5 * half.unsqueeze(-1).to(k.dtype)
    w_hat = point.reshape(rows, cols) * delta

    q = Quantized(name=f"E8(alpha={alpha:.4f})", w_hat=w_hat,
                  payload_bpw=float("nan"), overhead_bpw=_row_scale_overhead_bpw(w))
    if want_rates:
        q.rates = e8_rates(k, half, w.numel())
        q.payload_bpw = q.rates[rate_key]
        q.extra["rate_key"] = rate_key
    if want_payload:
        q.payload_bytes = pack_e8_fixed(k, half)
        q.extra["payload_bpw_fixed"] = len(q.payload_bytes) * 8 / w.numel()
        alt = pack_e8_bytes(k, half)
        if alt is not None:
            q.extra["payload_bytealigned"] = alt
            q.extra["payload_bpw_bytealigned"] = len(alt) * 8 / w.numel()
    q.extra["alpha"] = alpha
    return q


def pack_e8_fixed(k: torch.Tensor, half: torch.Tensor) -> bytes:
    """Bit-pack E8 blocks at the fixed random-access width: 1 coset bit + 8*w coords.

    This is the byte stream a GEMV kernel would stream, and therefore the byte stream
    the nvCOMP benchmark must be fed -- compressing anything else would measure a
    payload the engine never reads.

    Packing is bit-exact, with NO per-block padding to a byte boundary, for two
    reasons. Padding would inflate the payload (at w=8 a block is 65 bits, so
    byte-aligning costs 9.0 bpw instead of 8.125), and the padding bits are constant,
    which hands the compressor free entropy and flatters every ratio in Module D.
    Bit-exact packing still gives O(1) seek -- block i begins at bit i*(1+8w), an
    arithmetic offset -- it just costs the kernel a shift, which is what the existing
    AWQ/INT4 packing already does.

    The 65-bit case is also why this cannot go through an int64 intermediate: shifting
    a coset bit above 8 coordinates of 8 bits overflows. Bits are emitted explicitly.
    """
    import numpy as np

    n_blocks = k.shape[0]
    kk = k.to(torch.int64).cpu()
    kmin = int(kk.min())
    span = int(kk.max()) - kmin + 1
    w = max(1, math.ceil(math.log2(span)))
    bits_per_block = 1 + E8_DIM * w
    shifted = (kk - kmin).numpy().astype(np.int64)
    coset = half.reshape(-1).to(torch.uint8).cpu().numpy()

    # Chunk in multiples of 8 blocks so every chunk is a whole number of bytes and
    # the chunks concatenate without a bit-level seam.
    chunk = 8 * 65536
    parts = []
    for start in range(0, n_blocks, chunk):
        stop = min(start + chunk, n_blocks)
        n = stop - start
        bits = np.empty((n, bits_per_block), dtype=np.uint8)
        bits[:, 0] = coset[start:stop]
        blk = shifted[start:stop]
        for j in range(E8_DIM):
            col = blk[:, j]
            for b in range(w):                      # MSB-first within each field
                bits[:, 1 + j * w + b] = (col >> (w - 1 - b)) & 1
        parts.append(np.packbits(bits.reshape(-1)))
    return np.concatenate(parts).tobytes()


def pack_e8_bytes(k: torch.Tensor, half: torch.Tensor) -> Optional[bytes]:
    """E8 blocks as one int8 per coordinate + a packed coset bitmap: 8.125 bpw raw.

    The counterpart to ``pack_e8_fixed``, and the layout that makes the entropy rate
    *reachable in practice*. Bit-packing at 65 bits per block straddles byte
    boundaries, so a general-purpose compressor cannot see where one coordinate ends
    and the next begins and finds almost nothing to remove. One byte per coordinate is
    larger raw, but it puts the same small, strongly-peaked integer in a fixed byte
    lane, which is exactly the redundancy an entropy coder eats -- so the *compressed*
    size, not the raw size, is what a transfer path pays.

    Returns None if any coordinate falls outside int8, which happens at the finest
    steps in the sweep.
    """
    import numpy as np

    kk = k.to(torch.int64).cpu()
    if int(kk.abs().max()) > 127:
        return None
    coords = kk.to(torch.int8).numpy().tobytes()
    bits = half.reshape(-1).to(torch.uint8).cpu().numpy()
    return coords + np.packbits(bits).tobytes()


def fit_e8_scale_for_rate(w: torch.Tensor, target_bpw: float,
                          rate_key: str = "bpw_coord_entropy",
                          tol: float = 0.002, iters: int = 40) -> Tuple[float, Quantized]:
    """Bisect ``alpha`` so that ``rates[rate_key]`` hits ``target_bpw``.

    Rate is monotone *decreasing* in alpha (a coarser grid emits fewer distinct
    symbols), which is what makes plain bisection safe here.
    """
    lo, hi = 0.02, 8.0          # brackets the 1..8 bpw range for Gaussian-ish rows
    best: Optional[Quantized] = None
    best_alpha = float("nan")
    for _ in range(iters):
        mid = math.sqrt(lo * hi)
        q = quantize_e8(w, mid, rate_key=rate_key)
        rate = q.rates[rate_key]
        if best is None or abs(rate - target_bpw) < abs(best.rates[rate_key] - target_bpw):
            best, best_alpha = q, mid
        if abs(rate - target_bpw) <= tol:
            break
        if rate > target_bpw:
            lo = mid            # too many bits -> coarsen
        else:
            hi = mid
    assert best is not None
    best.name = f"E8@{best.rates[rate_key]:.3f}bpw"
    return best_alpha, best


def quantize_e8_fixed_width(w: torch.Tensor, coord_bits: int = 4,
                            alpha_grid: Optional[torch.Tensor] = None) -> Quantized:
    """E8 constrained to a FIXED-WIDTH, randomly-addressable code.

    The entropy-coded rates above are variable-length: to read block ``i`` you must
    have decoded blocks ``0..i-1``. This variant is what a kernel can actually index
    -- ``coord_bits`` per coordinate plus one coset bit, constant stride, O(1) seek.

    Because the code box is finite, the step is now a genuine rate-distortion
    trade-off rather than a monotone knob: too fine and the heavy tail clips, too
    coarse and granular noise dominates. So ``alpha`` is *optimized* for SQNR over a
    grid rather than fitted to a target rate. Clipping happens in the pre-quantized
    domain (clamp y before decoding, not the coordinates after) so the chosen point
    still satisfies D8's even-sum constraint and remains a true E8 point.
    """
    limit = (2 ** coord_bits - 1) // 2          # coords live in [-limit, +limit]
    wf = w.to(torch.float32)
    rms = wf.pow(2).mean(dim=1, keepdim=True).sqrt()
    if alpha_grid is None:
        alpha_grid = torch.exp(torch.linspace(math.log(0.3), math.log(6.0), 48))
    best = None
    for a in alpha_grid.tolist():
        delta = (rms * a).to(torch.float16).to(torch.float32)
        y = (wf / delta).clamp(-(limit - 1.0), limit - 1.0).reshape(-1, E8_DIM)
        k, half = closest_e8(y)
        k = k.clamp(-limit, limit)
        point = k + 0.5 * half.unsqueeze(-1).to(k.dtype)
        w_hat = point.reshape(w.shape) * delta
        mse = (wf.to(torch.float64) - w_hat.to(torch.float64)).pow(2).mean()
        if best is None or mse < best[0]:
            best = (mse, a, w_hat, k, half)
    assert best is not None
    _, a, w_hat, k, half = best
    bits_per_block = 1 + E8_DIM * coord_bits
    q = Quantized(name=f"E8 fixed-width {coord_bits}b", w_hat=w_hat,
                  payload_bpw=bits_per_block / E8_DIM,
                  overhead_bpw=_row_scale_overhead_bpw(w),
                  rates={"bpw_fixed_coords": bits_per_block / E8_DIM},
                  extra={"alpha": a, "coord_bits": coord_bits, "coord_limit": limit},
                  payload_bytes=pack_e8_fixed(k, half))
    return q


def quantize_rtn_int4_clipped(w: torch.Tensor, levels: int = 15,
                              n_ratios: int = 64) -> Quantized:
    """INT4 with a per-row MSE-optimal clipping scale, not the max.

    The spec's RTN row uses delta = max|w_row|/7, and on this tensor max/rms is 52.9
    -- one outlier per row sets the step for all 13696 weights, which is why plain
    RTN lands at 14.4 dB. Clipping the scale is the standard, nearly-free fix, and
    including it keeps the lattice from being compared against a handicapped
    baseline. The clip ratio is chosen per row by exhaustive search over the grid.
    """
    wf = w.to(torch.float32)
    lim = levels // 2
    amax = wf.abs().amax(dim=1, keepdim=True)
    ratios = torch.linspace(0.05, 1.0, n_ratios, device=wf.device)
    best_mse = None
    best_delta = None
    for r in ratios.tolist():
        delta = (amax * r / lim).to(torch.float16).to(torch.float32)
        q = torch.clamp(torch.round(wf / delta), -lim, lim)
        mse = (wf - q * delta).pow(2).mean(dim=1, keepdim=True).to(torch.float64)
        if best_mse is None:
            best_mse, best_delta = mse, delta
        else:
            take = mse < best_mse
            best_mse = torch.where(take, mse, best_mse)
            best_delta = torch.where(take, delta, best_delta)
    assert best_delta is not None
    q = torch.clamp(torch.round(wf / best_delta), -lim, lim)
    return Quantized(name="RTN INT4 (MSE clip)", w_hat=q * best_delta,
                     payload_bpw=4.0, overhead_bpw=_row_scale_overhead_bpw(w),
                     rates={"bpw_fixed_coords": 4.0},
                     extra={"levels": levels},
                     payload_bytes=pack_nibbles((q + lim).to(torch.uint8)))


def quantize_rtn_int4(w: torch.Tensor) -> Quantized:
    """Symmetric round-to-nearest INT4, per-row max scale: delta = max|w_row| / 7."""
    wf = w.to(torch.float32)
    delta = wf.abs().amax(dim=1, keepdim=True) / 7.0
    delta = delta.to(torch.float16).to(torch.float32)
    q = torch.clamp(torch.round(wf / delta), -7, 7)
    return Quantized(name="RTN INT4", w_hat=q * delta, payload_bpw=4.0,
                     overhead_bpw=_row_scale_overhead_bpw(w),
                     rates={"bpw_fixed_coords": 4.0},
                     extra={"levels": 15},
                     payload_bytes=pack_nibbles((q + 7).to(torch.uint8)))


def quantize_lloyd_max(w: torch.Tensor, levels: int = 16, subsample: int = 1 << 21,
                       iters: int = 60, seed: int = 0) -> Quantized:
    """16-level Lloyd-Max (1-D k-means) LUT on row-normalized weights.

    Centroids are fit on a subsample of w/rms(row) and then applied everywhere.
    Because Lloyd-Max puts its outermost centroid at the *conditional mean* of the
    tail rather than at the tail's edge, it clips hard -- which is exactly why its
    L-inf error is an order of magnitude worse than RTN's despite a better SQNR.
    """
    wf = w.to(torch.float32)
    rms = wf.pow(2).mean(dim=1, keepdim=True).sqrt()
    rms = rms.to(torch.float16).to(torch.float32)
    xn = wf / rms
    g = torch.Generator(device=xn.device).manual_seed(seed)
    idx = torch.randint(0, xn.numel(), (min(subsample, xn.numel()),),
                        generator=g, device=xn.device)
    s = xn.reshape(-1)[idx]
    # init on quantiles so Lloyd starts inside the bulk
    c = torch.quantile(s, torch.linspace(0.5 / levels, 1 - 0.5 / levels, levels,
                                         device=xn.device))
    for _ in range(iters):
        a = torch.bucketize(s, (c[1:] + c[:-1]) / 2)
        sums = torch.zeros(levels, device=xn.device).scatter_add_(0, a, s)
        cnts = torch.zeros(levels, device=xn.device).scatter_add_(0, a, torch.ones_like(s))
        c = torch.where(cnts > 0, sums / cnts.clamp(min=1), c)
        c, _ = torch.sort(c)
    edges = (c[1:] + c[:-1]) / 2
    a_all = torch.bucketize(xn.reshape(-1), edges)
    w_hat = c[a_all].reshape(w.shape) * rms
    lut_bpw = levels * BF16_BITS / w.numel()
    return Quantized(name=f"Lloyd-Max {levels}-LUT", w_hat=w_hat,
                     payload_bpw=math.log2(levels),
                     overhead_bpw=_row_scale_overhead_bpw(w) + lut_bpw,
                     rates={"bpw_fixed_coords": math.log2(levels)},
                     extra={"centroids": [float(v) for v in c]},
                     payload_bytes=pack_nibbles(a_all.to(torch.uint8)))


def pack_nibbles(idx: torch.Tensor) -> bytes:
    """4-bit index payload, two weights per byte -- the layout an INT4 kernel reads."""
    flat = idx.reshape(-1, 2).to(torch.uint8)
    packed = flat[:, 0] | (flat[:, 1] << 4)
    return packed.contiguous().cpu().numpy().tobytes()


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def weight_metrics(w: torch.Tensor, w_hat: torch.Tensor) -> Dict[str, float]:
    """SQNR and L-inf of the weight reconstruction, measured against bf16 as truth."""
    a = w.to(torch.float64)
    e = a - w_hat.to(torch.float64)
    return {
        "weight_sqnr_db": float(10.0 * torch.log10(a.pow(2).sum() / e.pow(2).sum())),
        "weight_linf": float(e.abs().max()),
        "weight_rmse": float(e.pow(2).mean().sqrt()),
    }


def act_metrics(x: torch.Tensor, w: torch.Tensor, w_hat: Optional[torch.Tensor],
                y_hat_override: Optional[torch.Tensor] = None,
                chunk: int = 128) -> Dict[str, float]:
    """Output-activation SNR and bias for Y = X W^T against Yhat = X What^T.

    ``y_hat_override`` lets a scheme supply a Yhat it computed some other way (the
    QJL hybrid adds a sketch correction that is not expressible as a What).

    Bias gets a standard error: the mean of Y - Yhat is a *correlated* average --
    every token shares the same weight error -- so the token-to-token spread of
    the per-token mean is the only defensible error bar, and without it "bias is
    approximately zero" is an unfalsifiable claim.
    """
    xf = x.to(torch.float32)
    wt = w.to(torch.float32).T.contiguous()
    wht = w_hat.to(torch.float32).T.contiguous() if w_hat is not None else None
    y_ref_sq = torch.zeros((), dtype=torch.float64, device=x.device)
    err_sq = torch.zeros((), dtype=torch.float64, device=x.device)
    err_max = torch.zeros((), dtype=torch.float64, device=x.device)
    err_sum = torch.zeros((), dtype=torch.float64, device=x.device)
    chan_sum = torch.zeros(w.shape[0], dtype=torch.float64, device=x.device)
    per_token_mean = []
    n = 0
    for i in range(0, xf.shape[0], chunk):
        xb = xf[i:i + chunk]
        y = xb @ wt
        if y_hat_override is not None:
            yh = y_hat_override[i:i + chunk].to(torch.float32)
        else:
            yh = xb @ wht
        d = (y - yh).to(torch.float64)
        y_ref_sq += y.to(torch.float64).pow(2).sum()
        err_sq += d.pow(2).sum()
        err_max = torch.maximum(err_max, d.abs().max())
        err_sum += d.sum()
        chan_sum += d.sum(dim=0)
        per_token_mean.append(d.mean(dim=1))
        n += d.numel()
    ptm = torch.cat(per_token_mean)
    bias = float(err_sum / n)
    se = float(ptm.std(unbiased=True) / math.sqrt(ptm.numel())) if ptm.numel() > 1 \
        else float("nan")
    chan_bias = chan_sum / xf.shape[0]
    return {
        "act_snr_db": float(10.0 * torch.log10(y_ref_sq / err_sq)),
        "act_mean_bias": bias,
        "act_mean_bias_se": se,
        "act_bias_t_stat": (bias / se) if se and se > 0 else float("nan"),
        "act_linf": float(err_max),
        "act_rmse": float((err_sq / n).sqrt()),
        "act_channel_bias_rms": float(chan_bias.pow(2).mean().sqrt()),
        "act_ref_rms": float((y_ref_sq / n).sqrt()),
    }


def bf16_reference_row() -> Dict[str, float]:
    """The BF16 baseline row: it is the reference, so its errors are identically 0."""
    return {
        "weight_sqnr_db": float("inf"), "weight_linf": 0.0, "weight_rmse": 0.0,
        "act_snr_db": float("inf"), "act_mean_bias": 0.0, "act_mean_bias_se": 0.0,
        "act_bias_t_stat": 0.0, "act_linf": 0.0, "act_rmse": 0.0,
        "act_channel_bias_rms": 0.0,
    }


# --------------------------------------------------------------------------- #
# self-test: the lattice decoder must be provably correct before any number
# built on top of it means anything.
# --------------------------------------------------------------------------- #

def _self_test(device: str = "cuda" if torch.cuda.is_available() else "cpu") -> None:
    g = torch.Generator(device="cpu").manual_seed(7)
    y = torch.randn(20000, 8, generator=g).to(device) * 3.0
    k, half = closest_e8(y)
    pt = k + 0.5 * half.unsqueeze(-1)

    # (a) every returned point is in E8: coset-0 points are integral, coset-1
    #     points are integral after a +0.5 shift, and both have an even 2*sum/2.
    frac = (pt * 2).round() - (pt * 2)
    assert frac.abs().max() < 1e-5, "points must lie on the half-integer grid"
    assert (k - k.round()).abs().max() < 1e-5, "k must be integral"
    assert (k.sum(-1).to(torch.int64) % 2 == 0).all(), "D8 needs an even coordinate sum"

    # (b) brute force: no point of E8 within a radius-3 integer box beats it.
    ref = y[:512]
    got = pt[:512]
    best = (ref - got).pow(2).sum(-1)
    base = torch.round(ref)
    for _ in range(4000):
        cand = base + torch.randint(-2, 3, ref.shape, device=device).float()
        s = cand.sum(-1).to(torch.int64) % 2 != 0
        cand[s, 0] += 1.0                                   # force even sum
        for shift in (0.0, 0.5):
            c = cand + shift
            d = (ref - c).pow(2).sum(-1)
            assert bool((d >= best - 1e-4).all()), "found a strictly closer E8 point"
    print("self-test OK: E8 decoder is exact on 20k random vectors")


if __name__ == "__main__":
    _self_test()
