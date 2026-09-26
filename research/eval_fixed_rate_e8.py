"""Fixed-rate 4.00 bpw E8 with outlier mitigation -- no entropy coder anywhere.

The previous study (see FINDINGS.md) concluded that E8's 3.5 bpw headline is an
*entropy-coded* rate, and that a fixed-width E8 loses to 4-bit scalar schemes. This
module attacks that conclusion directly: it builds the tightest honest fixed-rate E8
-- exactly 32 bits per 8 weights, one register, closed-form decode, zero compression
-- and asks whether outlier mitigation can lift it past the scalar baselines on real
activations.

THE 32-BIT COSET PACK, AND WHY THE PARITY TRICK NEEDS CARE
----------------------------------------------------------
Eight coordinates in [-8, +7] would cost 8*4 = 32 bits, leaving no room for the coset
flag. The flag is bought back from D8's even-sum constraint: with sum(k) = 0 (mod 2),
the low bit of k_7 is implied by the other seven, so only its high 3 bits need storing.

    bit  0     coset flag (0 -> Z^8, 1 -> (Z+1/2)^8)
    bits 1-28  k_0..k_6 as offset-binary u_i = k_i + 8, four bits each
    bits 29-31 high three bits of u_7
    decode     u_7 = (high3 << 1) | (sum(u_0..u_6) & 1)

That last line is valid because sum(k_i) even <=> sum(u_i) even (the 8*8 = 64 offset is
even), so u_7 = sum_{i<7} u_i (mod 2). It holds in BOTH cosets: ``closest_e8`` returns a
point as k + 0.5*coset with k in D8, and adding 1/2 to all eight coordinates shifts the
sum by 4, which preserves parity.

**This is exactly why the coordinates cannot simply be clamped after decoding.** Clamping
one coordinate flips the parity of the sum, the even-sum invariant breaks, and the decoder
then reconstructs the wrong k_7 -- silently, on the minority of blocks that saturate. The
quantizer here is therefore box-constrained *by construction*
(``closest_d8_boxed``): round into the box, then repair parity with the cheapest single
+-1 move that stays inside the box. ``_self_test`` checks the pack round-trips bit-exactly
over the real tensor, which is the assertion the whole 4.00 bpw claim rests on.

THE TWO MITIGATIONS ARE AIMED AT DIFFERENT OUTLIERS
---------------------------------------------------
Worth stating plainly before reading the table, because the spec's two strategies are not
substitutes:

* **FWHT incoherence** disperses *weight* spikes, which is what causes coordinate
  saturation in a [-8, +7] box (this tensor has max|w|/rms = 52.9).
* **Sparse outlier retention** keyed on ||X[:, k]||_2 removes *activation* spikes, which is
  what dominates the output error -- but it does nothing about saturation, since a channel
  can have a huge activation and an unremarkable weight.

So each is measured on both axes, and the combination is measured too.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import e8_lattice_engine as e8  # noqa: E402

ARTIFACT_DIR = os.path.join(HERE, "artifacts")
E8_DIM = 8
BOX_LO, BOX_HI = -8, 7
SCALE_BITS = 16
BF16_BITS = 16


# --------------------------------------------------------------------------- #
# box-constrained E8
# --------------------------------------------------------------------------- #

def closest_d8_boxed(y: torch.Tensor, lo: int = BOX_LO, hi: int = BOX_HI) -> torch.Tensor:
    """Nearest point of {k in Z^8 : sum(k) even, lo <= k_i <= hi} for each row of y.

    Round-and-clamp gives the nearest point of Z^8 inside the box; if its sum is odd, the
    parity must be repaired, and the cheapest repair is a single coordinate moved by +-1
    (an even-magnitude move cannot change parity, and +-3 is strictly worse than +-1).
    Moving cost is exact and closed form: shifting coordinate j up costs 1 - 2d_j and down
    costs 1 + 2d_j, for d_j = y_j - r_j. Moves that would leave the box are excluded, which
    is what makes the result a true projection onto the boxed lattice rather than a clamped
    approximation of one.
    """
    r = torch.round(y).clamp(float(lo), float(hi))
    d = y - r
    odd = (r.sum(dim=-1).to(torch.int64) & 1).bool()
    if not bool(odd.any()):
        return r
    inf = torch.full_like(d, float("inf"))
    cost_up = torch.where(r < hi, 1.0 - 2.0 * d, inf)
    cost_dn = torch.where(r > lo, 1.0 + 2.0 * d, inf)
    cost = torch.cat([cost_up, cost_dn], dim=-1)               # [..., 16]
    pick = cost.argmin(dim=-1, keepdim=True)
    coord = pick % E8_DIM
    step = torch.where(pick < E8_DIM, 1.0, -1.0).to(r.dtype)
    step = step * odd.unsqueeze(-1).to(r.dtype)                # no move where sum is even
    return r.scatter_add(-1, coord, step)


def closest_e8_boxed(y: torch.Tensor, lo: int = BOX_LO,
                     hi: int = BOX_HI) -> Tuple[torch.Tensor, torch.Tensor]:
    """Nearest boxed-E8 point; returns (k, coset) with the point equal to k + 0.5*coset."""
    a = closest_d8_boxed(y, lo, hi)
    b = closest_d8_boxed(y - 0.5, lo, hi)
    da = (y - a).pow(2).sum(dim=-1)
    db = (y - (b + 0.5)).pow(2).sum(dim=-1)
    take_b = db < da
    k = torch.where(take_b.unsqueeze(-1), b, a)
    return k, take_b.to(torch.uint8)


# --------------------------------------------------------------------------- #
# the 32-bit register pack
# --------------------------------------------------------------------------- #

def box_for(coord_bits: int) -> Tuple[int, int]:
    """Two's-complement coordinate box for a given width: 4 bits -> [-8, +7]."""
    half = 1 << (coord_bits - 1)
    return -half, half - 1


def pack_e8_fixed_rate(k: torch.Tensor, coset: torch.Tensor,
                       coord_bits: int = 4) -> torch.Tensor:
    """[N, 8] coords + [N] coset -> one word per block of exactly ``8 * coord_bits`` bits.

    The parity trick generalizes: the coset flag costs one bit and k_7's low bit is free,
    so the total is 1 + 7*b + (b-1) = 8*b bits for ANY coordinate width b -- i.e. exactly
    b bits per weight, with the coset carried for free. b=4 is the 32-bit register case.
    """
    lo, _ = box_for(coord_bits)
    u = k.to(torch.int64) - lo                                 # offset binary, [0, 2^b-1]
    mask = (1 << coord_bits) - 1
    word = coset.to(torch.int64).reshape(-1).clone()           # bit 0
    for i in range(7):
        word |= (u[:, i] & mask) << (1 + coord_bits * i)       # bits 1 .. 7b
    word |= (u[:, 7] >> 1) << (1 + coord_bits * 7)             # top b-1 bits
    return word


def unpack_e8_fixed_rate(word: torch.Tensor,
                         coord_bits: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
    """The decoder a kernel would run: 8 coordinates + coset out of one register."""
    lo, _ = box_for(coord_bits)
    mask = (1 << coord_bits) - 1
    w = word.to(torch.int64)
    coset = (w & 1).to(torch.uint8)
    u = torch.empty(w.shape[0], E8_DIM, dtype=torch.int64, device=word.device)
    for i in range(7):
        u[:, i] = (w >> (1 + coord_bits * i)) & mask
    parity = u[:, :7].sum(dim=1) & 1                           # the algebraic recovery
    high = (w >> (1 + coord_bits * 7)) & (mask >> 1)
    u[:, 7] = (high << 1) | parity
    return (u + lo).to(torch.int16), coset


# --------------------------------------------------------------------------- #
# randomized Hadamard (incoherence) rotation
# --------------------------------------------------------------------------- #

def fwht(a: torch.Tensor) -> torch.Tensor:
    """Orthonormal fast Walsh-Hadamard transform along the last axis (power-of-two)."""
    n = a.shape[-1]
    assert n & (n - 1) == 0, "FWHT needs a power-of-two length"
    shape = a.shape
    a = a.clone()
    h = 1
    while h < n:
        a = a.reshape(*shape[:-1], n // (2 * h), 2, h)
        x = a[..., 0, :].clone()
        y = a[..., 1, :].clone()
        a[..., 0, :] = x + y
        a[..., 1, :] = x - y
        a = a.reshape(*shape)
        h *= 2
    return a / math.sqrt(n)


@dataclass
class Rotation:
    """An orthogonal Q applied along the reduction axis, shared by X and W.

    Y = X W^T = (X Q)(W Q)^T for any orthogonal Q, so rotating both sides leaves the layer
    output exactly unchanged while redistributing the weight spikes that make a [-8, +7]
    box saturate.

    13696 = 128 * 107, and no Hadamard matrix of order 107 exists, so a single full-length
    FWHT is impossible. Q is therefore ``diag(signs)`` followed by a 128-point FWHT inside
    each of 107 blocks -- optionally composed with a dense random orthogonal on the
    107-axis (``mix_blocks``), which is what makes the mixing global rather than confined
    within each block of 128 channels.
    """
    signs: torch.Tensor
    block: int
    nblocks: int
    mix: Optional[torch.Tensor] = None

    def apply(self, t: torch.Tensor) -> torch.Tensor:
        out = t.to(torch.float32) * self.signs
        out = fwht(out.reshape(*t.shape[:-1], self.nblocks, self.block))
        if self.mix is not None:
            out = torch.einsum("...bc,ab->...ac", out, self.mix)
        return out.reshape(*t.shape[:-1], self.nblocks * self.block)


def make_rotation(dim: int, block: int = 128, seed: int = 0,
                  mix_blocks: bool = False, device: str = "cuda") -> Rotation:
    assert dim % block == 0, f"{dim} is not a multiple of {block}"
    nb = dim // block
    g = torch.Generator(device="cpu").manual_seed(seed)
    signs = (torch.randint(0, 2, (dim,), generator=g).float() * 2 - 1).to(device)
    mix = None
    if mix_blocks:
        q, _ = torch.linalg.qr(torch.randn(nb, nb, generator=g))
        mix = q.to(device)
    return Rotation(signs=signs, block=block, nblocks=nb, mix=mix)


# --------------------------------------------------------------------------- #
# fixed-rate quantizer
# --------------------------------------------------------------------------- #

@dataclass
class FixedRateResult:
    w_hat: torch.Tensor
    payload_bpw: float
    overhead_bpw: float
    sat_at_bound: float          # fraction of coordinates sitting on -8 or +7
    sat_clamped: float           # fraction the box actually moved (true saturation)
    extra: Dict[str, object] = field(default_factory=dict)

    @property
    def total_bpw(self) -> float:
        return self.payload_bpw + self.overhead_bpw


def _alpha_grid(n: int = 40, lo: float = 0.12, hi: float = 8.0,
                device: str = "cuda") -> torch.Tensor:
    return torch.exp(torch.linspace(math.log(lo), math.log(hi), n)).to(device)


def quantize_e8_fixed_rate(w: torch.Tensor, grid: Optional[torch.Tensor] = None,
                           coord_bits: int = 4,
                           verify_pack: bool = True) -> FixedRateResult:
    """Boxed E8 at exactly ``coord_bits`` bpw, with a per-row MSE-optimal step.

    The step is chosen per output row from a grid: the row's FP16 scale is stored either
    way, so per-row optimization is free, and inside a finite box the step is a genuine
    rate-distortion trade-off (too fine and the tail saturates, too coarse and granular
    noise dominates) rather than a monotone knob. Choosing it per row rather than globally
    is worth several dB on this tensor, because row scales span a wide range.
    """
    grid = grid if grid is not None else _alpha_grid(device=str(w.device))
    lo, hi = box_for(coord_bits)
    wf = w.to(torch.float32)
    rows, cols = wf.shape
    assert cols % E8_DIM == 0, "input dim must be a multiple of 8"
    rms = wf.pow(2).mean(dim=1, keepdim=True).sqrt().clamp(min=1e-30)

    best_mse = torch.full((rows, 1), float("inf"), device=wf.device, dtype=torch.float64)
    best_delta = torch.ones((rows, 1), device=wf.device, dtype=torch.float32)
    for a in grid.tolist():
        delta = (rms * a).to(torch.float16).to(torch.float32)
        k, coset = closest_e8_boxed((wf / delta).reshape(-1, E8_DIM), lo, hi)
        pt = (k + 0.5 * coset.unsqueeze(-1).to(k.dtype)).reshape(rows, cols)
        mse = (wf - pt * delta).pow(2).mean(dim=1, keepdim=True).to(torch.float64)
        take = mse < best_mse
        best_mse = torch.where(take, mse, best_mse)
        best_delta = torch.where(take, delta, best_delta)
        del k, coset, pt, mse

    # final pass at the chosen per-row step
    y = (wf / best_delta).reshape(-1, E8_DIM)
    k, coset = closest_e8_boxed(y, lo, hi)
    w_hat = (k + 0.5 * coset.unsqueeze(-1).to(k.dtype)).reshape(rows, cols) * best_delta

    # saturation, two ways: sitting on the boundary vs actually moved by the box
    sat_bound = float(((k <= lo) | (k >= hi)).to(torch.float64).mean())
    k_free, _ = e8.closest_e8(y)
    sat_clamped = float(((k_free < lo) | (k_free > hi)).to(torch.float64).mean())
    del k_free

    if verify_pack:
        word = pack_e8_fixed_rate(k, coset, coord_bits)
        k2, c2 = unpack_e8_fixed_rate(word, coord_bits)
        if not (torch.equal(k2.to(torch.int64), k.to(torch.int64))
                and torch.equal(c2, coset)):
            raise RuntimeError(
                f"the {8 * coord_bits}-bit coset pack did not round-trip: the even-sum "
                f"invariant the parity recovery depends on has been violated, so the "
                f"{coord_bits}.00 bpw format is not lossless and every number built on "
                f"it is meaningless")
        del word, k2, c2

    n_blocks = (rows * cols) // E8_DIM
    return FixedRateResult(
        w_hat=w_hat,
        payload_bpw=n_blocks * (E8_DIM * coord_bits) / w.numel(),
        overhead_bpw=rows * SCALE_BITS / w.numel(),
        sat_at_bound=sat_bound, sat_clamped=sat_clamped,
        extra={"coord_bits": coord_bits, "box": [lo, hi],
               "alpha_min": float((best_delta / rms).min()),
               "alpha_max": float((best_delta / rms).max()),
               "pack_verified": bool(verify_pack)})


def select_outlier_channels(x: torch.Tensor, frac: float) -> torch.Tensor:
    """Top-``frac`` input channels by activation L2 norm ||X[:, k]||_2."""
    n = max(1, int(round(frac * x.shape[1])))
    norms = x.to(torch.float32).pow(2).sum(dim=0)
    return torch.topk(norms, n).indices.sort().values


def quantize_e8_sparse_outliers(w: torch.Tensor, outlier_idx: torch.Tensor,
                                grid: Optional[torch.Tensor] = None,
                                coord_bits: int = 4) -> FixedRateResult:
    """Keep the outlier input channels in BF16, quantize the compacted bulk at 4.00 bpw.

    The bulk columns are *gathered* before blocking, not masked in place, so the retained
    channels do not burn 4 bits each -- which is what makes the rate match the intended
    (1-a)*4.00 + a*16.0 accounting instead of 4.00 + a*16.0. The tail is zero-padded to a
    multiple of 8 (at most 7 columns of 13696, so under 0.06%).
    """
    rows, cols = w.shape
    mask = torch.ones(cols, dtype=torch.bool, device=w.device)
    mask[outlier_idx] = False
    bulk_idx = mask.nonzero(as_tuple=True)[0]
    n_bulk, n_out = int(bulk_idx.numel()), int(outlier_idx.numel())

    padded = int(math.ceil(n_bulk / E8_DIM) * E8_DIM)
    bulk = torch.zeros(rows, padded, dtype=torch.float32, device=w.device)
    bulk[:, :n_bulk] = w.to(torch.float32)[:, bulk_idx]
    res = quantize_e8_fixed_rate(bulk, grid, coord_bits)

    w_hat = torch.empty(rows, cols, dtype=torch.float32, device=w.device)
    w_hat[:, bulk_idx] = res.w_hat[:, :n_bulk]
    # BF16 is the reference dtype, so the retained columns are exact by construction
    w_hat[:, outlier_idx] = w.to(torch.float32)[:, outlier_idx]

    block_bits = E8_DIM * coord_bits
    bits = (rows * (padded // E8_DIM) * block_bits    # E8 payload for the bulk
            + rows * n_out * BF16_BITS               # retained columns
            + cols                                   # one-bit-per-channel index bitmap
            + rows * SCALE_BITS)                     # per-row FP16 scale
    payload = (rows * (padded // E8_DIM) * block_bits
               + rows * n_out * BF16_BITS) / w.numel()
    return FixedRateResult(
        w_hat=w_hat,
        payload_bpw=payload,
        overhead_bpw=bits / w.numel() - payload,
        sat_at_bound=res.sat_at_bound, sat_clamped=res.sat_clamped,
        extra={"n_outlier_channels": n_out, "outlier_frac": n_out / cols,
               "pad_columns": padded - n_bulk, "total_bpw_exact": bits / w.numel(),
               **res.extra})


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def eval_output(x_ref: torch.Tensor, w_ref: torch.Tensor,
                x_q: torch.Tensor, w_hat: torch.Tensor,
                chunk: int = 128) -> Dict[str, float]:
    """Act-SNR, cosine similarity (global and worst token), and L-inf of Y - Yhat.

    ``x_q``/``w_hat`` may live in a rotated basis; since Y = (XQ)(WQ)^T exactly, the
    comparison against the unrotated reference stays valid and no inverse is needed.
    Cosine similarity is the metric that matters for a decode step -- it is what survives
    the next RMSNorm -- and it is far more demanding than SNR: for isotropic error,
    CosSim ~= 1 - 10^(-SNR/10)/2, so 0.999 needs roughly 27 dB.
    """
    wt = w_ref.to(torch.float32).T.contiguous()
    wht = w_hat.to(torch.float32).T.contiguous()
    f64 = torch.float64
    y_sq = torch.zeros((), dtype=f64, device=x_ref.device)
    yh_sq = torch.zeros((), dtype=f64, device=x_ref.device)
    dot = torch.zeros((), dtype=f64, device=x_ref.device)
    err_sq = torch.zeros((), dtype=f64, device=x_ref.device)
    err_max = torch.zeros((), dtype=f64, device=x_ref.device)
    err_sum = torch.zeros((), dtype=f64, device=x_ref.device)
    tok_cos: List[torch.Tensor] = []
    n = 0
    for i in range(0, x_ref.shape[0], chunk):
        y = (x_ref[i:i + chunk].to(torch.float32) @ wt).to(f64)
        yh = (x_q[i:i + chunk].to(torch.float32) @ wht).to(f64)
        d = y - yh
        y_sq += y.pow(2).sum()
        yh_sq += yh.pow(2).sum()
        dot += (y * yh).sum()
        err_sq += d.pow(2).sum()
        err_max = torch.maximum(err_max, d.abs().max())
        err_sum += d.sum()
        tok_cos.append((y * yh).sum(1)
                       / (y.norm(dim=1) * yh.norm(dim=1)).clamp(min=1e-300))
        n += d.numel()
    tc = torch.cat(tok_cos)
    return {
        "act_snr_db": float(10.0 * torch.log10(y_sq / err_sq)),
        "cos_global": float(dot / (y_sq.sqrt() * yh_sq.sqrt())),
        "cos_token_min": float(tc.min()),
        "cos_token_mean": float(tc.mean()),
        "act_linf": float(err_max),
        "act_mean_bias": float(err_sum / n),
        "act_rms": float((y_sq / n).sqrt()),
    }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def load_activations(layer: int) -> torch.Tensor:
    path = os.path.join(ARTIFACT_DIR, f"calib_act_layer{layer}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing -- run: python research/capture_activations.py "
            f"--layers {layer}")
    return torch.load(path, weights_only=False)["x"]


def run_layer(layer: int, model_dir: str, device: str, alphas: List[float],
              seed: int, grid_n: int) -> List[Dict[str, object]]:
    w = e8.load_down_proj(model_dir, layer, device=device)
    x = load_activations(layer).to(device)
    wf = w.to(torch.float32)
    grid = _alpha_grid(grid_n, device=device)
    rows: List[Dict[str, object]] = []

    def add(name: str, res: Optional[FixedRateResult], w_hat: torch.Tensor,
            total_bpw: float, x_q: Optional[torch.Tensor] = None,
            w_ref: Optional[torch.Tensor] = None, wm_ref: Optional[torch.Tensor] = None):
        wm = e8.weight_metrics(wm_ref if wm_ref is not None else wf, w_hat)
        om = eval_output(x, wf, x_q if x_q is not None else x, w_hat)
        rec: Dict[str, object] = {
            "scheme": name, "total_bpw": total_bpw,
            "weight_sqnr_db": wm["weight_sqnr_db"], "weight_linf": wm["weight_linf"],
            **om,
        }
        if res is not None:
            rec.update({"sat_at_bound_pct": 100 * res.sat_at_bound,
                        "sat_clamped_pct": 100 * res.sat_clamped,
                        "payload_bpw": res.payload_bpw,
                        "overhead_bpw": res.overhead_bpw, "extra": res.extra})
        rows.append(rec)
        print(f"    {name:<38} bpw={total_bpw:6.3f}  SNR={rec['act_snr_db']:7.3f}  "
              f"cos={rec['cos_global']:.6f}  min_tok={rec['cos_token_min']:.6f}  "
              f"sat={rec.get('sat_clamped_pct', float('nan')):.3f}%")

    print(f"\n  layer {layer}: baselines")
    rows.append({"scheme": "BF16 baseline", "total_bpw": 16.0,
                 "weight_sqnr_db": float("inf"), "weight_linf": 0.0,
                 "act_snr_db": float("inf"), "cos_global": 1.0,
                 "cos_token_min": 1.0, "cos_token_mean": 1.0,
                 "act_linf": 0.0, "act_mean_bias": 0.0})
    print(f"    {'BF16 baseline':<38} bpw={16.0:6.3f}  SNR={'inf':>7}  cos=1.000000")
    q = e8.quantize_rtn_int4(wf)
    add("RTN INT4", None, q.w_hat, q.total_bpw)
    q = e8.quantize_lloyd_max(wf)
    add("Lloyd-Max 16-LUT", None, q.w_hat, q.total_bpw)

    print(f"  layer {layer}: fixed-rate E8 candidates")
    direct = quantize_e8_fixed_rate(wf, grid)
    add("E8 4.00bpw (direct)", direct, direct.w_hat, direct.total_bpw)

    for mix in (False, True):
        rot = make_rotation(wf.shape[1], 128, seed, mix_blocks=mix, device=device)
        w_rot = rot.apply(wf)
        x_rot = rot.apply(x)
        r = quantize_e8_fixed_rate(w_rot, grid)
        label = "E8 4.00bpw + FWHT" + (" (+block mix)" if mix else "")
        add(label, r, r.w_hat, r.total_bpw, x_q=x_rot, wm_ref=w_rot)
        del rot, w_rot, x_rot, r
        torch.cuda.empty_cache()

    for a in alphas:
        idx = select_outlier_channels(x, a)
        r = quantize_e8_sparse_outliers(wf, idx, grid)
        add(f"E8 4.00bpw + sparse outliers a={a:.1%}", r, r.w_hat,
            float(r.extra["total_bpw_exact"]))
        del r
        torch.cuda.empty_cache()

    # the combination, since the two mitigations target different outliers
    rot = make_rotation(wf.shape[1], 128, seed, mix_blocks=False, device=device)
    w_rot, x_rot = rot.apply(wf), rot.apply(x)
    idx = select_outlier_channels(x_rot, alphas[-1])
    r = quantize_e8_sparse_outliers(w_rot, idx, grid)
    add(f"E8 4.00bpw + FWHT + sparse a={alphas[-1]:.1%}", r, r.w_hat,
        float(r.extra["total_bpw_exact"]), x_q=x_rot, wm_ref=w_rot)
    del rot, w_rot, x_rot, r
    torch.cuda.empty_cache()

    # Beyond the 4.00 bpw brief: CosSim > 0.999 means Act-SNR > 27 dB (the two are the
    # same number, see eval_output), and nothing at 4 bpw is close. The same pack at 5
    # bits per coordinate is still exactly 5.00 bpw, so this prices the target.
    print(f"  layer {layer}: what CosSim > 0.999 costs (beyond the 4.00 bpw brief)")
    r5 = quantize_e8_fixed_rate(wf, grid, coord_bits=5)
    add("E8 5.00bpw (direct)", r5, r5.w_hat, r5.total_bpw)
    del r5
    torch.cuda.empty_cache()
    idx = select_outlier_channels(x, alphas[1])
    r5 = quantize_e8_sparse_outliers(wf, idx, grid, coord_bits=5)
    add(f"E8 5.00bpw + sparse a={alphas[1]:.1%}", r5, r5.w_hat,
        float(r5.extra["total_bpw_exact"]))
    del r5
    torch.cuda.empty_cache()
    return rows


def markdown_table(rows: List[Dict[str, object]], layer: int) -> str:
    def g(r, k, spec, dash="--"):
        v = r.get(k)
        if v is None:
            return dash
        if isinstance(v, float):
            if math.isinf(v):
                return "inf"
            if math.isnan(v):
                return dash
        return format(v, spec)

    out = [f"### Layer {layer} `down_proj` [4096, 13696]", "",
           "| Scheme | Eff. bpw | Act-SNR (dB) | CosSim (global) | CosSim (min token) "
           "| Linf(Y) | Sat. clamped | Sat. at bound | W-SQNR (dB) |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        def pct(key):
            v = r.get(key)
            return "--" if v is None else format(v, ".3f") + "%"
        flag = " **" if (isinstance(r.get("cos_global"), float)
                         and r["cos_global"] > 0.999) else ""
        out.append(
            f"| {r['scheme']}{flag} | {g(r, 'total_bpw', '.3f')} | "
            f"{g(r, 'act_snr_db', '.2f')} | {g(r, 'cos_global', '.6f')} | "
            f"{g(r, 'cos_token_min', '.6f')} | {g(r, 'act_linf', '.4f')} | "
            f"{pct('sat_clamped_pct')} | {pct('sat_at_bound_pct')} | "
            f"{g(r, 'weight_sqnr_db', '.2f')} |")
    out += ["", "`Sat. clamped` = coordinates the box actually moved; `Sat. at bound` = "
            "coordinates resting on the box edge. `**` marks CosSim > 0.999."]
    return "\n".join(out)


def _self_test(device: str = "cuda") -> None:
    """The pack must be lossless and the boxed decode must respect the box and parity."""
    g = torch.Generator(device="cpu").manual_seed(3)
    for coord_bits in (4, 5):
        lo, hi = box_for(coord_bits)
        y = (torch.randn(200000, 8, generator=g) * 6.0).to(device)   # forces saturation
        k, coset = closest_e8_boxed(y, lo, hi)
        assert int(k.min()) >= lo and int(k.max()) <= hi, "coordinate left the box"
        assert bool((k.sum(-1).to(torch.int64) % 2 == 0).all()), "even-sum broken"
        k2, c2 = unpack_e8_fixed_rate(pack_e8_fixed_rate(k, coset, coord_bits), coord_bits)
        assert torch.equal(k2.to(torch.int64), k.to(torch.int64)), "coords lost in pack"
        assert torch.equal(c2, coset), "coset lost in the pack"

        # optimal among boxed candidates reachable by a single +-1 parity repair
        best = (y - (k + 0.5 * coset.unsqueeze(-1))).pow(2).sum(-1)
        base = torch.round(y).clamp(lo, hi)
        for _ in range(300):
            cand = (base + torch.randint(-2, 3, y.shape, device=device).float()
                    ).clamp(lo, hi)
            odd = cand.sum(-1).to(torch.int64) % 2 != 0
            cand[odd, 0] = torch.where(cand[odd, 0] < hi,
                                       cand[odd, 0] + 1, cand[odd, 0] - 1)
            for shift in (0.0, 0.5):
                d = (y - (cand + shift)).pow(2).sum(-1)
                assert bool((d >= best - 1e-4).all()), "a closer boxed point exists"

    q = make_rotation(1024, 128, 0, mix_blocks=True, device=device)
    t = torch.randn(4, 1024, device=device)
    assert abs(float(q.apply(t).norm() - t.norm())) < 1e-2, "rotation is not orthogonal"
    print("self-test OK: boxed E8 exact, 32-bit pack lossless, rotation orthogonal")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", default=e8.DEFAULT_MODEL_DIR)
    ap.add_argument("--layers", type=int, nargs="+", default=[10, 3])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.001, 0.005, 0.01])
    ap.add_argument("--grid", type=int, default=40, help="step-search grid points")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "fixed_rate_e8_report.json"))
    ap.add_argument("--md-out", default=os.path.join(HERE, "FIXED_RATE_E8.md"))
    args = ap.parse_args()

    if args.self_test:
        _self_test(args.device)
        return 0
    t0 = time.time()
    print(f"device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    _self_test(args.device)

    report: Dict[str, object] = {"meta": {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__, "model_dir": args.model_dir,
        "box": [BOX_LO, BOX_HI], "bits_per_block": 32, "alphas": args.alphas,
        "grid_points": args.grid, "seed": args.seed,
    }}
    md = ["# Fixed-rate 4.00 bpw E8 + outlier isolation", "",
          "Zero-entropy path: 8 coordinates in one 32-bit register, closed-form decode, "
          "no nvCOMP anywhere. Generated by `research/eval_fixed_rate_e8.py`.", ""]
    for layer in args.layers:
        rows = run_layer(layer, args.model_dir, args.device, args.alphas,
                         args.seed, args.grid)
        report[f"layer{layer}"] = rows
        md.append(markdown_table(rows, layer))
        md.append("")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    with open(args.md_out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    print(f"\nwrote {args.out} and {args.md_out} in {time.time() - t0:.0f}s")
    # the report files are UTF-8; the Windows console may not be
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    print("\n" + "\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
