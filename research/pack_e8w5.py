#!/usr/bin/env python3
"""E8W5-A16 reference packer: the offline half of the 5-bit companded-E8 kernel format.

Part V (``eval_multirate_sweep.py``) found that at b=5 a *companded* E8 lattice is the best
format measured at that rate, and Part IV's ``boundary_diagnostic()`` bounds what it can
win: +0.522 dB over NF5 in the weight domain at this coordinate width. This module turns
that quantizer into a byte-exact, 16-byte-aligned weight format a CUDA kernel can stream,
and verifies the two properties a format must have before a kernel is written against it:

1. the pack round-trips bit-exactly against the encoder's own (k, coset) output, and
2. the *integer decode a kernel would actually run* -- modelled op for op, including the
   parity recovery and the bit-spread -- reproduces the reference reconstruction exactly.

Inherited unchanged from the research, so Part V's dB still applies to it: the boxed-E8
encoder (``closest_e8_boxed``), the quantile warp (``PiecewiseWarp``), the conditional-mean
2x32-entry codebook, and the lambda search.

NEW here, and therefore NOT covered by any measured dB in Part V: per-group-128 scales in
place of Part V's per-row scales, and the optional MMA coordinate permutation. Both are
measured by ``--neutrality``; read docs/E8W5_FORMAT_SPEC.md section 9 before quoting a
number from this file.

    python research/pack_e8w5.py --self-test              :: pack/decode verification, no model
    python research/pack_e8w5.py --neutrality             :: what the two new choices cost
    python research/pack_e8w5.py --pack-tensor --layer 10 :: pack a real tensor, print the BPW
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

import e8_lattice_engine as e8                                       # noqa: E402
from eval_fixed_rate_e8 import (                                     # noqa: E402
    _alpha_grid, box_for, closest_e8_boxed)
from eval_companded_e8 import PiecewiseWarp                          # noqa: E402
from eval_1d_vs_e8_kl import levels_nf4                              # noqa: E402

F64 = torch.float64

# --------------------------------------------------------------------------- #
# format constants -- these ARE the format; changing one changes the file format
# --------------------------------------------------------------------------- #

COORD_BITS = 5                          # b
E8_DIM = 8                              # coordinates per lattice block
N_LEVELS = 1 << COORD_BITS              # 32 levels per coordinate
LUT_N = 2 * N_LEVELS                    # 64 codebook entries, indexed by (coset, u)
GROUP = 128                             # weights per scale group == 16 blocks == 80 bytes
BLOCKS_PER_GROUP = GROUP // E8_DIM      # 16
LO, HI = box_for(COORD_BITS)            # -16, +15
SCALE_BITS = 16
BF16_BITS = 16
LUT_BITS = 16
COL_INDEX_BITS = 32                     # one int32 per retained outlier column

# plane-L bit geometry. Nibble i holds u_i & 0xF for i in 0..6; nibble 7 holds
# (u_7 & 0xE) | coset, because u_7's bit 0 is implied by D8's even-sum parity.
COSET_BIT = 4 * (E8_DIM - 1)            # 28
PARITY_MASK = 0x01111111                # bit 0 of nibbles 0..6
COSET_CLEAR = 0xEFFFFFFF                # everything except bit 28


# --------------------------------------------------------------------------- #
# coordinate layout: which 8 weights of a row form one E8 block
# --------------------------------------------------------------------------- #

LAYOUT_K_LINEAR = "k_linear"
LAYOUT_MMA_M16N8K16 = "mma_m16n8k16"
LAYOUTS = (LAYOUT_K_LINEAR, LAYOUT_MMA_M16N8K16)


def mma_m16n8k16_perm() -> torch.Tensor:
    """K offsets, inside one aligned 32-column window, of the 8 coordinates of each block.

    For ``mma.sync.aligned.m16n8k16`` with .f16 operands, the B-operand fragment gives lane
    ``t`` the four k values {2j, 2j+1, 2j+8, 2j+9} for j = t % 4, at n = t >> 2. Two
    consecutive k16 tiles therefore hand one lane exactly eight k values -- and eight is the
    E8 block size.

    This matters because the pack is *not* decodable coordinate-by-coordinate: u_7's low bit
    is recovered from the parity of the other seven, so whichever thread owns one coordinate
    of a block must own all eight. Grouping a block as 8 consecutive k (LAYOUT_K_LINEAR)
    spreads it over four MMA lanes, none of which can decode it alone. Grouping it by this
    permutation instead makes block ownership and lane ownership the same thing, with no
    ``__shfl_sync`` and no wasted decode.

    Returns a length-32 gather: position ``8*j + s`` is the K offset of coordinate ``s`` of
    block ``j`` within the window. All 32 offsets stay inside one window, hence inside one
    GROUP=128 scale group, which is what makes the block's single scale well defined.
    """
    rows: List[List[int]] = []
    for j in range(4):
        quad = [2 * j, 2 * j + 1, 2 * j + 8, 2 * j + 9]
        rows.append(quad + [16 + t for t in quad])
    perm = torch.tensor(rows, dtype=torch.int64).reshape(-1)
    assert perm.numel() == 32 and torch.equal(perm.sort().values,
                                              torch.arange(32)), "not a permutation"
    return perm


def layout_gather(n_cols: int, layout: str, device="cpu") -> Optional[torch.Tensor]:
    """Column gather that puts a row into block order, or None for the identity."""
    if layout == LAYOUT_K_LINEAR:
        return None
    if layout != LAYOUT_MMA_M16N8K16:
        raise ValueError(f"unknown layout {layout!r}")
    if n_cols % 32:
        raise ValueError(f"{LAYOUT_MMA_M16N8K16} needs a multiple of 32 columns, got {n_cols}")
    perm = mma_m16n8k16_perm().to(device)
    base = torch.arange(0, n_cols, 32, device=device).unsqueeze(1)
    return (base + perm.unsqueeze(0)).reshape(-1)


def invert_gather(g: Optional[torch.Tensor], n_cols: int) -> Optional[torch.Tensor]:
    if g is None:
        return None
    inv = torch.empty_like(g)
    inv[g] = torch.arange(n_cols, device=g.device)
    return inv


# --------------------------------------------------------------------------- #
# the pack -- two planes, both independently 16-byte aligned per scale group
# --------------------------------------------------------------------------- #

def pack_planes(k: torch.Tensor, coset: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """[nblk, 8] coordinates + [nblk] coset -> (plane_lo uint32, plane_hi uint8).

    Plane L, one uint32 per block: nibble i = u_i & 0xF for i in 0..6, and nibble 7 =
    (u_7 & 0xE) | coset -- the coset flag takes the slot that D8's parity frees.
    Plane H, one uint8 per block: bit i = u_i >> 4.

    40 bits per block = exactly 5.000 bpw payload, and per GROUP=128 weights the two planes
    are 64 B and 16 B, i.e. four and one aligned 16-byte vectors with zero padding.
    """
    u = (k.to(torch.int64) - LO)
    if int(u.min()) < 0 or int(u.max()) > N_LEVELS - 1:
        raise ValueError("coordinate outside the 5-bit box")
    if int((u.sum(dim=-1) & 1).max()) != 0:
        raise ValueError("odd coordinate sum: not a D8 point, so u_7 bit 0 is unrecoverable")

    lo4 = torch.zeros(u.shape[0], dtype=torch.int64, device=u.device)
    for i in range(E8_DIM - 1):
        lo4 |= (u[:, i] & 0xF) << (4 * i)
    lo4 |= ((u[:, 7] & 0xE) | coset.to(torch.int64).reshape(-1)) << COSET_BIT

    hi = torch.zeros(u.shape[0], dtype=torch.int64, device=u.device)
    for i in range(E8_DIM):
        hi |= ((u[:, i] >> 4) & 1) << i
    return lo4.to(torch.uint32), hi.to(torch.uint8)


def unpack_planes(plane_lo: torch.Tensor,
                  plane_hi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """The inverse of :func:`pack_planes`, written the plain way (reference, not kernel)."""
    lo4 = plane_lo.to(torch.int64)
    hi = plane_hi.to(torch.int64)
    coset = (lo4 >> COSET_BIT) & 1
    u = torch.empty(lo4.shape[0], E8_DIM, dtype=torch.int64, device=lo4.device)
    for i in range(E8_DIM - 1):
        u[:, i] = ((lo4 >> (4 * i)) & 0xF) | (((hi >> i) & 1) << 4)
    parity = torch.zeros_like(coset)
    for i in range(E8_DIM - 1):
        parity ^= (u[:, i] & 1)
    u[:, 7] = (((lo4 >> COSET_BIT) & 0xE) | parity) | (((hi >> 7) & 1) << 4)
    return (u + LO).to(torch.int16), coset.to(torch.uint8)


def cuda_decode_mirror(plane_lo: torch.Tensor, plane_hi: torch.Tensor,
                       codebook: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mirror of ``e8w5_decode_block`` in research/kernels/e8w5_dequant.cuh, op for op.

    Kept deliberately in the kernel's idiom rather than the reference's, so that the .cuh
    and this function can only agree if the kernel's integer algebra is right:

      * parity via ``__popc(lo4 & 0x01111111) & 1`` -- here XOR of the same seven bits;
      * the bit-28 fixup that makes nibble i equal u_i & 0xF for ALL eight coordinates,
        turning coordinate 7 from a special case into a uniform one;
      * the three-step spread of plane H's eight bits to the eight nibble positions;
      * a 6-bit codebook index whose top bit IS the coset flag.

    Returns (idx [nblk, 8] int64, values [nblk, 8] float32 read out of ``codebook``).
    """
    lo4 = plane_lo.to(torch.int64)
    hi = plane_hi.to(torch.int64)

    coset = (lo4 >> COSET_BIT) & 1                       # 1 shift + 1 and
    parity = torch.zeros_like(lo4)                       # __popc(lo4 & PARITY_MASK) & 1
    for i in range(E8_DIM - 1):
        parity ^= (lo4 >> (4 * i)) & 1
    lo4f = (lo4 & COSET_CLEAR) | (parity << COSET_BIT)   # 1 lop3: nibble plane now uniform

    x = hi                                               # spread bit i -> bit 4i
    x = (x | (x << 12)) & 0x000F000F
    x = (x | (x << 6)) & 0x03030303
    x = (x | (x << 3)) & 0x11111111
    spread = x

    base = coset << COORD_BITS                           # the codebook index's bit 5
    idx = torch.empty(lo4.shape[0], E8_DIM, dtype=torch.int64, device=lo4.device)
    for i in range(E8_DIM):
        nib = (lo4f >> (4 * i)) & 0xF                    # bfe.u32 d, L, 4i, 4
        msb = ((spread >> (4 * i)) & 1) << 4             # bfe + shl, or one shifted mask
        idx[:, i] = base | msb | nib                     # lop3: three-input OR
    return idx, codebook.to(torch.float32)[idx]


# --------------------------------------------------------------------------- #
# fitting: per-group step search + conditional-mean codebook + lambda search
# --------------------------------------------------------------------------- #

@dataclass
class E8W5Tensor:
    """A packed tensor. ``plane_lo``/``plane_hi`` are in *block* order, not K order."""
    plane_lo: torch.Tensor                  # uint32  [N, n_blocks]
    plane_hi: torch.Tensor                  # uint8   [N, n_blocks]
    scales: torch.Tensor                    # float16 [N, n_groups]
    codebook: torch.Tensor                  # float16 [64]
    outlier_cols: torch.Tensor              # int32   [n_out]
    outlier_vals: torch.Tensor              # bfloat16[N, n_out]
    shape: Tuple[int, int] = (0, 0)
    layout: str = LAYOUT_K_LINEAR
    bulk_cols: int = 0                      # real bulk columns before padding
    pad_cols: int = 0
    lam: float = 1.0
    meta: Dict[str, object] = field(default_factory=dict)


def _group_view(buf: torch.Tensor, group: Optional[int]) -> Tuple[torch.Tensor, int]:
    """[rows, cols] -> [rows, n_groups, g]. ``group=None`` means one group per row."""
    rows, cols = buf.shape
    g = cols if group is None else group
    if cols % g:
        raise ValueError(f"{cols} columns is not a multiple of the group size {g}")
    return buf.reshape(rows, cols // g, g), cols // g


def fit_e8w5(bulk: torch.Tensor, *, layout: str = LAYOUT_K_LINEAR,
             group: Optional[int] = GROUP, grid: Optional[torch.Tensor] = None,
             lambda_grid: Optional[List[float]] = None, iters: int = 2,
             n_knots: int = 2048) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                           torch.Tensor, torch.Tensor, Dict[str, object]]:
    """Fit codebook + scales and encode ``bulk`` [rows, cols], cols a multiple of the group.

    Returns (k, coset, scales, codebook, w_hat, meta). ``k``/``coset`` are in block order
    under ``layout``; ``w_hat`` is back in the caller's column order so it can be scored
    directly against ``bulk``.

    The step and the codebook are interdependent -- the codebook is the conditional mean of
    the *normalized* weights, which move when the step moves -- so they alternate, exactly
    as ``_bulk_e8_companded`` does. lambda is global to the tensor because the decoder's
    codebook depends on it; it is NOT a per-row parameter.
    """
    dev = bulk.device
    grid = _alpha_grid(40, device=str(dev)) if grid is None else grid
    lams = [1.0] if lambda_grid is None else list(lambda_grid)
    rows, cols = bulk.shape

    gather = layout_gather(cols, layout, device=dev)
    inv = invert_gather(gather, cols)
    buf = bulk if gather is None else bulk[:, gather]

    gv, n_groups = _group_view(buf, group)
    rms = gv.pow(2).mean(dim=2, keepdim=True).sqrt().clamp(min=1e-30)   # [rows, ngrp, 1]
    base_warp = PiecewiseWarp((gv / rms).reshape(-1), n_knots=n_knots, n_levels=N_LEVELS)

    def encode(vals: torch.Tensor, warp: PiecewiseWarp):
        t = warp.forward(vals).reshape(-1, E8_DIM)
        return closest_e8_boxed(t, LO, HI)

    def decode(k: torch.Tensor, c: torch.Tensor, warp: PiecewiseWarp,
               lut: Optional[torch.Tensor]) -> torch.Tensor:
        if lut is None:
            return warp.inverse(k + 0.5 * c.unsqueeze(-1).to(k.dtype))
        idx = c.unsqueeze(-1).to(torch.int64) * N_LEVELS + (k.to(torch.int64) - LO)
        return lut[idx].to(torch.float32)

    best = None
    for lam in lams:
        warp = base_warp.with_lambda(lam) if lam < 1.0 else base_warp
        lut: Optional[torch.Tensor] = None
        best_delta = rms.clone()
        best_mse = torch.full((rows, n_groups, 1), float("inf"), device=dev, dtype=F64)
        for _ in range(iters):
            best_mse = torch.full((rows, n_groups, 1), float("inf"), device=dev, dtype=F64)
            for a in grid.tolist():
                delta = (rms * a).to(torch.float16).to(torch.float32)
                k, c = encode(gv / delta, warp)
                rec = decode(k, c, warp, lut).reshape(rows, n_groups, -1) * delta
                mse = (gv - rec).to(F64).pow(2).mean(dim=2, keepdim=True)
                take = mse < best_mse
                best_mse = torch.where(take, mse, best_mse)
                best_delta = torch.where(take, delta, best_delta)
                del k, c, rec, mse
            # refit the 64-entry codebook as conditional means of the normalized weights
            v = gv / best_delta
            k, c = encode(v, warp)
            idx = (c.unsqueeze(-1).to(torch.int64) * N_LEVELS
                   + (k.to(torch.int64) - LO)).reshape(-1)
            vals = v.reshape(-1).to(F64)
            sums = torch.zeros(LUT_N, device=dev, dtype=F64).scatter_add_(0, idx, vals)
            cnts = torch.zeros(LUT_N, device=dev, dtype=F64).scatter_add_(
                0, idx, torch.ones_like(vals))
            ladder = torch.arange(LO, HI + 1, device=dev, dtype=torch.float32)
            fallback = warp.inverse(torch.stack([ladder, ladder + 0.5]).reshape(-1)).to(F64)
            lut = torch.where(cnts > 0, sums / cnts.clamp(min=1), fallback)
            del v, k, c, idx, vals, sums, cnts
        total = float(best_mse.mean())
        if best is None or total < best[0]:
            best = (total, lam, lut.clone(), best_delta.clone())
    assert best is not None
    _, lam_best, lut, delta = best
    warp = base_warp.with_lambda(lam_best) if lam_best < 1.0 else base_warp

    # the codebook is what the kernel reads, so round it to the stored precision BEFORE
    # scoring: a dB measured against a float64 table is a dB the kernel cannot reproduce.
    codebook = lut.to(torch.float16)
    k, c = encode(gv / delta, warp)
    rec = (codebook.to(torch.float32)[
        c.unsqueeze(-1).to(torch.int64) * N_LEVELS + (k.to(torch.int64) - LO)]
    ).reshape(rows, n_groups, -1) * delta
    w_hat = rec.reshape(rows, cols)
    if inv is not None:
        w_hat = w_hat[:, inv]

    sat = float(((k <= LO) | (k >= HI)).to(F64).mean())
    meta = {"lambda": lam_best, "lambda_grid": lams, "sat_at_bound": sat,
            "n_groups": n_groups, "group": (group if group is not None else cols),
            "layout": layout, "warp_knots": n_knots, "iters": iters}
    return k, c, delta.reshape(rows, n_groups).to(torch.float16), codebook, w_hat, meta


def pack_tensor(w: torch.Tensor, *, outlier_cols: Optional[torch.Tensor] = None,
                layout: str = LAYOUT_K_LINEAR, group: Optional[int] = GROUP,
                grid: Optional[torch.Tensor] = None,
                lambda_grid: Optional[List[float]] = None,
                iters: int = 2) -> Tuple[E8W5Tensor, torch.Tensor]:
    """Full pipeline on a [N, K] weight tensor. Returns (packed, w_hat[N, K]).

    Outlier columns are *gathered out* before quantizing, so a retained channel does not
    also burn a lattice code -- the rate is (1-a)*bulk + a*16, not bulk + a*16. The bulk is
    then zero-padded up to a multiple of the group size, which is why ``K % 128 == 0`` is
    not by itself enough to avoid padding: K minus the outlier count generally is not.
    """
    rows, cols = w.shape
    dev = w.device
    wf = w.to(torch.float32)
    if outlier_cols is None:
        outlier_cols = torch.empty(0, dtype=torch.int64, device=dev)
    outlier_cols = outlier_cols.to(dev).to(torch.int64).sort().values

    keep = torch.ones(cols, dtype=torch.bool, device=dev)
    keep[outlier_cols] = False
    bulk_idx = keep.nonzero(as_tuple=True)[0]
    n_bulk = int(bulk_idx.numel())
    g = cols if group is None else group
    step = g if layout == LAYOUT_K_LINEAR else max(g, 32)
    padded = int(math.ceil(n_bulk / step) * step)

    buf = torch.zeros(rows, padded, dtype=torch.float32, device=dev)
    buf[:, :n_bulk] = wf[:, bulk_idx]

    k, c, scales, codebook, bulk_hat, meta = fit_e8w5(
        buf, layout=layout, group=group, grid=grid, lambda_grid=lambda_grid, iters=iters)
    plane_lo, plane_hi = pack_planes(k, c)
    n_blocks = padded // E8_DIM
    plane_lo = plane_lo.reshape(rows, n_blocks)
    plane_hi = plane_hi.reshape(rows, n_blocks)

    w_hat = torch.empty(rows, cols, dtype=torch.float32, device=dev)
    w_hat[:, bulk_idx] = bulk_hat[:, :n_bulk]
    out_vals = wf[:, outlier_cols].to(torch.bfloat16)
    w_hat[:, outlier_cols] = out_vals.to(torch.float32)

    packed = E8W5Tensor(plane_lo=plane_lo, plane_hi=plane_hi, scales=scales,
                        codebook=codebook, outlier_cols=outlier_cols.to(torch.int32),
                        outlier_vals=out_vals, shape=(rows, cols), layout=layout,
                        bulk_cols=n_bulk, pad_cols=padded - n_bulk,
                        lam=float(meta["lambda"]), meta=meta)
    return packed, w_hat


# --------------------------------------------------------------------------- #
# rate accounting -- every stored bit, counted once
# --------------------------------------------------------------------------- #

def bpw_accounting(p: E8W5Tensor, rows: Optional[int] = None) -> Dict[str, float]:
    """Exact achieved bits per weight, including scales, codebook, indices and padding.

    ``rows`` overrides the packed object's row count. Every term except the per-tensor
    codebook is exactly linear in rows, so the accounting for a full [N, K] tensor is exact
    even when the fit that produced ``p`` only ran on a slice of the rows -- which is how
    this gets measured while a sweep is holding the machine's memory.
    """
    rows = p.shape[0] if rows is None else rows
    cols = p.shape[1]
    n_weights = rows * cols
    n_blocks = int(p.plane_lo.shape[1])
    n_groups = int(p.scales.shape[1])
    n_out = int(p.outlier_cols.numel())

    parts = {
        "plane_lo": rows * n_blocks * 32,
        "plane_hi": rows * n_blocks * 8,
        "scales": rows * n_groups * SCALE_BITS,
        "codebook": LUT_N * LUT_BITS,
        "outlier_vals": rows * n_out * BF16_BITS,
        "outlier_cols": n_out * COL_INDEX_BITS,
    }
    total = sum(parts.values())
    out = {f"bpw_{k}": v / n_weights for k, v in parts.items()}
    out.update({
        "bits_total": float(total),
        "bpw_total": total / n_weights,
        "bpw_payload": (parts["plane_lo"] + parts["plane_hi"]
                        + parts["outlier_vals"]) / n_weights,
        "bpw_metadata": (parts["scales"] + parts["codebook"]
                         + parts["outlier_cols"]) / n_weights,
        "pad_columns": float(p.pad_cols),
        "pad_waste_bpw": rows * p.pad_cols * COORD_BITS / n_weights,
        "bytes_total": total / 8.0,
    })
    return out


# --------------------------------------------------------------------------- #
# a group-scaled NF5 baseline, so the margin can be re-checked at G=128
# --------------------------------------------------------------------------- #

def fit_nf5(bulk: torch.Tensor, *, group: Optional[int] = GROUP,
            grid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """NF5 with the same per-group MSE-optimal step search the lattice gets.

    Part III's rule: a baseline gets the same tuning as the candidate, or the comparison is
    worthless. NF5 *is* companded Z^8, which is exactly why it is the baseline that bounds
    what the lattice's cell shape can add.
    """
    dev = bulk.device
    grid = _alpha_grid(40, device=str(dev)) if grid is None else grid
    gv, n_groups = _group_view(bulk, group)
    levels = levels_nf4(bulk.reshape(-1), n=N_LEVELS).to(dev).to(torch.float32)
    levels, _ = torch.sort(levels)
    edges = (levels[1:] + levels[:-1]) / 2
    # levels_nf4 returns quantiles in RAW weight units, so the step carries the 1/span
    # normalization that quantize_lut_1d applies. Without it the grid never reaches the
    # right scale and the baseline saturates -- a silently crippled baseline, which is the
    # one failure mode Part III warns about.
    span = float(levels.abs().max())
    rms = gv.pow(2).mean(dim=2, keepdim=True).sqrt().clamp(min=1e-30)
    best_mse = torch.full((bulk.shape[0], n_groups, 1), float("inf"), device=dev, dtype=F64)
    best = torch.zeros_like(gv)
    for a in grid.tolist():
        delta = (rms * a / span).to(torch.float16).to(torch.float32)
        q = torch.bucketize((gv / delta).contiguous(), edges)
        rec = levels[q] * delta
        mse = (gv - rec).to(F64).pow(2).mean(dim=2, keepdim=True)
        take = mse < best_mse
        best_mse = torch.where(take, mse, best_mse)
        best = torch.where(take, rec, best)
        del q, rec, mse
    return best.reshape(bulk.shape)


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #

def _sqnr_db(ref: torch.Tensor, got: torch.Tensor) -> float:
    num = ref.to(F64).pow(2).sum()
    den = (ref.to(F64) - got.to(F64)).pow(2).sum().clamp(min=1e-300)
    return float(10.0 * torch.log10(num / den))


def self_test(device: str = "cpu", n_blocks: int = 1 << 16,
              seed: int = 0) -> Dict[str, object]:
    """Round-trip and kernel-mirror verification on exhaustive + random coordinates.

    No model and no GPU needed: these are properties of the bit layout, so they are checked
    against the encoder's own output rather than against a dB number.
    """
    torch.manual_seed(seed)
    out: Dict[str, object] = {}

    # 1. every coordinate value at every coordinate position, in both cosets
    ladder = torch.arange(LO, HI + 1, device=device, dtype=torch.int64)
    ks, cs = [], []
    for c in (0, 1):
        for pos in range(E8_DIM):
            blk = torch.zeros(ladder.numel(), E8_DIM, dtype=torch.int64, device=device)
            blk[:, pos] = ladder
            # repair parity in a *different* coordinate so `pos` keeps its exact value
            fix = 1 if pos != 1 else 2
            blk[:, fix] += (blk.sum(dim=1) & 1)
            ks.append(blk)
            cs.append(torch.full((ladder.numel(),), c, dtype=torch.uint8, device=device))
    k_ex = torch.cat(ks)
    c_ex = torch.cat(cs)

    # 2. random points straight off the boxed-E8 encoder, i.e. the real distribution
    y = torch.randn(n_blocks, E8_DIM, device=device) * 6.0
    k_rd, c_rd = closest_e8_boxed(y, LO, HI)
    k_all = torch.cat([k_ex, k_rd.to(torch.int64)])
    c_all = torch.cat([c_ex, c_rd])

    lo4, hi = pack_planes(k_all, c_all)
    k_rt, c_rt = unpack_planes(lo4, hi)
    out["roundtrip_bit_exact"] = bool(
        torch.equal(k_rt.to(torch.int64), k_all) and torch.equal(c_rt, c_all))
    out["blocks_checked"] = int(k_all.shape[0])
    out["coords_checked"] = int(k_all.numel())
    out["distinct_coord_values_seen"] = int((k_all - LO).unique().numel())
    out["both_cosets_seen"] = bool(c_all.unique().numel() == 2)

    # 3. the kernel's integer algebra must produce the codebook index the packer meant
    codebook = torch.arange(LUT_N, device=device, dtype=torch.float16)   # index == value
    idx_mirror, val_mirror = cuda_decode_mirror(lo4, hi, codebook)
    idx_ref = c_all.to(torch.int64).unsqueeze(-1) * N_LEVELS + (k_all - LO)
    out["mirror_index_bit_exact"] = bool(torch.equal(idx_mirror, idx_ref))
    out["mirror_value_exact"] = bool(torch.equal(val_mirror, idx_ref.to(torch.float32)))
    out["index_range"] = [int(idx_mirror.min()), int(idx_mirror.max())]

    # 4. alignment arithmetic -- the claims the layout section of the spec makes
    out["bits_per_block"] = E8_DIM * COORD_BITS
    out["bytes_plane_lo_per_group"] = BLOCKS_PER_GROUP * 4
    out["bytes_plane_hi_per_group"] = BLOCKS_PER_GROUP * 1
    out["group_bytes_total"] = BLOCKS_PER_GROUP * 5
    out["both_planes_16B_aligned"] = ((BLOCKS_PER_GROUP * 4) % 16 == 0
                                      and (BLOCKS_PER_GROUP * 1) % 16 == 0)

    # 5. the MMA permutation must be a permutation, and must not straddle a scale group
    perm = mma_m16n8k16_perm()
    out["mma_perm_valid"] = bool(torch.equal(perm.sort().values, torch.arange(32)))
    out["mma_block_within_one_group"] = bool(32 <= GROUP)
    return out


def load_weight_rows(model_dir: str, key: str, rows: Optional[int] = None,
                     device: str = "cpu") -> torch.Tensor:
    """Read the first ``rows`` rows of one safetensors tensor by byte range, without mmap.

    ``e8.load_down_proj`` mmaps the whole 4.3 GB shard, which fails with a commit-limit
    error (Windows error 1455) while a full-model sweep is holding 24 GB. Rows are
    contiguous in a row-major safetensors payload, so the slice we want is one seek and one
    read -- and reading 14 MB instead of mapping 4.3 GB is the difference between this
    experiment running beside the sweep and not running at all.
    """
    import struct

    with open(os.path.join(model_dir, "model.safetensors.index.json"), "r",
              encoding="utf-8") as fh:
        shard = json.load(fh)["weight_map"][key]
    path = os.path.join(model_dir, shard)
    with open(path, "rb") as fh:
        (hdr_len,) = struct.unpack("<Q", fh.read(8))
        hdr = json.loads(fh.read(hdr_len).decode("utf-8"))
        info = hdr[key]
        dtype = {"BF16": torch.bfloat16, "F16": torch.float16,
                 "F32": torch.float32}[info["dtype"]]
        shape = info["shape"]
        if len(shape) != 2:
            raise ValueError(f"{key} is not a matrix: {shape}")
        n_rows = shape[0] if rows is None else min(rows, shape[0])
        itemsize = torch.empty(0, dtype=dtype).element_size()
        row_bytes = shape[1] * itemsize
        start = 8 + hdr_len + info["data_offsets"][0]
        fh.seek(start)
        raw = fh.read(n_rows * row_bytes)
    t = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(n_rows, shape[1])
    return t.to(device)


def neutrality(model_dir: str, layer: int, rows: int, cols: int, grid_n: int,
               device: str) -> Dict[str, object]:
    """What the two choices Part V did NOT measure actually cost, on a real tensor slice.

    (a) per-group-128 scales instead of per-row scales, and
    (b) the MMA coordinate permutation instead of 8-consecutive-K blocks.

    NF5 is fitted at each scale granularity too, because the margin over NF5 is the claim
    the format rests on and NF5 also gains from finer scales.
    """
    key = f"model.layers.{layer}.mlp.down_proj.weight"
    w = load_weight_rows(model_dir, key, rows=rows, device=device).to(torch.float32)
    sl = w[:, :cols].contiguous()
    del w
    grid = _alpha_grid(grid_n, device=device)
    res: Dict[str, object] = {"slice": [rows, cols], "layer": layer, "grid_points": grid_n}
    cells = []
    for group in (None, GROUP):
        gname = "per_row" if group is None else f"group_{GROUP}"
        nf = fit_nf5(sl, group=group, grid=grid)
        nf_db = _sqnr_db(sl, nf)
        for layout in LAYOUTS:
            t0 = time.time()
            _, _, _, _, w_hat, meta = fit_e8w5(sl, layout=layout, group=group, grid=grid)
            db = _sqnr_db(sl, w_hat)
            cells.append({"scales": gname, "layout": layout, "e8w5_wsqnr_db": db,
                          "nf5_wsqnr_db": nf_db, "margin_db": db - nf_db,
                          "lambda": meta["lambda"], "sat_at_bound": meta["sat_at_bound"],
                          "seconds": round(time.time() - t0, 1)})
            print(f"  {gname:10s} {layout:14s} E8W5 {db:6.3f} dB   NF5 {nf_db:6.3f} dB"
                  f"   margin {db - nf_db:+.3f} dB   lam={meta['lambda']}"
                  f"   [{cells[-1]['seconds']}s]", flush=True)
    res["cells"] = cells
    by = {(c["scales"], c["layout"]): c for c in cells}
    res["layout_delta_db"] = {
        s: (by[(s, LAYOUT_MMA_M16N8K16)]["e8w5_wsqnr_db"]
            - by[(s, LAYOUT_K_LINEAR)]["e8w5_wsqnr_db"])
        for s in ("per_row", f"group_{GROUP}")}
    res["scale_delta_db"] = {
        l: (by[(f"group_{GROUP}", l)]["e8w5_wsqnr_db"] - by[("per_row", l)]["e8w5_wsqnr_db"])
        for l in LAYOUTS}
    return res


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--neutrality", action="store_true")
    ap.add_argument("--pack-tensor", action="store_true")
    ap.add_argument("--model-dir", default=e8.DEFAULT_MODEL_DIR)
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--device", default="cpu",
                    help="cpu by default: the GPU may be busy with a sweep")
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--full-rows", type=int, default=0,
                    help="report the BPW table for this many rows instead of the packed "
                         "row count; every term but the codebook is linear in rows")
    ap.add_argument("--cols", type=int, default=4096)
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--layout", default=LAYOUT_K_LINEAR, choices=LAYOUTS)
    ap.add_argument("--outlier-frac", type=float, default=0.001)
    ap.add_argument("--out", default=os.path.join(HERE, "e8w5_verification.json"))
    args = ap.parse_args()

    if not (args.self_test or args.neutrality or args.pack_tensor):
        ap.error("pick at least one of --self-test / --neutrality / --pack-tensor")

    report: Dict[str, object] = {"format": "E8W5-A16", "coord_bits": COORD_BITS,
                                 "group": GROUP, "codebook_entries": LUT_N}

    if args.self_test:
        print("== self-test: pack round-trip and CUDA decode mirror ==")
        st = self_test(device=args.device)
        for k, v in st.items():
            print(f"  {k:32s} {v}")
        ok = all(bool(st[k]) for k in ("roundtrip_bit_exact", "mirror_index_bit_exact",
                                       "mirror_value_exact", "both_planes_16B_aligned",
                                       "mma_perm_valid", "both_cosets_seen"))
        print(f"  => {'PASS' if ok else 'FAIL'}")
        report["self_test"] = st
        if not ok:
            print("self-test FAILED", file=sys.stderr)
            return 1

    if args.pack_tensor:
        print(f"\n== packing layer {args.layer} down_proj ({args.layout}) ==")
        w = load_weight_rows(args.model_dir,
                             f"model.layers.{args.layer}.mlp.down_proj.weight",
                             rows=(args.rows or None),
                             device=args.device).to(torch.float32)
        n_out = max(1, int(args.outlier_frac * w.shape[1])) if args.outlier_frac > 0 else 0
        # weight-side stand-in for the activation-energy selection the sweep uses; a real
        # packer must be handed the calibration-chosen columns (see spec section 8).
        out_cols = (torch.topk(w.abs().amax(dim=0), n_out).indices
                    if n_out else torch.empty(0, dtype=torch.int64, device=w.device))
        t0 = time.time()
        packed, w_hat = pack_tensor(w, outlier_cols=out_cols, layout=args.layout,
                                    grid=_alpha_grid(args.grid, device=args.device))
        acct = bpw_accounting(packed, rows=args.full_rows or None)
        db = _sqnr_db(w, w_hat)
        print(f"  shape {packed.shape}  bulk {packed.bulk_cols} + pad {packed.pad_cols}"
              f"  blocks/row {packed.plane_lo.shape[1]}"
              f"  groups/row {packed.scales.shape[1]}")
        print(f"  lambda {packed.lam}  W-SQNR {db:.3f} dB  ({time.time() - t0:.0f}s)")
        for k, v in sorted(acct.items()):
            print(f"  {k:24s} {v:.6f}")
        # the stored planes must decode to exactly the reconstruction that was scored
        k_rt, c_rt = unpack_planes(packed.plane_lo.reshape(-1), packed.plane_hi.reshape(-1))
        idx = c_rt.to(torch.int64).unsqueeze(-1) * N_LEVELS + (k_rt.to(torch.int64) - LO)
        rec = (packed.codebook.to(torch.float32)[idx]
               .reshape(packed.shape[0], -1, GROUP)
               * packed.scales.to(torch.float32).unsqueeze(-1)).reshape(packed.shape[0], -1)
        if packed.layout != LAYOUT_K_LINEAR:
            g = layout_gather(rec.shape[1], packed.layout, w.device)
            rec = rec[:, invert_gather(g, int(g.numel()))]
        keep = torch.ones(packed.shape[1], dtype=torch.bool, device=w.device)
        keep[packed.outlier_cols.to(torch.int64)] = False
        same = torch.equal(rec[:, :packed.bulk_cols],
                           w_hat[:, keep.nonzero(as_tuple=True)[0]])
        print(f"  planes decode to the scored reconstruction: {same}")
        report["pack_tensor"] = {"wsqnr_db": db, "lambda": packed.lam,
                                 "bulk_cols": packed.bulk_cols,
                                 "pad_cols": packed.pad_cols, "layout": packed.layout,
                                 "planes_match_scored_reconstruction": bool(same), **acct}

    if args.neutrality:
        print("\n== neutrality: what per-group scales and the MMA permutation cost ==")
        report["neutrality"] = neutrality(args.model_dir, args.layer, args.rows, args.cols,
                                          args.grid, args.device)
        print(f"  layout delta (MMA - linear): {report['neutrality']['layout_delta_db']}")
        print(f"  scale  delta (G128 - row):   {report['neutrality']['scale_delta_db']}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, default=str)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
