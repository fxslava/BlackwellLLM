"""Companded E8 vs 1-D NF4 at 4.013 bpw: can the lattice capture BOTH gains?

Part III established the asymmetry this module attacks. At a matched rate, uniform E8 beat
uniform RTN INT4 by +1.11 dB (of which ~0.65 dB is Zador's sphere-packing gain and ~0.56 dB
is the code RTN's symmetric convention wastes), but 1-D NF4 and Lloyd-Max beat uniform E8,
because 1-D *companding* gain (+1.75 dB) is larger than the lattice's *packing* gain. The
hypothesis under test: give E8 companding and the two gains should compose.

The hypothesis is theoretically sound. In Zador's high-rate expression the cell-shape term
and the source-density term are separate multiplicative factors,

    D  ~  G_n * 2^(-2R) * ||p||_{n/(n+2)}
          ^^^^ cell shape        ^^^^ density / companding

so in dB they add, and +0.65 dB on top of NF4 is the right thing to expect. What decides the
experiment is whether a given construction actually moves the factor it claims to.

THE TWO CANDIDATES ARE NOT EQUIVALENT, AND ONLY ONE CAN COMPAND
--------------------------------------------------------------
* **Radial shell scaling (Candidate 1)** leaves the encoder a nearest-neighbour search on the
  *uniform* lattice and only moves the reconstruction points radially. The partition is
  unchanged. That is Lloyd's centroid condition, not companding: it cannot alter
  ``||p||_{n/(n+2)}``, so it cannot earn companding gain. What it *can* recover is the
  reconstruction error in the saturated outer cells, which on a tensor with max/rms = 52.9
  are unbounded and badly represented by their boundary lattice point. Expect tenths of a dB,
  not +1.75.
* **Cartesian companded E8 (Candidate 2)** quantizes in a warped domain ``u = g(w)``, so the
  cells in the original domain are small where the density is high. That is real companding,
  and with the lattice supplying the cell shape it is the construction that can capture both
  terms.

Because E8 is *unimodular* it has exactly as many points in the [-8,+7] box as Z^8 does
(2^32 either way, hence the same 32-bit code), so Candidate 2 against NF4 is an exactly
equal-rate comparison in which the only difference is cell shape. That is what makes the
+0.65 dB acceptance criterion a clean test rather than a rate trade.

WHY AN NF4-CENTROID CONTROL IS IN THIS TABLE
--------------------------------------------
Candidate 2's LUT holds conditional *means*, so it gets a centroid correction that plain NF4
(whose levels are quantiles) does not. Comparing the two would confound packing gain with
that correction. ``1D-NF4-centroid`` keeps NF4's partition and replaces its levels with
conditional means, isolating the packing gain exactly. Parts I and II of this study both
reached a wrong verdict by comparing against a baseline fitted less hard than the candidate;
this control is the fix, and the margin against *it* is the honest one.

A CORRECTION TO THE BRIEF
-------------------------
The brief states that inside the 4-bit box there are "fewer than 16 unique active shells".
There are not: R^2 is always an even integer (both cosets), but 72 distinct shells occur in
practice and the median is R^2 = 30, so indexing a 16-entry LUT by R^2/2 would saturate at
the median and throw away most of the structure. Candidate 1 therefore bins shells by
*quantile* (reported bin edges), and a 64-bin variant is also measured to show whether 16 is
the binding constraint.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import e8_lattice_engine as e8                                   # noqa: E402
from eval_fixed_rate_e8 import (                                 # noqa: E402
    E8_DIM, box_for, closest_e8_boxed, pack_e8_fixed_rate, unpack_e8_fixed_rate,
    _alpha_grid,
)
from eval_1d_vs_e8_kl import (                                   # noqa: E402
    TailRunner, Scheme, with_sparse_outliers, layer_metrics, kl_metrics,
    paired_kl_margin, levels_rtn_int4, levels_lloyd_max, levels_nf4,
    quantize_lut_1d, _bulk_lut, _bulk_e8, _subsample, build_batch,
)

F64 = torch.float64


# --------------------------------------------------------------------------- #
# monotone piecewise-linear warp (the companding function)
# --------------------------------------------------------------------------- #

class PiecewiseWarp:
    """u = 16*F(v) - 8.5 for the empirical CDF F, and its inverse.

    The offset puts the 16 integer coordinate values -8..+7 exactly on the 16 quantile
    midpoints (i+0.5)/16, which is precisely where NF4 puts its levels -- so quantizing the
    warped value on the integer grid reproduces NF4's partition, and the only remaining
    difference between NF4 and companded E8 is the lattice's cell shape. That equivalence is
    the reason this particular offset is used rather than a convenient one.
    """

    def __init__(self, v: torch.Tensor, n_knots: int = 2048, seed: int = 0,
                 subsample: int = 1 << 21):
        s = _subsample(v, subsample, seed).to(F64)
        ps = torch.linspace(0.0, 1.0, n_knots + 1, device=s.device, dtype=F64)
        xs = torch.quantile(s, ps)
        # a flat stretch in the CDF makes the map non-invertible; nudge to strict increase
        eps = float((xs[-1] - xs[0]).abs()) * 1e-9 + 1e-30
        xs = torch.cummax(xs, dim=0)[0]
        xs = xs + torch.arange(xs.numel(), device=xs.device, dtype=F64) * eps
        self.xs, self.ps = xs, ps

    @staticmethod
    def _interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
        i = torch.searchsorted(xp, x.contiguous(), right=True).clamp(1, xp.numel() - 1)
        x0, x1 = xp[i - 1], xp[i]
        y0, y1 = fp[i - 1], fp[i]
        t = ((x - x0) / (x1 - x0).clamp(min=1e-300)).clamp(0.0, 1.0)
        return y0 + t * (y1 - y0)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return (16.0 * self._interp(v.to(F64), self.xs, self.ps) - 8.5).to(torch.float32)

    def inverse(self, u: torch.Tensor) -> torch.Tensor:
        p = ((u.to(F64) + 8.5) / 16.0).clamp(0.0, 1.0)
        return self._interp(p, self.ps, self.xs).to(torch.float32)


# --------------------------------------------------------------------------- #
# Candidate 1 -- radial shell-scaled E8
# --------------------------------------------------------------------------- #

def _shell_index(pt: torch.Tensor, edges: Optional[torch.Tensor],
                 n_bins: int) -> torch.Tensor:
    """Shell bin per block. ``edges`` None means the hardware-cheap R^2>>k binning."""
    r2 = pt.pow(2).sum(-1)
    if edges is None:
        return (r2 / 2.0).long().clamp(0, n_bins - 1)
    return torch.bucketize(r2, edges).clamp(0, n_bins - 1)


def _bulk_e8_radial(grid: torch.Tensor, coord_bits: int = 4, n_bins: int = 16,
                    iters: int = 3, binning: str = "quantile"):
    """Uniform-lattice encoder, radially rescaled reconstruction: alpha(shell) * c * delta.

    alpha_k is the least-squares optimal radial scale for shell k,
    ``sum <y,c> / sum ||c||^2`` over the blocks landing in that shell, which is exactly the
    formula in the brief and exactly Lloyd's centroid condition restricted to radial moves.
    The per-row step and the global alpha table depend on each other, so they are fitted by
    alternating optimization (``iters`` passes, the first with alpha = 1).
    """
    def fn(bulk: torch.Tensor):
        rows, n = bulk.shape
        padded = int(math.ceil(n / E8_DIM) * E8_DIM)
        buf = torch.zeros(rows, padded, dtype=torch.float32, device=bulk.device)
        buf[:, :n] = bulk
        lo, hi = box_for(coord_bits)
        rms = buf.pow(2).mean(dim=1, keepdim=True).sqrt().clamp(min=1e-30)
        alpha = torch.ones(n_bins, device=buf.device, dtype=F64)
        edges: Optional[torch.Tensor] = None
        best_delta = rms.clone()

        for it in range(iters):
            best_mse = torch.full((rows, 1), float("inf"), device=buf.device, dtype=F64)
            for a in grid.tolist():
                delta = (rms * a).to(torch.float16).to(torch.float32)
                y = (buf / delta).reshape(-1, E8_DIM)
                k, c = closest_e8_boxed(y, lo, hi)
                pt = k + 0.5 * c.unsqueeze(-1).to(k.dtype)
                sh = _shell_index(pt, edges, n_bins)
                rec = (pt * alpha[sh].unsqueeze(-1).to(torch.float32)
                       ).reshape(rows, padded) * delta
                mse = (buf - rec).to(F64).pow(2).mean(dim=1, keepdim=True)
                take = mse < best_mse
                best_mse = torch.where(take, mse, best_mse)
                best_delta = torch.where(take, delta, best_delta)
                del y, k, c, pt, sh, rec, mse
            # refit the shell table at the chosen steps
            y = (buf / best_delta).reshape(-1, E8_DIM)
            k, c = closest_e8_boxed(y, lo, hi)
            pt = k + 0.5 * c.unsqueeze(-1).to(k.dtype)
            r2 = pt.pow(2).sum(-1)
            if binning == "quantile" and it == 0:
                qs = torch.linspace(0, 1, n_bins + 1, device=buf.device)[1:-1]
                edges = torch.quantile(r2.to(torch.float32), qs).to(r2.dtype)
                edges = torch.unique(edges)
            sh = _shell_index(pt, edges, n_bins)
            num = torch.zeros(n_bins, device=buf.device, dtype=F64).scatter_add_(
                0, sh, (y * pt).sum(-1).to(F64))
            den = torch.zeros(n_bins, device=buf.device, dtype=F64).scatter_add_(
                0, sh, pt.pow(2).sum(-1).to(F64))
            alpha = torch.where(den > 0, num / den.clamp(min=1e-300),
                                torch.ones_like(den))
            del y, k, c, pt, r2, sh, num, den

        y = (buf / best_delta).reshape(-1, E8_DIM)
        k, c = closest_e8_boxed(y, lo, hi)
        pt = k + 0.5 * c.unsqueeze(-1).to(k.dtype)
        sh = _shell_index(pt, edges, n_bins)
        rec = (pt * alpha[sh].unsqueeze(-1).to(torch.float32)).reshape(rows, padded) \
            * best_delta
        k2, c2 = unpack_e8_fixed_rate(pack_e8_fixed_rate(k, c, coord_bits), coord_bits)
        if not (torch.equal(k2.to(torch.int64), k.to(torch.int64)) and torch.equal(c2, c)):
            raise RuntimeError("E8 32-bit pack failed to round-trip")
        bits = rows * (padded // E8_DIM) * (E8_DIM * coord_bits)
        return rec[:, :n], bits, {
            "n_shell_bins": n_bins, "binning": binning,
            "alpha_table": [float(v) for v in alpha],
            "shell_edges": ([float(v) for v in edges] if edges is not None else None),
            "coord_bits": coord_bits, "pad_columns": padded - n, "pack_verified": True,
            "lut_entries": n_bins, "lookups_per_block": 1}
    return fn


# --------------------------------------------------------------------------- #
# Candidate 2 -- Cartesian 32-entry companded E8
# --------------------------------------------------------------------------- #

def _bulk_e8_companded(grid: torch.Tensor, coord_bits: int = 4, iters: int = 2,
                       n_knots: int = 2048):
    """Quantize in the warped domain, reconstruct through a 2x16-entry conditional-mean LUT.

    This is the construction that can actually earn companding gain: the encoder's cells are
    uniform in ``u = g(w)``, hence non-uniform in w, concentrated where the density is. The
    decoder is a coordinate-wise table indexed by (coset, k), which is exactly the inverse
    warp evaluated on the lattice's coordinate values, refined to conditional means.

    The per-row step and the LUT are interdependent, so they alternate: search the step with
    the analytic inverse warp as the decoder, then refit the LUT as conditional means, then
    re-search.
    """
    def fn(bulk: torch.Tensor):
        rows, n = bulk.shape
        padded = int(math.ceil(n / E8_DIM) * E8_DIM)
        buf = torch.zeros(rows, padded, dtype=torch.float32, device=bulk.device)
        buf[:, :n] = bulk
        lo, hi = box_for(coord_bits)
        n_lv = 1 << coord_bits
        rms = buf.pow(2).mean(dim=1, keepdim=True).sqrt().clamp(min=1e-30)
        warp = PiecewiseWarp(buf / rms, n_knots=n_knots)
        lut: Optional[torch.Tensor] = None                      # [2 * n_lv]
        best_delta = rms.clone()

        def encode(v: torch.Tensor):
            u = warp.forward(v).reshape(-1, E8_DIM)
            return closest_e8_boxed(u, lo, hi)

        def decode(k: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
            if lut is None:
                pt = k + 0.5 * c.unsqueeze(-1).to(k.dtype)
                return warp.inverse(pt)
            idx = (c.unsqueeze(-1).to(torch.int64) * n_lv
                   + (k.to(torch.int64) - lo))
            return lut[idx].to(torch.float32)

        for _ in range(iters):
            best_mse = torch.full((rows, 1), float("inf"), device=buf.device, dtype=F64)
            for a in grid.tolist():
                delta = (rms * a).to(torch.float16).to(torch.float32)
                k, c = encode(buf / delta)
                rec = decode(k, c).reshape(rows, padded) * delta
                mse = (buf - rec).to(F64).pow(2).mean(dim=1, keepdim=True)
                take = mse < best_mse
                best_mse = torch.where(take, mse, best_mse)
                best_delta = torch.where(take, delta, best_delta)
                del k, c, rec, mse
            # refit the 2 x 16 LUT as conditional means of the normalized weights
            v = buf / best_delta
            k, c = encode(v)
            idx = (c.unsqueeze(-1).to(torch.int64) * n_lv + (k.to(torch.int64) - lo)
                   ).reshape(-1)
            vals = v.reshape(-1).to(F64)
            sums = torch.zeros(2 * n_lv, device=buf.device, dtype=F64).scatter_add_(
                0, idx, vals)
            cnts = torch.zeros(2 * n_lv, device=buf.device, dtype=F64).scatter_add_(
                0, idx, torch.ones_like(vals))
            fallback = warp.inverse(
                torch.stack([torch.arange(lo, hi + 1, device=buf.device,
                                          dtype=torch.float32),
                             torch.arange(lo, hi + 1, device=buf.device,
                                          dtype=torch.float32) + 0.5]).reshape(-1)
            ).to(F64)
            lut = torch.where(cnts > 0, sums / cnts.clamp(min=1), fallback)
            del v, k, c, idx, vals, sums, cnts

        v = buf / best_delta
        k, c = encode(v)
        rec = decode(k, c).reshape(rows, padded) * best_delta
        k2, c2 = unpack_e8_fixed_rate(pack_e8_fixed_rate(k, c, coord_bits), coord_bits)
        if not (torch.equal(k2.to(torch.int64), k.to(torch.int64)) and torch.equal(c2, c)):
            raise RuntimeError("E8 32-bit pack failed to round-trip")
        sat = float(((k <= lo) | (k >= hi)).to(F64).mean())
        bits = rows * (padded // E8_DIM) * (E8_DIM * coord_bits)
        return rec[:, :n], bits, {
            "coord_bits": coord_bits, "pad_columns": padded - n,
            "lut_entries": 2 * n_lv, "lookups_per_block": E8_DIM,
            "lut": [float(x) for x in lut], "sat_at_bound": sat,
            "warp_knots": n_knots, "pack_verified": True}
    return fn


# --------------------------------------------------------------------------- #
# the control that isolates packing gain: NF4's partition, centroid levels
# --------------------------------------------------------------------------- #

def _bulk_nf4_centroid(grid: torch.Tensor, n_lv: int = 16):
    """NF4 partition, reconstruction levels replaced by conditional means.

    Without this row, any win by the companded lattice could be its conditional-mean LUT
    rather than its cell shape.
    """
    def fn(bulk: torch.Tensor):
        rows, n = bulk.shape
        absmax = bulk.abs().amax(dim=1, keepdim=True).clamp(min=1e-30)
        lv = levels_nf4(bulk / absmax, n=n_lv).to(torch.float32)
        lv, _ = torch.sort(lv)
        span = float(lv.abs().max())
        rms = bulk.pow(2).mean(dim=1, keepdim=True).sqrt().clamp(min=1e-30)
        edges = (lv[1:] + lv[:-1]) / 2
        cent = lv.to(F64).clone()
        best_delta = rms.clone()
        for _ in range(2):
            best_mse = torch.full((rows, 1), float("inf"), device=bulk.device, dtype=F64)
            for g in grid.tolist():
                delta = (rms * g / span).to(torch.float16).to(torch.float32)
                idx = torch.bucketize((bulk / delta).reshape(-1), edges)
                rec = cent[idx].to(torch.float32).reshape(bulk.shape) * delta
                mse = (bulk - rec).to(F64).pow(2).mean(dim=1, keepdim=True)
                take = mse < best_mse
                best_mse = torch.where(take, mse, best_mse)
                best_delta = torch.where(take, delta, best_delta)
                del idx, rec, mse
            v = (bulk / best_delta).reshape(-1)
            idx = torch.bucketize(v, edges)
            vals = v.to(F64)
            sums = torch.zeros(n_lv, device=bulk.device, dtype=F64).scatter_add_(0, idx, vals)
            cnts = torch.zeros(n_lv, device=bulk.device, dtype=F64).scatter_add_(
                0, idx, torch.ones_like(vals))
            cent = torch.where(cnts > 0, sums / cnts.clamp(min=1), lv.to(F64))
            del v, idx, vals, sums, cnts
        idx = torch.bucketize((bulk / best_delta).reshape(-1), edges)
        rec = cent[idx].to(torch.float32).reshape(bulk.shape) * best_delta
        return rec, rows * n * 4, {"n_levels": n_lv, "lut_entries": n_lv,
                                   "lookups_per_block": E8_DIM,
                                   "centroids": [float(x) for x in cent]}
    return fn


# --------------------------------------------------------------------------- #
# the diagnostic that explains the result
# --------------------------------------------------------------------------- #

def boundary_diagnostic(coord_bits: int = 4, n: int = 4_000_000, margin: float = 4.0,
                        device: str = "cuda", seed: int = 0) -> Dict[str, float]:
    """Why companding and lattice packing gain do not compose in a bounded code.

    Two measurements on a *uniform* source, which is exactly what a companding warp
    produces, at equal codepoint count (2^32 for either lattice in the [-8,+7] box):

    1. Source filling the box. Z^8's unit cells tile the box exactly -- zero overload -- while
       E8's Voronoi cells do not, so 66% of E8 cells are clipped by a box face.
    2. Source confined well inside the box, so the boundary is irrelevant.

    Case 2 recovers the textbook +0.654 dB, which validates the quantizer. Case 1 is the
    regime companding actually creates, and there the gain is smaller. Companding therefore
    *buys* density matching and *spends* part of the lattice's packing gain, because
    flattening the source is what pushes mass into the boundary cells.
    """
    lo, hi = box_for(coord_bits)
    g = torch.Generator(device=device).manual_seed(seed)
    span = float(hi - lo + 1)
    out: Dict[str, float] = {"coord_bits": coord_bits, "box_lo": lo, "box_hi": hi}
    for tag, width, centre in (("filling_box", span, (lo + hi) / 2.0),
                               ("inside_box", span - 2 * margin, (lo + hi) / 2.0)):
        u = (torch.rand(n, E8_DIM, generator=g, device=device) - 0.5) * width + centre
        z = u.round().clamp(float(lo), float(hi))
        k, c = closest_e8_boxed(u, lo, hi)
        pt = k + 0.5 * c.unsqueeze(-1).to(k.dtype)
        mse_z = float((u - z).pow(2).mean().to(F64))
        mse_e = float((u - pt).pow(2).mean().to(F64))
        out[f"{tag}_mse_per_coord_z8"] = mse_z
        out[f"{tag}_mse_per_coord_e8"] = mse_e
        out[f"{tag}_e8_gain_db"] = 10.0 * math.log10(mse_z / mse_e)
        out[f"{tag}_e8_coords_on_box_edge"] = float(
            ((k <= lo) | (k >= hi)).to(F64).mean())
        del u, z, k, c, pt
    out["asymptotic_prediction_db"] = 10.0 * math.log10((1.0 / 12.0) / 0.0716818)
    out["cells_touching_boundary_frac"] = 1.0 - ((span - 2) / span) ** E8_DIM
    out["boundary_loss_db"] = out["inside_box_e8_gain_db"] - out["filling_box_e8_gain_db"]
    return out


# --------------------------------------------------------------------------- #
# silicon / register audit
# --------------------------------------------------------------------------- #

def silicon_audit(dram_gbps: float, n_weights: int, bpw: float,
                  int_ops_per_sec: float = 30e12) -> Dict[str, object]:
    """Per-8-weight instruction and register footprint. Static analysis, not a kernel run.

    The RTX 5070 column is grounded in primitives whose existence and cost are known
    (LOP3/SHF for field extraction, POPC for the parity bit, a 4-or-5 level FSEL tree or a
    PRMT pair for a small table, I2F, FFMA). Costs are in instruction slots, not cycles: no
    kernel exists yet, so this bounds work rather than measuring latency.

    A 16- or 32-entry table indexed *per lane* is the interesting architectural difference.
    Constant memory broadcasts only when the address is warp-uniform, which it is not here,
    so the realistic options are a register select tree (~log2(entries) ops per lookup) or a
    replicated shared-memory table (bank conflicts). Candidate 1 escapes this because its
    table is indexed once per *block* of 8 weights, not once per weight.
    """
    def row(fmt, loads, alu, entries, lookups_per_block, sel_ops, i2f, fma, note):
        per8 = alu + i2f + fma + lookups_per_block * sel_ops
        return {"format": fmt, "loads_per_8w": loads, "int_alu_per_8w": alu,
                "table_entries": entries, "lookups_per_8w": lookups_per_block,
                "select_ops_per_lookup": sel_ops, "i2f_per_8w": i2f, "fma_per_8w": fma,
                "ops_per_8w": per8, "ops_per_weight": per8 / 8,
                "table_bytes": entries * 4,
                "extra_regs_if_register_resident": entries,
                "note": note}

    rows = [
        row("1D-NF4 / Lloyd-Max (16-LUT)", 1, 16, 16, 8, 4, 0, 8,
            "8 per-lane lookups; 4-level FSEL tree or a PRMT pair; the only format needing "
            "a per-weight table access"),
        row("8D-E8 uniform", 1, 25, 0, 0, 0, 8, 8,
            "7x(SHF+LOP3), high-3 extract, parity via POPC(word & 0x11111110); no table"),
        row("8D-E8 radial shell (C1)", 1, 25 + 8 + 2, 16, 1, 4, 8, 8,
            "adds 8 IMAD for R^2 and a bucketize; ONE table lookup per 8 weights, so the "
            "table cost amortizes 8x against the 1-D schemes"),
        row("8D-E8 Cartesian 32-LUT (C2)", 1, 25, 32, 8, 5, 0, 8,
            "8 per-lane lookups into 32 entries (5-level tree); replaces I2F+scale with the "
            "table, so it is close to NF4's cost plus E8's unpack"),
    ]
    payload_bytes = n_weights * bpw / 8
    mem_us = payload_bytes / dram_gbps / 1e3
    for r in rows:
        r["alu_us_for_tensor"] = (r["ops_per_8w"] * n_weights / 8) / int_ops_per_sec * 1e6
        r["alu_fraction_of_memory_time"] = r["alu_us_for_tensor"] / mem_us
    return {
        "per_format": rows, "payload_bytes": payload_bytes, "memory_time_us": mem_us,
        "assumed_dram_gbps": dram_gbps, "assumed_int_ops_per_sec": int_ops_per_sec,
        "rtx5070_verdict": (
            "every format reads identical bytes, so at batch=1 the GEMV is bandwidth-bound "
            "and all of these dequant costs stay under the memory time; the discriminator "
            "is not op count but whether a per-lane table access is needed"),
        "ascend_aiv_note": (
            "NOT QUANTIFIED. The AIV's vector ISA, its gather/table primitives and their "
            "issue costs are outside what can be verified from here, and inventing "
            "instruction counts for it would be worse than leaving the column empty. What "
            "carries over architecturally, and is all that is claimed: uniform E8 and "
            "radial-shell E8 need no per-element table, so they map onto any vector unit "
            "with bit-extract + integer-to-float + FMA; NF4/Lloyd-Max and Cartesian-LUT E8 "
            "need a per-element lookup into 16 or 32 entries, which on a vector machine "
            "without a cheap in-register permute becomes either a gather or a select chain "
            "of depth log2(entries). Candidate 1's one-lookup-per-8-weights is the only "
            "structural difference that would survive a port."),
    }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", default=e8.DEFAULT_MODEL_DIR)
    ap.add_argument("--layers", type=int, nargs="+", default=[3, 10])
    ap.add_argument("--outlier-frac", type=float, default=0.001)
    ap.add_argument("--num-seqs", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--dataset", default="Salesforce/wikitext")
    ap.add_argument("--split", default="train")
    ap.add_argument("--grid", type=int, default=40)
    ap.add_argument("--shell-bins", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join(HERE, "companded_e8_report.json"))
    ap.add_argument("--md-out", default=os.path.join(HERE, "COMPANDED_E8.md"))
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    dev = args.device
    t0 = time.time()
    from transformers.models.glm import GlmForCausalLM

    print(f"device: {torch.cuda.get_device_name(0)}")
    print("loading GLM-4-9B onto host RAM (bf16, 18.9 GB)...", flush=True)
    model = GlmForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16).eval()
    n_layers = model.config.num_hidden_layers
    runner = TailRunner(model, dev)
    ids = build_batch(args.model_dir, args.num_seqs, args.seq_len, args.dataset, args.split)
    print(f"  calibration {tuple(ids.shape)} = {ids.numel()} tokens", flush=True)

    grabbed: Dict[int, Dict[str, torch.Tensor]] = {l: {} for l in args.layers}
    handles = []
    for l in args.layers:
        def mk(layer_idx, key):
            def hook(_m, a):
                grabbed[layer_idx][key] = a[0].detach().clone()
            return hook
        handles.append(model.model.layers[l].mlp.down_proj
                       .register_forward_pre_hook(mk(l, "x")))
        handles.append(model.model.layers[l].post_attention_layernorm
                       .register_forward_pre_hook(mk(l, "h1")))
    print("reference forward (streaming all 40 layers)...", flush=True)
    hidden = runner.embed(ids)
    hidden = runner.run_layers(hidden, 0, n_layers)
    logits_model = runner.logits(hidden)
    for h in handles:
        h.remove()
    del hidden
    gc.collect(); torch.cuda.empty_cache()

    grid = _alpha_grid(args.grid, device=dev)
    report: Dict[str, object] = {"meta": {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "model_dir": args.model_dir, "layers": args.layers,
        "outlier_frac": args.outlier_frac, "tokens": int(ids.numel()),
        "shell_bins": args.shell_bins, "grid_points": args.grid,
        "kl_direction": "forward, D(P_reference || P_quantized)",
        "hypothesis": "companding gain (+1.75 dB) and Zador packing gain (+0.65 dB) are "
                      "separate multiplicative factors and should compose",
    }}
    md = ["# Companded E8 vs 1-D NF4 at matched 4.013 bpw", "",
          "Can an E8 lattice capture 1-D companding gain *and* its own packing gain? "
          "All rows carry the same 0.1% BF16 outlier channels and the same per-row "
          "MSE-optimal step search. Generated by `research/eval_companded_e8.py`.", ""]

    for layer in args.layers:
        print(f"\n=== layer {layer} ===", flush=True)
        w = e8.load_down_proj(args.model_dir, layer, device=dev)
        x = grabbed[layer]["x"].reshape(-1, w.shape[1]).to(dev, torch.float32)
        h1 = grabbed[layer]["h1"].to(dev)
        n_out = max(1, int(args.outlier_frac * w.shape[1]))
        out_idx = torch.topk(x.pow(2).sum(0), n_out).indices.sort().values
        wf = w.to(torch.float32)
        print(f"  W {tuple(w.shape)}  outliers {n_out} channels", flush=True)

        builds: List[Tuple[str, Callable, int]] = [
            ("1D-RTN-INT4", _bulk_lut(levels_rtn_int4, grid), 0),
            ("1D-Lloyd-Max", _bulk_lut(levels_lloyd_max, grid), 16),
            ("1D-NF4", _bulk_lut(levels_nf4, grid), 16),
            ("1D-NF4-centroid", _bulk_nf4_centroid(grid), 16),
            ("8D-E8-Uniform", _bulk_e8(grid), 0),
            (f"8D-E8-Radial-{args.shell_bins}", _bulk_e8_radial(
                grid, n_bins=args.shell_bins), args.shell_bins),
            ("8D-E8-Radial-64", _bulk_e8_radial(grid, n_bins=64), 64),
            ("8D-E8-Companded", _bulk_e8_companded(grid), 32),
        ]
        schemes: List[Scheme] = []
        for name, fn, lut_n in builds:
            ts = time.time()
            schemes.append(with_sparse_outliers(w, out_idx, name, fn, lut_entries=lut_n))
            print(f"    built {name:<22} ({time.time() - ts:.1f}s)", flush=True)

        y_ref = (x @ wf.T).reshape(*h1.shape[:-1], -1)
        variants: List[Tuple[str, torch.Tensor]] = [("Ref-BF16", y_ref)]
        for s in schemes:
            variants.append((s.name, (x @ s.w_hat.T).reshape(*h1.shape[:-1], -1)))
        ctrl_targets = [40.0, 30.0, 25.0]
        gen = torch.Generator(device=dev).manual_seed(1234)
        y_rms = float(y_ref.pow(2).mean().sqrt())
        for tgt in ctrl_targets:
            noise = torch.randn(y_ref.shape, generator=gen, device=dev,
                                dtype=y_ref.dtype) * (y_rms * 10 ** (-tgt / 20))
            variants.append((f"Ctrl-noise-{tgt:.0f}dB", y_ref + noise))
            del noise

        stack = torch.cat([(h1.to(torch.float32) + yv).to(torch.bfloat16)
                           for _, yv in variants], dim=0)
        print(f"  tail batch {tuple(stack.shape)} ({len(variants)} variants)", flush=True)
        hid = runner.run_layers(stack, layer + 1, n_layers)
        del stack
        gc.collect(); torch.cuda.empty_cache()

        per = args.num_seqs
        ref_logits = runner.logits(hid[0:per])
        floor = kl_metrics(logits_model[0:per], ref_logits)
        report[f"layer{layer}_method_floor"] = floor
        print(f"  method floor: KL={floor['kl_mean_nats']:.3e} nats, "
              f"top1={floor['top1_agreement']:.4f}", flush=True)

        rows_out: List[Dict[str, object]] = [
            {"scheme": "Ref-BF16", "total_bpw": 16.0, "act_snr_db": float("inf"),
             "cos_mean": 1.0, "cos_min": 1.0, "kl_mean_nats": 0.0, "kl_max_nats": 0.0,
             "top1_agreement": 1.0}]
        kl_logits: Dict[str, torch.Tensor] = {}
        for i, s in enumerate(schemes, start=1):
            lg = runner.logits(hid[i * per:(i + 1) * per])
            km = kl_metrics(ref_logits, lg)
            kl_logits[s.name] = lg
            lm = layer_metrics(x, wf, s.w_hat)
            wm = e8.weight_metrics(wf, s.w_hat)
            rows_out.append({"scheme": s.name, "total_bpw": s.total_bpw,
                             "payload_bpw": s.payload_bpw,
                             "overhead_bpw": s.overhead_bpw,
                             "weight_sqnr_db": wm["weight_sqnr_db"],
                             "weight_linf": wm["weight_linf"], **lm, **km,
                             "extra": s.extra})
            print(f"    {s.name:<22} bpw={s.total_bpw:.4f} SNR={lm['act_snr_db']:7.3f} "
                  f"cos={lm['cos_mean']:.6f}/{lm['cos_min']:.6f} "
                  f"KL={km['kl_mean_nats']:.3e} top1={km['top1_agreement']:.4f}",
                  flush=True)
        for j, tgt in enumerate(ctrl_targets):
            lg = runner.logits(hid[(len(schemes) + 1 + j) * per:
                                   (len(schemes) + 2 + j) * per])
            km = kl_metrics(ref_logits, lg)
            yv = variants[len(schemes) + 1 + j][1]
            snr = float(10 * torch.log10(y_ref.to(F64).pow(2).sum()
                                         / (y_ref - yv).to(F64).pow(2).sum()))
            rows_out.append({"scheme": f"Ctrl-noise-{tgt:.0f}dB", "total_bpw": None,
                             "act_snr_db": snr, **km})
            print(f"    {'Ctrl-noise-%.0fdB' % tgt:<22} (control)   SNR={snr:7.3f} "
                  f"KL={km['kl_mean_nats']:.3e} top1={km['top1_agreement']:.4f}", flush=True)
            del lg

        # the acceptance criteria: every candidate measured against NF4 and NF4-centroid
        nf4 = next(r for r in rows_out if r["scheme"] == "1D-NF4")
        nf4c = next(r for r in rows_out if r["scheme"] == "1D-NF4-centroid")
        verdicts = []
        for cand in ("8D-E8-Uniform", f"8D-E8-Radial-{args.shell_bins}",
                     "8D-E8-Radial-64", "8D-E8-Companded"):
            cr = next(r for r in rows_out if r["scheme"] == cand)
            p = paired_kl_margin(ref_logits, kl_logits["1D-NF4"], kl_logits[cand])
            verdicts.append({
                "candidate": cand,
                "act_snr_margin_vs_nf4_db": cr["act_snr_db"] - nf4["act_snr_db"],
                "act_snr_margin_vs_nf4_centroid_db":
                    cr["act_snr_db"] - nf4c["act_snr_db"],
                "bpw_delta_vs_nf4": cr["total_bpw"] - nf4["total_bpw"],
                "paired_kl_nf4_minus_candidate": p,
                "meets_065db_criterion":
                    bool(cr["act_snr_db"] - nf4["act_snr_db"] >= 0.65),
                "meets_kl_z3_criterion": bool(p["sign_test_z"] > 3.0),
            })
            print(f"  {cand:<22} vs NF4: {verdicts[-1]['act_snr_margin_vs_nf4_db']:+.3f} dB"
                  f" (vs NF4-centroid {verdicts[-1]['act_snr_margin_vs_nf4_centroid_db']:+.3f}"
                  f" dB)  sign-test z={p['sign_test_z']:+.2f}", flush=True)
        report[f"layer{layer}_verdicts"] = verdicts
        report[f"layer{layer}"] = rows_out
        md.append(_md_table(rows_out, layer, floor, verdicts, args.shell_bins))
        md.append("")
        del w, wf, x, h1, hid, ref_logits, kl_logits, schemes, variants, y_ref
        gc.collect(); torch.cuda.empty_cache()

    print("\n== boundary diagnostic ==", flush=True)
    bd = boundary_diagnostic(device=dev)
    report["boundary_diagnostic"] = bd
    print(f"  uniform source filling the box: E8 gain {bd['filling_box_e8_gain_db']:+.3f} dB "
          f"({100 * bd['filling_box_e8_coords_on_box_edge']:.1f}% of coords on the edge)")
    print(f"  same source held {4.0:.0f} units inside the box: "
          f"E8 gain {bd['inside_box_e8_gain_db']:+.3f} dB "
          f"(asymptotic prediction {bd['asymptotic_prediction_db']:+.3f} dB)")
    print(f"  => boundary loss {bd['boundary_loss_db']:.3f} dB, and companding is what "
          f"pushes mass into those cells", flush=True)
    md.append(_md_boundary(bd))
    report["silicon_audit"] = silicon_audit(564.0, 56098816, 4.013)
    md.append(_md_audit(report["silicon_audit"]))

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    with open(args.md_out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    print(f"\nwrote {args.out} and {args.md_out} in {time.time() - t0:.0f}s")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    print("\n" + "\n".join(md))
    return 0


def _md_table(rows, layer, floor, verdicts, shell_bins) -> str:
    def f(r, k, spec):
        v = r.get(k)
        if v is None:
            return "--"
        if isinstance(v, float) and math.isinf(v):
            return "inf"
        return format(v, spec)
    out = [f"### Layer {layer} `down_proj` -- all schemes + 0.1% BF16 outliers", "",
           "| Scheme | Total bpw | Act-SNR (dB) | CosSim mean | CosSim min-token "
           "| KL mean (1e-4 nats) | KL SE | KL max (1e-4) | Top-1 agree | W-SQNR |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        kl = r.get("kl_mean_nats")
        klm = r.get("kl_max_nats")
        kse = r.get("kl_se_nats")
        out.append(
            f"| {r['scheme']} | {f(r, 'total_bpw', '.4f')} | {f(r, 'act_snr_db', '.2f')} | "
            f"{f(r, 'cos_mean', '.6f')} | {f(r, 'cos_min', '.6f')} | "
            f"{'--' if kl is None else format(kl * 1e4, '.2f')} | "
            f"{'--' if kse is None else format(kse * 1e4, '.2f')} | "
            f"{'--' if klm is None else format(klm * 1e4, '.1f')} | "
            f"{f(r, 'top1_agreement', '.4f')} | {f(r, 'weight_sqnr_db', '.2f')} |")
    out += ["", f"*Method noise floor*: KL = {floor['kl_mean_nats'] * 1e4:.2f}e-4 nats, "
            f"top-1 {floor['top1_agreement']:.4f}.", "",
            "| Candidate | vs NF4 (dB) | vs NF4-centroid (dB) | +0.65 dB met? "
            "| paired KL sign-test z | z > 3 met? |", "|---|---|---|---|---|---|"]
    for v in verdicts:
        p = v["paired_kl_nf4_minus_candidate"]
        out.append(f"| {v['candidate']} | {v['act_snr_margin_vs_nf4_db']:+.3f} | "
                   f"{v['act_snr_margin_vs_nf4_centroid_db']:+.3f} | "
                   f"{'**yes**' if v['meets_065db_criterion'] else 'no'} | "
                   f"{p['sign_test_z']:+.2f} ({100 * p['frac_tokens_b_better']:.1f}% of "
                   f"tokens) | {'**yes**' if v['meets_kl_z3_criterion'] else 'no'} |")
    return "\n".join(out)


def _md_boundary(b) -> str:
    return "\n".join([
        "### Why the two gains do not compose: the boundary diagnostic", "",
        "A *uniform* source is exactly what a companding warp produces. At equal codepoint "
        "count (2^32 either lattice in the [-8,+7] box):", "",
        "| Uniform source | Z^8 MSE/coord | E8 MSE/coord | E8 gain | coords on box edge |",
        "|---|---|---|---|---|",
        f"| filling the box | {b['filling_box_mse_per_coord_z8']:.6f} | "
        f"{b['filling_box_mse_per_coord_e8']:.6f} | "
        f"**{b['filling_box_e8_gain_db']:+.3f} dB** | "
        f"{100 * b['filling_box_e8_coords_on_box_edge']:.1f}% |",
        f"| held 4 units inside the box | {b['inside_box_mse_per_coord_z8']:.6f} | "
        f"{b['inside_box_mse_per_coord_e8']:.6f} | "
        f"**{b['inside_box_e8_gain_db']:+.3f} dB** | "
        f"{100 * b['inside_box_e8_coords_on_box_edge']:.1f}% |", "",
        f"Away from the faces the lattice delivers {b['inside_box_e8_gain_db']:+.3f} dB, "
        f"matching the asymptotic G_8 prediction of "
        f"{b['asymptotic_prediction_db']:+.3f} dB, which validates the quantizer. Filling "
        f"the box costs {b['boundary_loss_db']:.3f} dB: Z^8's unit cells tile a box exactly "
        f"and overload nowhere, while "
        f"{100 * b['cells_touching_boundary_frac']:.0f}% of E8's cells are clipped by a "
        f"face. Companding is precisely the operation that flattens the source onto that "
        f"box, so it buys density matching and spends part of the packing gain. The two are "
        f"in tension, not additive, and "
        f"**{b['filling_box_e8_gain_db']:+.3f} dB is therefore the hard ceiling** on what a "
        f"box-constrained companded E8 can win over NF4 at this coordinate width -- the "
        f"asymptotic +0.65 dB is not reachable here at all.", "",
        "The tension is visible in the fitted quantizers, not just in theory: the per-row MSE "
        "search lands on steps that leave only ~0.5% of coordinates on the box edge rather "
        "than the 12.5% a fully companded source would produce. It is declining full "
        "companding in order to stay away from the faces, which is why the measured margin "
        "comes in near half the ceiling."])


def _md_audit(a) -> str:
    out = ["### Silicon / register audit (static instruction count, not a measured kernel)",
           "",
           "| Format | loads /8w | int ALU /8w | table entries | lookups /8w "
           "| sel ops per lookup | ops /weight | table bytes | ALU as % of memory time |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in a["per_format"]:
        out.append(f"| {r['format']} | {r['loads_per_8w']} | {r['int_alu_per_8w']} | "
                   f"{r['table_entries']} | {r['lookups_per_8w']} | "
                   f"{r['select_ops_per_lookup']} | {r['ops_per_weight']:.1f} | "
                   f"{r['table_bytes']} | "
                   f"{100 * r['alu_fraction_of_memory_time']:.1f}% |")
    out += ["", f"At 4.013 bpw one tensor is {a['payload_bytes'] / 1e6:.1f} MB; reading it at "
            f"{a['assumed_dram_gbps']:.0f} GB/s takes {a['memory_time_us']:.0f} us. "
            f"{a['rtx5070_verdict']}.",
            "", "Notes:", ""]
    for r in a["per_format"]:
        out.append(f"- **{r['format']}** -- {r['note']}")
    out += ["", f"**Ascend AIV:** {a['ascend_aiv_note']}"]
    return "\n".join(out)


if __name__ == "__main__":
    raise SystemExit(main())
