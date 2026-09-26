"""Module C -- Quantized Johnson-Lindenstrauss (QJL) 1-bit sketching of the E8 residual.

The idea under test: spend 3.5 bpw on E8, then spend a further 0.5 bpw on a 1-bit
sketch of the residual E = W - What, and recover part of X@E^T online. If it works,
it is a cheaper way to buy accuracy than making the E8 grid denser.

THE ESTIMATOR (and why it is not the formula in the spec)
--------------------------------------------------------
The spec writes ``Yhat = X What + gamma * (X S^T) (*) B`` with ``(*)`` elementwise.
That does not typecheck: ``X S^T`` is [tokens, m] and ``B`` is [out_features, m], so
there is no elementwise product, and no contraction is named. What is implemented
here is the standard QJL estimator those symbols are reaching for, which is the one
with a provable guarantee. For a random projection row ``s ~ N(0, I_D)``:

    E[ <s,x> * sign(<s,e>) ] = sqrt(2/pi) * <x, e/||e||>

so averaging over m rows and rescaling gives an UNBIASED estimate of the inner
product the quantizer threw away:

    <x,e> ~= sqrt(pi/2) * (||e|| / m) * < S x , sign(S e) >

Per (output row r, block b) we therefore store ``sign(S e_rb)`` -- m bits -- and the
residual norm ``||e_rb||``. ``S x_b`` is computed once per token and shared across all
4096 output rows, which is what makes the scheme cheap; that sharing is the whole
reason QJL is interesting.

WHAT THE THEORY PREDICTS BEFORE WE MEASURE
-----------------------------------------
The estimator is unbiased but noisy. Its standard deviation is

    std(T) = ||x|| * ||e|| * sqrt(pi / (2m))

while the quantity it is estimating, for a residual with no preferred direction, has
rms ``||x|| * ||e|| / sqrt(D)``. The ratio of added noise to removed signal is

    noise / signal = sqrt(pi * D / (2m))

which at the spec's ``m = D/2`` is ``sqrt(pi) = 1.77``, INDEPENDENT of D. So at
m = D/2 the sketch injects 1.77x more error than it removes, and Act-SNR should get
about 20*log10(1.77) = 4.97 dB WORSE, not better. Break-even needs
``m >= pi*D/2 = 1.571*D`` -- more sketch dimensions than the block has weights,
i.e. above 1.57 bpw. ``qjl_theory`` returns these numbers and
``sweep_m`` tests them against measurement; the isotropy of the weight residual
(established by the prior analysis) is exactly the property that makes this
prediction hold.

NORM BOOKKEEPING
----------------
The spec budgets 0.5 bpw = m/D = 32/64 bits, counting the sign bits ONLY. The per-
block norm is real storage too: one FP16 per 64 weights is a further 0.25 bpw, half
the sketch budget again. Because an E8 residual is near-homoscedastic inside a row
(the Voronoi cell scales with the row's single step size), a per-ROW norm costs
0.001 bpw and loses almost nothing -- ``norm_mode`` selects which, and both are
reported so the 0.5 bpw claim can be checked rather than assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

SCALE_BITS = 16


class UnsupportedProjection(ValueError):
    """Raised when a projection kind cannot supply the requested m."""


# --------------------------------------------------------------------------- #
# projection
# --------------------------------------------------------------------------- #

def make_projection(m: int, block: int, kind: str = "rademacher",
                    seed: int = 0, device: str = "cuda") -> torch.Tensor:
    """An [m, block] projection whose rows have E||s||^2 = block, like N(0, I_block).

    ``rademacher`` -- iid +-1. ``gaussian`` -- iid N(0,1), the distribution the
    estimator's guarantee is stated for. ``fwht`` -- m rows of H @ diag(d) for a
    64-point Hadamard H and random signs d, i.e. the structured variant that a kernel
    would apply in O(D log D) with no matrix at all. All three share the same scaling
    so gamma is identical across them.

    ``fwht`` beats the iid variants at equal m because its rows are *orthogonal*, so
    the m sign measurements are non-redundant -- but for the same reason it saturates
    at m = block: there are only ``block`` orthogonal rows available, and asking for
    more duplicates them and adds no information. Any m > block is therefore wasted
    storage in this mode, which the sweep shows as a flat curve past that point.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    if kind == "gaussian":
        s = torch.randn(m, block, generator=g)
    elif kind == "rademacher":
        s = (torch.randint(0, 2, (m, block), generator=g).float() * 2.0 - 1.0)
    elif kind == "fwht":
        if block & (block - 1):
            raise ValueError("fwht needs a power-of-two block")
        h = torch.ones(1, 1)
        while h.shape[0] < block:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        d = torch.randint(0, 2, (block,), generator=g).float() * 2.0 - 1.0
        if m > block:
            raise UnsupportedProjection(
                f"fwht has only {block} orthogonal rows; m={m} would have to repeat "
                f"them, which double-weights those measurements and makes the "
                f"estimator worse than at m={block} rather than merely no better")
        s = h[torch.randperm(block, generator=g)[:m]] * d
    else:
        raise ValueError(f"unknown projection {kind!r}")
    return s.to(device)


# --------------------------------------------------------------------------- #
# encode / apply
# --------------------------------------------------------------------------- #

@dataclass
class QJLSketch:
    signs: torch.Tensor          # [rows, nblocks, m] in {-1,+1}, stored as int8
    norms: torch.Tensor          # [rows, nblocks] fp32 (already broadcast if row mode)
    s: torch.Tensor              # [m, block]
    m: int
    block: int
    norm_mode: str
    sign_bpw: float
    norm_bpw: float

    @property
    def total_bpw(self) -> float:
        return self.sign_bpw + self.norm_bpw


def qjl_encode(residual: torch.Tensor, m: int, block: int = 64,
               kind: str = "rademacher", seed: int = 0,
               norm_mode: str = "row") -> QJLSketch:
    """Sketch E = W - What into m sign bits per ``block`` weights."""
    rows, cols = residual.shape
    assert cols % block == 0, "input dim must be a multiple of the QJL block"
    nb = cols // block
    e = residual.to(torch.float32).reshape(rows, nb, block)
    s = make_projection(m, block, kind, seed, str(residual.device))
    proj = torch.einsum("rbd,md->rbm", e, s)
    signs = torch.where(proj >= 0, 1, -1).to(torch.int8)

    block_norms = e.pow(2).sum(-1).sqrt()                      # [rows, nb]
    if norm_mode == "block":
        norms = block_norms
        norm_bpw = SCALE_BITS / block
    elif norm_mode == "row":
        # one norm per row, applied to every block in it
        row_norm = block_norms.pow(2).mean(dim=1, keepdim=True).sqrt()
        norms = row_norm.expand_as(block_norms).contiguous()
        norm_bpw = SCALE_BITS / cols
    else:
        raise ValueError(f"unknown norm_mode {norm_mode!r}")
    return QJLSketch(signs=signs, norms=norms, s=s, m=m, block=block,
                     norm_mode=norm_mode, sign_bpw=m / block, norm_bpw=norm_bpw)


def qjl_correction(x: torch.Tensor, sk: QJLSketch, chunk: int = 128) -> torch.Tensor:
    """The additive correction to Y, shape [tokens, rows].

    gamma = sqrt(pi/2)/m folds in the sign-correlation constant; the per-block norm
    multiplies the sign matrix because it varies with (row, block) and so cannot be
    folded into the token-side projection.
    """
    rows, nb = sk.norms.shape
    gamma = math.sqrt(math.pi / 2.0) / sk.m
    scaled = sk.signs.to(torch.float32) * sk.norms.unsqueeze(-1)       # [rows, nb, m]
    out = torch.empty(x.shape[0], rows, device=x.device, dtype=torch.float32)
    for i in range(0, x.shape[0], chunk):
        xb = x[i:i + chunk].to(torch.float32).reshape(-1, nb, sk.block)
        sx = torch.einsum("nbd,md->nbm", xb, sk.s)                     # shared per token
        out[i:i + chunk] = gamma * torch.einsum("nbm,rbm->nr", sx, scaled)
    return out


def qjl_theory(m: int, block: int) -> Dict[str, float]:
    """Closed-form prediction for what the sketch does to the error, before measuring."""
    ratio = math.sqrt(math.pi * block / (2.0 * m))
    return {
        "noise_over_signal": ratio,
        "predicted_act_snr_delta_db": -20.0 * math.log10(ratio),
        "break_even_m": math.pi * block / 2.0,
        "break_even_bpw": (math.pi * block / 2.0) / block,
    }


# --------------------------------------------------------------------------- #
# the head-to-head the study is for
# --------------------------------------------------------------------------- #

def sweep_m(x: torch.Tensor, w: torch.Tensor, w_hat: torch.Tensor,
            m_values: List[int], block: int = 64, kind: str = "rademacher",
            seed: int = 0, norm_mode: str = "row",
            act_metrics_fn=None) -> List[dict]:
    """Act-SNR of ``What + QJL(m)`` across m, next to the closed-form prediction.

    Sweeping m is what turns the theory section above from an assertion into a test:
    the predicted crossing at m = 1.571*block either shows up in the measured curve
    or it does not.
    """
    assert act_metrics_fn is not None, "pass e8_lattice_engine.act_metrics"
    residual = w.to(torch.float32) - w_hat.to(torch.float32)
    base = act_metrics_fn(x, w, w_hat)
    rows = []
    for m in m_values:
        try:
            sk = qjl_encode(residual, m, block, kind, seed, norm_mode)
        except UnsupportedProjection as exc:
            rows.append({"m": m, "block": block, "projection": kind,
                         "skipped": str(exc)})
            continue
        corr = qjl_correction(x, sk)
        y_hat = _assemble(x, w_hat, corr)
        met = act_metrics_fn(x, w, None, y_hat_override=y_hat)
        th = qjl_theory(m, block)
        rows.append({
            "m": m, "block": block, "projection": kind, "norm_mode": norm_mode,
            "sign_bpw": sk.sign_bpw, "norm_bpw": sk.norm_bpw,
            "qjl_total_bpw": sk.total_bpw,
            "act_snr_db": met["act_snr_db"],
            "act_snr_delta_db": met["act_snr_db"] - base["act_snr_db"],
            "predicted_act_snr_delta_db": th["predicted_act_snr_delta_db"],
            "act_mean_bias": met["act_mean_bias"],
            "act_mean_bias_se": met["act_mean_bias_se"],
            "act_channel_bias_rms": met["act_channel_bias_rms"],
            "act_linf": met["act_linf"],
        })
        del sk, corr, y_hat
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def _assemble(x: torch.Tensor, w_hat: torch.Tensor, corr: torch.Tensor,
              chunk: int = 128) -> torch.Tensor:
    """Y = X What^T + correction, materialized so act_metrics can consume it."""
    wht = w_hat.to(torch.float32).T.contiguous()
    out = torch.empty(x.shape[0], w_hat.shape[0], device=x.device, dtype=torch.float32)
    for i in range(0, x.shape[0], chunk):
        out[i:i + chunk] = x[i:i + chunk].to(torch.float32) @ wht + corr[i:i + chunk]
    return out


def residual_homoscedasticity(residual: torch.Tensor, block: int = 64) -> Dict[str, float]:
    """How much per-block residual norms vary inside a row.

    This is the measurement that justifies (or refuses) the cheap ``norm_mode='row'``:
    if the spread is small, one FP16 per row replaces one per block and the sketch
    really does cost 0.5 bpw instead of 0.75.
    """
    rows, cols = residual.shape
    nb = cols // block
    n = residual.to(torch.float32).reshape(rows, nb, block).pow(2).sum(-1).sqrt()
    rel = n.std(dim=1) / n.mean(dim=1).clamp(min=1e-20)
    return {
        "block_norm_rel_std_mean": float(rel.mean()),
        "block_norm_rel_std_max": float(rel.max()),
    }
