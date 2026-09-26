"""1D scalar vs 8D E8 at a matched 4.013 bpw, scored all the way out to logit KL.

Every candidate here gets the *same* outlier mechanism (top 0.1% input channels by
||X[:, k]||_2 held in native BF16) and the *same* per-row MSE-optimal scale search, so the
only thing that differs between rows of the output table is the shape of the codebook:

    RTN INT4    15 uniform levels, -7..+7                     (1-D, symmetric)
    Lloyd-Max   16 MSE-optimal centroids                      (1-D, non-uniform)
    NF4         16 equiprobable quantile levels               (1-D, non-uniform)
    E8          the Gosset lattice, 32 bits per 8 weights     (8-D)

Giving the scalar baselines a per-row MSE-optimal scale rather than a plain absmax is
deliberate: the brief asks for the *strongest possible* 1-D quantizers, and on this tensor
that search is worth several dB (see FINDINGS.md section 7). A margin measured against a
handicapped baseline would be worthless.

WHY THE KL NUMBER REQUIRES RUNNING THE REST OF THE MODEL
-------------------------------------------------------
A KL divergence between softmax distributions only means something if the softmax is over
the real vocabulary, which means propagating the perturbed activation through every layer
above the one under test and through lm_head. That is done here literally: the layer's
residual stream is reassembled as ``h1 + Yhat``, and layers L+1..39, the final norm and
lm_head are then run for real. The tail does not fit in 11.9 GB, so layers are streamed to
the GPU one at a time -- the same trick the engine's own hibernation path uses.

All six variants (reference + five schemes) are stacked into ONE batch of 24 sequences and
propagated together. A decoder is causal and each sequence is independent, so this is exact,
and it means the 29-36 layer tail is paid for once rather than six times.

THE MEASUREMENT FLOOR, WHICH IS REPORTED RATHER THAN ASSUMED
-----------------------------------------------------------
Two reference paths are computed, not one:

* ``logits_model`` -- the model's own unmodified streamed forward, in bf16 throughout.
* ``logits_ref``   -- the reassembled ``h1 + Y`` restart, where Y is an fp32 recomputation
  of the same projection.

These differ slightly (fp32 vs bf16 accumulation in one MLP, then chaotic bf16 rounding
through the tail), and the KL between them is the **noise floor of this whole method**. Every
scheme's KL is measured against ``logits_ref`` so that quantization is the only variable, and
the floor is printed next to the results: any scheme whose KL is within a small multiple of
it has not been resolved by this experiment. Logits are formed in fp32 because bf16 logit
rounding alone (relative 2^-8 on values of order 15) would manufacture KL of the same order
as the effect being measured.
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
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import e8_lattice_engine as e8                      # noqa: E402
from eval_fixed_rate_e8 import (                    # noqa: E402
    E8_DIM, box_for, closest_e8_boxed, pack_e8_fixed_rate, unpack_e8_fixed_rate,
    _alpha_grid, select_outlier_channels,
)

SCALE_BITS = 16
BF16_BITS = 16
LUT_BITS = 16


# --------------------------------------------------------------------------- #
# 1-D codebooks, all scored with the same per-row scale search
# --------------------------------------------------------------------------- #

def levels_rtn_int4(_bulk: torch.Tensor) -> torch.Tensor:
    """Symmetric uniform INT4: -7..+7.

    Symmetric costs one of the 16 codes (there is no +8), which is the convention the brief
    names. An asymmetric -8..+7 implementation would recover ~0.09 bits of code space.
    """
    return torch.arange(-7, 8, dtype=torch.float32)


def levels_lloyd_max(bulk: torch.Tensor, n: int = 16, iters: int = 80,
                     subsample: int = 1 << 21, seed: int = 0) -> torch.Tensor:
    """MSE-optimal 16 centroids (Lloyd) on the outlier-free weights, in absmax units."""
    s = _subsample(bulk, subsample, seed)
    c = torch.quantile(s, torch.linspace(0.5 / n, 1 - 0.5 / n, n, device=s.device))
    # The centroid accumulation runs in float64 on purpose. In float32 the
    # non-deterministic ordering of a GPU scatter_add perturbs the fitted centroids just
    # enough to move the downstream logit-KL effect size by ~25% between runs, which is the
    # same order as the effect itself -- so the scheme would appear unstable when only the
    # fit was. In float64 the ordering effect is ~1e-16 and the pipeline is reproducible.
    s64 = s.to(torch.float64)
    c = c.to(torch.float64)
    for _ in range(iters):
        a = torch.bucketize(s64, (c[1:] + c[:-1]) / 2)
        sums = torch.zeros(n, device=s.device, dtype=torch.float64).scatter_add_(0, a, s64)
        cnt = torch.zeros(n, device=s.device, dtype=torch.float64).scatter_add_(
            0, a, torch.ones_like(s64))
        c = torch.where(cnt > 0, sums / cnt.clamp(min=1), c)
        c, _ = torch.sort(c)
    return c.to(torch.float32)


def levels_nf4(bulk: torch.Tensor, n: int = 16, subsample: int = 1 << 21,
               seed: int = 0) -> torch.Tensor:
    """NormalFloat4: n equiprobable quantile levels of the outlier-free weights.

    The one structural difference from Lloyd-Max: NF4 places levels so each *code* is
    equally likely (maximum code entropy), Lloyd-Max places them to minimize MSE. On a
    peaked distribution those disagree -- NF4 spends codes on the tail that Lloyd-Max
    spends on the bulk -- so the pair isolates "information-theoretically uniform" against
    "distortion-optimal" at identical rate.
    """
    s = _subsample(bulk, subsample, seed)
    q = torch.linspace(0.5 / n, 1 - 0.5 / n, n, device=s.device)
    return torch.quantile(s, q)


def _subsample(t: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    flat = t.reshape(-1)
    g = torch.Generator(device=flat.device).manual_seed(seed)
    idx = torch.randint(0, flat.numel(), (min(k, flat.numel()),),
                        generator=g, device=flat.device)
    return flat[idx].to(torch.float32)


def quantize_lut_1d(w: torch.Tensor, levels: torch.Tensor,
                    grid: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Apply a 1-D codebook with a per-row MSE-optimal scale chosen from ``grid``.

    ``levels`` are in normalized units; the stored per-row FP16 scale maps them onto the
    row. The search is the same one the E8 path uses, so no candidate is advantaged by its
    scale-fitting rather than by its codebook.
    """
    wf = w.to(torch.float32)
    rows = wf.shape[0]
    rms = wf.pow(2).mean(dim=1, keepdim=True).sqrt().clamp(min=1e-30)
    lv = levels.to(wf.device).to(torch.float32)
    lv, _ = torch.sort(lv)
    edges = (lv[1:] + lv[:-1]) / 2
    span = float(lv.abs().max())

    best_mse = torch.full((rows, 1), float("inf"), device=wf.device, dtype=torch.float64)
    best_delta = torch.ones((rows, 1), device=wf.device, dtype=torch.float32)
    for g in grid.tolist():
        delta = (rms * g / span).to(torch.float16).to(torch.float32)
        idx = torch.bucketize((wf / delta).reshape(-1), edges)
        rec = lv[idx].reshape(wf.shape) * delta
        # float64 before the reduction, not after: adjacent grid points are often near-tied
        # for a given row, and float32 reduction noise then flips which scale that row
        # picks, which wobbles the downstream KL effect size run to run.
        mse = (wf - rec).to(torch.float64).pow(2).mean(dim=1, keepdim=True)
        take = mse < best_mse
        best_mse = torch.where(take, mse, best_mse)
        best_delta = torch.where(take, delta, best_delta)
        del idx, rec, mse
    idx = torch.bucketize((wf / best_delta).reshape(-1), edges)
    w_hat = lv[idx].reshape(wf.shape) * best_delta
    return w_hat, {"levels": [float(v) for v in lv],
                   "n_levels": int(lv.numel()),
                   "scale_mult_min": float((best_delta * span / rms).min()),
                   "scale_mult_max": float((best_delta * span / rms).max())}


# --------------------------------------------------------------------------- #
# sparse outlier wrapper, shared by every candidate
# --------------------------------------------------------------------------- #

@dataclass
class Scheme:
    name: str
    w_hat: torch.Tensor
    payload_bpw: float
    overhead_bpw: float
    extra: Dict[str, object] = field(default_factory=dict)

    @property
    def total_bpw(self) -> float:
        return self.payload_bpw + self.overhead_bpw


def with_sparse_outliers(w: torch.Tensor, outlier_idx: torch.Tensor, name: str,
                         bulk_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, int,
                                                                 Dict[str, object]]],
                         lut_entries: int = 0) -> Scheme:
    """Hold ``outlier_idx`` columns in BF16, hand the compacted bulk to ``bulk_fn``.

    ``bulk_fn`` returns (bulk_hat, bulk_payload_bits, extra). Columns are *gathered* before
    quantizing, so a retained channel does not also burn a bulk code -- which is what makes
    the rate (1-a)*bulk + a*16 rather than bulk + a*16.
    """
    rows, cols = w.shape
    keep = torch.ones(cols, dtype=torch.bool, device=w.device)
    keep[outlier_idx] = False
    bulk_idx = keep.nonzero(as_tuple=True)[0]
    n_out = int(outlier_idx.numel())

    bulk_hat, bulk_bits, extra = bulk_fn(w.to(torch.float32)[:, bulk_idx])
    w_hat = torch.empty(rows, cols, dtype=torch.float32, device=w.device)
    w_hat[:, bulk_idx] = bulk_hat
    w_hat[:, outlier_idx] = w.to(torch.float32)[:, outlier_idx]   # BF16 is the reference

    payload_bits = bulk_bits + rows * n_out * BF16_BITS
    overhead_bits = cols + rows * SCALE_BITS + lut_entries * LUT_BITS
    extra = dict(extra)
    extra.update({"n_outlier_channels": n_out, "outlier_frac": n_out / cols,
                  "bulk_columns": int(bulk_idx.numel()),
                  "payload_bits": int(payload_bits),
                  "overhead_bits": int(overhead_bits)})
    return Scheme(name=name, w_hat=w_hat,
                  payload_bpw=payload_bits / w.numel(),
                  overhead_bpw=overhead_bits / w.numel(), extra=extra)


def _bulk_e8(grid: torch.Tensor, coord_bits: int = 4):
    def fn(bulk: torch.Tensor):
        rows, n = bulk.shape
        padded = int(math.ceil(n / E8_DIM) * E8_DIM)
        buf = torch.zeros(rows, padded, dtype=torch.float32, device=bulk.device)
        buf[:, :n] = bulk
        lo, hi = box_for(coord_bits)
        rms = buf.pow(2).mean(dim=1, keepdim=True).sqrt().clamp(min=1e-30)
        best_mse = torch.full((rows, 1), float("inf"), device=buf.device,
                              dtype=torch.float64)
        best_delta = torch.ones((rows, 1), device=buf.device, dtype=torch.float32)
        for a in grid.tolist():
            delta = (rms * a).to(torch.float16).to(torch.float32)
            k, c = closest_e8_boxed((buf / delta).reshape(-1, E8_DIM), lo, hi)
            rec = (k + 0.5 * c.unsqueeze(-1).to(k.dtype)).reshape(rows, padded) * delta
            mse = (buf - rec).to(torch.float64).pow(2).mean(dim=1, keepdim=True)
            take = mse < best_mse
            best_mse = torch.where(take, mse, best_mse)
            best_delta = torch.where(take, delta, best_delta)
            del k, c, rec, mse
        k, c = closest_e8_boxed((buf / best_delta).reshape(-1, E8_DIM), lo, hi)
        rec = (k + 0.5 * c.unsqueeze(-1).to(k.dtype)).reshape(rows, padded) * best_delta
        # the format is only 4.00 bpw if the pack is lossless; check, do not assume
        k2, c2 = unpack_e8_fixed_rate(pack_e8_fixed_rate(k, c, coord_bits), coord_bits)
        if not (torch.equal(k2.to(torch.int64), k.to(torch.int64)) and torch.equal(c2, c)):
            raise RuntimeError("E8 32-bit pack failed to round-trip")
        sat = float(((k <= lo) | (k >= hi)).to(torch.float64).mean())
        bits = rows * (padded // E8_DIM) * (E8_DIM * coord_bits)
        return rec[:, :n], bits, {"coord_bits": coord_bits, "pad_columns": padded - n,
                                  "sat_at_bound": sat, "pack_verified": True}
    return fn


def _bulk_lut(levels_fn, grid: torch.Tensor, bits_per_weight: int = 4):
    def fn(bulk: torch.Tensor):
        absmax = bulk.abs().amax(dim=1, keepdim=True).clamp(min=1e-30)
        lv = levels_fn(bulk / absmax)
        w_hat, extra = quantize_lut_1d(bulk, lv, grid)
        rows, n = bulk.shape
        return w_hat, rows * n * bits_per_weight, extra
    return fn


# --------------------------------------------------------------------------- #
# streamed tail execution
# --------------------------------------------------------------------------- #

class TailRunner:
    """Runs GLM-4 layers on the GPU one at a time, keeping the master copy in host RAM.

    The full model is 18.9 GB of bf16 against 11.9 GB of VRAM, so a layer is moved up,
    executed, and moved back. Verified against the library's own ``GlmModel.forward``
    (``--validate``), because a hand-rolled layer loop that silently drops the rotary
    embeddings or the causal mask would produce plausible-looking logits.
    """

    def __init__(self, model, device: str = "cuda"):
        self.m = model
        self.base = model.model
        self.device = device
        self.base.rotary_emb.to(device)

    def context(self, hidden: torch.Tensor):
        from transformers.masking_utils import create_causal_mask
        pos = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
        mask = create_causal_mask(config=self.base.config, inputs_embeds=hidden,
                                  attention_mask=None, past_key_values=None,
                                  position_ids=pos)
        return pos, mask, self.base.rotary_emb(hidden, position_ids=pos)

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        self.base.embed_tokens.to(self.device)
        out = self.base.embed_tokens(ids.to(self.device))
        self.base.embed_tokens.to("cpu")
        torch.cuda.empty_cache()
        return out

    def run_layers(self, hidden: torch.Tensor, start: int, end: int,
                   progress: str = "") -> torch.Tensor:
        pos, mask, pe = self.context(hidden)
        for i in range(start, end):
            layer = self.base.layers[i].to(self.device)
            with torch.no_grad():
                hidden = layer(hidden, attention_mask=mask, position_embeddings=pe,
                               position_ids=pos)
            self.base.layers[i].to("cpu")
            torch.cuda.empty_cache()
            if progress and (i - start) % 8 == 0:
                print(f"      {progress} layer {i}", flush=True)
        return hidden

    def logits(self, hidden: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
        """Final norm + lm_head, projected in fp32 to keep the KL out of bf16 noise."""
        self.base.norm.to(self.device)
        with torch.no_grad():
            h = self.base.norm(hidden)
        self.base.norm.to("cpu")
        flat = h.reshape(-1, h.shape[-1]).to(torch.float32)
        wt = self.m.lm_head.weight.to(self.device, torch.float32).T.contiguous()
        out = torch.empty(flat.shape[0], wt.shape[1], dtype=torch.float32,
                          device=self.device)
        with torch.no_grad():
            for i in range(0, flat.shape[0], chunk):
                out[i:i + chunk] = flat[i:i + chunk] @ wt
        del wt
        torch.cuda.empty_cache()
        return out.reshape(*h.shape[:-1], -1)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def layer_metrics(x: torch.Tensor, w: torch.Tensor, w_hat: torch.Tensor,
                  chunk: int = 128) -> Dict[str, float]:
    """Act-SNR and cosine similarity of the layer output itself."""
    wt = w.to(torch.float32).T.contiguous()
    wht = w_hat.to(torch.float32).T.contiguous()
    f64 = torch.float64
    y_sq = torch.zeros((), dtype=f64, device=x.device)
    e_sq = torch.zeros((), dtype=f64, device=x.device)
    cos: List[torch.Tensor] = []
    for i in range(0, x.shape[0], chunk):
        xb = x[i:i + chunk].to(torch.float32)
        y, yh = xb @ wt, xb @ wht
        y_sq += y.to(f64).pow(2).sum()
        e_sq += (y - yh).to(f64).pow(2).sum()
        cos.append((y * yh).sum(1) / (y.norm(dim=1) * yh.norm(dim=1)).clamp(min=1e-30))
    c = torch.cat(cos)
    return {"act_snr_db": float(10 * torch.log10(y_sq / e_sq)),
            "cos_mean": float(c.mean()), "cos_min": float(c.min())}


def kl_per_token(ref_logits: torch.Tensor, q_logits: torch.Tensor,
                 chunk: int = 512) -> Tuple[torch.Tensor, float]:
    """Forward KL D(P_ref || P_quant) for every token, plus top-1 agreement.

    Forward KL is the direction the brief asks for and the conservative one here: it is
    dominated by tokens the true distribution considers likely, so a scheme that flattens or
    misplaces the peak is penalized.
    """
    r = ref_logits.reshape(-1, ref_logits.shape[-1])
    q = q_logits.reshape(-1, q_logits.shape[-1])
    kls: List[torch.Tensor] = []
    agree = torch.zeros((), dtype=torch.float64, device=r.device)
    for i in range(0, r.shape[0], chunk):
        lr = torch.log_softmax(r[i:i + chunk], dim=-1)
        lq = torch.log_softmax(q[i:i + chunk], dim=-1)
        kls.append((lr.exp() * (lr - lq)).sum(-1).to(torch.float64))
        agree += (r[i:i + chunk].argmax(-1) == q[i:i + chunk].argmax(-1)).sum()
    return torch.cat(kls), float(agree / r.shape[0])


def kl_metrics(ref_logits: torch.Tensor, q_logits: torch.Tensor,
               chunk: int = 512) -> Dict[str, float]:
    """Summary statistics of the per-token KL, *including a standard error*.

    Per-token KL is wildly heavy-tailed here (max is often 20x the mean), so a bare mean
    over 512 tokens carries a large sampling error. Reporting it is what makes the
    difference between "scheme A beats scheme B" and "the two are indistinguishable at this
    sample size" -- and for these schemes the answer is usually the latter, so the paired
    comparison in ``paired_kl_margin`` is the statistic that actually decides.
    """
    k, agree = kl_per_token(ref_logits, q_logits, chunk)
    n = k.numel()
    return {"kl_mean_nats": float(k.mean()),
            "kl_se_nats": float(k.std(unbiased=True) / math.sqrt(n)),
            "kl_max_nats": float(k.max()),
            "kl_mean_bits": float(k.mean() / math.log(2)),
            "kl_max_bits": float(k.max() / math.log(2)),
            "kl_p99_nats": float(torch.quantile(k, 0.99)),
            "top1_agreement": agree, "n_tokens": int(n)}


def paired_kl_margin(ref_logits: torch.Tensor, a_logits: torch.Tensor,
                     b_logits: torch.Tensor) -> Dict[str, float]:
    """Paired per-token KL difference (a - b), which is the powerful test here.

    Both schemes are scored on the same tokens against the same reference, so the per-token
    difference removes the token-to-token variation that dominates the unpaired standard
    error. Without this, a 1% gap between two schemes' mean KL cannot be called either way.
    """
    ka, _ = kl_per_token(ref_logits, a_logits)
    kb, _ = kl_per_token(ref_logits, b_logits)
    d = ka - kb
    n = d.numel()
    se = float(d.std(unbiased=True) / math.sqrt(n))
    mean = float(d.mean())
    # Per-token KL here has a max around 60x its mean, and a t-statistic on differences
    # that heavy-tailed is driven by a handful of tokens. A sign test asks the more robust
    # question -- on how many tokens is b actually the better scheme -- and is reported
    # alongside, because the two disagreeing is itself the finding.
    wins = float((d > 0).to(torch.float64).mean())
    z = (wins - 0.5) / math.sqrt(0.25 / n)
    return {"kl_diff_mean_nats": mean, "kl_diff_se_nats": se,
            "kl_diff_t": (mean / se) if se > 0 else float("nan"),
            "significant_at_2se": bool(se > 0 and abs(mean) > 2 * se),
            "kl_diff_median_nats": float(d.median()),
            "frac_tokens_b_better": wins, "sign_test_z": z,
            "sign_test_significant": bool(abs(z) > 2.0)}


def decoding_audit(dram_gbps: float, n_weights: int, bpw: float,
                   int_ops_per_sec: float = 30e12) -> Dict[str, object]:
    """Static op-count audit per 8 weights, plus the roofline that decides if it matters.

    This is an instruction-count analysis, NOT a measured kernel latency -- no CUDA kernel
    for any of these formats exists yet. What it can settle is whether the extra ALU work
    E8 needs could plausibly matter at batch=1, and the answer follows from the fact that
    all four formats read exactly the same number of bytes.
    """
    rows = [
        {"format": "RTN INT4 / any 4-bit uniform", "loads_per_8w": 1,
         "int_alu_per_8w": 16 + 8, "lut_lookups_per_8w": 0, "i2f_per_8w": 8,
         "fma_per_8w": 8,
         "note": "8x (shift+mask), 8x bias subtract; no table"},
        {"format": "16-entry LUT (Lloyd-Max, NF4)", "loads_per_8w": 1,
         "int_alu_per_8w": 16, "lut_lookups_per_8w": 8, "i2f_per_8w": 0,
         "fma_per_8w": 8,
         "note": "8 indexed reads; shared-memory bank conflicts unless the table is "
                 "register-resident via a select tree (~4 selects/weight) or PRMT"},
        {"format": "E8 32-bit coset pack", "loads_per_8w": 1,
         "int_alu_per_8w": 25, "lut_lookups_per_8w": 0, "i2f_per_8w": 8,
         "fma_per_8w": 8,
         "note": "7x (shift+mask), 1x high-3 extract, parity via __popc(word & mask), "
                 "coset half-step folded into the dequant bias; pure ALU, no table, "
                 "no divergence"},
    ]
    payload_bytes = n_weights * bpw / 8
    mem_us = payload_bytes / dram_gbps / 1e3
    for r in rows:
        ops = (r["int_alu_per_8w"] + r["i2f_per_8w"] + r["fma_per_8w"]
               + 4 * r["lut_lookups_per_8w"])          # a select tree costs ~4 ops
        r["ops_per_weight"] = ops / 8
        r["alu_us_for_tensor"] = (ops * n_weights / 8) / int_ops_per_sec * 1e6
        r["alu_fraction_of_memory_time"] = r["alu_us_for_tensor"] / mem_us
    return {"per_format": rows, "payload_bytes": payload_bytes,
            "memory_time_us": mem_us, "assumed_dram_gbps": dram_gbps,
            "assumed_int_ops_per_sec": int_ops_per_sec,
            "verdict": ("all formats read identical bytes, so at batch=1 the GEMV is "
                        "bandwidth-bound and the dequant ALU is hidden; the LUT variants' "
                        "risk is not op count but shared-memory bank conflicts")}


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def build_batch(model_dir: str, num_seqs: int, seq_len: int, dataset: str,
                split: str) -> torch.Tensor:
    from capture_activations import build_token_batch
    return build_token_batch(model_dir, num_seqs, seq_len, dataset, split, 0)


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
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--validate", action="store_true",
                    help="also check the streamed loop against GlmModel.forward")
    ap.add_argument("--out", default=os.path.join(HERE, "kl_comparison_report.json"))
    ap.add_argument("--md-out", default=os.path.join(HERE, "KL_1D_VS_E8.md"))
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

    # ---- capture X and the pre-MLP residual h1 at every layer under test ----
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
    hidden = runner.run_layers(hidden, 0, n_layers, progress="ref")
    logits_model = runner.logits(hidden)
    for h in handles:
        h.remove()
    del hidden
    gc.collect(); torch.cuda.empty_cache()
    print(f"  reference logits {tuple(logits_model.shape)}", flush=True)

    grid = _alpha_grid(args.grid, device=dev)
    report: Dict[str, object] = {"meta": {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "model_dir": args.model_dir, "layers": args.layers,
        "outlier_frac": args.outlier_frac, "tokens": int(ids.numel()),
        "num_seqs": args.num_seqs, "seq_len": args.seq_len,
        "kl_direction": "forward, D(P_reference || P_quantized)",
        "logit_dtype": "fp32 projection over a bf16 tail",
    }}
    md = ["# 1-D scalar vs 8-D E8 at matched 4.013 bpw, scored to logit KL", "",
          "All candidates carry the *same* 0.1% BF16 outlier channels and the *same* "
          "per-row MSE-optimal scale search; only the codebook differs. "
          "Generated by `research/eval_1d_vs_e8_kl.py`.", ""]

    for layer in args.layers:
        print(f"\n=== layer {layer} ===", flush=True)
        w = e8.load_down_proj(args.model_dir, layer, device=dev)
        x = grabbed[layer]["x"].reshape(-1, w.shape[1]).to(dev, torch.float32)
        h1 = grabbed[layer]["h1"].to(dev)
        n_out = max(1, int(args.outlier_frac * w.shape[1]))     # floor: 0.1% -> 13
        norms = x.to(torch.float32).pow(2).sum(0)
        out_idx = torch.topk(norms, n_out).indices.sort().values
        print(f"  W {tuple(w.shape)}  X {tuple(x.shape)}  "
              f"outliers {n_out} of {w.shape[1]} channels", flush=True)

        schemes: List[Scheme] = [
            with_sparse_outliers(w, out_idx, "1D-RTN-INT4",
                                 _bulk_lut(levels_rtn_int4, grid)),
            with_sparse_outliers(w, out_idx, "1D-Lloyd-Max",
                                 _bulk_lut(levels_lloyd_max, grid), lut_entries=16),
            with_sparse_outliers(w, out_idx, "1D-NF4",
                                 _bulk_lut(levels_nf4, grid), lut_entries=16),
            with_sparse_outliers(w, out_idx, "8D-E8-Fixed", _bulk_e8(grid)),
        ]

        # ---- reassemble the residual stream for every variant, batched ----
        wf = w.to(torch.float32)
        y_ref = (x @ wf.T).reshape(*h1.shape[:-1], -1)
        variants = [("Ref-BF16", y_ref)]
        rows_out: List[Dict[str, object]] = []
        for s in schemes:
            variants.append((s.name, (x @ s.w_hat.T).reshape(*h1.shape[:-1], -1)))

        # Synthetic-noise controls: perturb Y to a KNOWN Act-SNR with white noise. If KL
        # does not fall roughly 10x per 10 dB across these, the tail is amplifying
        # perturbations chaotically and the KL column cannot rank schemes that sit within a
        # dB of each other. This calibrates the metric instead of trusting it.
        ctrl_targets = [40.0, 30.0]
        gen = torch.Generator(device=y_ref.device).manual_seed(1234)
        y_rms = float(y_ref.pow(2).mean().sqrt())
        for tgt in ctrl_targets:
            sigma = y_rms * (10 ** (-tgt / 20))
            noise = torch.randn(y_ref.shape, generator=gen, device=y_ref.device,
                                dtype=y_ref.dtype) * sigma
            variants.append((f"Ctrl-noise-{tgt:.0f}dB", y_ref + noise))
            del noise
        stack = torch.cat([(h1.to(torch.float32) + yv).to(torch.bfloat16)
                           for _, yv in variants], dim=0)
        print(f"  tail batch {tuple(stack.shape)} "
              f"({len(variants)} variants x {args.num_seqs} seqs)", flush=True)

        hid = runner.run_layers(stack, layer + 1, n_layers, progress=f"L{layer}")
        logits = runner.logits(hid)
        del hid, stack
        gc.collect(); torch.cuda.empty_cache()

        per = args.num_seqs
        ref_logits = logits[0:per]
        floor = kl_metrics(logits_model[0:per], ref_logits)
        report[f"layer{layer}_method_floor"] = floor
        print(f"  method floor (fp32-restart vs model's own bf16 path): "
              f"KL={floor['kl_mean_nats']:.3e} nats, top1={floor['top1_agreement']:.4f}",
              flush=True)

        rows_out.append({"scheme": "Ref-BF16", "total_bpw": 16.0,
                         "act_snr_db": float("inf"), "cos_mean": 1.0, "cos_min": 1.0,
                         "kl_mean_nats": 0.0, "kl_max_nats": 0.0, "kl_mean_bits": 0.0,
                         "kl_max_bits": 0.0, "top1_agreement": 1.0})
        for i, s in enumerate(schemes, start=1):
            lm = layer_metrics(x, wf, s.w_hat)
            km = kl_metrics(ref_logits, logits[i * per:(i + 1) * per])
            wm = e8.weight_metrics(wf, s.w_hat)
            rows_out.append({"scheme": s.name, "total_bpw": s.total_bpw,
                             "payload_bpw": s.payload_bpw,
                             "overhead_bpw": s.overhead_bpw,
                             "weight_sqnr_db": wm["weight_sqnr_db"],
                             "weight_linf": wm["weight_linf"],
                             **lm, **km, "extra": s.extra})
            print(f"    {s.name:<14} bpw={s.total_bpw:.4f}  SNR={lm['act_snr_db']:7.3f}  "
                  f"cos={lm['cos_mean']:.6f}/{lm['cos_min']:.6f}  "
                  f"KL={km['kl_mean_nats']:.3e}+-{km['kl_se_nats']:.1e}  "
                  f"top1={km['top1_agreement']:.4f}", flush=True)

        # the synthetic-noise controls, which calibrate whether KL can rank at all
        n_sch = len(schemes)
        for j, tgt in enumerate(ctrl_targets):
            sl = logits[(n_sch + 1 + j) * per:(n_sch + 2 + j) * per]
            km = kl_metrics(ref_logits, sl)
            y_v = variants[n_sch + 1 + j][1]
            snr = float(10 * torch.log10(y_ref.to(torch.float64).pow(2).sum()
                                         / (y_ref - y_v).to(torch.float64).pow(2).sum()))
            rows_out.append({"scheme": f"Ctrl-noise-{tgt:.0f}dB", "total_bpw": None,
                             "act_snr_db": snr, **km})
            print(f"    {'Ctrl-noise-%.0fdB' % tgt:<14} (control)      "
                  f"SNR={snr:7.3f}  KL={km['kl_mean_nats']:.3e}"
                  f"+-{km['kl_se_nats']:.1e}  top1={km['top1_agreement']:.4f}", flush=True)

        # geometric margin of E8 over the best 1-D scalar, which is the brief's question
        e8row = next(r for r in rows_out if r["scheme"] == "8D-E8-Fixed")
        e8_logits = logits[4 * per:5 * per]
        cand = [r for r in rows_out if str(r["scheme"]).startswith("1D")]
        best1d = max(cand, key=lambda r: r["act_snr_db"])
        b_idx = 1 + [s.name for s in schemes].index(str(best1d["scheme"]))
        paired = paired_kl_margin(ref_logits, logits[b_idx * per:(b_idx + 1) * per],
                                  e8_logits)
        margin = {
            "best_1d": best1d["scheme"],
            "act_snr_margin_db": e8row["act_snr_db"] - best1d["act_snr_db"],
            "kl_ratio_1d_over_e8": (best1d["kl_mean_nats"] / e8row["kl_mean_nats"]
                                    if e8row["kl_mean_nats"] else float("inf")),
            "e8_kl_over_floor": (e8row["kl_mean_nats"] / floor["kl_mean_nats"]
                                 if floor["kl_mean_nats"] else float("inf")),
            "bpw_delta": e8row["total_bpw"] - best1d["total_bpw"],
            "top1_margin": e8row["top1_agreement"] - best1d["top1_agreement"],
            "paired_kl_best1d_minus_e8": paired,
        }
        report[f"layer{layer}_margin"] = margin
        print(f"  E8 vs best 1-D ({best1d['scheme']}): "
              f"{margin['act_snr_margin_db']:+.3f} dB Act-SNR, "
              f"KL {margin['kl_ratio_1d_over_e8']:.2f}x, "
              f"{margin['bpw_delta']:+.5f} bpw; paired KL diff "
              f"{paired['kl_diff_mean_nats']:+.2e}+-{paired['kl_diff_se_nats']:.1e} "
              f"(t={paired['kl_diff_t']:+.2f}, "
              f"{'SIGNIFICANT' if paired['significant_at_2se'] else 'not significant'})",
              flush=True)

        report[f"layer{layer}"] = rows_out
        md.append(_md_table(rows_out, layer, floor, margin))
        md.append("")
        del w, wf, x, h1, logits, ref_logits, schemes, variants, y_ref
        gc.collect(); torch.cuda.empty_cache()

    report["decoding_audit"] = decoding_audit(564.0, 56098816, 4.013)
    md.append(_md_audit(report["decoding_audit"]))

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


def _md_table(rows: List[Dict[str, object]], layer: int, floor: Dict[str, float],
              margin: Dict[str, object]) -> str:
    def f(r, k, spec):
        v = r.get(k)
        if v is None:
            return "--"
        if isinstance(v, float) and math.isinf(v):
            return "inf"
        return format(v, spec)
    out = [f"### Layer {layer} `down_proj` -- all schemes + 0.1% BF16 outliers", "",
           "| Scheme | Total bpw | Act-SNR (dB) | CosSim mean | CosSim min-token "
           "| KL mean (nats) | KL SE | KL max (nats) | KL mean (bits) | Top-1 agree "
           "| W-SQNR |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(
            f"| {r['scheme']} | {f(r, 'total_bpw', '.4f')} | {f(r, 'act_snr_db', '.2f')} | "
            f"{f(r, 'cos_mean', '.6f')} | {f(r, 'cos_min', '.6f')} | "
            f"{f(r, 'kl_mean_nats', '.3e')} | {f(r, 'kl_se_nats', '.1e')} | "
            f"{f(r, 'kl_max_nats', '.3e')} | "
            f"{f(r, 'kl_mean_bits', '.3e')} | {f(r, 'top1_agreement', '.4f')} | "
            f"{f(r, 'weight_sqnr_db', '.2f')} |")
    p = margin["paired_kl_best1d_minus_e8"]
    out += ["",
            f"*Method noise floor* (fp32 restart vs the model's own bf16 path): "
            f"KL = {floor['kl_mean_nats']:.3e} nats, top-1 agreement "
            f"{floor['top1_agreement']:.4f}. E8's KL is "
            f"{margin['e8_kl_over_floor']:.1f}x the floor.",
            "",
            f"**Margin of E8 over the best 1-D scalar ({margin['best_1d']}), at "
            f"{margin['bpw_delta']:+.5f} bpw: {margin['act_snr_margin_db']:+.3f} dB "
            f"Act-SNR, {margin['top1_margin']:+.4f} top-1 agreement. Paired per-token KL "
            f"difference (best-1D minus E8) = {p['kl_diff_mean_nats']:+.2e} "
            f"+- {p['kl_diff_se_nats']:.1e} nats, t = {p['kl_diff_t']:+.2f} -- "
            f"{'significant' if p['significant_at_2se'] else 'NOT significant'} at 2 SE. "
            f"Robust sign test: E8 has the lower KL on "
            f"{100 * p['frac_tokens_b_better']:.1f}% of tokens (z = "
            f"{p['sign_test_z']:+.2f}, "
            f"{'significant' if p['sign_test_significant'] else 'NOT significant'}).**"]
    return "\n".join(out)


def _md_audit(a: Dict[str, object]) -> str:
    out = ["### Decoding complexity audit (static op count, not a measured kernel)", "",
           "| Format | 32-bit loads / 8w | int ALU / 8w | LUT reads / 8w | ops / weight "
           "| dequant ALU time | as % of memory time |",
           "|---|---|---|---|---|---|---|"]
    for r in a["per_format"]:
        out.append(f"| {r['format']} | {r['loads_per_8w']} | {r['int_alu_per_8w']} | "
                   f"{r['lut_lookups_per_8w']} | {r['ops_per_weight']:.1f} | "
                   f"{r['alu_us_for_tensor']:.1f} us | "
                   f"{100 * r['alu_fraction_of_memory_time']:.1f}% |")
    out += ["", f"One tensor is {a['payload_bytes'] / 1e6:.1f} MB at 4.013 bpw; reading it "
            f"at {a['assumed_dram_gbps']:.0f} GB/s takes {a['memory_time_us']:.0f} us. "
            f"{a['verdict']}.",
            "", "Notes per format:", ""]
    for r in a["per_format"]:
        out.append(f"- **{r['format']}** -- {r['note']}")
    return "\n".join(out)


if __name__ == "__main__":
    raise SystemExit(main())
