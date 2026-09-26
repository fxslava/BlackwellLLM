"""Multi-rate sweep b in {3,4,5}: where does 8-D lattice geometry overtake 1-D companding?

Part IV found the mechanism that caps companded E8 at 4 bpw. A cubic lattice tiles the code
box exactly and overloads nowhere; E8's 240-facet Voronoi cells do not, so a fraction of them
are clipped by the box faces, and companding -- which flattens the source onto that box -- is
what drives mass into exactly those cells. At b=4 that cost 0.249 dB, pulling the packing
ceiling from +0.654 dB down to +0.405 dB.

The boundary fraction shrinks geometrically with coordinate width,

    f_boundary(b) = 1 - (1 - 2/2^b)^8      ->   89.9% (b=3), 65.6% (b=4), 40.3% (b=5)

so the ceiling should rise with b, and somewhere there ought to be a cross-over: below it
1-D companding wins, above it lattice geometry does. This module measures the ceiling
directly for each b (Tier 1, no model needed) and then checks whether the real tensors follow
(Tier 2, full logit KL).

TWO BASELINE CHOICES WORTH STATING UP FRONT
-------------------------------------------
Parts I, II and III of this study each reached a wrong verdict by comparing against a
baseline fitted less hard than the candidate, so both debatable choices here are made in the
baseline's favour:

* **RTN INT-b is asymmetric** (-2^(b-1) .. 2^(b-1)-1, all 2^b codes). Part III used the
  symmetric convention the brief named, which throws away one code and 0.56 dB at b=4 --
  enough to have flattered the lattice. Asymmetric RTN is the honest uniform baseline, and it
  has exactly as many codepoints in the box as E8 does (E8 is unimodular), so uniform E8 vs
  uniform RTN is an exactly equal-rate test of cell shape alone. A symmetric row is kept at
  b=4 only, to keep Part III's numbers traceable.
* **NF-b levels are fitted to the empirical weights**, not to N(0,1). The brief specifies a
  Gaussian fit, but these weights have kurtosis 4.25, so the empirical fit is strictly
  stronger and is what the candidate must beat. A Gaussian-fitted row is kept at b=4 to
  quantify what the canonical QLoRA choice costs.

Both extra rows are diagnostics, not part of the sweep proper.
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
    TailRunner, Scheme, with_sparse_outliers, layer_metrics, kl_per_token,
    levels_nf4, quantize_lut_1d, _bulk_lut, _bulk_e8, build_batch,
)
from eval_companded_e8 import _bulk_e8_companded, silicon_audit  # noqa: E402

F64 = torch.float64
BITWIDTHS = (3, 4, 5)


# --------------------------------------------------------------------------- #
# codebooks, generalized over b
# --------------------------------------------------------------------------- #

def _b_of(name: str) -> int:
    """Coordinate width from a scheme name; every name carries exactly one digit run."""
    import re
    m = re.search(r"\d+", name)
    if m is None:
        raise ValueError(f"no bitwidth in scheme name {name!r}")
    return int(m.group())


def levels_uniform(b: int, symmetric: bool = False) -> torch.Tensor:
    """Uniform INT-b levels. Asymmetric uses all 2^b codes; symmetric forfeits one."""
    half = 1 << (b - 1)
    lo = -(half - 1) if symmetric else -half
    return torch.arange(lo, half, dtype=torch.float32)


def levels_nf_gaussian(n: int) -> torch.Tensor:
    """Canonical NormalFloat: n equiprobable quantiles of N(0,1), normalized to max|.| = 1."""
    p = (torch.arange(n, dtype=F64) + 0.5) / n
    q = math.sqrt(2.0) * torch.erfinv(2.0 * p - 1.0)
    return (q / q.abs().max()).to(torch.float32)


def _lut_bits(n_levels: int) -> int:
    return n_levels * 16


# --------------------------------------------------------------------------- #
# Tier 1 -- the packing ceiling as a function of b
# --------------------------------------------------------------------------- #

def boundary_diagnostic(b: int, n: int = 10_000_000, device: str = "cuda",
                        seed: int = 0, chunk: int = 1_000_000) -> Dict[str, float]:
    """E8 vs Z^8 MSE on a uniform source at width b, with and without the box.

    A uniform source is what a companding warp produces, and at equal codepoint count
    (2^(8b) for either lattice in the box, since E8 is unimodular) the only thing that can
    differ is cell shape. The source is uniform on [lo-0.5, hi+0.5]^8, the support Z^8's
    levels tile exactly. Two conditions, differing only in whether the quantizer is bounded:

    * ``unbounded`` -- quantize with the *infinite* lattices, no clamping. Granular MSE is
      spatially periodic, so a uniform source over any region a few cells across yields the
      exact asymptotic MSE with no edge bias whatsoever: 1/12 for Z^8 and G_8 for E8, hence
      +0.654 dB at every b. This is cleaner than Part IV's "hold the source inside the box"
      formulation, which needed a margin the b=3 box is too small to give.
    * ``boxed``     -- the same source through the real bounded quantizers, so overload is
      included. This is the regime companding creates, and it is the true ceiling on what
      companded E8 can win over NF-b, because NF-b *is* companded Z^8.

    Their difference is the boundary loss. Accumulated in float64 over chunks so 10^7
    vectors fit at any b.
    """
    lo, hi = box_for(b)
    width = float(hi - lo + 1)
    centre = (lo + hi) / 2.0
    g = torch.Generator(device=device).manual_seed(seed)
    out: Dict[str, float] = {"b": b, "box_lo": lo, "box_hi": hi, "n_samples": n,
                             "source_width": width}
    acc = {t: {"z": torch.zeros((), dtype=F64, device=device),
               "e": torch.zeros((), dtype=F64, device=device),
               "edge": torch.zeros((), dtype=F64, device=device)}
           for t in ("unbounded", "boxed")}
    seen = 0
    for start in range(0, n, chunk):
        m = min(chunk, n - start)
        u = (torch.rand(m, E8_DIM, generator=g, device=device) - 0.5) * width + centre
        # unbounded: pure granular error, no clipping anywhere
        ku, cu = e8.closest_e8(u)
        ptu = ku + 0.5 * cu.unsqueeze(-1).to(ku.dtype)
        acc["unbounded"]["z"] += (u - u.round()).to(F64).pow(2).sum()
        acc["unbounded"]["e"] += (u - ptu).to(F64).pow(2).sum()
        # boxed: the real fixed-rate quantizers
        zb = u.round().clamp(float(lo), float(hi))
        kb, cb = closest_e8_boxed(u, lo, hi)
        ptb = kb + 0.5 * cb.unsqueeze(-1).to(kb.dtype)
        acc["boxed"]["z"] += (u - zb).to(F64).pow(2).sum()
        acc["boxed"]["e"] += (u - ptb).to(F64).pow(2).sum()
        acc["boxed"]["edge"] += ((kb <= lo) | (kb >= hi)).to(F64).sum()
        seen += m * E8_DIM
        del u, ku, cu, ptu, zb, kb, cb, ptb
    for tag in ("unbounded", "boxed"):
        mse_z = float(acc[tag]["z"] / seen)
        mse_e = float(acc[tag]["e"] / seen)
        out[f"{tag}_mse_z8"] = mse_z
        out[f"{tag}_mse_e8"] = mse_e
        out[f"{tag}_e8_gain_db"] = 10.0 * math.log10(mse_z / mse_e)
    out["boxed_coords_on_edge"] = float(acc["boxed"]["edge"] / seen)
    out["asymptotic_prediction_db"] = 10.0 * math.log10((1.0 / 12.0) / 0.0716818)
    out["f_boundary_predicted"] = 1.0 - (1.0 - 2.0 / (1 << b)) ** E8_DIM
    out["boundary_loss_db"] = out["unbounded_e8_gain_db"] - out["boxed_e8_gain_db"]
    out["ceiling_db"] = out["boxed_e8_gain_db"]
    return out


# --------------------------------------------------------------------------- #
# Tier 2 -- lm_head held resident so extra variants are nearly free
# --------------------------------------------------------------------------- #

class ResidentHead:
    """Final norm + fp32 lm_head kept on the GPU across all variants.

    Part III moved the 2.48 GB fp32 lm_head up and down for every variant, which is fine for
    7 of them and wasteful for 18. Holding it resident makes each additional scheme cost one
    matmul, which is what lets this sweep carry 14 schemes plus controls in one pass.
    """

    def __init__(self, runner: TailRunner, device: str = "cuda"):
        self.runner, self.device = runner, device
        runner.base.norm.to(device)
        self.wt = runner.m.lm_head.weight.to(device, torch.float32).T.contiguous()

    def __call__(self, hidden: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
        with torch.no_grad():
            h = self.runner.base.norm(hidden)
        flat = h.reshape(-1, h.shape[-1]).to(torch.float32)
        out = torch.empty(flat.shape[0], self.wt.shape[1], dtype=torch.float32,
                          device=self.device)
        with torch.no_grad():
            for i in range(0, flat.shape[0], chunk):
                out[i:i + chunk] = flat[i:i + chunk] @ self.wt
        return out.reshape(*h.shape[:-1], -1)

    def close(self) -> None:
        del self.wt
        self.runner.base.norm.to("cpu")
        torch.cuda.empty_cache()


def paired_from_kl(ka: torch.Tensor, kb: torch.Tensor) -> Dict[str, float]:
    """Paired stats on per-token KL vectors: t-test plus the robust sign test.

    Both schemes are scored on the same tokens against the same reference, so the per-token
    difference removes the token-to-token variation that dominates an unpaired comparison.
    Per-token KL is heavy-tailed enough (max ~60x mean) that the sign test is the one to
    believe when the two disagree.
    """
    d = (ka - kb).to(F64)
    n = d.numel()
    se = float(d.std(unbiased=True) / math.sqrt(n))
    mean = float(d.mean())
    wins = float((d > 0).to(F64).mean())
    z = (wins - 0.5) / math.sqrt(0.25 / n)
    return {"kl_diff_mean_nats": mean, "kl_diff_se_nats": se,
            "kl_diff_t": (mean / se) if se > 0 else float("nan"),
            "kl_diff_median_nats": float(d.median()),
            "frac_tokens_b_better": wins, "sign_test_z": z,
            "sign_test_significant": bool(abs(z) > 2.0),
            "meets_z3": bool(z > 3.0)}


def kl_summary(k: torch.Tensor, agree: float) -> Dict[str, float]:
    n = k.numel()
    return {"kl_mean_nats": float(k.mean()),
            "kl_se_nats": float(k.std(unbiased=True) / math.sqrt(n)),
            "kl_max_nats": float(k.max()),
            "kl_p99_nats": float(torch.quantile(k, 0.99)),
            "top1_agreement": agree, "n_tokens": int(n)}


# --------------------------------------------------------------------------- #
# scheme construction for one bitwidth
# --------------------------------------------------------------------------- #

def scheme_specs(bitwidths: Tuple[int, ...], grid: torch.Tensor, extras_at: int = 4,
                 lambda_grid: Optional[List[float]] = None
                 ) -> List[Tuple[str, Callable, int, int]]:
    """(name, bulk_fn, lut_entries, b) for every row, in table order.

    ``b`` is carried explicitly rather than parsed back out of the name: "E8-Uniform-3"
    contains the digit 8 before the 3, so name-parsing silently assigned every lattice row
    b=8 and dropped it from the per-rate comparisons.
    """
    specs: List[Tuple[str, Callable, int, int]] = []
    for b in bitwidths:
        n_lv = 1 << b
        specs += [
            (f"RTN-INT{b}",
             _bulk_lut(lambda x, b=b: levels_uniform(b), grid, bits_per_weight=b), 0, b),
            (f"E8-Uniform-{b}", _bulk_e8(grid, coord_bits=b), 0, b),
            (f"NF{b}",
             _bulk_lut(lambda x, n=n_lv: levels_nf4(x, n=n), grid, bits_per_weight=b),
             n_lv, b),
            (f"E8-Companded-{b}",
             _bulk_e8_companded(grid, coord_bits=b, lambda_grid=lambda_grid),
             2 * n_lv, b),
        ]
        if b == extras_at:
            specs += [
                (f"RTN-INT{b}-sym",
                 _bulk_lut(lambda x, b=b: levels_uniform(b, symmetric=True), grid,
                           bits_per_weight=b), 0, b),
                (f"NF{b}-gaussian",
                 _bulk_lut(lambda x, n=n_lv: levels_nf_gaussian(n), grid,
                           bits_per_weight=b), n_lv, b),
            ]
    return specs


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", default=e8.DEFAULT_MODEL_DIR)
    ap.add_argument("--layers", type=int, nargs="+", default=[3, 10])
    ap.add_argument("--bitwidths", type=int, nargs="+", default=list(BITWIDTHS))
    ap.add_argument("--outlier-frac", type=float, default=0.001)
    ap.add_argument("--num-seqs", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--dataset", default="Salesforce/wikitext")
    ap.add_argument("--split", default="train")
    ap.add_argument("--grid", type=int, default=40)
    ap.add_argument("--lambda-grid", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0],
                    help="companding strength candidates; 0 is uniform E8, 1 is the full "
                         "quantile warp. Including 0 guarantees companded E8 can never be "
                         "worse than uniform E8.")
    ap.add_argument("--diag-samples", type=int, default=10_000_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tier1-only", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "multirate_report.json"))
    ap.add_argument("--md-out", default=os.path.join(HERE, "MULTIRATE_SWEEP.md"))
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    dev = args.device
    t0 = time.time()
    bws = tuple(args.bitwidths)
    report: Dict[str, object] = {"meta": {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "bitwidths": list(bws), "layers": args.layers,
        "outlier_frac": args.outlier_frac, "diag_samples": args.diag_samples,
        "rtn_convention": "asymmetric (all 2^b codes); a symmetric row is kept at b=4",
        "nf_fit": "empirical quantiles; a N(0,1)-fitted row is kept at b=4",
    }}
    md = ["# Multi-rate sweep b in {3,4,5}: lattice geometry vs 1-D companding", "",
          "Does the packing ceiling rise with coordinate width, and where does 8-D geometry "
          "overtake 1-D companding? Generated by `research/eval_multirate_sweep.py`.", ""]

    # ---------------- Tier 1 ----------------
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"== Tier 1: packing ceiling vs b ({args.diag_samples/1e6:.0f} M vectors each) ==")
    diags = {}
    for b in bws:
        d = boundary_diagnostic(b, n=args.diag_samples, device=dev)
        diags[b] = d
        print(f"  b={b}: unbounded {d['unbounded_e8_gain_db']:+.3f} dB   "
              f"boxed(ceiling) {d['boxed_e8_gain_db']:+.3f} dB   "
              f"loss {d['boundary_loss_db']:.3f} dB   "
              f"f_boundary={100*d['f_boundary_predicted']:.1f}%   "
              f"coords on edge {100*d['boxed_coords_on_edge']:.1f}%", flush=True)
    report["tier1_boundary"] = {str(b): diags[b] for b in bws}
    md.append(_md_tier1(diags, bws))
    md.append("")
    if args.tier1_only:
        _write(report, md, args)
        return 0

    # ---------------- Tier 2 ----------------
    from transformers.models.glm import GlmForCausalLM
    print("\nloading GLM-4-9B onto host RAM (bf16, 18.9 GB)...", flush=True)
    model = GlmForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16).eval()
    n_layers = model.config.num_hidden_layers
    runner = TailRunner(model, dev)
    ids = build_batch(args.model_dir, args.num_seqs, args.seq_len, args.dataset, args.split)

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
    head = ResidentHead(runner, dev)
    logits_model = head(hidden)
    for h in handles:
        h.remove()
    del hidden
    gc.collect(); torch.cuda.empty_cache()

    grid = _alpha_grid(args.grid, device=dev)
    for layer in args.layers:
        print(f"\n=== layer {layer} ===", flush=True)
        w = e8.load_down_proj(args.model_dir, layer, device=dev)
        x = grabbed[layer]["x"].reshape(-1, w.shape[1]).to(dev, torch.float32)
        h1 = grabbed[layer]["h1"].to(dev)
        wf = w.to(torch.float32)
        n_out = max(1, int(args.outlier_frac * w.shape[1]))
        out_idx = torch.topk(x.pow(2).sum(0), n_out).indices.sort().values
        print(f"  outliers {n_out} of {w.shape[1]} channels", flush=True)

        y_ref = (x @ wf.T).reshape(*h1.shape[:-1], -1)
        variants: List[Tuple[str, torch.Tensor]] = [("Ref-BF16", y_ref)]
        # Build one scheme at a time and free its dequantized matrix as soon as its output
        # and metrics exist: 14 of them at 224 MB each would otherwise sit alongside the
        # resident 2.48 GB fp32 lm_head.
        built: List[Dict[str, object]] = []
        for name, fn, lut_n, b_of in scheme_specs(bws, grid,
                                                  lambda_grid=args.lambda_grid):
            ts = time.time()
            s = with_sparse_outliers(w, out_idx, name, fn, lut_entries=lut_n)
            y = (x @ s.w_hat.T).reshape(*h1.shape[:-1], -1)
            built.append({"scheme": name, "b": b_of, "total_bpw": s.total_bpw,
                          "payload_bpw": s.payload_bpw, "overhead_bpw": s.overhead_bpw,
                          **e8.weight_metrics(wf, s.w_hat),
                          **layer_metrics(x, wf, s.w_hat), "extra": s.extra})
            variants.append((name, y))
            print(f"    built {name:<18} bpw={s.total_bpw:.4f} "
                  f"W-SQNR={built[-1]['weight_sqnr_db']:6.2f} "
                  f"Act-SNR={built[-1]['act_snr_db']:7.3f} ({time.time() - ts:.0f}s)",
                  flush=True)
            del s
            gc.collect(); torch.cuda.empty_cache()
        ctrl = [40.0, 30.0, 25.0]
        gen = torch.Generator(device=dev).manual_seed(1234)
        y_rms = float(y_ref.pow(2).mean().sqrt())
        for tgt in ctrl:
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
        ref_logits = head(hid[0:per])
        kfloor, afloor = kl_per_token(logits_model[0:per], ref_logits)
        floor = kl_summary(kfloor, afloor)
        report[f"layer{layer}_method_floor"] = floor
        print(f"  method floor: KL={floor['kl_mean_nats']:.3e}, "
              f"top1={floor['top1_agreement']:.4f}", flush=True)

        rows: List[Dict[str, object]] = [
            {"scheme": "Ref-BF16", "b": None, "total_bpw": 16.0,
             "act_snr_db": float("inf"), "cos_mean": 1.0, "cos_min": 1.0,
             "kl_mean_nats": 0.0, "kl_max_nats": 0.0, "top1_agreement": 1.0}]
        kvecs: Dict[str, torch.Tensor] = {}
        for i, rec in enumerate(built, start=1):
            lg = head(hid[i * per:(i + 1) * per])
            k, ag = kl_per_token(ref_logits, lg)
            del lg
            kvecs[str(rec["scheme"])] = k
            rows.append({**rec, **kl_summary(k, ag)})
            print(f"    {rec['scheme']:<18} SNR={rec['act_snr_db']:7.3f} "
                  f"cos={rec['cos_mean']:.6f}/{rec['cos_min']:.6f} "
                  f"KL={float(k.mean()):.3e} top1={ag:.4f}", flush=True)
        for j, tgt in enumerate(ctrl):
            lg = head(hid[(len(built) + 1 + j) * per:(len(built) + 2 + j) * per])
            k, ag = kl_per_token(ref_logits, lg)
            del lg
            yv = variants[len(built) + 1 + j][1]
            snr = float(10 * torch.log10(y_ref.to(F64).pow(2).sum()
                                         / (y_ref - yv).to(F64).pow(2).sum()))
            rows.append({"scheme": f"Ctrl-noise-{tgt:.0f}dB", "b": None,
                         "total_bpw": None, "act_snr_db": snr, **kl_summary(k, ag)})

        # per-b verdicts: companded E8 and uniform E8 against NF-b at the same rate
        verdicts = []
        for b in bws:
            nf = next(r for r in rows if r["scheme"] == f"NF{b}")
            for cand in (f"E8-Uniform-{b}", f"E8-Companded-{b}"):
                cr = next(r for r in rows if r["scheme"] == cand)
                p = paired_from_kl(kvecs[f"NF{b}"], kvecs[cand])
                verdicts.append({
                    "b": b, "candidate": cand,
                    "act_snr_margin_vs_nf_db": cr["act_snr_db"] - nf["act_snr_db"],
                    "weight_sqnr_margin_vs_nf_db":
                        cr["weight_sqnr_db"] - nf["weight_sqnr_db"],
                    "bpw_delta": cr["total_bpw"] - nf["total_bpw"],
                    "ceiling_db": diags[b]["ceiling_db"],
                    "fraction_of_ceiling": ((cr["weight_sqnr_db"] - nf["weight_sqnr_db"])
                                            / diags[b]["ceiling_db"]
                                            if diags[b]["ceiling_db"] else float("nan")),
                    "top1_margin": cr["top1_agreement"] - nf["top1_agreement"],
                    "paired_kl_nf_minus_candidate": p,
                })
            v = verdicts[-1]
            print(f"  b={b}: E8-Companded vs NF{b}  "
                  f"Act-SNR {v['act_snr_margin_vs_nf_db']:+.3f} dB  "
                  f"W-SQNR {v['weight_sqnr_margin_vs_nf_db']:+.3f} dB "
                  f"(ceiling {v['ceiling_db']:+.3f})  "
                  f"KL z={v['paired_kl_nf_minus_candidate']['sign_test_z']:+.2f}", flush=True)
        report[f"layer{layer}_verdicts"] = verdicts
        report[f"layer{layer}"] = rows
        md.append(_md_tier2(rows, layer, floor, verdicts, bws))
        md.append("")
        del w, wf, x, h1, hid, ref_logits, kvecs, built, variants, y_ref
        gc.collect(); torch.cuda.empty_cache()

    head.close()
    report["silicon_audit"] = silicon_audit(564.0, 56098816, 4.013)
    md.append(_md_pareto(report, args.layers, bws))
    _write(report, md, args)
    print(f"\nwrote {args.out} and {args.md_out} in {time.time() - t0:.0f}s")
    return 0


def _write(report, md, args) -> None:
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    with open(args.md_out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    print("\n" + "\n".join(md))


def _md_tier1(diags, bws) -> str:
    out = ["## Tier 1 -- the packing ceiling as a function of coordinate width", "",
           "E8 vs Z^8 on a *uniform* source (what companding produces) at equal codepoint "
           "count. `unbounded` quantizes with the infinite lattices, isolating pure granular "
           "error; `boxed` uses the real bounded quantizers, so overload is included.", "",
           "| b | levels/coord | f_boundary (predicted) | coords on edge (boxed) "
           "| unbounded gain | boxed gain = **ceiling** | boundary loss |",
           "|---|---|---|---|---|---|---|"]
    for b in bws:
        d = diags[b]
        out.append(f"| {b} | {1 << b} | {100 * d['f_boundary_predicted']:.1f}% | "
                   f"{100 * d['boxed_coords_on_edge']:.1f}% | "
                   f"{d['unbounded_e8_gain_db']:+.3f} dB | "
                   f"**{d['boxed_e8_gain_db']:+.3f} dB** | "
                   f"{d['boundary_loss_db']:.3f} dB |")
    d0 = diags[bws[0]]
    out += ["", f"The `unbounded` column reproduces the asymptotic "
            f"10*log10((1/12)/G_8) = {d0['asymptotic_prediction_db']:+.3f} dB at every width, "
            f"which is simultaneously the check that the lattice decoder is correct there. "
            f"The `boxed` column is the ceiling on what companded E8 can win over NF-b, "
            f"because NF-b *is* companded Z^8."]
    return "\n".join(out)


def _md_tier2(rows, layer, floor, verdicts, bws) -> str:
    def f(r, k, spec):
        v = r.get(k)
        if v is None:
            return "--"
        if isinstance(v, float) and math.isinf(v):
            return "inf"
        return format(v, spec)
    out = [f"## Tier 2 -- layer {layer} `down_proj` (all rows + 0.1% BF16 outliers)", "",
           "| Scheme | b | Total bpw | W-SQNR | Act-SNR | CosSim mean | CosSim min-token "
           "| KL mean (1e-4) | KL max (1e-4) | Top-1 |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        kl, klm = r.get("kl_mean_nats"), r.get("kl_max_nats")
        out.append(
            f"| {r['scheme']} | {f(r, 'b', 'd')} | {f(r, 'total_bpw', '.4f')} | "
            f"{f(r, 'weight_sqnr_db', '.2f')} | {f(r, 'act_snr_db', '.2f')} | "
            f"{f(r, 'cos_mean', '.6f')} | {f(r, 'cos_min', '.6f')} | "
            f"{'--' if kl is None else format(kl * 1e4, '.2f')} | "
            f"{'--' if klm is None else format(klm * 1e4, '.1f')} | "
            f"{f(r, 'top1_agreement', '.4f')} |")
    out += ["", f"*Method noise floor*: KL = {floor['kl_mean_nats'] * 1e4:.2f}e-4 nats, "
            f"top-1 {floor['top1_agreement']:.4f}.", "",
            f"### Layer {layer} -- margins against NF-b at the same width", "",
            "| b | Candidate | Act-SNR vs NF | W-SQNR vs NF | ceiling | frac of ceiling "
            "| KL sign-test z | z>3? |", "|---|---|---|---|---|---|---|---|"]
    for v in verdicts:
        p = v["paired_kl_nf_minus_candidate"]
        out.append(f"| {v['b']} | {v['candidate']} | "
                   f"{v['act_snr_margin_vs_nf_db']:+.3f} | "
                   f"{v['weight_sqnr_margin_vs_nf_db']:+.3f} | "
                   f"{v['ceiling_db']:+.3f} | "
                   f"{v['fraction_of_ceiling']:.2f} | "
                   f"{p['sign_test_z']:+.2f} ({100 * p['frac_tokens_b_better']:.1f}%) | "
                   f"{'**yes**' if p['meets_z3'] else 'no'} |")
    return "\n".join(out)


def _md_pareto(report, layers, bws) -> str:
    out = ["## Pareto-optimal format per rate", "",
           "| b | layer | best by Act-SNR | best by logit KL | best by min-token CosSim |",
           "|---|---|---|---|---|"]
    for b in bws:
        for layer in layers:
            rows = [r for r in report.get(f"layer{layer}", [])
                    if r.get("b") == b and r.get("act_snr_db") is not None]
            if not rows:
                continue
            best_snr = max(rows, key=lambda r: r["act_snr_db"])
            best_kl = min(rows, key=lambda r: r["kl_mean_nats"])
            best_cos = max(rows, key=lambda r: r.get("cos_min", -1))
            out.append(f"| {b} | {layer} | {best_snr['scheme']} "
                       f"({best_snr['act_snr_db']:.2f} dB) | {best_kl['scheme']} "
                       f"({best_kl['kl_mean_nats'] * 1e4:.2f}e-4) | "
                       f"{best_cos['scheme']} ({best_cos['cos_min']:.6f}) |")
    return "\n".join(out)


if __name__ == "__main__":
    raise SystemExit(main())
