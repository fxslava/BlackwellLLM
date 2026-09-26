"""Full-model multi-rate sweep: every weight tensor of GLM-4-9B at b in {3,4,5}.

Part V found a non-monotone Pareto curve on two probe tensors (layer 3 and layer 10
``down_proj``): the lattice wins at b=3, ties at b=4, wins decisively at b=5. This module
asks whether that survives generalization to all 240 weight tensors -- 6 projections x 40
layers, 8.24 B parameters -- and whether the b=4 trough is a property of the format or of the
two tensors it was found on.

TWO PHASES, BECAUSE THEY ANSWER DIFFERENT QUESTIONS
---------------------------------------------------
*Phase A* quantizes each tensor in isolation and scores it against the *reference*
activations that tensor really sees. Isolation is the point: with the whole model quantized,
a tensor's apparent error is contaminated by everything upstream, so per-tensor comparisons
between schemes would measure drift rather than the scheme. This phase answers where the
trough is, how lambda distributes with depth, and which tensors are fragile.

*Phase B* quantizes the entire model at once and runs a real forward to logits, which is the
only way a KL divergence over the vocabulary means anything. Run for the contender configs
plus a sensitivity-allocated mixed-precision config built from Phase A's own numbers.

LEAKAGE CONTROL
---------------
Codebooks, per-row scales and lambda are fitted on the *weights*, which carry no token
information, so they cannot leak. The one token-dependent decision is which channels to keep
in BF16 -- chosen by ||X[:,k]||_2 -- so that selection is made on a calibration split and
every reported Act-SNR, cosine, top-1 and KL is measured on a disjoint validation split. Both
are reported (``act_snr_db`` vs ``act_snr_db_calib``) so the size of the optimism is visible
rather than asserted to be small.

MEMORY DISCIPLINE (12 GB cap)
-----------------------------
No dequantized matrix is ever held past the point where its metrics exist: each scheme is
fitted, scored, and freed before the next is built, and Phase B restores each layer's
original weights immediately after running it. ``gate_up_proj`` is 112 M parameters, so a
single careless retention costs 450 MB.
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

import e8_lattice_engine as e8                                      # noqa: E402
from eval_fixed_rate_e8 import _alpha_grid                          # noqa: E402
from eval_1d_vs_e8_kl import (                                      # noqa: E402
    TailRunner, with_sparse_outliers, kl_per_token, levels_nf4, _bulk_lut, _bulk_e8,
)
from eval_companded_e8 import _bulk_e8_companded                    # noqa: E402
from eval_multirate_sweep import (                                  # noqa: E402
    levels_uniform, ResidentHead, paired_from_kl, kl_summary, boundary_diagnostic,
)

F64 = torch.float64

# (module path, the input group whose captured activation feeds it)
TENSORS: List[Tuple[str, str]] = [
    ("self_attn.q_proj", "qkv"),
    ("self_attn.k_proj", "qkv"),
    ("self_attn.v_proj", "qkv"),
    ("self_attn.o_proj", "o"),
    ("mlp.gate_up_proj", "gateup"),
    ("mlp.down_proj", "down"),
]
CAPTURE_AT = {"self_attn.q_proj": "qkv", "self_attn.o_proj": "o",
              "mlp.gate_up_proj": "gateup", "mlp.down_proj": "down"}
SCHEME_KINDS = ("RTN", "NF", "E8-Uniform", "E8-Companded")


# --------------------------------------------------------------------------- #
# checkpoint access
# --------------------------------------------------------------------------- #

def load_weight(model_dir: str, layer: int, tensor: str,
                device: str = "cpu") -> torch.Tensor:
    """mmap one ``model.layers.<l>.<tensor>.weight`` out of the shards."""
    from safetensors import safe_open
    key = f"model.layers.{layer}.{tensor}.weight"
    with open(os.path.join(model_dir, "model.safetensors.index.json"),
              "r", encoding="utf-8") as fh:
        shard = json.load(fh)["weight_map"][key]
    with safe_open(os.path.join(model_dir, shard), framework="pt", device="cpu") as fh:
        return fh.get_tensor(key).to(device)


def bulk_fn_for(kind: str, b: int, grid: torch.Tensor,
                lam_grid: List[float], lam_scan: torch.Tensor):
    """The bulk quantizer for one (scheme kind, bitwidth), plus its LUT entry count."""
    n_lv = 1 << b
    if kind == "RTN":
        return _bulk_lut(lambda x, b=b: levels_uniform(b), grid, bits_per_weight=b), 0
    if kind == "NF":
        return (_bulk_lut(lambda x, n=n_lv: levels_nf4(x, n=n), grid, bits_per_weight=b),
                n_lv)
    if kind == "E8-Uniform":
        return _bulk_e8(grid, coord_bits=b), 0
    if kind == "E8-Companded":
        return (_bulk_e8_companded(grid, coord_bits=b, lambda_grid=lam_grid,
                                   lam_scan_grid=lam_scan), 2 * n_lv)
    raise ValueError(kind)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def output_metrics(x: torch.Tensor, w: torch.Tensor, w_hat: torch.Tensor,
                   chunk: int = 256) -> Dict[str, float]:
    """Act-SNR, mean/worst-token cosine, and the reference energy the allocator needs."""
    wt = w.to(torch.float32).T.contiguous()
    wht = w_hat.to(torch.float32).T.contiguous()
    y_sq = torch.zeros((), dtype=F64, device=x.device)
    e_sq = torch.zeros((), dtype=F64, device=x.device)
    cos: List[torch.Tensor] = []
    for i in range(0, x.shape[0], chunk):
        xb = x[i:i + chunk].to(torch.float32)
        y, yh = xb @ wt, xb @ wht
        y_sq += y.to(F64).pow(2).sum()
        e_sq += (y - yh).to(F64).pow(2).sum()
        cos.append((y * yh).sum(1) / (y.norm(dim=1) * yh.norm(dim=1)).clamp(min=1e-30))
        del y, yh
    c = torch.cat(cos)
    return {"act_snr_db": float(10 * torch.log10(y_sq / e_sq)),
            "cos_mean": float(c.mean()), "cos_min": float(c.min()),
            "ref_energy": float(y_sq)}


# --------------------------------------------------------------------------- #
# Phase A -- per-tensor isolated sweep
# --------------------------------------------------------------------------- #

def phase_a(model_dir: str, layers: List[int], bitwidths: List[int],
            acts: Dict[int, Dict[str, torch.Tensor]], n_calib: int,
            grid: torch.Tensor, lam_grid: List[float],
            lam_scan: Optional[torch.Tensor],
            outlier_frac: float, device: str, verbose_every: int = 1,
            checkpoint: Optional[str] = None) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    t0 = time.time()
    for li, layer in enumerate(layers):
        for tensor, group in TENSORS:
            w = load_weight(model_dir, layer, tensor, device=device)
            wf = w.to(torch.float32)
            xa = acts[layer][group].to(device, torch.float32)
            x_cal, x_val = xa[:n_calib], xa[n_calib:]
            n_out = max(1, int(outlier_frac * w.shape[1]))
            # selection uses the CALIBRATION split only
            out_idx = torch.topk(x_cal.pow(2).sum(0), n_out).indices.sort().values
            for b in bitwidths:
                for kind in SCHEME_KINDS:
                    fn, lut_n = bulk_fn_for(kind, b, grid, lam_grid, lam_scan)
                    s = with_sparse_outliers(w, out_idx, f"{kind}-{b}", fn,
                                             lut_entries=lut_n)
                    wm = e8.weight_metrics(wf, s.w_hat)
                    om = output_metrics(x_val, wf, s.w_hat)
                    om_cal = output_metrics(x_cal, wf, s.w_hat)
                    rec: Dict[str, object] = {
                        "layer": layer, "tensor": tensor, "group": group,
                        "kind": kind, "b": b, "params": int(w.numel()),
                        "total_bpw": s.total_bpw, "payload_bpw": s.payload_bpw,
                        "overhead_bpw": s.overhead_bpw,
                        "weight_sqnr_db": wm["weight_sqnr_db"],
                        "weight_linf": wm["weight_linf"],
                        "n_outlier_channels": n_out, **om,
                        "act_snr_db_calib": om_cal["act_snr_db"],
                        "act_snr_leak_db": om_cal["act_snr_db"] - om["act_snr_db"],
                    }
                    if "lambda" in s.extra:
                        rec["lambda"] = float(s.extra["lambda"])
                    if "sat_at_bound" in s.extra:
                        rec["sat_at_bound"] = float(s.extra["sat_at_bound"])
                    rows.append(rec)
                    del s
                    gc.collect(); torch.cuda.empty_cache()
            del w, wf, xa, x_cal, x_val
            gc.collect(); torch.cuda.empty_cache()
        if verbose_every and (li % verbose_every == 0 or li == len(layers) - 1):
            done = (li + 1) / len(layers)
            el = time.time() - t0
            print(f"  layer {layer:2d} done ({li+1}/{len(layers)}), "
                  f"{el/60:.1f} min elapsed, ~{el/done/60:.0f} min total", flush=True)
        if checkpoint:
            # a 90-minute phase should not lose everything to a late failure
            with open(checkpoint, "w", encoding="utf-8") as fh:
                json.dump({"layers_done": layers[:li + 1], "per_tensor": rows}, fh,
                          default=str)
    return rows


# --------------------------------------------------------------------------- #
# Phase B -- whole-model quantized forward to logits
# --------------------------------------------------------------------------- #

def allocate_mixed(rows: List[Dict[str, object]], bitwidths: List[int],
                   target_bpw: float,
                   kind_at: Dict[int, str]) -> Tuple[Dict[Tuple[int, str], int], float]:
    """Greedy marginal-return bit allocation across tensors at a target average rate.

    Error contribution of a tensor is ``ref_energy * 10^(-ActSNR/10)`` -- the actual squared
    output error it injects -- so upgrading the tensor with the largest error reduction per
    extra bit is the right greedy step, and it uses Phase A's measured numbers rather than a
    proxy like parameter count.
    """
    lo = min(bitwidths)
    idx: Dict[Tuple[int, str, int], Dict[str, object]] = {}
    for r in rows:
        if r["kind"] == kind_at[int(r["b"])]:
            idx[(int(r["layer"]), str(r["tensor"]), int(r["b"]))] = r
    keys = sorted({(k[0], k[1]) for k in idx})
    bits = {k: lo for k in keys}

    def err(k, b):
        r = idx[(k[0], k[1], b)]
        return float(r["ref_energy"]) * 10 ** (-float(r["act_snr_db"]) / 10)

    total_params = sum(int(idx[(k[0], k[1], lo)]["params"]) for k in keys)
    spent = sum(bits[k] * int(idx[(k[0], k[1], lo)]["params"]) for k in keys)
    budget = target_bpw * total_params
    ups = sorted(bitwidths)
    while True:
        best = None
        for k in keys:
            cur = bits[k]
            nxt = next((b for b in ups if b > cur), None)
            if nxt is None:
                continue
            p = int(idx[(k[0], k[1], lo)]["params"])
            cost = (nxt - cur) * p
            if spent + cost > budget:
                continue
            gain = (err(k, cur) - err(k, nxt)) / cost
            if best is None or gain > best[0]:
                best = (gain, k, nxt, cost)
        if best is None:
            break
        _, k, nxt, cost = best
        bits[k] = nxt
        spent += cost
    return bits, spent / total_params


def quantize_layer_(layer_mod, model_dir: str, layer: int,
                    plan: Dict[str, Tuple[str, int]], acts_group: Dict[str, torch.Tensor],
                    n_calib: int, grid: torch.Tensor, lam_grid: List[float],
                    lam_scan: torch.Tensor, outlier_frac: float,
                    device: str) -> Dict[str, torch.Tensor]:
    """Replace this layer's weights with quantized ones; return the originals to restore."""
    saved: Dict[str, torch.Tensor] = {}
    for tensor, group in TENSORS:
        kind, b = plan[tensor]
        mod = layer_mod
        for part in tensor.split("."):
            mod = getattr(mod, part)
        saved[tensor] = mod.weight.data.clone()
        w = mod.weight.data
        x_cal = acts_group[group][:n_calib].to(device, torch.float32)
        n_out = max(1, int(outlier_frac * w.shape[1]))
        out_idx = torch.topk(x_cal.pow(2).sum(0), n_out).indices.sort().values
        fn, lut_n = bulk_fn_for(kind, b, grid, lam_grid, lam_scan)
        s = with_sparse_outliers(w, out_idx, f"{kind}-{b}", fn, lut_entries=lut_n)
        mod.weight.data = s.w_hat.to(w.dtype)
        del s, x_cal
        gc.collect(); torch.cuda.empty_cache()
    return saved


def restore_layer_(layer_mod, saved: Dict[str, torch.Tensor]) -> None:
    for tensor, w in saved.items():
        mod = layer_mod
        for part in tensor.split("."):
            mod = getattr(mod, part)
        mod.weight.data = w


def full_model_logits(runner: TailRunner, head: ResidentHead, ids: torch.Tensor,
                      n_layers: int, plans: Dict[int, Dict[str, Tuple[str, int]]],
                      model_dir: str, acts: Dict[int, Dict[str, torch.Tensor]],
                      n_calib: int, grid: torch.Tensor, lam_grid: List[float],
                      lam_scan: torch.Tensor, outlier_frac: float,
                      device: str) -> torch.Tensor:
    """One forward with every layer's weights quantized per ``plans``, restoring as it goes."""
    hidden = runner.embed(ids)
    pos, mask, pe = runner.context(hidden)
    for i in range(n_layers):
        layer = runner.base.layers[i].to(device)
        saved = quantize_layer_(layer, model_dir, i, plans[i], acts[i], n_calib,
                                grid, lam_grid, lam_scan, outlier_frac, device)
        with torch.no_grad():
            hidden = layer(hidden, attention_mask=mask, position_embeddings=pe,
                           position_ids=pos)
        restore_layer_(layer, saved)
        del saved
        runner.base.layers[i].to("cpu")
        gc.collect(); torch.cuda.empty_cache()
    return head(hidden)


# --------------------------------------------------------------------------- #
# aggregation / reporting
# --------------------------------------------------------------------------- #

def pct(vals: List[float], q: float) -> float:
    if not vals:
        return float("nan")
    t = torch.tensor(vals, dtype=torch.float64)
    return float(torch.quantile(t, q))


def aggregate(rows: List[Dict[str, object]], bitwidths: List[int]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    groups = sorted({str(r["group"]) for r in rows})
    for b in bitwidths:
        for kind in SCHEME_KINDS:
            sel = [r for r in rows if int(r["b"]) == b and r["kind"] == kind]
            if not sel:
                continue
            a = [float(r["act_snr_db"]) for r in sel]
            wq = [float(r["weight_sqnr_db"]) for r in sel]
            cm = [float(r["cos_min"]) for r in sel]
            lk = [float(r["act_snr_leak_db"]) for r in sel]
            rec: Dict[str, object] = {
                "n_tensors": len(sel),
                "act_snr_mean": sum(a) / len(a), "act_snr_min": min(a),
                "act_snr_p05": pct(a, 0.05),
                "w_sqnr_mean": sum(wq) / len(wq), "w_sqnr_min": min(wq),
                "cos_min_worst": min(cm), "cos_min_mean": sum(cm) / len(cm),
                "leak_mean_db": sum(lk) / len(lk), "leak_max_db": max(lk),
                "mean_total_bpw": sum(float(r["total_bpw"]) for r in sel) / len(sel),
            }
            lams = [float(r["lambda"]) for r in sel if "lambda" in r]
            if lams:
                rec["lambda_mean"] = sum(lams) / len(lams)
                rec["lambda_hist"] = {f"{v:.2f}": lams.count(v)
                                      for v in sorted(set(lams))}
            for g in groups:
                gs = [float(r["act_snr_db"]) for r in sel if r["group"] == g]
                if gs:
                    rec[f"act_snr_mean_{g}"] = sum(gs) / len(gs)
            out[f"b{b}_{kind}"] = rec
    return out


def md_tables(rows: List[Dict[str, object]], agg: Dict[str, object],
              bitwidths: List[int], tier1: Dict[int, Dict[str, float]],
              phase_b: Optional[Dict[str, object]], n_layers_done: int) -> str:
    md: List[str] = [
        "# Full-model multi-rate sweep: all weight tensors at b in {3,4,5}", "",
        f"{len(rows)} fits over {n_layers_done} layers x {len(TENSORS)} projections x "
        f"{len(bitwidths)} bitwidths x {len(SCHEME_KINDS)} schemes. Codebooks, scales and "
        "lambda are fitted on weights; outlier channels are chosen on a calibration split "
        "and every activation metric is measured on a disjoint validation split. "
        "Generated by `research/eval_full_model_sweep.py`.", "",
        "## Packing ceiling per width (Tier 1, no model involved)", "",
        "| b | unbounded gain | boxed ceiling | boundary loss | edge coords |",
        "|---|---|---|---|---|"]
    for b in bitwidths:
        d = tier1[b]
        md.append(f"| {b} | {d['unbounded_e8_gain_db']:+.3f} dB | "
                  f"**{d['boxed_e8_gain_db']:+.3f} dB** | {d['boundary_loss_db']:.3f} dB | "
                  f"{100*d['boxed_coords_on_edge']:.1f}% |")

    md += ["", "## Cross-layer aggregates (validation split)", "",
           "| b | scheme | mean bpw | Act-SNR mean | Act-SNR p05 | Act-SNR min "
           "| W-SQNR mean | worst-token cos (min) | mean lambda | leak (mean dB) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for b in bitwidths:
        for kind in SCHEME_KINDS:
            r = agg.get(f"b{b}_{kind}")
            if not r:
                continue
            lam = f"{r['lambda_mean']:.3f}" if "lambda_mean" in r else "--"
            md.append(f"| {b} | {kind} | {r['mean_total_bpw']:.4f} | "
                      f"{r['act_snr_mean']:.2f} | {r['act_snr_p05']:.2f} | "
                      f"{r['act_snr_min']:.2f} | {r['w_sqnr_mean']:.2f} | "
                      f"{r['cos_min_worst']:.6f} | {lam} | {r['leak_mean_db']:+.3f} |")

    md += ["", "## Is the b=4 trough layer-type dependent?", "",
           "Act-SNR mean by projection group, E8-Companded minus NF at the same width "
           "(positive = the lattice wins):", "",
           "| b | " + " | ".join(sorted({str(r['group']) for r in rows})) + " |",
           "|---" * (1 + len({str(r['group']) for r in rows})) + "|"]
    groups = sorted({str(r["group"]) for r in rows})
    for b in bitwidths:
        cells = []
        for g in groups:
            e8c = agg.get(f"b{b}_E8-Companded", {}).get(f"act_snr_mean_{g}")
            nf = agg.get(f"b{b}_NF", {}).get(f"act_snr_mean_{g}")
            cells.append("--" if e8c is None or nf is None
                         else f"{float(e8c) - float(nf):+.3f}")
        md.append(f"| {b} | " + " | ".join(cells) + " |")

    md += ["", "## Lambda by depth (E8-Companded)", "",
           "| b | " + " | ".join(f"layers {a}-{a+9}" for a in range(0, 40, 10)) + " |",
           "|---|---|---|---|---|"]
    for b in bitwidths:
        cells = []
        for a in range(0, 40, 10):
            lams = [float(r["lambda"]) for r in rows
                    if int(r["b"]) == b and r["kind"] == "E8-Companded"
                    and "lambda" in r and a <= int(r["layer"]) < a + 10]
            cells.append("--" if not lams else f"{sum(lams)/len(lams):.3f}")
        md.append(f"| {b} | " + " | ".join(cells) + " |")

    md += ["", "## Most vulnerable tensors at b=3 (lowest validation Act-SNR)", "",
           "| rank | layer | tensor | scheme | Act-SNR | worst-token cos | W-SQNR |",
           "|---|---|---|---|---|---|---|"]
    worst = sorted([r for r in rows if int(r["b"]) == 3
                    and r["kind"] == "E8-Companded"],
                   key=lambda r: float(r["act_snr_db"]))[:12]
    for i, r in enumerate(worst, 1):
        md.append(f"| {i} | {r['layer']} | {r['tensor']} | {r['kind']}-{r['b']} | "
                  f"{float(r['act_snr_db']):.2f} | {float(r['cos_min']):.6f} | "
                  f"{float(r['weight_sqnr_db']):.2f} |")

    if phase_b:
        md += ["", "## Phase B -- whole model quantized, real logits (validation tokens)", "",
               "| config | mean bpw | KL mean (1e-4 nats) | KL max | top-1 agreement "
               "| vs NF at same rate: sign-test z |", "|---|---|---|---|---|---|"]
        for name, r in phase_b["configs"].items():
            z = r.get("sign_test_z_vs_nf")
            md.append(f"| {name} | {float(r['mean_bpw']):.4f} | "
                      f"{float(r['kl_mean_nats'])*1e4:.2f} | "
                      f"{float(r['kl_max_nats'])*1e4:.1f} | "
                      f"{float(r['top1_agreement']):.4f} | "
                      f"{'--' if z is None else format(float(z), '+.2f')} |")
        md += ["", f"*Method floor* (fp32 head over the bf16 tail): "
               f"KL = {float(phase_b['floor']['kl_mean_nats'])*1e4:.2f}e-4 nats, "
               f"top-1 {float(phase_b['floor']['top1_agreement']):.4f}.", ""]
        if "mixed_plan_summary" in phase_b:
            md += ["Mixed-precision allocation (greedy marginal return on Phase A "
                   "measurements):", ""] + phase_b["mixed_plan_summary"]
    return "\n".join(md)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", default=e8.DEFAULT_MODEL_DIR)
    ap.add_argument("--layers", default="all",
                    help="'all', or a comma/space separated list of layer indices")
    ap.add_argument("--bitwidths", default="3,4,5")
    ap.add_argument("--outlier-frac", type=float, default=0.001)
    ap.add_argument("--calib-seqs", type=int, default=4)
    ap.add_argument("--val-seqs", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--dataset", default="Salesforce/wikitext")
    ap.add_argument("--split", default="train")
    ap.add_argument("--grid", type=int, default=40,
                    help="step-grid size. Kept at 40 (matching Parts IV-V) because coarser "
                         "grids are NOT scheme-neutral: at 24 points the companded lattice "
                         "loses 0.13 dB against NF's 0.045 dB, since it has two coupled "
                         "parameters and a coarse step grid damages the joint optimum more. "
                         "Trimming it would have biased the very comparison being made")
    ap.add_argument("--lam-scan", type=int, default=0,
                    help="coarse step-grid size for a two-stage lambda search; 0 disables "
                         "it and scores every lambda on the full grid. Left off by default "
                         "because a 12-point scan mis-ranks lambda (it picked 0.25 over the "
                         "correct 0.0 at b=4, costing 0.10 dB of weight SQNR)")
    ap.add_argument("--phase-b-grid", type=int, default=40,
                    help="step-grid size for Phase B's whole-model refits (see --grid: a "
                         "coarser grid would bias NF against the lattice)")
    ap.add_argument("--lambda-grid", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--mixed-target-bpw", type=float, default=4.0)
    ap.add_argument("--diag-samples", type=int, default=4_000_000)
    ap.add_argument("--skip-fullmodel", action="store_true")
    ap.add_argument("--phase-b-configs", nargs="+",
                    default=["NF-4", "E8-Companded-4", "mixed"],
                    help="which whole-model configs to run; 'mixed' is the "
                         "sensitivity-allocated one")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join(HERE, "full_model_report.json"))
    ap.add_argument("--md-out", default=os.path.join(HERE, "FULL_MODEL_SWEEP.md"))
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    dev = args.device
    t0 = time.time()
    bws = [int(v) for v in args.bitwidths.replace(",", " ").split()]
    grid = _alpha_grid(args.grid, device=dev)
    lam_scan = _alpha_grid(args.lam_scan, device=dev) if args.lam_scan > 0 else None
    print(f"device: {torch.cuda.get_device_name(0)}", flush=True)

    print("== Tier 1: packing ceiling per width ==", flush=True)
    tier1 = {b: boundary_diagnostic(b, n=args.diag_samples, device=dev) for b in bws}
    for b in bws:
        d = tier1[b]
        print(f"  b={b}: unbounded {d['unbounded_e8_gain_db']:+.3f} dB   "
              f"ceiling {d['boxed_e8_gain_db']:+.3f} dB   "
              f"loss {d['boundary_loss_db']:.3f} dB", flush=True)

    from transformers.models.glm import GlmForCausalLM
    print("\nloading GLM-4-9B onto host RAM (bf16, 18.9 GB)...", flush=True)
    model = GlmForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16).eval()
    n_layers = model.config.num_hidden_layers
    layers = (list(range(n_layers)) if args.layers.strip() == "all"
              else [int(v) for v in args.layers.replace(",", " ").split()])
    runner = TailRunner(model, dev)

    from capture_activations import build_token_batch
    n_seq = args.calib_seqs + args.val_seqs
    ids = build_token_batch(args.model_dir, n_seq, args.seq_len, args.dataset,
                            args.split, 0)
    print(f"  batch {tuple(ids.shape)}: {args.calib_seqs} calibration + "
          f"{args.val_seqs} validation sequences (disjoint)", flush=True)

    # ---- capture every tensor's reference input, once ----
    # capture for EVERY layer, not just the Phase A subset: Phase B quantizes the whole
    # model and needs each layer's calibration input to pick its outlier channels
    acts: Dict[int, Dict[str, torch.Tensor]] = {l: {} for l in range(n_layers)}
    handles = []
    for l in range(n_layers):
        mod_layer = model.model.layers[l]
        for path, group in CAPTURE_AT.items():
            mod = mod_layer
            for part in path.split("."):
                mod = getattr(mod, part)

            def mk(layer_idx, g):
                def hook(_m, a):
                    acts[layer_idx][g] = a[0].detach().reshape(
                        -1, a[0].shape[-1]).to("cpu", torch.bfloat16)
                return hook
            handles.append(mod.register_forward_pre_hook(mk(l, group)))
    print("reference forward (streaming all 40 layers)...", flush=True)
    hidden = runner.embed(ids)
    hidden = runner.run_layers(hidden, 0, n_layers)
    head = ResidentHead(runner, dev)
    ref_logits_full = head(hidden)
    for h in handles:
        h.remove()
    del hidden
    gc.collect(); torch.cuda.empty_cache()
    n_cal_tok = args.calib_seqs * args.seq_len
    ref_logits_val = ref_logits_full[args.calib_seqs:].clone()
    head.close()
    del ref_logits_full
    gc.collect(); torch.cuda.empty_cache()
    cap_gb = sum(v.numel() * 2 for d in acts.values() for v in d.values()) / 1e9
    print(f"  captured activations: {cap_gb:.2f} GB on host", flush=True)

    report: Dict[str, object] = {"meta": {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "model_dir": args.model_dir, "n_layers": n_layers, "layers": layers,
        "bitwidths": bws, "outlier_frac": args.outlier_frac,
        "calib_tokens": n_cal_tok, "val_tokens": args.val_seqs * args.seq_len,
        "lambda_grid": args.lambda_grid, "grid_points": args.grid,
        "lam_scan_points": args.lam_scan,
        "leakage_control": ("outlier channels selected on the calibration split; all "
                            "activation and KL metrics reported on the validation split"),
    }, "tier1_boundary": {str(b): tier1[b] for b in bws}}

    print(f"\n== Phase A: {len(layers)*len(TENSORS)*len(bws)*len(SCHEME_KINDS)} fits "
          f"over {len(layers)} layers ==", flush=True)
    rows = phase_a(args.model_dir, layers, bws, acts, n_cal_tok, grid,
                   args.lambda_grid, lam_scan, args.outlier_frac, dev,
                   checkpoint=args.out + ".partial")
    report["per_tensor"] = rows
    agg = aggregate(rows, bws)
    report["aggregate"] = agg
    print("\n  cross-layer means (validation split):", flush=True)
    for b in bws:
        line = [f"    b={b}:"]
        for kind in SCHEME_KINDS:
            r = agg.get(f"b{b}_{kind}")
            if r:
                line.append(f"{kind}={r['act_snr_mean']:.2f}")
        print("  ".join(line), flush=True)

    phase_b: Optional[Dict[str, object]] = None
    if not args.skip_fullmodel and layers != list(range(n_layers)):
        print("\n== Phase B skipped: it quantizes the whole model, so it needs "
              "--layers all ==", flush=True)
    elif not args.skip_fullmodel:
        print("\n== Phase B: whole-model quantized forwards ==", flush=True)
        head = ResidentHead(runner, dev)
        grid_b = _alpha_grid(args.phase_b_grid, device=dev)
        configs: List[Tuple[str, Dict[int, Dict[str, Tuple[str, int]]], float]] = []
        wanted = set(args.phase_b_configs)
        for b in bws:
            for kind in ("NF", "E8-Companded"):
                if f"{kind}-{b}" not in wanted:
                    continue
                plan = {l: {t: (kind, b) for t, _ in TENSORS} for l in range(n_layers)}
                bpw = sum(float(r["total_bpw"]) * int(r["params"]) for r in rows
                          if int(r["b"]) == b and r["kind"] == kind) / max(
                    sum(int(r["params"]) for r in rows
                        if int(r["b"]) == b and r["kind"] == kind), 1)
                configs.append((f"{kind}-{b} (uniform)", plan, bpw))
        # sensitivity-allocated mixed precision, using Phase A's own measurements
        kind_at = {3: "E8-Companded", 4: "NF", 5: "E8-Companded"}
        kind_at = {b: kind_at.get(b, "NF") for b in bws}
        bits, achieved = allocate_mixed(rows, bws, args.mixed_target_bpw, kind_at)
        plan_mixed: Dict[int, Dict[str, Tuple[str, int]]] = {
            l: {t: (kind_at[bits[(l, t)]], bits[(l, t)]) for t, _ in TENSORS}
            for l in range(n_layers)}
        hist: Dict[int, int] = {}
        for v in bits.values():
            hist[v] = hist.get(v, 0) + 1
        print(f"  mixed plan: {hist}, achieved {achieved:.3f} bpw payload", flush=True)
        if "mixed" in wanted:
            configs.append((f"Mixed-{args.mixed_target_bpw:.1f}bpw", plan_mixed, achieved))

        results: Dict[str, object] = {}
        kvecs: Dict[str, torch.Tensor] = {}
        for name, plan, bpw in configs:
            ts = time.time()
            lg = full_model_logits(runner, head, ids, n_layers, plan, args.model_dir,
                                   acts, n_cal_tok, grid_b, args.lambda_grid, lam_scan,
                                   args.outlier_frac, dev)
            k, ag = kl_per_token(ref_logits_val, lg[args.calib_seqs:])
            del lg
            kvecs[name] = k
            results[name] = {"mean_bpw": bpw, **kl_summary(k, ag)}
            print(f"    {name:<26} bpw={bpw:.4f} KL={float(k.mean()):.3e} "
                  f"top1={ag:.4f}  ({time.time()-ts:.0f}s)", flush=True)
            gc.collect(); torch.cuda.empty_cache()
        for b in bws:
            a, c = f"NF-{b} (uniform)", f"E8-Companded-{b} (uniform)"
            if a in kvecs and c in kvecs:
                p = paired_from_kl(kvecs[a], kvecs[c])
                results[c]["sign_test_z_vs_nf"] = p["sign_test_z"]
                results[c]["paired_vs_nf"] = p
        mix = f"Mixed-{args.mixed_target_bpw:.1f}bpw"
        if mix in kvecs and "NF-4 (uniform)" in kvecs:
            p = paired_from_kl(kvecs["NF-4 (uniform)"], kvecs[mix])
            results[mix]["sign_test_z_vs_nf"] = p["sign_test_z"]
            results[mix]["paired_vs_nf4"] = p
        # Unlike Parts III-V, this phase has NO reconstruction floor: both the reference and
        # every quantized run go through the identical streamed forward, so the unquantized
        # path compared with itself is exactly zero and the whole KL is attributable to
        # quantization. That is the one methodological gain of doing it at full-model scale.
        phase_b = {"configs": results,
                   "floor": {"kl_mean_nats": 0.0, "top1_agreement": 1.0,
                             "note": "exactly zero by construction: no h1+Yhat splice, "
                                     "both paths are the same real forward"},
                   "mixed_bits_hist": {str(k): v for k, v in hist.items()},
                   "mixed_plan_summary": [
                       f"- {v} tensors at b={k} ({kind_at[k]})" for k, v in
                       sorted(hist.items())]}
        report["phase_b"] = phase_b
        head.close()

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    md = md_tables(rows, agg, bws, tier1, phase_b, len(layers))
    with open(args.md_out, "w", encoding="utf-8") as fh:
        fh.write(md + "\n")
    print(f"\nwrote {args.out} and {args.md_out} in {(time.time()-t0)/60:.1f} min")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    print("\n" + md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
