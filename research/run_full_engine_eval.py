"""Master orchestration -- the full E8 / QJL / nvCOMP evaluation, into bench_report.json.

Runs, in order:

  1. the activation capture (Module A) if its artifact is missing;
  2. every quantization scheme against layer 10's ``down_proj``, scoring each on both
     weight SQNR and real Output Act-SNR (Module B);
  3. the E8 rate sweep from 3.0 to 4.5 bpw, under all five rate definitions;
  4. the QJL hybrid, plus the m-sweep and projection comparison that test whether
     0.5 bpw of sketch beats 0.5 bpw of extra lattice density (Module C);
  5. nvCOMP v5 decompression on the payloads the kernel would actually stream
     (Module D);
  6. a cross-layer replication on layer 3, so no conclusion rests on one layer.

USAGE
    python research/run_full_engine_eval.py
    python research/run_full_engine_eval.py --skip-nvcomp --layers 10
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from typing import Dict, List, Optional

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import e8_lattice_engine as e8  # noqa: E402
import qjl_residual as qjl      # noqa: E402

ARTIFACT_DIR = os.path.join(HERE, "artifacts")
PRIMARY_RATE_KEY = "bpw_coord_entropy_parity"

# The reference numbers this study is asked to verify, from the prior offline
# weight-only analysis. Kept here so the report can state agreement or disagreement
# rather than quietly restating them as if they were freshly measured.
PRIOR_CLAIMS = {
    "RTN INT4":          {"weight_sqnr_db": 14.36, "weight_linf": 0.0508, "total_bpw": 4.001},
    "Lloyd-Max 16-LUT":  {"weight_sqnr_db": 19.65, "weight_linf": 0.5776, "total_bpw": 4.001},
    "Pure E8 @3.5bpw":   {"weight_sqnr_db": 20.203, "weight_linf": 0.0137, "total_bpw": 3.501},
}


def ensure_activations(layers: List[int], model_dir: str, num_seqs: int,
                       seq_len: int) -> Dict[int, dict]:
    """Load the capture artifacts, running Module A for any that are missing."""
    missing = [l for l in layers
               if not os.path.exists(os.path.join(ARTIFACT_DIR, f"calib_act_layer{l}.pt"))]
    if missing:
        print(f"capture artifacts missing for layers {missing}; running Module A")
        cmd = [sys.executable, os.path.join(HERE, "capture_activations.py"),
               "--model-dir", model_dir, "--layers", *[str(l) for l in layers],
               "--num-seqs", str(num_seqs), "--seq-len", str(seq_len)]
        subprocess.run(cmd, check=True)
    out = {}
    for l in layers:
        out[l] = torch.load(os.path.join(ARTIFACT_DIR, f"calib_act_layer{l}.pt"),
                            weights_only=False)
    return out


def build_schemes(w: torch.Tensor, want_payloads: bool = True) -> List[e8.Quantized]:
    """Every scheme in the evaluation matrix, all on the same per-row FP16 scale."""
    schemes: List[e8.Quantized] = []

    t = time.time()
    q = e8.quantize_rtn_int4(w)
    schemes.append(q)
    q = e8.quantize_rtn_int4_clipped(w)
    schemes.append(q)
    q = e8.quantize_lloyd_max(w)
    schemes.append(q)
    print(f"  scalar baselines: {time.time() - t:.1f}s")

    for target in (3.5, 4.0):
        t = time.time()
        alpha, q = e8.fit_e8_scale_for_rate(w, target, rate_key=PRIMARY_RATE_KEY)
        # re-run once with the payload so the nvCOMP module gets real bytes
        if want_payloads:
            q2 = e8.quantize_e8(w, alpha, want_rates=True, want_payload=True)
            q2.name = q.name
            q = q2
        q.name = f"{'Pure' if target == 3.5 else 'Dense'} E8 @{target:.1f}bpw"
        schemes.append(q)
        print(f"  {q.name}: alpha={alpha:.4f} ({time.time() - t:.1f}s)")

    t = time.time()
    q = e8.quantize_e8_fixed_width(w, coord_bits=4)
    q.name = "E8 fixed-width 4b"
    schemes.append(q)
    print(f"  {q.name} ({time.time() - t:.1f}s)")
    return schemes


def score(schemes: List[e8.Quantized], x: torch.Tensor,
          w: torch.Tensor) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    base = e8.bf16_reference_row()
    rows.append({"scheme": "BF16 baseline", "payload_bpw": 16.0, "overhead_bpw": 0.0,
                 "total_bpw": 16.0, "bpw_fixed_random_access": 16.0, **base})
    for q in schemes:
        wm = e8.weight_metrics(w, q.w_hat)
        am = e8.act_metrics(x, w, q.w_hat)
        row: Dict[str, object] = {
            "scheme": q.name,
            "payload_bpw": q.payload_bpw,
            "overhead_bpw": q.overhead_bpw,
            "total_bpw": q.total_bpw,
            "bpw_fixed_random_access": q.rates.get("bpw_fixed_coords"),
            "bpw_payload_as_packed": q.extra.get("payload_bpw_fixed"),
            **wm, **am,
        }
        # bias is only interpretable against the scale of the output it biases
        row["act_bias_over_ref_rms"] = (am["act_mean_bias"] / am["act_ref_rms"]
                                        if am["act_ref_rms"] else float("nan"))
        if q.rates:
            row["rates"] = {k: v for k, v in q.rates.items()}
        if q.extra:
            row["extra"] = {k: v for k, v in q.extra.items()
                            if k != "centroids" and not isinstance(v, bytes)}
        if q.name in PRIOR_CLAIMS:
            c = PRIOR_CLAIMS[q.name]
            row["prior_claim"] = c
            row["prior_sqnr_delta_db"] = wm["weight_sqnr_db"] - c["weight_sqnr_db"]
        rows.append(row)
    return rows


def rate_sweep(w: torch.Tensor, lo: float = 3.0, hi: float = 4.5,
               steps: int = 7) -> List[Dict[str, float]]:
    """SQNR versus rate for E8, under every rate definition at once.

    The same quantized tensor has five defensible bit rates; tabulating them side by
    side is what exposes that the headline 3.5 bpw and the kernel-addressable rate are
    not the same number.
    """
    out = []
    for target in [lo + (hi - lo) * i / (steps - 1) for i in range(steps)]:
        alpha, q = e8.fit_e8_scale_for_rate(w, target, rate_key=PRIMARY_RATE_KEY)
        m = e8.weight_metrics(w, q.w_hat)
        rec = {"target_bpw": target, "alpha": alpha,
               "weight_sqnr_db": m["weight_sqnr_db"], "weight_linf": m["weight_linf"]}
        rec.update({k: v for k, v in q.rates.items()})
        out.append(rec)
        del q
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def qjl_block(w: torch.Tensor, w_hat_35: torch.Tensor, x: torch.Tensor,
              block: int, m_values: List[int],
              projections: List[str]) -> Dict[str, object]:
    residual = w.to(torch.float32) - w_hat_35.to(torch.float32)
    hom = qjl.residual_homoscedasticity(residual, block)
    del residual
    out: Dict[str, object] = {"block": block, "homoscedasticity": hom,
                              "theory": {str(m): qjl.qjl_theory(m, block) for m in m_values}}
    for proj in projections:
        print(f"    projection={proj} m={m_values}")
        out[f"sweep_{proj}"] = qjl.sweep_m(
            x, w, w_hat_35, m_values, block=block, kind=proj,
            norm_mode="row", act_metrics_fn=e8.act_metrics)
    # the spec's exact configuration, with both norm accountings
    out["spec_config"] = []
    for norm_mode in ("row", "block"):
        r = qjl.sweep_m(x, w, w_hat_35, [block // 2], block=block, kind="rademacher",
                        norm_mode=norm_mode, act_metrics_fn=e8.act_metrics)[0]
        out["spec_config"].append(r)
    return out


def layer_params(model_dir: str) -> Dict[str, int]:
    """Parameter counts for one GLM-4 decoder layer, read from the checkpoint config.

    down_proj is the tensor under study, but it is only part of a layer, and a prefetch
    has to cover the whole layer inside the same compute window. Extrapolating from one
    tensor to the layer is the difference between a codec that "fits the 0.46 ms budget"
    and one that does not.
    """
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as fh:
        c = json.load(fh)
    h, i = c["hidden_size"], c["intermediate_size"]
    kv = c["num_key_value_heads"] * c["head_dim"]
    parts = {
        "q_proj": h * h, "k_proj": h * kv, "v_proj": h * kv, "o_proj": h * h,
        "gate_up_proj": h * 2 * i, "down_proj": i * h,
    }
    parts["total"] = sum(parts.values())
    parts["num_hidden_layers"] = c["num_hidden_layers"]
    return parts


def layer_budget_analysis(nvcomp_rows: List[Dict[str, object]], model_dir: str,
                          n_weights: int, budget_ms: float,
                          dram_gbps: float) -> Dict[str, object]:
    """Does a whole layer's decompression fit the compute window, not just one tensor?"""
    lp = layer_params(model_dir)
    frac = n_weights / lp["total"]
    per: List[Dict[str, object]] = []
    for r in nvcomp_rows:
        if "error" in r or not r.get("fits_layer_budget"):
            continue
        # scale this tensor's measured decompress time up to the whole layer
        layer_ms = float(r["decomp_ms_mean"]) / frac
        per.append({
            "payload": r["payload"], "algorithm": r["algorithm"],
            "decomp_gbps": r["decomp_gbps"], "effective_bpw": r.get("effective_bpw"),
            "tensor_decomp_ms": r["decomp_ms_mean"],
            "layer_decomp_ms": layer_ms,
            "layer_fits": bool(layer_ms <= budget_ms),
            "layer_over_factor": layer_ms / budget_ms,
        })
    per.sort(key=lambda d: d["layer_decomp_ms"])
    # The window itself shrinks once weights are quantized: the layer's own read is
    # smaller, so there is LESS compute time to hide a prefetch behind. ``dram_gbps``
    # must be the peak one-directional figure -- a d2d copy moves every byte twice, so
    # its per-direction rate is half the bandwidth a read-only GEMV stream can use.
    quantized_read_ms = {
        f"{bpw:.3f}bpw": lp["total"] * bpw / 8 / 1e9 / (dram_gbps / 1e3)
        for bpw in (16.0, 8.125, 4.0, 3.761)
    }
    return {
        "layer_params": lp["total"], "down_proj_params": n_weights,
        "down_proj_fraction": frac, "num_hidden_layers": lp["num_hidden_layers"],
        "budget_ms": budget_ms,
        "dram_gbps_peak_estimate": dram_gbps,
        "dram_gbps_note": ("peak = a d2d copy's read+write traffic over its time; the "
                           "per-direction copy rate is half this and would understate "
                           "a read-only weight stream by 2x"),
        "layer_read_ms_at_peak_bw": quantized_read_ms,
        "per_codec": per,
    }


def fmt(v: object, spec: str = ".3f", width: int = 0) -> str:
    if v is None:
        return "n/a".rjust(width)
    if isinstance(v, float):
        if math.isinf(v):
            return "inf".rjust(width)
        if math.isnan(v):
            return "n/a".rjust(width)
        return format(v, spec).rjust(width)
    return str(v).rjust(width)


def print_matrix(rows: List[Dict[str, object]], nvcomp_best: Dict[str, float]) -> None:
    hdr = (f"{'Scheme':<22} {'Payload':>8} {'Ovh':>7} {'Total':>7} {'FixedRA':>8} "
           f"{'W-SQNR':>8} {'Act-SNR':>8} {'MeanBias':>11} {'bias t':>7} "
           f"{'W-Linf':>8} {'nvCOMP':>9}")
    print("\n" + "=" * len(hdr))
    print("EVALUATION MATRIX -- GLM-4-9B layer 10 down_proj [4096, 13696]")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        nv = nvcomp_best.get(str(r["scheme"]))
        print(f"{str(r['scheme']):<22} "
              f"{fmt(r['payload_bpw'], '.3f', 8)} {fmt(r['overhead_bpw'], '.4f', 7)} "
              f"{fmt(r['total_bpw'], '.3f', 7)} "
              f"{fmt(r.get('bpw_fixed_random_access'), '.3f', 8)} "
              f"{fmt(r['weight_sqnr_db'], '.2f', 8)} {fmt(r['act_snr_db'], '.2f', 8)} "
              f"{fmt(r['act_mean_bias'], '+.3e', 11)} "
              f"{fmt(r.get('act_bias_t_stat'), '+.2f', 7)} "
              f"{fmt(r['weight_linf'], '.4f', 8)} "
              f"{fmt(nv, '.1f', 9)}")
    print("-" * len(hdr))
    print("Payload/Ovh/Total/FixedRA in bits per weight. W-SQNR, Act-SNR in dB.")
    print("FixedRA = the fixed-width, randomly-addressable rate a GEMV kernel needs;")
    print("Total uses the entropy-coded rate, which requires sequential decode.")
    print("nvCOMP = best decompress GB/s over the tested codecs for that payload.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", default=e8.DEFAULT_MODEL_DIR)
    ap.add_argument("--layers", type=int, nargs="+", default=[10, 3],
                    help="first layer is primary; the rest are replication checks")
    ap.add_argument("--num-seqs", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--qjl-block", type=int, default=64)
    ap.add_argument("--qjl-m", type=int, nargs="+", default=[8, 16, 32, 64, 96, 128])
    ap.add_argument("--qjl-projections", nargs="+",
                    default=["rademacher", "gaussian", "fwht"])
    ap.add_argument("--nvcomp-codecs", nargs="+", default=None)
    ap.add_argument("--nvcomp-iters", type=int, default=50)
    ap.add_argument("--budget-ms", type=float, default=0.46)
    ap.add_argument("--skip-nvcomp", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=os.path.join(HERE, "bench_report.json"))
    args = ap.parse_args()

    dev = args.device
    primary = args.layers[0]
    t0 = time.time()

    report: Dict[str, object] = {
        "meta": {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "host": platform.node(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__,
            "python": sys.version.split()[0],
            "model_dir": args.model_dir,
            "primary_layer": primary,
            "primary_rate_key": PRIMARY_RATE_KEY,
            "budget_ms": args.budget_ms,
        }
    }
    try:
        from nvidia import nvcomp as _nv
        report["meta"]["nvcomp"] = _nv.__version__
    except Exception:
        report["meta"]["nvcomp"] = None

    print("== Module A: calibration activations ==")
    caps = ensure_activations(args.layers, args.model_dir, args.num_seqs, args.seq_len)
    report["calibration"] = {
        str(l): {k: v for k, v in blob.items() if k != "x"} for l, blob in caps.items()}
    x = caps[primary]["x"].to(dev)
    print(f"  layer {primary}: X {tuple(x.shape)}  "
          f"rms={caps[primary]['x_rms']:.4f} kurtosis={caps[primary]['x_kurtosis']:.0f}")

    print(f"\n== Module B: quantizing layer {primary} down_proj ==")
    w = e8.load_down_proj(args.model_dir, primary, device=dev)
    wf = w.to(torch.float32)
    report["weight_stats"] = {
        "shape": list(w.shape), "params": int(w.numel()), "dtype": str(w.dtype),
        "rms": float(wf.pow(2).mean().sqrt()), "absmax": float(wf.abs().max()),
        "kurtosis": float(wf.pow(4).mean() / wf.pow(2).mean() ** 2),
        "absmax_over_rms": float(wf.abs().max() / wf.pow(2).mean().sqrt()),
    }
    print(f"  W {tuple(w.shape)} rms={report['weight_stats']['rms']:.5f} "
          f"kurtosis={report['weight_stats']['kurtosis']:.2f} "
          f"absmax/rms={report['weight_stats']['absmax_over_rms']:.1f}")
    del wf

    schemes = build_schemes(w)
    rows = score(schemes, x, w)

    print(f"\n== Module B: E8 rate sweep 3.0 -> 4.5 bpw ==")
    report["rate_sweep"] = rate_sweep(w)
    for r in report["rate_sweep"]:
        print(f"  target={r['target_bpw']:.2f}  parity={r[PRIMARY_RATE_KEY]:.3f}  "
              f"coordH={r['bpw_coord_entropy']:.3f}  fixedRA={r['bpw_fixed_coords']:.3f}  "
              f"SQNR={r['weight_sqnr_db']:.2f}dB  Linf={r['weight_linf']:.4f}")

    print(f"\n== Module C: QJL residual sketch (block={args.qjl_block}) ==")
    w_hat_35 = next(q for q in schemes if q.name.startswith("Pure E8")).w_hat
    report["qjl"] = qjl_block(w, w_hat_35, x, args.qjl_block, args.qjl_m,
                              args.qjl_projections)
    spec_row = report["qjl"]["spec_config"][0]
    hybrid_base = next(r for r in rows if str(r["scheme"]).startswith("Pure E8"))
    rows.append({
        "scheme": "Hybrid E8+QJL",
        "payload_bpw": hybrid_base["payload_bpw"],
        "overhead_bpw": float(hybrid_base["overhead_bpw"]) + spec_row["qjl_total_bpw"],
        "total_bpw": float(hybrid_base["total_bpw"]) + spec_row["qjl_total_bpw"],
        "bpw_fixed_random_access": (float(hybrid_base["bpw_fixed_random_access"])
                                   + spec_row["qjl_total_bpw"]),
        "weight_sqnr_db": float("nan"),     # the sketch corrects Y, not W
        "weight_linf": float("nan"),
        "weight_rmse": float("nan"),
        "act_snr_db": spec_row["act_snr_db"],
        "act_mean_bias": spec_row["act_mean_bias"],
        "act_mean_bias_se": spec_row["act_mean_bias_se"],
        "act_bias_t_stat": (spec_row["act_mean_bias"] / spec_row["act_mean_bias_se"]
                            if spec_row["act_mean_bias_se"] else float("nan")),
        "act_linf": spec_row["act_linf"],
        "act_channel_bias_rms": spec_row["act_channel_bias_rms"],
        "qjl": {k: spec_row[k] for k in ("m", "sign_bpw", "norm_bpw", "norm_mode",
                                        "act_snr_delta_db",
                                        "predicted_act_snr_delta_db")},
    })
    for proj in args.qjl_projections:
        print(f"  {proj}:")
        for r in report["qjl"][f"sweep_{proj}"]:
            if "skipped" in r:
                print(f"    m={r['m']:4d} skipped: {r['skipped']}")
                continue
            print(f"    m={r['m']:4d} ({r['sign_bpw']:.3f} bpw)  "
                  f"Act-SNR={r['act_snr_db']:7.3f} dB  "
                  f"delta={r['act_snr_delta_db']:+7.3f} dB  "
                  f"predicted={r['predicted_act_snr_delta_db']:+7.3f} dB")

    nvcomp_best: Dict[str, float] = {}
    if not args.skip_nvcomp:
        print(f"\n== Module D: nvCOMP v5 decompression ==")
        import nvcomp_stream_bench as nb
        payloads: Dict[str, bytes] = {}
        for q in schemes:
            if q.payload_bytes:
                payloads[q.name] = q.payload_bytes
            alt = q.extra.get("payload_bytealigned")
            if isinstance(alt, bytes):
                # the same quantization, byte-aligned: raw size is larger but an
                # entropy coder can actually reach the entropy rate through it
                payloads[f"{q.name} [byte-aligned]"] = alt
        report["nvcomp"] = nb.bench_payloads(
            payloads, args.nvcomp_codecs, args.nvcomp_iters,
            budget_ms=args.budget_ms, device=dev, n_weights=int(w.numel()))
        for r in report["nvcomp"]:
            if "error" in r:
                continue
            k = str(r["payload"])
            nvcomp_best[k] = max(nvcomp_best.get(k, 0.0), float(r["decomp_gbps"]))
        # DRAM-bandwidth reference on a buffer far too big to sit in L2
        report["nvcomp_dram_reference"] = nb.measure_d2d_bandwidth(int(512e6), device=dev)
        print(f"\n  512 MB d2d reference: "
              f"{report['nvcomp_dram_reference']['copy_read_gbps']:.0f} GB/s read "
              f"(the 24-57 MB figures above are partly L2-resident and so optimistic)")
        report["budget_analysis"] = layer_budget_analysis(
            report["nvcomp"], args.model_dir, int(w.numel()), args.budget_ms,
            report["nvcomp_dram_reference"]["copy_rw_gbps"])
        ba = report["budget_analysis"]
        print(f"\n  whole-layer extrapolation ({ba['layer_params']/1e6:.0f} M params/layer, "
              f"down_proj is {100*ba['down_proj_fraction']:.0f}% of it):")
        print(f"    measured peak bandwidth {ba['dram_gbps_peak_estimate']:.0f} GB/s; "
              f"one layer's own weight read: "
              + ", ".join(f"{k}={v:.3f} ms"
                          for k, v in ba["layer_read_ms_at_peak_bw"].items()))
        for r in ba["per_codec"]:
            print(f"    {r['payload']:<34} {r['algorithm']:<9} "
                  f"{r['decomp_gbps']:6.1f} GB/s -> layer {r['layer_decomp_ms']:6.3f} ms "
                  f"vs {args.budget_ms} ms budget: "
                  f"{'FITS' if r['layer_fits'] else 'OVER by %.1fx' % r['layer_over_factor']}")

    print(f"\n== Cross-layer replication ==")
    report["replication"] = {}
    for l in args.layers[1:]:
        wl = e8.load_down_proj(args.model_dir, l, device=dev)
        xl = caps[l]["x"].to(dev)
        _, q35 = e8.fit_e8_scale_for_rate(wl, 3.5, rate_key=PRIMARY_RATE_KEY)
        qlm = e8.quantize_lloyd_max(wl)
        rep = []
        for q in (q35, qlm):
            rep.append({"scheme": q.name, "total_bpw": q.total_bpw,
                        **e8.weight_metrics(wl, q.w_hat),
                        **e8.act_metrics(xl, wl, q.w_hat)})
        sk = qjl.qjl_encode(wl.to(torch.float32) - q35.w_hat, args.qjl_block // 2,
                            args.qjl_block, "rademacher", 0, "row")
        yh = qjl._assemble(xl, q35.w_hat, qjl.qjl_correction(xl, sk))
        rep.append({"scheme": "Hybrid E8+QJL", "total_bpw": q35.total_bpw + sk.total_bpw,
                    **e8.act_metrics(xl, wl, None, y_hat_override=yh)})
        report["replication"][str(l)] = rep
        for r in rep:
            print(f"  layer {l:2d} {r['scheme']:<22} "
                  f"Act-SNR={r['act_snr_db']:7.3f} dB  "
                  f"W-SQNR={fmt(r.get('weight_sqnr_db'), '.2f')} dB")
        del wl, xl, q35, qlm, sk, yh
        torch.cuda.empty_cache()

    report["matrix"] = rows
    print_matrix(rows, nvcomp_best)

    report["meta"]["wall_seconds"] = round(time.time() - t0, 1)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"\nwrote {args.out} ({os.path.getsize(args.out) / 1e3:.1f} kB) "
          f"in {report['meta']['wall_seconds']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
