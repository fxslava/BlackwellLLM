"""Render a LongBench run's results as markdown, and diff two runs per sample.

    python benchmarks/report_tables.py benchmarks/results/e8w5_baseline
    python benchmarks/report_tables.py <new-run> --against benchmarks/results/e8w5_baseline

The diff mode is the point of keeping per-sample predictions: after a Hessian or
Hadamard pass, "the task average moved by 1.2" is not evidence on 50 samples --
the standard error is larger than that. What *is* evidence is which specific
samples changed and in which direction, so `--against` pairs samples by (task,
_id) and reports the per-sample deltas, the win/loss split, and a paired
t-statistic over the samples both runs scored.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def load(run_dir: Path) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    with open(run_dir / "predictions.jsonl", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows[(r["task"], str(r["_id"]))] = r
    return summary, rows


def results_table(summary: dict[str, Any]) -> str:
    out = ["| task | metric | n | score | +/- stderr | mean prompt tok | truncated | s/sample |",
           "|---|---|---:|---:|---:|---:|---:|---:|"]
    for task, t in summary["tasks"].items():
        if not t.get("n"):
            out.append(f"| {task} | — | 0 | — | — | — | — | — |")
            continue
        se = f"{t['score_stderr']:.2f}" if t.get("score_stderr") is not None else "—"
        out.append(f"| `{task}` | {t['metric']} | {t['n']} | **{t['score']:.2f}** | {se} "
                   f"| {t['prompt_tokens_mean']:.0f} | {100 * t['truncated_frac']:.0f}% "
                   f"| {t['wall_s_mean']:.1f} |")
    return "\n".join(out)


def perf_table(summary: dict[str, Any]) -> str:
    out = ["| task | prefill ms/token | mean prefill | decode ms/token | stop reasons |",
           "|---|---:|---:|---:|---|"]
    for task, t in summary["tasks"].items():
        if not t.get("n"):
            continue
        dec = f"{t['decode_ms_per_token']:.1f}" if t.get("decode_ms_per_token") else "—"
        stops = ", ".join(f"{k}={v}" for k, v in t["stop_reasons"].items())
        out.append(f"| `{task}` | {t['prefill_ms_per_token']:.2f} "
                   f"| {t['prefill_ms_mean'] / 1000:.1f} s | {dec} | {stops} |")
    return "\n".join(out)


def secondary_table(summary: dict[str, Any]) -> str:
    lines = ["| task | secondary metric | value |", "|---|---|---:|"]
    any_row = False
    for task, t in summary["tasks"].items():
        for name, val in (t.get("secondary") or {}).items():
            lines.append(f"| `{task}` | {name} | {val:.2f} |")
            any_row = True
    return "\n".join(lines) if any_row else "_(no secondary metrics recorded)_"


def diff(new_dir: Path, base_dir: Path) -> str:
    new_sum, new_rows = load(new_dir)
    base_sum, base_rows = load(base_dir)
    shared = sorted(set(new_rows) & set(base_rows))
    if not shared:
        return "No samples in common — are these runs of the same task subset?"

    out = [f"Comparing **{new_dir.name}** against **{base_dir.name}** "
           f"({len(shared)} samples in common)\n",
           "| task | n | base | new | delta | better | worse | same | paired t |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for task in sorted({t for t, _ in shared}):
        keys = [k for k in shared if k[0] == task]
        deltas = [new_rows[k]["score"] - base_rows[k]["score"] for k in keys]
        base_mean = 100 * statistics.fmean(base_rows[k]["score"] for k in keys)
        new_mean = 100 * statistics.fmean(new_rows[k]["score"] for k in keys)
        better = sum(1 for d in deltas if d > 1e-9)
        worse = sum(1 for d in deltas if d < -1e-9)
        same = len(deltas) - better - worse
        # Paired t over the per-sample deltas: the only honest significance test
        # here, because the two runs saw the identical samples.
        if len(deltas) > 1 and statistics.pstdev(deltas) > 0:
            sd = statistics.stdev(deltas)
            t_stat = statistics.fmean(deltas) / (sd / len(deltas) ** 0.5) if sd else 0.0
            t_txt = f"{t_stat:+.2f}"
        else:
            t_txt = "—"
        out.append(f"| `{task}` | {len(keys)} | {base_mean:.2f} | {new_mean:.2f} "
                   f"| {new_mean - base_mean:+.2f} | {better} | {worse} | {same} | {t_txt} |")

    out.append("\nA |t| below ~2.0 means the two runs are statistically "
               "indistinguishable on this sample count, whatever the averages say.\n")
    out.append("### Largest per-sample regressions\n")
    ranked = sorted(shared, key=lambda k: new_rows[k]["score"] - base_rows[k]["score"])
    for k in ranked[:8]:
        d = new_rows[k]["score"] - base_rows[k]["score"]
        if d >= -1e-9:
            break
        out.append(f"- `{k[0]}` `{k[1]}` {base_rows[k]['score']:.3f} -> "
                   f"{new_rows[k]['score']:.3f} ({d:+.3f})")
        out.append(f"  - base: `{base_rows[k]['prediction'][:140]!r}`")
        out.append(f"  - new : `{new_rows[k]['prediction'][:140]!r}`")
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--against", default=None, help="baseline run dir to diff against")
    args = p.parse_args()
    run_dir = Path(args.run_dir)

    if args.against:
        print(diff(run_dir, Path(args.against)))
        return 0

    summary, rows = load(run_dir)
    cfg, eng, vram = summary["config"], summary["engine"], summary["vram"]
    print(f"### Configuration\n")
    print(f"- model `{cfg['model']}`, ctx {cfg['ctx']}, input budget "
          f"{cfg['max_input_tokens']} tokens, {cfg['sampling']}")
    # .get(): a summary written before a field existed must still render, or the
    # baseline stops being readable the moment the harness grows a knob.
    print(f"- engine path `{cfg.get('kv_mode', '?')}/{cfg.get('prefill', '?')}`, "
          f"gpu_layers {cfg.get('gpu_layers') if cfg.get('gpu_layers') is not None else 'all'}")
    print(f"- weights {eng['weights_gb']} GB, KV/dynamic {eng['kv_dynamic_mb']} MB, "
          f"engine restarts {eng['restarts']}")
    print(f"- GPU total used {vram['baseline_mb']} MB -> {vram['peak_total_mb']} MB peak "
          f"(delta {vram['peak_over_baseline_mb']} MB)")
    print(f"- wall clock {summary['wall_clock_s'] / 60:.1f} min\n")
    print("### Results\n")
    print(results_table(summary))
    print("\n### Latency\n")
    print(perf_table(summary))
    print("\n### Secondary metrics\n")
    print(secondary_table(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
