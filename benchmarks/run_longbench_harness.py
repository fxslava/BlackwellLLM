"""Automated LongBench harness for BlackwellLLM.

Drives `blackwell_llm.exe --serve` over a stdin/stdout pipe, scores the
predictions with `metrics.py`, and records the latency / VRAM profile that makes
a quantization A/B meaningful.

    python benchmarks/run_longbench_harness.py \
        --model F:/AI/models/GLM-4-9B-e8w5 \
        --tasks qasper,multifieldqa_en,lcc,trec \
        --limit 50 \
        --max-input-tokens 2000 \
        --output-dir benchmarks/results/e8w5_baseline

Protocol fidelity. The four task prompts, their `max_gen`, the middle-truncation
rule and the decision of which tasks get a chat template are taken from the
official LongBench `pred.py` / `config/*.json`; the scorers live in `metrics.py`.
Truncation itself happens inside the engine, against the engine's own tokenizer
(`--serve` takes a `max_input_tokens` budget), so no second tokenizer can
disagree with the one doing inference.

**Read `--max-input-tokens` before quoting any score from this harness.** Cost per
prompt grows super-linearly with context on this engine (batch=1 GEMV projections
plus an attention term that grows with the KV length). Measured on GLM-4-9B E8W5
/ RTX 5070, x64-Release:

    --kv-mode paged --prefill fast   (default)
        prefill_ms(N) = 15.61*N + 6.26e-4*N^2        N = 1983..9142, ctx 16384
    --kv-mode continuous --prefill loop  (one Forward() per token)
        prefill_ms(N) = 17.2*N  + 6.43e-3*N^2        N = 348..2550

The default path is 4.7x faster at N = 16000 (6.8 min vs 32 min per prompt): it
skips the lm_head for every prompt token whose logits are discarded, and tiles the
prompt through the engine's batched chunk path, which cuts the quadratic
attention coefficient by 21x. It is NOT free of the quadratic term -- budget
accordingly.

A run truncated to fit a time budget is NOT comparable to a published LongBench
number for the same task; the summary records the budget for exactly that reason.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics  # noqa: E402  (same-directory module, kept importable without a package)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXE = REPO_ROOT / "out/build/x64-Release/src/apps/Release/blackwell_llm.exe"
DEFAULT_MODEL = "F:/AI/models/GLM-4-9B-e8w5"
HF_DATASET = "zai-org/LongBench"        # THUDM/LongBench redirects here

# Measured on GLM-4-9B E8W5, RTX 5070, x64-Release (docs/LONGBENCH_E8W5_BASELINE.md).
# Used only to size per-request timeouts and to project a run's cost before it
# starts -- never as a substitute for a measurement. Keyed by the engine path,
# because the two differ by ~4.7x at 16k tokens.
#   (linear ms/token, quadratic ms/token^2)
PREFILL_COST = {
    ("paged", "fast"):           (15.61, 6.26e-4),
    ("paged", "loop"):           (17.2, 6.43e-3),
    ("continuous", "fast"):      (17.2, 6.43e-3),   # capacity 1: no chunk tiling
    ("continuous", "loop"):      (17.2, 6.43e-3),
}


# --------------------------------------------------------------------------- #
# task configuration -- official LongBench prompts, max_gen and chat-template rule
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TaskSpec:
    prompt: str
    max_gen: int
    chat: bool        # official pred.py wraps all but trec/lcc/triviaqa/samsum/... in the template
    metric_name: str


TASKS: dict[str, TaskSpec] = {
    "qasper": TaskSpec(
        prompt=(
            "You are given a scientific article and a question. Answer the question as "
            "concisely as you can, using a single phrase or sentence if possible. If the "
            "question cannot be answered based on the information in the article, write "
            '"unanswerable". If the question is a yes/no question, answer "yes", "no", or '
            '"unanswerable". Do not provide any explanation.\n\nArticle: {context}\n\n '
            "Answer the question based on the above article as concisely as you can, using "
            "a single phrase or sentence if possible. If the question cannot be answered "
            'based on the information in the article, write "unanswerable". If the question '
            'is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any '
            "explanation.\n\nQuestion: {input}\n\nAnswer:"
        ),
        max_gen=128, chat=True, metric_name="qa_f1"),
    "multifieldqa_en": TaskSpec(
        prompt=(
            "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the "
            "following question based on the above text, only give me the answer and do not "
            "output any other words.\n\nQuestion: {input}\nAnswer:"
        ),
        max_gen=64, chat=True, metric_name="qa_f1"),
    "lcc": TaskSpec(
        prompt="Please complete the code given below. \n{context}Next line of code:\n",
        max_gen=64, chat=False, metric_name="edit_sim"),
    "trec": TaskSpec(
        prompt=("Please determine the type of the question below. Here are some examples of "
                "questions.\n\n{context}\n{input}"),
        max_gen=64, chat=False, metric_name="classification"),
}


# Characters per token, used ONLY to project a run's cost before it starts.
# ~4.0 for the English prose and code in these four tasks.
CHARS_PER_TOKEN_EST = 4.0


def estimate_prefill_seconds(n_tokens: int, kv_mode: str = "paged",
                             prefill: str = "fast") -> float:
    lin, quad = PREFILL_COST.get((kv_mode, prefill), PREFILL_COST[("continuous", "loop")])
    return (lin * n_tokens + quad * n_tokens * n_tokens) / 1000.0


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #

def ensure_data(tasks: list[str], cache_dir: Path) -> dict[str, Path]:
    """Return {task: jsonl path}, downloading LongBench's data.zip once if needed.

    The HF repo is script-based (`LongBench.py` + `data.zip`) and `datasets` >= 4
    no longer executes dataset scripts, so `load_dataset` cannot be used here --
    the zip is fetched directly and the needed task files extracted from it.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {t: cache_dir / f"{t}.jsonl" for t in tasks}
    missing = [t for t, p in paths.items() if not p.exists()]
    if not missing:
        return paths

    print(f"[data] {len(missing)} task file(s) missing {missing}; fetching {HF_DATASET}/data.zip")
    from huggingface_hub import hf_hub_download
    zip_path = hf_hub_download(repo_id=HF_DATASET, filename="data.zip", repo_type="dataset")
    with zipfile.ZipFile(zip_path) as zf:
        members = {Path(n).name: n for n in zf.namelist() if n.endswith(".jsonl")}
        for task in missing:
            name = f"{task}.jsonl"
            if name not in members:
                raise SystemExit(f"[data] {name} not in data.zip; available: "
                                 f"{sorted(members)[:10]}...")
            with zf.open(members[name]) as src, open(paths[task], "wb") as dst:
                shutil.copyfileobj(src, dst)
            print(f"[data] extracted {name} -> {paths[task]}")
    return paths


def load_samples(path: Path, limit: int | None) -> list[dict[str, Any]]:
    """First `limit` samples, in file order -- the official ordering, so a rerun
    with the same limit scores the same subset."""
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if limit is not None and len(out) >= limit:
                break
    return out


# --------------------------------------------------------------------------- #
# engine client
# --------------------------------------------------------------------------- #

RESULT_RE = re.compile(r"^<<<RESULT (.*)>>>$")
WEIGHTS_RE = re.compile(r"Contiguous VRAM weights footprint:\s*([0-9.]+)\s*GB")
DYNAMIC_RE = re.compile(r"Total dynamic memory consumption:\s*([0-9.]+)\s*MB")


class EngineTimeout(RuntimeError):
    pass


class EngineDied(RuntimeError):
    pass


class ServeClient:
    """One `blackwell_llm.exe --serve` process, read by a pump thread.

    A pump thread rather than a blocking read because there is no `select` on
    Windows pipes, and a per-request timeout is not optional here: a 16k-token
    prompt legitimately takes half an hour, so "no output yet" and "hung" can only
    be told apart against a deadline derived from the prompt size.
    """

    def __init__(self, exe: Path, model: str, ctx: int, gpu_layers: int | None,
                 log_path: Path, kv_mode: str = "paged", prefill: str = "fast"):
        self.exe, self.model, self.ctx, self.gpu_layers = exe, model, ctx, gpu_layers
        self.kv_mode, self.prefill = kv_mode, prefill
        self.log = open(log_path, "a", encoding="utf-8")
        self.proc: subprocess.Popen[bytes] | None = None
        self.lines: queue.Queue[str | None] = queue.Queue()
        self.weights_gb: float | None = None
        self.dynamic_mb: float | None = None
        self.restarts = 0

    # -- lifecycle ---------------------------------------------------------- #

    def start(self) -> None:
        cmd = [str(self.exe), "--model", self.model, "--ctx", str(self.ctx), "--serve",
               "--kv-mode", self.kv_mode, "--prefill", self.prefill]
        if self.gpu_layers is not None:
            cmd += ["--gpu-layers", str(self.gpu_layers)]
        self.log.write(f"\n=== spawn {time.strftime('%H:%M:%S')}: {' '.join(cmd)}\n")
        self.log.flush()
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, bufsize=0)
        threading.Thread(target=self._pump, args=(self.proc,), daemon=True).start()

        t0 = time.time()
        while True:
            line = self._next_line(timeout=300)
            if line.startswith("<<<READY"):
                # The arena's VRAM figures are NOT available yet: the DLL's
                # std::cout is block-buffered onto this pipe and flushes at
                # process exit, so they are parsed by close() time, not now.
                print(f"[engine] ready in {time.time() - t0:.1f}s  {line}")
                return

    def _pump(self, proc: subprocess.Popen[bytes]) -> None:
        assert proc.stdout is not None
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if not line.startswith("<<<") and not line.startswith("TEXT "):
                # Engine-internal chatter: keep it out of the protocol stream but
                # on disk, and mine it for the authoritative VRAM figures (there is
                # no per-process VRAM query on WDDM).
                self.log.write(line + "\n")
                m = WEIGHTS_RE.search(line)
                if m:
                    self.weights_gb = float(m.group(1))
                m = DYNAMIC_RE.search(line)
                if m:
                    self.dynamic_mb = float(m.group(1))
                continue
            self.lines.put(line)
        self.log.flush()
        self.lines.put(None)   # EOF sentinel

    def _next_line(self, timeout: float) -> str:
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty as exc:
            raise EngineTimeout(f"no output for {timeout:.0f}s") from exc
        if line is None:
            raise EngineDied("engine closed stdout")
        return line

    def kill(self) -> None:
        if self.proc is None:
            return
        # taskkill /T because the engine has been observed to outlive its own
        # shutdown message (DirectStorage/driver teardown), and a survivor holds
        # 7.2 GiB of VRAM plus a lock on the exe.
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)],
                           capture_output=True, timeout=30)
        except Exception:
            pass
        try:
            self.proc.kill()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=30)
        except Exception:
            pass
        self.proc = None
        while not self.lines.empty():           # drop stale lines from the dead process
            try:
                self.lines.get_nowait()
            except queue.Empty:
                break

    def close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                assert self.proc.stdin is not None
                self.proc.stdin.write(b"QUIT\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=20)
            except Exception:
                pass
        self.kill()
        self.log.close()

    def restart(self) -> None:
        self.restarts += 1
        print(f"[engine] restarting (#{self.restarts})")
        self.kill()
        self.start()

    # -- one request -------------------------------------------------------- #

    def generate(self, prompt: str, max_new: int, max_input_tokens: int,
                 timeout: float, temperature: float = 0.0, top_p: float = 1.0,
                 chat: bool = True) -> dict[str, Any]:
        assert self.proc is not None and self.proc.stdin is not None
        blob = prompt.encode("utf-8")
        header = (f"GEN {len(blob)} {max_new} {max_input_tokens} "
                  f"{temperature:.3f} {top_p:.3f} {1 if chat else 0}\n").encode("utf-8")
        try:
            self.proc.stdin.write(header + blob + b"\n")
            self.proc.stdin.flush()
        except OSError as exc:
            raise EngineDied(f"write failed: {exc}") from exc

        fields: dict[str, str] = {}
        text: str | None = None
        error: str | None = None
        deadline = time.time() + timeout
        while True:
            line = self._next_line(timeout=max(1.0, deadline - time.time()))
            m = RESULT_RE.match(line)
            if m:
                fields = dict(kv.split("=", 1) for kv in m.group(1).split() if "=" in kv)
            elif line.startswith("TEXT "):
                text = _unescape(line[len("TEXT "):])
            elif line.startswith("<<<ERROR"):
                error = line
            elif line.startswith("<<<END>>>"):
                break

        if error is not None:
            raise RuntimeError(error)
        return {
            "text": text or "",
            "prompt_tokens": int(fields.get("prompt_tokens", 0)),
            "gen_tokens": int(fields.get("gen_tokens", 0)),
            "prefill_ms": float(fields.get("prefill_ms", 0.0)),
            "decode_ms": float(fields.get("decode_ms", 0.0)),
            "stop": fields.get("stop", "?"),
            "truncated": fields.get("trunc") == "1",
            "prefill_path": fields.get("prefill", "?"),
        }


def _unescape(s: str) -> str:
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            nxt = s[i + 1]
            out.append({"n": "\n", "r": "\r", "t": "\t", "\\": "\\"}.get(nxt, nxt))
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


# --------------------------------------------------------------------------- #
# VRAM sampling
# --------------------------------------------------------------------------- #

class VramSampler:
    """Polls total GPU memory in use.

    Per-process VRAM is unavailable on this platform (WDDM -- nvidia-smi reports
    N/A per compute app), so this is whole-device usage: the engine's share is
    `peak - baseline`, and the baseline is whatever the desktop already held. The
    engine's own arena log remains the authoritative figure for weights + KV; this
    exists to catch an unexpected allocation on top of them.
    """

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.baseline_mb: int | None = None

    @staticmethod
    def _read_mb() -> int | None:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=15)
            return int(out.stdout.strip().splitlines()[0])
        except Exception:
            return None

    def start(self) -> None:
        self.baseline_mb = self._read_mb()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            mb = self._read_mb()
            if mb is not None:
                self.samples.append(mb)

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return {
            "baseline_mb": self.baseline_mb,
            "peak_total_mb": max(self.samples) if self.samples else None,
            "peak_over_baseline_mb": (max(self.samples) - self.baseline_mb
                                      if self.samples and self.baseline_mb is not None else None),
            "n_samples": len(self.samples),
        }


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #

@dataclass
class TaskRun:
    task: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)


def build_prompt(task: str, sample: dict[str, Any]) -> str:
    return TASKS[task].prompt.format(context=sample["context"], input=sample["input"])


def already_done(path: Path) -> set[tuple[str, str]]:
    """(task, _id) pairs present in an existing predictions file, for --resume."""
    done: set[tuple[str, str]] = set()
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "task" in row and "_id" in row:
                done.add((row["task"], str(row["_id"])))
    return done


def run(args: argparse.Namespace) -> int:
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in tasks if t not in TASKS]
    if unknown:
        raise SystemExit(f"unknown task(s) {unknown}; known: {sorted(TASKS)}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    data = ensure_data(tasks, Path(args.data_dir))
    if args.fetch_only:
        print("[data] fetch-only: done")
        return 0

    samples = {t: load_samples(data[t], args.limit) for t in tasks}
    budget = args.max_input_tokens

    # Cost projection up front: a run that will not finish should say so before it
    # holds the GPU for hours, not after.
    # Per-sample, not budget-times-count: most LongBench samples are shorter than
    # the budget, and assuming otherwise overstates a run by ~1.5x. Token counts
    # are estimated from characters (the engine owns the only real tokenizer here),
    # so this is a projection with a +/-15% error bar, not a promise.
    projected = 0.0
    for t in tasks:
        cap = min(budget, args.ctx - TASKS[t].max_gen - 64)
        for sample in samples[t]:
            est_tokens = min(cap, len(build_prompt(t, sample)) / CHARS_PER_TOKEN_EST)
            projected += estimate_prefill_seconds(int(est_tokens), args.kv_mode, args.prefill)
    print(f"[plan] {sum(len(s) for s in samples.values())} samples across {len(tasks)} task(s), "
          f"input budget {budget} tokens, ctx {args.ctx}")
    lin, quad = PREFILL_COST[(args.kv_mode, args.prefill)]
    print(f"[plan] engine path {args.kv_mode}/{args.prefill}; projected prefill time "
          f"~{projected / 3600:.2f} h (model: {lin} ms/token + {quad:.2e} ms/token^2)")
    if args.dry_run:
        for t in tasks:
            print(f"[dry-run] {t}: {len(samples[t])} samples, max_gen={TASKS[t].max_gen}, "
                  f"chat={TASKS[t].chat}")
            if samples[t]:
                p = build_prompt(t, samples[t][0])
                print(f"           first prompt {len(p)} chars, head: {p[:160]!r}")
        return 0

    done = already_done(pred_path) if args.resume else set()
    if done:
        print(f"[resume] skipping {len(done)} sample(s) already in {pred_path}")

    vram = VramSampler()
    vram.start()
    client = ServeClient(Path(args.exe), args.model, args.ctx, args.gpu_layers,
                         out_dir / "engine.log", args.kv_mode, args.prefill)
    client.start()

    runs = {t: TaskRun(t) for t in tasks}
    wall_t0 = time.time()
    pred_fh = open(pred_path, "a", encoding="utf-8")
    try:
        for task in tasks:
            spec = TASKS[task]
            task_budget = min(budget, args.ctx - spec.max_gen - 64)
            timeout = (60.0 + 3.0 * estimate_prefill_seconds(task_budget, args.kv_mode,
                                                             args.prefill)
                       + 0.5 * spec.max_gen)
            print(f"\n=== {task}: {len(samples[task])} samples, max_gen={spec.max_gen}, "
                  f"chat={spec.chat}, budget={task_budget}, timeout={timeout:.0f}s")

            for i, sample in enumerate(samples[task]):
                key = (task, str(sample.get("_id", i)))
                if key in done:
                    continue
                prompt = build_prompt(task, sample)
                t0 = time.time()
                try:
                    res = client.generate(prompt, spec.max_gen, task_budget, timeout,
                                          chat=spec.chat)
                except (EngineTimeout, EngineDied, RuntimeError) as exc:
                    # One bad sample must not end the run: record it, restart the
                    # engine (a timeout leaves the protocol stream desynchronised)
                    # and continue.
                    print(f"  [{i + 1}/{len(samples[task])}] FAILED {type(exc).__name__}: "
                          f"{str(exc)[:160]}")
                    runs[task].failures.append({"_id": sample.get("_id", i),
                                                "error": f"{type(exc).__name__}: {exc}"})
                    try:
                        client.restart()
                    except Exception as exc2:
                        print(f"  [fatal] engine will not restart: {exc2}")
                        raise
                    continue

                scored = metrics.score_sample(task, res["text"], sample["answers"],
                                              sample.get("all_classes"))
                row = {
                    "task": task,
                    "_id": sample.get("_id", i),
                    "length_words": sample.get("length"),
                    "prompt_tokens": res["prompt_tokens"],
                    "gen_tokens": res["gen_tokens"],
                    "prefill_ms": res["prefill_ms"],
                    "decode_ms": res["decode_ms"],
                    "wall_s": round(time.time() - t0, 2),
                    "stop": res["stop"],
                    "truncated": res["truncated"],
                    "prefill_path": res["prefill_path"],
                    "prediction": res["text"],
                    "answers": sample["answers"],
                    **scored,
                }
                runs[task].rows.append(row)
                pred_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                pred_fh.flush()
                print(f"  [{i + 1}/{len(samples[task])}] score={scored['score']:.3f} "
                      f"tok={res['prompt_tokens']} gen={res['gen_tokens']} "
                      f"stop={res['stop']}{' TRUNC' if res['truncated'] else ''} "
                      f"{row['wall_s']:.1f}s")
    finally:
        pred_fh.close()
        client.close()
        vram_stats = vram.stop()

    summary = summarize(runs, args, vram_stats, client, time.time() - wall_t0)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    print_summary(summary)
    print(f"\n[out] {pred_path}\n[out] {out_dir / 'summary.json'}\n[out] {out_dir / 'engine.log'}")
    return 0


def summarize(runs: dict[str, TaskRun], args: argparse.Namespace,
              vram_stats: dict[str, Any], client: ServeClient,
              wall_s: float) -> dict[str, Any]:
    per_task = {}
    for task, tr in runs.items():
        rows = tr.rows
        if not rows:
            per_task[task] = {"n": 0, "failures": len(tr.failures)}
            continue
        scores = [r["score"] for r in rows]
        prompt_tokens = [r["prompt_tokens"] for r in rows]
        secondary = {k: round(100 * statistics.fmean([r[k] for r in rows]), 2)
                     for k in rows[0]
                     if k not in {"task", "_id", "length_words", "prompt_tokens", "gen_tokens",
                                  "prefill_ms", "decode_ms", "wall_s", "stop", "truncated",
                                  "prediction", "answers", "score", "prefill_path"}}
        per_task[task] = {
            "n": len(rows),
            "failures": len(tr.failures),
            "metric": TASKS[task].metric_name,
            "score": round(100 * statistics.fmean(scores), 2),
            # Standard error of the mean: 50 samples is a small subset, and a
            # score quoted without it invites reading noise as a regression.
            "score_stderr": (round(100 * statistics.stdev(scores) / len(scores) ** 0.5, 2)
                             if len(scores) > 1 else None),
            "secondary": secondary,
            "prompt_tokens_mean": round(statistics.fmean(prompt_tokens), 1),
            "prompt_tokens_max": max(prompt_tokens),
            "truncated_frac": round(sum(r["truncated"] for r in rows) / len(rows), 3),
            "stop_reasons": {s: sum(1 for r in rows if r["stop"] == s)
                             for s in sorted({r["stop"] for r in rows})},
            "prefill_ms_mean": round(statistics.fmean(r["prefill_ms"] for r in rows), 1),
            "prefill_ms_per_token": round(
                statistics.fmean(r["prefill_ms"] / max(1, r["prompt_tokens"]) for r in rows), 2),
            "decode_ms_per_token": round(
                statistics.fmean(r["decode_ms"] / r["gen_tokens"]
                                 for r in rows if r["gen_tokens"] > 0), 2)
            if any(r["gen_tokens"] for r in rows) else None,
            "wall_s_mean": round(statistics.fmean(r["wall_s"] for r in rows), 1),
        }
    return {
        "config": {
            "model": args.model, "exe": str(args.exe), "ctx": args.ctx,
            "max_input_tokens": args.max_input_tokens, "limit": args.limit,
            "tasks": list(runs), "gpu_layers": args.gpu_layers,
            "kv_mode": args.kv_mode, "prefill": args.prefill,
            "sampling": "greedy (temperature=0)",
        },
        "engine": {
            "weights_gb": client.weights_gb,
            "kv_dynamic_mb": client.dynamic_mb,
            "restarts": client.restarts,
        },
        "vram": vram_stats,
        "wall_clock_s": round(wall_s, 1),
        "tasks": per_task,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 78)
    print(f"{'task':<18}{'metric':<16}{'n':>4}{'score':>9}{'+/-':>7}"
          f"{'tok':>7}{'trunc':>7}{'s/sample':>10}")
    print("-" * 78)
    for task, t in summary["tasks"].items():
        if not t.get("n"):
            print(f"{task:<18}{'-':<16}{0:>4}")
            continue
        se = f"{t['score_stderr']:.2f}" if t["score_stderr"] is not None else "-"
        print(f"{task:<18}{t['metric']:<16}{t['n']:>4}{t['score']:>9.2f}{se:>7}"
              f"{t['prompt_tokens_mean']:>7.0f}{t['truncated_frac']:>7.2f}"
              f"{t['wall_s_mean']:>10.1f}")
    print("=" * 78)
    eng, vram = summary["engine"], summary["vram"]
    print(f"weights {eng['weights_gb']} GB | KV/dynamic {eng['kv_dynamic_mb']} MB | "
          f"engine restarts {eng['restarts']}")
    print(f"GPU total used: baseline {vram['baseline_mb']} MB -> peak "
          f"{vram['peak_total_mb']} MB (delta {vram['peak_over_baseline_mb']} MB)")
    cfg = summary["config"]
    print(f"engine path {cfg['kv_mode']}/{cfg['prefill']} | ctx {cfg['ctx']} | "
          f"input budget {cfg['max_input_tokens']} | {cfg['sampling']}")
    print(f"wall clock {summary['wall_clock_s'] / 60:.1f} min")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL, help="checkpoint directory")
    p.add_argument("--exe", default=str(DEFAULT_EXE), help="blackwell_llm.exe (Release)")
    p.add_argument("--tasks", default=",".join(TASKS), help="comma-separated LongBench tasks")
    p.add_argument("--limit", type=int, default=50, help="samples per task (0 = all)")
    p.add_argument("--output-dir", default=str(REPO_ROOT / "benchmarks/results/e8w5_baseline"))
    p.add_argument("--data-dir", default=str(REPO_ROOT / "benchmarks/data"))
    p.add_argument("--ctx", type=int, default=16384, help="engine context length")
    p.add_argument("--max-input-tokens", type=int, default=8000,
                   help="middle-truncation budget for the prompt. Cost is quadratic in "
                        "this number -- see the module docstring")
    p.add_argument("--gpu-layers", type=int, default=None,
                   help="layers resident in VRAM (default: all)")
    p.add_argument("--kv-mode", choices=("paged", "continuous"), default="paged",
                   help="paged raises the engine's token capacity, which is what lets "
                        "prefill run in batched chunks (default: paged)")
    p.add_argument("--prefill", choices=("fast", "loop"), default="fast",
                   help="fast = PrefillTokens; loop = per-token Forward() (reference)")
    p.add_argument("--resume", action="store_true",
                   help="skip samples already present in predictions.jsonl")
    p.add_argument("--dry-run", action="store_true", help="show the plan, run nothing")
    p.add_argument("--fetch-only", action="store_true", help="download/extract data and exit")
    args = p.parse_args(argv)
    if args.limit == 0:
        args.limit = None
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
