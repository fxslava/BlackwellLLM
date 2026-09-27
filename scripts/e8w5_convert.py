#!/usr/bin/env python3
"""Stream a HuggingFace checkpoint into the engine's E8W5 format, one tensor at a time.

Reads a source checkpoint, quantizes every linear projection weight to 5-bit companded-E8
(docs/E8W5_FORMAT_SPEC.md), copies everything else through byte-faithfully, and writes a
checkpoint VRAMArena can load with quant_method="e8w5".

Emitted per quantized projection ``<base>.weight``, replacing it:
    <base>.plane_lo   int32 [out, in/8]    one word per 8-weight E8 block
    <base>.plane_hi   int32 [out, in/32]   one byte per block, four blocks per word
    <base>.scales     f16   [out, in/128]
    <base>.codebook   f16   [64]           fitted per tensor

int32 rather than uint32/uint8 deliberately: AWQ already ships I32 planes and F16 scales
through this loader, so those are the dtypes the path is known to carry. The bytes are
identical either way -- plane_hi as int32[in/32] is the same little-endian image as
uint8[in/8], which is exactly what the kernel reads (one uint32 per four blocks).

WHAT THIS VERSION DOES NOT DO: outlier channel retention. Every K column goes into the
lattice bulk. The research's dB figures for b=5 all include 0.1% BF16 outliers AND used
per-row scales; per-group-128 scales absorb much of what sparse retention was compensating
for, but "much" is a measurement, not an assumption -- run
``python research/pack_e8w5.py --neutrality`` and read section 9 of the integration doc
before trusting a converted model on a spiky layer. Keeping outliers out of v1 also keeps
the format exactly what the kernel consumes: no gather, no padding, no dead tensors, and no
double-counting hazard for a future kernel that does add them.

    python scripts/e8w5_convert.py --src F:/AI/models/GLM-4-9B-Chat-1M-hf \\
                                   --dst F:/AI/models/GLM-4-9B-e8w5 --device cuda
    python scripts/e8w5_convert.py --src ... --dst ... --layers 0 1 --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The one sanctioned direction of dependency: a script may import the research harness, so
# the shipping quantizer is literally the one the research measured. src/ and tests/ never do.
sys.path.insert(0, os.path.join(REPO, "research"))

from pack_e8w5 import (                                              # noqa: E402
    COORD_BITS, E8_DIM, GROUP, LUT_N, fit_e8w5, pack_planes, unpack_planes, LO, N_LEVELS)
from eval_fixed_rate_e8 import _alpha_grid                           # noqa: E402

# Projections the LinearDispatcher routes; everything else is copied through. Fused and
# split MLP variants both appear across the families this engine loads (GLM-4 fuses
# gate_up_proj, Llama/Qwen split it), so match on the base name rather than a fixed list.
QUANTIZE_RE = re.compile(
    r"^model\.layers\.(\d+)\.("
    r"self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    r"|mlp\.(gate_up_proj|gate_proj|up_proj|down_proj)"
    r")\.weight$")


def is_quantizable(name: str, layers: Optional[List[int]]) -> bool:
    m = QUANTIZE_RE.match(name)
    if not m:
        return False
    return layers is None or int(m.group(1)) in layers


def sqnr_db(ref: torch.Tensor, got: torch.Tensor) -> float:
    num = ref.to(torch.float64).pow(2).sum()
    den = (ref.to(torch.float64) - got.to(torch.float64)).pow(2).sum().clamp(min=1e-300)
    return float(10.0 * torch.log10(num / den))


def quantize_tensor(w: torch.Tensor, grid: torch.Tensor,
                    lambda_grid: Optional[List[float]],
                    verify: bool) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    """[out, in] -> the four E8W5 tensors, plus stats. Verifies the pack round-trips."""
    rows, cols = w.shape
    if cols % GROUP:
        raise ValueError(
            f"in_features={cols} is not a multiple of the {GROUP}-weight scale group; "
            "E8W5 has no sub-group padding path (spec section 8)")

    k, c, scales, codebook, w_hat, meta = fit_e8w5(
        w.to(torch.float32), group=GROUP, grid=grid, lambda_grid=lambda_grid)
    plane_lo, plane_hi = pack_planes(k, c)

    if verify:
        # The format is only 5.00 bpw if the pack is lossless. Check, do not assume.
        k2, c2 = unpack_planes(plane_lo, plane_hi)
        if not (torch.equal(k2.to(torch.int64), k.to(torch.int64)) and torch.equal(c2, c)):
            raise RuntimeError("E8W5 pack failed to round-trip")

    n_blocks = cols // E8_DIM
    lo_i32 = plane_lo.to(torch.int64).to(torch.int32).reshape(rows, n_blocks)
    # Fold four blocks' bytes into one int32, little-endian, matching the kernel's read.
    hi = plane_hi.to(torch.int64).reshape(rows, n_blocks // 4, 4)
    hi_i32 = (hi[..., 0] | (hi[..., 1] << 8) | (hi[..., 2] << 16) | (hi[..., 3] << 24))
    hi_i32 = hi_i32.to(torch.int32) - (((hi_i32 >> 31) & 1) << 32).to(torch.int32)

    out = {
        "plane_lo": lo_i32.cpu().contiguous(),
        "plane_hi": hi_i32.reshape(rows, n_blocks // 4).cpu().contiguous(),
        "scales": scales.to(torch.float16).cpu().contiguous(),
        "codebook": codebook.to(torch.float16).cpu().contiguous(),
    }
    stats = {"wsqnr_db": sqnr_db(w.to(torch.float32), w_hat),
             "lambda": meta["lambda"], "sat_at_bound": meta["sat_at_bound"],
             "groups": int(scales.shape[1])}
    return out, stats


class ShardWriter:
    """Accumulates tensors and flushes a safetensors shard once a byte budget is reached.

    This is what keeps the converter's resident set to roughly one shard: the source is read
    tensor by tensor and the output never holds more than `budget_bytes` of finished work.
    """

    def __init__(self, dst: str, budget_bytes: int, prefix: str = "model"):
        self.dst = dst
        self.budget = budget_bytes
        self.prefix = prefix
        self.buf: Dict[str, torch.Tensor] = {}
        self.bytes = 0
        self.shards: List[Tuple[str, List[str]]] = []
        self.weight_map: Dict[str, str] = {}
        self.total_bytes = 0

    def add(self, name: str, t: torch.Tensor) -> None:
        self.buf[name] = t
        n = t.numel() * t.element_size()
        self.bytes += n
        self.total_bytes += n
        if self.bytes >= self.budget:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        from safetensors.torch import save_file
        idx = len(self.shards) + 1
        fname = f"{self.prefix}-{idx:05d}.safetensors"
        save_file(self.buf, os.path.join(self.dst, fname),
                  metadata={"format": "pt"})
        for k in self.buf:
            self.weight_map[k] = fname
        self.shards.append((fname, list(self.buf.keys())))
        print(f"    [shard] wrote {fname}  ({self.bytes / 2**20:.0f} MiB, "
              f"{len(self.buf)} tensors)", flush=True)
        self.buf.clear()
        self.bytes = 0

    def write_index(self) -> None:
        self.flush()
        # Rename to the canonical -of- form now that the shard count is known.
        total = len(self.shards)
        renames = {}
        for i, (fname, keys) in enumerate(self.shards, start=1):
            final = f"{self.prefix}-{i:05d}-of-{total:05d}.safetensors"
            os.replace(os.path.join(self.dst, fname), os.path.join(self.dst, final))
            renames[fname] = final
        self.weight_map = {k: renames[v] for k, v in self.weight_map.items()}
        index = {"metadata": {"total_size": self.total_bytes},
                 "weight_map": self.weight_map}
        with open(os.path.join(self.dst, "model.safetensors.index.json"), "w",
                  encoding="utf-8") as f:
            json.dump(index, f, indent=2)


def write_config(src: str, dst: str) -> None:
    """Copy config.json with a quantization_config block the ConfigLoader recognises."""
    with open(os.path.join(src, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["quantization_config"] = {
        "quant_method": "e8w5",
        "bits": COORD_BITS,
        "group_size": GROUP,
        # Informational: the decoder needs none of this, the codebook carries it.
        "codebook_entries": LUT_N,
        "lattice": "E8",
        "format_doc": "docs/E8W5_FORMAT_SPEC.md",
    }
    with open(os.path.join(dst, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def copy_aux_files(src: str, dst: str) -> None:
    """Tokenizer and friends: the engine reads the checkpoint's own tokenizer."""
    for name in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model",
                 "special_tokens_map.json", "generation_config.json", "vocab.json",
                 "merges.txt", "added_tokens.json", "chat_template.jinja"):
        p = os.path.join(src, name)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(dst, name))


def convert(src: str, dst: str, device: str, grid_n: int,
            lambda_grid: Optional[List[float]], layers: Optional[List[int]],
            budget_mib: int, verify: bool, dry_run: bool) -> Dict[str, object]:
    from safetensors import safe_open

    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(src, "model.safetensors.index.json"), "r",
              encoding="utf-8") as f:
        src_index = json.load(f)
    weight_map: Dict[str, str] = src_index["weight_map"]

    # Group by shard so each source file is opened once and read in its own order.
    by_shard: Dict[str, List[str]] = {}
    for name, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(name)

    grid = _alpha_grid(grid_n, device=device)
    writer = ShardWriter(dst, budget_mib * 2**20)
    report: Dict[str, object] = {"src": src, "dst": dst, "device": device,
                                 "grid_points": grid_n, "tensors": []}
    n_quant = 0
    n_copy = 0
    src_bytes = 0
    t0 = time.time()

    for shard in sorted(by_shard):
        print(f"[shard] {shard}", flush=True)
        with safe_open(os.path.join(src, shard), framework="pt", device="cpu") as fh:
            for name in sorted(by_shard[shard]):
                t = fh.get_tensor(name)
                src_bytes += t.numel() * t.element_size()
                if not is_quantizable(name, layers):
                    n_copy += 1
                    if not dry_run:
                        writer.add(name, t)
                    continue

                base = name[: -len(".weight")]
                w = t.to(device)
                ts = time.time()
                packed, stats = quantize_tensor(w, grid, lambda_grid, verify)
                del w
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                if not dry_run:
                    for suffix, tensor in packed.items():
                        writer.add(f"{base}.{suffix}", tensor)
                n_quant += 1
                rec = {"tensor": name, "shape": list(t.shape), **stats,
                       "seconds": round(time.time() - ts, 1)}
                report["tensors"].append(rec)
                print(f"    {name:52s} {tuple(t.shape)}  W-SQNR {stats['wsqnr_db']:6.2f} dB"
                      f"  lam={stats['lambda']}  [{rec['seconds']}s]", flush=True)
                del t, packed

    if not dry_run:
        writer.write_index()
        write_config(src, dst)
        copy_aux_files(src, dst)

    dst_bytes = writer.total_bytes
    report.update({
        "quantized_tensors": n_quant, "copied_tensors": n_copy,
        "src_bytes": src_bytes, "dst_bytes": dst_bytes,
        "compression_x": (src_bytes / dst_bytes) if dst_bytes else 0.0,
        "seconds_total": round(time.time() - t0, 1),
    })
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="source HF checkpoint directory")
    ap.add_argument("--dst", required=True, help="output directory")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--grid", type=int, default=40,
                    help="step-search grid points per group (40 = the research default)")
    ap.add_argument("--lambda-grid", type=float, nargs="+", default=None,
                    help="companding strengths to fit over; default 1.0 only. The "
                         "full-model sweep shows layer-0 q/k_proj prefer 0.25, so "
                         "'--lambda-grid 0.0 0.25 0.5 1.0' is the faithful setting")
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="restrict quantization to these layer indices (others copied "
                         "through unquantized) -- for a cheap partial-conversion smoke test")
    ap.add_argument("--shard-mib", type=int, default=1536)
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the per-tensor pack round-trip check")
    ap.add_argument("--dry-run", action="store_true",
                    help="quantize and report quality, write nothing")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    rep = convert(args.src, args.dst, args.device, args.grid, args.lambda_grid,
                  args.layers, args.shard_mib, not args.no_verify, args.dry_run)

    print(f"\nquantized {rep['quantized_tensors']} tensors, copied {rep['copied_tensors']}")
    print(f"src {rep['src_bytes'] / 2**30:.2f} GiB -> dst {rep['dst_bytes'] / 2**30:.2f} GiB"
          f"  ({rep['compression_x']:.2f}x)  in {rep['seconds_total']}s")
    if rep["tensors"]:
        db = [t["wsqnr_db"] for t in rep["tensors"]]
        print(f"W-SQNR over quantized tensors: min {min(db):.2f} / "
              f"mean {sum(db) / len(db):.2f} / max {max(db):.2f} dB")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1)
        print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
