"""Module D -- nvCOMP v5 GPU decompression benchmark for the quantized payloads.

The question this answers is narrow and practical: if a layer's weights live on NVMe
or in host RAM as a compressed blob, can the GPU turn them back into the kernel's
input format fast enough to hide behind the compute window of the layer in front of
it (~0.46 ms)?

WHAT IS COMPRESSED
------------------
The *fixed-width, random-access payload* each scheme would actually stream -- E8's
bit-packed (coset bit + 8 coordinates) blocks, and the scalar schemes' packed 4-bit
indices. Compressing anything else (the fp32 dequantized tensor, say) would measure
a byte stream the engine never reads and would flatter the ratio enormously.

MEASUREMENT DISCIPLINE
----------------------
Four things that silently invalidate a GPU codec benchmark, and what is done here:

* *Stream mismatch.* The codec is constructed with ``cuda_stream`` set to the same
  torch stream the timing events are recorded on, so the events actually bracket the
  decode instead of racing an internal stream.
* *Hidden synchronization.* ``decode`` calls ``configure_decompression`` internally
  unless a config is supplied, and that call synchronizes. The config is therefore
  built once, outside the loop.
* *Allocation inside the loop.* The output buffer is preallocated and reused, so the
  number is decompression, not cudaMalloc.
* *Unverified output.* Every configuration round-trips once and compares bytes before
  it is timed. An impressively fast broken decode is the classic failure here.

THE COMPARISON THAT MATTERS
---------------------------
Throughput is reported against two reference points, because "GB/s" alone decides
nothing:

* the **0.46 ms layer compute budget** -- can a prefetch hide inside it;
* the **measured device-to-device copy bandwidth** for the same payload -- if
  decompression is slower than simply reading the uncompressed bytes from VRAM, then
  compression only ever pays off on the *transfer* path (NVMe/PCIe at ~7 GB/s), never
  as a VRAM-resident format. That distinction is the actual engineering conclusion.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Optional

import torch

DEFAULT_CODECS = ["GDeflate", "Bitcomp", "LZ4", "ANS", "Zstd"]
LAYER_BUDGET_MS = 0.46


def _to_device_bytes(payload: bytes, device: str = "cuda") -> torch.Tensor:
    return torch.frombuffer(bytearray(payload), dtype=torch.uint8).to(device)


def measure_d2d_bandwidth(nbytes: int, iters: int = 50, device: str = "cuda") -> Dict[str, float]:
    """Device-to-device copy bandwidth for a buffer of this size.

    The honest ceiling for any VRAM-resident scheme: whatever a plain copy of the same
    payload costs is the bar decompression has to beat to be worth doing in VRAM.
    """
    src = torch.empty(nbytes, dtype=torch.uint8, device=device)
    dst = torch.empty_like(src)
    stream = torch.cuda.current_stream()
    for _ in range(5):
        dst.copy_(src)
    stream.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record(stream)
    for _ in range(iters):
        dst.copy_(src)
    end.record(stream)
    stream.synchronize()
    ms = start.elapsed_time(end) / iters
    # a copy touches the bytes twice (read + write)
    return {"copy_ms": ms, "copy_read_gbps": nbytes / 1e9 / (ms / 1e3),
            "copy_rw_gbps": 2 * nbytes / 1e9 / (ms / 1e3)}


def bench_one(payload: bytes, algorithm: str, iters: int = 50,
              chunk_size: int = 1 << 16, device: str = "cuda",
              budget_ms: float = LAYER_BUDGET_MS,
              n_weights: Optional[int] = None) -> Dict[str, object]:
    """Compress ``payload`` once, then time ``iters`` warm decompressions."""
    from nvidia import nvcomp

    stream = torch.cuda.current_stream()
    src = _to_device_bytes(payload, device)
    codec = nvcomp.Codec(algorithm=algorithm, cuda_stream=stream.cuda_stream,
                         device_id=torch.cuda.current_device(),
                         uncomp_chunk_size=chunk_size)
    arr = nvcomp.as_array(src, cuda_stream=stream.cuda_stream)
    comp = codec.encode(arr)
    stream.synchronize()
    comp_bytes = int(comp.buffer_size)

    # correctness first: an unverified decode is not a measurement.
    cfg = codec.decompression_config(comp)
    out_buf = torch.empty(src.numel(), dtype=torch.uint8, device=device)
    out_arr = nvcomp.as_array(out_buf, cuda_stream=stream.cuda_stream)
    codec.decode(comp, out=out_arr, decompression_config=cfg)
    stream.synchronize()
    if not bool(torch.equal(out_buf, src)):
        return {"algorithm": algorithm, "error": "round-trip mismatch"}

    for _ in range(10):                                        # warm the codec
        codec.decode(comp, out=out_arr, decompression_config=cfg)
    stream.synchronize()

    per_iter: List[float] = []
    for _ in range(iters):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record(stream)
        codec.decode(comp, out=out_arr, decompression_config=cfg)
        end.record(stream)
        stream.synchronize()
        per_iter.append(start.elapsed_time(end))
    per_iter.sort()
    mean_ms = sum(per_iter) / len(per_iter)
    uncomp = src.numel()
    out: Dict[str, object] = {}
    if n_weights:
        # the number that decides whether an entropy rate is real: what the layer
        # costs on the transfer path once this codec has squeezed it.
        out["effective_bpw"] = comp_bytes * 8 / n_weights
        out["raw_bpw"] = uncomp * 8 / n_weights
    out.update({
        "algorithm": algorithm,
        "chunk_size": chunk_size,
        "uncompressed_bytes": uncomp,
        "compressed_bytes": comp_bytes,
        "compression_ratio": uncomp / comp_bytes,
        "decomp_ms_mean": mean_ms,
        "decomp_ms_p50": per_iter[len(per_iter) // 2],
        "decomp_ms_p99": per_iter[min(len(per_iter) - 1, int(0.99 * len(per_iter)))],
        "decomp_ms_min": per_iter[0],
        "decomp_gbps": uncomp / 1e9 / (mean_ms / 1e3),
        "read_gbps": comp_bytes / 1e9 / (mean_ms / 1e3),
        "fits_layer_budget": bool(mean_ms <= budget_ms),
        "budget_ms": budget_ms,
        "iters": iters,
    })
    return out


def bench_payloads(payloads: Dict[str, bytes], codecs: Optional[List[str]] = None,
                   iters: int = 50, chunk_size: int = 1 << 16,
                   device: str = "cuda", budget_ms: float = LAYER_BUDGET_MS,
                   verbose: bool = True,
                   n_weights: Optional[int] = None) -> List[Dict[str, object]]:
    codecs = codecs or DEFAULT_CODECS
    results: List[Dict[str, object]] = []
    for pname, payload in payloads.items():
        if payload is None:
            continue
        bw = measure_d2d_bandwidth(len(payload), device=device)
        if verbose:
            print(f"\n  payload {pname}: {len(payload)/1e6:.2f} MB  "
                  f"(d2d copy {bw['copy_ms']:.3f} ms -> {bw['copy_read_gbps']:.0f} GB/s read)")
        for algo in codecs:
            try:
                r = bench_one(payload, algo, iters, chunk_size, device, budget_ms,
                              n_weights=n_weights)
            except Exception as exc:                          # a codec may be absent
                r = {"algorithm": algo, "error": f"{type(exc).__name__}: {exc}"}
            r["payload"] = pname
            r.update({f"d2d_{k}": v for k, v in bw.items()})
            if "error" not in r:
                r["decomp_vs_copy"] = r["decomp_ms_mean"] / bw["copy_ms"]
                if verbose:
                    eff = (f" -> {r['effective_bpw']:.3f} bpw" if "effective_bpw" in r
                           else "")
                    print(f"    {algo:10s} ratio={r['compression_ratio']:.3f}{eff}  "
                          f"{r['decomp_ms_mean']:7.3f} ms  {r['decomp_gbps']:7.1f} GB/s  "
                          f"{'FITS' if r['fits_layer_budget'] else 'OVER'} {budget_ms} ms  "
                          f"({r['decomp_vs_copy']:.0f}x a d2d copy)")
            elif verbose:
                print(f"    {algo:10s} {r['error']}")
            results.append(r)
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--payload", action="append", default=[],
                    help="name=path to a raw payload file; repeatable")
    ap.add_argument("--synthetic-mb", type=float, default=24.0,
                    help="if no --payload given, benchmark a payload of this size")
    ap.add_argument("--codecs", nargs="+", default=DEFAULT_CODECS)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--chunk-size", type=int, default=1 << 16)
    ap.add_argument("--budget-ms", type=float, default=LAYER_BUDGET_MS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA unavailable"); return 1
    print(f"device: {torch.cuda.get_device_name(0)}")

    payloads: Dict[str, bytes] = {}
    for spec in args.payload:
        name, _, path = spec.partition("=")
        with open(path, "rb") as fh:
            payloads[name] = fh.read()
    if not payloads:
        n = int(args.synthetic_mb * 1e6)
        g = torch.Generator().manual_seed(0)
        payloads["synthetic-4bit"] = torch.randint(
            0, 256, (n,), generator=g, dtype=torch.uint8).numpy().tobytes()

    res = bench_payloads(payloads, args.codecs, args.iters, args.chunk_size,
                         budget_ms=args.budget_ms)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
