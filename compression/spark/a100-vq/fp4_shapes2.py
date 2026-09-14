"""Shape-by-shape FP4 vs FP8, with the weights actually coming from DRAM.

v1 of this benchmark timed 200 back-to-back calls on one matrix and reported 418 GB/s -- above
this box's 234 GB/s DRAM peak -- because a 25 MB weight stays in L2 once it is there. A verify
step reads each weight exactly once against an 11 GB working set, so every read misses. Here each
rep uses a different copy of the weight, enough copies to exceed L2, which is the condition the
step runs in.
"""
from __future__ import annotations

import os
import sys

import torch

WORK = os.path.expanduser("~/dsv41-spark/work")
sys.path.insert(0, WORK)
sys.path.insert(0, os.path.join(WORK, "tools"))
from engine.v41_engine import V41Engine                        # noqa: E402
import v41_ref as R                                            # noqa: E402
import engine.fastdecode as FD                                 # noqa: E402
from fp4_linear import quantize_fp8_to_fp4                     # noqa: E402
from fp8_linear import FP8Weight                               # noqa: E402

md = os.path.expanduser("~/dsv41-spark/models/DeepSeek-V4.1-Flash")
os.environ["DSV41_DENSE_FP4"] = "off"
eng = V41Engine(md, max_seq=8192, spec=True,
                trace_stats="results/trace-full-20260910/stats/coverage.json",
                arena_gb=30.0, transient_slots=64, keep_free_gb=12.0, prune_keep=None,
                expert_format="cb3")
W = eng.model.W
M = FD.T_VERIFY
COPIES = int(os.environ.get("COPIES", "12"))     # 12 x 25 MB = 300 MB, well past L2
print(f"M={M}  copies per shape = {COPIES}", flush=True)


def ev_rot(objs, call, n=None):
    n = n or len(objs) * 8
    for i in range(len(objs)):
        call(objs[i])
    torch.cuda.synchronize()
    e = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
         for _ in range(n)]
    for i, (a, b) in enumerate(e):
        a.record(); call(objs[i % len(objs)]); b.record()
    torch.cuda.synchronize()
    v = sorted(x.elapsed_time(y) for x, y in e)
    return v[len(v) // 2] * 1000


print(f"\n{'name':6s} {'K':>6s} {'N':>7s} {'grid':>6s} {'fp8 MB':>7s} {'fp4 MB':>7s} "
      f"{'fp8 us':>8s} {'fp4 us':>8s} {'fp8 GB/s':>9s} {'fp4 GB/s':>9s} {'win':>5s} {'dus':>7s}")
b0 = W.layers[0]
rows = []
for nm in ("wq_a", "wq_b", "wkv", "wo_b"):
    w8 = getattr(b0, nm)
    if not isinstance(w8, FP8Weight):
        continue
    K = w8.w.shape[1] * (2 if getattr(w8, "packed", False) else 1)
    N = w8.w.shape[0]
    x = torch.randn(M, w8.K if hasattr(w8, "K") else K, dtype=torch.bfloat16, device="cuda")
    c8 = [w8] + [FP8Weight(w8.w.clone(), w8.s.clone()) for _ in range(COPIES - 1)]
    t8 = ev_rot(c8, lambda w: R.dense(x, w))
    b8 = (w8.w.numel() + w8.s.numel() * 4) / 2**20
    del c8
    torch.cuda.empty_cache()
    w4 = quantize_fp8_to_fp4(w8)
    c4 = [quantize_fp8_to_fp4(w8) for _ in range(COPIES)]
    t4 = ev_rot(c4, lambda w: R.dense(x, w))
    b4 = (w4.w.numel() + w4.s.numel() * 4) / 2**20
    del c4, w4
    torch.cuda.empty_cache()
    g8 = b8 * 2**20 / (t8 * 1e-6) / 1e9
    g4 = b4 * 2**20 / (t4 * 1e-6) / 1e9
    grid = -(-N // 128)
    rows.append((nm, K, N, grid, b8, b4, t8, t4, g8, g4))
    print(f"{nm:6s} {K:6d} {N:7d} {grid:6d} {b8:7.2f} {b4:7.2f} {t8:8.1f} {t4:8.1f} "
          f"{g8:9.1f} {g4:9.1f} {('fp4' if t4 < t8 else 'FP8'):>5s} {t8-t4:7.1f}")

f8 = sum(r[6] for r in rows) * 40 / 1000
f4 = sum(r[7] for r in rows) * 40 / 1000
mb8 = sum(r[4] for r in rows) * 40 / 1024
mb4 = sum(r[5] for r in rows) * 40 / 1024
best = sum(min(r[6], r[7]) for r in rows) * 40 / 1000
print(f"\n40 layers: all-fp8 {f8:6.3f} ms / {mb8:5.3f} GB    all-fp4 {f4:6.3f} ms / {mb4:5.3f} GB")
print(f"           per-shape best-of {best:6.3f} ms  (fp4 where it wins, fp8 where it does not)")
print(f"  all-fp4 aggregate {mb4*1024*2**20/(f4*1e-3)/1e9:6.1f} GB/s")
print(f"  latency floor: the two small shapes cost "
      f"{sum(r[7] for r in rows if r[3] < 48)*40/1000:.3f} ms of the {f4:.3f} ms for "
      f"{sum(r[5] for r in rows if r[3] < 48)*40/1024:.3f} GB of the {mb4:.3f} GB")
