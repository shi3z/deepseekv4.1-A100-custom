"""Phase 3: the gate GEMM across the decode M values and both N, current path vs new.

The verify block is 2 rows today but DSV41_BLOCK can make it 6, and the MTP gate is N=128, so the
kernel has to hold up on more than the one shape it was tuned on. DRAM-resident (a copy per rep).
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

WORK = os.path.expanduser("~/dsv41-spark/work")
sys.path.insert(0, WORK)
sys.path.insert(0, os.path.join(WORK, "tools"))
from engine.v41_engine import V41Engine                        # noqa: E402
import v41_ref as R                                            # noqa: E402
from gate_gemm import gate_linear, plan                        # noqa: E402

md = os.path.expanduser("~/dsv41-spark/models/DeepSeek-V4.1-Flash")
eng = V41Engine(md, max_seq=8192, spec=True,
                trace_stats="results/trace-full-20260910/stats/coverage.json",
                arena_gb=20.0, transient_slots=64, keep_free_gb=12.0, prune_keep=None,
                expert_format="cb3")
W = eng.model.W
C = 12


def ev_rot(objs, call, n=256):
    for o in objs:
        call(o)
    torch.cuda.synchronize()
    e = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
         for _ in range(n)]
    for i, (a, b) in enumerate(e):
        a.record(); call(objs[i % len(objs)]); b.record()
    torch.cuda.synchronize()
    v = sorted(a.elapsed_time(b) for a, b in e)
    return v[len(v) // 2] * 1000, v[int(len(v) * .95)] * 1000


print(f"{'src':5s} {'N':>5s} {'K':>6s} {'M':>3s} {'cur us':>8s} {'p95':>7s} {'new us':>8s} "
      f"{'p95':>7s} {'speedup':>8s} {'maxdiff':>10s}")
for tag, w32 in (("main", W.layers[0].gate_w), ("mtp", W.mtp[0].gate_w)):
    N, K = w32.shape
    c32 = [w32.clone() for _ in range(C)]
    c16 = [w.to(torch.bfloat16).contiguous() for w in c32]
    for M in (1, 2, 4, 6, 8):
        x16 = torch.randn(M, K, device="cuda").to(torch.bfloat16)
        x32 = x16.float()
        best = None
        for bn in (32, 64, 128):
            for tgt in (48, 96, 192):
                for nw in (2, 4):
                    try:
                        t = ev_rot(c16, lambda w: gate_linear(x16, w, block_n=bn, target=tgt,
                                                              num_warps=nw))
                    except Exception:
                        continue
                    if best is None or t[0] < best[0][0]:
                        best = (t, bn, tgt, nw)
        cur = ev_rot(c32, lambda w: R.mm(x32, w))
        y0 = R.mm(x32, w32)
        y1 = gate_linear(x16, c16[0], block_n=best[1], target=best[2], num_warps=best[3])
        print(f"{tag:5s} {N:5d} {K:6d} {M:3d} {cur[0]:8.1f} {cur[1]:7.1f} {best[0][0]:8.1f} "
              f"{best[0][1]:7.1f} {100*(1-best[0][0]/cur[0]):7.1f}% {float((y1-y0).abs().max()):10.2e}"
              f"   bn={best[1]} tgt={best[2]} w={best[3]}", flush=True)
    del c32, c16
    torch.cuda.empty_cache()
