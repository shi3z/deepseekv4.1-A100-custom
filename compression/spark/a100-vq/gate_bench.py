"""Is a bf16-storage fp32-accumulate gate GEMM actually faster than cuBLAS on the fp32 copy?

fp32_skinny's own note says this kernel shape loses to cuBLAS at N=512 even with fp32 weights, so
halving the bytes is not enough on its own. Real weights, DRAM-resident (one copy per rep, past
L2), at the decode M the verify block uses. Accuracy is only worth measuring if the speed is there.
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
import engine.fastdecode as FD                                 # noqa: E402
from gate_gemm import gate_linear, plan                        # noqa: E402

md = os.path.expanduser("~/dsv41-spark/models/DeepSeek-V4.1-Flash")
eng = V41Engine(md, max_seq=8192, spec=True,
                trace_stats="results/trace-full-20260910/stats/coverage.json",
                arena_gb=20.0, transient_slots=64, keep_free_gb=12.0, prune_keep=None,
                expert_format="cb3")
W = eng.model.W
M = FD.T_VERIFY
w32 = W.layers[0].gate_w
N, K = w32.shape
print(f"gate [{N}, {K}]  fp32 {w32.numel()*4/2**20:.2f} MB  bf16 {w32.numel()*2/2**20:.2f} MB  M={M}",
      flush=True)
C = 12
c32 = [w32.clone() for _ in range(C)]
c16 = [w.to(torch.bfloat16).contiguous() for w in c32]
x32 = torch.randn(M, K, device="cuda")
x16 = x32.to(torch.bfloat16)
x32 = x16.float()                       # exactly a bf16 activation promoted, as the engine does


def ev_rot(objs, call, n=128):
    for o in objs:
        call(o)
    torch.cuda.synchronize()
    e = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
         for _ in range(n)]
    for i, (a, b) in enumerate(e):
        a.record(); call(objs[i % len(objs)]); b.record()
    torch.cuda.synchronize()
    v = sorted(a.elapsed_time(b) for a, b in e)
    return v[len(v) // 2] * 1000


mb32 = w32.numel() * 4 / 2**20
mb16 = w32.numel() * 2 / 2**20
t_cublas = ev_rot(c32, lambda w: F.linear(x32, w))
t_mm = ev_rot(c32, lambda w: R.mm(x32, w))
print(f"\n{'path':38s} {'us':>8s} {'MB':>7s} {'GB/s':>8s}")
print(f"{'cuBLAS F.linear(fp32 x, fp32 w)':38s} {t_cublas:8.1f} {mb32:7.2f} "
      f"{mb32*2**20/(t_cublas*1e-6)/1e9:8.1f}")
print(f"{'R.mm (what the engine calls)':38s} {t_mm:8.1f} {mb32:7.2f} "
      f"{mb32*2**20/(t_mm*1e-6)/1e9:8.1f}", flush=True)

best = None
for bn in (32, 64, 128, 192, 384):
    for tgt in (48, 96, 192):
        for nw in (2, 4, 8):
            try:
                t = ev_rot(c16, lambda w: gate_linear(x16, w, block_n=bn, target=tgt, num_warps=nw))
            except Exception as e:
                continue
            b_, sk = plan(N, K, bn, tgt)
            if best is None or t < best[0]:
                best = (t, bn, tgt, nw, sk)
print(f"{'gate_linear (bf16 w, fp32 accum)':38s} {best[0]:8.1f} {mb16:7.2f} "
      f"{mb16*2**20/(best[0]*1e-6)/1e9:8.1f}   block_n={best[1]} target={best[2]} "
      f"warps={best[3]} split_k={best[4]}", flush=True)
print(f"\n  vs cuBLAS: {100*(best[0]-t_mm)/t_mm:+6.1f} %   "
      f"per step over 40 layers: {(best[0]-t_mm)*40/1000:+6.3f} ms", flush=True)

y_ref = R.mm(x32, w32)
y_new = gate_linear(x16, c16[0], block_n=best[1], target=best[2], num_warps=best[3])
d = (y_new - y_ref).abs()
print(f"\nnumerics vs the reference path: max|d| {float(d.max()):.3e}  "
      f"mean|d| {float(d.mean()):.3e}  rel {float(d.max()/y_ref.abs().max()):.3e}", flush=True)
