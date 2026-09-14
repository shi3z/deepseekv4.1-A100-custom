"""Is the small-N attention projection latency-bound on too few blocks? Sweep the tile, no edits.

`pick_block_n` returns 32 for every decode-sized call, so wkv (N=512) launches 16 programs onto
48 SMs and wq_a (N=1280) launches 40. Both take ~47 us for very different byte counts, which is
the signature of a serial K loop rather than a bandwidth limit. fp4_linear already accepts
block_n / num_warps / num_stages, so the hypothesis can be tested before touching anything.
"""
from __future__ import annotations

import itertools
import os
import sys

import torch

WORK = os.path.expanduser("~/dsv41-spark/work")
sys.path.insert(0, WORK)
sys.path.insert(0, os.path.join(WORK, "tools"))
from engine.v41_engine import V41Engine                        # noqa: E402
import engine.fastdecode as FD                                 # noqa: E402
from fp4_linear import quantize_fp8_to_fp4, fp4_linear         # noqa: E402
from fp8_linear import FP8Weight, fp8_linear                   # noqa: E402

md = os.path.expanduser("~/dsv41-spark/models/DeepSeek-V4.1-Flash")
os.environ["DSV41_DENSE_FP4"] = "off"
eng = V41Engine(md, max_seq=8192, spec=True,
                trace_stats="results/trace-full-20260910/stats/coverage.json",
                arena_gb=30.0, transient_slots=64, keep_free_gb=12.0, prune_keep=None,
                expert_format="cb3")
W = eng.model.W
M = FD.T_VERIFY
COPIES = 12
b0 = W.layers[0]
print(f"M={M}, {COPIES} DRAM-resident copies per shape\n", flush=True)


def ev_rot(objs, call, n=96):
    for o in objs:
        call(o)
    torch.cuda.synchronize()
    e = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
         for _ in range(n)]
    for i, (a, b) in enumerate(e):
        a.record(); call(objs[i % len(objs)]); b.record()
    torch.cuda.synchronize()
    v = sorted(x.elapsed_time(y) for x, y in e)
    return v[len(v) // 2] * 1000


for nm in ("wq_a", "wkv", "wq_b", "wo_b"):
    w8 = getattr(b0, nm)
    K = w8.K if hasattr(w8, "K") else w8.w.shape[1]
    N = w8.w.shape[0]
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    c4 = [quantize_fp8_to_fp4(w8) for _ in range(COPIES)]
    mb = (c4[0].w.numel() + c4[0].s.numel() * 4) / 2**20
    best = (1e9, None)
    out = []
    for bn, nw, ns in itertools.product((8, 16, 32, 64, 128), (1, 2, 4, 8), (2, 3, 4)):
        if N % bn:
            continue
        try:
            t = ev_rot(c4, lambda w: fp4_linear(x, w, block_n=bn, num_warps=nw, num_stages=ns))
        except Exception:
            continue
        out.append((t, bn, nw, ns))
        if t < best[0]:
            best = (t, (bn, nw, ns))
    out.sort()
    cur = ev_rot(c4, lambda w: fp4_linear(x, w))
    print(f"{nm:6s} N={N:6d} K={K:5d} {mb:6.2f} MB   default(bn=32,w4,s3) {cur:7.1f} us "
          f"({mb*2**20/(cur*1e-6)/1e9:6.1f} GB/s)")
    for t, bn, nw, ns in out[:5]:
        print(f"        bn={bn:4d} warps={nw} stages={ns}  grid={-(-N//bn):5d}  {t:7.1f} us "
              f"({mb*2**20/(t*1e-6)/1e9:6.1f} GB/s)  {100*(t-cur)/cur:+6.1f} %")
    del c4
    torch.cuda.empty_cache()
    print(flush=True)
