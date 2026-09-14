"""A decode-time tile policy for fp4_linear, chosen over every shape the step actually launches.

The sweep on the attention projections said BLOCK_N=16 with one warp beats the default 32/4 on
everything small, but `pick_block_n` is global and the LM head (N=129280, 394 MB, already at
182 GB/s) goes through the same function. A policy that helps a 1.5 MB projection and hurts a
394 MB head would be a net loss, so the head is measured here too before anything changes.
"""
from __future__ import annotations

import os
import sys

import torch

WORK = os.path.expanduser("~/dsv41-spark/work")
sys.path.insert(0, WORK)
sys.path.insert(0, os.path.join(WORK, "tools"))
from engine.v41_engine import V41Engine                        # noqa: E402
import engine.fastdecode as FD                                 # noqa: E402
from fp4_linear import FP4Weight, fp4_linear, quantize_to_fp4  # noqa: E402

md = os.path.expanduser("~/dsv41-spark/models/DeepSeek-V4.1-Flash")
eng = V41Engine(md, max_seq=8192, spec=True,
                trace_stats="results/trace-full-20260910/stats/coverage.json",
                arena_gb=float(os.environ.get("ARENA_GB", "20")), transient_slots=64,
                keep_free_gb=12.0, prune_keep=None, expert_format="cb3")
W = eng.model.W
M = FD.T_VERIFY
b0 = W.layers[0]
qa, kv = b0.wq_a, b0.wkv


def copies(make, n, budget_mb=1200):
    out = []
    for _ in range(n):
        w = make()
        out.append(w)
        if sum((c.w.numel() + c.s.numel()) for c in out) / 2**20 > budget_mb:
            break
    return out


CASES = []
CASES.append(("wqkv(1792)", lambda: FP4Weight(torch.cat([qa.w, kv.w], 0).contiguous(),
                                              torch.cat([qa.s, kv.s], 0).contiguous(),
                                              qa.N + kv.N, qa.K), 12))
CASES.append(("wq_b(32768)", lambda: FP4Weight(b0.wq_b.w.clone(), b0.wq_b.s.clone(),
                                               b0.wq_b.N, b0.wq_b.K), 12))
CASES.append(("wo_b(5120)", lambda: FP4Weight(b0.wo_b.w.clone(), b0.wo_b.s.clone(),
                                              b0.wo_b.N, b0.wo_b.K), 12))
head = W.head
if isinstance(head, FP4Weight):
    CASES.append(("head(129280)", lambda: FP4Weight(head.w.clone(), head.s.clone(),
                                                    head.N, head.K), 3))


def ev_rot(objs, call, n=64):
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


print(f"M={M}\n{'shape':14s} {'MB':>7s} {'default':>9s} "
      + " ".join(f"{f'bn{bn}/w{nw}':>10s}" for bn, nw in
                 ((16, 1), (16, 2), (32, 1), (32, 4), (64, 4))), flush=True)
for name, make, nc in CASES:
    cs = copies(make, nc)
    x = torch.randn(M, cs[0].K, dtype=torch.bfloat16, device="cuda")
    mb = (cs[0].w.numel() + cs[0].s.numel()) / 2**20
    d = ev_rot(cs, lambda w: fp4_linear(x, w))
    line = f"{name:14s} {mb:7.2f} {d:9.1f}"
    for bn, nw in ((16, 1), (16, 2), (32, 1), (32, 4), (64, 4)):
        try:
            t = ev_rot(cs, lambda w: fp4_linear(x, w, block_n=bn, num_warps=nw, num_stages=3))
            line += f" {t:7.1f}{100*(t-d)/d:+3.0f}%"
        except Exception:
            line += f" {'--':>10s}"
    print(line + f"   ({len(cs)} copies, {len(cs)*mb:.0f} MB)", flush=True)
    del cs
    torch.cuda.empty_cache()
