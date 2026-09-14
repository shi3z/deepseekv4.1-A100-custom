"""hc_post as one kernel: y[j] = post[j]*x + sum_i comb[i,j]*residual[i], hc = 4.

Currently an einsum over a [2,4,4] mixing matrix, two fp32 casts, a broadcast multiply, an add and
a cast back -- six launches for 5120 columns of four-way mixing, twice per layer, 80 times a step.
"""
from __future__ import annotations

import os
import sys

import torch
import triton
import triton.language as tl

WORK = os.path.expanduser("~/dsv41-spark/work")
sys.path.insert(0, WORK)
sys.path.insert(0, os.path.join(WORK, "tools"))
import v41_ref as R                                            # noqa: E402


@triton.jit
def _hc_post_kernel(X, RES, POST, COMB, Y, D, HC: tl.constexpr, BLOCK: tl.constexpr):
    s = tl.program_id(0)
    blk = tl.program_id(1)
    d = blk * BLOCK + tl.arange(0, BLOCK)
    md = d < D
    xv = tl.load(X + s * D + d, mask=md, other=0.0).to(tl.float32)
    for j in tl.static_range(HC):
        acc = tl.load(POST + s * HC + j).to(tl.float32) * xv
        for i in tl.static_range(HC):
            r = tl.load(RES + (s * HC + i) * D + d, mask=md, other=0.0).to(tl.float32)
            acc += tl.load(COMB + (s * HC + i) * HC + j).to(tl.float32) * r
        tl.store(Y + (s * HC + j) * D + d, acc.to(tl.bfloat16), mask=md)


def fused_hc_post(x, residual, post, comb, block=1024, warps=4):
    s, hc, d = residual.shape
    y = torch.empty((s, hc, d), dtype=x.dtype, device=x.device)
    _hc_post_kernel[(s, triton.cdiv(d, block))](x, residual, post.float().contiguous(),
                                                comb.float().contiguous(), y, d,
                                                HC=hc, BLOCK=block, num_warps=warps)
    return y


def ev(fn, n=300):
    for _ in range(40):
        fn()
    torch.cuda.synchronize()
    e = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
         for _ in range(n)]
    for a, b in e:
        a.record(); fn(); b.record()
    torch.cuda.synchronize()
    v = sorted(a.elapsed_time(b) for a, b in e)
    return v[len(v) // 2] * 1000


T, HC, D = 2, 4, 5120
x = torch.randn(T, D, dtype=torch.bfloat16, device="cuda")
res = torch.randn(T, HC, D, dtype=torch.bfloat16, device="cuda")
post = torch.randn(T, HC, device="cuda")
comb = torch.randn(T, HC, HC, device="cuda")
base = ev(lambda: R.hc_post(x, res, post, comb))
ref = R.hc_post(x, res, post, comb)
best = None
for blk in (256, 512, 1024, 2048):
    for w in (1, 2, 4, 8):
        try:
            t = ev(lambda: fused_hc_post(x, res, post, comb, blk, w))
        except Exception:
            continue
        got = fused_hc_post(x, res, post, comb, blk, w)
        ok = bool(torch.equal(ref, got))
        d = float((ref.float() - got.float()).abs().max())
        if best is None or t < best[0]:
            best = (t, blk, w, ok, d)
print(f"hc_post [{T},{HC},{D}]  torch {base:7.2f} us   fused {best[0]:7.2f} us "
      f"({100*(1-best[0]/base):+6.1f} %)  block={best[1]} warps={best[2]} exact={best[3]} "
      f"max|d|={best[4]:.2e}")
for calls in (40, 80):
    print(f"   x{calls} calls/step = {(base-best[0])*calls/1000:+.3f} ms/step eager, "
          f"{(base-best[0])*calls/1000*0.57:+.3f} expected in-graph")
