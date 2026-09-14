"""Does hc_post become exact if the accumulation follows torch's order?

torch computes `mixed = einsum(comb, residual)` and then `post*x + mixed`; the kernel started from
`post*x` and accumulated the four comb terms into it. Same values, different order, and the
teacher-forced NLL moved 0.245 % / 0.415 % because of it. Three orderings are tried here against
the torch result on real-shaped data.
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
import elem_fused as EF                                        # noqa: E402


@triton.jit
def _k(X, RES, POST, COMB, Y, D, HC: tl.constexpr, ORDER: tl.constexpr, BLOCK: tl.constexpr):
    s = tl.program_id(0)
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    md = d < D
    xv = tl.load(X + s * D + d, mask=md, other=0.0).to(tl.float32)
    for j in tl.static_range(HC):
        if ORDER == 0:                       # post*x first, then accumulate comb (the first try)
            acc = tl.load(POST + s * HC + j).to(tl.float32) * xv
            for i in tl.static_range(HC):
                r = tl.load(RES + (s * HC + i) * D + d, mask=md, other=0.0).to(tl.float32)
                acc += tl.load(COMB + (s * HC + i) * HC + j).to(tl.float32) * r
        else:                                # mixed first, then add post*x -- torch's order
            mixed = tl.zeros([BLOCK], dtype=tl.float32)
            for i in tl.static_range(HC):
                r = tl.load(RES + (s * HC + i) * D + d, mask=md, other=0.0).to(tl.float32)
                mixed += tl.load(COMB + (s * HC + i) * HC + j).to(tl.float32) * r
            acc = tl.load(POST + s * HC + j).to(tl.float32) * xv + mixed
        tl.store(Y + (s * HC + j) * D + d, acc.to(tl.bfloat16), mask=md)


def fused(x, res, post, comb, order, block=1024, warps=4):
    s, hc, d = res.shape
    y = torch.empty((s, hc, d), dtype=x.dtype, device=x.device)
    _k[(s, triton.cdiv(d, block))](x, res, post.float().contiguous(), comb.float().contiguous(),
                                   y, d, HC=hc, ORDER=order, BLOCK=block, num_warps=warps)
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


torch.manual_seed(0)
for T in (2, 6):
    HC, D = 4, 5120
    x = torch.randn(T, D, dtype=torch.bfloat16, device="cuda")
    res = torch.randn(T, HC, D, dtype=torch.bfloat16, device="cuda")
    post = torch.randn(T, HC, device="cuda")
    comb = torch.randn(T, HC, HC, device="cuda")
    EF.HC_POST = False          # the reference must be the torch path, not the kernel
    ref = R.hc_post(x, res, post, comb)
    base = ev(lambda: R.hc_post(x, res, post, comb))
    for order, name in ((0, "post*x then comb"), (1, "comb then post*x (torch order)")):
        got = fused(x, res, post, comb, order)
        t = ev(lambda: fused(x, res, post, comb, order))
        diff = int((ref != got).sum())
        print(f"T={T} {name:32s} {t:7.2f} us (torch {base:7.2f})  exact={torch.equal(ref, got)}  "
              f"differing {diff}/{ref.numel()}  max|d| {float((ref.float()-got.float()).abs().max()):.3e}",
              flush=True)
