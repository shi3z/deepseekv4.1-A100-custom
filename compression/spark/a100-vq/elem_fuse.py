"""Two elementwise chains of the decode step, fused into one kernel each, checked for bit-equality.

expert_ffn spends 34.3 us a layer turning two [2, 2304] bf16 GEMM results into one, and _layer_b
spends 20.5 us adding two [2, 5120] bf16 tensors -- 18 kB and 40 kB of data. Neither is bandwidth:
it is seven and four launches per layer, 440 per step between them. Fusing each into a single
kernel keeps the arithmetic identical (fp32 in registers, one rounding at the end), which is the
thing to verify before anything else.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _swiglu_kernel(G, U, O, n, limit, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    g = tl.load(G + off, mask=m, other=0.0).to(tl.float32)
    u = tl.load(U + off, mask=m, other=0.0).to(tl.float32)
    u = tl.minimum(tl.maximum(u, -limit), limit)      # clamp(up, -limit, limit)
    g = tl.minimum(g, limit)                          # clamp(gate, max=limit)
    h = (g * tl.sigmoid(g)) * u                       # silu(gate) * up
    tl.store(O + off, h.to(tl.bfloat16), mask=m)


def fused_swiglu(g, u, limit):
    o = torch.empty_like(g)
    n = g.numel()
    _swiglu_kernel[(triton.cdiv(n, 1024),)](g, u, o, n, float(limit), BLOCK=1024, num_warps=4)
    return o


@triton.jit
def _merge_kernel(R, S, O, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    r = tl.load(R + off, mask=m, other=0.0).to(tl.float32)
    s = tl.load(S + off, mask=m, other=0.0).to(tl.float32)
    tl.store(O + off, (r + s).to(tl.bfloat16), mask=m)


def fused_merge(r, s):
    o = torch.empty_like(r)
    n = r.numel()
    _merge_kernel[(triton.cdiv(n, 1024),)](r, s, o, n, BLOCK=1024, num_warps=4)
    return o


def ev(fn, n=400):
    for _ in range(50):
        fn()
    torch.cuda.synchronize()
    e = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
         for _ in range(n)]
    for a, b in e:
        a.record(); fn(); b.record()
    torch.cuda.synchronize()
    v = sorted(a.elapsed_time(b) for a, b in e)
    return v[len(v) // 2] * 1000, v[int(len(v) * .95)] * 1000


LIM = 10.0
print(f"{'chain':10s} {'T':>3s} {'N':>6s} {'torch us':>9s} {'p95':>7s} {'fused us':>9s} "
      f"{'p95':>7s} {'gain':>7s} {'x40 ms':>8s} {'bit-exact':>10s}")
for T in (1, 2, 4, 6, 8):
    g = (torch.randn(T, 2304, device="cuda") * 6).to(torch.bfloat16)
    u = (torch.randn(T, 2304, device="cuda") * 6).to(torch.bfloat16)

    def torch_swiglu():
        gate = g.float()
        up = u.float()
        up = torch.clamp(up, min=-LIM, max=LIM)
        gate = torch.clamp(gate, max=LIM)
        return (F.silu(gate) * up).to(torch.bfloat16)

    a = ev(torch_swiglu)
    b = ev(lambda: fused_swiglu(g, u, LIM))
    r1, r2 = torch_swiglu(), fused_swiglu(g, u, LIM)
    ex = bool(torch.equal(r1, r2))
    print(f"{'swiglu':10s} {T:3d} {2304:6d} {a[0]:9.2f} {a[1]:7.2f} {b[0]:9.2f} {b[1]:7.2f} "
          f"{100*(1-b[0]/a[0]):6.1f}% {(a[0]-b[0])*40/1000:8.3f} {str(ex):>10s}")
    if not ex:
        d = (r1.float() - r2.float()).abs()
        print(f"           max|d| {float(d.max()):.3e}  differing {int((r1!=r2).sum())}/{r1.numel()}")

    rr = torch.randn(T, 5120, dtype=torch.bfloat16, device="cuda")
    ss = torch.randn(T, 5120, dtype=torch.bfloat16, device="cuda")

    def torch_merge():
        out = rr.float()
        out += ss.float()
        return out.to(torch.bfloat16)

    a = ev(torch_merge)
    b = ev(lambda: fused_merge(rr, ss))
    r1, r2 = torch_merge(), fused_merge(rr, ss)
    ex = bool(torch.equal(r1, r2))
    print(f"{'merge':10s} {T:3d} {5120:6d} {a[0]:9.2f} {a[1]:7.2f} {b[0]:9.2f} {b[1]:7.2f} "
          f"{100*(1-b[0]/a[0]):6.1f}% {(a[0]-b[0])*40/1000:8.3f} {str(ex):>10s}", flush=True)
    if not ex:
        d = (r1.float() - r2.float()).abs()
        print(f"           max|d| {float(d.max()):.3e}  differing {int((r1!=r2).sum())}/{r1.numel()}")
