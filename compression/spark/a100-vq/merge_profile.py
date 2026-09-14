"""What the MoE output merge actually costs, and what a fused version could return.

_layer_b runs, per layer:  out = moe.float(); out += shared.float(); out.to(bf16)
at [T, 5120]. At decode T that is 123 kB of temporaries per layer -- 4.9 MB across 40 layers
against ~11 GB of weights -- so this cannot be a bandwidth problem; if it costs anything it is
four tiny kernel launches per layer. Measured with CUDA events at the real shape, plus a fused
Triton version to bound the win before writing anything into the engine.
"""
from __future__ import annotations

import os
import sys

import torch
import triton
import triton.language as tl


@triton.jit
def _merge_kernel(R, S, O, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    msk = off < n
    r = tl.load(R + off, mask=msk, other=0.0).to(tl.float32)
    s = tl.load(S + off, mask=msk, other=0.0).to(tl.float32)
    tl.store(O + off, (r + s).to(tl.bfloat16), mask=msk)


def fused_merge(r, s, out=None):
    """bf16(fp32(r) + fp32(s)) -- the same two upcasts, the same fp32 add and the same single
    rounding the torch sequence performs, with no fp32 temporary and one launch."""
    n = r.numel()
    o = out if out is not None else torch.empty_like(r)
    _merge_kernel[(triton.cdiv(n, 1024),)](r, s, o, n, BLOCK=1024, num_warps=4)
    return o


def ev(fn, n=500):
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


DIM = 5120
print(f"{'T':>3s} {'torch us':>9s} {'p95':>7s} {'fused us':>9s} {'p95':>7s} {'saved us':>9s} "
      f"{'x40 ms':>8s} {'bit-exact':>10s} {'bytes':>9s}")
for T in (1, 2, 4, 6, 8, 16):
    r = torch.randn(T, DIM, dtype=torch.bfloat16, device="cuda")
    s = torch.randn(T, DIM, dtype=torch.bfloat16, device="cuda")

    def torch_path():
        out = r.float()
        out += s.float()
        return out.to(torch.bfloat16)

    a = ev(torch_path)
    b = ev(lambda: fused_merge(r, s))
    ref, new = torch_path(), fused_merge(r, s)
    exact = bool(torch.equal(ref, new))
    by = T * DIM * (2 + 2 + 2)          # two bf16 reads + one bf16 write
    print(f"{T:3d} {a[0]:9.2f} {a[1]:7.2f} {b[0]:9.2f} {b[1]:7.2f} {a[0]-b[0]:9.2f} "
          f"{(a[0]-b[0])*40/1000:8.3f} {str(exact):>10s} {by/1024:8.1f}K", flush=True)
    if not exact:
        d = (ref.float() - new.float()).abs()
        print(f"      max|d| {float(d.max()):.3e}  mean|d| {float(d.mean()):.3e}  "
              f"differing elements {int((ref != new).sum())} of {ref.numel()}")
