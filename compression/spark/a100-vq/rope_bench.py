"""One kernel for RoPE instead of cast, view_as_complex, complex multiply, view_as_real, cast, cat.

`_rope` rotates only the last 64 of a 512-wide row and concatenates the untouched head back on, so
the torch form materialises the rotated tail, then a fresh tensor for the concatenation. The copy
census put this path at 419 launches a step -- the largest single group. Here the untouched part is
copied and the tail rotated in the same kernel, with no complex tensor and no cat.
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
def _rope_kernel(X, FR, FI, Y, D, RD, sx, sy, HEADS, INV: tl.constexpr,
                 BLOCK: tl.constexpr, BLOCK_H: tl.constexpr):
    row = tl.program_id(0)
    tok = row // HEADS
    keep = D - RD
    c = tl.arange(0, BLOCK)
    mk = c < keep
    tl.store(Y + row * sy + c, tl.load(X + row * sx + c, mask=mk, other=0), mask=mk)
    j = tl.arange(0, BLOCK_H)
    mj = j < (RD // 2)
    a = tl.load(X + row * sx + keep + 2 * j, mask=mj, other=0.0).to(tl.float32)
    b = tl.load(X + row * sx + keep + 2 * j + 1, mask=mj, other=0.0).to(tl.float32)
    fr = tl.load(FR + tok * (RD // 2) + j, mask=mj, other=0.0)
    fi = tl.load(FI + tok * (RD // 2) + j, mask=mj, other=0.0)
    if INV:
        fi = -fi
    tl.store(Y + row * sy + keep + 2 * j, (a * fr - b * fi).to(tl.bfloat16), mask=mj)
    tl.store(Y + row * sy + keep + 2 * j + 1, (a * fi + b * fr).to(tl.bfloat16), mask=mj)


def fused_rope(x, fq, rd, inverse=False):
    """x [..., D] bf16, fq [T, rd/2] complex64 -> bf16 [..., D]; rotary on the last rd only."""
    D = x.shape[-1]
    x2 = x.reshape(-1, D)
    M = x2.shape[0]
    T = fq.shape[0]
    f = torch.view_as_real(fq)                       # [T, rd/2, 2] fp32 view, no copy
    y = torch.empty_like(x2)
    _rope_kernel[(M,)](x2, f[..., 0].contiguous(), f[..., 1].contiguous(), y, D, rd,
                       x2.stride(0), y.stride(0), M // T, inverse,
                       BLOCK=triton.next_power_of_2(D - rd),
                       BLOCK_H=triton.next_power_of_2(rd // 2), num_warps=4)
    return y.view(x.shape)


def torch_rope(x, fq, rd, inverse=False):
    return torch.cat([x[..., :-rd], R.apply_rotary(x[..., -rd:], fq, inverse=inverse)], dim=-1)


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
    return v[len(v) // 2] * 1000


RD = 64
T = 2
print(f"{'case':22s} {'shape':18s} {'torch us':>9s} {'fused us':>9s} {'gain':>7s} {'calls':>6s} "
      f"{'ms/step':>9s} {'graph est':>10s} {'exact':>6s}")
CASES = [("q [T,64,512]", (T, 64, 512), 43), ("kv [T,512]", (T, 512), 43),
         ("o inverse [T,4096]", (T, 4096), 40), ("indexer [T,32,128]", (T, 32, 128), 6)]
tot = 0.0
for name, shape, calls in CASES:
    x = torch.randn(*shape, dtype=torch.bfloat16, device="cuda")
    fq = torch.randn(T, RD // 2, 2, device="cuda").contiguous()
    fqc = torch.view_as_complex(fq)
    inv = "inverse" in name
    a = ev(lambda: torch_rope(x, fqc, RD, inv))
    b = ev(lambda: fused_rope(x, fqc, RD, inv))
    r0, r1 = torch_rope(x, fqc, RD, inv), fused_rope(x, fqc, RD, inv)
    ex = bool(torch.equal(r0, r1))
    ms = (a - b) * calls / 1000
    tot += ms
    d = (r0.float() - r1.float()).abs().max().item()
    print(f"{name:22s} {str(shape):18s} {a:9.2f} {b:9.2f} {100*(1-b/a):6.1f}% {calls:6d} "
          f"{ms:9.3f} {ms*0.57:10.3f} {str(ex):>6s}" + (f"  max|d| {d:.2e}" if not ex else ""))
print(f"\ntotal: {tot:.3f} ms/step eager, {tot*0.57:.3f} ms/step expected in-graph")
