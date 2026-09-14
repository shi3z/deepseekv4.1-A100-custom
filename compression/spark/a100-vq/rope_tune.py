"""Second try at the fused RoPE: contiguous tail load, even/odd split in registers, warp sweep.

v1 read the rotated pair with a stride of two and lost 2.3x on the q shape, which is 128 short
rows. Here the 64-wide tail is loaded in one contiguous block and split the way the CB3 kernel
splits its nibbles, and several rows are handled per program so a launch has enough to do.
"""
from __future__ import annotations

import itertools
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
def _rope2(X, FR, FI, Y, D, RD, sx, sy, M, HEADS, INV: tl.constexpr,
           ROWS: tl.constexpr, BLOCK: tl.constexpr, BLOCK_RD: tl.constexpr):
    r0 = tl.program_id(0) * ROWS
    for i in tl.static_range(ROWS):
        row = r0 + i
        if row < M:
            keep = D - RD
            c = tl.arange(0, BLOCK)
            mk = c < keep
            tl.store(Y + row * sy + c, tl.load(X + row * sx + c, mask=mk, other=0), mask=mk)
            k = tl.arange(0, BLOCK_RD)
            mr = k < RD
            t = tl.load(X + row * sx + keep + k, mask=mr, other=0.0).to(tl.float32)
            a, b = tl.split(tl.reshape(t, [BLOCK_RD // 2, 2]))   # even / odd of adjacent pairs
            j = tl.arange(0, BLOCK_RD // 2)
            mj = j < (RD // 2)
            tok = row // HEADS
            fr = tl.load(FR + tok * (RD // 2) + j, mask=mj, other=0.0)
            fi = tl.load(FI + tok * (RD // 2) + j, mask=mj, other=0.0)
            if INV:
                fi = -fi
            re = a * fr - b * fi
            im = a * fi + b * fr
            out = tl.reshape(tl.join(re, im), [BLOCK_RD])
            tl.store(Y + row * sy + keep + k, out.to(tl.bfloat16), mask=mr)


def fused_rope(x, fq, rd, inverse=False, rows=1, warps=4):
    D = x.shape[-1]
    x2 = x.reshape(-1, D)
    M, T = x2.shape[0], fq.shape[0]
    f = torch.view_as_real(fq)
    y = torch.empty_like(x2)
    _rope2[(triton.cdiv(M, rows),)](x2, f[..., 0].contiguous(), f[..., 1].contiguous(), y, D, rd,
                                    x2.stride(0), y.stride(0), M, M // T, inverse,
                                    ROWS=rows, BLOCK=triton.next_power_of_2(D - rd),
                                    BLOCK_RD=triton.next_power_of_2(rd), num_warps=warps)
    return y.view(x.shape)


def torch_rope(x, fq, rd, inverse=False):
    return torch.cat([x[..., :-rd], R.apply_rotary(x[..., -rd:], fq, inverse=inverse)], dim=-1)


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


RD, T = 64, 2
for name, shape, calls in (("q [T,64,512]", (T, 64, 512), 43), ("kv [T,512]", (T, 512), 43),
                           ("o inv [T,4096]", (T, 4096), 40)):
    x = torch.randn(*shape, dtype=torch.bfloat16, device="cuda")
    fq = torch.view_as_complex(torch.randn(T, RD // 2, 2, device="cuda").contiguous())
    inv = "inv" in name
    base = ev(lambda: torch_rope(x, fq, RD, inv))
    best = None
    for rows, w in itertools.product((1, 2, 4, 8), (1, 2, 4, 8)):
        try:
            t = ev(lambda: fused_rope(x, fq, RD, inv, rows, w))
        except Exception as e:
            continue
        ok = torch.equal(torch_rope(x, fq, RD, inv), fused_rope(x, fq, RD, inv, rows, w))
        if best is None or t < best[0]:
            best = (t, rows, w, ok)
    if best is None:
        print(f"{name:16s} all configs failed"); continue
    print(f"{name:16s} torch {base:7.2f} us   best fused {best[0]:7.2f} us "
          f"({100*(1-best[0]/base):+6.1f} %)  rows={best[1]} warps={best[2]} exact={best[3]}  "
          f"x{calls} = {(base-best[0])*calls/1000:+.3f} ms/step", flush=True)
