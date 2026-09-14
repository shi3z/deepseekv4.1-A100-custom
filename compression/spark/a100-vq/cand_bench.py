"""Each fusion candidate timed on its own, at the shape and the per-step count the decode uses.

The op table double-counts: an `aten::copy_` row and a `void at::native::...copy...` row are the
same GPU work seen at two levels. Timing the candidate sequences directly avoids that, and it is
the only way to get "what would one kernel cost instead".
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
def _rmsnorm_kernel(X, W, Y, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < n_cols
    x = tl.load(X + row * n_cols + cols, mask=m, other=0.0).to(tl.float32)
    v = tl.sum(x * x, axis=0) / n_cols
    x = x * tl.rsqrt(v + eps)
    w = tl.load(W + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(Y + row * n_cols + cols, (w * x).to(tl.bfloat16), mask=m)


def fused_rmsnorm(x, w, eps):
    y = torch.empty_like(x)
    n = x.shape[-1]
    _rmsnorm_kernel[(x.shape[0],)](x, w, y, n, eps, BLOCK=triton.next_power_of_2(n), num_warps=8)
    return y


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


T = 2
EPS = 1e-6
print(f"{'candidate':30s} {'shape':16s} {'now us':>8s} {'fused us':>9s} {'save us':>8s} "
      f"{'calls':>6s} {'ms/step':>9s} {'graph est':>10s} {'exact':>6s}")
rows = []
for tag, N, calls in (("rmsnorm q_norm", 1280, 43), ("rmsnorm kv_norm", 512, 43),
                      ("rmsnorm (other)", 5120, 43)):
    x = torch.randn(T, N, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(N, dtype=torch.bfloat16, device="cuda")
    a = ev(lambda: R.rmsnorm(x, w, EPS))
    b = ev(lambda: fused_rmsnorm(x, w, EPS))
    r0, r1 = R.rmsnorm(x, w, EPS), fused_rmsnorm(x, w, EPS)
    ex = bool(torch.equal(r0, r1))
    d = (r0.float() - r1.float()).abs().max().item()
    ms = (a - b) * calls / 1000
    rows.append((tag, ms))
    print(f"{tag:30s} [{T},{N}]{'':>7s} {a:8.2f} {b:9.2f} {a-b:8.2f} {calls:6d} {ms:9.3f} "
          f"{ms*0.57:10.3f} {str(ex):>6s}" + (f"   max|d| {d:.2e}" if not ex else ""))

# rope
rd = 64
x = torch.randn(T, 64, 640, dtype=torch.bfloat16, device="cuda")
fq = torch.randn(T, rd // 2, dtype=torch.complex64, device="cuda")


def rope_now():
    return torch.cat([x[..., :-rd], R.apply_rotary(x[..., -rd:], fq)], dim=-1)


a = ev(rope_now)
print(f"{'rope (q, cat+complex)':30s} [{T},64,640]{'':>3s} {a:8.2f} {'-':>9s} {'-':>8s} "
      f"{86:6d} {'-':>9s} {'-':>10s}")

# hc_post / hc_pre
h = torch.randn(T, 4, 5120, dtype=torch.bfloat16, device="cuda")
post = torch.randn(T, 4, dtype=torch.float32, device="cuda")
comb = torch.randn(T, 4, 4, dtype=torch.float32, device="cuda")
out = torch.randn(T, 5120, dtype=torch.bfloat16, device="cuda")
try:
    a = ev(lambda: R.hc_post(out, h, post, comb))
    print(f"{'hc_post':30s} [{T},4,5120]{'':>3s} {a:8.2f} {'-':>9s} {'-':>8s} {40:6d} "
          f"{a*40/1000:9.3f} {a*40/1000*0.57:10.3f}")
except Exception as e:
    print(f"hc_post: {e}")
print(f"\nrmsnorm total: {sum(m for _, m in rows):.3f} ms/step eager, "
      f"{sum(m for _, m in rows)*0.57:.3f} ms/step expected in-graph")
