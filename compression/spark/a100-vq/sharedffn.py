"""The shared-expert FFN, measured: the two GEMMs that read the same x, the tile they run on,
and the elementwise chain between them.

`fp8_linear` hardcodes BLOCK_N=128, so w1/w3 at N=2304 launch 18 programs onto 48 SMs -- the same
grid starvation that cost 27 % of the attention projections' time. And `expert_ffn` materialises
gate and up in fp32, clamps each, silus one, multiplies and casts back: seven small kernels per
layer on [T, I] tensors that are 18 kB. Whether either is worth fusing is a measurement, not a
guess, so all of it is timed at the real shapes with the weights coming from DRAM.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F
import triton

WORK = os.path.expanduser("~/dsv41-spark/work")
sys.path.insert(0, WORK)
sys.path.insert(0, os.path.join(WORK, "tools"))
from engine.v41_engine import V41Engine                        # noqa: E402
import v41_ref as R                                            # noqa: E402
import engine.fastdecode as FD                                 # noqa: E402
from fp8_linear import FP8Weight, fp8_linear, _fp8_linear_kernel  # noqa: E402

md = os.path.expanduser("~/dsv41-spark/models/DeepSeek-V4.1-Flash")
eng = V41Engine(md, max_seq=8192, spec=True,
                trace_stats="results/trace-full-20260910/stats/coverage.json",
                arena_gb=20.0, transient_slots=64, keep_free_gb=12.0, prune_keep=None,
                expert_format="cb3")
W = eng.model.W
b0 = W.layers[0]
T = FD.T_VERIFY
lim = eng.args.swiglu_limit
print(f"T={T} swiglu_limit={lim}", flush=True)
for nm in ("sh_w1", "sh_w2", "sh_w3"):
    w = getattr(b0, nm)
    print(f"  {nm}: {type(w).__name__} N={w.N} K={w.K} "
          f"{(w.w.numel()+w.s.numel()*4)/2**20:.2f} MB  grid@128={triton.cdiv(w.N,128)}", flush=True)

C = 10


def ev(fn, n=300):
    for _ in range(30):
        fn()
    torch.cuda.synchronize()
    e = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
         for _ in range(n)]
    for a, b in e:
        a.record(); fn(); b.record()
    torch.cuda.synchronize()
    v = sorted(a.elapsed_time(b) for a, b in e)
    return v[len(v) // 2] * 1000


def call_tiled(x, w, bn, nw, ns):
    y = torch.empty(x.size(0), w.N, dtype=torch.bfloat16, device=x.device)
    _fp8_linear_kernel[(triton.cdiv(w.N, bn), triton.cdiv(x.size(0), 16))](
        x, w.w, w.s, y, x.size(0), w.N, w.K, x.stride(0), w.w.stride(0), w.s.stride(0),
        y.stride(0), BLOCK_M=16, BLOCK_N=bn, BLOCK_K=128, num_warps=nw, num_stages=ns)
    return y


print(f"\n{'mat':6s} {'N':>6s} {'MB':>6s} {'cur us':>8s} {'GB/s':>7s} {'best us':>8s} "
      f"{'GB/s':>7s} {'gain':>7s}  config")
tot_cur = tot_best = 0.0
x = torch.randn(T, b0.sh_w1.K, dtype=torch.bfloat16, device="cuda")
for nm in ("sh_w1", "sh_w3", "sh_w2"):
    w0 = getattr(b0, nm)
    cs = [FP8Weight(w0.w.clone(), w0.s.clone()) for _ in range(C)]
    xin = x if nm != "sh_w2" else torch.randn(T, w0.K, dtype=torch.bfloat16, device="cuda")
    mb = (w0.w.numel() + w0.s.numel() * 4) / 2**20
    cur = ev(lambda: fp8_linear(xin, cs[0]))
    cur = ev(lambda: [fp8_linear(xin, c) for c in cs][0], n=60) / C
    best = None
    for bn in (16, 32, 64, 128, 256):
        if w0.N % bn:
            continue
        for nw in (1, 2, 4, 8):
            for ns in (2, 3):
                try:
                    t = ev(lambda: [call_tiled(xin, c, bn, nw, ns) for c in cs][0], n=60) / C
                except Exception:
                    continue
                if best is None or t < best[0]:
                    best = (t, bn, nw, ns)
    tot_cur += cur; tot_best += best[0]
    print(f"{nm:6s} {w0.N:6d} {mb:6.2f} {cur:8.1f} {mb*2**20/(cur*1e-6)/1e9:7.1f} "
          f"{best[0]:8.1f} {mb*2**20/(best[0]*1e-6)/1e9:7.1f} {100*(1-best[0]/cur):6.1f}%  "
          f"bn={best[1]} warps={best[2]} stages={best[3]} grid={triton.cdiv(w0.N,best[1])}",
          flush=True)
    del cs
    torch.cuda.empty_cache()
print(f"\n  3 GEMMs per layer: {tot_cur:.1f} -> {tot_best:.1f} us   "
      f"x40 layers: {tot_cur*40/1000:.3f} -> {tot_best*40/1000:.3f} ms/step "
      f"(save {(tot_cur-tot_best)*40/1000:.3f} ms)", flush=True)

# ---- the elementwise chain between the GEMMs
g = torch.randn(T, b0.sh_w1.N, dtype=torch.bfloat16, device="cuda")
u = torch.randn(T, b0.sh_w1.N, dtype=torch.bfloat16, device="cuda")


def chain():
    gate = g.float()
    up = u.float()
    up = torch.clamp(up, min=-lim, max=lim)
    gate = torch.clamp(gate, max=lim)
    h = F.silu(gate) * up
    return h.to(torch.bfloat16)


t_chain = ev(chain)
print(f"  elementwise chain (2 casts, 2 clamps, silu, mul, cast): {t_chain:7.2f} us/layer "
      f"= {t_chain*40/1000:.3f} ms/step   intermediates {T*b0.sh_w1.N*4/1024:.1f} KB", flush=True)

# ---- the routed+shared merge in _layer_b
r = torch.randn(T, 5120, dtype=torch.bfloat16, device="cuda")
s = torch.randn(T, 5120, dtype=torch.bfloat16, device="cuda")


def merge():
    out = r.float()
    out += s.float()
    return out.to(torch.bfloat16)


t_merge = ev(merge)
print(f"  routed+shared merge (2 casts, add, cast):               {t_merge:7.2f} us/layer "
      f"= {t_merge*40/1000:.3f} ms/step", flush=True)
