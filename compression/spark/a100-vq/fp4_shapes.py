"""Shape-by-shape: what the 177 _fp4_linear calls of a verify step actually are, and whether FP4
is the faster format for each of them.

The attention group is four matrices per layer whose sizes differ by 8x. A format that halves the
bytes still loses if the small ones are latency-bound, so every shape is timed in both formats on
the same weights, at the decode M the verify block really uses.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

import torch

WORK = os.path.expanduser("~/dsv41-spark/work")
sys.path.insert(0, WORK)
sys.path.insert(0, os.path.join(WORK, "tools"))
from engine.v41_engine import V41Engine                        # noqa: E402
import v41_ref as R                                            # noqa: E402
import engine.fastdecode as FD                                 # noqa: E402
from fp4_linear import quantize_fp8_to_fp4, FP4Weight          # noqa: E402
from fp8_linear import FP8Weight                               # noqa: E402

md = os.path.expanduser("~/dsv41-spark/models/DeepSeek-V4.1-Flash")
os.environ["DSV41_DENSE_FP4"] = "off"          # load fp8, quantize copies here so both exist
eng = V41Engine(md, max_seq=8192, spec=True,
                trace_stats="results/trace-full-20260910/stats/coverage.json",
                arena_gb=40.0, transient_slots=64, keep_free_gb=12.0, prune_keep=None,
                expert_format="cb3")
W = eng.model.W
M = int(os.environ.get("DECODE_M", str(FD.T_VERIFY)))
print(f"M = {M} (T_VERIFY)", flush=True)


def fp8_bytes(w):
    return w.w.numel() + w.s.numel() * 4


def fp4_bytes(w):
    return w.w.numel() + w.s.numel() * 4


def ev(fn, n=200):
    for _ in range(30):
        fn()
    torch.cuda.synchronize()
    e = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
         for _ in range(n)]
    for a, b in e:
        a.record(); fn(); b.record()
    torch.cuda.synchronize()
    v = sorted(x.elapsed_time(y) for x, y in e)
    return v[len(v) // 2] * 1000          # us


NAMES = ("wq_a", "wq_b", "wkv", "wo_b")
per_layer = []
b0 = W.layers[0]
print(f"\n{'name':8s} {'K':>7s} {'N':>7s} {'fp8 MB':>8s} {'fp4 MB':>8s} "
      f"{'fp8 us':>8s} {'fp4 us':>8s} {'fp8 GB/s':>9s} {'fp4 GB/s':>9s} {'winner':>7s}")
tot = defaultdict(float)
for nm in NAMES:
    w8 = getattr(b0, nm)
    if not isinstance(w8, FP8Weight):
        print(f"{nm}: not an FP8Weight ({type(w8).__name__})"); continue
    w4 = quantize_fp8_to_fp4(w8)
    K = w8.K if hasattr(w8, "K") else w8.w.shape[1]
    N = w8.w.shape[0]
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    t8 = ev(lambda: R.dense(x, w8))
    t4 = ev(lambda: R.dense(x, w4))
    b8, b4 = fp8_bytes(w8) / 2**20, fp4_bytes(w4) / 2**20
    g8, g4 = b8 * 2**20 / (t8 * 1e-6) / 1e9, b4 * 2**20 / (t4 * 1e-6) / 1e9
    win = "fp4" if t4 < t8 else "FP8"
    print(f"{nm:8s} {K:7d} {N:7d} {b8:8.2f} {b4:8.2f} {t8:8.1f} {t4:8.1f} "
          f"{g8:9.1f} {g4:9.1f} {win:>7s}")
    tot["fp8_us"] += t8 * 40; tot["fp4_us"] += t4 * 40
    tot["fp8_mb"] += b8 * 40; tot["fp4_mb"] += b4 * 40
    per_layer.append((nm, t8, t4))
    del w4
    torch.cuda.empty_cache()

# the head, which _fp4_linear also serves, twice per pair
h = W.head
hb = h.w.numel() + h.s.numel() * 4 if not torch.is_tensor(h) else h.numel() * 2
xh = torch.randn(M, 5120, dtype=torch.bfloat16, device="cuda")
th = ev(lambda: R.head_logits(xh, h))
print(f"{'head':8s} {5120:7d} {129280:7d} {'':8s} {hb/2**20:8.2f} {'':8s} {th:8.1f} "
      f"{'':9s} {hb/(th*1e-6)/1e9:9.1f}")

print(f"\n40 layers of attention: fp8 {tot['fp8_us']/1000:7.3f} ms / {tot['fp8_mb']/1024:5.3f} GB"
      f"   fp4 {tot['fp4_us']/1000:7.3f} ms / {tot['fp4_mb']/1024:5.3f} GB")
print(f"  fp4 saves {tot['fp8_mb']/1024 - tot['fp4_mb']/1024:.3f} GB and "
      f"{(tot['fp8_us']-tot['fp4_us'])/1000:.3f} ms per verify step")
print(f"  fp4 aggregate {tot['fp4_mb']*2**20/(tot['fp4_us']*1e-6)/1e9:.1f} GB/s   "
      f"fp8 aggregate {tot['fp8_mb']*2**20/(tot['fp8_us']*1e-6)/1e9:.1f} GB/s")
