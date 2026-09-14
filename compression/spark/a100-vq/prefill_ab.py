"""Prefill-sized MoE call: CB3 v3 (which has a dedicated FP4-scratch path) vs CBF8 (which does not)."""
import os, sys, time, itertools
import torch
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work/tools"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/a100-vq"))
import cb3_moe as C3, fp4_moe as F4, cbf8_moe as F8
from engine.codebook_sim import CodebookSim
from cbf8 import CBF8
dev = "cuda"; torch.manual_seed(0)
DIM, INTER = F4.DIM, F4.INTER
S = 32; T = int(os.environ.get("T", 512)); K = 6
w = lambda n, k: torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=dev)
sc = lambda n, k: torch.randint(124, 126, (n, k // 32), dtype=torch.uint8, device=dev)
cb = C3.CB3ArenaV2(S, dev); cb.sim = CodebookSim(3, dev)
f8 = F8.CBF8Arena(S, dev).attach(CBF8(dev))
for i in range(S):
    args = (w(INTER, DIM), sc(INTER, DIM), w(DIM, INTER), sc(DIM, INTER), w(INTER, DIM), sc(INTER, DIM))
    cb.load_slot(i, *args); f8.load_slot(i, *args)
x = torch.randn(T, DIM, dtype=torch.bfloat16, device=dev) * 0.3
slots = torch.randint(0, S, (T, K), dtype=torch.int32, device=dev)
wts = torch.rand(T, K, device=dev)

def bench(fn, n=5):
    for _ in range(2): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / n * 1e3

a = bench(lambda: C3.moe_forward_v3(x, slots, wts, cb, 10.0))
d = bench(lambda: F8.moe_forward_cbf8(x, slots, wts, f8, 10.0))
print(f"T={T} K={K}:  CB3 v3 {a:8.2f} ms   CBF8 (decode cfg) {d:8.2f} ms  ({(d-a)/a*100:+.0f} %)")
best = (1e9, None)
for bm, bn, nw, ns in itertools.product((64, 128, 256), (32, 64, 128), (4, 8), (1, 2)):
    try:
        t = bench(lambda: F8.moe_forward_cbf8(x, slots, wts, f8, 10.0, block_m=bm,
                                              cfg_up=(bn, nw, ns), cfg_down=(bn, nw, ns)), n=3)
    except Exception:
        continue
    if t < best[0]:
        best = (t, (bm, bn, nw, ns))
        print(f"        BM={bm} cfg {(bn, nw, ns)} {t:8.2f} ms", flush=True)
print(f"        CBF8 best prefill (BM, BN, warps, stages)={best[1]} {best[0]:8.2f} ms  "
      f"({(best[0]-a)/a*100:+.0f} % vs CB3)")
