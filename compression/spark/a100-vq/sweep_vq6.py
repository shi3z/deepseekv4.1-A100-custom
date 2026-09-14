"""(BM, BN, warps, stages) for the VQ12 up and down kernels, timed through the real entry point."""
import os, sys, time, itertools
import torch
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work/tools"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/a100-vq"))
import cb3_moe as C3, fp4_moe as F4, vq12_moe as VQM
import vq6_moe as V6
from engine.codebook_sim import CodebookSim
from vq12 import VQ12
from vq6 import VQ6

dev = "cuda"; torch.manual_seed(0)
DIM, INTER = F4.DIM, F4.INTER
S = 16; T = int(os.environ.get("T", 6)); K = 6
vq = VQ12(os.path.expanduser("~/dsv41-spark/a100-vq/vq_3.0.npz"), dev)
w = lambda n, k: torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=dev)
sc = lambda n, k: torch.randint(124, 126, (n, k // 32), dtype=torch.uint8, device=dev)
vqa = V6.VQ6Arena(S, dev).attach(VQ6(os.path.expanduser("~/dsv41-spark/a100-vq/vq2_3.npz"), dev))
cb = C3.CB3ArenaV2(S, dev); cb.sim = CodebookSim(3, dev)
for i in range(S):
    args = (w(INTER, DIM), sc(INTER, DIM), w(DIM, INTER), sc(DIM, INTER), w(INTER, DIM), sc(INTER, DIM))
    vqa.load_slot(i, *args); cb.load_slot(i, *args)
x = torch.randn(T, DIM, dtype=torch.bfloat16, device=dev) * 0.3
slots = torch.randint(0, S, (T, K), dtype=torch.int32, device=dev)
wts = torch.rand(T, K, device=dev)
BM = F4._pick_bm(T * K)
print(f"T={T} K={K} -> BM={BM}")

def bench(fn, n=100):
    try:
        for _ in range(10): fn()
    except Exception as e:
        return float("inf")
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / n * 1e3

base_up, base_dn = V6._VQ6_UP_CFG[BM], V6._VQ6_DOWN_CFG[BM]
print("cb3 v3", round(bench(lambda: C3.moe_forward_v3(x, slots, wts, cb, 10.0)), 3),
      "vq6 base", round(bench(lambda: V6.moe_forward_vq6(x, slots, wts, vqa, 10.0)), 3))
for which in ("up", "down"):
    rows = []
    for bn, nw, ns in itertools.product((16, 32, 64, 128), (1, 2, 4, 8), (1, 2, 3, 4)):
        cfg = (bn, nw, ns)
        u, d = (cfg, base_dn) if which == "up" else (base_up, cfg)
        rows.append((bench(lambda: V6.moe_forward_vq6(x, slots, wts, vqa, 10.0, cfg_up=u, cfg_down=d)), cfg))
    rows.sort()
    print(f"{which}: best " + "  ".join(f"{c} {t:.3f}" for t, c in rows[:6]))
