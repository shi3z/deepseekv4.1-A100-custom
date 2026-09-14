"""CB3 v3 vs VQ12 v1 (three dependent loads) vs VQ12 v2 (one fused fp16 gather)."""
import os, sys, time
import torch
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work/tools"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/a100-vq"))
import cb3_moe as C3, fp4_moe as F4, vq12_moe as VQM
from engine.codebook_sim import CodebookSim
from vq12 import VQ12

dev = "cuda"; torch.manual_seed(0)
DIM, INTER = F4.DIM, F4.INTER
S = 16
T = int(os.environ.get("T", 6)); K = int(os.environ.get("K", 6))
vq = VQ12(os.path.expanduser("~/dsv41-spark/a100-vq/vq_3.0.npz"), dev)
w = lambda n, k: torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=dev)
sc = lambda n, k: torch.randint(124, 126, (n, k // 32), dtype=torch.uint8, device=dev)
cb = C3.CB3ArenaV2(S, dev); cb.sim = CodebookSim(3, dev)
vqa = VQM.VQ12Arena(S, dev).attach(vq)
for i in range(S):
    args = (w(INTER, DIM), sc(INTER, DIM), w(DIM, INTER), sc(DIM, INTER), w(INTER, DIM), sc(INTER, DIM))
    cb.load_slot(i, *args); vqa.load_slot(i, *args)
x = torch.randn(T, DIM, dtype=torch.bfloat16, device=dev) * 0.3
slots = torch.randint(0, S, (T, K), dtype=torch.int32, device=dev)
wts = torch.rand(T, K, device=dev)

y1 = VQM.moe_forward_vq(x, slots, wts, vqa, 10.0).float()
y2 = VQM.moe_forward_vq2(x, slots, wts, vqa, 10.0).float()
rel = float((y2 - y1).norm() / y1.norm())
print(f"v2 vs v1: max abs {float((y2-y1).abs().max()):.6f}, rel {rel:.2e}  "
      f"{'IDENTICAL' if rel == 0 else 'DIFFERS'}")

def bench(fn, n=200):
    for _ in range(20): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / n * 1e3

a = bench(lambda: C3.moe_forward_v3(x, slots, wts, cb, 10.0))
b = bench(lambda: VQM.moe_forward_vq(x, slots, wts, vqa, 10.0))
c = bench(lambda: VQM.moe_forward_vq2(x, slots, wts, vqa, 10.0))
print(f"T={T} K={K}:  CB3 v3 {a:.3f}   VQ12 v1 {b:.3f} (+{(b-a)/a*100:.1f} %)   "
      f"VQ12 v2 {c:.3f} (+{(c-a)/a*100:.1f} %)   v2 vs v1 {(c-b)/b*100:+.1f} %")
