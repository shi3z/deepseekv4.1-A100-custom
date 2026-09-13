"""CB3 v3 vs VQ12 (dim 4) vs VQ6 (dim 2): correctness against the format's own dequant, then time."""
import os, sys, time
import torch
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work/tools"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/a100-vq"))
import cb3_moe as C3, fp4_moe as F4, vq12_moe as VQM, vq6_moe as V6
from engine.codebook_sim import CodebookSim
from vq12 import VQ12
from vq6 import VQ6

dev = "cuda"; torch.manual_seed(0)
DIM, INTER = F4.DIM, F4.INTER
S = 16; T = int(os.environ.get("T", 6)); K = 6
vq = VQ12(os.path.expanduser("~/dsv41-spark/a100-vq/vq_3.0.npz"), dev)
v6 = VQ6(os.path.expanduser("~/dsv41-spark/a100-vq/vq2_3.npz"), dev)
w = lambda n, k: torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=dev)
sc = lambda n, k: torch.randint(124, 126, (n, k // 32), dtype=torch.uint8, device=dev)
cb = C3.CB3ArenaV2(S, dev); cb.sim = CodebookSim(3, dev)
vqa = VQM.VQ12Arena(S, dev).attach(vq)
v6a = V6.VQ6Arena(S, dev).attach(v6)
W = []
for i in range(S):
    args = (w(INTER, DIM), sc(INTER, DIM), w(DIM, INTER), sc(DIM, INTER), w(INTER, DIM), sc(INTER, DIM))
    cb.load_slot(i, *args); vqa.load_slot(i, *args); v6a.load_slot(i, *args); W.append(args)
x = torch.randn(T, DIM, dtype=torch.bfloat16, device=dev) * 0.3
slots = torch.randint(0, S, (T, K), dtype=torch.int32, device=dev)
wts = torch.rand(T, K, device=dev)

# reference: the same routing done in torch against VQ6's own dequantised weights
def ref(arena):
    out = torch.zeros(T, DIM, dtype=torch.float32, device=dev)
    for t in range(T):
        for kk in range(K):
            s = int(slots[t, kk])
            w1, w2, w3 = arena.dequant_slot(s)
            g = (x[t].float() @ w1.float().T).clamp(max=10.0)
            u = (x[t].float() @ w3.float().T).clamp(-10.0, 10.0)
            h = g * torch.sigmoid(g) * u
            out[t] += float(wts[t, kk]) * (h @ w2.float().T)
    return out

r = ref(v6a)
y = V6.moe_forward_vq6(x, slots, wts, v6a, 10.0).float()
print(f"VQ6 kernel vs its own dequant: rel err {float((y-r).norm()/r.norm()):.2e}")

def bench(fn, n=200):
    for _ in range(20): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / n * 1e3

a = bench(lambda: C3.moe_forward_v3(x, slots, wts, cb, 10.0))
b = bench(lambda: VQM.moe_forward_vq(x, slots, wts, vqa, 10.0))
c = bench(lambda: V6.moe_forward_vq6(x, slots, wts, v6a, 10.0))
print(f"T={T} K={K}:  CB3 v3 {a:.3f}   VQ12 {b:.3f} (+{(b-a)/a*100:.1f} %)   "
      f"VQ6 {c:.3f} ({(c-a)/a*100:+.1f} %)")
