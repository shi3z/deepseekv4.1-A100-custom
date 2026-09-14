"""Numerical check of vq12_moe against a plain bf16 reference on the same quantised weights."""
import os, sys, time, torch
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work/tools"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/a100-vq"))
import fp4_moe as F4, cb3_moe as C3
import vq12_moe as VQM
from vq12 import VQ12

dev = "cuda"
torch.manual_seed(0)
vq = VQ12(os.path.expanduser("~/dsv41-spark/a100-vq/vq_3.0.npz"))
S, T, K = 4, 6, 6
ar = VQM.VQ12Arena(S, dev).attach(vq)
DIM, INTER = F4.DIM, F4.INTER
ref = []
for s in range(S):
    w1 = torch.randint(0, 256, (INTER, DIM // 2), dtype=torch.uint8, device=dev)
    w3 = torch.randint(0, 256, (INTER, DIM // 2), dtype=torch.uint8, device=dev)
    w2 = torch.randint(0, 256, (DIM, INTER // 2), dtype=torch.uint8, device=dev)
    s1 = torch.randint(120, 126, (INTER, DIM // 32), dtype=torch.uint8, device=dev)
    s3 = torch.randint(120, 126, (INTER, DIM // 32), dtype=torch.uint8, device=dev)
    s2 = torch.randint(120, 126, (DIM, INTER // 32), dtype=torch.uint8, device=dev)
    ar.load_slot(s, w1, s1, w2, s2, w3, s3)
    ref.append(ar.dequant_slot(s))          # what the kernel must compute with
x = torch.randn(T, DIM, dtype=torch.bfloat16, device=dev) * 0.5
slots = torch.randint(0, S, (T, K), dtype=torch.int32, device=dev)
wts = torch.rand(T, K, device=dev)
y = VQM.moe_forward_vq(x, slots, wts, ar)

exp = torch.zeros(T, DIM, dtype=torch.float32, device=dev)
for t in range(T):
    for k in range(K):
        w1b, w2b, w3b = ref[int(slots[t, k])]
        g = (x[t].float() @ w1b.float().T).clamp(max=10.0)
        u = (x[t].float() @ w3b.float().T).clamp(-10.0, 10.0)
        hh = (g * torch.sigmoid(g) * u * float(wts[t, k])).to(torch.bfloat16)
        exp[t] += hh.float() @ w2b.float().T
err = (y.float() - exp).norm() / exp.norm()
print(f"vq12 kernel vs reference: rel err {err:.5f}  ({'OK' if err < 2e-2 else 'MISMATCH'})")

def bench(fn, it=30):
    for _ in range(5): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / it * 1e3

c3 = C3.CB3ArenaV2(S, dev)
from engine.codebook_sim import CodebookSim
c3.sim = CodebookSim(3, dev)
for s in range(S):
    c3.load_slot(s, *[t for t in (torch.randint(0,256,(INTER,DIM//2),dtype=torch.uint8,device=dev),
                                   torch.randint(120,126,(INTER,DIM//32),dtype=torch.uint8,device=dev),
                                   torch.randint(0,256,(DIM,INTER//2),dtype=torch.uint8,device=dev),
                                   torch.randint(120,126,(DIM,INTER//32),dtype=torch.uint8,device=dev),
                                   torch.randint(0,256,(INTER,DIM//2),dtype=torch.uint8,device=dev),
                                   torch.randint(120,126,(INTER,DIM//32),dtype=torch.uint8,device=dev))])
tv = bench(lambda: VQM.moe_forward_vq(x, slots, wts, ar))
t2 = bench(lambda: C3.moe_forward_v2(x, slots, wts, c3))
t3 = bench(lambda: C3.moe_forward_v3(x, slots, wts, c3))
print(f"decode {T}x{K}: vq12 {tv:.3f} ms | cb3 v2 {t2:.3f} ms | cb3 v3 (PTX) {t3:.3f} ms")
