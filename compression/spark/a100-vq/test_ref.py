"""Is the reference in test_vq12_moe right? Run CB3's own v2 forward against the same construction."""
import os, sys, torch
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work/tools"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/a100-vq"))
import fp4_moe as F4, cb3_moe as C3
from engine.codebook_sim import CodebookSim
import vq12_moe as VQM
from vq12 import VQ12

dev = "cuda"; torch.manual_seed(0)
DIM, INTER = F4.DIM, F4.INTER
S, T, K = 4, 6, 6
raw = []
for s in range(S):
    raw.append((torch.randint(0,256,(INTER,DIM//2),dtype=torch.uint8,device=dev),
                torch.randint(120,126,(INTER,DIM//32),dtype=torch.uint8,device=dev),
                torch.randint(0,256,(DIM,INTER//2),dtype=torch.uint8,device=dev),
                torch.randint(120,126,(DIM,INTER//32),dtype=torch.uint8,device=dev),
                torch.randint(0,256,(INTER,DIM//2),dtype=torch.uint8,device=dev),
                torch.randint(120,126,(INTER,DIM//32),dtype=torch.uint8,device=dev)))
x = torch.randn(T, DIM, dtype=torch.bfloat16, device=dev) * 0.5
slots = torch.randint(0, S, (T, K), dtype=torch.int32, device=dev)
wts = torch.rand(T, K, device=dev)

def expected(deq):
    out = torch.zeros(T, DIM, dtype=torch.float32, device=dev)
    for t in range(T):
        for k in range(K):
            w1b, w2b, w3b = deq[int(slots[t, k])]
            g = (x[t].float() @ w1b.float().T).clamp(max=10.0)
            u = (x[t].float() @ w3b.float().T).clamp(-10.0, 10.0)
            hh = (g * torch.sigmoid(g) * u * float(wts[t, k])).to(torch.bfloat16)
            out[t] += hh.float() @ w2b.float().T
    return out

c3 = C3.CB3ArenaV2(S, dev); c3.sim = CodebookSim(3, dev)
for s in range(S): c3.load_slot(s, *raw[s])
d3 = [c3.dequant_slot(s) for s in range(S)]
y3 = C3.moe_forward_v2(x, slots, wts, c3).float()
e3 = expected(d3)
print(f"CB3 v2   vs the same reference: rel err {float((y3-e3).norm()/e3.norm()):.5f}")

vq = VQ12(os.path.expanduser("~/dsv41-spark/a100-vq/vq_3.0.npz"))
av = VQM.VQ12Arena(S, dev).attach(vq)
for s in range(S): av.load_slot(s, *raw[s])
dv = [av.dequant_slot(s) for s in range(S)]
yv = VQM.moe_forward_vq(x, slots, wts, av).float()
ev = expected(dv)
print(f"VQ12     vs the same reference: rel err {float((yv-ev).norm()/ev.norm()):.5f}")
print(f"VQ12 vs CB3 outputs (different quantisers, so a real gap is expected): "
      f"{float((yv-y3).norm()/y3.norm()):.4f}")
