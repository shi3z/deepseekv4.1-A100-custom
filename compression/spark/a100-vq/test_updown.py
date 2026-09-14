"""Up and down kernels separately, against a reference built from the arena's OWN weights."""
import os, sys, torch
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work/tools"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/a100-vq"))
import fp4_moe as F4
import vq12_moe as VQM
from vq12 import VQ12
dev="cuda"; torch.manual_seed(0); DIM, INTER = F4.DIM, F4.INTER
vq = VQ12(os.path.expanduser("~/dsv41-spark/a100-vq/vq_3.0.npz"))
S, T, K = 2, 4, 2
ar = VQM.VQ12Arena(S, dev).attach(vq)
for s in range(S):
    ar.load_slot(s, torch.randint(0,256,(INTER,DIM//2),dtype=torch.uint8,device=dev),
                    torch.randint(124,126,(INTER,DIM//32),dtype=torch.uint8,device=dev),
                    torch.randint(0,256,(DIM,INTER//2),dtype=torch.uint8,device=dev),
                    torch.randint(124,126,(DIM,INTER//32),dtype=torch.uint8,device=dev),
                    torch.randint(0,256,(INTER,DIM//2),dtype=torch.uint8,device=dev),
                    torch.randint(124,126,(INTER,DIM//32),dtype=torch.uint8,device=dev))
W = [ar.dequant_slot(s) for s in range(S)]          # (w1, w2, w3) bf16, what the kernel must use
x = torch.randn(T, DIM, dtype=torch.bfloat16, device=dev) * 0.3
slots = torch.randint(0, S, (T, K), dtype=torch.int32, device=dev)
wts = torch.rand(T, K, device=dev)
P = T * K
BM = F4._pick_bm(P); bn1, nw1, ns1 = VQM.C3._UP_CFG[BM]; bn2, nw2, ns2 = VQM.C3._DOWN_CFG[BM]
bs, bp, NB = F4.build_routing(slots, ar.slots, BM)
wgt = wts.reshape(-1).float().contiguous()
h = torch.empty((P, INTER), dtype=torch.bfloat16, device=dev)
VQM._vq_up_kernel[(NB, INTER // bn1)](
    x, ar.w1_lo, ar.w1_hi, ar.s1, ar.w3_lo, ar.w3_hi, ar.s3, h, ar.lut, ar.tab,
    wgt, bs, bp, x.stride(0), h.stride(0), 10.0,
    TOPK=K, N=INTER, K=DIM, BM=BM, BN=bn1, num_warps=nw1, num_stages=ns1)
torch.cuda.synchronize()
href = torch.zeros(P, INTER, dtype=torch.float32, device=dev)
for t in range(T):
    for k in range(K):
        w1b, w2b, w3b = W[int(slots[t, k])]
        g = (x[t].float() @ w1b.float().T).clamp(max=10.0)
        u = (x[t].float() @ w3b.float().T).clamp(-10.0, 10.0)
        href[t*K+k] = g * torch.sigmoid(g) * u * float(wts[t, k])
eh = (h.float() - href).norm() / href.norm()
print(f"UP   kernel: rel err {eh:.5f}   ({'OK' if eh < 2e-2 else 'WRONG'})")
parts = torch.empty((P, DIM), dtype=torch.float32, device=dev)
VQM._vq_down_kernel[(NB, DIM // bn2)](
    href.to(torch.bfloat16), ar.w2_lo, ar.w2_hi, ar.s2, parts, ar.lut, ar.tab, bs, bp,
    INTER, parts.stride(0), TOPK=K, N=DIM, K=INTER, BM=BM, BN=bn2, NTOK=T,
    num_warps=nw2, num_stages=ns2)
torch.cuda.synchronize()
y = parts.view(K, T, DIM).sum(0)
yref = torch.zeros(T, DIM, dtype=torch.float32, device=dev)
for t in range(T):
    for k in range(K):
        yref[t] += href[t*K+k].float() @ W[int(slots[t, k])][1].float().T
ey = (y - yref).norm() / yref.norm()
print(f"DOWN kernel: rel err {ey:.5f}   ({'OK' if ey < 2e-2 else 'WRONG'})")
