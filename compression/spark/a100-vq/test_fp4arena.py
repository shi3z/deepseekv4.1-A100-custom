"""Same VQ-quantised weights, run through the stock FP4 kernel: isolates data vs kernel."""
import os, sys, torch
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work/tools"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/a100-vq"))
import fp4_moe as F4
import vq12_moe as VQM
from vq12 import VQ12
dev="cuda"; torch.manual_seed(0)
DIM, INTER = F4.DIM, F4.INTER
S, T, K = 4, 6, 6
vq = VQ12(os.path.expanduser("~/dsv41-spark/a100-vq/vq_3.0.npz"))
av = VQM.VQ12Arena(S, dev).attach(vq)
af = F4.ExpertArena(S, dev)
raw=[]
for s in range(S):
    r=(torch.randint(0,256,(INTER,DIM//2),dtype=torch.uint8,device=dev),
       torch.randint(120,126,(INTER,DIM//32),dtype=torch.uint8,device=dev),
       torch.randint(0,256,(DIM,INTER//2),dtype=torch.uint8,device=dev),
       torch.randint(120,126,(DIM,INTER//32),dtype=torch.uint8,device=dev),
       torch.randint(0,256,(INTER,DIM//2),dtype=torch.uint8,device=dev),
       torch.randint(120,126,(INTER,DIM//32),dtype=torch.uint8,device=dev))
    raw.append(r); av.load_slot(s, *r)
    # the same weights after VQ, written back as ordinary FP4
    q1 = vq.unpack(av.w1_lo[s], av.w1_hi[s]); q3 = vq.unpack(av.w3_lo[s], av.w3_hi[s]); q2 = vq.unpack(av.w2_lo[s], av.w2_hi[s])
    af.load_slot(s, q1, r[1], q2, r[3], q3, r[5])
x = torch.randn(T, DIM, dtype=torch.bfloat16, device=dev)*0.5
slots = torch.randint(0,S,(T,K),dtype=torch.int32,device=dev)
wts = torch.rand(T,K,device=dev)
yf = F4.moe_forward(x, slots, wts, af).float()
yv = VQM.moe_forward_vq(x, slots, wts, av).float()
print(f"VQ12 kernel vs the stock FP4 kernel on identical weights: rel err {float((yv-yf).norm()/yf.norm()):.5f}")
