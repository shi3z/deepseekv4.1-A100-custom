"""Read the arena through the kernel's own addressing and compare with vq12.unpack."""
import os, sys, torch, triton, triton.language as tl
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work/tools"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/a100-vq"))
import fp4_moe as F4
import vq12_moe as VQM
from vq12 import VQ12

@triton.jit
def probe(lo_ptr, hi_ptr, lut_ptr, out_ptr, slot, blk,
          N: tl.constexpr, K: tl.constexpr, BN: tl.constexpr):
    KL: tl.constexpr = K // 4
    KH: tl.constexpr = K // 8
    offs_n = tl.arange(0, BN)
    lo = lo_ptr + slot * (N * KL) + offs_n[:, None] * KL + tl.arange(0, 64)[None, :] + blk * 64
    hi = hi_ptr + slot * (N * KH) + offs_n[:, None] * KH + tl.arange(0, 32)[None, :] + blk * 32
    L = tl.load(lo); H = tl.load(hi)
    L0, L1, L2, L3, L4, L5, L6, L7 = VQM._split8w(L, BN, 8)
    H0, H1, H2, H3, H4, H5, H6, H7 = VQM._split8w(H, BN, 4)
    for i in tl.static_range(8):
        Lk = L0 if i==0 else (L1 if i==1 else (L2 if i==2 else (L3 if i==3 else (L4 if i==4 else (L5 if i==5 else (L6 if i==6 else L7))))))
        Hk = H0 if i==0 else (H1 if i==1 else (H2 if i==2 else (H3 if i==3 else (H4 if i==4 else (H5 if i==5 else (H6 if i==6 else H7))))))
        tl.store(out_ptr + offs_n[:, None] * 128 + i * 16 + tl.arange(0, 16)[None, :],
                 VQM._grp_packed_vq(Lk, Hk, lut_ptr, BN))

dev="cuda"; torch.manual_seed(0)
DIM, INTER = F4.DIM, F4.INTER
vq = VQ12(os.path.expanduser("~/dsv41-spark/a100-vq/vq_3.0.npz"))
ar = VQM.VQ12Arena(2, dev).attach(vq)
w1 = torch.randint(0,256,(INTER,DIM//2),dtype=torch.uint8,device=dev)
s1 = torch.randint(120,126,(INTER,DIM//32),dtype=torch.uint8,device=dev)
w2 = torch.randint(0,256,(DIM,INTER//2),dtype=torch.uint8,device=dev)
s2 = torch.randint(120,126,(DIM,INTER//32),dtype=torch.uint8,device=dev)
ar.load_slot(1, w1, s1, w2, s2, w1, s1)
ref_all = vq.unpack(ar.w1_lo[1], ar.w1_hi[1])          # [INTER, DIM/2]
BN = 16
for blk in (0, 3, 19):
    out = torch.zeros(BN, 128, dtype=torch.uint8, device=dev)
    probe[(1,)](ar.w1_lo, ar.w1_hi, ar.lut, out, 1, blk, N=INTER, K=DIM, BN=BN)
    torch.cuda.synchronize()
    ref = ref_all[:BN, blk*128:(blk+1)*128]
    print(f"block {blk:2d}: matches {bool((out==ref).all())}  ({float((out==ref).float().mean())*100:.1f}%)")
# and does the packed arena equal a re-pack of the original?
lo0, hi0, _ = vq.pack(w1, s1)
print(f"arena planes == vq.pack(w1): lo {bool((ar.w1_lo[1]==lo0).all())} hi {bool((ar.w1_hi[1]==hi0).all())}")
