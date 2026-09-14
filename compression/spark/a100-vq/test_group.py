"""One scale group at a time: does group i use x[32i:32i+32] and scale i and its own 32 weights?"""
import os, sys, torch, triton, triton.language as tl
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work/tools"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/a100-vq"))
import fp4_moe as F4
import vq12_moe as VQM
from vq12 import VQ12

@triton.jit
def one_group(x_ptr, lo_ptr, hi_ptr, s_ptr, out_ptr, lut_ptr, tab_ptr,
              N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, G: tl.constexpr):
    KL: tl.constexpr = K // 4
    KH: tl.constexpr = K // 8
    SG: tl.constexpr = K // 32
    offs_m = tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    x_base = x_ptr + offs_m[:, None] * K
    xk = 2 * tl.arange(0, 16)[None, :]
    lo = tl.load(lo_ptr + offs_n[:, None] * KL + tl.arange(0, 64)[None, :])
    hi = tl.load(hi_ptr + offs_n[:, None] * KH + tl.arange(0, 32)[None, :])
    sc = tl.load(s_ptr + offs_n[:, None] * SG + tl.arange(0, 8)[None, :])
    L = VQM._split8w(lo, BN, 8)
    H = VQM._split8w(hi, BN, 4)
    S = VQM._split8(sc, BN)
    acc = VQM._chunk_dot_lut(x_base + G * 32, xk, offs_m[:, None] >= 0,
                             VQM._grp_packed_vq(L[G], H[G], lut_ptr, BN), S[G], tab_ptr)
    tl.store(out_ptr + offs_m[:, None] * BN + offs_n[None, :], acc)

dev="cuda"; torch.manual_seed(0); DIM, INTER = F4.DIM, F4.INTER
vq = VQ12(os.path.expanduser("~/dsv41-spark/a100-vq/vq_3.0.npz"))
ar = VQM.VQ12Arena(1, dev).attach(vq)
ar.load_slot(0, torch.randint(0,256,(INTER,DIM//2),dtype=torch.uint8,device=dev),
                torch.randint(124,126,(INTER,DIM//32),dtype=torch.uint8,device=dev),
                torch.randint(0,256,(DIM,INTER//2),dtype=torch.uint8,device=dev),
                torch.randint(124,126,(DIM,INTER//32),dtype=torch.uint8,device=dev),
                torch.randint(0,256,(INTER,DIM//2),dtype=torch.uint8,device=dev),
                torch.randint(124,126,(INTER,DIM//32),dtype=torch.uint8,device=dev))
w1 = ar.dequant_slot(0)[0].float()        # [INTER, DIM]
BM, BN = 16, 128
x = (torch.randn(BM, DIM, dtype=torch.bfloat16, device=dev) * 0.3)
for G in range(8):
    out = torch.empty(BM, BN, dtype=torch.float32, device=dev)
    one_group[(1,)](x, ar.w1_lo, ar.w1_hi, ar.s1, out, ar.lut, ar.tab,
                    N=INTER, K=DIM, BM=BM, BN=BN, G=G)
    torch.cuda.synchronize()
    ref = x[:, G*32:(G+1)*32].float() @ w1[:BN, G*32:(G+1)*32].T
    e = float((out - ref).norm() / ref.norm())
    # which k slice does it actually match?
    hits = [j for j in range(8) if float((out - (x[:, j*32:(j+1)*32].float() @ w1[:BN, j*32:(j+1)*32].T)).norm()
                                         / ref.norm()) < 1e-2]
    print(f"group {G}: rel err {e:.5f}   matches k-slice(s) {hits}")
