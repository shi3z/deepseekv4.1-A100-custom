"""Same group, same tile -- but the tile reaches the dot two ways: straight from the registers
that built it, or stored to memory and loaded back plainly. If only the second is right, the
fault is the register layout of the interleave-built tile, not its values."""
import os, sys, torch, triton, triton.language as tl
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work/tools"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/a100-vq"))
import fp4_moe as F4
import vq12_moe as VQM
from vq12 import VQ12

@triton.jit
def direct(x_ptr, lo_ptr, hi_ptr, s_ptr, out_ptr, tile_ptr, lut_ptr, tab_ptr,
           K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    KL: tl.constexpr = K // 4
    KH: tl.constexpr = K // 8
    SG: tl.constexpr = K // 32
    om = tl.arange(0, BM); on = tl.arange(0, BN)
    lo = tl.load(lo_ptr + on[:, None] * KL + tl.arange(0, 64)[None, :])
    hi = tl.load(hi_ptr + on[:, None] * KH + tl.arange(0, 32)[None, :])
    sc = tl.load(s_ptr + on[:, None] * SG + tl.arange(0, 8)[None, :])
    L = VQM._split8w(lo, BN, 8); H = VQM._split8w(hi, BN, 4); S = VQM._split8(sc, BN)
    t = VQM._grp_packed_vq(L[0], H[0], lut_ptr, BN)
    tl.store(tile_ptr + on[:, None] * 16 + tl.arange(0, 16)[None, :], t)     # publish the tile
    acc = VQM._chunk_dot_lut(x_ptr + om[:, None] * K, 2 * tl.arange(0, 16)[None, :],
                             om[:, None] >= 0, t, S[0], tab_ptr)
    tl.store(out_ptr + om[:, None] * BN + on[None, :], acc)

@triton.jit
def viamem(x_ptr, tile_ptr, s_ptr, out_ptr, tab_ptr,
           K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    SG: tl.constexpr = K // 32
    om = tl.arange(0, BM); on = tl.arange(0, BN)
    t = tl.load(tile_ptr + on[:, None] * 16 + tl.arange(0, 16)[None, :])     # plain load
    sc = tl.load(s_ptr + on[:, None] * SG + tl.arange(0, 8)[None, :])
    S = VQM._split8(sc, BN)
    acc = VQM._chunk_dot_lut(x_ptr + om[:, None] * K, 2 * tl.arange(0, 16)[None, :],
                             om[:, None] >= 0, t, S[0], tab_ptr)
    tl.store(out_ptr + om[:, None] * BN + on[None, :], acc)

dev="cuda"; torch.manual_seed(0); DIM, INTER = F4.DIM, F4.INTER
vq = VQ12(os.path.expanduser("~/dsv41-spark/a100-vq/vq_3.0.npz"))
ar = VQM.VQ12Arena(1, dev).attach(vq)
ar.load_slot(0, torch.randint(0,256,(INTER,DIM//2),dtype=torch.uint8,device=dev),
                torch.randint(124,126,(INTER,DIM//32),dtype=torch.uint8,device=dev),
                torch.randint(0,256,(DIM,INTER//2),dtype=torch.uint8,device=dev),
                torch.randint(124,126,(DIM,INTER//32),dtype=torch.uint8,device=dev),
                torch.randint(0,256,(INTER,DIM//2),dtype=torch.uint8,device=dev),
                torch.randint(124,126,(INTER,DIM//32),dtype=torch.uint8,device=dev))
w1 = ar.dequant_slot(0)[0].float()
BM, BN = 16, 128
x = torch.randn(BM, DIM, dtype=torch.bfloat16, device=dev) * 0.3
tile = torch.zeros(BN, 16, dtype=torch.uint8, device=dev)
o1 = torch.empty(BM, BN, dtype=torch.float32, device=dev)
o2 = torch.empty(BM, BN, dtype=torch.float32, device=dev)
direct[(1,)](x, ar.w1_lo, ar.w1_hi, ar.s1, o1, tile, ar.lut, ar.tab, K=DIM, BM=BM, BN=BN)
torch.cuda.synchronize()
viamem[(1,)](x, tile, ar.s1, o2, ar.tab, K=DIM, BM=BM, BN=BN)
torch.cuda.synchronize()
ref = x[:, :32].float() @ w1[:BN, :32].T
# is the published tile right?
q = vq.unpack(ar.w1_lo[0], ar.w1_hi[0])[:BN, :16]
print(f"published tile correct: {bool((tile == q).all())}")
print(f"dot from registers : rel err {float((o1-ref).norm()/ref.norm()):.5f}")
print(f"dot after a memory round trip: rel err {float((o2-ref).norm()/ref.norm()):.5f}")
