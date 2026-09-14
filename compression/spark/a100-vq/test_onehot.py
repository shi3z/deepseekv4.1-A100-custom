"""Which k does the up kernel actually read? One-hot x, one expert, read h back."""
import os, sys, torch
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/work/tools"))
sys.path.insert(0, os.path.expanduser("~/dsv41-spark/a100-vq"))
import fp4_moe as F4, cb3_moe as C3
import vq12_moe as VQM
from vq12 import VQ12, FP4_VALS
dev="cuda"; DIM, INTER = F4.DIM, F4.INTER
vq = VQ12(os.path.expanduser("~/dsv41-spark/a100-vq/vq_3.0.npz"))
ar = VQM.VQ12Arena(1, dev).attach(vq)

# w1: row 0 carries a distinct code per k within the first 32; everything else is code 0.
codes = torch.zeros(INTER, DIM, dtype=torch.uint8, device=dev)
codes[0, :32] = torch.tensor([1,2,3,4,5,6,7,9,10,11,12,13,14,15,1,2,
                              3,4,5,6,7,9,10,11,12,13,14,15,1,2,3,4], dtype=torch.uint8, device=dev)
w1 = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
s1 = torch.full((INTER, DIM // 32), 127, dtype=torch.uint8, device=dev)
w3 = (torch.full((INTER, DIM), 2, dtype=torch.uint8, device=dev)[:, 0::2] |
      (torch.full((INTER, DIM), 2, dtype=torch.uint8, device=dev)[:, 1::2] << 4)).contiguous()
w2 = torch.zeros(DIM, INTER // 2, dtype=torch.uint8, device=dev)
s2 = torch.full((DIM, INTER // 32), 127, dtype=torch.uint8, device=dev)
ar.load_slot(0, w1, s1, w2, s2, w3, s1)
# what the arena really holds after VQ, as codes
q1 = vq.unpack(ar.w1_lo[0], ar.w1_hi[0])
qc = torch.empty(INTER, DIM, dtype=torch.uint8, device=dev)
qc[:, 0::2] = q1 & 0xF; qc[:, 1::2] = q1 >> 4
want = FP4_VALS.to(dev)[qc[0, :32].long()]

h = torch.empty((1, INTER), dtype=torch.bfloat16, device=dev)
wgt = torch.ones(1, device=dev)
bs = torch.zeros(1, dtype=torch.int32, device=dev)
bp = torch.zeros(16, dtype=torch.int32, device=dev)
got = []
for k0 in range(32):
    x = torch.zeros(1, DIM, dtype=torch.bfloat16, device=dev); x[0, k0] = 1.0
    VQM._vq_up_kernel[(1, INTER // 128)](
        x, ar.w1_lo, ar.w1_hi, ar.s1, ar.w3_lo, ar.w3_hi, ar.s3, h, ar.lut, ar.tab,
        wgt, bs, bp, x.stride(0), h.stride(0), 1e30,
        TOPK=1, N=INTER, K=DIM, BM=16, BN=128, num_warps=4, num_stages=1)
    torch.cuda.synchronize()
    g = float(h[0, 0])            # silu(gate)*up*1 with up = 1.0 -> g = silu(gate)
    got.append(g)
import math
def silu(v): return v / (1 + math.exp(-v))
exp = [silu(float(want[k])) for k in range(32)]
mis = [k for k in range(32) if abs(got[k] - exp[k]) > 0.05]
print("k where the kernel disagrees:", mis[:16], f"({len(mis)}/32)")
for k in range(8):
    src = [j for j in range(32) if abs(got[k] - exp[j]) < 0.02]
    print(f"  x at k={k:2d} -> kernel used k={src}  (expected [{k}])")
